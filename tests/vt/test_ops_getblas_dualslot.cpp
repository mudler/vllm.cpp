// #837 GetBlas dual-slot host lifetime + product-call seam (research c24b).
// Production GetBlas and these tests both execute RocmProductGetBlasOn.
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include <doctest/doctest.h>

#include "vt/rocm/rocm_getblas_dualslot.h"


#ifndef VLLM_CPP_SOURCE_DIR
#define VLLM_CPP_SOURCE_DIR "."
#endif

namespace {

std::string ReadText(const char* rel) {
  const std::string path = std::string(VLLM_CPP_SOURCE_DIR) + "/" + rel;
  std::ifstream in(path);
  REQUIRE(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

struct Rec {
  int create[2] = {0, 0};
  int destroy[2] = {0, 0};
  int set_stream[2] = {0, 0};
  int set_device_n = 0;
  int last_set_device = -999;
  uintptr_t handle[2] = {0, 0};
  int stream[2] = {0, 0};
  int next_id = 1;
  bool capturing = false;
  int cur_dev = -1;
};

struct FakeHooks {
  using handle_t = uintptr_t;
  using stream_t = int;
  Rec* rec = nullptr;
  int last_create_slot = -1;

  static handle_t NullHandle() { return 0; }
  static stream_t NullStream() { return 0; }
  static bool IsNull(handle_t h) { return h == 0; }

  bool StreamIsCapturing(stream_t) const { return rec->capturing; }
  int GetDevice() const { return rec->cur_dev; }
  void SetDevice(int d) {
    rec->set_device_n++;
    rec->last_set_device = d;
    rec->cur_dev = d;
  }
  handle_t Create() {
    const int slot = last_create_slot >= 0 ? last_create_slot : 0;
    rec->create[slot]++;
    const handle_t h = static_cast<handle_t>(rec->next_id++);
    rec->handle[slot] = h;
    return h;
  }
  void Destroy(handle_t h) {
    for (int i = 0; i < 2; ++i) {
      if (rec->handle[i] == h) {
        rec->destroy[i]++;
        rec->handle[i] = 0;
        return;
      }
    }
    rec->destroy[0]++;
  }
  void SetStream(handle_t h, stream_t s) {
    for (int i = 0; i < 2; ++i) {
      if (rec->handle[i] == h) {
        rec->set_stream[i]++;
        rec->stream[i] = s;
        return;
      }
    }
  }
};

struct TrackingHooks : FakeHooks {
  handle_t Create() {
    last_create_slot = (rec->last_set_device == 1) ? 1 : 0;
    return FakeHooks::Create();
  }
};

inline int SwappedSlotIndex(int device) { return (device == 0) ? 1 : 0; }

struct Table {
  Rec rec;
  uintptr_t h0 = 0;
  uintptr_t h1 = 0;
};

// Product hops go through RocmProductGetBlasOn — same function as GetBlas.
template <class Eng, class Hooks, class Forward = vt::rocm::RocmGetBlasForward>
void RunProductHops(Eng& eng, Hooks& hooks, Table* t, bool change_s0_stream) {
  hooks.rec = &t->rec;
  t->h0 = vt::rocm::RocmProductGetBlasOn<Eng, Hooks, Forward>(eng, 0, /*s0=*/1, hooks);
  t->h1 = vt::rocm::RocmProductGetBlasOn<Eng, Hooks, Forward>(eng, 1, /*s1=*/2, hooks);
  t->h0 = vt::rocm::RocmProductGetBlasOn<Eng, Hooks, Forward>(eng, 0, /*s0=*/1, hooks);
  t->h1 = vt::rocm::RocmProductGetBlasOn<Eng, Hooks, Forward>(eng, 1, /*s1=*/2, hooks);
  if (change_s0_stream) {
    (void)vt::rocm::RocmProductGetBlasOn<Eng, Hooks, Forward>(eng, 0, /*s0'=*/3, hooks);
  }
}

bool ProductTableHolds(const Table& t, bool after_stream_change) {
  if (t.rec.create[0] != 1) return false;
  if (t.rec.create[1] != 1) return false;
  if (t.rec.destroy[0] != 0) return false;
  if (t.rec.destroy[1] != 0) return false;
  if (t.rec.handle[0] != t.h0) return false;
  if (t.rec.handle[1] != t.h1) return false;
  if (t.h0 == 0 || t.h1 == 0 || t.h0 == t.h1) return false;
  if (after_stream_change) {
    if (t.rec.stream[0] != 3) return false;
    if (t.rec.stream[1] != 2) return false;
  }
  return true;
}

struct SingleTlsEngine {
  using handle_t = uintptr_t;
  using stream_t = int;
  struct Tls {
    int dev = -1;
    stream_t stream = 0;
    handle_t handle = 0;
  } tls;
  handle_t Get(int device, stream_t stream, TrackingHooks& hooks) {
    if (!hooks.StreamIsCapturing(stream)) {
      if (hooks.GetDevice() != device) hooks.SetDevice(device);
    }
    if (hooks.IsNull(tls.handle) || tls.dev != device) {
      if (!hooks.IsNull(tls.handle)) {
        hooks.Destroy(tls.handle);
        tls.handle = 0;
      }
      if (!hooks.StreamIsCapturing(stream)) hooks.SetDevice(device);
      hooks.last_create_slot = (device == 1) ? 1 : 0;
      tls.handle = hooks.Create();
      tls.dev = device;
      tls.stream = 0;
    }
    if (tls.stream != stream) {
      hooks.SetStream(tls.handle, stream);
      tls.stream = stream;
    }
    return tls.handle;
  }
};

struct NoStreamHooks : TrackingHooks {
  void SetStream(handle_t, stream_t) {}
};

struct CaptureAlwaysSetEngine {
  vt::rocm::GetBlasDualSlotEngine<TrackingHooks> inner;
  uintptr_t Get(int device, int stream, TrackingHooks& hooks) {
    hooks.SetDevice(device);
    return inner.Get(device, stream, hooks);
  }
};

// c24b mutants: product call forwards the wrong device or a null stream.
struct ForwardDev0 {
  template <class Engine, class Hooks>
  static typename Engine::handle_t apply(Engine& engine, int /*device*/,
                                         typename Engine::stream_t stream,
                                         Hooks& hooks) {
    return engine.Get(0, stream, hooks);
  }
};

struct ForwardNullStream {
  template <class Engine, class Hooks>
  static typename Engine::handle_t apply(Engine& engine, int device,
                                         typename Engine::stream_t /*stream*/,
                                         Hooks& hooks) {
    return engine.Get(device, Hooks::NullStream(), hooks);
  }
};

}  // namespace

TEST_CASE("getblas slot index") {
  CHECK(vt::rocm::GetBlasSlotIndex(0) == 0);
  CHECK(vt::rocm::GetBlasSlotIndex(1) == 1);
  CHECK(vt::rocm::GetBlasSlotIndex(2) == 0);
  CHECK(vt::rocm::GetBlasSlotIndex(-1) == 0);
}

TEST_CASE("getblas first use fills slot 0 only") {
  using Eng = vt::rocm::GetBlasDualSlotEngine<TrackingHooks>;
  Eng eng;
  TrackingHooks hooks;
  Rec rec;
  hooks.rec = &rec;
  const auto h0 = vt::rocm::RocmProductGetBlasOn(eng, 0, 1, hooks);
  CHECK(h0 != 0);
  CHECK(eng.tls_slots[0].handle == h0);
  CHECK(eng.tls_slots[1].handle == 0);
  CHECK(rec.create[0] == 1);
  CHECK(rec.create[1] == 0);
}

TEST_CASE("getblas product lifetime table via product-call seam") {
  using Eng = vt::rocm::GetBlasDualSlotEngine<TrackingHooks>;
  Eng eng;
  TrackingHooks hooks;
  Table t;
  RunProductHops(eng, hooks, &t, /*change_s0_stream=*/true);
  CHECK(ProductTableHolds(t, /*after_stream_change=*/true));
  CHECK(eng.tls_slots[0].handle == t.h0);
  CHECK(eng.tls_slots[1].handle == t.h1);
  CHECK(t.rec.create[0] == 1);
  CHECK(t.rec.create[1] == 1);
  CHECK(t.rec.destroy[0] == 0);
  CHECK(t.rec.destroy[1] == 0);
  CHECK(t.rec.stream[0] == 3);
  CHECK(t.rec.stream[1] == 2);
}

TEST_CASE("getblas hop 1->0->1 keeps both handles") {
  using Eng = vt::rocm::GetBlasDualSlotEngine<TrackingHooks>;
  Eng eng;
  TrackingHooks hooks;
  Rec rec;
  hooks.rec = &rec;
  const auto h0 = vt::rocm::RocmProductGetBlasOn(eng, 0, 1, hooks);
  const auto h1 = vt::rocm::RocmProductGetBlasOn(eng, 1, 2, hooks);
  CHECK(vt::rocm::RocmProductGetBlasOn(eng, 0, 1, hooks) == h0);
  CHECK(vt::rocm::RocmProductGetBlasOn(eng, 1, 2, hooks) == h1);
  CHECK(rec.destroy[0] == 0);
  CHECK(rec.destroy[1] == 0);
  CHECK(rec.create[0] == 1);
  CHECK(rec.create[1] == 1);
}

TEST_CASE("getblas capture path does not SetDevice") {
  using Eng = vt::rocm::GetBlasDualSlotEngine<TrackingHooks>;
  Eng eng;
  TrackingHooks hooks;
  Rec rec;
  rec.capturing = true;
  rec.cur_dev = 0;
  hooks.rec = &rec;
  (void)vt::rocm::RocmProductGetBlasOn(eng, 1, 9, hooks);
  CHECK(rec.set_device_n == 0);
  CHECK(rec.last_set_device == -999);
}

TEST_CASE("getblas RED swapped selector fills the wrong slot") {
  using Eng = vt::rocm::GetBlasDualSlotEngine<TrackingHooks, SwappedSlotIndex>;
  Eng eng;
  TrackingHooks hooks;
  Rec rec;
  hooks.rec = &rec;
  const auto h = vt::rocm::RocmProductGetBlasOn(eng, 0, 1, hooks);
  CHECK(eng.tls_slots[0].handle == 0);
  CHECK(eng.tls_slots[1].handle == h);
}

TEST_CASE("getblas RED destroy-on-hop fails table") {
  SingleTlsEngine eng;
  TrackingHooks hooks;
  Table t;
  RunProductHops(eng, hooks, &t, /*change_s0_stream=*/false);
  CHECK(t.rec.destroy[0] >= 1);
  CHECK_FALSE(ProductTableHolds(t, /*after_stream_change=*/false));
}

TEST_CASE("getblas RED missing stream rebind fails table") {
  using Eng = vt::rocm::GetBlasDualSlotEngine<NoStreamHooks>;
  Eng eng;
  NoStreamHooks hooks;
  Table t;
  RunProductHops(eng, hooks, &t, /*change_s0_stream=*/true);
  CHECK(t.rec.stream[0] != 3);
  CHECK_FALSE(ProductTableHolds(t, /*after_stream_change=*/true));
}

TEST_CASE("getblas RED capture SetDevice fails capture invariant") {
  CaptureAlwaysSetEngine eng;
  TrackingHooks hooks;
  Rec rec;
  rec.capturing = true;
  rec.cur_dev = 0;
  hooks.rec = &rec;
  (void)eng.Get(1, 9, hooks);
  CHECK(rec.set_device_n >= 1);
}

TEST_CASE("getblas RED product call forwards device 0") {
  using Eng = vt::rocm::GetBlasDualSlotEngine<TrackingHooks>;
  Eng eng;
  TrackingHooks hooks;
  Table t;
  RunProductHops<Eng, TrackingHooks, ForwardDev0>(eng, hooks, &t, false);
  CHECK(t.h0 == t.h1);
  CHECK_FALSE(ProductTableHolds(t, false));
}

TEST_CASE("getblas RED product call forwards null stream") {
  using Eng = vt::rocm::GetBlasDualSlotEngine<TrackingHooks>;
  Eng eng;
  TrackingHooks hooks;
  Table t;
  RunProductHops<Eng, TrackingHooks, ForwardNullStream>(eng, hooks, &t, true);
  CHECK(t.rec.stream[0] != 3);
  CHECK_FALSE(ProductTableHolds(t, true));
}

TEST_CASE("getblas product source uses RocmProductGetBlasOn with device+stream") {
  const std::string hip = ReadText("src/vt/rocm/rocm_matmul_hipblaslt.hip");
  CHECK(hip.find("RocmProductGetBlasOn") != std::string::npos);
  CHECK(hip.find("RocmProductGetBlasOn(tls_slots, device, stream, hooks)") !=
        std::string::npos);
  CHECK(hip.find("tls_slots.Get(0,") == std::string::npos);
  CHECK(hip.find("Get(device, nullptr") == std::string::npos);
  CHECK(hip.find("static thread_local Tls tls;") == std::string::npos);
}

TEST_CASE("getblas hop 0-1-0-1 restores device on revisit") {
  using Eng = vt::rocm::GetBlasDualSlotEngine<TrackingHooks>;
  Eng eng;
  TrackingHooks hooks;
  Rec rec;
  hooks.rec = &rec;
  (void)vt::rocm::RocmProductGetBlasOn(eng, 0, 1, hooks);
  CHECK(rec.cur_dev == 0);
  (void)vt::rocm::RocmProductGetBlasOn(eng, 1, 2, hooks);
  CHECK(rec.cur_dev == 1);
  (void)vt::rocm::RocmProductGetBlasOn(eng, 0, 1, hooks);
  CHECK(rec.cur_dev == 0);
  CHECK(rec.last_set_device == 0);
  (void)vt::rocm::RocmProductGetBlasOn(eng, 1, 2, hooks);
  CHECK(rec.cur_dev == 1);
  CHECK(rec.last_set_device == 1);
}

TEST_CASE("getblas product 0-1-0-1 handle identity") {
  using Eng = vt::rocm::GetBlasDualSlotEngine<TrackingHooks>;
  Eng eng;
  TrackingHooks hooks;
  Rec rec;
  hooks.rec = &rec;
  const auto h0a = vt::rocm::RocmProductGetBlasOn(eng, 0, 1, hooks);
  const auto h1a = vt::rocm::RocmProductGetBlasOn(eng, 1, 2, hooks);
  const auto h0b = vt::rocm::RocmProductGetBlasOn(eng, 0, 1, hooks);
  const auto h1b = vt::rocm::RocmProductGetBlasOn(eng, 1, 2, hooks);
  CHECK(h0a == h0b);
  CHECK(h1a == h1b);
  CHECK(h0a != h1a);
  CHECK(rec.destroy[0] == 0);
  CHECK(rec.destroy[1] == 0);
  CHECK(rec.cur_dev == 1);
}

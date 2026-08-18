// #839 host lifetime + product-policy mutations for prefill peer.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "vt/rocm/rocm_gemma4_prefill_dequant_cache.h"

#ifndef VLLM_CPP_SOURCE_DIR
#define VLLM_CPP_SOURCE_DIR "."
#endif

namespace {

std::string ReadHip() {
  const std::string path = std::string(VLLM_CPP_SOURCE_DIR) + "/src/vt/rocm/rocm_gemma4_experts.hip";
  std::ifstream in(path);
  REQUIRE(in.good());
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string ReadHeader() {
  const std::string path =
      std::string(VLLM_CPP_SOURCE_DIR) + "/include/vt/rocm/rocm_gemma4_prefill_dequant_cache.h";
  std::ifstream in(path);
  REQUIRE(in.good());
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string ExtractFinish(const std::string& hip) {
  const auto a = hip.rfind("bool FinishGemma4Fp8ExpertGeGLUPrefillPeer");
  REQUIRE(a != std::string::npos);
  const auto b = hip.find("void PinGemma4Fp8ExpertHostCache", a);
  REQUIRE(b != std::string::npos);
  return hip.substr(a, b - a);
}

std::string ExtractRestoreComputeOrThrow(const std::string& hip) {
  const auto a = hip.find("void RestoreComputeOrThrow(int compute_dev)");
  REQUIRE(a != std::string::npos);
  const auto b = hip.find("\nbool RetirePinThenUnpin", a);
  REQUIRE(b != std::string::npos);
  return hip.substr(a, b - a);
}

// The HIP half of the two compile gates below needs a HIP compiler. This file
// used to name one absolute path from the contributor's box
// ("/opt/rocm-7.2.4/core-7.14/bin/hipcc") and CHECK that it returned 0, so on
// every machine without that exact path the gate FAILED instead of reporting
// that it had not run -- the whole build-test-cpu and both sanitize-cpu legs of
// #1047, green on main. Resolve the toolchain the way
// tests/vt/test_ops_getblas_product.cpp:15-21 resolves its precondition, and say
// out loud when it is absent.
//
// Absent hipcc does NOT exit 77 here, unlike getblas: that file is one HIP gate
// and nothing else, while this one carries 21 host-lifetime cases that are the
// only gate CI has on this change. Exiting would take those with it. The g++ leg
// of both compile gates still runs and still asserts, so neither case reports a
// zero-assertion pass; what is lost off ROCm is the HIP-header compile, which no
// CI runner here can perform. See `## Owed` in the spec.
const char* ResolveHipcc() {
  static const std::string resolved = [] {
    std::vector<std::string> candidates;
    for (const char* var : {"VLLM_CPP_HIPCC", "HIPCC"}) {
      if (const char* v = std::getenv(var); v != nullptr && v[0] != '\0') candidates.emplace_back(v);
    }
    if (const char* rp = std::getenv("ROCM_PATH"); rp != nullptr && rp[0] != '\0') {
      candidates.emplace_back(std::string(rp) + "/bin/hipcc");
    }
    candidates.emplace_back("hipcc");  // PATH
    candidates.emplace_back("/opt/rocm/bin/hipcc");
    for (const auto& c : candidates) {
      const std::string probe = c + " --version >/dev/null 2>&1";
      if (std::system(probe.c_str()) == 0) return c;
    }
    return std::string{};
  }();
  return resolved.empty() ? nullptr : resolved.c_str();
}

// Loud, and on stderr, so a reader of a CI log can tell a gate that ran from one
// that could not. Never silent: a skip nobody can see is the failure mode this
// repository's exit-77 convention exists to prevent.
void ReportHipccNotRun(const char* what) {
  std::fprintf(stderr,
               "\n*** HIP COMPILE GATE NOT RUN — no hipcc resolved, this is NOT a pass ***\n"
               "%s\n"
               "Set VLLM_CPP_HIPCC=/path/to/hipcc (or put hipcc on PATH, or set ROCM_PATH) to run it.\n",
               what);
}

struct RestoreGateRc {
  int compile = 127;
  int run = 127;
};

RestoreGateRc CompileAndRunProductRestore(const std::string& restore_fn, const std::string& compiler) {
  const std::string path = "/tmp/vllm_prefill_peer_restore_gate.cpp";
  const std::string bin = "/tmp/vllm_prefill_peer_restore_gate.bin";
  std::ofstream out(path);
  if (!out.good()) return {126, 126};
  out << R"GATE(
#include <stdexcept>
namespace vt {
namespace rocm {
struct RestoreFailed : std::runtime_error {
  RestoreFailed() : std::runtime_error("restore failed") {}
};
}
}
using hipError_t = int;
constexpr hipError_t hipSuccess = 0;
static hipError_t g_set_rc = 1;
hipError_t hipSetDevice(int) { return g_set_rc; }
)GATE";
  out << restore_fn << '\n';
  out << R"GATE(
int probe() {
  try {
    RestoreComputeOrThrow(3);
    return 0;
  } catch (const vt::rocm::RestoreFailed&) {
    return 1;
  } catch (...) {
    return 2;
  }
}
int main() {
  g_set_rc = 1;
  if (probe() != 1) return 10;
  g_set_rc = 0;
  if (probe() != 0) return 11;
  return 0;
}
)GATE";
  out.close();
  const std::string cmd = std::string("HIP_VISIBLE_DEVICES= ") + compiler + " -std=c++17 " + path +
                          " -o " + bin +
                          " >/tmp/vllm_prefill_peer_restore_gate.out 2>/tmp/vllm_prefill_peer_restore_gate.err";
  const int crc = std::system(cmd.c_str());
  if (crc != 0) return {crc, 127};
  return {0, std::system(bin.c_str())};
}

bool ProductFinishRetiresCstBeforeReuse(const std::string& finish) {
  const auto sync = finish.find("hipStreamSynchronize(cst)");
  const auto success =
      finish.find("tls.pending_M = 0;\n  RestoreComputeOrThrow(compute_dev); return true;");
  return sync != std::string::npos && success != std::string::npos && sync < success;
}

int CompileFinishCatchTu(const std::string& finish, const std::string& compiler) {
  const auto c1 = finish.find("} catch (const vt::rocm::RestoreFailed&)");
  if (c1 == std::string::npos) return 127;
  const std::string catches = finish.substr(c1);  // includes function closer
  const std::string path = "/tmp/vllm_prefill_peer_finish_catch_gate.cpp";
  std::ofstream out(path);
  if (!out.good()) return 126;
  out << R"GATE(
#include <exception>
namespace vt {
struct Queue {
  struct {
    int index = 0;
  } device;
  void* handle = nullptr;
};
namespace rocm {
struct RestoreFailed : std::exception {};
}
}
struct PeerSlot {
  int cache_pin = -1;
  bool rollback_armed = false;
  int pending_M = 0;
  bool eq_live = false;
  struct {
    void* handle = nullptr;
  } eq;
};
struct PeerTls {
  PeerSlot s[1];
};
PeerTls& PeerPipeTls() {
  static PeerTls t;
  return t;
}
using hipStream_t = void*;
bool RetirePinThenUnpin(PeerSlot&, hipStream_t, hipStream_t) { return true; }
void RestoreComputeOrThrow(int) {}
bool gate(vt::Queue& compute_q, int slot) {
  try {
    hipStream_t cst = static_cast<hipStream_t>(compute_q.handle);
    (void)cst;
    throw vt::rocm::RestoreFailed{};
  )GATE";
  out << catches;
  out.close();
  const std::string cmd = std::string("HIP_VISIBLE_DEVICES= ") + compiler +
                          " -fsyntax-only -std=c++17 " + path +
                          " -o /tmp/vllm_prefill_peer_finish_catch_gate.o >/tmp/vllm_prefill_peer_finish_catch_gate.out 2>/tmp/vllm_prefill_peer_finish_catch_gate.err";
  return std::system(cmd.c_str());
}

}  // namespace

TEST_CASE("prefill peer source has Launch/Finish and shared cache") {
  const std::string hip = ReadHip();
  CHECK(hip.find("LaunchGemma4Fp8ExpertGeGLUPrefillPeer") != std::string::npos);
  CHECK(hip.find("FinishGemma4Fp8ExpertGeGLUPrefillPeer") != std::string::npos);
  CHECK(hip.find("cache_pin") != std::string::npos);
  CHECK(hip.find("PeerSlot") != std::string::npos);
  CHECK(hip.find("PrefillDequantCacheT<HipPrefillCacheHooks>") != std::string::npos);
  CHECK(hip.find("ChoosePrefillRetire") != std::string::npos);
  CHECK(hip.find("this_gen_ev_e") != std::string::npos);
  CHECK(hip.find("RestoreComputeOrThrow") != std::string::npos);
  CHECK(hip.find("SameDevLife") != std::string::npos);
  const auto prefill = hip.find("bool RunGemma4Fp8ExpertGeGLUPrefillOnExpertDevice");
  const auto pinhost = hip.find("void PinGemma4Fp8ExpertHostCache");
  REQUIRE(prefill != std::string::npos);
  REQUIRE(pinhost != std::string::npos);
  CHECK(hip.find("LaunchGemma4Fp8ExpertGeGLUPrefillPeer", prefill) != std::string::npos);
}

TEST_CASE("prefill peer Ensure rejects live pin instead of FreeAll") {
  vt::rocm::PrefillDequantCacheHost cache;
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    REQUIRE(cache.Ensure(0, 4, 8));
  }
  void* gu = nullptr;
  void* dn = nullptr;
  int pin = -1;
  const char key = 'k';
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    REQUIRE(cache.GetLocked(&key, &gu, &dn, &pin));
    CHECK(cache.LivePins() == 1);
    CHECK_FALSE(cache.Ensure(0, 5, 8));
    CHECK(cache.I == 4);
    cache.UnpinLocked(pin);
  }
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    CHECK(cache.Ensure(0, 5, 8));
    CHECK(cache.I == 5);
  }
}

TEST_CASE("prefill peer pinned slot cannot be rewritten by second worker") {
  vt::rocm::PrefillDequantCacheHost cache;
  const char ka = 'a';
  const char kb = 'b';
  void* gu = nullptr;
  void* dn = nullptr;
  int pin_a = -1;
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    REQUIRE(cache.Ensure(0, 4, 8));
    REQUIRE(cache.GetLocked(&ka, &gu, &dn, &pin_a));
    int pin_b = -1;
    CHECK_FALSE(cache.GetLocked(&kb, &gu, &dn, &pin_b));
    CHECK(cache.slots[pin_a].key == &ka);
    cache.UnpinLocked(pin_a);
    REQUIRE(cache.GetLocked(&kb, &gu, &dn, &pin_b));
    CHECK(cache.slots[pin_b].key == &kb);
  }
}

TEST_CASE("prefill peer Launch/Finish pairing and fail-after-enqueue retires") {
  using vt::rocm::PrefillPeerFailAt;
  vt::rocm::PrefillDequantCacheHost cache;
  vt::rocm::PrefillPeerLife slot;
  const char key = 'k';
  REQUIRE(vt::rocm::HostLaunch(cache, slot, 0, &key, 8, PrefillPeerFailAt::None));
  CHECK(slot.pending_M == 8);
  CHECK(slot.cache_pin >= 0);
  CHECK(cache.LivePins() == 1);
  REQUIRE(vt::rocm::HostFinish(cache, slot, 8, PrefillPeerFailAt::None));
  CHECK(slot.pending_M == 0);
  CHECK(cache.LivePins() == 0);
  CHECK(slot.compute_restored);
  CHECK_FALSE(slot.ev_e_recorded);
  CHECK_FALSE(slot.this_gen_ev_e);

  vt::rocm::PrefillPeerLife s2;
  REQUIRE_FALSE(vt::rocm::HostLaunch(cache, s2, 0, &key, 8, PrefillPeerFailAt::AfterPinEnqueue));
  CHECK(cache.LivePins() == 0);
  CHECK(s2.work_enqueued);

  vt::rocm::PrefillPeerLife s3;
  REQUIRE_FALSE(vt::rocm::HostLaunch(cache, s3, 0, &key, 8, PrefillPeerFailAt::RecordEvE));
  CHECK(cache.LivePins() == 0);

  vt::rocm::PrefillPeerLife s3b;
  REQUIRE_FALSE(vt::rocm::HostLaunch(cache, s3b, 0, &key, 8, PrefillPeerFailAt::AfterRecord));
  CHECK(cache.LivePins() == 0);

  vt::rocm::PrefillPeerLife s4;
  REQUIRE(vt::rocm::HostLaunch(cache, s4, 0, &key, 8, PrefillPeerFailAt::None));
  REQUIRE_FALSE(vt::rocm::HostFinish(cache, s4, 8, PrefillPeerFailAt::AfterWait));
  CHECK(cache.LivePins() == 0);

  vt::rocm::PrefillPeerLife s5;
  REQUIRE(vt::rocm::HostLaunch(cache, s5, 0, &key, 8, PrefillPeerFailAt::None));
  REQUIRE_FALSE(vt::rocm::HostFinish(cache, s5, 8, PrefillPeerFailAt::AfterCopy));
  CHECK(cache.LivePins() == 0);
}

TEST_CASE("prefill peer wrapper uses slot 0 only") {
  const std::string hip = ReadHip();
  CHECK(hip.find("LaunchGemma4Fp8ExpertGeGLUPrefillPeer(compute_q, /*slot=*/0") != std::string::npos);
  CHECK(hip.find("FinishGemma4Fp8ExpertGeGLUPrefillPeer(compute_q, /*slot=*/0") != std::string::npos);
  CHECK(hip.find("PeerSlot s[1]") != std::string::npos);
  CHECK(hip.find("PeerSlot s[2]") == std::string::npos);
  CHECK(hip.find("if (slot != 0 || !x_compute") != std::string::npos);
  CHECK(hip.find("if (slot != 0 || !y_compute") != std::string::npos);
  CHECK(hip.find("slot < 0 || slot > 1") == std::string::npos);
}

TEST_CASE("prefill peer failed retirement quarantines pin") {
  vt::rocm::PrefillDequantCacheHost cache;
  vt::rocm::PrefillPeerLife slot;
  const char key = 'k';
  REQUIRE(vt::rocm::HostLaunch(cache, slot, 0, &key, 8, vt::rocm::PrefillPeerFailAt::None));
  REQUIRE_FALSE(vt::rocm::HostFinish(cache, slot, 8, vt::rocm::PrefillPeerFailAt::None,
                                     /*retire_ok=*/false));
  CHECK(slot.quarantined);
  CHECK(slot.cache_pin >= 0);
  CHECK(cache.LivePins() == 1);
}

TEST_CASE("prefill peer partial Ensure alloc is freed") {
  vt::rocm::PrefillDequantCacheHost cache;
  cache.hooks.fail_malloc_at = 2;
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    CHECK_FALSE(cache.Ensure(0, 4, 8));
    CHECK(cache.LiveAllocs() == 0);
    CHECK(cache.hooks.frees >= 1);
  }
}

TEST_CASE("prefill peer failed ready record keeps fill lease (no cross-stream reuse)") {
  vt::rocm::PrefillDequantCacheHost cache;
  const char ka = 'a';
  const char kb = 'b';
  void* gu = nullptr;
  void* dn = nullptr;
  int pin = -1;
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    REQUIRE(cache.Ensure(0, 4, 8));
    REQUIRE(cache.GetLocked(&ka, &gu, &dn, &pin));
    cache.UnpinLocked(pin);
    cache.hooks.fail_record = true;
    int fail_pin = -1;
    CHECK_FALSE(cache.GetLocked(&kb, &gu, &dn, &fail_pin));
    CHECK(fail_pin >= 0);
    CHECK(cache.slots[0].key == nullptr);
    CHECK_FALSE(cache.slots[0].ready);
    CHECK(cache.slots[0].filling);
    CHECK(cache.slots[0].fill_failed);
    CHECK(cache.LivePins() >= 1);
    CHECK(cache.hooks.fill_calls >= 1);
    int reuse = -1;
    CHECK_FALSE(cache.GetLocked(&ka, &gu, &dn, &reuse));
    CHECK(cache.slots[0].key == nullptr);
    REQUIRE_FALSE(cache.RetireFillLocked(fail_pin, /*retire_ok=*/false));
    CHECK(cache.slots[0].fill_failed);
    CHECK(cache.LivePins() >= 1);
  }
}

TEST_CASE("prefill peer stale prior ev_e is not current rollback target") {
  vt::rocm::PrefillPeerLife life;
  life.ev_e_recorded = true;  // leftover from previous Finish
  life.this_gen_ev_e = false;
  life.rollback_armed = true;
  life.cache_pin = -1;
  CHECK(vt::rocm::ChoosePrefillRetire(life) == vt::rocm::PrefillRetireTarget::ComputeStream);
  life.this_gen_ev_e = true;
  CHECK(vt::rocm::ChoosePrefillRetire(life) == vt::rocm::PrefillRetireTarget::RecordedEvent);
  life.this_gen_ev_e = false;
  life.rollback_armed = false;
  life.work_on_expert = true;
  CHECK(vt::rocm::ChoosePrefillRetire(life) == vt::rocm::PrefillRetireTarget::ExpertStream);
}

TEST_CASE("prefill peer two-invocation: second enqueue fail syncs current stream") {
  vt::rocm::PrefillDequantCacheHost cache;
  vt::rocm::PrefillPeerLife slot;
  const char key = 'k';
  REQUIRE(vt::rocm::HostLaunch(cache, slot, 0, &key, 8, vt::rocm::PrefillPeerFailAt::None));
  REQUIRE(vt::rocm::HostFinish(cache, slot, 8, vt::rocm::PrefillPeerFailAt::None));
  CHECK_FALSE(slot.this_gen_ev_e);
  CHECK_FALSE(slot.ev_e_recorded);
  REQUIRE_FALSE(
      vt::rocm::HostLaunch(cache, slot, 0, &key, 8, vt::rocm::PrefillPeerFailAt::AfterFirstEnqueue));
  CHECK(vt::rocm::ChoosePrefillRetire(slot) != vt::rocm::PrefillRetireTarget::RecordedEvent);
  CHECK(slot.work_enqueued);
}

TEST_CASE("prefill peer restore failure is fatal not false") {
  vt::rocm::PrefillDequantCacheHost cache;
  vt::rocm::PrefillPeerLife slot;
  const char key = 'k';
  CHECK_THROWS_AS(
      vt::rocm::HostLaunch(cache, slot, 0, &key, 8, vt::rocm::PrefillPeerFailAt::None,
                           /*retire_ok=*/true, /*restore_ok=*/false),
      vt::rocm::RestoreFailed);
  CHECK(slot.pending_M == 0);
  CHECK(slot.cache_pin < 0);
  CHECK(cache.LivePins() == 0);
}

TEST_CASE("prefill peer same-dev quarantine blocks reenter and reconfigure") {
  vt::rocm::SameDevLife life;
  CHECK(life.CanEnter());
  CHECK(life.CanReconfigure());
  life.Quarantine(0, 0);
  CHECK_FALSE(life.CanEnter());
  CHECK_FALSE(life.CanReconfigure());
  life.ClearPin();
  CHECK(life.CanEnter());
}

TEST_CASE("prefill peer same-dev pin held through GEMM readers") {
  vt::rocm::PrefillDequantCacheHost cache;
  vt::rocm::SameDevLife life;
  vt::rocm::SameDevSession sess;
  sess.cache = &cache;
  sess.life = &life;
  const char key = 'k';
  REQUIRE(sess.Acquire(&key, 0));
  CHECK(cache.LivePins() == 1);
  CHECK(life.cache_pin >= 0);
  REQUIRE(sess.RunGemmReaders());
  CHECK_FALSE(sess.unpinned_before_gemm);
  CHECK(cache.LivePins() == 1);
  REQUIRE(sess.Retire(/*sync_ok=*/true));
  CHECK(cache.LivePins() == 0);
}

TEST_CASE("prefill peer same-dev early unpin before GEMM is RED") {
  vt::rocm::PrefillDequantCacheHost cache;
  vt::rocm::SameDevLife life;
  vt::rocm::SameDevSession sess;
  sess.cache = &cache;
  sess.life = &life;
  const char key = 'k';
  REQUIRE(sess.Acquire(&key, 0));
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    cache.UnpinLocked(sess.pin);  // product mutation analogue
  }
  CHECK_FALSE(sess.RunGemmReaders());
  CHECK(sess.unpinned_before_gemm);
}

TEST_CASE("prefill peer LifeFromSlot does not remap this_gen to ev_e_recorded") {
  struct FakeSlot {
    int cache_pin = 3;
    int cache_dev = 1;
    bool this_gen_ev_e = false;
    bool ev_e_recorded = true;
    bool rollback_armed = true;
    bool quarantined = false;
    bool work_on_compute = true;
    bool work_on_expert = false;
    bool fill_lease = false;
  } tls;
  const auto life = vt::rocm::LifeFromSlot(tls);
  CHECK_FALSE(life.this_gen_ev_e);
  CHECK(life.ev_e_recorded);
  CHECK(vt::rocm::ChoosePrefillRetire(life) == vt::rocm::PrefillRetireTarget::ComputeStream);
}

TEST_CASE("prefill peer product RestoreComputeOrThrow no-op mutation is RED") {
  const std::string restore = ExtractRestoreComputeOrThrow(ReadHip());
  CHECK(restore.find("if (hipSetDevice(compute_dev) != hipSuccess) throw vt::rocm::RestoreFailed{}") !=
        std::string::npos);
  const auto gxx = CompileAndRunProductRestore(restore, "g++");
  CHECK(gxx.compile == 0);
  CHECK(gxx.run == 0);

  const std::string line =
      "  if (hipSetDevice(compute_dev) != hipSuccess) throw vt::rocm::RestoreFailed{};\n";
  std::string mut = restore;
  const auto pos = mut.find(line);
  REQUIRE(pos != std::string::npos);
  mut.replace(pos, line.size(), "  (void)compute_dev;\n");
  const auto mut_gxx = CompileAndRunProductRestore(mut, "g++");
  CHECK(mut_gxx.compile == 0);
  // A mutant that fails to COMPILE reads as a caught defect while proving
  // nothing, so the compile status is asserted separately from the run status.
  CHECK(mut_gxx.run != 0);

  if (const char* hipcc = ResolveHipcc(); hipcc != nullptr) {
    const std::string hip_cxx = std::string(hipcc) + " -x c++";
    const auto hip = CompileAndRunProductRestore(restore, hip_cxx);
    CHECK(hip.compile == 0);
    CHECK(hip.run == 0);
    const auto mut_hip = CompileAndRunProductRestore(mut, hip_cxx);
    CHECK(mut_hip.compile == 0);
    CHECK(mut_hip.run != 0);
  } else {
    ReportHipccNotRun("RestoreComputeOrThrow no-op mutation, HIP-compiler leg");
  }

  const std::string hdr = ReadHeader();
  CHECK(hdr.find("if (std::uncaught_exceptions() > 0) return;") == std::string::npos);
  CHECK(hdr.find("if (!set(dev)) std::terminate();") != std::string::npos);
  CHECK(hdr.find("PublishThenRestoreOrThrow") != std::string::npos);
}

TEST_CASE("prefill peer HostLaunch fill-lease fail retires outside lock") {
  vt::rocm::PrefillDequantCacheHost cache;
  vt::rocm::PrefillPeerLife slot;
  const char key = 'k';
  cache.hooks.fail_record = true;
  REQUIRE_FALSE(vt::rocm::HostLaunch(cache, slot, 0, &key, 8, vt::rocm::PrefillPeerFailAt::None));
  CHECK(slot.fill_lease == false);  // cleared on successful retire
  CHECK(slot.cache_pin < 0);
  CHECK(cache.LivePins() == 0);
  CHECK_FALSE(cache.slots[0].filling);
  CHECK_FALSE(cache.slots[0].fill_failed);
  cache.hooks.fail_record = false;
  vt::rocm::PrefillPeerLife s2;
  REQUIRE(vt::rocm::HostLaunch(cache, s2, 0, &key, 8, vt::rocm::PrefillPeerFailAt::None));
  CHECK(s2.cache_pin >= 0);
}

TEST_CASE("prefill peer fill-lease failed retire quarantines") {
  vt::rocm::PrefillDequantCacheHost cache;
  vt::rocm::PrefillPeerLife slot;
  const char key = 'k';
  cache.hooks.fail_record = true;
  REQUIRE_FALSE(vt::rocm::HostLaunch(cache, slot, 0, &key, 8, vt::rocm::PrefillPeerFailAt::None,
                                     /*retire_ok=*/false));
  CHECK(slot.quarantined);
  CHECK(slot.cache_pin >= 0);
  CHECK(cache.LivePins() >= 1);
  const bool lease_held = cache.slots[0].filling || cache.slots[0].fill_failed;
  CHECK(lease_held);
}


TEST_CASE("prefill peer two-compute-stream output-copy retirement") {
  vt::rocm::OutputCopyGate gate;
  gate.Enqueue(1);
  CHECK_FALSE(gate.CanReuseScratch());
  CHECK_FALSE(gate.Retire(2));
  CHECK_FALSE(gate.CanReuseScratch());
  REQUIRE(gate.Retire(1));
  CHECK(gate.CanReuseScratch());

  vt::rocm::PrefillDequantCacheHost cache;
  vt::rocm::PrefillPeerLife slot;
  const char key = 'k';
  REQUIRE(vt::rocm::HostLaunch(cache, slot, 0, &key, 8, vt::rocm::PrefillPeerFailAt::None));
  // Copy on stream 1, retire on stream 2: product mutation analogue is RED.
  CHECK_FALSE(vt::rocm::HostFinish(cache, slot, 8, vt::rocm::PrefillPeerFailAt::None, true, true, 1, 2));
  CHECK_FALSE(slot.output_copy.CanReuseScratch());
  CHECK(slot.pending_M == 8);
  vt::rocm::PrefillPeerLife reuse;
  reuse.output_copy = slot.output_copy;
  CHECK_FALSE(vt::rocm::HostLaunch(cache, reuse, 0, &key, 8, vt::rocm::PrefillPeerFailAt::None));
  REQUIRE(vt::rocm::HostFinish(cache, slot, 8, vt::rocm::PrefillPeerFailAt::None, true, true, 1, 1));
  CHECK(slot.output_copy.CanReuseScratch());
  CHECK(slot.pending_M == 0);
}

TEST_CASE("prefill peer Finish catch HIP/C++ compile gate") {
  const std::string finish = ExtractFinish(ReadHip());
  CHECK(CompileFinishCatchTu(finish, "g++") == 0);
  std::string mut_cst = finish;
  const std::string decl = "    hipStream_t cst = static_cast<hipStream_t>(compute_q.handle);\n";
  auto p1 = mut_cst.find(decl);
  REQUIRE(p1 != std::string::npos);
  mut_cst.erase(p1, decl.size());
  auto p2 = mut_cst.find(decl);
  if (p2 != std::string::npos) mut_cst.erase(p2, decl.size());
  CHECK(CompileFinishCatchTu(mut_cst, "g++") != 0);

  if (const char* hipcc = ResolveHipcc(); hipcc != nullptr) {
    const std::string hip_cxx = std::string(hipcc) + " -x c++";
    CHECK(CompileFinishCatchTu(finish, hip_cxx) == 0);
    CHECK(CompileFinishCatchTu(mut_cst, hip_cxx) != 0);
  } else {
    ReportHipccNotRun("Finish catch compile gate, HIP-compiler leg");
  }
}

TEST_CASE("prefill peer product two-stream output-copy mutation is RED") {
  const std::string finish = ExtractFinish(ReadHip());
  CHECK(ProductFinishRetiresCstBeforeReuse(finish));
  const std::string block =
      "  if (hipStreamSynchronize(cst) != hipSuccess) {\n"
      "    tls.quarantined = true;\n"
      "    RestoreComputeOrThrow(compute_dev); return false;\n"
      "  }\n";
  std::string mut = finish;
  const auto pos = mut.find(block);
  REQUIRE(pos != std::string::npos);
  mut.erase(pos, block.size());
  CHECK_FALSE(ProductFinishRetiresCstBeforeReuse(mut));
}

TEST_CASE("prefill peer source: product uses shared retire/restore policy") {
  const std::string hip = ReadHip();
  CHECK(hip.find("ChoosePrefillRetire(PeerLifeView") != std::string::npos);
  CHECK(hip.find("LifeFromSlot") != std::string::npos);
  CHECK(hip.find("RestoreComputeOrThrow") != std::string::npos);
  CHECK(hip.find("bool RestoreComputeDevOrFatal") == std::string::npos);
  CHECK(hip.find("life.Quarantine") != std::string::npos);
  CHECK(hip.find("life.PersistPin") != std::string::npos);
  CHECK(hip.find("work_on_compute = true") != std::string::npos);
  CHECK(hip.find("FailLaunchRestore(tls, est, cst, compute_dev)") != std::string::npos);
  CHECK(hip.find("ReleaseObservedPinLocked") != std::string::npos);
  CHECK(hip.find("PublishThenRestoreOrThrow") != std::string::npos);
  CHECK(hip.find("tls.fill_lease = true") != std::string::npos);
  const auto fin = hip.find("bool FinishGemma4Fp8ExpertGeGLUPrefillPeer");
  REQUIRE(fin != std::string::npos);
  const auto fin_def = hip.rfind("bool FinishGemma4Fp8ExpertGeGLUPrefillPeer");
  REQUIRE(fin_def != std::string::npos);
  const auto fin_end = hip.find("void PinGemma4Fp8ExpertHostCache", fin_def);
  REQUIRE(fin_end != std::string::npos);
  const std::string finish = hip.substr(fin_def, fin_end - fin_def);
  const auto catch_all = finish.find("catch (...)");
  REQUIRE(catch_all != std::string::npos);
  const std::string catch_body = finish.substr(catch_all);
  CHECK(catch_body.find("hipStream_t cst = static_cast<hipStream_t>(compute_q.handle);") !=
        std::string::npos);
  const auto sync_cst = finish.find("hipStreamSynchronize(cst)");
  const auto success_clear =
      finish.find("tls.pending_M = 0;\n  RestoreComputeOrThrow(compute_dev); return true;");
  REQUIRE(sync_cst != std::string::npos);
  REQUIRE(success_clear != std::string::npos);
  CHECK(sync_cst < success_clear);
  const auto ret = hip.find("bool RetirePinThenUnpin");
  REQUIRE(ret != std::string::npos);
  const auto ret_end = hip.find("void FailLaunchRestore", ret);
  REQUIRE(ret_end != std::string::npos);
  const std::string retire = hip.substr(ret, ret_end - ret);
  CHECK(retire.find("(void)hipEventSynchronize") == std::string::npos);
  CHECK(retire.find("ChoosePrefillRetire") != std::string::npos);
  CHECK(retire.find("PrefillRetireTarget::ComputeStream") != std::string::npos);
  CHECK(retire.find("PrefillRetireTarget::ExpertStream") != std::string::npos);
  CHECK(retire.find("ReleaseObservedPinLocked") != std::string::npos);
  CHECK(retire.find("UnpinLocked(tls.cache_pin)") == std::string::npos);

  // After first successful enqueue is armed, bare restore-return is forbidden.
  const auto armed = hip.find("tls.rollback_armed = true");
  REQUIRE(armed != std::string::npos);
  const auto rec_end = hip.find("PublishThenRestoreOrThrow", armed);
  REQUIRE(rec_end != std::string::npos);
  const std::string after = hip.substr(armed, rec_end - armed);
  CHECK(after.find("RestoreComputeOrThrow(compute_dev); return false;") == std::string::npos);
  CHECK(after.find("FailLaunchRestore(tls, est, cst, compute_dev)") != std::string::npos);

  // Same-dev: no UnpinLocked between GetLocked and MatmulBT.
  const auto sd = hip.find("if (expert_dev == compute_dev)");
  REQUIRE(sd != std::string::npos);
  const auto launch = hip.find("LaunchGemma4Fp8ExpertGeGLUPrefillPeer(compute_q, /*slot=*/0", sd);
  REQUIRE(launch != std::string::npos);
  const std::string same = hip.substr(sd, launch - sd);
  const auto gl = same.find("GetLocked");
  const auto mm = same.find("MatmulBT");
  REQUIRE(gl != std::string::npos);
  REQUIRE(mm != std::string::npos);
  const auto unpin = same.find("UnpinLocked", gl);
  const bool unpin_after_gemm = unpin == std::string::npos || unpin > mm;
  CHECK(unpin_after_gemm);
}

TEST_CASE("prefill peer this_gen remap mutation is RED") {
  const std::string hip = ReadHip();
  CHECK(hip.find("life.this_gen_ev_e = tls.ev_e_recorded") == std::string::npos);
  CHECK(hip.find("PeerLifeView(const PeerSlot& tls) { return vt::rocm::LifeFromSlot(tls); }") !=
        std::string::npos);
}

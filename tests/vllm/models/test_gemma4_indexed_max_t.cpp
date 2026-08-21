// #838 host gates: product-loop tensor oracle, single-scale, retire-before-pool.
#include <doctest/doctest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "vllm/model_executor/models/gemma4_indexed_gate.h"

#ifndef VLLM_CPP_SOURCE_DIR
#define VLLM_CPP_SOURCE_DIR "."
#endif

using vllm::Gemma4ApplyHostExpertScaleOnce;
using vllm::Gemma4IndexedCall;
using vllm::Gemma4IndexedDispatchTokens;
using vllm::Gemma4IndexedHelperHits;
using vllm::Gemma4IndexedHostApplyToken;
using vllm::Gemma4IndexedHostSerialRef;
using vllm::Gemma4IndexedOkT;
using vllm::Gemma4IndexedArgsEq;
using vllm::Gemma4IndexedArm;
using vllm::Gemma4IndexedFailPathRetireThenMaybeRelease;
using vllm::Gemma4IndexedHelperArgs;
using vllm::Gemma4IndexedMayReleaseToPool;
using vllm::Gemma4IndexedOracleClose;
using vllm::Gemma4IndexedPackArgs;
using vllm::Gemma4IndexedRunSelectedArm;
using vllm::Gemma4IndexedScratchChoice;
using vllm::Gemma4IndexedScratchKind;
using vllm::Gemma4IndexedScratchKindFor;
using vllm::Gemma4IndexedScratchValidForT;
using vllm::Gemma4IndexedSelectArm;
using vllm::Gemma4IndexedTokenOffsets;
using vllm::ParseGemma4DecodeIndexedMaxT;
using vllm::kGemma4PrefillBatchMinT;

namespace {

std::string ReadText(const char* rel) {
  const std::string path = std::string(VLLM_CPP_SOURCE_DIR) + "/" + rel;
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Slice ONE function definition, signature through its brace-matched closing
// brace. The obvious alternative -- slice from this symbol to the next one --
// is not a slice of this function at all; it is a slice of the GAP, so any
// unrelated insertion after the closing brace lands inside it and reds a source
// invariant that the function still satisfies. Skips a forward declaration by
// requiring the first `{` to precede the first `;`. Braces inside string or
// character literals would fool the matcher; the functions asserted on here
// contain none, and a reviewer adding one must re-check this helper.
std::string FunctionBody(const std::string& src, const std::string& signature) {
  for (auto sig = src.find(signature); sig != std::string::npos;
       sig = src.find(signature, sig + 1)) {
    const auto open = src.find('{', sig);
    if (open == std::string::npos) return {};
    const auto semi = src.find(';', sig);
    if (semi != std::string::npos && semi < open) continue;  // declaration
    int depth = 0;
    for (size_t i = open; i < src.size(); ++i) {
      if (src[i] == '{') {
        ++depth;
      } else if (src[i] == '}' && --depth == 0) {
        return src.substr(sig, i - sig + 1);
      }
    }
    return {};
  }
  return {};
}

struct WritingHelper {
  bool peer_expected = false;
  int fail_at = -1;
  bool fail_after_enqueue = false;
  int calls = 0;
  int restores = 0;
  int64_t H = 0;
  int top_k = 0;

  bool operator()(const Gemma4IndexedCall<float, float>& c) {
    ++calls;
    REQUIRE(c.peer == peer_expected);
    const auto want = Gemma4IndexedTokenOffsets(c.t, H, top_k);
    REQUIRE(c.off.x_elems == want.x_elems);
    REQUIRE(c.off.y_elems == want.y_elems);
    REQUIRE(c.off.route == want.route);
    if (fail_at >= 0 && c.t == fail_at) {
      if (fail_after_enqueue) {
        Gemma4IndexedHostApplyToken(c.y, c.x, c.ri, c.rw, H, top_k);
      }
      return false;
    }
    Gemma4IndexedHostApplyToken(c.y, c.x, c.ri, c.rw, H, top_k);
    return true;
  }
};

void FillNontrivial(std::vector<float>& x, std::vector<int32_t>& ri, std::vector<float>& rw,
                    int64_t T, int64_t H, int top_k) {
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t h = 0; h < H; ++h) {
      x[static_cast<size_t>(t * H + h)] = (h == 0 && t == 1) ? 0.f : static_cast<float>(t + 1) * 0.25f +
                                                                         static_cast<float>(h) * 0.125f;
    }
    for (int g = 0; g < top_k; ++g) {
      ri[static_cast<size_t>(t * top_k + g)] = static_cast<int32_t>((t + g) % 8);
      rw[static_cast<size_t>(t * top_k + g)] = (g == 3 && t == 0) ? 0.f : 0.5f + 0.05f * static_cast<float>(g);
    }
  }
}

}  // namespace

TEST_CASE("gemma4 indexed-max-t: env parse") {
  CHECK(ParseGemma4DecodeIndexedMaxT(nullptr) == 63);
  CHECK(ParseGemma4DecodeIndexedMaxT("1") == 1);
  CHECK(ParseGemma4DecodeIndexedMaxT("64") == 63);
}

TEST_CASE("gemma4 indexed-max-t: host predicate table") {
  const int64_t unset63 = ParseGemma4DecodeIndexedMaxT(nullptr);
  const int64_t env1 = ParseGemma4DecodeIndexedMaxT("1");
  CHECK(Gemma4IndexedOkT(1, unset63, 8, true));
  CHECK(Gemma4IndexedOkT(19, unset63, 8, true));
  CHECK_FALSE(Gemma4IndexedOkT(19, env1, 8, true));
  CHECK_FALSE(Gemma4IndexedOkT(64, unset63, 8, true));
  CHECK(Gemma4IndexedOkT(63, unset63, 8, true));
  CHECK(kGemma4PrefillBatchMinT == 64);
}

TEST_CASE("gemma4 indexed-max-t: tensor oracle T={2,19,63} x {same,peer}") {
  for (bool peer : {false, true}) {
    for (int64_t T : {int64_t{2}, int64_t{19}, int64_t{63}}) {
      const int64_t H = 8;
      const int top_k = 8;
      std::vector<float> y(static_cast<size_t>(T * H), 99.f);
      std::vector<float> x(static_cast<size_t>(T * H), 0.f);
      std::vector<int32_t> ri(static_cast<size_t>(T * top_k), 0);
      std::vector<float> rw(static_cast<size_t>(T * top_k), 0.f);
      std::vector<float> ref(static_cast<size_t>(T * H), 0.f);
      FillNontrivial(x, ri, rw, T, H, top_k);
      Gemma4IndexedHostSerialRef(x.data(), ri.data(), rw.data(), ref.data(), T, H, top_k);
      WritingHelper fake;
      fake.peer_expected = peer;
      fake.H = H;
      fake.top_k = top_k;
      const uint64_t hits0 = Gemma4IndexedHelperHits().load();
      const auto disp = Gemma4IndexedDispatchTokens(
          T, H, top_k, peer, y.data(), x.data(), ri.data(), rw.data(),
          [&](const Gemma4IndexedCall<float, float>& c) { return fake(c); }, [&] { ++fake.restores; });
      REQUIRE(disp.ok);
      CHECK(disp.hits == static_cast<uint64_t>(T));
      CHECK(fake.calls == static_cast<int>(T));
      CHECK(disp.y_owner == static_cast<void*>(y.data()));
      CHECK(Gemma4IndexedHelperHits().load() == hits0 + static_cast<uint64_t>(T));
      float mad = 0.f;
      REQUIRE(Gemma4IndexedOracleClose(y.data(), ref.data(), T * H, &mad));
      CHECK(mad == doctest::Approx(0.f));
      bool any_nz = false, any_z = false;
      for (float v : ref) {
        if (v == 0.f) any_z = true;
        else any_nz = true;
      }
      CHECK(any_nz);
      CHECK(any_z);
      y[0] = 123.f;
      CHECK(y[0] == 123.f);
    }
  }
}

TEST_CASE("gemma4 indexed-max-t: RED wrong stride corrupts output") {
  const int64_t T = 19, H = 8;
  const int top_k = 8;
  std::vector<float> y(static_cast<size_t>(T * H), 0.f);
  std::vector<float> x(static_cast<size_t>(T * H), 0.f);
  std::vector<int32_t> ri(static_cast<size_t>(T * top_k), 0);
  std::vector<float> rw(static_cast<size_t>(T * top_k), 0.f);
  std::vector<float> ref(static_cast<size_t>(T * H), 0.f);
  FillNontrivial(x, ri, rw, T, H, top_k);
  Gemma4IndexedHostSerialRef(x.data(), ri.data(), rw.data(), ref.data(), T, H, top_k);
  for (int64_t t = 0; t < T; ++t) {
    const int64_t bad = t;  // t as t, not t*H
    Gemma4IndexedHostApplyToken(y.data() + bad, x.data() + t * H, ri.data() + t * top_k,
                                rw.data() + t * top_k, H, top_k);
  }
  float mad = 0.f;
  CHECK_FALSE(Gemma4IndexedOracleClose(y.data(), ref.data(), T * H, &mad));
}

TEST_CASE("gemma4 indexed-max-t: RED T=1 TLS owner is invalid for T>1") {
  const int64_t T = 19, H = 8;
  float tls1[8] = {};
  std::vector<float> owned(static_cast<size_t>(T * H), 0.f);
  CHECK(Gemma4IndexedScratchKindFor(1) == Gemma4IndexedScratchKind::TlsT1);
  CHECK(Gemma4IndexedScratchKindFor(T) == Gemma4IndexedScratchKind::OwnedTH);
  Gemma4IndexedScratchChoice tls{Gemma4IndexedScratchKind::TlsT1, tls1, H};
  CHECK_FALSE(Gemma4IndexedScratchValidForT(tls, T, H));
  Gemma4IndexedScratchChoice good{Gemma4IndexedScratchKindFor(T), owned.data(), T * H};
  CHECK(Gemma4IndexedScratchValidForT(good, T, H));
  CHECK(good.y != static_cast<void*>(tls1));
}

TEST_CASE("gemma4 indexed-max-t: release-before-retire is RED; fail-path retires while owned") {
  CHECK_FALSE(Gemma4IndexedMayReleaseToPool(/*enqueued=*/true, /*retire_observed=*/false));
  CHECK(Gemma4IndexedMayReleaseToPool(true, true));
  CHECK(Gemma4IndexedMayReleaseToPool(false, false));
  bool owned = true, released = false, retire_ok = false;
  CHECK(Gemma4IndexedFailPathRetireThenMaybeRelease(true, owned, released, retire_ok,
                                                    [] { return true; }));
  CHECK(retire_ok);
  CHECK(released);
  CHECK_FALSE(owned);
  owned = true;
  released = false;
  retire_ok = true;
  CHECK_FALSE(Gemma4IndexedFailPathRetireThenMaybeRelease(true, owned, released, retire_ok,
                                                          [] { return false; }));
  CHECK_FALSE(retire_ok);
  CHECK_FALSE(released);
  CHECK(owned);  // quarantined
}

TEST_CASE("gemma4 indexed-max-t: independent serial ref RED on candidate arithmetic/route") {
  const int64_t T = 19, H = 8;
  const int top_k = 8;
  std::vector<float> x(static_cast<size_t>(T * H), 0.f);
  std::vector<int32_t> ri(static_cast<size_t>(T * top_k), 0);
  std::vector<float> rw(static_cast<size_t>(T * top_k), 0.f);
  std::vector<float> ref(static_cast<size_t>(T * H), 0.f);
  std::vector<float> bad(static_cast<size_t>(T * H), 0.f);
  FillNontrivial(x, ri, rw, T, H, top_k);
  Gemma4IndexedHostSerialRef(x.data(), ri.data(), rw.data(), ref.data(), T, H, top_k);
  for (int64_t t = 0; t < T; ++t) {
    // mutated candidate: extra *2, does not go through SerialRef
    Gemma4IndexedHostApplyToken(bad.data() + t * H, x.data() + t * H, ri.data() + t * top_k,
                                rw.data() + t * top_k, H, top_k);
    for (int64_t h = 0; h < H; ++h) bad[static_cast<size_t>(t * H + h)] *= 2.f;
  }
  float mad = 0.f;
  CHECK_FALSE(Gemma4IndexedOracleClose(bad.data(), ref.data(), T * H, &mad));
}

TEST_CASE("gemma4 indexed-max-t: production selector arm/args identity; swap is RED") {
  int same_n = 0, peer_n = 0;
  float same_out = 0.f, peer_out = 0.f;
  auto same = [&] {
    ++same_n;
    same_out = 1.f;
    return true;
  };
  auto peer = [&] {
    ++peer_n;
    peer_out = 2.f;
    return true;
  };
  CHECK(Gemma4IndexedSelectArm(true, false) == Gemma4IndexedArm::SameDev);
  CHECK(Gemma4IndexedSelectArm(false, true) == Gemma4IndexedArm::Peer);
  REQUIRE(Gemma4IndexedRunSelectedArm(Gemma4IndexedArm::SameDev, same, peer));
  CHECK(same_n == 1);
  CHECK(peer_n == 0);
  CHECK(same_out == 1.f);
  REQUIRE(Gemma4IndexedRunSelectedArm(Gemma4IndexedArm::Peer, same, peer));
  CHECK(peer_n == 1);
  CHECK(peer_out == 2.f);
  auto swapped = [&](Gemma4IndexedArm arm) {
    return Gemma4IndexedRunSelectedArm(arm, peer, same);
  };
  same_n = peer_n = 0;
  REQUIRE(swapped(Gemma4IndexedArm::SameDev));
  CHECK(peer_n == 1);
  CHECK(same_n == 0);
  CHECK(peer_out == 2.f);
  float y = 0, x = 0, rw = 0;
  int32_t ri = 0;
  const auto want = Gemma4IndexedPackArgs(&y, &x, &ri, &rw);
  const auto swapped_args = Gemma4IndexedPackArgs(&x, &y, &ri, &rw);
  CHECK(Gemma4IndexedArgsEq(want, Gemma4IndexedPackArgs(&y, &x, &ri, &rw)));
  CHECK_FALSE(Gemma4IndexedArgsEq(want, swapped_args));
}

TEST_CASE("gemma4 indexed-max-t: fallback scale is once, not s^2") {
  const int64_t T = 19;
  const int top_k = 8;
  const int64_t E = 8;
  std::vector<float> orig(static_cast<size_t>(T * top_k), 0.5f);
  std::vector<int32_t> hi(static_cast<size_t>(T * top_k));
  std::vector<float> hscale(static_cast<size_t>(E), 3.f);
  for (size_t i = 0; i < hi.size(); ++i) hi[i] = static_cast<int32_t>(i % 8);
  std::vector<float> fallback = orig;
  Gemma4ApplyHostExpertScaleOnce(fallback.data(), hi.data(), hscale.data(), T, top_k, E, false);
  for (size_t i = 0; i < orig.size(); ++i) {
    CHECK(fallback[i] == doctest::Approx(orig[i] * 3.f));
    CHECK(fallback[i] != doctest::Approx(orig[i] * 9.f));
  }
}

TEST_CASE("gemma4 indexed-max-t: source invariants") {
  const std::string moe = ReadText("src/vllm/model_executor/models/gemma4_moe.cpp");
  const std::string hip = ReadText("src/vt/rocm/rocm_gemma4_experts.hip");
  REQUIRE_FALSE(moe.empty());
  REQUIRE_FALSE(hip.empty());
  CHECK(moe.find("Gemma4IndexedDispatchTokens") != std::string::npos);
  CHECK(moe.find("Gemma4IndexedSelectArm") != std::string::npos);
  CHECK(moe.find("Gemma4IndexedRunSelectedArm") != std::string::npos);
  CHECK(moe.find("Gemma4IndexedScratchKindFor") != std::string::npos);
  CHECK(moe.find("retire-before-acc_idx-dtor") != std::string::npos);
  CHECK(moe.find("RetireGemma4Fp8TopKIndexedPeer") != std::string::npos);
  CHECK(moe.find("rw_idx_owned") != std::string::npos);
  CHECK(moe.find("struct RwIdxTls") != std::string::npos);
  CHECK(moe.find("static thread_local RwIdxTls rwt") != std::string::npos);
  // T=1 scaled rw is TLS-stable; only T>1 Release()s a pooled copy.
  const auto rwt_at = moe.find("struct RwIdxTls");
  REQUIRE(rwt_at != std::string::npos);
  const auto t1_arm = moe.find("Gemma4IndexedScratchKind::TlsT1", rwt_at);
  REQUIRE(t1_arm != std::string::npos);
  const auto owned_emplace = moe.find("rw_idx_owned.emplace", rwt_at);
  REQUIRE(owned_emplace != std::string::npos);
  CHECK(owned_emplace > t1_arm);  // pooled emplace is the T>1 branch
  CHECK(moe.find("rwt.buf->Release") == std::string::npos);
  CHECK(moe.find("rw_idx_owned->Release()") != std::string::npos);
  // Fresh copy + scale land on the TLS/owned dest every call, never on `rw`.
  const auto copy_at = moe.find("d.b.Copy(d.q, scaled->ptr(), rw.ptr()", rwt_at);
  REQUIRE(copy_at != std::string::npos);
  const auto key_if = moe.find("rwt.dev != compute_dev || rwt.n != n || !rwt.buf", rwt_at);
  REQUIRE(key_if != std::string::npos);
  CHECK(copy_at > key_if);  // copy is outside the TLS-miss emplace
  CHECK(moe.find("ApplyExpertScaleRw(d.q, static_cast<float*>(scaled->ptr())", rwt_at) !=
        std::string::npos);
  CHECK(moe.find("ApplyExpertScaleRw(d.q, static_cast<float*>(rw.ptr())", rwt_at) ==
        std::string::npos);
  CHECK(moe.find("ExpertGeGLUFp8TopKIndexedBatched") == std::string::npos);
  const auto serial = ReadText("include/vllm/model_executor/models/gemma4_indexed_gate.h");
  const auto sref = serial.find("Gemma4IndexedHostSerialRef");
  REQUIRE(sref != std::string::npos);
  const auto sref_end = serial.find("Gemma4IndexedOracleClose", sref);
  REQUIRE(sref_end != std::string::npos);
  CHECK(serial.substr(sref, sref_end - sref).find("Gemma4IndexedHostApplyToken") == std::string::npos);
  CHECK(hip.find("retire_fail") != std::string::npos);
  CHECK(hip.find("RetireGemma4Fp8TopKIndexedPeer") != std::string::npos);
  CHECK(hip.find("RestoreComputeDev") != std::string::npos);
  // The guarantee: RetireGemma4Fp8TopKIndexedPeer synchronizes the compute
  // stream, keeps that result, and returns it -- it never discards the status
  // and never reports unconditional success. Asserted on the function's own
  // body, so an unrelated definition added after it cannot red this case.
  const std::string retire = FunctionBody(hip, "bool RetireGemma4Fp8TopKIndexedPeer");
  REQUIRE_FALSE(retire.empty());
  REQUIRE(retire.back() == '}');
  CHECK(retire.find("RunGemma4Fp8ExpertGeGLUPrefillOnExpertDevice") == std::string::npos);
  CHECK(retire.find("hipStreamSynchronize(cst)") != std::string::npos);
  CHECK(retire.find("(void)hipStreamSynchronize") == std::string::npos);
  CHECK(retire.find("return true;") == std::string::npos);
  CHECK(retire.find("return ok;") != std::string::npos);
}

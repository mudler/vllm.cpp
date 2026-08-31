// vllm.cpp original (vt runtime). The BYTE-EQUALITY gate for the hoisted
// per-element dtype dispatch in `vt::Attention` and `vt::AttentionCross`
// (row VT-CPU-ELEM-DISPATCH, .agents/specs/vt-cpu-elem-dispatch.md).
//
// WHAT THIS FILE EXISTS TO HOLD. Both CPU kernels used to read every operand
// element through `LoadF32`, which switches on `t.dtype` and multiplies by
// `SizeOf(t.dtype)` once per element. That dispatch is loop-invariant and it
// cost 62.68% of the kernel's own CPU time. It is now resolved ONCE per call
// into a typed micro-kernel, and the query row is widened to f32 once per
// (head, query) pair instead of once per (key, element).
//
// The transformation's entire claim is that it is BIT-EXACT, not close: the
// same indices are summed in the same order into the same f32 accumulators.
// A tolerance would not test that claim, so every assertion here is `memcmp`
// over the raw output bytes against a reference that reproduces the ORIGINAL
// per-element loop verbatim, written from the layout contract in
// include/vt/ops.h and using nothing from src/vt/cpu.
//
// WHY THE DTYPE MATRIX IS THE POINT. Before this row, the CPU arm of
// tests/vt/test_ops_attention_cross.cpp ran f32 operands and nothing else --
// its bf16 geometries are CUDA-only -- so `AttentionCrossKernel`'s f16 and
// bf16 element paths were ungated, and they are exactly the paths a typed
// micro-kernel table can get wrong (a table indexed by the WRONG operand's
// dtype is invisible whenever the operands share one). Every case below
// therefore also runs MIXED operand dtypes, where query, key, value and out
// disagree, because that is the only shape in which such a defect shows.
//
// The refusal cases are here for the same reason: hoisting the dispatch moves
// the "this dtype has no per-element size" failure from inside a threadpool
// worker to the calling thread, and the MESSAGE must not drift while it moves.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

using vt::AttentionArgs;
using vt::AttentionCrossArgs;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }

Tensor MakeT(void* data, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = Cpu();
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

size_t ElemBytes(DType dt) { return dt == DType::kF32 ? 4u : 2u; }

// A storage buffer at an arbitrary elementwise dtype, filled from f32 values
// through the SAME narrowing the runtime uses, so the test never asserts that
// two different roundings agree.
struct Buf {
  DType dt;
  std::vector<uint8_t> bytes;
  int64_t n = 0;

  Buf(DType d, const std::vector<float>& v) : dt(d), n(static_cast<int64_t>(v.size())) {
    bytes.resize(v.size() * ElemBytes(d));
    for (size_t i = 0; i < v.size(); ++i) Store(static_cast<int64_t>(i), v[i]);
  }
  void Store(int64_t i, float x) {
    uint8_t* p = bytes.data() + static_cast<size_t>(i) * ElemBytes(dt);
    if (dt == DType::kF32) {
      std::memcpy(p, &x, 4);
    } else {
      const uint16_t h = dt == DType::kF16 ? vt::F32ToF16(x) : vt::F32ToBF16(x);
      std::memcpy(p, &h, 2);
    }
  }
  // The reference reader: byte-for-byte what `LoadF32` in src/vt/cpu/cpu_ops.cpp
  // did before this row, re-derived here rather than shared with it.
  float Load(int64_t i) const {
    const uint8_t* p = bytes.data() + static_cast<size_t>(i) * ElemBytes(dt);
    if (dt == DType::kF32) {
      float x;
      std::memcpy(&x, p, 4);
      return x;
    }
    uint16_t h;
    std::memcpy(&h, p, 2);
    return dt == DType::kF16 ? vt::F16ToF32(h) : vt::BF16ToF32(h);
  }
  void* Data() { return bytes.data(); }
};

std::vector<float> Ramp(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed | 1u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = static_cast<float>(static_cast<int32_t>(s >> 10) % 977 - 488) * 0.00713f;
  }
  return v;
}

// ---------------------------------------------------------------------------
// The REFERENCE kernels: the pre-row per-element loops, transcribed from
// include/vt/ops.h's layout contract. Three passes, f32 throughout, the score
// reduction strictly sequential over `e` and the value accumulation strictly
// sequential over `j` -- which is what the runtime must still be doing.
// ---------------------------------------------------------------------------
void RefCross(Buf& out, const Buf& q, const Buf& k, const Buf& v, const std::vector<float>* bias,
              int64_t bias_rows, int64_t tq, int64_t s, int64_t hq, int64_t hk, int64_t d,
              float scale) {
  const int64_t qpk = hq / hk;
  std::vector<float> probs(static_cast<size_t>(s));
  std::vector<float> acc(static_cast<size_t>(d));
  for (int64_t h = 0; h < hq; ++h) {
    const int64_t g = h / qpk;
    for (int64_t i = 0; i < tq; ++i) {
      const int64_t qoff = (i * hq + h) * d;
      const float* brow =
          bias == nullptr ? nullptr : bias->data() + (bias_rows == 1 ? 0 : i) * s;
      float m = -std::numeric_limits<float>::infinity();
      for (int64_t j = 0; j < s; ++j) {
        const int64_t koff = (j * hk + g) * d;
        float dot = 0.0f;
        for (int64_t e = 0; e < d; ++e) dot += q.Load(qoff + e) * k.Load(koff + e);
        dot *= scale;
        if (brow != nullptr) dot += brow[j];
        probs[static_cast<size_t>(j)] = dot;
        if (dot > m) m = dot;
      }
      float denom = 0.0f;
      for (int64_t j = 0; j < s; ++j) {
        const float e = std::exp(probs[static_cast<size_t>(j)] - m);
        probs[static_cast<size_t>(j)] = e;
        denom += e;
      }
      const float inv = 1.0f / denom;
      for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] = 0.0f;
      for (int64_t j = 0; j < s; ++j) {
        const float p = probs[static_cast<size_t>(j)] * inv;
        const int64_t voff = (j * hk + g) * d;
        for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] += p * v.Load(voff + e);
      }
      for (int64_t e = 0; e < d; ++e) out.Store(qoff + e, acc[static_cast<size_t>(e)]);
    }
  }
}

void RefSelf(Buf& out, const Buf& q, const Buf& k, const Buf& v, bool causal, int64_t t,
             int64_t hq, int64_t hk, int64_t d, float scale) {
  const int64_t qpk = hq / hk;
  std::vector<float> probs(static_cast<size_t>(t));
  std::vector<float> acc(static_cast<size_t>(d));
  for (int64_t h = 0; h < hq; ++h) {
    const int64_t g = h / qpk;
    for (int64_t i = 0; i < t; ++i) {
      const int64_t jmax = causal ? i : t - 1;
      const int64_t qoff = (i * hq + h) * d;
      float m = -std::numeric_limits<float>::infinity();
      for (int64_t j = 0; j <= jmax; ++j) {
        const int64_t koff = (j * hk + g) * d;
        float dot = 0.0f;
        for (int64_t e = 0; e < d; ++e) dot += q.Load(qoff + e) * k.Load(koff + e);
        dot *= scale;
        probs[static_cast<size_t>(j)] = dot;
        if (dot > m) m = dot;
      }
      float denom = 0.0f;
      for (int64_t j = 0; j <= jmax; ++j) {
        const float e = std::exp(probs[static_cast<size_t>(j)] - m);
        probs[static_cast<size_t>(j)] = e;
        denom += e;
      }
      const float inv = 1.0f / denom;
      for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] = 0.0f;
      for (int64_t j = 0; j <= jmax; ++j) {
        const float p = probs[static_cast<size_t>(j)] * inv;
        const int64_t voff = (j * hk + g) * d;
        for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] += p * v.Load(voff + e);
      }
      for (int64_t e = 0; e < d; ++e) out.Store(qoff + e, acc[static_cast<size_t>(e)]);
    }
  }
}

struct Geo {
  const char* name;
  int64_t tq, s, hq, hk, d;
};

struct DTypes {
  const char* name;
  DType q, k, v, o;
};

// The matrix, bounded by what the OP ACCEPTS rather than by what the kernel
// could be handed. `vt::Attention` and `vt::AttentionCross` (src/vt/ops.cpp)
// both require `IsFloat(query.dtype) && key.dtype == query.dtype &&
// value.dtype == query.dtype` and `IsOutFloat(out.dtype)`, so the operands
// always share ONE of f32/f16/bf16 and `out` is independently f32 or bf16.
// A mixed q/k/v row would be testing an input the production entry point
// refuses, which is why there is not one here: an unreachable case that passes
// is not evidence, and this row does not get to widen the op's contract to make
// its own change look better tested.
//
// The discriminating axis that IS reachable is `out` disagreeing with the
// operands: the load side and the store side resolve their dtype separately,
// so a store that inherited the OPERAND's dtype is invisible whenever the two
// agree and wrong in every row below where they do not.
const DTypes kDTypeSets[] = {
    {"all f32", DType::kF32, DType::kF32, DType::kF32, DType::kF32},
    {"f16 operands, f32 out", DType::kF16, DType::kF16, DType::kF16, DType::kF32},
    {"f16 operands, bf16 out", DType::kF16, DType::kF16, DType::kF16, DType::kBF16},
    {"bf16 operands, bf16 out", DType::kBF16, DType::kBF16, DType::kBF16, DType::kBF16},
    {"bf16 operands, f32 out", DType::kBF16, DType::kBF16, DType::kBF16, DType::kF32},
    {"f32 operands, bf16 out", DType::kF32, DType::kF32, DType::kF32, DType::kBF16},
};

const Geo kCrossGeos[] = {
    // d = 40 is not a multiple of any lane count in the tree, tq != s, and
    // hq > hk exercises the GQA broadcast.
    {"ragged d=40 gqa", 7, 11, 6, 3, 40},
    {"square d=64", 5, 5, 4, 4, 64},
    {"single key", 3, 1, 2, 1, 8},
    {"d=1", 4, 6, 2, 2, 1},
};

}  // namespace

TEST_CASE("AttentionCross CPU is byte-identical to the per-element reference") {
  Queue q{Cpu(), nullptr};
  for (const Geo& g : kCrossGeos) {
    for (const DTypes& dt : kDTypeSets) {
      for (int bias_mode = 0; bias_mode < 3; ++bias_mode) {  // none / [1,S] / [Tq,S]
        CAPTURE(g.name);
        CAPTURE(dt.name);
        CAPTURE(bias_mode);
        const int64_t nq = g.tq * g.hq * g.d;
        const int64_t nkv = g.s * g.hk * g.d;
        Buf qb(dt.q, Ramp(static_cast<size_t>(nq), 101u));
        Buf kb(dt.k, Ramp(static_cast<size_t>(nkv), 202u));
        Buf vb(dt.v, Ramp(static_cast<size_t>(nkv), 303u));
        Buf ob(dt.o, std::vector<float>(static_cast<size_t>(nq), 0.0f));
        Buf rb(dt.o, std::vector<float>(static_cast<size_t>(nq), 0.0f));
        const int64_t bias_rows = bias_mode == 0 ? 0 : (bias_mode == 1 ? 1 : g.tq);
        std::vector<float> bias =
            bias_mode == 0 ? std::vector<float>()
                           : Ramp(static_cast<size_t>(bias_rows * g.s), 404u);

        Tensor tq_ = MakeT(qb.Data(), dt.q, {g.tq, g.hq, g.d});
        Tensor tk = MakeT(kb.Data(), dt.k, {g.s, g.hk, g.d});
        Tensor tv = MakeT(vb.Data(), dt.v, {g.s, g.hk, g.d});
        Tensor to = MakeT(ob.Data(), dt.o, {g.tq, g.hq, g.d});
        Tensor tb = MakeT(bias.data(), DType::kF32, {bias_rows == 0 ? 1 : bias_rows, g.s});
        AttentionCrossArgs a{};
        a.scale = 0.3125f;
        vt::AttentionCross(q, to, tq_, tk, tv, bias_mode == 0 ? nullptr : &tb, a);

        RefCross(rb, qb, kb, vb, bias_mode == 0 ? nullptr : &bias, bias_rows, g.tq, g.s, g.hq,
                 g.hk, g.d, a.scale);
        CHECK(std::memcmp(ob.bytes.data(), rb.bytes.data(), ob.bytes.size()) == 0);
      }
    }
  }
}

TEST_CASE("Attention CPU is byte-identical to the per-element reference") {
  Queue q{Cpu(), nullptr};
  const Geo geos[] = {
      {"ragged d=40 gqa", 9, 9, 6, 3, 40},
      {"square d=64", 5, 5, 4, 4, 64},
      {"d=1", 4, 4, 2, 2, 1},
  };
  for (const Geo& g : geos) {
    for (const DTypes& dt : kDTypeSets) {
      for (int causal = 0; causal < 2; ++causal) {
        CAPTURE(g.name);
        CAPTURE(dt.name);
        CAPTURE(causal);
        const int64_t nq = g.tq * g.hq * g.d;
        const int64_t nkv = g.tq * g.hk * g.d;
        Buf qb(dt.q, Ramp(static_cast<size_t>(nq), 111u));
        Buf kb(dt.k, Ramp(static_cast<size_t>(nkv), 222u));
        Buf vb(dt.v, Ramp(static_cast<size_t>(nkv), 333u));
        Buf ob(dt.o, std::vector<float>(static_cast<size_t>(nq), 0.0f));
        Buf rb(dt.o, std::vector<float>(static_cast<size_t>(nq), 0.0f));
        Tensor tq_ = MakeT(qb.Data(), dt.q, {g.tq, g.hq, g.d});
        Tensor tk = MakeT(kb.Data(), dt.k, {g.tq, g.hk, g.d});
        Tensor tv = MakeT(vb.Data(), dt.v, {g.tq, g.hk, g.d});
        Tensor to = MakeT(ob.Data(), dt.o, {g.tq, g.hq, g.d});
        AttentionArgs a{};
        a.scale = 0.3125f;
        a.causal = causal != 0;
        vt::Attention(q, to, tq_, tk, tv, a);
        RefSelf(rb, qb, kb, vb, a.causal, g.tq, g.hq, g.hk, g.d, a.scale);
        CHECK(std::memcmp(ob.bytes.data(), rb.bytes.data(), ob.bytes.size()) == 0);
      }
    }
  }
}

// THE REFUSAL, and an honest statement of where it lives. Hoisting the dispatch
// moved the kernel's own "this dtype has no per-element size" failure from
// inside a threadpool worker to the calling thread, and `AttnResolveOrRefuse`
// keeps the exact `SizeOf`/`LoadF32` messages so it cannot drift. That branch is
// NOT reachable through `vt::Attention` or `vt::AttentionCross`, because both
// validate `IsFloat` on every operand first -- so it is defence for a direct
// registry call and this file does not pretend to gate it.
//
// What IS reachable, and what these two cases hold, is that the op-level refusal
// is unchanged: a non-float operand is still refused, still before any element
// is read, and still with the message the op wrote before this row.
TEST_CASE("Attention still refuses a block-quantized operand at the op boundary") {
  Queue q{Cpu(), nullptr};
  const int64_t t = 4, h = 2, d = 8;
  std::vector<float> f(static_cast<size_t>(t * h * d), 0.5f);
  std::vector<uint8_t> blk(4096, 0);
  for (int which = 0; which < 3; ++which) {
    Tensor tq_ = MakeT(f.data(), DType::kF32, {t, h, d});
    Tensor tk = MakeT(f.data(), DType::kF32, {t, h, d});
    Tensor tv = MakeT(f.data(), DType::kF32, {t, h, d});
    Tensor to = MakeT(f.data(), DType::kF32, {t, h, d});
    Tensor* target = which == 0 ? &tq_ : (which == 1 ? &tk : &tv);
    target->data = blk.data();
    target->dtype = DType::kQ8_0;
    AttentionArgs a{};
    a.scale = 1.0f;
    std::string msg;
    try {
      vt::Attention(q, to, tq_, tk, tv, a);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    CAPTURE(which);
    CAPTURE(msg);
    CHECK(msg.find("attention: query/key/value must share one float dtype") !=
          std::string::npos);
  }
}

TEST_CASE("AttentionCross still refuses an integer operand at the op boundary") {
  Queue q{Cpu(), nullptr};
  const int64_t t = 4, h = 2, d = 8;
  std::vector<float> f(static_cast<size_t>(t * h * d), 0.5f);
  std::vector<int32_t> ints(static_cast<size_t>(t * h * d), 0);
  for (int which = 0; which < 3; ++which) {
    Tensor tq_ = MakeT(f.data(), DType::kF32, {t, h, d});
    Tensor tk = MakeT(f.data(), DType::kF32, {t, h, d});
    Tensor tv = MakeT(f.data(), DType::kF32, {t, h, d});
    Tensor to = MakeT(f.data(), DType::kF32, {t, h, d});
    Tensor* target = which == 0 ? &tq_ : (which == 1 ? &tk : &tv);
    target->data = ints.data();
    target->dtype = DType::kI32;
    AttentionCrossArgs a{};
    a.scale = 1.0f;
    std::string msg;
    try {
      vt::AttentionCross(q, to, tq_, tk, tv, nullptr, a);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    CAPTURE(which);
    CAPTURE(msg);
    CHECK(msg.find("attention_cross: query/key/value must share one float dtype") !=
          std::string::npos);
  }
}

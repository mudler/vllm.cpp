// CPU punica LoRA apply — ports the executable spec of the shrink/expand ops
// and the single-linear apply.
//
// UPSTREAM tests re-expressed (${VLLM_SOURCE} @ 555967922):
//   tests/lora/test_punica_ops.py:36-74   _cpu_bgmv_shrink / _cpu_bgmv_expand
//                                         (per-LoRA matmul references, idx<0 skip)
//   vllm/lora/punica_wrapper/punica_cpu.py:166-195 add_shrink (W2: the
//                                         multi-slice shrink, whose rank_a is
//                                         narrower than the buffer on the
//                                         fully-sharded path)
//   tests/lora/test_punica_ops.py:142-290 check_lora_shrink/expand_kernel
//   tests/lora/test_lora_functions.py     scaling / optimize
//   vllm/lora/punica_wrapper/punica_cpu.py:265  add_lora_linear (end-to-end)
//   vllm/lora/layers/base_linear.py:215         _apply_lora_to_output
//
// The reference here is an independent double-precision per-LoRA matmul loop
// (mirrors _cpu_bgmv_shrink/_cpu_bgmv_expand), NOT the code under test.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "vllm/lora/lora_weights.h"
#include "vllm/lora/punica.h"

using vllm::lora::AddLoraLinear;
using vllm::lora::AddShrink;
using vllm::lora::BgmvExpand;
using vllm::lora::BgmvExpandSlice;
using vllm::lora::BgmvShrink;
using vllm::lora::LoRALayerWeights;
using vllm::lora::LoRALinear;

namespace {

// Deterministic pseudo-random fill in [-0.5, 0.5)*scale (LCG, no <random> dep).
std::vector<float> Rand(int64_t n, uint32_t seed, float scale = 0.2f) {
  std::vector<float> v(static_cast<size_t>(n));
  uint32_t s = seed;
  for (int64_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[static_cast<size_t>(i)] =
        (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * scale;
  }
  return v;
}

// Reference: out[t,r] = scaling * sum_i x[t,i]*a[s][r,i], s = idx[t].
// Skips the same slots the kernel skips (punica_cpu.cpp BgmvShrink/AddShrink:
// `slot < 0 || slot >= num_slots`). Both halves matter: a caller that exercises
// the out-of-range guard passes `idx[t] == num_slots`, and reading `a` at that
// slot runs off the end of the stacked adapter. `num_slots` is not a parameter,
// so derive it from the only place it is knowable here -- `a`'s own extent.
std::vector<double> RefShrink(const std::vector<float>& x, int64_t T,
                              int64_t in_dim, const std::vector<float>& a,
                              int64_t rank, const std::vector<int32_t>& idx,
                              double scaling) {
  std::vector<double> out(static_cast<size_t>(T * rank), 0.0);
  const int64_t a_stride = rank * in_dim;
  const int64_t num_slots = static_cast<int64_t>(a.size()) / a_stride;
  for (int64_t t = 0; t < T; ++t) {
    const int32_t s = idx[static_cast<size_t>(t)];
    if (s < 0 || s >= num_slots) continue;
    for (int64_t r = 0; r < rank; ++r) {
      double acc = 0.0;
      for (int64_t i = 0; i < in_dim; ++i) {
        acc += static_cast<double>(x[static_cast<size_t>(t * in_dim + i)]) *
               static_cast<double>(a[static_cast<size_t>(s * a_stride + r * in_dim + i)]);
      }
      out[static_cast<size_t>(t * rank + r)] = scaling * acc;
    }
  }
  return out;
}

void CheckClose(const std::vector<float>& got, const std::vector<double>& ref,
                double tol = 1e-4) {
  REQUIRE(got.size() == ref.size());
  for (size_t i = 0; i < got.size(); ++i) {
    CHECK(std::abs(static_cast<double>(got[i]) - ref[i]) <= tol);
  }
}

}  // namespace

TEST_CASE("Optimize folds scaling into lora_b (scaling==1 is a no-op)") {
  // test_lora_functions.py / lora_weights.py:36-42.
  LoRALayerWeights w("m", /*rank=*/4, /*alpha=*/8, /*a=*/{1.0f, 2.0f},
                     /*b=*/{3.0f, 4.0f}, /*in=*/2, /*out=*/1);
  CHECK(w.scaling == doctest::Approx(2.0));  // alpha/rank = 8/4
  w.Optimize();
  CHECK(w.scaling == doctest::Approx(1.0));
  CHECK(w.lora_b[0] == doctest::Approx(6.0f));  // 3*2
  CHECK(w.lora_b[1] == doctest::Approx(8.0f));  // 4*2
  // Idempotent once folded.
  auto before = w.lora_b;
  w.Optimize();
  CHECK(w.lora_b == before);
}

TEST_CASE("bgmv_shrink matches per-LoRA matmul reference, idx<0 skipped") {
  // test_punica_ops.py:36-48,142-213.
  const int64_t T = 6, in_dim = 8, rank = 4, num_slots = 3;
  const double scaling = 0.5;
  auto x = Rand(T * in_dim, 11);
  auto a = Rand(num_slots * rank * in_dim, 22);
  std::vector<int32_t> idx = {0, 1, 2, -1, 1, 0};  // one base-model token (-1)

  std::vector<float> out(static_cast<size_t>(T * rank), 0.0f);
  BgmvShrink(x.data(), T, in_dim, a.data(), num_slots, rank, idx.data(), scaling,
             out.data());

  auto ref = RefShrink(x, T, in_dim, a, rank, idx, scaling);
  CheckClose(out, ref);
  // The -1 token stayed zero (skip semantics).
  for (int64_t r = 0; r < rank; ++r) CHECK(out[static_cast<size_t>(3 * rank + r)] == 0.0f);
}

TEST_CASE("bgmv_expand add_inputs accumulates onto the base output") {
  // test_punica_ops.py:51-74,216-290 (add_inputs=True path).
  const int64_t T = 4, rank = 3, out_dim = 5, num_slots = 2;
  auto buf = Rand(T * rank, 33);
  auto b = Rand(num_slots * out_dim * rank, 44);
  std::vector<int32_t> idx = {0, 1, -1, 0};
  auto base = Rand(T * out_dim, 55, 1.0f);

  std::vector<float> y = base;
  BgmvExpand(buf.data(), T, rank, b.data(), num_slots, out_dim, idx.data(),
             /*add_inputs=*/true, y.data());

  // ref: y = base + buf @ b[s]^T for s>=0, unchanged for s<0.
  const int64_t b_stride = out_dim * rank;
  for (int64_t t = 0; t < T; ++t) {
    const int32_t s = idx[static_cast<size_t>(t)];
    for (int64_t o = 0; o < out_dim; ++o) {
      double acc = base[static_cast<size_t>(t * out_dim + o)];
      if (s >= 0) {
        for (int64_t r = 0; r < rank; ++r) {
          acc += static_cast<double>(buf[static_cast<size_t>(t * rank + r)]) *
                 static_cast<double>(b[static_cast<size_t>(s * b_stride + o * rank + r)]);
        }
      }
      CHECK(std::abs(static_cast<double>(y[static_cast<size_t>(t * out_dim + o)]) - acc) <= 1e-4);
    }
  }
}

TEST_CASE("bgmv_expand_slice writes only the [offset,offset+slice) window") {
  // lora_ops.py:110-128 — merged-projection slice write.
  const int64_t T = 2, rank = 2, out_dim = 3, num_slots = 1, y_width = 7,
                offset = 2;
  auto buf = Rand(T * rank, 66);
  auto b = Rand(num_slots * out_dim * rank, 77);
  std::vector<int32_t> idx = {0, 0};
  std::vector<float> y(static_cast<size_t>(T * y_width), 9.0f);  // sentinel fill

  BgmvExpandSlice(buf.data(), T, rank, b.data(), num_slots, out_dim, idx.data(),
                  y_width, offset, /*slice_size=*/out_dim, /*add_inputs=*/false,
                  y.data());

  for (int64_t t = 0; t < T; ++t) {
    for (int64_t c = 0; c < y_width; ++c) {
      const float v = y[static_cast<size_t>(t * y_width + c)];
      if (c < offset || c >= offset + out_dim) {
        CHECK(v == 9.0f);  // outside the window untouched
      }
    }
  }
}

TEST_CASE("add_lora_linear == scale * x @ A^T @ B^T added to base output") {
  // punica_cpu.py:265 end-to-end + RED-first: the un-applied base differs.
  const int64_t T = 5, in_dim = 6, out_dim = 4, rank = 3, num_slots = 2;
  const double scaling = 1.5;
  auto x = Rand(T * in_dim, 101);
  auto a = Rand(num_slots * rank * in_dim, 202);
  auto b = Rand(num_slots * out_dim * rank, 303);
  auto base = Rand(T * out_dim, 404, 1.0f);
  std::vector<int32_t> idx = {0, 1, -1, 1, 0};

  std::vector<float> y = base;
  AddLoraLinear(y.data(), x.data(), T, in_dim, out_dim, a.data(), b.data(),
                num_slots, rank, idx.data(), scaling);

  // Double reference: delta = scale * (x @ a[s]^T) @ b[s]^T.
  const int64_t a_stride = rank * in_dim, b_stride = out_dim * rank;
  std::vector<double> ref(static_cast<size_t>(T * out_dim));
  for (int64_t t = 0; t < T; ++t) {
    const int32_t s = idx[static_cast<size_t>(t)];
    for (int64_t o = 0; o < out_dim; ++o) {
      double delta = 0.0;
      if (s >= 0) {
        for (int64_t r = 0; r < rank; ++r) {
          double sh = 0.0;
          for (int64_t i = 0; i < in_dim; ++i) {
            sh += static_cast<double>(x[static_cast<size_t>(t * in_dim + i)]) *
                  static_cast<double>(a[static_cast<size_t>(s * a_stride + r * in_dim + i)]);
          }
          delta += scaling * sh *
                   static_cast<double>(b[static_cast<size_t>(s * b_stride + o * rank + r)]);
        }
      }
      ref[static_cast<size_t>(t * out_dim + o)] =
          static_cast<double>(base[static_cast<size_t>(t * out_dim + o)]) + delta;
    }
  }
  CheckClose(y, ref);

  // RED-first: the base-only output (no LoRA applied) must NOT match ref for
  // the LoRA-active tokens — proves the apply actually did work.
  bool base_differs = false;
  for (int64_t t = 0; t < T; ++t) {
    if (idx[static_cast<size_t>(t)] < 0) continue;
    for (int64_t o = 0; o < out_dim; ++o) {
      if (std::abs(static_cast<double>(base[static_cast<size_t>(t * out_dim + o)]) -
                   ref[static_cast<size_t>(t * out_dim + o)]) > 1e-3) {
        base_differs = true;
      }
    }
  }
  CHECK(base_differs);
  // The base-model token (-1) is unchanged by the apply.
  for (int64_t o = 0; o < out_dim; ++o) {
    CHECK(y[static_cast<size_t>(2 * out_dim + o)] ==
          doctest::Approx(base[static_cast<size_t>(2 * out_dim + o)]));
  }
}

TEST_CASE("LoRALinear apply on one ReplicatedLinear vs reference (rank pad)") {
  // base_linear.py:100-238 — create/set/apply. max_rank_ > adapter rank tests
  // the zero-padding of the stacked slot. Scaling is folded via SetLora.
  const int64_t T = 4, in_dim = 5, out_dim = 3, max_rank = 4, num_slots = 2;
  LoRALinear layer(num_slots, max_rank, in_dim, out_dim);

  // Two adapters of different (<= max_rank) rank and different scaling.
  const int rank0 = 2, rank1 = 4;
  LoRALayerWeights ad0("proj", rank0, /*alpha=*/rank0 * 3,  // scaling 3.0
                       Rand(rank0 * in_dim, 1), Rand(out_dim * rank0, 2), in_dim,
                       out_dim);
  LoRALayerWeights ad1("proj", rank1, /*alpha=*/rank1,  // scaling 1.0
                       Rand(rank1 * in_dim, 3), Rand(out_dim * rank1, 4), in_dim,
                       out_dim);
  layer.SetLora(0, ad0);
  layer.SetLora(1, ad1);

  auto x = Rand(T * in_dim, 5);
  auto base = Rand(T * out_dim, 6, 1.0f);
  std::vector<int32_t> idx = {0, 1, -1, 0};

  std::vector<float> y = base;
  layer.ApplyLoraToOutput(y.data(), x.data(), T, idx.data());

  // Reference from the ORIGINAL adapter tensors (scaling explicit here).
  auto ref_token = [&](int64_t t, const LoRALayerWeights& ad) {
    std::vector<double> out(static_cast<size_t>(out_dim));
    for (int64_t o = 0; o < out_dim; ++o) {
      double delta = 0.0;
      for (int64_t r = 0; r < ad.rank; ++r) {
        double sh = 0.0;
        for (int64_t i = 0; i < in_dim; ++i) {
          sh += static_cast<double>(x[static_cast<size_t>(t * in_dim + i)]) *
                static_cast<double>(ad.lora_a[static_cast<size_t>(r * in_dim + i)]);
        }
        delta += ad.scaling * sh *
                 static_cast<double>(ad.lora_b[static_cast<size_t>(o * ad.rank + r)]);
      }
      out[static_cast<size_t>(o)] =
          static_cast<double>(base[static_cast<size_t>(t * out_dim + o)]) + delta;
    }
    return out;
  };

  for (int64_t t = 0; t < T; ++t) {
    std::vector<double> ref;
    if (idx[static_cast<size_t>(t)] == 0) ref = ref_token(t, ad0);
    else if (idx[static_cast<size_t>(t)] == 1) ref = ref_token(t, ad1);
    else {  // base token
      ref.resize(static_cast<size_t>(out_dim));
      for (int64_t o = 0; o < out_dim; ++o)
        ref[static_cast<size_t>(o)] = base[static_cast<size_t>(t * out_dim + o)];
    }
    for (int64_t o = 0; o < out_dim; ++o) {
      CHECK(std::abs(static_cast<double>(y[static_cast<size_t>(t * out_dim + o)]) -
                     ref[static_cast<size_t>(o)]) <= 1e-4);
    }
  }

  // ResetLora clears a slot back to base behaviour.
  layer.ResetLora(0);
  std::vector<int32_t> only0 = {0, 0, 0, 0};
  std::vector<float> y2 = base;
  layer.ApplyLoraToOutput(y2.data(), x.data(), T, only0.data());
  CHECK(y2 == base);  // slot 0 zeroed => identity
}

TEST_CASE("add_shrink leaves a -1 slot's buffer row UNTOUCHED, not zeroed") {
  // punica_cpu.py:192-195 -> bgmv_shrink. A base-model token's slot is -1
  // (utils.py:104-110) and the triton kernel early-exits on it; the caller's
  // zero-fill of the buffer is what makes the following expand contribute
  // nothing. Pre-filling with a SENTINEL instead of zero separates "skipped"
  // from "computed to zero", which is the only difference a dropped guard
  // makes -- through a layer the expand skips the same token, so the wrong
  // buffer never reaches the output, and the read at `a_stacked - slot_stride`
  // is silent.
  const int64_t T = 4, in_dim = 6, rank = 3, num_slots = 2, n_slices = 2;
  const std::vector<float> x = Rand(T * in_dim, 11u);
  const std::vector<float> a0 = Rand(num_slots * rank * in_dim, 22u);
  const std::vector<float> a1 = Rand(num_slots * rank * in_dim, 33u);
  // Two base tokens, one active slot each side, and one out-of-range slot.
  const std::vector<int32_t> idx = {1, -1, 0, num_slots};
  constexpr float kSentinel = -12345.0f;

  std::vector<float> buffers(static_cast<size_t>(n_slices * T * rank), kSentinel);
  AddShrink(buffers.data(), x.data(), T, in_dim, {a0.data(), a1.data()}, num_slots,
            /*rank_a=*/rank, /*buffer_rank=*/rank, idx.data(), /*scale=*/1.0);

  const std::vector<double> ref0 = RefShrink(x, T, in_dim, a0, rank, idx, 1.0);
  const std::vector<double> ref1 = RefShrink(x, T, in_dim, a1, rank, idx, 1.0);
  for (int64_t s = 0; s < n_slices; ++s) {
    const std::vector<double>& ref = s == 0 ? ref0 : ref1;
    for (int64_t t = 0; t < T; ++t) {
      const bool skipped = idx[static_cast<size_t>(t)] < 0 ||
                           idx[static_cast<size_t>(t)] >= num_slots;
      for (int64_t r = 0; r < rank; ++r) {
        const float got =
            buffers[static_cast<size_t>(s * T * rank + t * rank + r)];
        if (skipped) {
          REQUIRE(got == kSentinel);
        } else {
          REQUIRE(got != kSentinel);
          CHECK(std::abs(static_cast<double>(got) -
                         ref[static_cast<size_t>(t * rank + r)]) <= 1e-4);
        }
      }
    }
  }
}

TEST_CASE("add_shrink fills only rank_a columns and strides lora_a by rank_a") {
  // lora_ops.py:80 -- `output_tensor[:, :outputs.shape[1]]`. The stacked lora_a
  // has `max_rank / tp_size` rows on the fully-sharded path
  // (base_linear.py:112-116), narrower than the buffer the expand later reads,
  // so the per-slot stride is rank_a * in_dim and the columns past rank_a stay
  // as the caller left them. Every layer-level case runs at tp_size == 1 where
  // the two ranks coincide, so this contract is only visible here.
  const int64_t T = 3, in_dim = 5, rank_a = 2, buffer_rank = 4, num_slots = 3;
  const std::vector<float> a = Rand(num_slots * rank_a * in_dim, 44u);
  const std::vector<float> x = Rand(T * in_dim, 55u);
  const std::vector<int32_t> idx = {2, 0, 1};

  std::vector<float> buf(static_cast<size_t>(T * buffer_rank), 0.0f);
  AddShrink(buf.data(), x.data(), T, in_dim, {a.data()}, num_slots, rank_a,
            buffer_rank, idx.data(), /*scale=*/1.0);

  for (int64_t t = 0; t < T; ++t) {
    const int64_t s = idx[static_cast<size_t>(t)];
    for (int64_t r = 0; r < buffer_rank; ++r) {
      double want = 0.0;
      if (r < rank_a) {
        for (int64_t i = 0; i < in_dim; ++i) {
          want += static_cast<double>(x[static_cast<size_t>(t * in_dim + i)]) *
                  static_cast<double>(
                      a[static_cast<size_t>(s * rank_a * in_dim + r * in_dim + i)]);
        }
      }
      CHECK(std::abs(static_cast<double>(
                         buf[static_cast<size_t>(t * buffer_rank + r)]) -
                     want) <= 1e-5);
    }
  }
}

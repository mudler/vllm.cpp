// Qwen4-Exp W5d-4 — the MoE weight adapter. See qwen4_exp_moe.h for the four
// differences between `Qwen4ExpMoeWeights` and `MoeBlockWeights`, the oracle
// anchors, and what lands unreached.
#include "vllm/model_executor/models/qwen4_exp_moe.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace {

std::string ShapeStr(const OwnedTensor& t) {
  std::string s = "[";
  for (int i = 0; i < t.rank; ++i) {
    if (i != 0) s += ",";
    s += std::to_string(t.shape[i]);
  }
  return s + "]";
}

// The one refusal spelling this file uses. Every message names the tensor and
// the part that is missing or wrong, because "the MoE weights did not fit" sends
// the next reader to the wrong file.
void Refuse(bool ok, const std::string& what) {
  VT_CHECK(ok, "qwen4_exp moe adapter: " + what +
                   ". See `.agents/specs/qwen4-exp-flash-next.md` and issue "
                   "#2249 (item 4).");
}

void RequireShape(const OwnedTensor& t, const char* name,
                  const std::vector<int64_t>& want) {
  Refuse(t.rank == static_cast<int>(want.size()),
         std::string(name) + " must be rank " + std::to_string(want.size()) +
             ", not rank " + std::to_string(t.rank) + " " + ShapeStr(t));
  for (size_t i = 0; i < want.size(); ++i) {
    Refuse(t.shape[i] == want[i],
           std::string(name) + " axis " + std::to_string(i) + " must be " +
               std::to_string(want[i]) + ", not " + std::to_string(t.shape[i]) +
               " (whole shape " + ShapeStr(t) + ")");
  }
}

// A zero-copy view of `src`'s bytes at [byte_off, byte_off + nbytes).
//
// `KeepAlive()` is taken BEFORE `data()`: on an owned buffer it moves the vector
// into a refcounted holder, and while `std::vector`'s move preserves the heap
// address, ordering the two calls means the view can never depend on that being
// true. Every layout marker the GEMM keys on is carried across — dropping
// `repacked` makes `kMatmulBTQuant` read i8mm-interleaved bytes as plain q8_0,
// which is the CIQ-G7 all-zero-token failure.
OwnedTensor BorrowView(OwnedTensor& src, size_t byte_off, size_t nbytes,
                       const std::vector<int64_t>& shape, bool nk) {
  Refuse(byte_off + nbytes <= src.bytes.size(),
         "a view of " + std::to_string(nbytes) + " bytes at offset " +
             std::to_string(byte_off) + " does not fit the " +
             std::to_string(src.bytes.size()) + "-byte source buffer");
  std::shared_ptr<const void> keep = src.bytes.KeepAlive();
  OwnedTensor v;
  v.bytes = OwnedBytes::Borrow(src.bytes.data() + byte_off, nbytes, std::move(keep));
  v.dtype = src.dtype;
  v.nk = nk;
  v.repacked = src.repacked;
  v.q8_0_aligned = src.q8_0_aligned;
  v.elem_kn_repacked = src.elem_kn_repacked;
  v.mmap_fd = src.mmap_fd;
  v.mmap_file_offset = src.mmap_file_offset + byte_off;
  v.rank = static_cast<int>(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) v.shape[i] = shape[i];
  return v;
}

// The whole of `src` as a view: same bytes, same rank, shape, dtype and `nk`.
OwnedTensor BorrowWhole(OwnedTensor& src) {
  std::vector<int64_t> shape(src.shape, src.shape + src.rank);
  return BorrowView(src, 0, src.bytes.size(), shape, src.nk);
}

// An f32 owned tensor re-rounded to bf16 at `shape` / `nk`. The seam's router
// and shared gate are both consumed against a bf16 activation, and the CUDA GEMM
// refuses a mixed (bf16, f32) pair by name; see the header.
OwnedTensor Bf16FromF32(const OwnedTensor& src, const char* name,
                        const std::vector<int64_t>& shape, bool nk) {
  Refuse(src.dtype == vt::DType::kF32,
         std::string(name) +
             " must be f32 as `LoadMoe` leaves it; the loader's residency "
             "routing changed under this adapter");
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  Refuse(src.bytes.size() == static_cast<size_t>(n) * sizeof(float),
         std::string(name) + " holds " + std::to_string(src.bytes.size()) +
             " bytes, not the " + std::to_string(static_cast<size_t>(n) * sizeof(float)) +
             " its shape needs");
  OwnedTensor o;
  o.dtype = vt::DType::kBF16;
  o.nk = nk;
  o.rank = static_cast<int>(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) o.shape[i] = shape[i];
  o.bytes.resize(static_cast<size_t>(n) * sizeof(uint16_t));
  const auto* in = reinterpret_cast<const float*>(src.bytes.data());
  auto* out = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (int64_t i = 0; i < n; ++i) out[i] = vt::F32ToBF16(in[i]);
  return o;
}

// Which of the seam's two expert arms this tower belongs to. `LoadStackedExperts`
// produces exactly two forms: a bf16 EXPANSION (`ExpandBf16`, nk = true) or the
// file's own quant BLOCKS (`OwnGgufQuantBlocks`).
bool IsStackedKeepQuant(const OwnedTensor& t) { return vt::IsBlockQuant(t.dtype); }

}  // namespace

HfConfig Qwen4ExpMoeHfConfig(const Qwen4ExpParams& p) {
  HfConfig c;
  c.hidden_size = p.hidden_size;
  c.num_experts = p.num_experts;
  c.num_experts_per_tok = p.num_experts_per_tok;
  c.moe_intermediate_size = p.moe_intermediate_size;
  c.shared_expert_intermediate_size = p.shared_expert_intermediate_size;
  return c;
}

MoeBlockWeights Qwen4ExpMoeBlockWeights(Qwen4ExpMoeWeights& moe,
                                        const Qwen4ExpParams& p) {
  const int64_t H = p.hidden_size;
  const int64_t E = p.num_experts;
  const int64_t I = p.moe_intermediate_size;
  const int64_t Is = p.shared_expert_intermediate_size;
  Refuse(H > 0 && E > 0 && I > 0 && Is > 0 && p.num_experts_per_tok > 0,
         "the config carries no MoE geometry (hidden_size " + std::to_string(H) +
             ", num_experts " + std::to_string(E) + ", moe_intermediate_size " +
             std::to_string(I) + ", shared_expert_intermediate_size " +
             std::to_string(Is) + ", num_experts_per_tok " +
             std::to_string(p.num_experts_per_tok) + ")");
  Refuse(p.num_experts_per_tok <= E,
         "num_experts_per_tok " + std::to_string(p.num_experts_per_tok) +
             " exceeds num_experts " + std::to_string(E));

  RequireShape(moe.router, "ffn_gate_inp (router)", {E, H});
  RequireShape(moe.shared_gate, "ffn_gate_inp_shexp (shared gate)", {H});
  RequireShape(moe.gate_exps, "ffn_gate_exps", {E, I, H});
  RequireShape(moe.up_exps, "ffn_up_exps", {E, I, H});
  RequireShape(moe.down_exps, "ffn_down_exps", {E, H, I});
  RequireShape(moe.shared_gate_proj, "ffn_gate_shexp", {Is, H});
  RequireShape(moe.shared_up_proj, "ffn_up_shexp", {Is, H});
  RequireShape(moe.shared_down_proj, "ffn_down_shexp", {H, Is});

  MoeBlockWeights m;

  // Router: upstream's `weight [E, H]` verbatim, re-rounded to bf16 and marked
  // nk so `MatmulBf16` takes `vt::MatmulBT` — the same orientation the qwen3_5
  // GGUF loader's default `expand_nk` produces, and the same one upstream's
  // `F.linear(hidden_states, self.weight)` reads.
  m.router_gate = Bf16FromF32(moe.router, "ffn_gate_inp (router)", {E, H}, /*nk=*/true);
  // Shared gate: upstream's `Linear(H, 1)` with its output axis restored, so the
  // seam's `MatmulF32D` produces the [T, 1] gate logit it feeds to sigmoid.
  m.shared_gate =
      Bf16FromF32(moe.shared_gate, "ffn_gate_inp_shexp (shared gate)", {H, 1}, /*nk=*/false);

  // The shared expert's three projections are `LoadMatmul` products already in
  // the form `MatmulF32D` reads, in whichever residency the policy chose. They
  // pass through as views rather than by assignment: copying an `OwnedTensor`
  // whose buffer is OWNED duplicates its bytes, which for the bf16 expansion is
  // 3 x 3.3 MB per layer that nothing would ever read twice. The source's own
  // `nk` is preserved rather than asserted — `LoadMatmul` sets it per residency
  // and this adapter has no business deciding it.
  m.shared_gate_proj = BorrowWhole(moe.shared_gate_proj);
  m.shared_up_proj = BorrowWhole(moe.shared_up_proj);
  m.shared_down_proj = BorrowWhole(moe.shared_down_proj);

  // ONE residency for all three towers. `MoeBlock` picks the whole expert path
  // from `expert_gate_kq` alone, so a mixed set is read as keep-quant and then
  // dereferences an empty up/down tower.
  const bool kq_gate = IsStackedKeepQuant(moe.gate_exps);
  const bool kq_up = IsStackedKeepQuant(moe.up_exps);
  const bool kq_down = IsStackedKeepQuant(moe.down_exps);
  Refuse(kq_gate == kq_up && kq_gate == kq_down,
         std::string("the three expert towers disagree on residency (ffn_gate_exps ") +
             vt::Name(moe.gate_exps.dtype) + ", ffn_up_exps " +
             vt::Name(moe.up_exps.dtype) + ", ffn_down_exps " +
             vt::Name(moe.down_exps.dtype) +
             "); `MoeBlock` selects the expert path from the GATE tower alone, so "
             "a mixed set reads as keep-quant and dereferences an empty tower");

  if (kq_gate) {
    // Keep-quant: the stacked towers reach the seam VERBATIM, re-declared rank 2
    // `[E*N, K]` because `vt::MatmulBTQuantGrouped` refuses rank 3.
    m.expert_gate_kq = BorrowView(moe.gate_exps, 0, moe.gate_exps.bytes.size(),
                                  {E * I, H}, moe.gate_exps.nk);
    m.expert_up_kq = BorrowView(moe.up_exps, 0, moe.up_exps.bytes.size(),
                                {E * I, H}, moe.up_exps.nk);
    m.expert_down_kq = BorrowView(moe.down_exps, 0, moe.down_exps.bytes.size(),
                                  {E * H, I}, moe.down_exps.nk);
    return m;
  }

  Refuse(moe.gate_exps.dtype == vt::DType::kBF16 &&
             moe.up_exps.dtype == vt::DType::kBF16 &&
             moe.down_exps.dtype == vt::DType::kBF16,
         std::string("the expert towers are ") + vt::Name(moe.gate_exps.dtype) +
             ", which is neither the block-quant arm nor the bf16 expansion "
             "`LoadStackedExperts` produces; the seam has no arm for it");

  // bf16: per-expert `[N, K]` nk = true views into the stacked buffer. NOT
  // copies — see the header; three copies per layer at the released geometry is
  // 240 GB across the stack.
  const size_t gu_bytes = static_cast<size_t>(I) * static_cast<size_t>(H) * sizeof(uint16_t);
  const size_t dn_bytes = static_cast<size_t>(H) * static_cast<size_t>(I) * sizeof(uint16_t);
  m.expert_gate.reserve(static_cast<size_t>(E));
  m.expert_up.reserve(static_cast<size_t>(E));
  m.expert_down.reserve(static_cast<size_t>(E));
  for (int64_t e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    m.expert_gate.push_back(
        BorrowView(moe.gate_exps, se * gu_bytes, gu_bytes, {I, H}, /*nk=*/true));
    m.expert_up.push_back(
        BorrowView(moe.up_exps, se * gu_bytes, gu_bytes, {I, H}, /*nk=*/true));
    m.expert_down.push_back(
        BorrowView(moe.down_exps, se * dn_bytes, dn_bytes, {H, I}, /*nk=*/true));
  }
  return m;
}

MoeBlockOutput RunQwen4ExpMoeBlock(vt::Queue& queue, const MoeBlockWeights& weights,
                                   const Qwen4ExpParams& p, const vt::Tensor& dh,
                                   int64_t T) {
  Refuse(T > 0, "T must be positive, not " + std::to_string(T));
  Refuse(dh.rank == 2 && dh.shape[0] == T && dh.shape[1] == p.hidden_size,
         "the hidden state must be [T, hidden_size] = [" + std::to_string(T) + "," +
             std::to_string(p.hidden_size) + "], not rank " + std::to_string(dh.rank));
  Refuse(dh.dtype == vt::DType::kBF16,
         "the hidden state must be bf16, which is the dtype every `MoeBlock` arm "
         "reads; a wider one is silently truncated by the download");
  const HfConfig cfg = Qwen4ExpMoeHfConfig(p);
  return RunMoeBlock(queue, weights, cfg, dh, T);
}

}  // namespace vllm

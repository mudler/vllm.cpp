// Qwen4-Exp W5b-5 — `Qwen4ExpTextAttention` as one production block, and W5d-3
// (#2249 item 2) — the same block over the PAGED K/V the engine allocates. See
// `qwen4_exp_qsa_block.h` for why this file exists, which four settings it is
// the sole enforcer of, what the two cache arms share, and what it deliberately
// does not cover.
//
// ALGORITHM ORACLE: transformers 5.16.0 (this row's accepted lane pin),
// `models/qwen4_exp/modeling_qwen4_exp.py`. Every line below cites the upstream
// line it mirrors. OP ORACLE: vLLM, through the `vt::` primitives — this block
// introduces no arithmetic of its own, which is the whole point of it being a
// composition rather than a kernel.
#include "vllm/model_executor/models/qwen4_exp_qsa_block.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // ResidentWeight
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vllm {
namespace {

using dense_attn::Dev;
using dense_attn::DBuf;
using vt::DType;
using vt::Tensor;

// ─── A HOST READ OF A TENSOR THAT MAY NOT BE HOST-RESIDENT (#2421) ──────────
// `count` elements of `t` starting at ELEMENT `offset`, written to `dst` on the
// host. `StageHostWords` only ENQUEUES the read; nothing is readable until the
// caller synchronises the queue once for every range it staged.
//
// THIS IS WHAT REPLACED THREE REFUSALS. This block resolves exactly three things
// on the host — group 2's page table, the two rope layouts' probe rows, and
// `kv_lens` — and each of the three used to refuse a tensor that was not
// CPU-resident BY NAME, which made every CUDA step stop at the first of them
// before any arithmetic ran. None of the three has a device-side consumer to
// hand the work to: two produce an index vector that is uploaded again
// immediately, and the third produces no tensor at all. So the read stays a host
// read, and the words it reads are COPIED, with the element range named at the
// call site so the copy's size is visible where the read is.
//
// IT IS NOT A SILENT DEVICE DEREFERENCE, which is what those refusals existed to
// prevent and what this file must never become: a host loop walking a device
// pointer faults on a discrete device and, on a unified-memory one, reads
// plausible bytes and produces a quietly wrong SELECTION that no downstream
// value gate can convict. `Backend::Copy` infers its direction from the pointer
// values (`cudaMemcpyDefault`, src/vt/cuda/cuda_backend.cu:116), so one entry
// point covers every arm, and the `kCPU` tensor takes the memcpy below and
// enqueues nothing at all.
//
// THE SPLIT INTO STAGE-THEN-SYNCHRONISE IS THE WHOLE POINT OF THE SHAPE.
// `CheckRopeLayoutsAgree` reads NINE ranges, and the cost of a device read here
// is the synchronise and not the four kilobytes; nine synchronises per QSA layer
// per step would be a decode cost paid to re-check a constant. The spec's
// `## Owed` carries the one that remains.
template <typename T>
void StageHostWords(Dev d, const Tensor& t, int64_t offset, int64_t count, T* dst) {
  VT_CHECK(offset >= 0 && count >= 0,
           "qwen4_exp qsa block: a host read names a negative element range");
  VT_CHECK(vt::SizeOf(t.dtype) == sizeof(T),
           "qwen4_exp qsa block: a host read's element width disagrees with the tensor dtype");
  VT_CHECK(t.IsContiguous(),
           "qwen4_exp qsa block: a host read needs a contiguous tensor — a strided one would "
           "copy the wrong elements rather than fail");
  VT_CHECK(offset + count <= t.Numel(),
           "qwen4_exp qsa block: a host read names elements outside the tensor");
  if (count == 0) return;
  VT_CHECK(t.data != nullptr, "qwen4_exp qsa block: a host read of an unallocated tensor");
  const auto bytes = static_cast<size_t>(count) * sizeof(T);
  const char* src = static_cast<const char*>(t.data) + static_cast<size_t>(offset) * sizeof(T);
  if (t.device.type == vt::DeviceType::kCPU) {
    std::memcpy(dst, src, bytes);
    return;
  }
  d.b.Copy(d.q, dst, src, bytes);
}

// One range, read and made readable. `Backend::Synchronize` is a no-op on the
// base class, so the CPU arm pays nothing for the unconditional call and the
// device arm cannot forget it.
template <typename T>
std::vector<T> HostWords(Dev d, const Tensor& t, int64_t offset, int64_t count) {
  std::vector<T> host(static_cast<size_t>(count));
  StageHostWords(d, t, offset, count, host.data());
  d.b.Synchronize(d.q);
  return host;
}

// A contiguous ROW-RANGE view of a contiguous tensor, reshaped. `t` is the
// owner, `start`/`count` index its OUTERMOST dimension, and `shape` describes
// the view. The element count must agree, which is the check that stops a
// reshape from quietly renaming a stride.
Tensor RowsView(const Tensor& t, int64_t start, int64_t count,
                const std::vector<int64_t>& shape) {
  VT_CHECK(t.rank >= 1 && t.IsContiguous(),
           "qwen4_exp qsa block: RowsView needs a contiguous tensor");
  VT_CHECK(start >= 0 && count >= 0 && start + count <= t.shape[0],
           "qwen4_exp qsa block: RowsView range outside the tensor");
  int64_t row_elems = 1;
  for (int i = 1; i < t.rank; ++i) row_elems *= t.shape[i];
  int64_t want = 1;
  for (int64_t s : shape) want *= s;
  VT_CHECK(want == count * row_elems,
           "qwen4_exp qsa block: RowsView shape does not cover the rows it names");
  Tensor v = dense_attn::MakeTensor(
      static_cast<char*>(t.data) +
          static_cast<size_t>(start * row_elems) * vt::SizeOf(t.dtype),
      t.dtype, t.device, shape);
  return v;
}

// The whole tensor under a different shape, same bytes.
Tensor Reshape(const Tensor& t, const std::vector<int64_t>& shape) {
  return RowsView(t, 0, t.rank == 0 ? 0 : t.shape[0], shape);
}

// ─── W5i: THE INDEXER SIDE CACHE'S PAGE TRANSLATION (#2249 item 3) ──────────
// The PHYSICAL rows of logical indexer positions [begin, begin + n), resolved
// through group 2's own block table:
//
//     row(pos) = table[pos / block_size] * block_size + pos % block_size
//
// which is the ordinary paged address and the same arithmetic the runner uses to
// build a slot mapping. Flattened, an `MLAAttentionSpec` group's pages are a
// contiguous `[num_pages * block_size, D]` array of such rows, so this vector is
// exactly the `idx` operand `vt::IndexSelect` and `vt::IndexCopy` take, and the
// gather and the scatter need NO NEW OP.
//
// THE TABLE IS READ ON THE HOST, and it ARRIVES on the host: `QsaBlockCore`
// resolves it once per call through `HostWords` above, whatever device it lives
// on, and hands the resolved words here. It used to be a `const Tensor&` that
// REFUSED a table that was not CPU-resident by name, which was correct — the
// alternative on offer at the time was a host loop over a device pointer — and
// which stopped every CUDA step inside this block. What the refusal was really
// asking for is a translation with a device-side home; until that exists the
// words are copied, explicitly, and the copy is priced in the spec's `## Owed`
// rather than hidden.
//
// TAKING A VECTOR RATHER THAN A TENSOR IS THE POINT. The type now says that this
// arithmetic runs on host words and cannot be handed a device pointer at all, so
// the residency question is settled at the ONE place that resolves it instead of
// re-asked at every use.
std::vector<int32_t> IndexerRows(const std::vector<int32_t>& table, int64_t block_size,
                                 int64_t begin, int64_t n) {
  const auto pages = static_cast<int64_t>(table.size());
  const int32_t* tb = table.data();
  std::vector<int32_t> rows(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const int64_t pos = begin + i;
    const int64_t lp = pos / block_size;
    VT_CHECK(lp < pages,
             "qwen4_exp qsa block: the indexer block table does not name logical page " +
                 std::to_string(lp));
    rows[static_cast<size_t>(i)] =
        static_cast<int32_t>(static_cast<int64_t>(tb[lp]) * block_size + pos % block_size);
  }
  return rows;
}


// ─── THE TWO ROPE LAYOUTS, CROSS-CHECKED ────────────────────────────────────
// The block is handed ONE set of angles TWICE: `vt::RopeFromCache` reads a
// PACKED bf16 `[P, rot]` cache (cos in the leading half, sin in the trailing
// half) and `vt::Qwen4ExpQsaCompress` reads SEPARATE f32 `[P, rot]` tables,
// because the two ops were ported from two upstreams that spell it differently.
// Nothing in the type system forces a caller to build both from one table, and
// a layer loop that does not diverges SILENTLY — the query roped with one set of
// angles and the pooled indexer keys with another, every value finite, the
// selection quietly wrong. The header used to say "the gate asserts they agree";
// nothing did, and the test's own builder derived one FROM the other, which is
// agreement by construction rather than an assertion.
//
// THE PROBE IS A BOUNDED SAMPLE OF ROWS, NOT THE WHOLE TABLE, and that is a cost
// decision stated rather than hidden. `P` is the context budget and this runs
// once per QSA layer per step, so a full comparison would be O(P * rot) per call
// — the same O(kv)-per-step shape the spec already carries under `## Owed`, paid
// to re-check a constant. Every construction difference that a layer loop can
// actually make is a property of the WHOLE table and shows at every
// non-degenerate row: a different theta, a different `rotary_dim`, an
// INTERLEAVED pack against a half-split one, swapped cos/sin halves, an
// off-by-one position offset, or a position-scaling factor (linear, YaRN)
// applied to one table and not the other. What the sample cannot see is a single
// corrupted row, and that is stated here rather than implied.
//
// ROW 0 IS NOT A PROBE WHEN A SECOND ROW EXISTS. cos is 1 and sin is 0 at every
// frequency there, so row 0 agrees under every difference in that list. The last
// row is a probe because a position-scaling difference is smallest at low
// positions and largest at the end of the table.
constexpr int kRopeProbeRows = 3;

// One bf16 ulp at magnitude 1. `cos_sin` is the block dtype (bf16) and `cos`/`sin`
// are f32, so two spellings of one angle differ by at most the rounding of a
// value in [-1, 1] — 2^-9 under round-to-nearest and 2^-8 even under truncation.
// Every difference the probe exists to catch is O(1), so this bound separates
// them by two orders of magnitude; it is a LAYOUT check, not an epsilon.
constexpr float kRopeAgreeTol = 1.0F / 128.0F;

void CheckRopeLayoutsAgree(Dev d, const Tensor& cos_sin, const Tensor& cos, const Tensor& sin) {
  // The comparison is a HOST read of both tables, and it STAYS one on a device
  // arm: the probe rows are copied to the host rather than the check being
  // skipped. Skipping is what a mute switch looks like, and refusing — which is
  // what this line did until #2421 — stopped every CUDA step here before any
  // arithmetic ran. The check the wave was owed a device-side home for is
  // O(kRopeProbeRows * rotary_dim) words, so the home it gets is a copy.
  VT_CHECK(cos.shape[0] == cos_sin.shape[0] && sin.shape[0] == cos_sin.shape[0],
           "qwen4_exp qsa block: the PACKED cos_sin cache and the SEPARATE cos/sin tables "
           "must have the same number of rows — they are two layouts of ONE [P, rotary_dim] "
           "table and two heights cannot have come from one build");
  const int64_t P = cos_sin.shape[0], rot = cos_sin.shape[1], half = rot / 2;
  const int64_t probes[kRopeProbeRows] = {P > 1 ? 1 : 0, P / 2, P - 1};
  // NINE RANGES, ONE SYNCHRONISE. The three probe rows are gathered out of three
  // tensors, so they are staged together and made readable once; on a CPU queue
  // every one of the nine is a memcpy and the synchronise is the base class's
  // no-op, which is why this costs the CPU arm nothing.
  std::vector<uint16_t> pk(static_cast<size_t>(kRopeProbeRows * rot));
  std::vector<float> cf(static_cast<size_t>(kRopeProbeRows * rot));
  std::vector<float> sf(static_cast<size_t>(kRopeProbeRows * rot));
  for (int i = 0; i < kRopeProbeRows; ++i) {
    const int64_t r = probes[i];
    StageHostWords(d, cos_sin, r * rot, rot, pk.data() + static_cast<size_t>(i * rot));
    StageHostWords(d, cos, r * rot, rot, cf.data() + static_cast<size_t>(i * rot));
    StageHostWords(d, sin, r * rot, rot, sf.data() + static_cast<size_t>(i * rot));
  }
  d.b.Synchronize(d.q);
  for (int i = 0; i < kRopeProbeRows; ++i) {
    const int64_t r = i * rot;
    for (int64_t j = 0; j < half; ++j) {
      const float pc = vt::BF16ToF32(pk[static_cast<size_t>(r + j)]);
      const float ps = vt::BF16ToF32(pk[static_cast<size_t>(r + half + j)]);
      VT_CHECK(std::fabs(pc - cf[static_cast<size_t>(r + j)]) <= kRopeAgreeTol &&
                   std::fabs(ps - sf[static_cast<size_t>(r + j)]) <= kRopeAgreeTol,
               "qwen4_exp qsa block: the PACKED cos_sin cache and the SEPARATE cos/sin "
               "tables do not describe the same angles — both must be built from ONE "
               "table, or the query and the pooled indexer keys are roped differently and "
               "nothing downstream can tell");
    }
  }
}

// ─── THE TWO CACHE ARMS, AS ONE DESCRIPTOR (W5d-3, #2249 item 2) ────────────
// Exactly one of the two pointers is set. The block body below reads this in
// precisely two places — where the new K/V rows are STORED and where the
// consumer ADDRESSES them — and is otherwise one copy of one function. That is
// the shape `RunGdnBlockPaged` established next door: one implementation, a
// second entry point, no second body to keep bit-identical by hand.
struct KvArm {
  const Qwen4ExpQsaCaches* contig = nullptr;
  const Qwen4ExpQsaPagedCaches* paged = nullptr;
};

}  // namespace

Qwen4ExpQsaSelection Qwen4ExpQsaIndex(Dev d, const Qwen4ExpQsaParams& qsa, float rms_norm_eps,
                                      const Tensor& q_index, const Tensor& index_key,
                                      const Tensor& k_norm_w, const Tensor& cos, const Tensor& sin,
                                      const Tensor& kv_lens, int64_t kv_len,
                                      bool round_intermediates_to_bf16, Tensor* logits) {
  const int64_t T = q_index.shape[0];
  const int64_t H = qsa.n_heads;
  const int64_t D = qsa.head_dim;
  const int64_t CR = qsa.compress_ratio;
  const int64_t topk = qsa.block_topk();

  VT_CHECK(q_index.rank == 3 && q_index.shape[1] == H && q_index.shape[2] == D,
           "qwen4_exp qsa indexer: q_index must be [T, indexer_n_heads, indexer_head_dim]");
  VT_CHECK(qsa.kv_heads == 1,
           "qwen4_exp qsa indexer: upstream requires indexer_kv_heads == 1 "
           "(configuration_qwen4_exp.py), and the side cache is one vector per state");
  VT_CHECK(index_key.rank == 2 && index_key.shape[1] == D,
           "qwen4_exp qsa indexer: the indexer side cache must be [max_kv, indexer_head_dim]");
  VT_CHECK(kv_len > 0 && kv_len <= index_key.shape[0],
           "qwen4_exp qsa indexer: kv_len outside the side cache");

  // ONLY COMPLETE BLOCKS PRODUCE A STATE (`(position + 1) % compress_ratio == 0`,
  // modeling_qwen4_exp.py:684). The ragged tail costs no state and is attended
  // from the raw KV cache unconditionally by the consumer, whatever the scores
  // said — so it is dropped HERE rather than scored.
  const int64_t complete_keys = (kv_len / CR) * CR;
  const int64_t nb = complete_keys / CR;

  // The pooled-key scratch. A DENSE `[nb, D]` array and not a paged one: the
  // side cache's paged store belongs to the wave that gives QSA a real KV-cache
  // group, which is blocked behind #2131, and the spec's `## Owed` says so.
  DBuf block_keys(d, index_key.dtype, {nb > 0 ? nb : 1, D});

  if (nb > 0) {
    vt::Qwen4ExpQsaCompressArgs cargs;
    cargs.compress_ratio = CR;
    // The rope span is the MODEL's `int(head_dim * partial_rotary_factor)`, which
    // upstream then requires to FIT `indexer_head_dim`
    // (configuration_qwen4_exp.py:225-231) — it is not derived from the indexer's
    // own width, and deriving it would make that requirement unfalsifiable.
    cargs.rotary_dim = cos.shape[1];
    cargs.eps = rms_norm_eps;
    cargs.round_intermediates_to_bf16 = round_intermediates_to_bf16;
    Tensor raw = RowsView(index_key, 0, complete_keys, {complete_keys, D});
    vt::Qwen4ExpQsaCompress(d.q, block_keys.t(), raw, k_norm_w, cos, sin, cargs);
  }

  // ─── THE FOUR SETTINGS, IN PRODUCTION CODE ─────────────────────────────────
  //
  // 1. `weights` ALL ONES. This is what collapses `vt::DsaIndexerLogits`'s fold
  //    `weights[t,h] * q_scale[t,h] * softmax_scale * n_head_scale` to the single
  //    constant `softmax_scale`, and that collapse is the entire claim that this
  //    op IS `Qwen4ExpTextQSAIndexer`'s block score. QSA has no `weights_proj`
  //    and no tensor a non-uniform value could come from. Spec mutation M27 is
  //    the paired red: non-uniform weights break the collapse and red the suite.
  VT_CHECK(q_index.dtype == DType::kF32 || q_index.dtype == DType::kBF16,
           "qwen4_exp qsa indexer: q_index must be f32 or bf16 (the op's fold operands "
           "must share one float dtype)");
  std::vector<uint8_t> ones_host(static_cast<size_t>(T * H) * vt::SizeOf(q_index.dtype));
  if (q_index.dtype == DType::kF32) {
    auto* p = reinterpret_cast<float*>(ones_host.data());
    for (int64_t i = 0; i < T * H; ++i) p[i] = 1.0f;
  } else {
    auto* p = reinterpret_cast<uint16_t*>(ones_host.data());
    const uint16_t one = vt::F32ToBF16(1.0f);
    for (int64_t i = 0; i < T * H; ++i) p[i] = one;
  }
  DBuf ones(d, q_index.dtype, {T, H}, ones_host.data());

  // 4. `win_end == kv_len / compress_ratio` PER QUERY TOKEN — the COMPLETE
  //    VISIBLE blocks, not the whole cache. Upstream forms its candidate blocks
  //    out of `local_visible_indices`, the nonzero columns of that query's own
  //    row of the causal mask (modeling_qwen4_exp.py:670-676), so a query at
  //    position p sees `(p + 1) / CR` blocks and not `kv_len / CR`. Spec
  //    mutation M28 is the paired red: widening this to the whole cache reds the
  //    suite. `win_start` is 0 because the visible prefix is contiguous — the
  //    op's `## Owed` records that an arbitrary visibility set is not expressible
  //    and that nothing yet detects one.
  std::vector<int32_t> ws_host(static_cast<size_t>(T), 0);
  std::vector<int32_t> we_host(static_cast<size_t>(T), 0);
  {
    VT_CHECK(kv_lens.rank == 1 && kv_lens.shape[0] == T && kv_lens.dtype == DType::kI32,
             "qwen4_exp qsa indexer: kv_lens must be i32 [T]");
    // READ ON THE HOST, whatever device it lives on. This used to REFUSE a
    // device-resident `kv_lens` by name; it is `[T]` i32 words whose only
    // product is `win_start`/`win_end`, two `[T]` i32 vectors uploaded three
    // lines below, so the window is built where it was always built and the
    // words are copied to get there (#2421).
    const std::vector<int32_t> kl = HostWords<int32_t>(d, kv_lens, 0, T);
    for (int64_t t = 0; t < T; ++t)
      we_host[static_cast<size_t>(t)] =
          static_cast<int32_t>(kl[static_cast<size_t>(t)] / CR);
  }
  DBuf win_start(d, DType::kI32, {T}, ws_host.data());
  DBuf win_end(d, DType::kI32, {T}, we_host.data());

  DBuf owned_logits;
  Tensor logits_t;
  if (logits != nullptr) {
    VT_CHECK(logits->rank == 2 && logits->shape[0] == T && logits->shape[1] == nb &&
                 logits->dtype == DType::kF32,
             "qwen4_exp qsa indexer: the logits tap must be f32 [T, kv_len / compress_ratio]");
    logits_t = *logits;
  } else {
    owned_logits = DBuf(d, DType::kF32, {T, nb > 0 ? nb : 1});
    logits_t = owned_logits.t();
  }

  DBuf block_ids(d, DType::kI32, {T, topk});
  DBuf counts(d, DType::kI32, {T});

  if (nb > 0) {
    vt::DsaIndexerLogitsArgs largs;
    // 3. `softmax_scale == index_head_dim ** -0.5`. Upstream divides the summed
    //    relu AFTER the head reduction (`/ math.sqrt(self.index_head_dim)`,
    //    modeling_qwen4_exp.py:693) and this fold multiplies BEFORE it —
    //    `c * sum_h r_h` against `sum_h c * r_h`, equal in exact arithmetic and up
    //    to an ulp apart in f32. The gate compares the LOGITS BY VALUE against
    //    the oracle's own pre-top-k tensor rather than resting on that argument.
    largs.softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));
    // 2. `n_head_scale == 1.0f`, NOT DeepSeek-V4's `n_head ** -0.5`. QSA's
    //    scoring line has no such factor and the checkpoint has no tensor for it.
    //    A wrong value here CANNOT move a selection — top-k is invariant under a
    //    positive rescale, which spec mutation M26 measures as a survival — so
    //    the only gate that convicts it is the value comparison on the logits.
    largs.n_head_scale = 1.0f;
    // `q_scale` stays null: that is upstream's `q_scale == 1`, the unquantized
    // arm. It is the ONE member of the fold a selection could see, and QSA has
    // no per-token-per-head quantization scale to put there.
    vt::DsaIndexerLogits(d.q, logits_t, q_index, block_keys.t(), ones.t(), win_start.t(),
                         win_end.t(), largs);
  }
  // `DsaTopkSelect` is the same all-select-below-k, ties-to-the-LOWER-index,
  // ASCENDING-emission top-k QSA needs, applied to the BLOCK axis instead of the
  // token axis. The ascending emission is load-bearing downstream: it is what
  // makes a sub-budget gather reduce over the same positions in the same order
  // dense attention would, and therefore bit-identical to it.
  //
  // With `nb == 0` every window is empty, so this writes the all-`-1` / all-zero
  // selection that the consumer reads as "ragged tail only" — which is exactly
  // upstream's `num_complete_blocks == 0` branch (modeling_qwen4_exp.py:698-700).
  vt::DsaTopkSelect(d.q, block_ids.t(), counts.t(), logits_t, win_start.t(), win_end.t());

  Qwen4ExpQsaSelection sel;
  sel.block_ids = block_ids.t();
  sel.counts = counts.t();
  sel.storage = std::make_shared<std::pair<std::shared_ptr<void>, std::shared_ptr<void>>>(
      block_ids.ReleaseShared(), counts.ReleaseShared());
  return sel;
}

namespace {

// ONE BLOCK BODY. `arm` selects the cache shape; see `KvArm` above for why the
// two entry points below are wrappers over this and not two functions.
Qwen4ExpQsaBlockOutput QsaBlockCore(Dev d, const Qwen4ExpQsaWeights& w,
                                    const Qwen4ExpParams& params, const Tensor& hidden,
                                    const Tensor& positions, const Tensor& cos_sin,
                                    const Tensor& cos, const Tensor& sin, const KvArm& arm,
                                    int64_t past_len, int64_t* keys_visited) {
  const bool paged = arm.paged != nullptr;
  VT_CHECK((arm.contig != nullptr) != paged,
           "qwen4_exp qsa block: exactly one cache arm must be set");
  const int64_t T = hidden.shape[0];
  const int64_t H = params.hidden_size;
  const int64_t Hq = params.num_attention_heads;
  const int64_t Hkv = params.num_key_value_heads;
  const int64_t Dh = params.head_dim;
  const int64_t rot = params.rotary_dim;
  const int64_t IdxH = params.qsa.n_heads;
  const int64_t IdxD = params.qsa.head_dim;
  const int64_t CR = params.qsa.compress_ratio;
  const auto eps = static_cast<float>(params.rms_norm_eps);
  const int64_t kv_len = past_len + T;

  VT_CHECK(hidden.rank == 2 && hidden.shape[1] == H,
           "qwen4_exp qsa block: hidden must be [T, hidden_size] — the gated residual's "
           "COLLAPSED output, never the hc_count-wide stream");
  // THE BLOCK DTYPE IS bf16, AND THAT IS INHERITED RATHER THAN CHOSEN. vLLM
  // resolves ONE model dtype and every layer inherits it (AGENTS.md, "Inherit
  // vLLM defaults"); concretely, every `vt::` output-gate op in this tree —
  // `SigmoidGateBf16`, `SharedExpertGate` — STORES bf16 on every backend, so an
  // f32 arm here would have to widen a shared dispatcher across five backends
  // this row cannot gate. Refused by name rather than silently rounded.
  VT_CHECK(hidden.dtype == DType::kBF16,
           "qwen4_exp qsa block: hidden must be bf16 (the model dtype). An f32 arm needs "
           "an f32 output gate, and every vt:: gate op stores bf16; see the spec's "
           "`## Owed`");
  VT_CHECK(Hq > 0 && Hkv > 0 && Hq % Hkv == 0,
           "qwen4_exp qsa block: num_attention_heads must be a multiple of num_key_value_heads");
  VT_CHECK(rot > 0 && rot % 2 == 0 && rot <= Dh && rot <= IdxD,
           "qwen4_exp qsa block: rotary_dim must be even and fit BOTH head_dim and "
           "indexer_head_dim (configuration_qwen4_exp.py:225-231)");
  VT_CHECK(cos_sin.rank == 2 && cos_sin.shape[1] == rot && cos_sin.dtype == hidden.dtype,
           "qwen4_exp qsa block: cos_sin must be the [P, rotary_dim] PACKED cos|sin cache "
           "vt::RopeFromCache reads, at the block dtype");
  VT_CHECK(cos.rank == 2 && sin.rank == 2 && cos.shape[1] == rot && sin.shape[1] == rot &&
               cos.dtype == DType::kF32 && sin.dtype == DType::kF32,
           "qwen4_exp qsa block: cos/sin must be the f32 [P, rotary_dim] FULL tables "
           "vt::Qwen4ExpQsaCompress reads — a second layout for the same angles, because "
           "the two ops were ported from two upstreams that spell it differently");
  // The two layouts are cross-checked rather than trusted. See
  // `CheckRopeLayoutsAgree` for what the bounded row sample can and cannot see.
  CheckRopeLayoutsAgree(d, cos_sin, cos, sin);
  VT_CHECK(past_len >= 0, "qwen4_exp qsa block: past_len must not be negative");
  if (!paged) {
    const Qwen4ExpQsaCaches& caches = *arm.contig;
    VT_CHECK(caches.key.rank == 3 && caches.value.rank == 3 && caches.key.shape[1] == Hkv &&
                 caches.key.shape[2] == Dh && caches.value.shape[1] == Hkv &&
                 caches.value.shape[2] == Dh,
             "qwen4_exp qsa block: key/value caches must be [max_kv, num_kv_heads, head_dim]");
    VT_CHECK(kv_len <= caches.key.shape[0] && kv_len <= caches.value.shape[0],
             "qwen4_exp qsa block: the new tokens do not fit the key/value caches");
  } else {
    const Qwen4ExpQsaPagedCaches& pc = *arm.paged;
    // An fp8 KV cache is REFUSED BY NAME. `vt::Qwen4ExpQsaGatherAttention` has
    // no dequantising read and no `k_scale`/`v_scale`, so reading fp8 bytes
    // through it is wrong tokens rather than a crash — the exact shape
    // `kv_cache_route.h` exists to prevent. The store would take the fp8 branch
    // and the read would not, which is that header's named failure verbatim.
    VT_CHECK(!dense_attn::IsFp8KvCache(pc.kv),
             "qwen4_exp qsa block: an fp8 paged KV cache is not supported — "
             "vt::Qwen4ExpQsaGatherAttention has no dequantising read. See the spec's "
             "`## Owed`");
    VT_CHECK(pc.kv.data != nullptr && pc.kv.num_blocks > 0 && pc.kv.block_size > 0,
             "qwen4_exp qsa block: the paged KV cache is unallocated");
    VT_CHECK(pc.kv.num_kv_heads == Hkv && pc.kv.head_size == Dh,
             "qwen4_exp qsa block: the paged KV cache head dims disagree with the config");
    VT_CHECK(pc.kv.dtype == hidden.dtype,
             "qwen4_exp qsa block: the paged KV cache dtype must be the block dtype — "
             "vt::ReshapeAndCache's `auto` path copies raw elements and does not cast");
    // THIS IS THE ONLY SITE THAT REFUSES IT NOW, and the sentence here used to
    // say otherwise. It read "`MakeQwen4ExpKVCache` already refuses a
    // `block_size` the compress ratio does not divide"; W5h DELETED that
    // refusal, correctly — it guarded a COMPRESSED page geometry and group 2 is
    // not one, so at `compress_ratio` 1 there is no division to truncate. The
    // reason the check survives HERE is a different one and it is group 0's:
    // it keeps a compress block of CR tokens inside ONE page, so the consumer
    // never has to resolve two pages for one selected block.
    VT_CHECK(pc.kv.block_size % CR == 0,
             "qwen4_exp qsa block: the KV page size must be a multiple of "
             "`indexer_compress_ratio`, which MakeQwen4ExpKVCache already requires");
    VT_CHECK(pc.block_table.rank == 2 && pc.block_table.shape[0] == 1 &&
                 pc.block_table.dtype == DType::kI32 && pc.block_table.IsContiguous(),
             "qwen4_exp qsa block: block_table must be a contiguous i32 [1, max_pages] — "
             "this block serves ONE sequence per call");
    VT_CHECK(pc.block_table.shape[1] * pc.kv.block_size >= kv_len,
             "qwen4_exp qsa block: the block table names fewer tokens than kv_len");
    VT_CHECK(pc.slot_mapping.rank == 1 && pc.slot_mapping.shape[0] == T &&
                 pc.slot_mapping.dtype == DType::kI64 && pc.slot_mapping.IsContiguous(),
             "qwen4_exp qsa block: slot_mapping must be a contiguous i64 [T]");
    // ─── GROUP 2, the indexer side cache (W5i, #2249 item 3) ────────────────
    // The runner's fused MLA page, `[num_pages, block_size, indexer_head_dim]`.
    // Rank 3 and not a flat `[max_kv, D]`: the page geometry has to be READABLE
    // to translate a logical position, and a shape is the one place it cannot
    // disagree with the buffer.
    VT_CHECK(pc.index_key.rank == 3 && pc.index_key.shape[2] == IdxD &&
                 pc.index_key.IsContiguous() && pc.index_key.data != nullptr,
             "qwen4_exp qsa block: the paged indexer side cache must be a "
             "contiguous [num_pages, block_size, indexer_head_dim] — the fused "
             "3-dim page an MLAAttentionSpec group is allocated as");
    VT_CHECK(pc.index_key.shape[0] > 0 && pc.index_key.shape[1] > 0,
             "qwen4_exp qsa block: the paged indexer side cache is unallocated");
    // GROUP 2'S OWN TABLE. Not `block_table` above: the two groups are allocated
    // from separate physical page pools, so reading one through the other's map
    // returns another sequence's keys with no shape error.
    VT_CHECK(pc.index_block_table.rank == 2 && pc.index_block_table.shape[0] == 1 &&
                 pc.index_block_table.dtype == DType::kI32 &&
                 pc.index_block_table.IsContiguous(),
             "qwen4_exp qsa block: index_block_table must be a contiguous i32 "
             "[1, max_pages] — this block serves ONE sequence per call");
    VT_CHECK(pc.index_block_table.shape[1] * pc.index_key.shape[1] >= kv_len,
             "qwen4_exp qsa block: the indexer block table names fewer tokens "
             "than kv_len");
    // The physical row index travels as i32, because that is what
    // `vt::IndexSelect` / `vt::IndexCopy` take. Refuse a cache whose last row
    // cannot be NAMED rather than wrap into a valid-looking small index.
    VT_CHECK(pc.index_key.shape[0] * pc.index_key.shape[1] <=
                 static_cast<int64_t>(INT32_MAX),
             "qwen4_exp qsa block: the paged indexer side cache has more rows "
             "than an i32 slot index can name");
  }

  // ─── THE INDEXER SIDE CACHE, AS A FLAT ARRAY OF ROWS ──────────────────────
  // Both arms end up here: the contiguous arm's cache already IS
  // `[max_kv, indexer_head_dim]` addressed by logical position, and the paged
  // arm's `[num_pages, block_size, D]` pages flatten — same bytes, no copy — into
  // `[num_pages * block_size, D]` addressed by PHYSICAL slot. `idx_page` is what
  // separates the two: on the paged arm a logical position must go through
  // `IndexerRows` before it names a row here, and on the contiguous arm it is the
  // row.
  // ─── GROUP 2'S BLOCK TABLE, RESOLVED ON THE HOST ONCE PER CALL (#2421) ────
  // `IndexerRows` runs TWICE below — the scatter of this step's rows, and the
  // gather of the visible prefix — and both read the same `[1, pages]` table. It
  // is resolved here, above both, so a device arm pays ONE round trip per block
  // call rather than one per use. Placed after the paged validation above,
  // because that is what establishes the rank, dtype and contiguity this read
  // relies on.
  std::vector<int32_t> idx_table;
  if (paged) {
    idx_table = HostWords<int32_t>(d, arm.paged->index_block_table, 0,
                                   arm.paged->index_block_table.shape[1]);
  }

  const int64_t idx_page = paged ? arm.paged->index_key.shape[1] : 0;
  Tensor index_rows =
      paged ? Reshape(arm.paged->index_key,
                      {arm.paged->index_key.shape[0] * idx_page,
                       arm.paged->index_key.shape[2]})
            : arm.contig->index_key;

  vt::RopeArgs rope;
  rope.rotary_dim = static_cast<int>(rot);
  // NeoX HALF-SPLIT, and the halves are swapped end for end against
  // DeepSeek-V4's indexer: `apply_rotary_pos_emb` rotates the LEADING
  // `rotary_dim` dims with `rotate_half` and concatenates the NoPE dims back
  // UNTOUCHED (modeling_qwen4_exp.py:566-608). `vt::RopeFromCache` with
  // `is_neox_style` does exactly that and leaves dims >= rotary_dim alone.
  rope.is_neox_style = true;

  // ─── THE INDEXER RUNS FIRST ────────────────────────────────────────────────
  // Upstream calls `self.indexer(...)` on the FIRST line of the attention
  // forward (:786), before any q/k/v projection, because the selection is an
  // input to the attention and not a refinement of it.
  //
  // `index_qk_proj` is ONE `nn.Linear` upstream, SPLIT by the converter into two
  // tensors at `indexer_n_heads * indexer_head_dim` ("one projection feeds
  // indexer q and k; split it, as minimax-m3 does", llama.cpp#27742). The split
  // is a slice of a row-major [N, K] weight, so projecting through the two
  // halves separately is the same arithmetic as projecting through the whole.
  DBuf q_index_raw(d, hidden.dtype, {T, IdxH * IdxD});
  vt::MatmulBT(d.q, q_index_raw.t(), hidden,
               dense_attn::ResidentWeight(d, w.idx_q_proj, {IdxH * IdxD, H}));

  // The indexer's raw key goes STRAIGHT INTO THE SIDE CACHE, un-normed and
  // un-roped, which is what `Cache.update_indexer` stores (:653) and what
  // `vt::Qwen4ExpQsaCompress` expects — it applies the norm and the block-start
  // rope itself. Storing a normed or roped key here would double-apply both.
  //
  // WHERE IT LANDS, and this is the second place the cache arm forks. The
  // CONTIGUOUS arm projects DIRECTLY INTO the rows this step owns, so nothing is
  // copied afterwards. The PAGED arm cannot: a step's tokens can cross a page
  // boundary, so there are no "the rows this step owns" to project into. It
  // stages `[T, IdxD]` and scatters with `vt::IndexCopy` at the physical rows
  // group 2's table names — the same stage-then-scatter shape the K/V takes
  // below, for the same reason.
  {
    DBuf idx_stage;
    if (paged) idx_stage = DBuf(d, index_rows.dtype, {T, IdxD});
    Tensor slot = paged ? idx_stage.t() : RowsView(index_rows, past_len, T, {T, IdxD});
    vt::MatmulBT(d.q, slot, hidden, dense_attn::ResidentWeight(d, w.idx_k_proj, {IdxD, H}));
    if (paged) {
      const std::vector<int32_t> rows = IndexerRows(idx_table, idx_page, past_len, T);
      DBuf ridx(d, DType::kI32, {T}, rows.data());
      vt::IndexCopy(d.q, index_rows, idx_stage.t(), ridx.t());
    }
  }

  // `q = self.q_layernorm(q)` then `apply_rotary_pos_emb(q, cos=current_cos,
  // sin=current_sin)` (:651-652). CURRENT positions, not the full table: the
  // cos/sin the indexer is handed cover every KEY position and the query slices
  // the last `seq_length` rows out of them. `RopeFromCache` takes the positions
  // themselves, so the slice is expressed as "these tokens' positions" instead.
  {
    Tensor flat = Reshape(q_index_raw.t(), {T * IdxH, IdxD});
    // `gemma = true` IS `Qwen4ExpTextRMSNorm`'s polarity: `output * (1.0 + weight)`
    // on a ZERO-initialised gamma (:177). The loader stores the raw HuggingFace
    // value — it INVERTS the converter's baked `+1` — so the `+1` belongs here,
    // and a port that passed the gamma through an `out * w` norm would apply a
    // near-zero scale and read as a checkpoint bug.
    vt::RmsNorm(d.q, flat, flat, dense_attn::ResidentWeight(d, w.idx_q_norm, {IdxD}),
                vt::RmsNormArgs{eps, /*gemma=*/true});
  }
  Tensor q_index = Reshape(q_index_raw.t(), {T, IdxH, IdxD});
  vt::RopeFromCache(d.q, q_index, nullptr, positions, cos_sin, rope);

  // The causal visible length of each query token. CONTIGUOUS PREFIX, which is
  // what a serving engine's ragged batch gives and what both `vt::` ops below
  // assume; upstream's general form reads an arbitrary visibility set out of a
  // padded batch's mask, and the ops' `## Owed` records that nothing here can
  // detect one.
  std::vector<int32_t> kv_lens_host(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    kv_lens_host[static_cast<size_t>(t)] = static_cast<int32_t>(past_len + t + 1);
  // ONE tensor for one value, and it is the DEVICE one (#2421). Until this
  // change the same `[T]` i32 words were wrapped TWICE — a CPU `Tensor` over
  // `kv_lens_host` for the indexer, whose refusal made a host tensor the only
  // legal operand, and this `DBuf` for the gather consumer. Two objects for one
  // value is a drift hazard for the sake of a refusal that is now gone, so the
  // gather's operand is the indexer's operand. On a CPU queue the `DBuf` IS host
  // memory and the indexer's read is a memcpy; on a device arm it is one round
  // trip, which is what makes that path REACHED from the production forward
  // rather than exercised only by a test.
  DBuf kv_lens(d, DType::kI32, {T}, kv_lens_host.data());

  // WHAT THE INDEXER READS. `Qwen4ExpQsaIndex` takes a CONTIGUOUS `[rows, D]`
  // addressed by logical position and its contract is unchanged by W5i, because
  // the paged arm GATHERS one for it: `vt::IndexSelect` over the visible prefix's
  // physical rows. That is upstream's own shape rather than a concession —
  // `Cache.update_indexer` returns the whole concatenated
  // `[batch, total_len, index_head_dim]` and the indexer `index_select`s out of
  // it (`cache_utils.py:350-351`, `modeling_qwen4_exp.py:679`).
  //
  // IT COSTS ONE EXTRA PASS OVER THE VISIBLE PREFIX PER LAYER PER STEP, and that
  // is recorded rather than hidden. It is not an asymptotic change:
  // `vt::Qwen4ExpQsaCompress` already streams all `complete_keys` rows every step
  // — it rebuilds every pooled block key and caches none, exactly as upstream
  // does — so the gather adds a constant to a pass that was already O(kv_len).
  // Folding the page resolution into that op instead would remove the constant;
  // the spec's `## Owed` carries it with the issue that owes the measurement.
  DBuf idx_gathered;
  Tensor index_visible = index_rows;
  if (paged) {
    idx_gathered = DBuf(d, index_rows.dtype, {kv_len, IdxD});
    const std::vector<int32_t> rows = IndexerRows(idx_table, idx_page, 0, kv_len);
    DBuf ridx(d, DType::kI32, {kv_len}, rows.data());
    vt::IndexSelect(d.q, idx_gathered.t(), index_rows, ridx.t());
    index_visible = idx_gathered.t();
  }
  Qwen4ExpQsaSelection sel = Qwen4ExpQsaIndex(
      d, params.qsa, eps, q_index, index_visible,
      dense_attn::ResidentWeight(d, w.idx_k_norm, {IdxD}), cos, sin, kv_lens.t(), kv_len,
      /*round_intermediates_to_bf16=*/index_visible.dtype == DType::kBF16);

  // ─── THE ATTENTION ─────────────────────────────────────────────────────────
  // `q_proj` emits `num_attention_heads * head_dim * 2` and is chunked PER HEAD
  // into the query and the OUTPUT GATE (:809-811) — not two heads' worth of
  // queries. `vt::AttnGateSplit` is exactly that layout, `[q(Dh) | gate(Dh)]`
  // per (t, head).
  DBuf qgate(d, hidden.dtype, {T, Hq * 2 * Dh});
  vt::MatmulBT(d.q, qgate.t(), hidden,
               dense_attn::ResidentWeight(d, w.q_proj, {Hq * 2 * Dh, H}));
  //
  // THE QUERY HALF IS SPLIT AT THE MODEL DTYPE, NOT WIDENED (#2488). vLLM reaches
  // this point through `torch.chunk` (qwen3_next.py:430) and hands `q` to `q_norm`
  // unwidened (:437), so `q` carries `model_config.dtype` — bf16 for this
  // architecture — from the GEMM to the norm.
  //
  // An earlier comment here justified an f32 `q` as upstream's `_norm(x.float())`
  // then `.type_as(x)`. That was RIGHT about the rounding count and WRONG about
  // what it licenses: upstream's `.float()` is a promotion inside the norm kernel's
  // registers (ple_layer.py:70 then :80 `.to(input_dtype)`), not a materialised
  // `[T, Hq, Dh]` allocation. The rounding count is one either way, because `qgate`
  // is ALREADY at `hidden.dtype` — so widening each element to f32 and narrowing it
  // back is the identity, and all it buys is `T*Hq*Dh*4` bytes where upstream moves
  // half that. `AGENTS.md` names this exactly: a token gate cannot see a dtype that
  // is too wide, so the gate on this line reads the dtype and the byte count.
  //
  // It is also what made #2477 a wall. An f32 `q` beside the bf16 gamma the loader
  // produces was the pairing the CUDA RmsNorm kernel refused; #2493 taught the
  // kernel that pairing and said it was treating a symptom. This is the cause.
  //
  // The GATE half stays f32: `vt::SigmoidGateBf16` takes an f32 gate on each of
  // its four backends. #2488 stays open for it.
  DBuf q_split(d, hidden.dtype, {T, Hq, Dh});
  DBuf gate(d, DType::kF32, {T, Hq, Dh});
  vt::AttnGateSplit(d.q, q_split.t(), gate.t(), qgate.t());

  // q_norm reads the split and stores the block dtype, which is upstream's
  // `_norm(x.float()) * (1 + w)` followed by `.to(input_dtype)`: one rounding, at
  // the store, exactly where upstream has one. Both operands are now the block
  // dtype, which is the pairing every backend's RmsNorm already served.
  DBuf q(d, hidden.dtype, {T, Hq, Dh});
  {
    Tensor src = Reshape(q_split.t(), {T * Hq, Dh});
    Tensor dst = Reshape(q.t(), {T * Hq, Dh});
    vt::RmsNorm(d.q, dst, src, dense_attn::ResidentWeight(d, w.q_norm, {Dh}),
                vt::RmsNormArgs{eps, /*gemma=*/true});
  }

  // WHERE THE NEW K/V ROWS LAND — the FIRST of the two places the cache arm is
  // read, and the one that has no shared shape.
  //
  // CONTIGUOUS ARM (W5b-5, unchanged): k and v are projected DIRECTLY INTO the
  // cache rows this step owns, so nothing is copied afterwards and there is no
  // second buffer that could drift from the cache.
  //
  // PAGED ARM (W5d-3): a step's tokens can cross a page boundary, so there are no
  // "the rows this step owns" to project into. k and v go to a staging buffer and
  // `dense_attn::WriteKvCache` scatters them at the slot mapping — which is
  // `dense_attn::AttnBlock`'s own order, and upstream's: `past_key_values.update`
  // is called AFTER the norm and the rope (:826), so the cache holds normed,
  // roped keys and raw values either way. The arithmetic reaching the cache is
  // the same in both arms; only the destination differs.
  DBuf k_stage;
  DBuf v_stage;
  if (paged) {
    k_stage = DBuf(d, hidden.dtype, {T, Hkv, Dh});
    v_stage = DBuf(d, hidden.dtype, {T, Hkv, Dh});
  }
  Tensor k_slot = paged ? k_stage.t() : RowsView(arm.contig->key, past_len, T, {T, Hkv, Dh});
  Tensor v_slot = paged ? v_stage.t() : RowsView(arm.contig->value, past_len, T, {T, Hkv, Dh});
  {
    DBuf k_raw(d, hidden.dtype, {T, Hkv * Dh});
    vt::MatmulBT(d.q, k_raw.t(), hidden, dense_attn::ResidentWeight(d, w.k_proj, {Hkv * Dh, H}));
    Tensor dst = Reshape(k_slot, {T * Hkv, Dh});
    Tensor src = Reshape(k_raw.t(), {T * Hkv, Dh});
    vt::RmsNorm(d.q, dst, src, dense_attn::ResidentWeight(d, w.k_norm, {Dh}),
                vt::RmsNormArgs{eps, /*gemma=*/true});
  }
  {
    Tensor dst = Reshape(v_slot, {T, Hkv * Dh});
    vt::MatmulBT(d.q, dst, hidden, dense_attn::ResidentWeight(d, w.v_proj, {Hkv * Dh, H}));
  }
  // ONE rope call over q and k together, as upstream does (:824). On the
  // contiguous arm the k operand IS the cache slice, so the cache holds the roped
  // key with no copy; on the paged arm it is the staging buffer, roped before the
  // scatter for the same reason `dense_attn::AttnBlock` ropes before its
  // `WriteKvCache`.
  vt::RopeFromCache(d.q, q.t(), &k_slot, positions, cos_sin, rope);

  if (paged) {
    Tensor kc_w = dense_attn::KvSlice(arm.paged->kv, d.q.device, 0);
    Tensor vc_w = dense_attn::KvSlice(arm.paged->kv, d.q.device, 1);
    dense_attn::WriteKvCache(d.q, arm.paged->kv, k_slot, v_slot, kc_w, vc_w,
                             arm.paged->slot_mapping);
  }

  // THE GATHER CONSUMER. Selected block `b` IS tokens [CR*b, CR*b + CR), expanded
  // as ADDRESSES inside the op and never materialised as a token buffer, plus
  // the ALWAYS-attended ragged tail. A sparse MASK over the dense cache would
  // agree with this VALUE FOR VALUE — `exp(-inf - m)` is exactly +0 — and would
  // forfeit the long-context lever this row exists for, which is why
  // `keys_visited` is forwarded and why the gate runs the NaN-poison and
  // unmapped-tail probes over this call rather than reading its shape.
  DBuf attn(d, hidden.dtype, {T, Hq, Dh});
  {
    vt::Qwen4ExpQsaAttnArgs aargs;
    // The MODEL's head_dim, not the indexer's (:764 `self.scaling = self.head_dim**-0.5`).
    aargs.scale = 1.0f / std::sqrt(static_cast<float>(Dh));
    aargs.compress_ratio = CR;
    aargs.keys_visited = keys_visited;
    // THE SECOND — and last — place the cache arm is read. The consumer resolves
    // a key row flatly or through the page table; nothing else about the call
    // changes, which is why this is one op with two address modes rather than
    // two ops (see `Qwen4ExpQsaAttnArgs::kv_block_table`).
    Tensor kc;
    Tensor vc;
    if (paged) {
      kc = dense_attn::KvSlice(arm.paged->kv, d.q.device, 0);
      vc = dense_attn::KvSlice(arm.paged->kv, d.q.device, 1);
      aargs.kv_block_table = &arm.paged->block_table;
      aargs.kv_block_size = arm.paged->kv.block_size;
    } else {
      kc = RowsView(arm.contig->key, 0, kv_len, {kv_len, Hkv, Dh});
      vc = RowsView(arm.contig->value, 0, kv_len, {kv_len, Hkv, Dh});
    }
    vt::Qwen4ExpQsaGatherAttention(d.q, attn.t(), q.t(), kc, vc, sel.block_ids, kv_lens.t(), aargs);
  }

  // `attn_output * torch.sigmoid(gate)` then `o_proj` (:838-840). The gate is
  // read at f32 because that is the operand `vt::SigmoidGateBf16` states on each
  // of its four backends. It is NOT because this value needs the width: `qgate` is
  // `hidden.dtype`, so the f32 gate holds bf16 values in f32 storage, exactly as
  // the query did before #2488's first half. Narrowing it means widening that op
  // on four backends for zero rounding change, which is why #2488 stays open.
  DBuf gated(d, DType::kBF16, {T, Hq * Dh});
  vt::SigmoidGateBf16(d.q, gated.t(), Reshape(attn.t(), {T, Hq * Dh}),
                      Reshape(gate.t(), {T, Hq * Dh}));

  DBuf out(d, hidden.dtype, {T, H});
  vt::MatmulBT(d.q, out.t(), gated.t(), dense_attn::ResidentWeight(d, w.o_proj, {H, Hq * Dh}));

  Qwen4ExpQsaBlockOutput r;
  r.tensor = out.t();
  r.storage = out.ReleaseShared();
  return r;
}

}  // namespace

Qwen4ExpQsaBlockOutput RunQwen4ExpQsaBlock(Dev d, const Qwen4ExpQsaWeights& w,
                                           const Qwen4ExpParams& params, const Tensor& hidden,
                                           const Tensor& positions, const Tensor& cos_sin,
                                           const Tensor& cos, const Tensor& sin,
                                           const Qwen4ExpQsaCaches& caches, int64_t past_len,
                                           int64_t* keys_visited) {
  KvArm arm;
  arm.contig = &caches;
  return QsaBlockCore(d, w, params, hidden, positions, cos_sin, cos, sin, arm, past_len,
                      keys_visited);
}

Qwen4ExpQsaBlockOutput RunQwen4ExpQsaBlockPaged(Dev d, const Qwen4ExpQsaWeights& w,
                                                const Qwen4ExpParams& params, const Tensor& hidden,
                                                const Tensor& positions, const Tensor& cos_sin,
                                                const Tensor& cos, const Tensor& sin,
                                                const Qwen4ExpQsaPagedCaches& caches,
                                                int64_t past_len, int64_t* keys_visited) {
  KvArm arm;
  arm.paged = &caches;
  return QsaBlockCore(d, w, params, hidden, positions, cos_sin, cos, sin, arm, past_len,
                      keys_visited);
}

}  // namespace vllm

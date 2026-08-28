// Qwen4-Exp (Qwen3.8-Flash-Next) W5b-4 DEVICE-ARM GATE — Qwen Sparse Attention
// as `vt::` ops over `vt::Tensor`. Issue #2167, row MODEL-MM-QWEN4-EXP, spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// WHAT IS UNDER TEST. Two new ops — `vt::Qwen4ExpQsaCompress` (the pooled-key
// side-cache build) and `vt::Qwen4ExpQsaGatherAttention` (the consumer) — plus
// the COMPOSITION that turns them into an indexer using two ops this tree
// already had: `vt::DsaIndexerLogits` for the ReLU-summed MQA block score and
// `vt::DsaTopkSelect` for the per-query top-k. The composition is gated here
// rather than assumed, because "these two existing ops already are the QSA
// score" is a claim about arithmetic and this file is where it is measured.
//
// WHAT IT IS COMPARED AGAINST. `fixtures/qwen4_exp_qsa_goldens.inc`, the output
// of the UNMODIFIED `Qwen4ExpTextQSAIndexer.forward` at transformers 5.16.0 —
// this row's accepted ALGORITHM lane pin, vLLM registering no `qwen4_exp` at
// origin/main 6a5e8f5979. The same file gates the W4 host reference
// (`test_qwen4_exp_qsa.cpp`), so the two arms answer to ONE oracle instead of to
// each other.
//
// THE CONTEXT LENGTH MATTERS AND IS STATED. The spec's `## Gates` requires any
// QSA gate to run past `indexer_budget` tokens, because at or below the budget
// every candidate is selected and a short-prompt gate cannot tell a correct port
// from one attending POOLED keys. The goldens run a scaled config
// (`token_budget` 8, `compress_ratio` 4, so `index_width` 11) at kv_len 23 —
// 2.1x the buffer width, genuinely sparse — and the last two cases in this file
// additionally run the RELEASED indexer config (`token_budget` 2048,
// `compress_ratio` 4, `index_head_dim` 128, `rotary_dim` 64) at **3002 tokens of
// context**, which is past 2048 and where 512 of the 750 complete blocks are
// SELECTED and the other 238 are discarded. The sub-budget control beside it
// runs at 2051 and selects everything, which is the measurement of WHY the
// requirement exists.
//
// AND ONE GATE THAT IS NOT ABOUT CORRECTNESS. `keys_visited` says the gather
// READ only the selected rows. A sparse mask over a dense cache is correct and
// agrees with the gather value for value; llama.cpp #27739 names the mechanism
// by which it is also not faster. The counter is incremented at the key-row read
// inside the kernel and compared against a count derived INDEPENDENTLY from the
// host reference's own expansion of the selection — never against the kernel's
// own idea of how many rows it picked. W4's first fresh review found the version
// where those two were the same quantity computed the same way, under which a
// body doing the full dense work passed the case named after this property.
//
// SCOPE, HONESTLY. CPU only: no CUDA arm of either op exists and one written on
// this host could not be gated on it. Nothing calls either op from a production
// entry point — `ModelRegistry::Forward` has no `qwen4_exp` arm — so this lands
// UNREACHED, as the spec's `## Owed` records. No token claim and no speed claim.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen4_exp_qsa.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

// THE FETCH-LEVEL PROBE'S ONLY PLATFORM DEPENDENCY. `mmap`/`mprotect` are POSIX
// and MSVC has neither, so the case below is compiled where they exist and is
// declared SKIPPED where they are not — a probe that cannot run must say so,
// never fail and never quietly vanish.
#if !defined(_WIN32) && (defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)))
#define VT_QSA_MPROTECT_PROBE 1
#include <sys/mman.h>
#include <unistd.h>

#include <csetjmp>
#include <csignal>
#else
#define VT_QSA_MPROTECT_PROBE 0
#endif

using vllm::qwen4_exp::QsaBlockScores;
using vllm::qwen4_exp::QsaCompressNormRope;
using vllm::qwen4_exp::QsaConfig;
using vllm::qwen4_exp::QsaGatherAttention;
using vllm::qwen4_exp::QsaMaskedAttention;
using vllm::qwen4_exp::QsaSelectedTokenIndices;
using vllm::qwen4_exp::QsaTopkBlocks;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Qwen4ExpQsaAttnArgs;
using vt::Qwen4ExpQsaCompressArgs;
using vt::Queue;
using vt::Tensor;

#include "fixtures/qwen4_exp_qsa_goldens.inc"  // NOLINT — golden literals

namespace g = qwen4_exp_qsa_goldens;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQ() { return Queue{Cpu(), nullptr}; }

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
    stride *= t.shape[i];
  }
  return t;
}

std::vector<float> Slice(const float* p, int64_t n) { return std::vector<float>(p, p + n); }

double RelL2(const std::vector<float>& a, const float* b, int64_t n) {
  double num = 0.0, den = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    num += d * d;
    den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return std::sqrt(num) / std::max(std::sqrt(den), 1e-30);
}

int64_t SelectedCount(const std::vector<int32_t>& idx) {
  int64_t n = 0;
  for (int32_t v : idx) {
    if (v < 0) break;
    ++n;
  }
  return n;
}

QsaConfig GoldenConfig() {
  QsaConfig cfg;
  cfg.index_n_heads = g::kIndexNHeads;
  cfg.index_kv_heads = g::kIndexKvHeads;
  cfg.index_head_dim = g::kIndexHeadDim;
  cfg.token_budget = g::kTokenBudget;
  cfg.compress_ratio = g::kCompressRatio;
  cfg.rotary_dim = g::kRotaryDim;
  cfg.rms_norm_eps = g::kRmsNormEps;
  return cfg;
}

struct Case {
  const char* name;
  int64_t seq;
  const float* q_raw;
  const float* k_raw;
  const float* cos;
  const float* sin;
  const float* q_norm_w;
  const float* k_norm_w;
  const float* q_post;
  const float* block_keys;
  const int32_t* selected;
  const float* attn_q;
  const float* attn_k;
  const float* attn_v;
  const float* attn_out;
};

const Case kSubBudget{"sub_budget",           g::kSubBudgetSeq,       g::kSubBudgetQRaw,
                      g::kSubBudgetKRaw,      g::kSubBudgetCos,       g::kSubBudgetSin,
                      g::kSubBudgetQNormW,    g::kSubBudgetKNormW,    g::kSubBudgetQPost,
                      g::kSubBudgetBlockKeys, g::kSubBudgetSelected,  g::kSubBudgetAttnQ,
                      g::kSubBudgetAttnK,     g::kSubBudgetAttnV,     g::kSubBudgetAttnOut};

const Case kOverBudget{"over_budget",           g::kOverBudgetSeq,      g::kOverBudgetQRaw,
                       g::kOverBudgetKRaw,      g::kOverBudgetCos,      g::kOverBudgetSin,
                       g::kOverBudgetQNormW,    g::kOverBudgetKNormW,   g::kOverBudgetQPost,
                       g::kOverBudgetBlockKeys, g::kOverBudgetSelected, g::kOverBudgetAttnQ,
                       g::kOverBudgetAttnK,     g::kOverBudgetAttnV,    g::kOverBudgetAttnOut};

// `q` after `q_layernorm` and the partial rope, [seq, H, D]. The oracle dumped
// it (`q_post`), so this reads the golden rather than recomputing it: the query
// preamble is W4's gate, not this one's, and recomputing it here would make this
// file's failures ambiguous between two ports.
std::vector<float> GoldenQ(const Case& c, const QsaConfig& cfg) {
  return Slice(c.q_post, c.seq * cfg.index_n_heads * cfg.index_head_dim);
}

// ── The composed indexer: vt::Qwen4ExpQsaCompress + the two DSA ops ──────────
//
// This IS the wave's design claim in executable form. `weights` is all ones and
// `n_head_scale` is 1, so `DsaIndexerLogits`'s fold
// `weights * q_scale * softmax_scale * n_head_scale` collapses to the single
// constant `index_head_dim ** -0.5` — QSA's scale, which has neither
// DeepSeek-V4's learned `weights_proj` nor its `n_head ** -0.5`.
struct Indexed {
  std::vector<float> block_keys;  // [nb_total, D]
  std::vector<int32_t> block_ids;  // [T, block_topk], ascending, -1 padded
  std::vector<int32_t> counts;     // [T]
  std::vector<int32_t> kv_lens;    // [T]
};

// `kv_lens[t]` is the causal visible length of query token t. `complete` blocks
// are `kv_lens[t] / compress_ratio`, which is the per-query window handed to
// both DSA ops.
Indexed RunIndexer(Queue& q, const QsaConfig& cfg, const std::vector<float>& raw_keys,
                   const std::vector<float>& k_norm_w, const std::vector<float>& cos,
                   const std::vector<float>& sin, const std::vector<float>& q_index,
                   const std::vector<int32_t>& kv_lens, bool round_bf16) {
  const int64_t D = cfg.index_head_dim;
  const int64_t H = cfg.index_n_heads;
  const int64_t T = static_cast<int64_t>(kv_lens.size());
  const int64_t num_keys = static_cast<int64_t>(raw_keys.size()) / D;
  const int64_t complete_keys = (num_keys / cfg.compress_ratio) * cfg.compress_ratio;
  const int64_t nb_total = complete_keys / cfg.compress_ratio;

  Indexed out;
  out.kv_lens = kv_lens;
  out.block_keys.assign(static_cast<size_t>(nb_total * D), 0.0f);

  std::vector<float> raw_complete(raw_keys.begin(), raw_keys.begin() + complete_keys * D);
  Tensor t_raw = MakeT(raw_complete.data(), DType::kF32, {complete_keys, D});
  std::vector<float> knw = k_norm_w;
  Tensor t_knw = MakeT(knw.data(), DType::kF32, {D});
  std::vector<float> c = cos, s = sin;
  Tensor t_cos = MakeT(c.data(), DType::kF32, {num_keys, cfg.rotary_dim});
  Tensor t_sin = MakeT(s.data(), DType::kF32, {num_keys, cfg.rotary_dim});
  Tensor t_bk = MakeT(out.block_keys.data(), DType::kF32, {nb_total, D});

  Qwen4ExpQsaCompressArgs cargs;
  cargs.compress_ratio = cfg.compress_ratio;
  cargs.rotary_dim = cfg.rotary_dim;
  cargs.eps = cfg.rms_norm_eps;
  cargs.round_intermediates_to_bf16 = round_bf16;
  vt::Qwen4ExpQsaCompress(q, t_bk, t_raw, t_knw, t_cos, t_sin, cargs);

  // The scoring window: complete blocks only, [0, kv_len / compress_ratio).
  std::vector<int32_t> win_start(static_cast<size_t>(T), 0);
  std::vector<int32_t> win_end(static_cast<size_t>(T), 0);
  for (int64_t t = 0; t < T; ++t) {
    win_end[static_cast<size_t>(t)] =
        static_cast<int32_t>(kv_lens[static_cast<size_t>(t)] / cfg.compress_ratio);
  }
  std::vector<float> ones(static_cast<size_t>(T * H), 1.0f);
  std::vector<float> logits(static_cast<size_t>(T * nb_total), 0.0f);
  std::vector<float> qi = q_index;

  Tensor t_q = MakeT(qi.data(), DType::kF32, {T, H, D});
  Tensor t_w = MakeT(ones.data(), DType::kF32, {T, H});
  Tensor t_ws = MakeT(win_start.data(), DType::kI32, {T});
  Tensor t_we = MakeT(win_end.data(), DType::kI32, {T});
  Tensor t_lg = MakeT(logits.data(), DType::kF32, {T, nb_total});

  vt::DsaIndexerLogitsArgs largs;
  largs.softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));
  largs.n_head_scale = 1.0f;
  vt::DsaIndexerLogits(q, t_lg, t_q, t_bk, t_w, t_ws, t_we, largs);

  const int64_t topk = cfg.block_topk();
  out.block_ids.assign(static_cast<size_t>(T * topk), -1);
  out.counts.assign(static_cast<size_t>(T), 0);
  Tensor t_ids = MakeT(out.block_ids.data(), DType::kI32, {T, topk});
  Tensor t_cnt = MakeT(out.counts.data(), DType::kI32, {T});
  vt::DsaTopkSelect(q, t_ids, t_cnt, t_lg, t_ws, t_we);
  return out;
}

// The HOST reference's own expansion of a device selection, used only to derive
// the EXPECTED read count. It never touches the kernel's counter.
std::vector<int32_t> ExpandHost(const QsaConfig& cfg, const std::vector<int32_t>& block_ids,
                               int64_t topk, int64_t t, int64_t kv_len) {
  std::vector<int64_t> blocks;
  for (int64_t j = 0; j < topk; ++j) {
    const int32_t b = block_ids[static_cast<size_t>(t * topk + j)];
    if (b < 0) break;
    blocks.push_back(b);
  }
  return QsaSelectedTokenIndices(blocks, kv_len / cfg.compress_ratio, kv_len, cfg);
}

// A deterministic, portable value source for the real-config cases. `std::mt19937`
// would do, but a 12-line LCG keeps the fixture reproducible without depending on
// a standard-library distribution's unspecified consumption pattern.
struct Lcg {
  uint64_t s;
  explicit Lcg(uint64_t seed) : s(seed) {}
  float Next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>(static_cast<int32_t>(s >> 33)) / 2147483648.0f - 0.5f;
  }
};

#if VT_QSA_MPROTECT_PROBE
// ── The fetch-level probe's trampoline ──────────────────────────────────
//
// A kernel that reads an unmapped page raises SIGSEGV, and an unhandled SIGSEGV
// kills the whole binary: ctest then reports a signal with no case name, which
// is indistinguishable from a build that never ran and is the "broken
// instruments fail toward a code verdict" trap. So the fault is caught and
// turned into a FAILING ASSERTION that names the row it died on.
//
// `siglongjmp` out of the handler formally skips the destructors of every frame
// it unwinds. Those frames belong to the kernel and hold only `std::vector`s, so
// a body that faults leaks a few kilobytes on its way to a red. The SHIPPED
// kernel never takes this path -- it is the mutant's exit, and a mutant does not
// have to be leak-clean to be convicted.
sigjmp_buf g_qsa_probe_jmp;
volatile sig_atomic_t g_qsa_probe_armed = 0;

extern "C" void QsaProbeFaultHandler(int sig) {
  if (g_qsa_probe_armed != 0) {
    g_qsa_probe_armed = 0;
    siglongjmp(g_qsa_probe_jmp, 1);
  }
  // Not ours. Put the default action back and return, so the faulting
  // instruction re-executes and dies exactly as it would have without us --
  // swallowing an unrelated SIGSEGV would hide a real defect.
  ::signal(sig, SIG_DFL);
}

// The `sigsetjmp` lives in its OWN function so that no non-volatile local of the
// test case is modified between the setjmp and the longjmp; that is both the
// standard's rule and what keeps `-Wclobbered` quiet under `-Werror`.
//
// The handler is installed HERE and doctest's own is put back before returning.
// doctest installs a fatal-condition handler around every case, and left in
// place it turns the mutant's fault into `FATAL ERROR: test case CRASHED` and
// ABANDONS the rest of the binary -- every remaining case reads as skipped, so
// one convicted mutant costs the verdict on every other property in the file.
// Catching it here makes the fault a failing CHECK in one case and lets the
// suite finish.
bool GatherFaulted(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                   const Tensor& value, const Tensor& block_ids, const Tensor& kv_lens,
                   const Qwen4ExpQsaAttnArgs& args) {
  struct sigaction sa;
  struct sigaction old_segv;
  struct sigaction old_bus;
  std::memset(&sa, 0, sizeof(sa));
  std::memset(&old_segv, 0, sizeof(old_segv));
  std::memset(&old_bus, 0, sizeof(old_bus));
  sa.sa_handler = &QsaProbeFaultHandler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  if (::sigaction(SIGSEGV, &sa, &old_segv) != 0) return true;
  if (::sigaction(SIGBUS, &sa, &old_bus) != 0) {
    ::sigaction(SIGSEGV, &old_segv, nullptr);
    return true;
  }
  // VOLATILE on purpose: a non-volatile local written between `sigsetjmp` and
  // `siglongjmp` has an indeterminate value on the longjmp path.
  volatile bool faulted = true;
  if (sigsetjmp(g_qsa_probe_jmp, 1) == 0) {
    g_qsa_probe_armed = 1;
    vt::Qwen4ExpQsaGatherAttention(q, out, query, key, value, block_ids, kv_lens, args);
    g_qsa_probe_armed = 0;
    faulted = false;
  }
  g_qsa_probe_armed = 0;
  ::sigaction(SIGSEGV, &old_segv, nullptr);
  ::sigaction(SIGBUS, &old_bus, nullptr);
  return faulted;
}

// An anonymous page-aligned mapping, restored to readable before it is returned
// so that a `munmap` on a PROT_NONE range is never the thing under test.
struct GuardedCache {
  void* base = nullptr;
  size_t bytes = 0;
  GuardedCache() = default;
  GuardedCache(const GuardedCache&) = delete;
  GuardedCache& operator=(const GuardedCache&) = delete;
  ~GuardedCache() {
    if (base != nullptr) {
      ::mprotect(base, bytes, PROT_READ | PROT_WRITE);
      ::munmap(base, bytes);
    }
  }
  bool Map(size_t n) {
    void* p = ::mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return false;
    base = p;
    bytes = n;
    return true;
  }
  float* Data() { return static_cast<float*>(base); }
};
#endif  // VT_QSA_MPROTECT_PROBE

}  // namespace

// ── The compressor, against the lane-pinned oracle ───────────────────────────

TEST_CASE("vt::Qwen4ExpQsaCompress: pooled, normed, block-start-roped keys match the oracle") {
  Queue q = CpuQ();
  const QsaConfig cfg = GoldenConfig();
  const int64_t D = cfg.index_head_dim;
  for (const Case* c : {&kSubBudget, &kOverBudget}) {
    CAPTURE(std::string(c->name));  // doctest stringifies a char* as a bool
    const int64_t complete = (c->seq / cfg.compress_ratio) * cfg.compress_ratio;
    const int64_t nb = complete / cfg.compress_ratio;
    std::vector<float> raw = Slice(c->k_raw, complete * D);
    std::vector<float> knw = Slice(c->k_norm_w, D);
    std::vector<float> cos = Slice(c->cos, c->seq * cfg.rotary_dim);
    std::vector<float> sin = Slice(c->sin, c->seq * cfg.rotary_dim);
    std::vector<float> got(static_cast<size_t>(nb * D), 0.0f);

    Tensor t_raw = MakeT(raw.data(), DType::kF32, {complete, D});
    Tensor t_knw = MakeT(knw.data(), DType::kF32, {D});
    Tensor t_cos = MakeT(cos.data(), DType::kF32, {c->seq, cfg.rotary_dim});
    Tensor t_sin = MakeT(sin.data(), DType::kF32, {c->seq, cfg.rotary_dim});
    Tensor t_out = MakeT(got.data(), DType::kF32, {nb, D});

    Qwen4ExpQsaCompressArgs args;
    args.compress_ratio = cfg.compress_ratio;
    args.rotary_dim = cfg.rotary_dim;
    args.eps = cfg.rms_norm_eps;
    args.round_intermediates_to_bf16 = true;
    vt::Qwen4ExpQsaCompress(q, t_out, t_raw, t_knw, t_cos, t_sin, args);

    // The same bound the host arm is held to: the oracle's mean reduces in
    // torch's order and ours in ascending token order over compress_ratio = 4
    // terms; everything after the pool is reproduced operation for operation.
    CHECK(RelL2(got, c->block_keys, nb * D) < 1e-6);
  }
}

TEST_CASE("vt::Qwen4ExpQsaCompress: the pool is a MEAN over a NON-overlapping window") {
  // A HAND-DERIVED case, for the reason the host suite states and which applies
  // unchanged to the op: `k_layernorm` runs on the POOLED key, and RMSNorm is
  // scale-invariant whenever its epsilon is negligible against the mean square.
  // At the published eps = 1e-6, storing a SUM instead of a mean therefore
  // changes nothing downstream — and `/4` is exact in binary floating point, so
  // not even a bf16 round-trip catches it.
  //
  // So probe where the norm is NOT scale-invariant. With eps dominating the mean
  // square the norm is linear in its input and mean-versus-sum is a factor of
  // compress_ratio. Everything else is neutralised: cos = 1 and sin = 0 make the
  // rope the identity, the norm weight is zero so `(1.0 + w)` is 1, and the
  // bf16 rounding is off — which is what makes every expected value EXACT.
  Queue q = CpuQ();
  constexpr int64_t D = 4, CR = 2, ROT = 2;
  std::vector<float> raw = {
      2.0f, 4.0f, 6.0f,  8.0f,   // block 0, key 0
      6.0f, 8.0f, 10.0f, 12.0f,  // block 0, key 1
      0.0f, 2.0f, 4.0f,  4.0f,   // block 1, key 0
      2.0f, 4.0f, 6.0f,  6.0f};  // block 1, key 1
  std::vector<float> w(D, 0.0f);
  std::vector<float> cos(4 * ROT, 1.0f);
  std::vector<float> sin(4 * ROT, 0.0f);
  std::vector<float> got(2 * D, 0.0f);

  Tensor t_raw = MakeT(raw.data(), DType::kF32, {4, D});
  Tensor t_w = MakeT(w.data(), DType::kF32, {D});
  Tensor t_cos = MakeT(cos.data(), DType::kF32, {4, ROT});
  Tensor t_sin = MakeT(sin.data(), DType::kF32, {4, ROT});
  Tensor t_out = MakeT(got.data(), DType::kF32, {2, D});

  Qwen4ExpQsaCompressArgs args;
  args.compress_ratio = CR;
  args.rotary_dim = ROT;
  args.eps = 10.0f;
  args.round_intermediates_to_bf16 = false;
  vt::Qwen4ExpQsaCompress(q, t_out, t_raw, t_w, t_cos, t_sin, args);

  // Block 0 MEAN-pools to [4, 6, 8, 10]: mean square 54, + eps = 64, so the norm
  // divides by exactly 8. Block 1 pools to [1, 3, 5, 5]: mean square 15, + eps =
  // 25, so it divides by exactly 5. An unnormalised SUM would pool block 0 to
  // [8, 12, 16, 20], mean square 216, + eps = 226, and emit 8/sqrt(226) =
  // 0.5322... where 0.5 is asserted.
  const std::vector<float> want = {0.5f, 0.75f, 1.0f, 1.25f, 0.2f, 0.6f, 1.0f, 1.0f};
  REQUIRE(got.size() == want.size());
  for (size_t i = 0; i < got.size(); ++i) CHECK(got[i] == want[i]);
}

TEST_CASE("vt::Qwen4ExpQsaCompress: the rope position is the block's FIRST token") {
  // The block-start position is invisible to every golden comparison that also
  // varies the keys, because a one-block phase error still produces a plausible
  // vector. Here the two blocks carry the SAME pooled key and the cos/sin table
  // differs only between row 0 and row 2, so the output can only distinguish
  // `position = CR*b` from `position = CR*b + CR - 1`.
  Queue q = CpuQ();
  constexpr int64_t D = 2, CR = 2, ROT = 2;
  std::vector<float> raw = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
  std::vector<float> w(D, 0.0f);
  // Rows 0 and 2 are the BLOCK-START positions; rows 1 and 3 are the block ends.
  // cos/sin at row 0 is the identity rotation, at row 1 a quarter turn, at row 2
  // a half turn, at row 3 a three-quarter turn.
  std::vector<float> cos = {1.0f, 1.0f, 0.0f, 0.0f, -1.0f, -1.0f, 0.0f, 0.0f};
  std::vector<float> sin = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, -1.0f, -1.0f};
  std::vector<float> got(2 * D, 0.0f);

  Tensor t_raw = MakeT(raw.data(), DType::kF32, {4, D});
  Tensor t_w = MakeT(w.data(), DType::kF32, {D});
  Tensor t_cos = MakeT(cos.data(), DType::kF32, {4, ROT});
  Tensor t_sin = MakeT(sin.data(), DType::kF32, {4, ROT});
  Tensor t_out = MakeT(got.data(), DType::kF32, {2, D});

  Qwen4ExpQsaCompressArgs args;
  args.compress_ratio = CR;
  args.rotary_dim = ROT;
  args.eps = 1e-6f;
  args.round_intermediates_to_bf16 = false;
  vt::Qwen4ExpQsaCompress(q, t_out, t_raw, t_w, t_cos, t_sin, args);

  // Pooled key is [1, 0] for both blocks; k_layernorm with a zero weight and a
  // negligible eps normalises it to [sqrt(2), 0]. `rotate_half` over a
  // rotary_dim of 2 pairs (d0, d1) as (-x1, x0), so the rotation is
  // (x0*cos0 - x1*sin0, x1*cos1 + x0*sin1) = (sqrt(2)*cos, sqrt(2)*sin).
  // Block 0 rotates at row 0 — cos 1, sin 0, the identity — and block 1 at
  // row 2 — cos -1, sin 0, a sign flip.
  const float n = std::sqrt(2.0f);
  REQUIRE(got.size() == 4);
  CHECK(got[0] == doctest::Approx(n).epsilon(1e-5));
  CHECK(got[1] == doctest::Approx(0.0f).epsilon(1e-5));
  CHECK(got[2] == doctest::Approx(-n).epsilon(1e-5));
  CHECK(got[3] == doctest::Approx(0.0f).epsilon(1e-5));
  // Had the rope read the block's LAST position (rows 1 and 3, a quarter and a
  // three-quarter turn) the answer would be (0, n) and (0, -n) — the two
  // components swapped. The expectation is therefore not merely tight, it is
  // ORTHOGONAL to the defect it exists to catch.
  CHECK(std::abs(got[0]) > std::abs(got[1]));
  CHECK(std::abs(got[2]) > std::abs(got[3]));
}

// ── The composed indexer, against the oracle's own selected sets ─────────────

TEST_CASE("qsa-device-indexer: DsaIndexerLogits + DsaTopkSelect reproduce the oracle") {
  // THE REUSE CLAIM, MEASURED. If the two existing DSA ops were not the QSA
  // score and the QSA top-k, this case is where that shows: the selected token
  // sets below are the transformers 5.16.0 output, ragged tail included, for
  // every query token of both fixtures.
  Queue q = CpuQ();
  const QsaConfig cfg = GoldenConfig();
  const int64_t D = cfg.index_head_dim;
  const int64_t topk = cfg.block_topk();
  for (const Case* c : {&kSubBudget, &kOverBudget}) {
    CAPTURE(std::string(c->name));  // doctest stringifies a char* as a bool
    std::vector<int32_t> kv_lens;
    for (int64_t t = 0; t < c->seq; ++t) kv_lens.push_back(static_cast<int32_t>(t + 1));
    const int64_t complete = (c->seq / cfg.compress_ratio) * cfg.compress_ratio;
    const Indexed idx =
        RunIndexer(q, cfg, Slice(c->k_raw, complete * D), Slice(c->k_norm_w, D),
                   Slice(c->cos, c->seq * cfg.rotary_dim),
                   Slice(c->sin, c->seq * cfg.rotary_dim), GoldenQ(*c, cfg), kv_lens,
                   /*round_bf16=*/true);
    for (int64_t t = 0; t < c->seq; ++t) {
      CAPTURE(t);
      const std::vector<int32_t> got = ExpandHost(cfg, idx.block_ids, topk, t, t + 1);
      REQUIRE(static_cast<int64_t>(got.size()) == g::kIndexWidth);
      for (int64_t j = 0; j < g::kIndexWidth; ++j) {
        CHECK(got[j] == c->selected[t * g::kIndexWidth + j]);
      }
    }
  }
}

// ── The gather consumer ──────────────────────────────────────────────────────

TEST_CASE("vt::Qwen4ExpQsaGatherAttention: reproduces the oracle's masked attention") {
  Queue q = CpuQ();
  const QsaConfig cfg = GoldenConfig();
  const int64_t D = cfg.index_head_dim, topk = cfg.block_topk();
  const int64_t HQ = g::kNumAttentionHeads, HKV = g::kNumKeyValueHeads, DH = g::kHeadDim;
  for (const Case* c : {&kSubBudget, &kOverBudget}) {
    CAPTURE(std::string(c->name));  // doctest stringifies a char* as a bool
    std::vector<int32_t> kv_lens;
    for (int64_t t = 0; t < c->seq; ++t) kv_lens.push_back(static_cast<int32_t>(t + 1));
    const int64_t complete = (c->seq / cfg.compress_ratio) * cfg.compress_ratio;
    Indexed idx = RunIndexer(q, cfg, Slice(c->k_raw, complete * D), Slice(c->k_norm_w, D),
                             Slice(c->cos, c->seq * cfg.rotary_dim),
                             Slice(c->sin, c->seq * cfg.rotary_dim), GoldenQ(*c, cfg),
                             kv_lens, /*round_bf16=*/true);

    // THE BATCH AXIS IS NEW. The host reference is one query token per call; the
    // whole sequence goes through the op in ONE call, so a kernel that computed
    // token 0 and broadcast it, or walked `block_ids` with the wrong row stride,
    // fails here and could not fail there.
    std::vector<float> qa = Slice(c->attn_q, c->seq * HQ * DH);
    std::vector<float> ka = Slice(c->attn_k, c->seq * HKV * DH);
    std::vector<float> va = Slice(c->attn_v, c->seq * HKV * DH);
    std::vector<float> out(static_cast<size_t>(c->seq * HQ * DH), 0.0f);

    Tensor t_q = MakeT(qa.data(), DType::kF32, {c->seq, HQ, DH});
    Tensor t_k = MakeT(ka.data(), DType::kF32, {c->seq, HKV, DH});
    Tensor t_v = MakeT(va.data(), DType::kF32, {c->seq, HKV, DH});
    Tensor t_ids = MakeT(idx.block_ids.data(), DType::kI32, {c->seq, topk});
    Tensor t_len = MakeT(idx.kv_lens.data(), DType::kI32, {c->seq});
    Tensor t_out = MakeT(out.data(), DType::kF32, {c->seq, HQ, DH});

    Qwen4ExpQsaAttnArgs args;
    args.scale = 1.0f / std::sqrt(static_cast<float>(DH));
    args.compress_ratio = cfg.compress_ratio;
    vt::Qwen4ExpQsaGatherAttention(q, t_out, t_q, t_k, t_v, t_ids, t_len, args);

    // The oracle reduces in torch's order over the padded row; we reduce over
    // the gathered subset. Same values, different summation order.
    CHECK(RelL2(out, c->attn_out, c->seq * HQ * DH) < 2e-3);
  }
}

TEST_CASE("vt::Qwen4ExpQsaGatherAttention: a sub-budget gather is BIT-IDENTICAL to dense") {
  // llama.cpp #27742 measures a max logit delta of 0.0 over all 2051 sub-budget
  // rows. This is that claim on the op: with every candidate selected the gather
  // reduces over exactly the dense sequence, in exactly the dense order. `want`
  // comes from the HOST masked reference over the full causal prefix — an
  // independent walk — and not from the op under test, because comparing the op
  // to a second call to itself only says it is deterministic.
  Queue q = CpuQ();
  const QsaConfig cfg = GoldenConfig();
  const Case& c = kSubBudget;
  const int64_t D = cfg.index_head_dim, topk = cfg.block_topk();
  const int64_t HQ = g::kNumAttentionHeads, HKV = g::kNumKeyValueHeads, DH = g::kHeadDim;
  REQUIRE(c.seq == cfg.index_width());  // the largest all-select context

  std::vector<int32_t> kv_lens;
  for (int64_t t = 0; t < c.seq; ++t) kv_lens.push_back(static_cast<int32_t>(t + 1));
  const int64_t complete = (c.seq / cfg.compress_ratio) * cfg.compress_ratio;
  Indexed idx = RunIndexer(q, cfg, Slice(c.k_raw, complete * D), Slice(c.k_norm_w, D),
                           Slice(c.cos, c.seq * cfg.rotary_dim),
                           Slice(c.sin, c.seq * cfg.rotary_dim), GoldenQ(c, cfg), kv_lens,
                           /*round_bf16=*/true);

  std::vector<float> qa = Slice(c.attn_q, c.seq * HQ * DH);
  std::vector<float> ka = Slice(c.attn_k, c.seq * HKV * DH);
  std::vector<float> va = Slice(c.attn_v, c.seq * HKV * DH);
  std::vector<float> out(static_cast<size_t>(c.seq * HQ * DH), 0.0f);

  Tensor t_q = MakeT(qa.data(), DType::kF32, {c.seq, HQ, DH});
  Tensor t_k = MakeT(ka.data(), DType::kF32, {c.seq, HKV, DH});
  Tensor t_v = MakeT(va.data(), DType::kF32, {c.seq, HKV, DH});
  Tensor t_ids = MakeT(idx.block_ids.data(), DType::kI32, {c.seq, topk});
  Tensor t_len = MakeT(idx.kv_lens.data(), DType::kI32, {c.seq});
  Tensor t_out = MakeT(out.data(), DType::kF32, {c.seq, HQ, DH});

  Qwen4ExpQsaAttnArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(DH));
  args.compress_ratio = cfg.compress_ratio;
  vt::Qwen4ExpQsaGatherAttention(q, t_out, t_q, t_k, t_v, t_ids, t_len, args);

  for (int64_t t = 0; t < c.seq; ++t) {
    CAPTURE(t);
    const int64_t kv_len = t + 1;
    const std::vector<float> q_row(qa.begin() + t * HQ * DH, qa.begin() + (t + 1) * HQ * DH);
    std::vector<int32_t> dense(cfg.index_width(), -1);
    for (int64_t j = 0; j < kv_len; ++j) dense[j] = static_cast<int32_t>(j);
    const std::vector<float> want =
        QsaMaskedAttention(q_row, ka, va, dense, kv_len, HQ, HKV, DH, nullptr);
    for (int64_t i = 0; i < HQ * DH; ++i) CHECK(out[t * HQ * DH + i] == want[i]);
  }
}

TEST_CASE("vt::Qwen4ExpQsaGatherAttention: the GATHER reads only the selected rows") {
  // The wave's whole point, and NOT a correctness property. A mask over a dense
  // cache is correct and passes every case above. `want` is derived from the
  // HOST reference's expansion of the device selection; `keys_visited` is
  // incremented inside the kernel at the key-row read. Two quantities, two
  // derivations. Assigning the counter from the selection instead — which is
  // what W4's first revision did — makes the two sides the same number computed
  // the same way, and a body doing the full dense work passes.
  Queue q = CpuQ();
  const QsaConfig cfg = GoldenConfig();
  const Case& c = kOverBudget;
  const int64_t D = cfg.index_head_dim, topk = cfg.block_topk();
  const int64_t HQ = g::kNumAttentionHeads, HKV = g::kNumKeyValueHeads, DH = g::kHeadDim;
  // Two softmax passes, each of which reads every row it visits, so a redundant
  // third pass is visible here rather than free. A single-pass online-softmax
  // rewrite would legitimately halve this and is where that gets re-derived.
  constexpr int64_t kReadsPerRowPerHead = 2;

  std::vector<int32_t> kv_lens;
  for (int64_t t = 0; t < c.seq; ++t) kv_lens.push_back(static_cast<int32_t>(t + 1));
  const int64_t complete = (c.seq / cfg.compress_ratio) * cfg.compress_ratio;
  Indexed idx = RunIndexer(q, cfg, Slice(c.k_raw, complete * D), Slice(c.k_norm_w, D),
                           Slice(c.cos, c.seq * cfg.rotary_dim),
                           Slice(c.sin, c.seq * cfg.rotary_dim), GoldenQ(c, cfg), kv_lens,
                           /*round_bf16=*/true);

  std::vector<float> qa = Slice(c.attn_q, c.seq * HQ * DH);
  std::vector<float> ka = Slice(c.attn_k, c.seq * HKV * DH);
  std::vector<float> va = Slice(c.attn_v, c.seq * HKV * DH);
  std::vector<float> out(static_cast<size_t>(c.seq * HQ * DH), 0.0f);

  int64_t want = 0, dense = 0, strictly_sparse_queries = 0;
  for (int64_t t = 0; t < c.seq; ++t) {
    const std::vector<int32_t> sel = ExpandHost(cfg, idx.block_ids, topk, t, t + 1);
    const int64_t sel_reads = SelectedCount(sel) * HQ * kReadsPerRowPerHead;
    const int64_t dense_reads = (t + 1) * HQ * kReadsPerRowPerHead;
    want += sel_reads;
    dense += dense_reads;
    if (sel_reads < dense_reads) ++strictly_sparse_queries;
  }

  Tensor t_q = MakeT(qa.data(), DType::kF32, {c.seq, HQ, DH});
  Tensor t_k = MakeT(ka.data(), DType::kF32, {c.seq, HKV, DH});
  Tensor t_v = MakeT(va.data(), DType::kF32, {c.seq, HKV, DH});
  Tensor t_ids = MakeT(idx.block_ids.data(), DType::kI32, {c.seq, topk});
  Tensor t_len = MakeT(idx.kv_lens.data(), DType::kI32, {c.seq});
  Tensor t_out = MakeT(out.data(), DType::kF32, {c.seq, HQ, DH});

  int64_t visited = -1;
  Qwen4ExpQsaAttnArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(DH));
  args.compress_ratio = cfg.compress_ratio;
  args.keys_visited = &visited;
  vt::Qwen4ExpQsaGatherAttention(q, t_out, t_q, t_k, t_v, t_ids, t_len, args);

  INFO("keys_visited ", visited, " selected-derived ", want, " dense ", dense);
  CHECK(visited == want);
  CHECK(visited < dense);
  // Above the budget the selection MUST discard blocks. A fixture that never
  // crossed it would leave the assertion above trivially true.
  CHECK(strictly_sparse_queries > 0);
}

TEST_CASE("vt::Qwen4ExpQsaGatherAttention: the unselected rows are NaN and the answer is finite") {
  // THE REPAIR FOR A SURVIVING MUTATION, and the strongest instrument in this
  // file. `keys_visited` is SELF-REPORTED: mutation M11 replaced the gather's
  // body with a dense masked walk over every one of the `kv_len` cached rows AND
  // set the counter from the selection, and it passed every other case here —
  // 10/10, 4167 assertions — exactly as W4's M22c passed 12 cases / 7251. A
  // counter a kernel writes cannot convict the kernel that writes it, and no
  // value comparison can either, because a mask that masks correctly agrees with
  // a gather value for value: `exp(-inf - m)` is exactly +0.
  //
  // What a mask CANNOT survive is a cache whose unselected rows are not numbers.
  // A gather never addresses them. A mask reads every value row and accumulates
  // `w * v` with `w == 0.0f`, and `0.0f * NaN` is NaN in IEEE-754 — so the
  // moment an unselected row is poisoned, a mask's output is NaN and a gather's
  // is unchanged. This is an OBSERVABLE OF THE WALK, not of the kernel's
  // bookkeeping, and it is what makes the property checkable rather than
  // asserted. Its honest limit is that it proves a row was never MULTIPLIED,
  // not that it was never FETCHED — a body that loads every row and throws the
  // unselected ones away passes this case. The case that convicts THAT is the
  // unmapped-tail probe below, which needs no paged production store because a
  // test may `mmap` its own cache.
  //
  // ONE query token, deliberately: the poison set is that token's complement,
  // and two tokens with different selections have no common complement to
  // poison.
  Queue q = CpuQ();
  const QsaConfig cfg = GoldenConfig();
  const Case& c = kOverBudget;
  const int64_t D = cfg.index_head_dim, topk = cfg.block_topk();
  const int64_t HQ = g::kNumAttentionHeads, HKV = g::kNumKeyValueHeads, DH = g::kHeadDim;
  const int64_t qi = c.seq - 1;         // the longest prefix in the fixture
  const int64_t kv = qi + 1;            // 23 cached tokens
  const int64_t complete = (c.seq / cfg.compress_ratio) * cfg.compress_ratio;

  const std::vector<int32_t> kv_lens(1, static_cast<int32_t>(kv));
  // ONE materialisation, then two iterators into it. Two calls to GoldenQ would
  // build two temporaries and the range would span different objects.
  const std::vector<float> q_all = GoldenQ(c, cfg);
  const std::vector<float> q_one(q_all.begin() + qi * cfg.index_n_heads * D,
                                 q_all.begin() + (qi + 1) * cfg.index_n_heads * D);
  Indexed idx = RunIndexer(q, cfg, Slice(c.k_raw, complete * D), Slice(c.k_norm_w, D),
                           Slice(c.cos, c.seq * cfg.rotary_dim),
                           Slice(c.sin, c.seq * cfg.rotary_dim), q_one, kv_lens,
                           /*round_bf16=*/true);

  const std::vector<int32_t> sel = ExpandHost(cfg, idx.block_ids, topk, 0, kv);
  const int64_t n_sel = SelectedCount(sel);
  // A poison set that is empty would make this case vacuous, which is the
  // "a gate that never fired is not a gate" failure. 23 cached, 11 attended.
  REQUIRE(n_sel < kv);

  const std::vector<float> qa_all = Slice(c.attn_q, c.seq * HQ * DH);
  std::vector<float> qa(qa_all.begin() + qi * HQ * DH, qa_all.begin() + (qi + 1) * HQ * DH);
  const std::vector<float> ka_clean = Slice(c.attn_k, c.seq * HKV * DH);
  const std::vector<float> va_clean = Slice(c.attn_v, c.seq * HKV * DH);

  std::vector<bool> keep(static_cast<size_t>(kv), false);
  for (int64_t j = 0; j < n_sel; ++j) keep[static_cast<size_t>(sel[static_cast<size_t>(j)])] = true;
  std::vector<float> ka = ka_clean, va = va_clean;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  int64_t poisoned = 0;
  for (int64_t p = 0; p < kv; ++p) {
    if (keep[static_cast<size_t>(p)]) continue;
    ++poisoned;
    for (int64_t h = 0; h < HKV; ++h) {
      for (int64_t d = 0; d < DH; ++d) {
        ka[static_cast<size_t>((p * HKV + h) * DH + d)] = nan;
        va[static_cast<size_t>((p * HKV + h) * DH + d)] = nan;
      }
    }
  }
  CHECK(poisoned == kv - n_sel);
  REQUIRE(poisoned > 0);

  std::vector<float> out(static_cast<size_t>(HQ * DH), 0.0f);
  std::vector<int32_t> ids = idx.block_ids, lens = idx.kv_lens;
  Tensor t_q = MakeT(qa.data(), DType::kF32, {1, HQ, DH});
  Tensor t_k = MakeT(ka.data(), DType::kF32, {c.seq, HKV, DH});
  Tensor t_v = MakeT(va.data(), DType::kF32, {c.seq, HKV, DH});
  Tensor t_ids = MakeT(ids.data(), DType::kI32, {1, topk});
  Tensor t_len = MakeT(lens.data(), DType::kI32, {1});
  Tensor t_out = MakeT(out.data(), DType::kF32, {1, HQ, DH});
  Qwen4ExpQsaAttnArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(DH));
  args.compress_ratio = cfg.compress_ratio;
  vt::Qwen4ExpQsaGatherAttention(q, t_out, t_q, t_k, t_v, t_ids, t_len, args);

  // Every output finite, AND bit-identical to the same gather over a CLEAN
  // cache. Finiteness alone would pass an implementation that read the poison
  // and then discarded the whole row; the equality says the poison never
  // entered the arithmetic at all.
  const std::vector<float> want =
      QsaGatherAttention(qa, ka_clean, va_clean, sel, kv, HQ, HKV, DH, nullptr);
  REQUIRE(want.size() == out.size());
  for (int64_t i = 0; i < HQ * DH; ++i) {
    CAPTURE(i);
    CHECK(std::isfinite(out[static_cast<size_t>(i)]));
    CHECK(out[static_cast<size_t>(i)] == want[static_cast<size_t>(i)]);
  }
}

// ── The same property, one layer deeper: the unselected tail is UNMAPPED ─────

#if VT_QSA_MPROTECT_PROBE
TEST_CASE("vt::Qwen4ExpQsaGatherAttention: the gather never FETCHES an unmapped unselected row") {
  // WHAT THIS ADDS OVER THE NaN CASE ABOVE, which is the whole reason it exists.
  // A NaN poison proves an unselected row was never MULTIPLIED into an
  // accumulator. It does not prove the row's bytes were never READ: a body that
  // loads every row and discards the unselected ones before the multiply passes
  // it untouched, and the key-row TRAFFIC is the cost llama.cpp #27739 measures,
  // not the multiply. The instrument that convicts a fetch is a cache whose
  // unselected pages are not readable at all, so a read of one is a SIGSEGV.
  //
  // AND IT DOES NOT NEED THE PAGED STORE. The structural version of this in
  // PRODUCTION is a block-table cache whose unselected blocks are simply not
  // mapped, and that store is genuinely owed and genuinely blocked. A TEST is
  // under no such constraint: it builds the cache itself, and `mmap` plus
  // `mprotect(PROT_NONE)` gives it the page-granular hole for free.
  //
  // THE SHAPE IS FORCED BY THE KERNEL. The gather addresses the cache as
  // `(p * HKV + kvh) * DH + d` and never reads `key.stride[0]`, so a guard page
  // BETWEEN rows is not available -- the unselected rows have to form ONE
  // contiguous run. Selecting blocks 0..511 and running at kv_len 3000, a
  // multiple of `compress_ratio` so the always-attended ragged tail is empty,
  // puts every unselected row in `[2048, 3000)` with nothing of interest after
  // it.
  Queue q = CpuQ();
  QsaConfig cfg;  // the released Qwen3.8-Flash-Next indexer values
  REQUIRE(cfg.token_budget == 2048);
  REQUIRE(cfg.block_topk() == 512);
  constexpr int64_t T = 1, HQ = 4, HKV = 2, DH = 32;
  const int64_t topk = cfg.block_topk();
  const int64_t kv = 3000;
  REQUIRE(kv % cfg.compress_ratio == 0);  // an empty ragged tail
  const int64_t complete = kv / cfg.compress_ratio;
  REQUIRE(topk < complete);  // genuinely sparse: 512 of 750 blocks

  const size_t row_floats = static_cast<size_t>(HKV * DH);
  const size_t cache_bytes = static_cast<size_t>(kv) * row_floats * sizeof(float);
  GuardedCache kmap, vmap;
  REQUIRE(kmap.Map(cache_bytes));
  REQUIRE(vmap.Map(cache_bytes));

  Lcg rng(0xfe7c40ded1234ULL);
  for (size_t i = 0; i < cache_bytes / sizeof(float); ++i) kmap.Data()[i] = rng.Next();
  for (size_t i = 0; i < cache_bytes / sizeof(float); ++i) vmap.Data()[i] = rng.Next();
  std::vector<float> qa(static_cast<size_t>(T * HQ * DH));
  for (float& v : qa) v = rng.Next();

  // The selection is handed in directly rather than scored: this case is about
  // the gather's FETCH, and an indexer run would only add a second thing that
  // could be wrong. Ascending and inside the complete blocks, as the op demands.
  std::vector<int32_t> ids(static_cast<size_t>(T * topk), -1);
  for (int64_t j = 0; j < topk; ++j) ids[static_cast<size_t>(j)] = static_cast<int32_t>(j);
  std::vector<int32_t> lens(static_cast<size_t>(T), static_cast<int32_t>(kv));

  Tensor t_q = MakeT(qa.data(), DType::kF32, {T, HQ, DH});
  Tensor t_k = MakeT(kmap.base, DType::kF32, {kv, HKV, DH});
  Tensor t_v = MakeT(vmap.base, DType::kF32, {kv, HKV, DH});
  Tensor t_ids = MakeT(ids.data(), DType::kI32, {T, topk});
  Tensor t_len = MakeT(lens.data(), DType::kI32, {T});

  Qwen4ExpQsaAttnArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(DH));
  args.compress_ratio = cfg.compress_ratio;

  // Pass 1: the cache is fully readable. This is the control the guarded run is
  // compared against, and it is taken FIRST because reading it afterwards would
  // mean unprotecting the pages the case exists to keep unreadable.
  std::vector<float> out_open(static_cast<size_t>(T * HQ * DH), 0.0f);
  Tensor t_out_open = MakeT(out_open.data(), DType::kF32, {T, HQ, DH});
  int64_t visited_open = -1;
  args.keys_visited = &visited_open;
  vt::Qwen4ExpQsaGatherAttention(q, t_out_open, t_q, t_k, t_v, t_ids, t_len, args);

  // The whole pages that lie STRICTLY INSIDE the unselected tail. The partial
  // pages at each end stay readable, because a page is the granularity mprotect
  // has and taking a page that holds a selected row would fault the honest
  // kernel.
  const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
  REQUIRE(page > 0);
  const size_t tail_lo =
      static_cast<size_t>(topk * cfg.compress_ratio) * row_floats * sizeof(float);
  const size_t guard_lo = ((tail_lo + page - 1) / page) * page;
  const size_t guard_hi = (cache_bytes / page) * page;
  // A guard of zero pages would make this case vacuous -- the "a gate that never
  // fired is not a gate" failure -- so it is REQUIREd, not assumed.
  REQUIRE(guard_hi > guard_lo);
  const size_t guard_bytes = guard_hi - guard_lo;
  INFO("guarding ", guard_bytes, " bytes (", guard_bytes / page, " pages) of the unselected tail [",
       tail_lo, ", ", cache_bytes, ") in BOTH the key and the value cache");
  REQUIRE(::mprotect(static_cast<char*>(kmap.base) + guard_lo, guard_bytes, PROT_NONE) == 0);
  REQUIRE(::mprotect(static_cast<char*>(vmap.base) + guard_lo, guard_bytes, PROT_NONE) == 0);

  // Pass 2: the same call over the same cache with the tail taken away. A
  // gather walks past it. A dense masked walk -- mutation M11's body, which the
  // NaN case reds and which every read-count assertion in this file passes --
  // dereferences the first guarded row and dies.
  std::vector<float> out_guarded(static_cast<size_t>(T * HQ * DH), 0.0f);
  Tensor t_out_guarded = MakeT(out_guarded.data(), DType::kF32, {T, HQ, DH});
  int64_t visited_guarded = -1;
  args.keys_visited = &visited_guarded;
  const bool faulted = GatherFaulted(q, t_out_guarded, t_q, t_k, t_v, t_ids, t_len, args);

  INFO("faulted ", faulted, " keys_visited ", visited_guarded);
  CHECK_FALSE(faulted);
  if (faulted) return;  // the tensors below hold nothing a fault left behind

  // The counts, so that a body which survived by reading NOTHING is not mistaken
  // for one that gathered. 512 blocks * 4 rows * 4 query heads * 2 softmax
  // passes = 16384, against a dense 3000 * 4 * 2 = 24000.
  // Re-derived here rather than shared with the case above: two softmax passes,
  // each of which reads every row it visits. A single-pass online-softmax
  // rewrite halves it, and that is owed work, not a free change.
  constexpr int64_t kReadsPerRowPerHead = 2;
  const int64_t want_reads = topk * cfg.compress_ratio * HQ * kReadsPerRowPerHead;
  const int64_t dense_reads = kv * HQ * kReadsPerRowPerHead;
  CHECK(visited_guarded == want_reads);
  CHECK(visited_guarded == visited_open);
  CHECK(visited_guarded < dense_reads);

  // And bit-identical to the unguarded run, which says the walk was the same one
  // and not a truncated version of it that stopped at the hole.
  for (int64_t i = 0; i < T * HQ * DH; ++i) {
    CAPTURE(i);
    CHECK(std::isfinite(out_guarded[static_cast<size_t>(i)]));
    CHECK(out_guarded[static_cast<size_t>(i)] == out_open[static_cast<size_t>(i)]);
  }
}
#else
TEST_CASE("vt::Qwen4ExpQsaGatherAttention: the gather never FETCHES an unmapped unselected row" *
          doctest::skip()) {
  // No POSIX `mmap`/`mprotect` on this platform. The NaN case above still runs
  // and is portable; this one reports SKIPPED rather than failing, because a
  // probe that cannot be built here proves nothing either way.
}
#endif  // VT_QSA_MPROTECT_PROBE

// ── The RELEASED indexer config, past 2048 tokens of context ─────────────────

TEST_CASE("qsa-device: the released config at 3002 tokens of context is genuinely sparse") {
  // THE GATE THE SPEC DEMANDS. At or below `indexer_budget` = 2048 every
  // candidate is selected, so a short-prompt gate cannot distinguish a correct
  // port from one attending POOLED keys. This case runs the RELEASED indexer
  // shape — index_n_heads 4, index_head_dim 128, token_budget 2048,
  // compress_ratio 4, rotary_dim 64 — at kv_len 3002: 750 complete blocks of
  // which 512 are selected, plus a 2-token ragged tail, so 2050 of 3002 rows are
  // attended and 952 are discarded. The 2051-token control below is the
  // measurement of why the requirement exists: there, nothing is discarded and
  // every assertion here would pass for a mask.
  //
  // The attention head shape is deliberately smaller than the released one
  // (4 query heads over 2 KV heads at head_dim 32, not 24 over 2 at 256): the
  // regime this case exists to reach is the SELECTION regime, which the indexer
  // config governs and the attention head shape does not, and the released head
  // shape would multiply the runtime by 48 to say the same thing.
  Queue q = CpuQ();
  QsaConfig cfg;  // the published Qwen3.8-Flash-Next indexer values
  REQUIRE(cfg.token_budget == 2048);
  REQUIRE(cfg.block_topk() == 512);
  const int64_t D = cfg.index_head_dim, topk = cfg.block_topk();
  constexpr int64_t HQ = 4, HKV = 2, DH = 32;

  struct Ctx {
    const char* name;
    int64_t kv_len;
    bool expect_sparse;
  };
  // 2051 == token_budget + compress_ratio - 1, the largest all-select context.
  const Ctx contexts[] = {{"past-budget", 3002, true}, {"sub-budget-control", 2051, false}};

  for (const Ctx& ctx : contexts) {
    CAPTURE(std::string(ctx.name));  // doctest stringifies a char* as a bool
    const int64_t kv = ctx.kv_len;
    const int64_t complete = (kv / cfg.compress_ratio) * cfg.compress_ratio;

    Lcg rng(0x5150c0ffee1234ULL ^ static_cast<uint64_t>(kv));
    std::vector<float> raw(static_cast<size_t>(complete * D));
    for (float& v : raw) v = rng.Next();
    std::vector<float> knw(static_cast<size_t>(D));
    for (float& v : knw) v = 0.1f * rng.Next();
    std::vector<float> cos(static_cast<size_t>(kv * cfg.rotary_dim));
    std::vector<float> sin(cos.size());
    for (int64_t p = 0; p < kv; ++p) {
      for (int64_t d = 0; d < cfg.rotary_dim; ++d) {
        const double theta = static_cast<double>(p) /
                             std::pow(10000.0, 2.0 * static_cast<double>(d % (cfg.rotary_dim / 2)) /
                                                   static_cast<double>(cfg.rotary_dim));
        cos[static_cast<size_t>(p * cfg.rotary_dim + d)] = static_cast<float>(std::cos(theta));
        sin[static_cast<size_t>(p * cfg.rotary_dim + d)] = static_cast<float>(std::sin(theta));
      }
    }
    // Three query tokens, all at the SAME kv_len: this is a decode step's shape,
    // and it keeps the case's cost linear in the thing being measured.
    constexpr int64_t T = 3;
    std::vector<float> qi(static_cast<size_t>(T * cfg.index_n_heads * D));
    for (float& v : qi) v = rng.Next();
    const std::vector<int32_t> kv_lens(T, static_cast<int32_t>(kv));

    const Indexed idx =
        RunIndexer(q, cfg, raw, knw, cos, sin, qi, kv_lens, /*round_bf16=*/false);

    std::vector<float> qa(static_cast<size_t>(T * HQ * DH));
    for (float& v : qa) v = rng.Next();
    std::vector<float> ka(static_cast<size_t>(kv * HKV * DH));
    for (float& v : ka) v = rng.Next();
    std::vector<float> va(static_cast<size_t>(kv * HKV * DH));
    for (float& v : va) v = rng.Next();
    std::vector<float> out(static_cast<size_t>(T * HQ * DH), 0.0f);

    Tensor t_q = MakeT(qa.data(), DType::kF32, {T, HQ, DH});
    Tensor t_k = MakeT(ka.data(), DType::kF32, {kv, HKV, DH});
    Tensor t_v = MakeT(va.data(), DType::kF32, {kv, HKV, DH});
    std::vector<int32_t> ids = idx.block_ids;
    std::vector<int32_t> lens = idx.kv_lens;
    Tensor t_ids = MakeT(ids.data(), DType::kI32, {T, topk});
    Tensor t_len = MakeT(lens.data(), DType::kI32, {T});
    Tensor t_out = MakeT(out.data(), DType::kF32, {T, HQ, DH});

    int64_t visited = -1;
    Qwen4ExpQsaAttnArgs args;
    args.scale = 1.0f / std::sqrt(static_cast<float>(DH));
    args.compress_ratio = cfg.compress_ratio;
    args.keys_visited = &visited;
    vt::Qwen4ExpQsaGatherAttention(q, t_out, t_q, t_k, t_v, t_ids, t_len, args);

    // THE SPARSITY, measured at the read against a selection-derived expectation.
    int64_t want_reads = 0, selected_total = 0;
    for (int64_t t = 0; t < T; ++t) {
      const std::vector<int32_t> sel = ExpandHost(cfg, ids, topk, t, kv);
      selected_total += SelectedCount(sel);
      want_reads += SelectedCount(sel) * HQ * 2;
    }
    const int64_t dense_reads = T * kv * HQ * 2;
    INFO("kv_len ", kv, " selected ", selected_total, " of ", T * kv, " keys_visited ", visited,
         " dense ", dense_reads);
    CHECK(visited == want_reads);
    if (ctx.expect_sparse) {
      // 2050 attended of 3002 cached, per query token.
      CHECK(selected_total == T * (cfg.block_topk() * cfg.compress_ratio + kv % cfg.compress_ratio));
      CHECK(visited < dense_reads);
    } else {
      // The control: below the budget the gather does exactly the dense work,
      // which is why a gate that never crosses 2048 cannot see a mask.
      CHECK(selected_total == T * kv);
      CHECK(visited == dense_reads);
    }

    // AND THE VALUES, against the host reference's own per-token gather over the
    // host reference's own expansion. The op is batched and the reference is per
    // token, so this is not a restatement.
    for (int64_t t = 0; t < T; ++t) {
      CAPTURE(t);
      const std::vector<int32_t> sel = ExpandHost(cfg, ids, topk, t, kv);
      const std::vector<float> q_row(qa.begin() + t * HQ * DH, qa.begin() + (t + 1) * HQ * DH);
      const std::vector<float> want =
          QsaGatherAttention(q_row, ka, va, sel, kv, HQ, HKV, DH, nullptr);
      for (int64_t i = 0; i < HQ * DH; ++i) CHECK(out[t * HQ * DH + i] == want[i]);
    }
  }
}

TEST_CASE("qsa-device: the composed indexer agrees with the host reference at released width") {
  // The goldens run index_head_dim 16 and 4 index heads over at most 5 blocks.
  // This runs the RELEASED indexer — 128-wide heads, 750 blocks, a top-k of 512
  // — where a reduction-order or window-bound defect has room to show and the
  // golden shapes have none. It compares the DEVICE composition against the HOST
  // reference's own score-and-select, block id for block id.
  Queue q = CpuQ();
  QsaConfig cfg;
  const int64_t D = cfg.index_head_dim, H = cfg.index_n_heads, topk = cfg.block_topk();
  constexpr int64_t kv = 3002, T = 3;
  const int64_t complete = (kv / cfg.compress_ratio) * cfg.compress_ratio;
  const int64_t nb = complete / cfg.compress_ratio;

  Lcg rng(0xabcdef0123456789ULL);
  std::vector<float> raw(static_cast<size_t>(complete * D));
  for (float& v : raw) v = rng.Next();
  std::vector<float> knw(static_cast<size_t>(D));
  for (float& v : knw) v = 0.1f * rng.Next();
  std::vector<float> cos(static_cast<size_t>(kv * cfg.rotary_dim), 1.0f);
  std::vector<float> sin(cos.size(), 0.0f);
  std::vector<float> qi(static_cast<size_t>(T * H * D));
  for (float& v : qi) v = rng.Next();
  const std::vector<int32_t> kv_lens(T, static_cast<int32_t>(kv));

  const Indexed idx = RunIndexer(q, cfg, raw, knw, cos, sin, qi, kv_lens, /*round_bf16=*/false);

  // The host arm, end to end and independently.
  const std::vector<float> host_keys =
      QsaCompressNormRope(raw, complete, knw, cos, sin, cfg, /*round_to_bf16=*/false);
  REQUIRE(static_cast<int64_t>(host_keys.size()) == nb * D);
  CHECK(RelL2(idx.block_keys, host_keys.data(), nb * D) < 1e-6);

  for (int64_t t = 0; t < T; ++t) {
    CAPTURE(t);
    const std::vector<float> q_row(qi.begin() + t * H * D, qi.begin() + (t + 1) * H * D);
    const std::vector<float> scores = QsaBlockScores(q_row, host_keys, nb, cfg);
    std::vector<int64_t> blocks = QsaTopkBlocks(scores, nb, std::min(topk, nb));
    std::sort(blocks.begin(), blocks.end());
    REQUIRE(static_cast<int64_t>(blocks.size()) == std::min(topk, nb));
    for (int64_t j = 0; j < static_cast<int64_t>(blocks.size()); ++j) {
      CHECK(idx.block_ids[static_cast<size_t>(t * topk + j)] ==
            static_cast<int32_t>(blocks[static_cast<size_t>(j)]));
    }
  }
}

// ── Refusals ─────────────────────────────────────────────────────────────────

TEST_CASE("the QSA ops refuse by name") {
  Queue q = CpuQ();
  constexpr int64_t D = 4, CR = 2, ROT = 2, N = 4;
  std::vector<float> raw(N * D, 0.5f);
  std::vector<float> w(D, 0.0f);
  std::vector<float> cos(N * ROT, 1.0f), sin(N * ROT, 0.0f);
  std::vector<float> bk(2 * D, 0.0f);
  Tensor t_raw = MakeT(raw.data(), DType::kF32, {N, D});
  Tensor t_w = MakeT(w.data(), DType::kF32, {D});
  Tensor t_cos = MakeT(cos.data(), DType::kF32, {N, ROT});
  Tensor t_sin = MakeT(sin.data(), DType::kF32, {N, ROT});
  Tensor t_bk = MakeT(bk.data(), DType::kF32, {2, D});

  Qwen4ExpQsaCompressArgs cok;
  cok.compress_ratio = CR;
  cok.rotary_dim = ROT;
  cok.eps = 1e-6f;

  SUBCASE("a partial trailing block, which writes no state upstream") {
    Tensor ragged = MakeT(raw.data(), DType::kF32, {3, D});
    Tensor out1 = MakeT(bk.data(), DType::kF32, {1, D});
    CHECK_THROWS_WITH_AS(
        vt::Qwen4ExpQsaCompress(q, out1, ragged, t_w, t_cos, t_sin, cok),
        doctest::Contains("whole number of COMPLETE blocks"), std::exception);
  }
  SUBCASE("a compress_ratio of 1, which is not a compressor") {
    Qwen4ExpQsaCompressArgs bad = cok;
    bad.compress_ratio = 1;
    CHECK_THROWS_WITH_AS(vt::Qwen4ExpQsaCompress(q, t_bk, t_raw, t_w, t_cos, t_sin, bad),
                         doctest::Contains("compress_ratio must be > 1"), std::exception);
  }
  SUBCASE("a rotary span wider than the index head") {
    // EVEN, so the evenness refusal below cannot cover for this one.
    Qwen4ExpQsaCompressArgs bad = cok;
    bad.rotary_dim = D + 2;
    std::vector<float> wide(N * (D + 2), 1.0f);
    Tensor c2 = MakeT(wide.data(), DType::kF32, {N, D + 2});
    CHECK_THROWS_WITH_AS(vt::Qwen4ExpQsaCompress(q, t_bk, t_raw, t_w, c2, c2, bad),
                         doctest::Contains("rotary_dim must fit"), std::exception);
  }
  SUBCASE("an odd rotary span, which rotate_half cannot pair") {
    // FITS the head, so only this refusal can fire on it.
    Qwen4ExpQsaCompressArgs bad = cok;
    bad.rotary_dim = D - 1;
    std::vector<float> odd(N * (D - 1), 1.0f);
    Tensor c2 = MakeT(odd.data(), DType::kF32, {N, D - 1});
    CHECK_THROWS_WITH_AS(vt::Qwen4ExpQsaCompress(q, t_bk, t_raw, t_w, c2, c2, bad),
                         doctest::Contains("rotary_dim must be even"), std::exception);
  }
  SUBCASE("cos/sin that do not cover every key position") {
    std::vector<float> shortc(2 * ROT, 1.0f);
    Tensor c2 = MakeT(shortc.data(), DType::kF32, {2, ROT});
    CHECK_THROWS_WITH_AS(vt::Qwen4ExpQsaCompress(q, t_bk, t_raw, t_w, c2, c2, cok),
                         doctest::Contains("cover every key position"), std::exception);
  }

  constexpr int64_t T = 2, HQ = 4, HKV = 2, DH = 8, KV = 8, TOPK = 2;
  std::vector<float> qa(T * HQ * DH, 0.1f), ka(KV * HKV * DH, 0.1f), va(KV * HKV * DH, 0.1f);
  std::vector<float> ao(T * HQ * DH, 0.0f);
  std::vector<int32_t> ids(T * TOPK, 0), lens(T, static_cast<int32_t>(KV));
  Tensor t_q = MakeT(qa.data(), DType::kF32, {T, HQ, DH});
  Tensor t_k = MakeT(ka.data(), DType::kF32, {KV, HKV, DH});
  Tensor t_v = MakeT(va.data(), DType::kF32, {KV, HKV, DH});
  Tensor t_ids = MakeT(ids.data(), DType::kI32, {T, TOPK});
  Tensor t_len = MakeT(lens.data(), DType::kI32, {T});
  Tensor t_out = MakeT(ao.data(), DType::kF32, {T, HQ, DH});
  Qwen4ExpQsaAttnArgs aok;
  aok.scale = 1.0f / std::sqrt(static_cast<float>(DH));
  aok.compress_ratio = 2;

  SUBCASE("an unset softmax scale") {
    Qwen4ExpQsaAttnArgs bad = aok;
    bad.scale = 0.0f;
    CHECK_THROWS_WITH_AS(
        vt::Qwen4ExpQsaGatherAttention(q, t_out, t_q, t_k, t_v, t_ids, t_len, bad),
        doctest::Contains("scale must be set explicitly"), std::exception);
  }
  SUBCASE("query heads that do not group over the KV heads") {
    std::vector<float> odd(T * 3 * DH, 0.1f), oddo(T * 3 * DH, 0.0f);
    Tensor q3 = MakeT(odd.data(), DType::kF32, {T, 3, DH});
    Tensor o3 = MakeT(oddo.data(), DType::kF32, {T, 3, DH});
    CHECK_THROWS_WITH_AS(
        vt::Qwen4ExpQsaGatherAttention(q, o3, q3, t_k, t_v, t_ids, t_len, aok),
        doctest::Contains("divisible by num_kv_heads"), std::exception);
  }
  SUBCASE("a kv_lens with the wrong length") {
    std::vector<int32_t> one(1, static_cast<int32_t>(KV));
    Tensor l1 = MakeT(one.data(), DType::kI32, {1});
    CHECK_THROWS_WITH_AS(
        vt::Qwen4ExpQsaGatherAttention(q, t_out, t_q, t_k, t_v, t_ids, l1, aok),
        doctest::Contains("kv_lens must be [tokens]"), std::exception);
  }
  SUBCASE("block ids that are not i32") {
    std::vector<float> f(T * TOPK, 0.0f);
    Tensor bad = MakeT(f.data(), DType::kF32, {T, TOPK});
    CHECK_THROWS_WITH_AS(
        vt::Qwen4ExpQsaGatherAttention(q, t_out, t_q, t_k, t_v, bad, t_len, aok),
        doctest::Contains("block_ids must be i32"), std::exception);
  }
  SUBCASE("a selected block past the end of the visible prefix") {
    // The kernel's own bound, not the dispatcher's: `kv_lens` is host-readable
    // here, but the id is only resolvable against it at the read.
    std::vector<int32_t> far(T * TOPK, static_cast<int32_t>(KV));  // block KV, not token KV
    Tensor t_far = MakeT(far.data(), DType::kI32, {T, TOPK});
    CHECK_THROWS_WITH_AS(
        vt::Qwen4ExpQsaGatherAttention(q, t_out, t_q, t_k, t_v, t_far, t_len, aok),
        doctest::Contains("selected block"), std::exception);
  }
}

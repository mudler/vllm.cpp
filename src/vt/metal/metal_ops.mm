// Metal backend — op kernels: encode/dispatch of the runtime-compiled MSL in
// metal_msl.h, plus the `RegisterOp` table entries. BACKEND-METAL-MLX, W0
// skeleton. Self-registering TU, copying the `src/vt/cpu/cpu_ops.cpp` Registrar
// idiom exactly, so adding this backend edited NO existing kernel file.
//
// WHAT THIS TU COVERS.
//   W0 seam proof: kAdd, kRelu, kSiluAndMul, kCastBf16, kCastF32, kLayerNorm,
//   kRmsNorm, the dense GEMM pair kMatmul/kMatmulBT, and the single kFusedChain
//   registration that inherits the portable fusion catalog. That set spans every
//   structural class the seam has to get right: flat elementwise, a rank-1
//   broadcast, a dtype-converting copy, TWO different row reductions, an optional
//   in-place residual stream, and the recipe interpreter.
//   M3a (THE FIRST MODEL): kEmbedding, kQkvSplit, kReshapeAndCache,
//   kPagedAttention, kGreedyArgmax — exactly the five ops OPT-125m
//   (`OPTForCausalLM`) needs beyond the W0 set, and no more. With these,
//   src/vllm/model_executor/models/opt.cpp runs END TO END on Apple GPU through
//   the ordinary engine stack. NOTE kPagedAttention is OURS regardless of MLX:
//   MLX has no paged-KV attention primitive at all
//   (.agents/specs/metal-mlx-reuse-study.md §5.3).
//
//   M3b (Qwen3-dense, `Qwen3ForCausalLM`): kRopeCosSinCache, kRopeFromCache and
//   kRopeNeox — the RoPE ops beyond OPT's set. Qwen3's dense attention preamble
//   defaults to the CACHE path (VT_QWEN3_ROPE_CACHE defaults ON): it builds the
//   per-step cos|sin cache with kRopeCosSinCache and applies it with
//   kRopeFromCache (both exercised by the SACRED gate). kRopeNeox serves the
//   VT_QWEN3_ROPE_CACHE=0 opt-out (in-place fp-transcendental rotation). With
//   these, qwen3.cpp runs END TO END on Apple GPU.
//
// WHAT IS STILL STUBBED: everything else — the whole quant tier, the GDN/MoE/MLA
// families, MRoPE (rank-2 rope-from-cache positions), and every sampler op except
// greedy argmax. `vt::GetOp` throws its normal "no kernel for op <Name> (id N) on
// device metal (type 2)" for them (a partial backend is a supported, tested
// state).
//
// DISPATCH MODEL (M3c-1): dispatches are BATCHED into one shared command buffer
// and committed at a flush point, not one buffer per op. See the Batch struct
// below for the three correctness facts that rests on, and
// .agents/specs/metal-dispatch-attribution.md for the measurement that forced
// it. VT_METAL_SYNC_DISPATCH=1 restores the old one-buffer-per-op behaviour.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "vt/metal_profile.h"

#include "metal_buffers.h"
#include "metal_context.h"
#include "vt/ops.h"

namespace vt::metal {
namespace {

// Storage dtype -> the shader-side code (metal_msl.h VT_DT_*).
uint32_t DtypeCode(DType d) {
  switch (d) {
    case DType::kF32: return 0;
    case DType::kF16: return 1;
    case DType::kBF16: return 2;
    default: break;
  }
  VT_CHECK(false, "metal: unsupported storage dtype (f32/f16/bf16 only in the W0 skeleton)");
  return 0;
}

// ---- Host mirrors of the metal_msl.h parameter structs. Field order and types
// must match the MSL declarations EXACTLY; MSL scalar `uint`/`float` are 4-byte
// with 4-byte alignment, same as here, and every struct is a multiple of 4 with
// no member exceeding 4 bytes, so the layouts coincide without padding surprises.
struct ElemParams { uint32_t n, d, a_dt, b_dt, out_dt, bcast; };
struct SiluMulParams { uint32_t t, d, x_dt, out_dt; };
struct RmsParams { uint32_t t, h, x_dt, w_dt, out_dt, res_dt, has_res, gemma, tg; float eps; };
struct LayerNormParams {
  uint32_t rows, d, x_dt, w_dt, b_dt, out_dt, has_w, has_b, tg;
  float eps;
};
struct GemmParams { uint32_t m, n, k, lda, a_dt, b_dt, out_dt, bt; };
struct EmbedParams { uint32_t rows, h, vocab, id_i64, tab_dt, out_dt; };
struct QkvSplitParams { uint32_t t, q_dim, k_dim, v_dim, in_dt, q_dt, k_dt, v_dt; };
// MSL `ulong` is 8-byte with 8-byte alignment, exactly like uint64_t here. Both
// structs below put ALL 8-byte members FIRST and then an even number of 4-byte
// members, so neither side can insert interior padding and the layouts coincide
// by construction rather than by coincidence (the static_asserts below pin it).
struct CacheParams {
  uint64_t k_blk_stride, k_pg_stride, v_blk_stride, v_pg_stride, k_tok_stride, v_tok_stride;
  uint32_t num_slots, n_elems, block_size, esz;
};
struct PagedAttnParams {
  uint64_t kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd;
  uint32_t num_reqs, hq, d, qpk, block_size, causal, tg;
  int32_t window_left, window_right, bt_row, bt_col;
  uint32_t q_dt, kc_dt, vc_dt, out_dt;
  float scale;
};
struct ArgmaxParams { uint32_t n, v, tg; };
struct RopeParams {
  uint32_t t, hq, hk, d, rot, rhalf, q_dt, k_dt, pos_i64;
  float base;
};
struct RopeCacheParams {
  uint32_t t, rot, rhalf, out_dt, pos_i64;
  float base;
};
struct RopeApplyParams {
  uint64_t q_s0, q_s1, k_s0, k_s1;
  uint32_t t, hq, hk, rot, rhalf, is_neox, q_dt, k_dt, cache_dt, pos_i64, has_k, pad;
};
struct BwParams { uint32_t n_chunks, chunk_f4, stride_f4, tg, chunks_per_tg, pad; };
struct QkNormRopeParams {
  uint64_t q_s0, q_s1, k_s0, k_s1;
  uint32_t t, hq, hk, d, rot, rhalf, is_neox, pos_i64;
  uint32_t q_dt, k_dt, qw_dt, kw_dt, cache_dt, gemma, tg, pad;
  float eps, pad2;
};
struct FStepGpu { uint32_t op, out, in0, in1, gemma, pad; };
struct FcParams { uint32_t t, h, nsteps, x_dt, w_dt, res_dt, out_dt, tg; float eps; };

static_assert(sizeof(RopeParams) == 40, "RopeParams layout must match the MSL struct");
static_assert(sizeof(RopeCacheParams) == 24, "RopeCacheParams layout must match the MSL struct");
static_assert(sizeof(RopeApplyParams) == 80, "RopeApplyParams layout must match the MSL struct");
static_assert(offsetof(RopeApplyParams, t) == 32, "RopeApplyParams: no interior padding");
static_assert(sizeof(ElemParams) == 24, "ElemParams layout must match the MSL struct");
static_assert(sizeof(FStepGpu) == 24, "FStepGpu layout must match the MSL struct");
static_assert(sizeof(CacheParams) == 64, "CacheParams layout must match the MSL struct");
// 6 x 8B + 16 x 4B = 112, with no trailing pad: the 4-byte member COUNT is even,
// which is what keeps the 8-byte alignment satisfied without either side
// inventing padding the other does not have.
static_assert(sizeof(PagedAttnParams) == 112, "PagedAttnParams layout must match the MSL struct");
static_assert(offsetof(PagedAttnParams, num_reqs) == 48, "PagedAttnParams: no interior padding");
static_assert(offsetof(CacheParams, num_slots) == 48, "CacheParams: no interior padding");

// --- VT_METAL_PROFILE: per-dispatch time attribution -----------------------
// Instruments has no Metal System Trace without a full Xcode, which the Apple
// box does not have, so the attribution the benchmark protocol requires
// ("trace the execution, not just the code") is collected in-process instead.
// The phases, which are exactly the host-build vs compute split:
//   encode_s  — PER DISPATCH: Encoder ctor through the appended dispatch
//   wait_s    — PER BATCH: wall time inside commit + waitUntilCompleted
//   gpu_s     — PER BATCH: MTLCommandBuffer GPUEndTime - GPUStartTime
// Since M3c-1 batched the dispatches, wait and GPU time are properties of the
// COMMIT, not of any one op, so they are reported on the total row only.
// gpu_s << wait_s means the backend is submit/synchronisation bound and tuning
// kernels cannot pay; gpu_s ~= wait_s means it is genuinely compute bound.
// OFF unless VT_METAL_PROFILE=1, and it adds nothing to the hot path but two
// clock reads next to an already-blocking round trip.
namespace vtprof {

using Row = vt::metal::ProfileRow;

struct Table {
  std::mutex mu;
  std::map<std::string, Row> rows;
  // The end-of-process report is tied to the ENVIRONMENT only, never to
  // SetProfileEnabled: a test that flips the flag must not print to stderr.
  const bool env_enabled = [] {
    const char* e = std::getenv("VT_METAL_PROFILE");
    return e != nullptr && e[0] == '1';
  }();
  std::atomic<bool> enabled{env_enabled};
  // Batch-level, NOT per-dispatch. Once dispatches share a command buffer
  // there is no per-dispatch wait or GPU time to attribute, so the flush
  // cost is accumulated here and the per-kernel rows keep only encode time.
  std::atomic<unsigned long long> commits{0};
  double commit_wait_s = 0.0;
  double commit_gpu_s = 0.0;

  void Add(const char* name, double encode_s, double wait_s, double gpu_s) {
    std::lock_guard<std::mutex> lock(mu);
    Row& r = rows[name];
    if (r.name.empty()) r.name = name;
    ++r.count;
    r.encode_s += encode_s;
    r.wait_s += wait_s;
    r.gpu_s += gpu_s;
  }

  ~Table() {
    if (!env_enabled) return;
    std::lock_guard<std::mutex> lock(mu);
    if (rows.empty()) return;
    Row tot;
    for (const auto& kv : rows) {
      tot.count += kv.second.count;
      tot.encode_s += kv.second.encode_s;
    }
    tot.wait_s = commit_wait_s;
    tot.gpu_s = commit_gpu_s;
    const unsigned long long n_commits = commits.load(std::memory_order_relaxed);
    std::fprintf(stderr,
                 "\n[vt metal-profile] dispatches=%llu commits=%llu "
                 "dispatches_per_commit=%.1f encode=%.3fs wait=%.3fs "
                 "gpu_busy=%.3fs gpu_busy_frac_of_wait=%.1f%%\n",
                 tot.count, n_commits,
                 n_commits > 0 ? static_cast<double>(tot.count) / n_commits : 0.0,
                 tot.encode_s, tot.wait_s, tot.gpu_s,
                 tot.wait_s > 0.0 ? 100.0 * tot.gpu_s / tot.wait_s : 0.0);
    std::fprintf(stderr,
                 "[vt metal-profile] %-28s %10s %10s %10s %10s %9s\n", "kernel",
                 "count", "encode_ms", "wait_ms", "gpu_ms", "gpu/wait");
    std::vector<std::pair<std::string, Row>> v(rows.begin(), rows.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
      return a.second.wait_s > b.second.wait_s;
    });
    for (const auto& kv : v) {
      std::fprintf(stderr,
                   "[vt metal-profile] %-28s %10llu %10.1f %10.1f %10.1f %8.1f%%\n",
                   kv.first.c_str(), kv.second.count, 1e3 * kv.second.encode_s,
                   1e3 * kv.second.wait_s, 1e3 * kv.second.gpu_s,
                   kv.second.wait_s > 0.0 ? 100.0 * kv.second.gpu_s / kv.second.wait_s
                                          : 0.0);
    }
  }
};

inline Table& Get() {
  static Table t;
  return t;
}

inline double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace vtprof

}  // namespace — closed so the public profile API below has external linkage

bool ProfileEnabled() {
  return vtprof::Get().enabled.load(std::memory_order_relaxed);
}

void SetProfileEnabled(bool on) {
  vtprof::Get().enabled.store(on, std::memory_order_relaxed);
}

std::vector<ProfileRow> GetProfileRows() {
  vtprof::Table& t = vtprof::Get();
  std::lock_guard<std::mutex> lock(t.mu);
  std::vector<ProfileRow> out;
  if (t.rows.empty()) return out;
  out.reserve(t.rows.size() + 1);
  ProfileRow tot;
  for (const auto& kv : t.rows) {
    out.push_back(kv.second);
    tot.count += kv.second.count;
    tot.encode_s += kv.second.encode_s;
  }
  // Since M3c-1 wait and GPU time are properties of the BATCH, not of any one
  // dispatch, so the total takes them from the flush accounting. `count` stays
  // the DISPATCH count, which is what makes count-vs-commits the batching ratio.
  tot.wait_s = t.commit_wait_s;
  tot.gpu_s = t.commit_gpu_s;
  out.push_back(tot);  // total last, with an empty name
  return out;
}

void ResetProfile() {
  vtprof::Table& t = vtprof::Get();
  std::lock_guard<std::mutex> lock(t.mu);
  t.rows.clear();
  t.commits.store(0, std::memory_order_relaxed);
  t.commit_wait_s = 0.0;
  t.commit_gpu_s = 0.0;
}

unsigned long long GetProfileCommits() {
  return vtprof::Get().commits.load(std::memory_order_relaxed);
}

namespace {

// --- M3c-1: the batched command buffer -------------------------------------
// Attribution (.agents/specs/metal-dispatch-attribution.md) measured 50,944
// command buffers for a 128-token generation, each costing a ~186 us
// commit+waitUntilCompleted round trip: 11.3 s, a third of the run, spent NOT
// computing. The fix is to stop making the command buffer the unit of the op.
//
// ONE command buffer and ONE compute encoder are kept open across dispatches
// and committed at a flush point. Correctness rests on three facts, each of
// which is pinned by a test in tests/vt/test_metal_backend.cpp:
//
//   1. ORDERING IS FREE INSIDE THE ENCODER. `computeCommandEncoder` creates a
//      MTLDispatchTypeSerial encoder, which serialises its dispatches, so a
//      read-modify-write chain composes exactly as it did when every op had its
//      own buffer. (Moving to a concurrent encoder would need explicit
//      memoryBarrier calls; we deliberately do not.)
//   2. THE HOST MUST NEVER SEE STALE BYTES. Metal storage here is Shared, so
//      Copy/Memset are host memcpy over the same pages the GPU writes, and Free
//      releases a buffer the GPU may still reference. Every one of those paths
//      flushes first (metal_backend.mm), as does Synchronize.
//   3. A HALF-BOUND ENCODER MUST NOT LEAK. A bind can throw (Resolve rejects
//      foreign memory) after the pipeline state is set; the shared encoder would
//      then carry that state into the NEXT op. The destructor flushes on that
//      path so the next dispatch starts from a clean encoder.
//
// The batch is process-wide because the MTLCommandQueue is: metal_backend.mm
// hands every vt::Queue the same underlying queue, so one batch preserves
// exactly the global ordering the synchronous design had.
//
// `VT_METAL_SYNC_DISPATCH=1` restores one-command-buffer-per-op. This is the
// debug mode the spec's risk section requires: batched, a command-buffer error
// covers a span of ops instead of naming one, so a bisect needs the old
// behaviour available without a rebuild.
struct Batch {
  std::recursive_mutex mu;
  id<MTLCommandBuffer> cmd = nil;
  id<MTLComputeCommandEncoder> enc = nil;
  unsigned long long pending = 0;
  // Name of the last dispatch encoded. Only meaningful when a commit carries
  // exactly ONE dispatch (VT_METAL_SYNC_DISPATCH), which is the mode used to
  // recover a PER-KERNEL GPU breakdown: batching deliberately makes wait/GPU a
  // property of the commit, so the per-kernel view has to come from somewhere.
  const char* last_kernel = "";
  const bool sync_mode = [] {
    const char* e = std::getenv("VT_METAL_SYNC_DISPATCH");
    return e != nullptr && e[0] == '1';
  }();
};

Batch& TheBatch() {
  static Batch b;
  return b;
}

// Commit whatever is open and block until the GPU is done with it. Caller holds
// `mu`. No-op when nothing is pending, so flush points can call it freely.
void FlushLocked(Batch& b) {
  if (b.enc == nil) return;
  [b.enc endEncoding];
  const bool prof = vtprof::Get().enabled.load(std::memory_order_relaxed);
  const double t_commit = prof ? vtprof::Now() : 0.0;
  [b.cmd commit];
  [b.cmd waitUntilCompleted];
  if (prof) {
    const double t_done = vtprof::Now();
    const double gpu = [b.cmd GPUEndTime] - [b.cmd GPUStartTime];
    vtprof::Table& tbl = vtprof::Get();
    tbl.commits.fetch_add(1, std::memory_order_relaxed);
    const double gpu_pos = gpu > 0.0 ? gpu : 0.0;
    {
      std::lock_guard<std::mutex> lock(tbl.mu);
      tbl.commit_wait_s += t_done - t_commit;
      tbl.commit_gpu_s += gpu_pos;
      // One dispatch in this commit => the whole GPU interval belongs to it.
      // With many dispatches batched there is no honest way to split it, so the
      // per-kernel columns stay 0 rather than being invented.
      if (b.pending == 1 && b.last_kernel[0] != '\0') {
        vtprof::Row& r = tbl.rows[b.last_kernel];
        if (r.name.empty()) r.name = b.last_kernel;
        r.wait_s += t_done - t_commit;
        r.gpu_s += gpu_pos;
      }
    }
  }
  // Capture the error BEFORE releasing, and clear the batch BEFORE throwing:
  // an exception must not leave a committed buffer installed as "pending".
  NSError* err = [b.cmd error];
  std::string msg;
  if (err != nil) msg = [[err localizedDescription] UTF8String];
  [b.enc release];
  [b.cmd release];
  b.enc = nil;
  b.cmd = nil;
  b.pending = 0;
  VT_CHECK(msg.empty(), std::string("metal: command buffer failed: ") + msg);
}

}  // namespace

// The flush entry point the backend's Synchronize/Copy/Memset/Free call, and
// the one op_provider reaches through Backend::FlushPending before running a
// portable CPU-reference kernel directly over Metal memory.
void FlushPendingBatch() {
  Batch& b = TheBatch();
  std::lock_guard<std::recursive_mutex> lock(b.mu);
  FlushLocked(b);
}

namespace {

// A small RAII-ish encode helper: appends one dispatch to the shared batch.
// Holds the batch lock for the whole encode, because a single
// MTLComputeCommandEncoder cannot be written by two threads at once (each op
// used to own its encoder, so this serialisation is new; it costs the ~4 us an
// encode takes, against the ~186 us it saves).
class Encoder {
 public:
  // `prof_label`, when given, is what the profile attributes this dispatch to.
  // It exists so ONE pipeline can be split by shape class in the report (a
  // decode GEMV and a prefill GEMM are the same kernel but not the same
  // performance problem), without inventing a second pipeline.
  explicit Encoder(const char* fn_name, const char* prof_label = nullptr)
      : lock_(TheBatch().mu) {
    prof_ = vtprof::Get().enabled.load(std::memory_order_relaxed);
    if (prof_) {
      prof_name_ = prof_label != nullptr ? prof_label : fn_name;
      t_ctor_ = vtprof::Now();
    }
    pool_ = [[NSAutoreleasePool alloc] init];
    auto& ctx = MetalContext::Get();
    Batch& b = TheBatch();
    if (b.enc == nil) {
      id<MTLCommandQueue> q = static_cast<id<MTLCommandQueue>>(ctx.command_queue());
      // RETAINED: [q commandBuffer] is autoreleased, and the batch outlives this
      // Encoder's pool by design. Released in FlushLocked.
      b.cmd = [[q commandBuffer] retain];
      VT_CHECK(b.cmd != nil, "metal: commandBuffer failed");
      b.enc = [[b.cmd computeCommandEncoder] retain];
      VT_CHECK(b.enc != nil, "metal: computeCommandEncoder failed");
    }
    enc_ = b.enc;
    auto pso = static_cast<id<MTLComputePipelineState>>(ctx.Pipeline(std::string(fn_name)));
    [enc_ setComputePipelineState:pso];
    max_threads_ = static_cast<size_t>([pso maxTotalThreadsPerThreadgroup]);
  }
  // A bind can THROW (Resolve rejects memory this backend did not allocate),
  // unwinding past the dispatch. The encoder is now SHARED, so the failure mode
  // changed: the half-bound state (pipeline set, some buffers bound, no
  // dispatch) would otherwise leak into the NEXT op, whose own binds might not
  // overwrite every index. Flushing here ends the encoder and commits the
  // dispatches already legitimately encoded, so the next Encoder starts clean.
  // `finished_` makes this idempotent.
  ~Encoder() {
    if (!finished_) {
      FlushLocked(TheBatch());
      finished_ = true;
    }
    [pool_ release];
  }

  Encoder(const Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;

  // Bind a vt::Tensor's storage, resolving the (possibly interior) pointer to
  // the owning MTLBuffer + offset.
  void BindTensor(const Tensor& t, int index, const char* what) {
    Resolved r = Resolve(t.data, what);
    [enc_ setBuffer:static_cast<id<MTLBuffer>>(r.buffer) offset:r.offset atIndex:index];
  }
  // Bind small host-side data by value (params structs, the recipe step list).
  // setBytes is the documented path for <4 KiB of per-dispatch constants.
  void BindBytes(const void* data, size_t bytes, int index) {
    VT_CHECK(bytes <= 4096, "metal: setBytes payload must stay under 4 KiB");
    [enc_ setBytes:data length:bytes atIndex:index];
  }

  // Flat dispatch: `n` threads, non-uniform threadgroups (Apple GPUs support
  // dispatchThreads:, so no manual grid rounding is needed).
  void DispatchFlat(int64_t n) {
    if (n <= 0) { Finish(); return; }
    // Clamp to THIS pipeline's limit, which the compiler may set below the
    // device max; dispatching past it is a hard Metal error.
    const NSUInteger tg = ChooseThreadgroupSize(n, max_threads_);
    [enc_ dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    Finish();
  }
  // Row dispatch: one THREADGROUP per row, `tg` threads each (llama.cpp
  // kernel_rms_norm shape).
  void DispatchRows(int64_t rows, uint32_t tg) {
    if (rows <= 0) { Finish(); return; }
    [enc_ dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    Finish();
  }
  // 2-D threadgroup grid: EXACTLY (gx, gy) threadgroups of `tg` threads each,
  // with no tiling arithmetic — the paged-attention shape, where one threadgroup
  // owns one (q-head, query token) pair and cooperatively reduces over keys.
  void DispatchGrid2D(int64_t gx, int64_t gy, uint32_t tg) {
    if (gx <= 0 || gy <= 0) { Finish(); return; }
    // A threadgroup wider than THIS pipeline's limit is undefined behaviour, and
    // in a Release build with no Metal validation layer it does not fault: it
    // silently computes garbage. `maxTotalThreadsPerThreadgroup` shrinks with
    // register and threadgroup-memory pressure, so a kernel that grows its tile
    // can cross the limit without any source-level hint. DispatchFlat already
    // clamps; this path asserts instead, because a 2-D kernel's threadgroup
    // shape is part of its correctness (clamping would silently drop simdgroups
    // and produce a wrong answer just as quietly).
    VT_CHECK(size_t(tg) <= max_threads_,
             std::string("metal: threadgroup of ") + std::to_string(tg) +
                 " threads exceeds this pipeline's maxTotalThreadsPerThreadgroup (" +
                 std::to_string(max_threads_) + ")");
    [enc_ dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(gx),
                                           static_cast<NSUInteger>(gy), 1)
         threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    Finish();
  }
  // 2-D tile dispatch: one threadgroup per (tile x tile) output tile, which is
  // the GEMM shape. dispatchThreadgroups (not dispatchThreads) because the tile
  // loop needs FULL threadgroups — the kernel bounds-checks its own edges.
  void DispatchTiles(int64_t cols, int64_t rows, uint32_t tile) {
    if (cols <= 0 || rows <= 0) { Finish(); return; }
    const NSUInteger gx = static_cast<NSUInteger>((cols + tile - 1) / tile);
    const NSUInteger gy = static_cast<NSUInteger>((rows + tile - 1) / tile);
    [enc_ dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
         threadsPerThreadgroup:MTLSizeMake(tile, tile, 1)];
    Finish();
  }

 private:
  // The dispatch is now APPENDED, not submitted. Nothing here blocks: the
  // command buffer stays open for the next op and is committed at a flush
  // point. That removal of ~186 us per op is the whole of M3c-1.
  void Finish() {
    finished_ = true;
    Batch& b = TheBatch();
    ++b.pending;
    b.last_kernel = prof_name_;
    if (prof_) {
      // Only encode time is per-dispatch now; wait and GPU time belong to the
      // batch and are accumulated in FlushLocked.
      vtprof::Get().Add(prof_name_, vtprof::Now() - t_ctor_, 0.0, 0.0);
    }
    // A batch that nothing ever flushes would grow without bound and hold every
    // buffer it references alive. The engine flushes constantly (the sampler
    // reads through Copy every step), so this cap is a backstop for pathological
    // callers, not a tuning knob.
    if (b.sync_mode || b.pending >= kMaxPendingDispatches) FlushLocked(b);
  }
  static constexpr unsigned long long kMaxPendingDispatches = 4096;

  std::unique_lock<std::recursive_mutex> lock_;
  NSAutoreleasePool* pool_ = nil;
  id<MTLComputeCommandEncoder> enc_ = nil;
  size_t max_threads_ = 0;
  bool finished_ = false;
  bool prof_ = false;
  const char* prof_name_ = "";
  double t_ctor_ = 0.0;
};

// ---------------------------------------------------------------------------
// Kernels. Every argument was already validated by the vt:: wrapper in
// src/vt/ops.cpp before GetOp dispatched here, so these only translate.
// ---------------------------------------------------------------------------

// cpu_layernorm.cpp:87-99 AddKernel.
void AddKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t n = a.Numel();
  const int64_t d = a.rank == 0 ? 1 : a.shape[a.rank - 1];
  const bool bcast = b.rank == 1 && a.rank != 1;
  ElemParams p{static_cast<uint32_t>(n), static_cast<uint32_t>(d), DtypeCode(a.dtype),
               DtypeCode(b.dtype), DtypeCode(out.dtype), bcast ? 1u : 0u};
  Encoder e("vt_add");
  e.BindTensor(a, 0, "add: a");
  e.BindTensor(b, 1, "add: b");
  e.BindTensor(out, 2, "add: out");
  e.BindBytes(&p, sizeof(p), 3);
  e.DispatchFlat(n);
}

// Dense GEMM, the NATIVE MSL provider for kMatmul / kMatmulBT (metal_msl.h
// vt_matmul). Registered at the default priority under vt::kNativeProviderName,
// so it stays the DEFAULT on Metal and the optional MLX provider
// (metal_mlx_provider.mm, VLLM_CPP_MLX) only displaces it when explicitly built
// in — and can be switched back off in the same binary with
// VT_OP_PROVIDER_DISABLE=mlx.
constexpr uint32_t kGemmTile = 16;
// Must match VT_GEMV_SGS in metal_msl.h; Apple simdgroups are 32-wide.
constexpr uint32_t kGemvSimdgroups = 8;
// Must match VT_MM_BM/VT_MM_BN/VT_MM_SGS in metal_msl.h.
constexpr int64_t kMmTile = 64;
constexpr uint32_t kMmSimdgroups = 8;

// Same-binary A/B lever for the GEMV fast path, which the benchmark protocol
// requires ("use a same-binary A/B") and which doubles as the bisect switch if
// a decode result is ever suspected of coming from this kernel.
// VT_METAL_NO_GEMV=1 routes m=1 back through the tile GEMM.
// Separate lever for the m>1 simdgroup GEMM, so it can be A/B'd INDEPENDENTLY of
// the m=1 GEMV. VT_METAL_NO_GEMV disables the whole fast-path family;
// VT_METAL_NO_MM disables only this kernel. Without the split, an A/B of one
// lever silently measures both.
bool MmEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_METAL_NO_MM");
    return !(e != nullptr && e[0] == '1');
  }();
  return on;
}

bool GemvEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_METAL_NO_GEMV");
    return !(e != nullptr && e[0] == '1');
  }();
  return on;
}
constexpr uint32_t kSimdWidth = 32;

void MatmulKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t m = a.shape[0], k = a.shape[1], n = b.shape[1];
  GemmParams p{static_cast<uint32_t>(m),       static_cast<uint32_t>(n),
               static_cast<uint32_t>(k),       static_cast<uint32_t>(a.stride[0]),
               DtypeCode(a.dtype),             DtypeCode(b.dtype),
               DtypeCode(out.dtype),           0u};
  Encoder e("vt_matmul", m == 1 ? "vt_matmul(gemv m=1)" : "vt_matmul(gemm m>1)");
  e.BindTensor(a, 0, "matmul: a");
  e.BindTensor(b, 1, "matmul: b");
  e.BindTensor(out, 2, "matmul: out");
  e.BindBytes(&p, sizeof(p), 3);
  e.DispatchTiles(n, m, kGemmTile);
}

void MatmulBTKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t m = a.shape[0], k = a.shape[1], n = b.shape[0];
  GemmParams p{static_cast<uint32_t>(m),       static_cast<uint32_t>(n),
               static_cast<uint32_t>(k),       static_cast<uint32_t>(a.stride[0]),
               DtypeCode(a.dtype),             DtypeCode(b.dtype),
               DtypeCode(out.dtype),           1u};
  // M3d: decode (m == 1) takes the GEMV kernel, which is the shape 21,464 of
  // 21,632 matmuls in a generation actually have. Everything else keeps the
  // tile GEMM. Routing on m alone keeps the gate narrow: same op, same operands,
  // one kernel swapped for a shape it is strictly better suited to.
  if (m == 1 && GemvEnabled()) {
    Encoder e("vt_matmul_bt_gemv");
    e.BindTensor(a, 0, "matmul_bt: a");
    e.BindTensor(b, 1, "matmul_bt: b");
    e.BindTensor(out, 2, "matmul_bt: out");
    e.BindBytes(&p, sizeof(p), 3);
    // One simdgroup per output column, kGemvSimdgroups of them per threadgroup.
    e.DispatchGrid2D((n + kGemvSimdgroups - 1) / kGemvSimdgroups, 1,
                     kGemvSimdgroups * kSimdWidth);
    return;
  }
  // m > 1: the 2-D blocked simdgroup GEMM. One 32x32 output tile per
  // threadgroup, 4 simdgroups of 32 threads.
  if (GemvEnabled() && MmEnabled()) {
    Encoder e("vt_matmul_bt_mm");
    e.BindTensor(a, 0, "matmul_bt: a");
    e.BindTensor(b, 1, "matmul_bt: b");
    e.BindTensor(out, 2, "matmul_bt: out");
    e.BindBytes(&p, sizeof(p), 3);
    e.DispatchGrid2D((n + kMmTile - 1) / kMmTile, (m + kMmTile - 1) / kMmTile,
                     kMmSimdgroups * kSimdWidth);
    return;
  }
  Encoder e("vt_matmul", "vt_matmul_bt(gemm m>1)");
  e.BindTensor(a, 0, "matmul_bt: a");
  e.BindTensor(b, 1, "matmul_bt: b");
  e.BindTensor(out, 2, "matmul_bt: out");
  e.BindBytes(&p, sizeof(p), 3);
  e.DispatchTiles(n, m, kGemmTile);
}

// cpu_layernorm.cpp:75-85 ReluKernel.
void ReluKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t n = x.Numel();
  ElemParams p{static_cast<uint32_t>(n), 1u, DtypeCode(x.dtype), 0u, DtypeCode(out.dtype), 0u};
  Encoder e("vt_relu");
  e.BindTensor(x, 0, "relu: x");
  e.BindTensor(out, 1, "relu: out");
  e.BindBytes(&p, sizeof(p), 2);
  e.DispatchFlat(n);
}

// cpu_ops.cpp:1436-1451 CastBf16Kernel / CastF32Kernel — one shader serves both
// (the CPU pair is likewise the same LoadF32/StoreF32 body twice).
void CastKernel(Queue&, Tensor& out, const Tensor& in) {
  const int64_t n = out.Numel();
  ElemParams p{static_cast<uint32_t>(n), 1u, DtypeCode(in.dtype), 0u, DtypeCode(out.dtype), 0u};
  Encoder e("vt_cast");
  e.BindTensor(in, 0, "cast: in");
  e.BindTensor(out, 1, "cast: out");
  e.BindBytes(&p, sizeof(p), 2);
  e.DispatchFlat(n);
}

// cpu_ops.cpp:252-264 SiluAndMulKernel.
void SiluAndMulKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t t = x.shape[0], d = x.shape[1] / 2;
  SiluMulParams p{static_cast<uint32_t>(t), static_cast<uint32_t>(d), DtypeCode(x.dtype),
                  DtypeCode(out.dtype)};
  Encoder e("vt_silu_and_mul");
  e.BindTensor(x, 0, "silu_and_mul: x");
  e.BindTensor(out, 1, "silu_and_mul: out");
  e.BindBytes(&p, sizeof(p), 2);
  e.DispatchFlat(t * d);
}

// cpu_ops.cpp:225-250 RmsNormKernel.
void RmsNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                   const RmsNormArgs& args, Tensor* residual) {
  const int64_t t = x.shape[0], h = x.shape[1];
  // Pipeline-specific cap: vt_rms_norm declares threadgroup scratch, so its
  // maxTotalThreadsPerThreadgroup can be LOWER than the device limit and
  // over-dispatching is a hard Metal error. Resolved BEFORE the Encoder is built
  // because `tg` also travels in the params struct — the shader's stride loop and
  // the host dispatch must agree exactly.
  const uint32_t tg =
      ChooseThreadgroupSize(h, MetalContext::Get().PipelineMaxThreads("vt_rms_norm"));
  RmsParams p{static_cast<uint32_t>(t),
              static_cast<uint32_t>(h),
              DtypeCode(x.dtype),
              DtypeCode(w.dtype),
              DtypeCode(out.dtype),
              residual != nullptr ? DtypeCode(residual->dtype) : 0u,
              residual != nullptr ? 1u : 0u,
              args.gemma ? 1u : 0u,
              tg,
              args.eps};
  Encoder e("vt_rms_norm");
  e.BindTensor(x, 0, "rmsnorm: x");
  e.BindTensor(w, 1, "rmsnorm: weight");
  e.BindTensor(out, 2, "rmsnorm: out");
  // Buffer 3 is always bound: an unbound buffer is undefined even when the
  // shader never reads it. With has_res == 0 it aliases `out` and is dead.
  e.BindTensor(residual != nullptr ? *residual : out, 3, "rmsnorm: residual");
  e.BindBytes(&p, sizeof(p), 4);
  e.DispatchRows(t, tg);
}

// cpu_layernorm.cpp:49-73 LayerNormKernel.
void LayerNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor* weight,
                     const Tensor* bias, const LayerNormArgs& args) {
  const int64_t d = x.shape[x.rank - 1];
  const int64_t rows = d == 0 ? 0 : x.Numel() / d;
  const uint32_t tg =
      ChooseThreadgroupSize(d, MetalContext::Get().PipelineMaxThreads("vt_layer_norm"));
  LayerNormParams p{static_cast<uint32_t>(rows),
                    static_cast<uint32_t>(d),
                    DtypeCode(x.dtype),
                    weight != nullptr ? DtypeCode(weight->dtype) : 0u,
                    bias != nullptr ? DtypeCode(bias->dtype) : 0u,
                    DtypeCode(out.dtype),
                    weight != nullptr ? 1u : 0u,
                    bias != nullptr ? 1u : 0u,
                    tg,
                    args.eps};
  Encoder e("vt_layer_norm");
  e.BindTensor(x, 0, "layer_norm: x");
  e.BindTensor(weight != nullptr ? *weight : x, 1, "layer_norm: weight");
  e.BindTensor(bias != nullptr ? *bias : x, 2, "layer_norm: bias");
  e.BindTensor(out, 3, "layer_norm: out");
  e.BindBytes(&p, sizeof(p), 4);
  e.DispatchRows(rows, tg);
}

// cpu_ops.cpp:1649-1702 FusedChainInterpKernel — the Tier-1 interpreter. ONE
// registration; every Tier-1-able recipe in include/vt/recipes.h realizes
// through it, and every non-Tier-1 recipe realizes through the device-agnostic
// Tier-0 composite in src/vt/ops.cpp, which re-enters this backend's standalone
// ops. That is the whole "2 lines -> all 10 recipes" property the spike claims.
void FusedChainKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& weight,
                      Tensor* residual, const FusedRecipe& r, float eps) {
  const int64_t t = x.shape[0], h = x.shape[1];
  VT_CHECK(r.n >= 1 && r.n <= kMaxFusedSteps, "metal fused_chain: bad step count");

  std::vector<FStepGpu> steps(static_cast<size_t>(r.n));
  for (int s = 0; s < r.n; ++s) {
    const FStep& st = r.steps[s];
    uint32_t op = 0;
    switch (st.op) {
      case FOp::kAdd: op = 0; break;
      case FOp::kMul: op = 1; break;
      case FOp::kSilu: op = 2; break;
      case FOp::kSigmoid: op = 3; break;
      case FOp::kRmsNorm:
        // Mirrors the CPU interpreter's assertion (cpu_ops.cpp:1674): the shader
        // hard-codes the mean-square reduction, so any other kind must not reach it.
        VT_CHECK(st.reduce == FReduce::kMeanSquare,
                 "metal fused_chain: rmsnorm needs kMeanSquare");
        op = 4;
        break;
      default:
        VT_CHECK(false, "metal fused_chain: non-Tier-1 opcode reached the interpreter");
    }
    // Canonical operand indices (cpu_ops.cpp:1621-1643): 0=x 1=weight 2=residual
    // 3=out, with 2 and 3 the only writable slots.
    VT_CHECK(st.out == 2 || st.out == 3, "metal fused_chain: step writes a read-only operand");
    VT_CHECK(st.in[0] <= 3 && st.in[1] <= 3, "metal fused_chain: operand index out of range");
    VT_CHECK(residual != nullptr || (st.out != 2 && st.in[0] != 2 && st.in[1] != 2),
             "metal fused_chain: recipe touches the residual slot but none was bound");
    steps[static_cast<size_t>(s)] =
        FStepGpu{op, st.out, st.in[0], st.in[1], st.gemma ? 1u : 0u, 0u};
  }

  const uint32_t tg =
      ChooseThreadgroupSize(h, MetalContext::Get().PipelineMaxThreads("vt_fused_chain"));
  FcParams p{static_cast<uint32_t>(t),
             static_cast<uint32_t>(h),
             static_cast<uint32_t>(r.n),
             DtypeCode(x.dtype),
             DtypeCode(weight.dtype),
             residual != nullptr ? DtypeCode(residual->dtype) : 0u,
             DtypeCode(out.dtype),
             tg,
             eps};
  Encoder e("vt_fused_chain");
  e.BindTensor(x, 0, "fused_chain: x");
  e.BindTensor(weight, 1, "fused_chain: weight");
  e.BindTensor(residual != nullptr ? *residual : out, 2, "fused_chain: residual");
  e.BindTensor(out, 3, "fused_chain: out");
  e.BindBytes(steps.data(), steps.size() * sizeof(FStepGpu), 4);
  e.BindBytes(&p, sizeof(p), 5);
  e.DispatchRows(t, tg);
}

// ---------------------------------------------------------------------------
// M3a — the five ops OPT-125m needs beyond the W0 set.
// ---------------------------------------------------------------------------

// cpu_ops.cpp:531-543 EmbeddingKernel.
void EmbeddingKernel(Queue&, Tensor& out, const Tensor& table, const Tensor& ids) {
  const int64_t rows = ids.shape[0], h = table.shape[1], vocab = table.shape[0];
  VT_CHECK(ids.dtype == DType::kI32 || ids.dtype == DType::kI64,
           "metal embedding: ids must be i32 or i64");
  EmbedParams p{static_cast<uint32_t>(rows),
                static_cast<uint32_t>(h),
                static_cast<uint32_t>(vocab),
                ids.dtype == DType::kI64 ? 1u : 0u,
                DtypeCode(table.dtype),
                DtypeCode(out.dtype)};
  Encoder e("vt_embedding");
  e.BindTensor(table, 0, "embedding: table");
  e.BindTensor(ids, 1, "embedding: ids");
  e.BindTensor(out, 2, "embedding: out");
  e.BindBytes(&p, sizeof(p), 3);
  e.DispatchFlat(rows * h);
}

// cpu_ops.cpp:1529-1543 QkvSplitKernel.
void QkvSplitKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& qkv) {
  const int64_t t = qkv.shape[0];
  VT_CHECK(t > 0, "metal qkv_split: empty batch");
  const int64_t q_dim = q_out.Numel() / t, k_dim = k_out.Numel() / t, v_dim = v_out.Numel() / t;
  QkvSplitParams p{static_cast<uint32_t>(t),   static_cast<uint32_t>(q_dim),
                   static_cast<uint32_t>(k_dim), static_cast<uint32_t>(v_dim),
                   DtypeCode(qkv.dtype),       DtypeCode(q_out.dtype),
                   DtypeCode(k_out.dtype),     DtypeCode(v_out.dtype)};
  Encoder e("vt_qkv_split");
  e.BindTensor(qkv, 0, "qkv_split: qkv");
  e.BindTensor(q_out, 1, "qkv_split: q");
  e.BindTensor(k_out, 2, "qkv_split: k");
  e.BindTensor(v_out, 3, "qkv_split: v");
  e.BindBytes(&p, sizeof(p), 4);
  e.DispatchFlat(t * (q_dim + k_dim + v_dim));
}

// cpu_cache.cpp:33-72 ReshapeAndCacheKernel. Raw-element copy => BIT-EXACT.
void ReshapeAndCacheKernel(Queue&, const Tensor& k, const Tensor& v, Tensor& k_cache,
                           Tensor& v_cache, const Tensor& slot_mapping) {
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = k_cache.shape[1];
  const int64_t n_elems = k_cache.shape[2] * k_cache.shape[3];  // one NHD page
  const size_t esz = SizeOf(k.dtype);
  VT_CHECK(esz == 2 || esz == 4, "metal reshape_and_cache: 2- or 4-byte elements only");
  VT_CHECK(SizeOf(k_cache.dtype) == esz && SizeOf(v.dtype) == esz &&
               SizeOf(v_cache.dtype) == esz,
           "metal reshape_and_cache: k/v and their caches must share an element width");
  CacheParams p{static_cast<uint64_t>(k_cache.stride[0]),
                static_cast<uint64_t>(k_cache.stride[1]),
                static_cast<uint64_t>(v_cache.stride[0]),
                static_cast<uint64_t>(v_cache.stride[1]),
                static_cast<uint64_t>(k.stride[0]),
                static_cast<uint64_t>(v.stride[0]),
                static_cast<uint32_t>(num_slots),
                static_cast<uint32_t>(n_elems),
                static_cast<uint32_t>(block_size),
                static_cast<uint32_t>(esz)};
  Encoder e("vt_reshape_and_cache");
  e.BindTensor(k, 0, "reshape_and_cache: k");
  e.BindTensor(v, 1, "reshape_and_cache: v");
  e.BindTensor(k_cache, 2, "reshape_and_cache: k_cache");
  e.BindTensor(v_cache, 3, "reshape_and_cache: v_cache");
  e.BindTensor(slot_mapping, 4, "reshape_and_cache: slot_mapping");
  e.BindBytes(&p, sizeof(p), 5);
  e.DispatchFlat(num_slots * n_elems);
}

// cpu_paged_attn.cpp:51-131 PagedAttentionKernel, in the algebraically identical
// ONLINE-softmax form (see metal_msl.h vt_paged_attention).
void PagedAttentionKernel(Queue&, Tensor& out, const Tensor& query, const Tensor& k_cache,
                          const Tensor& v_cache, const Tensor& block_table,
                          const Tensor& seq_lens, const Tensor& query_start_loc,
                          const PagedAttentionArgs& args) {
  const int64_t num_reqs = seq_lens.shape[0];
  const int64_t tq = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t num_kv_heads = k_cache.shape[2];
  // The shader's threadgroup accumulator is a fixed VT_PA_MAXD-wide array; an
  // over-wide head must be a loud vt:: error, never a silent truncation.
  VT_CHECK(d <= 256, "metal paged_attention: head_size > 256 is not implemented "
                     "(the threadgroup accumulator is VT_PA_MAXD wide)");
  VT_CHECK(num_kv_heads > 0 && hq % num_kv_heads == 0,
           "metal paged_attention: q-heads must be a multiple of kv-heads");
  // Decode launches one threadgroup per (head, query token), which for a single
  // token is just `hq` threadgroups — 16 for a Qwen3-1.7B-class model, far too
  // few to fill the GPU. Sizing the group by `d` alone caps it at 128 threads.
  // The score loop now scales with SIMDGROUP count (one simdgroup per key), so
  // a wider group buys real parallelism there even though the V accumulation is
  // still bounded by `d`. Measured experiment, not an assumption.
  // Any request with more than one query token is a PREFILL, which gets the
  // query-tiled pipeline: it serves VT_PA_QTILE consecutive tokens from one
  // threadgroup so each K row and V element is read once for the whole tile
  // instead of once per token. Decode keeps the untiled pipeline — it has one
  // token per request, so there is nothing to reuse and tiling would only cost
  // it grid parallelism and threadgroup memory.
  const bool tiled = tq > num_reqs;
  // PREFILL on the matrix units. vt_paged_attention scores with simd_sum dot
  // products and accumulates V with scalar FMAs, measured at 108 GFLOP/s against
  // the GEMM's ~2250 on the same device in the same forward. The mma kernel needs
  // exactly 8 simdgroups (its 4x2 simdgroup map) and a head that fits its tiles.
  // VT_METAL_NO_ATTN_MMA=1 routes back to the scalar kernel for a same-binary A/B.
  const uint32_t kMmaTg = 256;
  static const bool mma_off = [] {
    const char* e = std::getenv("VT_METAL_NO_ATTN_MMA");
    return e != nullptr && e[0] == '1';
  }();
  // Q and K are staged as bfloat for the mma, which is lossless ONLY if they are
  // already bf16; an f32 query would be silently truncated.
  const bool bf_in = query.dtype == DType::kBF16 && k_cache.dtype == DType::kBF16 &&
                     v_cache.dtype == DType::kBF16;
  const bool mma = tiled && !mma_off && bf_in && d <= 128 && (d % 8) == 0 &&
                   MetalContext::Get().PipelineMaxThreads("vt_paged_attention_mma") >= kMmaTg;
  const char* kname = mma ? "vt_paged_attention_mma"
                          : (tiled ? "vt_paged_attention_tiled" : "vt_paged_attention");
  const uint32_t tg =
      mma ? kMmaTg
          : ChooseThreadgroupSize(d * 4, MetalContext::Get().PipelineMaxThreads(kname));
  PagedAttnParams p{
      static_cast<uint64_t>(k_cache.stride[0]), static_cast<uint64_t>(k_cache.stride[1]),
      static_cast<uint64_t>(k_cache.stride[2]), static_cast<uint64_t>(v_cache.stride[0]),
      static_cast<uint64_t>(v_cache.stride[1]), static_cast<uint64_t>(v_cache.stride[2]),
      static_cast<uint32_t>(num_reqs),          static_cast<uint32_t>(hq),
      static_cast<uint32_t>(d),                 static_cast<uint32_t>(hq / num_kv_heads),
      static_cast<uint32_t>(k_cache.shape[1]),  args.causal ? 1u : 0u,
      tg,
      args.window_size.has_value() ? static_cast<int32_t>(args.window_size->left) : -1,
      args.window_size.has_value() ? static_cast<int32_t>(args.window_size->right) : -1,
      static_cast<int32_t>(block_table.stride[0]),
      static_cast<int32_t>(block_table.stride[1]),
      DtypeCode(query.dtype),                   DtypeCode(k_cache.dtype),
      DtypeCode(v_cache.dtype),                 DtypeCode(out.dtype),
      args.scale};
  Encoder e(kname);
  e.BindTensor(query, 0, "paged_attention: query");
  e.BindTensor(k_cache, 1, "paged_attention: k_cache");
  e.BindTensor(v_cache, 2, "paged_attention: v_cache");
  e.BindTensor(block_table, 3, "paged_attention: block_table");
  e.BindTensor(seq_lens, 4, "paged_attention: seq_lens");
  e.BindTensor(query_start_loc, 5, "paged_attention: query_start_loc");
  e.BindTensor(out, 6, "paged_attention: out");
  e.BindBytes(&p, sizeof(p), 7);
  // One threadgroup per (q-head, query token). Both extents are FULL
  // threadgroups, so dispatchThreadgroups (not dispatchThreads).
  e.DispatchGrid2D(hq, tq, tg);
}

// Fused Qwen3-dense attention preamble: RMSNorm(q) + RMSNorm(k) + partial NeoX
// RoPE in ONE dispatch, replacing three. See vt_attn_qk_norm_rope in metal_msl.h
// for why this is a launch-count win rather than a bandwidth one.
void AttnQkNormRopeKernel(Queue&, Tensor& q3, Tensor& k3, const Tensor& q_norm,
                          const Tensor& k_norm, const Tensor& cos_sin, const Tensor& positions,
                          const RmsNormArgs& na, const RopeArgs& ra) {
  VT_CHECK(q3.rank == 3 && k3.rank == 3, "metal qk_norm_rope: q/k must be [T,H,Dh]");
  const int64_t T = q3.shape[0], Hq = q3.shape[1], Dh = q3.shape[2];
  const int64_t Hk = k3.shape[1];
  VT_CHECK(k3.shape[0] == T && k3.shape[2] == Dh, "metal qk_norm_rope: k shape mismatch");
  const uint32_t rot = static_cast<uint32_t>(ra.rotary_dim > 0 ? ra.rotary_dim : Dh);
  VT_CHECK((rot & 1u) == 0u && rot <= Dh, "metal qk_norm_rope: bad rotary_dim");
  // 256 threads = 8 simdgroups, each owning one (token, head). The first version
  // sized the group by head_dim and launched one group per pair — 12288 of them
  // for a 512-token prefill, each paying a 7-step barrier reduction for 128
  // elements.
  const uint32_t tg = std::min<uint32_t>(
      256u, MetalContext::Get().PipelineMaxThreads("vt_attn_qk_norm_rope"));
  const uint32_t sgs = tg / 32u;
  QkNormRopeParams p{
      static_cast<uint64_t>(q3.stride[0]), static_cast<uint64_t>(q3.stride[1]),
      static_cast<uint64_t>(k3.stride[0]), static_cast<uint64_t>(k3.stride[1]),
      static_cast<uint32_t>(T),  static_cast<uint32_t>(Hq),
      static_cast<uint32_t>(Hk), static_cast<uint32_t>(Dh),
      rot, rot / 2u, 1u,
      positions.dtype == DType::kI64 ? 1u : 0u,
      DtypeCode(q3.dtype),     DtypeCode(k3.dtype),
      DtypeCode(q_norm.dtype), DtypeCode(k_norm.dtype),
      DtypeCode(cos_sin.dtype), na.gemma ? 1u : 0u, tg, 0u,
      na.eps, 0.0f};
  Encoder e("vt_attn_qk_norm_rope");
  e.BindTensor(q3, 0, "qk_norm_rope: q");
  e.BindTensor(k3, 1, "qk_norm_rope: k");
  e.BindTensor(q_norm, 2, "qk_norm_rope: q_norm");
  e.BindTensor(k_norm, 3, "qk_norm_rope: k_norm");
  e.BindTensor(cos_sin, 4, "qk_norm_rope: cos_sin");
  e.BindTensor(positions, 5, "qk_norm_rope: positions");
  e.BindBytes(&p, sizeof(p), 6);
  // One SIMDGROUP per (token, head), `sgs` pairs per threadgroup.
  const uint32_t pairs = static_cast<uint32_t>(T * (Hq + Hk));
  e.DispatchGrid2D((pairs + sgs - 1u) / sgs, 1u, tg);
}

// cpu_sample.cpp:40-57 GreedyArgmaxKernel — one threadgroup per logits row.
void GreedyArgmaxKernel(Queue&, Tensor& token_ids, const Tensor& logits) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  const uint32_t tg =
      ChooseThreadgroupSize(v, MetalContext::Get().PipelineMaxThreads("vt_greedy_argmax"));
  ArgmaxParams p{static_cast<uint32_t>(n), static_cast<uint32_t>(v), tg};
  Encoder e("vt_greedy_argmax");
  e.BindTensor(logits, 0, "greedy_argmax: logits");
  e.BindTensor(token_ids, 1, "greedy_argmax: token_ids");
  e.BindBytes(&p, sizeof(p), 2);
  e.DispatchRows(n, tg);
}

// cpu_ops.cpp:652-665 RopeNeoxKernel — the Qwen3-dense default rotation (M3b).
// In-place NeoX RoPE over q [T,Hq,D] and k [T,Hk,D]; one GPU thread per
// (token,head,pair). See metal_msl.h vt_rope_neox for the f32-transcendental
// numerics note (NMSE bar, not bit-exact).
void RopeNeoxKernel(Queue&, Tensor& qs, Tensor& ks, const Tensor& pos, const RopeArgs& args) {
  const int64_t t = qs.shape[0], hq = qs.shape[1], hk = ks.shape[1], d = qs.shape[2];
  const int rot = args.rotary_dim;
  const int64_t half = rot / 2;
  RopeParams p{static_cast<uint32_t>(t),   static_cast<uint32_t>(hq),
               static_cast<uint32_t>(hk),  static_cast<uint32_t>(d),
               static_cast<uint32_t>(rot), static_cast<uint32_t>(half),
               DtypeCode(qs.dtype),        DtypeCode(ks.dtype),
               pos.dtype == DType::kI64 ? 1u : 0u, args.base};
  Encoder e("vt_rope_neox");
  e.BindTensor(qs, 0, "rope_neox: q");
  e.BindTensor(ks, 1, "rope_neox: k");
  e.BindTensor(pos, 2, "rope_neox: positions");
  e.BindBytes(&p, sizeof(p), 3);
  e.DispatchFlat((t * hq + t * hk) * half);
}

// cpu_ops.cpp:751-768 RopeCosSinCacheKernel — fill cos_sin[T,rot] (M3b). One GPU
// thread per (token,pair). Same numerics note as vt_rope_neox.
void RopeCosSinCacheKernel(Queue&, Tensor& cos_sin, const Tensor& positions,
                           const RopeArgs& args) {
  const int64_t t = cos_sin.shape[0];
  const int rot = args.rotary_dim;
  const int64_t half = rot / 2;
  RopeCacheParams p{static_cast<uint32_t>(t),   static_cast<uint32_t>(rot),
                    static_cast<uint32_t>(half), DtypeCode(cos_sin.dtype),
                    positions.dtype == DType::kI64 ? 1u : 0u, args.base};
  Encoder e("vt_rope_cos_sin_cache");
  e.BindTensor(cos_sin, 0, "rope_cos_sin_cache: cos_sin");
  e.BindTensor(positions, 1, "rope_cos_sin_cache: positions");
  e.BindBytes(&p, sizeof(p), 2);
  e.DispatchFlat(t * half);
}

// cpu_ops.cpp:690-742 RopeFromCacheKernel — the Qwen3-dense DEFAULT rotation
// (VT_QWEN3_ROPE_CACHE defaults ON). Applies cos|sin READ from the cache; no
// in-kernel transcendentals, so it is bit-exact to the CPU oracle. See metal_msl.h
// vt_rope_from_cache. MRoPE (rank-2 positions) is not yet ported to Metal.
void RopeFromCacheKernel(Queue&, Tensor& qs, Tensor* ks, const Tensor& positions,
                         const Tensor& cache, const RopeArgs& args) {
  VT_CHECK(positions.rank == 1,
           "metal rope_from_cache: MRoPE (rank-2 positions) is not supported on "
           "Metal yet — Qwen3-dense uses rank-1 positions");
  const int64_t t = qs.shape[0], hq = qs.shape[1];
  const int64_t hk = ks != nullptr ? ks->shape[1] : 0;
  const int rot = args.rotary_dim;
  const int64_t half = rot / 2;
  RopeApplyParams p{
      static_cast<uint64_t>(qs.stride[0]), static_cast<uint64_t>(qs.stride[1]),
      ks != nullptr ? static_cast<uint64_t>(ks->stride[0]) : 0,
      ks != nullptr ? static_cast<uint64_t>(ks->stride[1]) : 0,
      static_cast<uint32_t>(t),   static_cast<uint32_t>(hq),
      static_cast<uint32_t>(hk),  static_cast<uint32_t>(rot),
      static_cast<uint32_t>(half), args.is_neox_style ? 1u : 0u,
      DtypeCode(qs.dtype),        ks != nullptr ? DtypeCode(ks->dtype) : DtypeCode(qs.dtype),
      DtypeCode(cache.dtype),     positions.dtype == DType::kI64 ? 1u : 0u,
      ks != nullptr ? 1u : 0u,    0u};
  Encoder e("vt_rope_from_cache");
  e.BindTensor(qs, 0, "rope_from_cache: q");
  e.BindTensor(ks != nullptr ? *ks : qs, 1, "rope_from_cache: k");
  e.BindTensor(positions, 2, "rope_from_cache: positions");
  e.BindTensor(cache, 3, "rope_from_cache: cos_sin cache");
  e.BindBytes(&p, sizeof(p), 4);
  e.DispatchFlat((t * hq + t * hk) * half);
}

struct Registrar {
  Registrar() {
    // Same guard as the backend registrar: a Metal-enabled build on a device-less
    // host registers nothing, so GetOp throws its normal not-registered error.
    if (!MetalContext::Available()) return;
    // static_cast against the ops.h aliases ties every kernel signature to the
    // registration contract at COMPILE time (the cpu_ops.cpp idiom).
    RegisterOp(OpId::kMatmul, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulKernel)));
    RegisterOp(OpId::kMatmulBT, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTKernel)));
    RegisterOp(OpId::kAdd, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<AddFn>(&AddKernel)));
    RegisterOp(OpId::kRelu, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<ReluFn>(&ReluKernel)));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernel)));
    RegisterOp(OpId::kCastBf16, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<CastBf16Fn>(&CastKernel)));
    RegisterOp(OpId::kCastF32, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<CastF32Fn>(&CastKernel)));
    RegisterOp(OpId::kLayerNorm, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<LayerNormFn>(&LayerNormKernel)));
    RegisterOp(OpId::kRmsNorm, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernel)));
    RegisterOp(OpId::kFusedChain, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<FusedChainFn>(&FusedChainKernel)));
    // --- M3a: the OPT-125m set.
    RegisterOp(OpId::kEmbedding, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingKernel)));
    RegisterOp(OpId::kQkvSplit, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<QkvSplitFn>(&QkvSplitKernel)));
    RegisterOp(OpId::kReshapeAndCache, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<ReshapeAndCacheFn>(&ReshapeAndCacheKernel)));
    RegisterOp(OpId::kPagedAttention, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<PagedAttentionFn>(&PagedAttentionKernel)));
    RegisterOp(OpId::kGreedyArgmax, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<GreedyArgmaxFn>(&GreedyArgmaxKernel)));
    // --- M3b: the two ops Qwen3-dense needs beyond OPT's set. The DEFAULT
    // (deterministic) dense attention path dispatches kRopeNeox (the rotation)
    // plus kRopeCosSinCache (built once per step by the attention preamble).
    RegisterOp(OpId::kRopeNeox, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<RopeFn>(&RopeNeoxKernel)));
    RegisterOp(OpId::kRopeCosSinCache, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<RopeCosSinCacheFn>(&RopeCosSinCacheKernel)));
    RegisterOp(OpId::kAttnQkNormRope, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<AttnQkNormRopeFn>(&AttnQkNormRopeKernel)));
    RegisterOp(OpId::kRopeFromCache, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<RopeFromCacheFn>(&RopeFromCacheKernel)));
  }
} registrar;

}  // namespace

// Test-only strided-read bandwidth probe; see vt_bw_probe in metal_msl.h. Lives
// outside the anonymous namespace so the Metal test can link it.
void BandwidthProbe(Queue&, void* src, void* out, uint32_t n_chunks, uint32_t chunk_f4,
                    uint32_t stride_f4) {
  // 256 threads walking 64 chunks each: enough work per threadgroup that this
  // measures memory, not dispatch.
  const uint32_t tg = std::min<uint32_t>(
      256u, MetalContext::Get().PipelineMaxThreads("vt_bw_probe"));
  const uint32_t chunks_per_tg = 64u;
  const uint32_t groups = (n_chunks + chunks_per_tg - 1) / chunks_per_tg;
  BwParams p{n_chunks, chunk_f4, stride_f4, tg, chunks_per_tg, 0u};
  Encoder e("vt_bw_probe");
  const Device dev{DeviceType::kMETAL, 0};
  Tensor s = Tensor::Contiguous(src, DType::kF32, dev, {1});
  Tensor o = Tensor::Contiguous(out, DType::kF32, dev, {1});
  e.BindTensor(s, 0, "bw_probe: src");
  e.BindTensor(o, 1, "bw_probe: out");
  e.BindBytes(&p, sizeof(p), 2);
  e.DispatchGrid2D(groups, 1, tg);
}

}  // namespace vt::metal

// LTX-2.5 CONV VIDEO VAE — ConvVideoDecoder (conv_video_decoder.py) ported 1:1
// from upstream `ltx_core` and gated against it by
// scripts/gen-ltx2-vae-goldens.py, which EXECUTES the upstream module at reduced
// dimensions on CPU.
//
// ─── SCOPE, so nothing is discovered later ───────────────────────────────────
//  * The DIFFUSION decoder (`NADiffusionDecoder` / `DiffusionVideoDecoder`) is
//    NOT ported. Ltx2VideoDecode refuses it BY NAME and never falls back — see
//    the header, and .agents/specs/ltx-2-5.md section 0 item 2.
//  * `attn_res_x` is refused too, for a different reason: at this upstream
//    revision the block cannot be CONSTRUCTED, because `_make_decoder_block`
//    passes `attention_head_dim` to `UNetMidBlock3D`, whose __init__ does not
//    accept it (conv_video_decoder.py:85-96 vs resnet.py:210-222). Upstream
//    raises TypeError; this raises with the same reason named.
//  * `dims == 2` / `dims == (2, 1)` (Conv2d and DualConv3d, convolution.py:27-71)
//    are not ported: the decoder is built with `convolution_dimensions=3`.
//  * The ENCODER half is out of this phase and owed.
//  * The TILED decode path (`tiled_decode`, conv_video_decoder.py:383-484) is NO
//    LONGER owed: it landed in row `LTX25-TILED-DECODE` (#644) and lives next
//    door in ltx2_video_vae_tiled.cpp, reached through `Ltx2VideoDecodeStreaming`.
//    This line named it as debt, and the row's spec cited this exact anchor as
//    the debt it pays, so leaving the two disagreeing was the record contradicting
//    the tree.
//
// ─── DTYPE: THIS IS THE CPU REFERENCE ARM, AND f32 IS NOT WHAT SHIPS ─────────
// Every buffer below is f32, and unlike the audio VAE next door that is NOT an
// upstream-grounded choice — it is the choice a reference arm makes, and it is
// annotated here because AGENTS.md requires an f32 on a model path to carry a
// reason, and because a too-WIDE dtype is the one defect a correctness gate
// structurally cannot report: it stays numerically right, the goldens stay green,
// and the only symptom is twice the bytes moved.
//
// Upstream does the OPPOSITE of what the audio VAE does. `ConvVideoDecoder.forward`
// runs in the CHECKPOINT's dtype: it casts in with `sample.to(weights_dtype)` on
// entry and back with `sample.to(output_dtype)` on exit
// (conv_video_decoder.py:283-286, 355-356). There is no autocast, no float32
// pin, and no spectral-metric argument of the kind that justifies the audio
// tower's f32 (ltx2_audio_vae.cpp:7-12 -> vocoder.py:585-595). So f32 here is
// the reference arm's convention and nothing more.
//
// The golden CANNOT catch this either, and that is worth stating plainly rather
// than leaving for someone to discover: the generator's `fill_from_stream` casts
// every upstream parameter to f32, so the oracle itself runs f32 and a dtype
// comparison against it is vacuous by construction.
//
// ─── THE ARITHMETIC IS f32 TOO, AND USED NOT TO BE (#1008) ───────────────────
// Storage being f32 says nothing about the width the arithmetic runs at, and
// until #1008 this file accumulated every convolution, GEMM, norm and softmax in
// `double` — a width no reference uses anywhere on this path. Upstream's ops are
// plain `nn.Conv3d` / `nn.Conv2d` / `F.normalize` / SDPA, which accumulate in the
// tensor dtype. That was MEASURED rather than assumed: on a reduction engineered
// so the widths separate, `F.conv3d` returns 0.0 for f32 AND for bf16 tensors
// while an f64 accumulator returns 2.5. The case
// "the decode's convolution accumulates in f32" in tests/vllm/models/
// test_ltx2_vae.cpp is that instrument, and it is the only gate here that can
// see the width — for the reason the paragraph above gives.
//
// What deliberately stays f64, each annotated at its site: the pinned config
// epsilons, the once-per-block scalars `sqrt(C)` and `1/sqrt(C)`, and the
// TimestepEmbedding frequency table, which is a constant precompute rather than
// a data path.
//
// ─── AND IT IS PARALLEL, WHICH IT ALSO USED NOT TO BE (#1009) ────────────────
// The convolutions dispatch through `vt::cpu::ParallelForRows`, the synchronous
// row-chunked parallel-for 10+ CPU kernels in this tree already use and that no
// line of this file used before. The partition is over OUTPUT lines only: the
// whole `ci * kernel^3` reduction stays inside one output element's body, so the
// blocked f32 order above is untouched and the result is bit-identical at any
// thread count and under any work-stealing assignment. Splitting the reduction
// axis `ic` instead would make the summation order a function of the thread
// count; it is rejected at the site. "the decode DISPATCHES its convolutions to
// the CPU threadpool" and "the decode is BIT-IDENTICAL across thread counts" in
// tests/vllm/models/test_ltx2_vae.cpp are the two instruments.
//
// PHASE L6 OWES THE PRODUCTION ARM — the bf16/NVFP4 decode that inherits the
// checkpoint dtype the way upstream does. Until it lands, this file is a
// correctness reference, not the shipping path, and no memory or throughput
// number should be taken from it.
#include "vllm/model_executor/models/ltx2_video_vae.h"
#include "vllm/model_executor/models/ltx2_kernels.h"
#include "vllm/model_executor/models/ltx2_video_vae_kernels.h"

#include "vllm/model_executor/models/ltx2_video_vae_encoder.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/minimax_h3.h"
#include "vt/backend.h"
#include "vt/cpu/cpu_threadpool.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm {

namespace {

// ─── RESIDENT VOLUME STORAGE (LTX25-VAE-DEVICE-RESIDENCY, #1451) ────────────
//
// One volume's bytes, ON THE QUEUE'S DEVICE. Before this existed the decode held
// every intermediate in a host `std::vector<float>`, so W5's convolution seam had
// to upload its input and download its output on EVERY `nn.Conv3d` call and the
// nine stages between the convolutions ran on the host in between. Upstream never
// moves the tensor back (conv_video_decoder.py's forward contains no `.cpu()` at
// all), so that was a divergence in memory behaviour and not only a cost.
//
// TWO BACKINGS, ONE INTERFACE, AND THE CPU ONE COPIES NOTHING. On the CPU queue
// the bytes live in a `std::vector<float>` and `ptr()` is a view over it, so the
// host arm moves no byte it did not move before this type existed -- which is
// what keeps the byte-identity claim also a no-regression claim, the same
// property `Conv3dThroughSeam` argued for its own CPU arm. On a real device the
// bytes are a backend allocation that lives as long as the volume does.
//
// THERE IS NO `operator[]` AND NO `begin()`, DELIBERATELY. A device pointer is
// not dereferenceable on the host, and on a UNIFIED-memory backend -- which is
// exactly what the `FakeXpuBackend` in tests/vllm/multimodal/test_diffusion_device_seam.cpp
// is -- a host dereference of one WORKS, silently, and no test on a box without
// a discrete GPU could see it. Removing the operators makes every such site a
// compile error instead: a caller that genuinely needs host bytes says `Host()`
// and gets a check that the volume is where it claims to be.
class VaeStore {
 public:
  VaeStore() = default;
  ~VaeStore() { Release(); }

  VaeStore(const VaeStore& other) { CopyFrom(other); }
  VaeStore& operator=(const VaeStore& other) {
    if (this != &other) {
      Release();
      CopyFrom(other);
    }
    return *this;
  }
  VaeStore(VaeStore&& other) noexcept { Steal(other); }
  VaeStore& operator=(VaeStore&& other) noexcept {
    if (this != &other) {
      Release();
      Steal(other);
    }
    return *this;
  }

  // `queue == nullptr` means the CPU queue, NOT "the old host path" -- the rule
  // this file has applied since W5.
  void Alloc(vt::Queue* queue, size_t n) {
    Release();
    n_ = n;
    queue_ = queue;
    if (!OnDevice()) {
      host_.assign(n, 0.0f);
      return;
    }
    backend_ = vt::TryGetBackend(queue_->device);
    VT_CHECK(backend_ != nullptr,
             "ltx2 video vae: the decode was handed a queue on a device with no registered "
             "backend, so the volume cannot be made resident on it");
    dev_ = backend_->Alloc(n_ == 0 ? 1 : n_ * sizeof(float));
  }

  // Allocate n floats ON THE SAME QUEUE as `other`. Almost every volume in this
  // file is derived from another one, and saying `Like(in.data, n)` cannot pick
  // the wrong queue the way re-deriving it at each site could -- a mismatch
  // there would put an output on the host while its input sits on a device, and
  // on a unified-memory backend that runs and produces the right pixels.
  void Like(const VaeStore& other, size_t n) { Alloc(other.queue_, n); }

  bool OnDevice() const {
    return queue_ != nullptr && queue_->device.type != vt::DeviceType::kCPU;
  }
  size_t size() const { return n_; }
  vt::Queue* queue() const { return queue_; }

  float* ptr() { return OnDevice() ? static_cast<float*>(dev_) : host_.data(); }
  const float* ptr() const {
    return OnDevice() ? static_cast<const float*>(dev_) : host_.data();
  }

  // Host bytes, with a check rather than a comment. Every caller of this is a
  // stage that has NOT been ported to a device arm, and each one names itself.
  std::vector<float>::iterator HostBegin() {
    VT_CHECK(!OnDevice(),
             "ltx2 video vae: a host loop asked for the bytes of a volume that is resident on a "
             "device -- the caller must download it first, or be ported to a device arm (#1451)");
    return host_.begin();
  }
  float* Host() {
    VT_CHECK(!OnDevice(),
             "ltx2 video vae: a host loop asked for the bytes of a volume that is resident on a "
             "device -- the caller must download it first, or be ported to a device arm (#1451)");
    return host_.data();
  }
  const float* Host() const {
    VT_CHECK(!OnDevice(),
             "ltx2 video vae: a host loop asked for the bytes of a volume that is resident on a "
             "device -- the caller must download it first, or be ported to a device arm (#1451)");
    return host_.data();
  }

  void Upload(const float* host) {
    if (n_ == 0) return;
    if (!OnDevice()) {
      std::copy(host, host + n_, host_.begin());
      return;
    }
    backend_->Copy(*queue_, dev_, host, n_ * sizeof(float));
  }
  void Download(float* host) const {
    if (n_ == 0) return;
    if (!OnDevice()) {
      std::copy(host_.begin(), host_.end(), host);
      return;
    }
    backend_->Copy(*queue_, host, dev_, n_ * sizeof(float));
    backend_->Synchronize(*queue_);
  }

 private:
  void Release() {
    if (dev_ != nullptr && backend_ != nullptr) backend_->Free(dev_);
    dev_ = nullptr;
    backend_ = nullptr;
    host_.clear();
    n_ = 0;
    queue_ = nullptr;
  }
  void CopyFrom(const VaeStore& other) {
    Alloc(other.queue_, other.n_);
    if (n_ == 0) return;
    if (!OnDevice()) {
      host_ = other.host_;
      return;
    }
    // DEVICE TO DEVICE. A copy that went through the host would be exactly the
    // round-trip this row removes, and `Volume hidden = input;` in
    // `ResnetBlock3d` is a copy on the hot path.
    backend_->Copy(*queue_, dev_, other.dev_, n_ * sizeof(float));
  }
  void Steal(VaeStore& other) {
    queue_ = other.queue_;
    backend_ = other.backend_;
    dev_ = other.dev_;
    host_ = std::move(other.host_);
    n_ = other.n_;
    other.dev_ = nullptr;
    other.backend_ = nullptr;
    other.n_ = 0;
    other.queue_ = nullptr;
  }

  vt::Queue* queue_ = nullptr;
  vt::Backend* backend_ = nullptr;
  void* dev_ = nullptr;
  std::vector<float> host_;
  size_t n_ = 0;
};

// ─── THE WEIGHTS, STAGED ONCE (LTX25-VAE-DEVICE-RESIDENCY, #1451) ───────────
//
// W5 uploaded a convolution's weight AND its bias on every call, so a decode
// re-sent the same bytes once per `nn.Conv3d` and once per tile and once per
// temporal group. Upstream stages the decoder's parameters onto the device at
// BUILD time and never moves them again (single_gpu_model_builder.py:273), and
// vLLM-Omni @ a4ea67a21 states the contract as "VAE(s) (always on GPU)"
// (vllm_omni/diffusion/models/interface.py:92).
//
// KEYED ON THE HOST POINTER, not on the parameter name. `Ltx2VaeWeights::Get`
// returns a reference into a map this class does not own and does not outlive,
// so the address of the first element identifies the tensor exactly, and it does
// so without this cache having to know the naming scheme of the decoder, the
// encoder or the upsampler. A name key would also be wrong for the same tensor
// reached under two prefixes.
//
// ITS LIFETIME IS THE DECODE. That is deliberate and it is a LIMIT, stated here
// rather than in a commit nobody re-reads: two decode calls stage the weights
// twice, so a tiled render pays it once per tile. Hoisting the cache to the
// engine's load, which is where upstream puts it, needs an owner on
// `Ltx2VideoEngine` and is owed -- see `## Owed` in
// .agents/specs/ltx25-vae-device-residency.md. Within one decode, which is what
// this row's gate measures, each weight is uploaded exactly once.
class VaeWeightCache {
 public:
  explicit VaeWeightCache(vt::Queue* queue) : queue_(queue) {
    if (queue_ != nullptr && queue_->device.type != vt::DeviceType::kCPU) {
      backend_ = vt::TryGetBackend(queue_->device);
      VT_CHECK(backend_ != nullptr,
               "ltx2 video vae: the decode was handed a queue on a device with no registered "
               "backend, so its weights cannot be staged onto it");
    }
  }
  ~VaeWeightCache() {
    if (backend_ == nullptr) return;
    for (const auto& kv : staged_) backend_->Free(kv.second);
  }
  VaeWeightCache(const VaeWeightCache&) = delete;
  VaeWeightCache& operator=(const VaeWeightCache&) = delete;

  // On the CPU queue this is the host pointer itself and NOTHING is copied,
  // which is what keeps the host arm byte-for-byte the cost it was.
  const float* Get(const std::vector<float>& host) {
    if (backend_ == nullptr) return host.data();
    auto it = staged_.find(host.data());
    if (it != staged_.end()) return static_cast<const float*>(it->second);
    const size_t bytes = host.size() * sizeof(float);
    void* dev = backend_->Alloc(bytes == 0 ? 1 : bytes);
    if (bytes > 0) backend_->Copy(*queue_, dev, host.data(), bytes);
    staged_.emplace(host.data(), dev);
    return static_cast<const float*>(dev);
  }

 private:
  vt::Queue* queue_ = nullptr;
  vt::Backend* backend_ = nullptr;
  std::map<const float*, void*> staged_;
};

// A PER-CALL host buffer, on the queue's device. Not everything a kernel reads
// is a weight: the spatial-noise plane is drawn fresh per block by
// `Ltx2NoiseStream`, and the timestep embedding is computed per block on the
// host. Those cannot go through `VaeWeightCache`, which is keyed on a host
// address that must outlive the decode -- a local vector's address is reused by
// the next local vector, and caching on it would hand a later block the earlier
// block's bytes.
//
// On the CPU queue this is the host pointer itself and nothing is copied.
class VaeScratch {
 public:
  VaeScratch(vt::Queue* queue, const std::vector<float>& host) {
    if (queue == nullptr || queue->device.type == vt::DeviceType::kCPU) {
      ptr_ = host.data();
      return;
    }
    store_.Alloc(queue, host.size());
    store_.Upload(host.data());
    ptr_ = store_.ptr();
  }
  VaeScratch(const VaeScratch&) = delete;
  VaeScratch& operator=(const VaeScratch&) = delete;
  const float* ptr() const { return ptr_; }

 private:
  VaeStore store_;
  const float* ptr_ = nullptr;
};

// A [C, T, H, W] volume at batch 1.
struct Volume {
  VaeStore data;
  int64_t channels = 0, t = 0, h = 0, w = 0;

  int64_t spatial() const { return t * h * w; }
  size_t At(int64_t c, int64_t ti, int64_t hi, int64_t wi) const {
    return static_cast<size_t>(((c * t + ti) * h + hi) * w + wi);
  }
};

// `ReflectIndex` and `SpatialIndex` USED TO LIVE HERE. They were the pad
// gather's index arithmetic, and they moved with the gather into the kLtx2Vae
// CPU arm (src/vt/cpu/cpu_ltx2_vae.cpp) when the pad stopped being a host loop
// (#1451). They are not duplicated: this file has no second copy, and the CUDA
// arm carries the same two functions so both devices reflect identically.

// ─── THE STAGE SEAM (LTX25-VAE-DEVICE-RESIDENCY, #1451) ─────────────────────
//
// W5 put the CONVOLUTION on the queue's device and left every stage between two
// convolutions as a host loop, so a non-CPU queue moved the whole volume back
// and forth around each one. Those loops now live in `vt::OpId::kLtx2Vae`
// (src/vt/cpu/cpu_ltx2_vae.cpp, src/vt/cuda/cuda_ltx2_vae.cu) and are reached
// ONLY through the two helpers below.
//
// ONE IMPLEMENTATION, TWO CALLERS, and that is the whole point. The decoder
// reaches a kernel with a device pointer on the queue it was given; the encoder
// — which is not ported to a resident volume in this wave — reaches the SAME
// kernel with a host pointer on the CPU queue. Neither is a transcription of the
// other, so the two cannot drift, and the committed goldens
// (tests/vllm/models/ltx2_vae_goldens.inc) gate both.
//
// The static_asserts pin `Ltx2PaddingMode` to the plain ints the kernel header
// carries. That header cannot include this model's headers — nvcc compiles it —
// so the two enumerations are declared twice and checked once, here.
static_assert(static_cast<int>(Ltx2PaddingMode::kZeros) == ltx2_vae::kLtx2VaePadZeros,
              "ltx2 vae pad: kZeros must agree with the kernel header");
static_assert(static_cast<int>(Ltx2PaddingMode::kReflect) == ltx2_vae::kLtx2VaePadReflect,
              "ltx2 vae pad: kReflect must agree with the kernel header");
static_assert(static_cast<int>(Ltx2PaddingMode::kReplicate) == ltx2_vae::kLtx2VaePadReplicate,
              "ltx2 vae pad: kReplicate must agree with the kernel header");

// `queue == nullptr` means the CPU queue, NOT "the old host path". There is one
// code path and the device is a property of the queue — the rule
// `Conv3dThroughSeam` already states below, applied to every other stage too.
vt::Queue VaeCpuQueue() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

const ltx2_vae::Ltx2VaeDeviceKernels& VaeKernels(const vt::Queue& q) {
  const ltx2_vae::Ltx2VaeDeviceKernels* k = ltx2_vae::Ltx2VaeDevice(q.device.type);
  VT_CHECK(k != nullptr,
           "ltx2 video vae: no kLtx2Vae kernel table is registered for this queue's device, so "
           "the decode's between-convolution stages cannot run on it");
  return *k;
}

// dst += src, elementwise, on the queue's device. This is `vt::Add`
// (include/vt/ops.h:2440) and NOT a new kernel: the shared surface already has
// an elementwise add with a CPU and a CUDA arm, and adding an eleventh entry to
// the VAE table for it would be the parallel path AGENTS.md forbids. It may
// alias in place, which is what both residual sites need.
void VaeAddInPlace(vt::Queue& q, float* dst, const float* src, int64_t n) {
  vt::Tensor d = vt::Tensor::Contiguous(dst, vt::DType::kF32, q.device, {n});
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(src), vt::DType::kF32, q.device, {n});
  vt::Add(q, d, d, a);
}

const ltx2::Ltx2DeviceKernels& VaeSiluKernels(const vt::Queue& q) {
  // The ungated SiLU is NOT duplicated into the VAE table. `vt::OpId::kLtx2`
  // already carries one (cpu_ltx2.cpp:188) and it is the same function
  // x / (1 + exp(-x)) the VAE's host `Silu` was; a second copy would be exactly
  // the parallel path AGENTS.md `## Shared seams` forbids.
  const ltx2::Ltx2DeviceKernels* k = ltx2::Ltx2Device(q.device.type);
  VT_CHECK(k != nullptr,
           "ltx2 video vae: no kLtx2 kernel table is registered for this queue's device, so the "
           "decode's SiLU cannot run on it");
  return *k;
}

// ─── THE DEVICE SEAM (LTX25-DEVICE-RESIDENCY W5, #1007) ──────────────────────
//
// `DevBuf` USED TO LIVE HERE. It was W5's per-call device allocation -- one
// `Alloc`, one `Copy` up, one `Copy` down and one `Free` for every operand of
// every convolution -- and it is gone with the round trips it served (#1451).
// The volume owns its own storage now (`VaeStore`) and the weights are staged
// once (`VaeWeightCache`), so nothing in this file allocates per call.
//
// Its removal also closes #1904 for this file by deletion rather than by
// migration: it was a hand-rolled second copy of `vllm::dense_attn::DBuf`
// (include/vllm/model_executor/models/dense_device_glue.h:109), and the right
// fix for a duplicated seam is to stop needing it. #1904 stays open for the
// audit it also asks for.


// The one convolution dispatch of the whole video VAE, decoder and encoder.
//
// `queue` is NULL for "the CPU queue", NOT for "the old host path": there is
// exactly one code path here and the device is a property of the queue. That is
// deliberate — a `queue != nullptr ? device : host` ternary would put the
// interesting branch where nothing in a CPU build can execute it, which is the
// shape #1426 already records for the DiT's device forward.
//
// Upstream decides the same thing the same way and never per call: the decoder
// is built onto a device once (`single_gpu_model_builder.py:267-288`, CUDA by
// default at `:273`) and the latent follows the weights
// (`conv_video_decoder.py:283-286`).
// The one convolution dispatch of the whole video VAE, decoder and encoder.
//
// `queue` is NULL for "the CPU queue", NOT for "the old host path": there is
// exactly one code path here and the device is a property of the queue.
//
// NOTHING IS UPLOADED OR DOWNLOADED HERE ANY MORE (#1451). W5 allocated four
// buffers per call, copied the input, the weight and the bias up and the output
// back down, and freed all four -- so the volume was on the host between every
// pair of convolutions and each weight was re-sent per call. Both operands now
// arrive already resident: `x` and `out` are views into the volume's own
// `VaeStore`, and `weight`/`bias` come from `VaeWeightCache`, which staged them
// once. Upstream never moves the tensor back
// (conv_video_decoder.py's forward contains no `.cpu()`), and this is what that
// looks like in this tree.
//
// THE USE-AFTER-FREE W5 GUARDED AGAINST IS GONE BY CONSTRUCTION rather than by
// scoping. W5's note here explained why its bias `DevBuf` had to outlive the
// dispatch: a `Free` between the kernel LAUNCH and the `Synchronize` frees
// memory a running kernel is still reading, and no test on a box without a GPU
// could see it. That whole class of hazard is removed by every operand
// outliving the decode instead of the call -- which is also what
// .agents/specs/ltx2-device-staged-view-uaf.md was opened for.
void Conv3dThroughSeam(vt::Queue* queue, const float* x, int64_t cin, int64_t tin, int64_t hin,
                       int64_t win, const float* weight, const float* bias, int64_t cout,
                       int64_t kernel, const vt::Conv3dArgs& args, float* out, int64_t tout,
                       int64_t hout, int64_t wout) {
  vt::Queue cpu_queue = VaeCpuQueue();
  vt::Queue& q = queue != nullptr ? *queue : cpu_queue;
  const int64_t wrows = cout * (cin / args.groups);
  vt::Tensor tx = vt::Tensor::Contiguous(const_cast<float*>(x), vt::DType::kF32, q.device,
                                         {cin, tin, hin, win});
  vt::Tensor tw = vt::Tensor::Contiguous(const_cast<float*>(weight), vt::DType::kF32, q.device,
                                         {wrows, kernel, kernel, kernel});
  vt::Tensor to = vt::Tensor::Contiguous(out, vt::DType::kF32, q.device, {cout, tout, hout, wout});
  vt::Tensor tb;
  if (bias != nullptr) {
    tb = vt::Tensor::Contiguous(const_cast<float*>(bias), vt::DType::kF32, q.device, {cout});
  }
  vt::Conv3d(q, to, tx, tw, bias != nullptr ? &tb : nullptr, args);
}

// CausalConv3d (convolution.py:266-317). Two things that are NOT interchangeable
// with MiniMax-H3's causal Conv3d:
//   * the temporal pad REPLICATES FRAME 0 `k_t - 1` times (H3 pads with zeros);
//   * the non-causal branch replicates the FIRST and LAST frame `(k_t - 1) / 2`
//     times each, so the output frame count is the same either way.
// Spatial padding is `k // 2` on each side in `spatial_padding_mode`.
//
// THE STRIDE IS APPLIED AFTER THE PAD, AND THE PAD DOES NOT KNOW ABOUT IT.
// `CausalConv3d.forward` concatenates `k_t - 1` copies of frame 0 and only then
// calls the strided `nn.Conv3d` (convolution.py:305-312), so a stride-2 temporal
// convolution still prepends TWO frames, not one. The video ENCODER is the only
// caller that passes a stride; every decoder call site keeps the defaults.
Volume CausalConv3d(vt::Queue* queue, VaeWeightCache* wcache, const Volume& in,
                    int64_t out_channels, int64_t kernel, bool causal, Ltx2PaddingMode mode,
                    const std::vector<float>& weight, const std::vector<float>* bias,
                    int64_t stride_t = 1, int64_t stride_h = 1, int64_t stride_w = 1) {
  const int64_t ci = in.channels;
  VT_CHECK(stride_t >= 1 && stride_h >= 1 && stride_w >= 1, "ltx2 conv3d: stride must be positive");
  VT_CHECK(static_cast<int64_t>(in.data.size()) == ci * in.spatial(),
           "ltx2 conv3d: input size does not match [C, T, H, W]");
  VT_CHECK(static_cast<int64_t>(weight.size()) == out_channels * ci * kernel * kernel * kernel,
           "ltx2 conv3d: weight size does not match the kernel");

  const int64_t pad_front = causal ? kernel - 1 : (kernel - 1) / 2;
  const int64_t pad_back = causal ? 0 : (kernel - 1) / 2;
  const int64_t pad_spatial = kernel / 2;
  const int64_t pt = in.t + pad_front + pad_back;
  const int64_t ph = in.h + 2 * pad_spatial;
  const int64_t pw = in.w + 2 * pad_spatial;

  // THE PAD IS A KERNEL NOW (#1451), not a host loop. It was the last piece of
  // the convolution stage still running on the host: W5 dispatched the reduction
  // and left the volume to be PADDED here, so even the convolution was not
  // resident. The gather is unchanged — one source element per destination
  // element, no reduction, so no arithmetic moved with it. The parallel
  // partition and the Amdahl argument (#1009) moved into the CPU arm with the
  // loop, src/vt/cpu/cpu_ltx2_vae.cpp.
  // THE PAD IS RESIDENT TOO. It lands in a buffer on the same queue as its
  // input, so the padded volume is never a host allocation on a device queue.
  VaeStore padded;
  padded.Like(in.data, static_cast<size_t>(ci * pt * ph * pw));
  vt::Queue conv_cpu = VaeCpuQueue();
  vt::Queue& cq = queue != nullptr ? *queue : conv_cpu;
  VaeKernels(cq).pad(cq, padded.ptr(), in.data.ptr(), ci, in.t, in.h, in.w, pad_front, pad_back,
                     pad_spatial, static_cast<int>(mode), vt::DType::kF32);

  Volume out;
  out.channels = out_channels;
  out.t = (pt - kernel) / stride_t + 1;
  out.h = (ph - kernel) / stride_h + 1;
  out.w = (pw - kernel) / stride_w + 1;
  VT_CHECK(pt >= kernel && ph >= kernel && pw >= kernel && out.t > 0 && out.h > 0 && out.w > 0,
           "ltx2 conv3d: empty output");
  out.data.Like(in.data, static_cast<size_t>(out_channels * out.spatial()));
  // THE REDUCTION IS DISPATCHED, NOT WRITTEN HERE (#1007, W5). It used to be a
  // `ParallelForRows` loop in this function, and the whole of what moved is
  // WHERE it runs: `vt::Conv3d` on the queue this decode was given, whose CPU
  // arm (src/vt/cpu/cpu_conv3d.cpp) is that loop transcribed with no arithmetic
  // change. The order is the op's published contract now — one f32 accumulator
  // per output element SEEDED WITH THE BIAS, then one f32 PARTIAL PER INPUT
  // CHANNEL swept (kt, kh, kw) — so this file's goldens are a real regression
  // gate on the MOVE rather than a re-baselined one: deleting the dispatch below
  // reds 12 of test_ltx2_vae's 44 cases, decoder and encoder alike.
  //
  // They are NOT the gate on the ORDER, and the difference matters. Mutating the
  // CPU kernel to a flat accumulator leaves test_ltx2_vae at 44/44 GREEN and
  // reds only tests/vt/test_ops_conv3d.cpp, which carries a case for exactly
  // this. The 5.00679e-06 figure recorded above is a mutation that ALSO narrowed
  // the accumulator width; order alone does not move these goldens at their
  // fixture scale.
  //
  // WHY THE PAD IS STILL BUILT HERE, and is not an argument to the op. torch
  // does the same: `nn.Conv3d` with a non-`zeros` `padding_mode` runs `F.pad`
  // and then a ZERO-padded convolution, and upstream materialises the temporal
  // pad itself with a `torch.concatenate` (convolution.py:305-311). So the op
  // is handed an already-padded volume and `pad_* = 0`, and LTX's padding-mode
  // enum stays out of a shared header.
  //
  // The parallel partition and the determinism argument that used to live here
  // moved with the loop: src/vt/cpu/cpu_conv3d.cpp states them, and
  // tests/vt/test_ops_conv3d.cpp gates them at 1/2/4/8 threads.
  vt::Conv3dArgs args;
  args.stride_t = stride_t;
  args.stride_h = stride_h;
  args.stride_w = stride_w;
  VT_CHECK(wcache != nullptr, "ltx2 video vae: a convolution was reached with no weight cache");
  Conv3dThroughSeam(queue, padded.ptr(), ci, pt, ph, pw, wcache->Get(weight),
                    bias != nullptr ? wcache->Get(*bias) : nullptr, out_channels, kernel, args,
                    out.data.ptr(), out.t, out.h, out.w);
  return out;
}

// make_linear_nd for dims == 3 (convolution.py:84-85): a 1x1x1 Conv3d.
Volume Linear3d(vt::Queue* queue, VaeWeightCache* wcache, const Volume& in, int64_t out_channels,
                const std::vector<float>& weight, const std::vector<float>& bias) {
  const int64_t n = in.spatial();
  VT_CHECK(static_cast<int64_t>(weight.size()) == out_channels * in.channels,
           "ltx2 video vae: linear3d weight does not match [out, in]");
  Volume out;
  out.channels = out_channels;
  out.t = in.t;
  out.h = in.h;
  out.w = in.w;
  out.data.Like(in.data, static_cast<size_t>(out_channels * n));
  vt::Queue cpu = VaeCpuQueue();
  vt::Queue& q = queue != nullptr ? *queue : cpu;
  // f32, bias-seeded, one partial per input channel: this is an `nn.Conv3d`
  // upstream too (make_linear_nd's dims==3 branch, convolution.py:84-85), so it
  // takes `vt::Conv3d`'s published accumulation contract rather than a GEMM's.
  VT_CHECK(wcache != nullptr, "ltx2 video vae: linear3d was reached with no weight cache");
  VaeKernels(q).linear_cn(q, out.data.ptr(), in.data.ptr(), wcache->Get(weight),
                          wcache->Get(bias), out_channels, in.channels, n, vt::DType::kF32);
  return out;
}

// f32: `F.silu` computes in the activation dtype. The kernel spells it
// x / (1 + exp(-x)) in float, which is what this loop was; the algebraically
// equivalent x * sigmoid(x) is NOT bit-identical and the goldens are held to the
// first.
void Silu(vt::Queue* queue, float* x, int64_t n) {
  vt::Queue cpu = VaeCpuQueue();
  vt::Queue& q = queue != nullptr ? *queue : cpu;
  VaeSiluKernels(q).silu(q, x, n, vt::DType::kF32);
}

// PixelNorm() with its DEFAULT eps of 1e-8 (normalization.py:22, reached bare
// from video_vae/resnet.py:46 and conv_video_decoder.py:243) — NOT the 1e-6 the
// audio VAE gets through build_normalization_layer.
void PixelNorm(vt::Queue* queue, float* x, int64_t channels, int64_t spatial, double eps) {
  vt::Queue cpu = VaeCpuQueue();
  vt::Queue& q = queue != nullptr ? *queue : cpu;
  // `eps` stays an f64 PARAMETER because it is a pinned config threshold
  // (Ltx2ConvVideoDecoderConfig::pixel_norm_eps). It is narrowed here, at the one
  // point it enters the arithmetic — which is where the host loop narrowed it.
  VaeKernels(q).pixel_norm(q, x, channels, spatial, static_cast<float>(eps), vt::DType::kF32);
}

// The fields the shared convolution/normalization primitives need, so ONE set of
// them serves both the decoder and the encoder rather than each half growing its
// own causal pad. Nothing here is a new degree of freedom: every member is read
// straight off whichever config the caller holds.
struct VideoConvSpec {
  Ltx2NormLayer norm_layer = Ltx2NormLayer::kPixelNorm;
  int64_t norm_num_groups = 32;
  double norm_eps = 1e-6;
  double pixel_norm_eps = 1e-8;
  Ltx2PaddingMode spatial_padding_mode = Ltx2PaddingMode::kZeros;
  // The per-CALL causal flag, i.e. what upstream passes as `causal=` rather than
  // what it passes to a constructor. The decoder takes it from `self.causal`
  // (conv_video_decoder.py:307); the encoder never passes it at all and so always
  // gets the `causal: bool = True` DEFAULT (convolution.py:304).
  bool causal = true;
  // The device this decode runs on (#1007, W5). NULL means the CPU queue, NOT
  // "the old host path": there is one code path and the device is a property of
  // the queue. It lives on the SPEC rather than on nine call sites because the
  // spec is already threaded to every one of them.
  vt::Queue* queue = nullptr;
  // The decode's staged weights. It lives on the spec for the same reason
  // `queue` does: the spec is already threaded to every call site, and a cache
  // constructed per call would stage the weights per call and be no cache.
  VaeWeightCache* wcache = nullptr;
};

VideoConvSpec SpecOf(const Ltx2ConvVideoDecoderConfig& config, vt::Queue* queue = nullptr) {
  VideoConvSpec spec;
  spec.queue = queue;
  spec.norm_layer = config.norm_layer;
  spec.norm_num_groups = config.norm_num_groups;
  spec.norm_eps = config.norm_eps;
  spec.pixel_norm_eps = config.pixel_norm_eps;
  spec.spatial_padding_mode = config.spatial_padding_mode;
  spec.causal = config.causal;
  return spec;
}

// The GroupNorm arm goes through the SAME table as everything else, and its
// kernel is `MiniMaxH3GroupNorm3d`'s loop transcribed with no arithmetic change
// — including the f64 mean and variance accumulators, which every committed
// golden on this path was taken through. Calling that host function directly
// would leave the volume on the host between two convolutions, which is the
// whole of #1451.
void ApplyNorm(const VideoConvSpec& config, float* x, int64_t channels, int64_t spatial,
               const Ltx2VaeWeights& weights, const std::string& prefix) {
  vt::Queue cpu = VaeCpuQueue();
  vt::Queue& q = config.queue != nullptr ? *config.queue : cpu;
  if (config.norm_layer == Ltx2NormLayer::kPixelNorm) {
    PixelNorm(config.queue, x, channels, spatial, config.pixel_norm_eps);
    return;
  }
  VT_CHECK(config.wcache != nullptr, "ltx2 video vae: a norm was reached with no weight cache");
  VaeKernels(q).group_norm(q, x, channels, spatial, config.norm_num_groups,
                           config.wcache->Get(weights.Get(prefix + ".weight")),
                           config.wcache->Get(weights.Get(prefix + ".bias")),
                           config.norm_eps, vt::DType::kF32);
}

// ---------------------------------------------------------------------------
// PixArtAlphaCombinedTimestepSizeEmbeddings (timestep_embedding.py:118-141) at
// batch 1: Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0) followed
// by TimestepEmbedding(256 -> embedding_dim) with a SiLU between its two linears.
// ---------------------------------------------------------------------------
std::vector<float> TimestepEmbedding(double timestep, int64_t embedding_dim,
                                     const Ltx2VaeWeights& weights, const std::string& prefix) {
  constexpr int64_t kProjChannels = 256;
  constexpr double kMaxPeriod = 10000.0;
  const int64_t half = kProjChannels / 2;
  // DELIBERATE f64 EXCEPTION, and the only one on this path. `proj` is a
  // transcendental CONSTANT table — 256 cos/sin values built once per block from
  // the timestep, never a per-element data-path accumulation — so it is off
  // every hot path, and evaluating it in f64 sits closer to the exact value that
  // upstream's f32 `torch.arange` table approximates. Everything downstream of
  // it is f32 (#1008).
  std::vector<double> proj(static_cast<size_t>(kProjChannels));
  for (int64_t i = 0; i < half; ++i) {
    // downscale_freq_shift = 0, so the divisor is exactly half_dim.
    const double exponent = -std::log(kMaxPeriod) * static_cast<double>(i) / static_cast<double>(half);
    const double angle = timestep * std::exp(exponent);
    // flip_sin_to_cos=True puts COS first (timestep_embedding.py:87-89).
    proj[static_cast<size_t>(i)] = std::cos(angle);
    proj[static_cast<size_t>(half + i)] = std::sin(angle);
  }

  const std::vector<float>& w1 = weights.Get(prefix + ".timestep_embedder.linear_1.weight");
  const std::vector<float>& b1 = weights.Get(prefix + ".timestep_embedder.linear_1.bias");
  const std::vector<float>& w2 = weights.Get(prefix + ".timestep_embedder.linear_2.weight");
  const std::vector<float>& b2 = weights.Get(prefix + ".timestep_embedder.linear_2.bias");
  VT_CHECK(static_cast<int64_t>(w1.size()) == embedding_dim * kProjChannels,
           "ltx2 timestep embedding: linear_1 shape does not match the embedding dim");

  // f32 for both `nn.Linear` accumulators and for the hidden activation between
  // them: upstream's TimestepEmbedder is two plain Linears with a SiLU, all in
  // the activation dtype. The frequency table above stays f64 — see its note.
  std::vector<float> hidden(static_cast<size_t>(embedding_dim));
  for (int64_t o = 0; o < embedding_dim; ++o) {
    float acc = b1[static_cast<size_t>(o)];
    for (int64_t i = 0; i < kProjChannels; ++i) {
      acc += static_cast<float>(proj[static_cast<size_t>(i)]) *
             w1[static_cast<size_t>(o * kProjChannels + i)];
    }
    hidden[static_cast<size_t>(o)] = acc / (1.0f + std::exp(-acc));  // SiLU
  }
  std::vector<float> out(static_cast<size_t>(embedding_dim));
  for (int64_t o = 0; o < embedding_dim; ++o) {
    float acc = b2[static_cast<size_t>(o)];
    for (int64_t i = 0; i < embedding_dim; ++i) {
      acc += hidden[static_cast<size_t>(i)] * w2[static_cast<size_t>(o * embedding_dim + i)];
    }
    out[static_cast<size_t>(o)] = acc;
  }
  return out;
}

// _feed_spatial_noise (resnet.py:104-119): ONE [H, W] draw, broadcast over batch,
// channels and TIME, scaled per channel. Drawing a full [C, T, H, W] block
// instead still yields a finite, plausible clip.
void FeedSpatialNoise(vt::Queue* queue, VaeWeightCache* wcache, Volume& x,
                      const std::vector<float>& per_channel_scale, Ltx2NoiseStream* noise) {
  VT_CHECK(noise != nullptr,
           "ltx2 video vae: a block sets inject_noise but no noise stream was supplied");
  // THE PLANE IS DRAWN ON THE HOST AND STAYS THE REPRODUCIBILITY SEAM. A
  // device-side generator would be a different stream, and every render this
  // project has captured is keyed to this one.
  const std::vector<float> plane = noise->Draw(x.h * x.w);
  VT_CHECK(static_cast<int64_t>(plane.size()) == x.h * x.w,
           "ltx2 video vae: the noise stream returned the wrong element count");
  vt::Queue cpu = VaeCpuQueue();
  vt::Queue& q = queue != nullptr ? *queue : cpu;
  VT_CHECK(wcache != nullptr, "ltx2 video vae: noise injection was reached with no weight cache");
  const VaeScratch plane_dev(queue, plane);
  VaeKernels(q).spatial_noise(q, x.data.ptr(), plane_dev.ptr(),
                              wcache->Get(per_channel_scale), x.channels, x.t, x.h, x.w,
                              vt::DType::kF32);
}

// One ada-LN group applied in place: x * (1 + scale) + shift, with the pair taken
// from `table[row]` plus `embed[row]` (resnet.py:135-147).
void ApplyAdaLn(vt::Queue* queue, VaeWeightCache* wcache, Volume& x,
                const std::vector<float>& table, const std::vector<float>& embed, int64_t rows,
                int64_t shift_row, int64_t scale_row) {
  const int64_t c = x.channels;
  VT_CHECK(static_cast<int64_t>(table.size()) == rows * c,
           "ltx2 video vae: scale_shift_table does not match the channel count");
  VT_CHECK(static_cast<int64_t>(embed.size()) == rows * c,
           "ltx2 video vae: timestep embedding does not match rows x channels");
  vt::Queue cpu = VaeCpuQueue();
  vt::Queue& q = queue != nullptr ? *queue : cpu;
  VT_CHECK(wcache != nullptr, "ltx2 video vae: ada-LN was reached with no weight cache");
  const VaeScratch embed_dev(queue, embed);
  VaeKernels(q).ada_ln(q, x.data.ptr(), wcache->Get(table), embed_dev.ptr(), c, x.spatial(), rows,
                       shift_row, scale_row, vt::DType::kF32);
}

// ResnetBlock3D.forward (resnet.py:121-186).
Volume ResnetBlock3d(const VideoConvSpec& config, const Ltx2VaeWeights& weights,
                     const std::string& prefix, const Volume& input, int64_t out_channels,
                     bool inject_noise, bool timestep_conditioning,
                     const std::vector<float>* timestep_embed, Ltx2NoiseStream* noise) {
  Volume hidden = input;
  ApplyNorm(config, hidden.data.ptr(), hidden.channels, hidden.spatial(), weights, prefix + ".norm1");
  if (timestep_conditioning) {
    VT_CHECK(timestep_embed != nullptr,
             "ltx2 video vae: a timestep-conditioned block needs a timestep embedding");
    // ada_values rows are (shift1, scale1, shift2, scale2).
    ApplyAdaLn(config.queue, config.wcache, hidden, weights.Get(prefix + ".scale_shift_table"), *timestep_embed, 4, 0, 1);
  }
  Silu(config.queue, hidden.data.ptr(), static_cast<int64_t>(hidden.data.size()));
  hidden = CausalConv3d(config.queue, config.wcache, hidden, out_channels, 3, config.causal,
                        config.spatial_padding_mode, weights.Get(prefix + ".conv1.conv.weight"),
                        &weights.Get(prefix + ".conv1.conv.bias"));
  if (inject_noise) {
    FeedSpatialNoise(config.queue, config.wcache, hidden, weights.Get(prefix + ".per_channel_scale1"), noise);
  }

  ApplyNorm(config, hidden.data.ptr(), hidden.channels, hidden.spatial(), weights, prefix + ".norm2");
  if (timestep_conditioning) {
    ApplyAdaLn(config.queue, config.wcache, hidden, weights.Get(prefix + ".scale_shift_table"), *timestep_embed, 4, 2, 3);
  }
  Silu(config.queue, hidden.data.ptr(), static_cast<int64_t>(hidden.data.size()));
  hidden = CausalConv3d(config.queue, config.wcache, hidden, out_channels, 3, config.causal,
                        config.spatial_padding_mode, weights.Get(prefix + ".conv2.conv.weight"),
                        &weights.Get(prefix + ".conv2.conv.bias"));
  if (inject_noise) {
    FeedSpatialNoise(config.queue, config.wcache, hidden, weights.Get(prefix + ".per_channel_scale2"), noise);
  }

  Volume residual = input;
  if (input.channels != out_channels) {
    // norm3 is GroupNorm with ONE group — a LayerNorm over (C, T, H, W) that
    // works in the (B, C, ...) layout without a rearrange (resnet.py:91-97).
    vt::Queue norm3_cpu = VaeCpuQueue();
    vt::Queue& n3q = config.queue != nullptr ? *config.queue : norm3_cpu;
    VaeKernels(n3q).group_norm(n3q, residual.data.ptr(), residual.channels, residual.spatial(), 1,
                               config.wcache->Get(weights.Get(prefix + ".norm3.weight")),
                               config.wcache->Get(weights.Get(prefix + ".norm3.bias")),
                               config.norm_eps, vt::DType::kF32);
    residual = Linear3d(config.queue, config.wcache, residual, out_channels, weights.Get(prefix + ".conv_shortcut.weight"),
                        weights.Get(prefix + ".conv_shortcut.bias"));
  }
  VT_CHECK(residual.data.size() == hidden.data.size(),
           "ltx2 video vae: resnet residual and main-branch shapes must match");
  {
    vt::Queue res_cpu = VaeCpuQueue();
    vt::Queue& rq = config.queue != nullptr ? *config.queue : res_cpu;
    VaeAddInPlace(rq, hidden.data.ptr(), residual.data.ptr(),
                  static_cast<int64_t>(hidden.data.size()));
  }
  return hidden;
}

// DepthToSpaceUpsample.forward (sampling.py:93-123). The channel unpack is
// `(c p1 p2 p3)` with p1 temporal and p2/p3 spatial, and a temporal stride of 2
// DROPS THE FIRST FRAME afterwards.
Volume DepthToSpaceUpsample(const VideoConvSpec& config, const Ltx2VaeWeights& weights,
                            const std::string& prefix, const Volume& x, int64_t st, int64_t sh,
                            int64_t sw, int64_t reduction, bool residual) {
  const int64_t stride_product = st * sh * sw;
  const int64_t conv_out_channels = stride_product * x.channels / reduction;

  vt::Queue cpu = VaeCpuQueue();
  vt::Queue& q = config.queue != nullptr ? *config.queue : cpu;

  // The three shape moves are kernels now (#1451). Each was a host loop that put
  // the volume back on the host between two convolutions.
  auto expand = [&](const Volume& packed) {
    Volume out;
    out.channels = packed.channels / stride_product;
    out.t = packed.t * st;
    out.h = packed.h * sh;
    out.w = packed.w * sw;
    out.data.Like(packed.data, static_cast<size_t>(out.channels * out.spatial()));
    VaeKernels(q).depth_to_space(q, out.data.ptr(), packed.data.ptr(), out.channels, packed.t,
                                 packed.h, packed.w, st, sh, sw, vt::DType::kF32);
    return out;
  };
  auto drop_first_frame = [&](const Volume& v) {
    Volume out;
    out.channels = v.channels;
    out.t = v.t - 1;
    out.h = v.h;
    out.w = v.w;
    out.data.Like(v.data, static_cast<size_t>(out.channels * out.spatial()));
    VaeKernels(q).frame_slice(q, out.data.ptr(), v.data.ptr(), v.channels, v.t, v.h, v.w,
                              /*drop=*/1, vt::DType::kF32);
    return out;
  };

  Volume skip;
  if (residual) {
    // The residual expands the INPUT itself and then repeats it up to the output
    // width (sampling.py:98-110).
    Volume expanded = expand(x);
    const int64_t repeat = stride_product / reduction;
    Volume repeated;
    repeated.channels = expanded.channels * repeat;
    repeated.t = expanded.t;
    repeated.h = expanded.h;
    repeated.w = expanded.w;
    repeated.data.Like(expanded.data, static_cast<size_t>(repeated.channels * repeated.spatial()));
    // torch's `repeat` TILES the whole tensor (sampling.py:108), so the block
    // index is the OUTER axis; `repeat_interleave` would put it inner and is a
    // different tensor.
    VaeKernels(q).channel_repeat(q, repeated.data.ptr(), expanded.data.ptr(), expanded.channels,
                                 expanded.spatial(), repeat, vt::DType::kF32);
    skip = st == 2 ? drop_first_frame(repeated) : repeated;
  }

  Volume packed = CausalConv3d(config.queue, config.wcache, x, conv_out_channels, 3, config.causal,
                               config.spatial_padding_mode,
                               weights.Get(prefix + ".conv.conv.weight"),
                               &weights.Get(prefix + ".conv.conv.bias"));
  Volume out = expand(packed);
  if (st == 2) out = drop_first_frame(out);
  if (residual) {
    VT_CHECK(skip.data.size() == out.data.size(),
             "ltx2 video vae: depth-to-space residual and main-branch shapes must match");
    VaeAddInPlace(q, out.data.ptr(), skip.data.ptr(), static_cast<int64_t>(out.data.size()));
  }
  return out;
}

// AttnBlock3D.forward (attention.py:58-69): SINGLE-HEAD spatial self-attention
// PER FRAME, with frames folded into the batch — there is deliberately no
// cross-frame interaction, so this block does not break temporal causality.
// THIS STAGE IS THE DECLARED STAGED REMAINDER OF #1451, and it is the ONE stage
// of the decode that still leaves the device. It is held back deliberately, not
// for time, and the reason is written here because a reader of this file is
// exactly who needs it:
//
//   * It is the only stage that needs an ATTENTION RUNG SELECTED. This tree has
//     `vt::Attention` (op 18, naive) and the fast rungs `vt::AttentionDenseFast`
//     (20), `DenseFlash` (21) and `DenseFa2` (22). They are SEPARATE ops with no
//     selector and no fallback notice, and picking the wrong one silently is
//     what #1549 and #1794 both were.
//   * It is the only stage whose port CHANGES THE NUMBERS. Its softmax and its
//     two 1x1 convolutions have an accumulation order that no shared attention
//     op reproduces, so unlike the ten kernels this row did land it cannot ride
//     tests/vllm/models/ltx2_vae_goldens.inc -- it needs its own red-first
//     re-gate and its own fresh review.
//
// So the volume is DOWNLOADED here and re-UPLOADED after, and a decode whose
// config carries an `attn` block pays two transfers per block. That is worse
// than nothing only if it is silent, which is why it is named here, in the
// commit body, in the pull request body, and under `## Owed` in
// .agents/specs/ltx25-vae-device-residency.md. The residency gate in
// tests/vllm/multimodal/test_diffusion_device_seam.cpp uses a decoder WITHOUT an
// attn block, so it measures the spine this row did make resident and does not
// quietly pass over this hole.
Volume AttnBlock3d(const Ltx2VaeWeights& weights, const std::string& prefix, const Volume& x) {
  const int64_t c = x.channels;
  const int64_t n = x.h * x.w;
  const std::vector<float>& gamma = weights.Get(prefix + ".norm.gamma");
  const std::vector<float>& qkv_w = weights.Get(prefix + ".to_qkv.weight");
  const std::vector<float>& qkv_b = weights.Get(prefix + ".to_qkv.bias");
  const std::vector<float>& proj_w = weights.Get(prefix + ".proj.weight");
  const std::vector<float>& proj_b = weights.Get(prefix + ".proj.bias");
  const double norm_scale = std::sqrt(static_cast<double>(c));
  const double attn_scale = 1.0 / std::sqrt(static_cast<double>(c));

  // See this function's header for why this stage is still on the host.
  //
  // THE HOST ARM STILL MOVES NO EXTRA BYTE. `Volume out = x` is the one copy
  // this block always made, and on the CPU queue the loops below read and write
  // those bytes in place exactly as they did before this row. The download and
  // the re-upload happen only when the volume is genuinely resident, which is
  // the case this stage has not been ported for. That is an ALLOCATION branch,
  // not a second arithmetic path: there is one copy of the loops below and both
  // devices run it.
  Volume out = x;
  std::vector<float> resident_in, resident_out;
  const float* xh = nullptr;
  float* outh = nullptr;
  const bool resident = x.data.OnDevice();
  if (resident) {
    resident_in.resize(x.data.size());
    x.data.Download(resident_in.data());
    resident_out = resident_in;
    xh = resident_in.data();
    outh = resident_out.data();
  } else {
    xh = x.data.Host();
    outh = out.data.Host();
  }
  // f32 activations, not f64. Upstream holds q/k/v and the attention output in
  // the tensor dtype (attention.py:63-67) and never promotes; these six buffers
  // are the block's whole scratch footprint, so the width is bytes as well as
  // arithmetic. `norm_scale` and `attn_scale` above stay f64 — upstream's
  // `channels**0.5` is a Python float evaluated once per block.
  std::vector<float> normed(static_cast<size_t>(c * n));
  std::vector<float> q(static_cast<size_t>(c * n)), k(static_cast<size_t>(c * n)),
      v(static_cast<size_t>(c * n));
  std::vector<float> scores(static_cast<size_t>(n));
  std::vector<float> attended(static_cast<size_t>(c * n));

  for (int64_t frame = 0; frame < x.t; ++frame) {
    // _RMSNorm2D: F.normalize(x, dim=1) * (sqrt(C) * gamma) — an L2 normalize with
    // torch's 1e-12 floor, not a mean-square RMS.
    for (int64_t i = 0; i < n; ++i) {
      // f32: `F.normalize(x, dim=1)` computes its norm in the input dtype
      // (attention.py:23). torch's 1e-12 floor stays f64 — it is a threshold.
      float sum_sq = 0.0f;
      for (int64_t ch = 0; ch < c; ++ch) {
        const float value = xh[x.At(ch, frame, i / x.w, i % x.w)];
        sum_sq += value * value;
      }
      const float inv = static_cast<float>(
          1.0 / std::max(std::sqrt(static_cast<double>(sum_sq)), kLtx2RmsNorm2dEps));
      // Same left-to-right association the f64 arm used; only the width changes.
      const float norm_scale_f = static_cast<float>(norm_scale);
      for (int64_t ch = 0; ch < c; ++ch) {
        normed[static_cast<size_t>(ch * n + i)] = xh[x.At(ch, frame, i / x.w, i % x.w)] * inv *
                                                  norm_scale_f * gamma[static_cast<size_t>(ch)];
      }
    }
    // to_qkv is a 1x1 Conv2d emitting [q | k | v] along the channel axis, and the
    // rearrange to tokens keeps that split on the LAST axis (attention.py:63-64).
    // f32: `to_qkv` is a 1x1 nn.Conv2d (attention.py:55), the same accumulator
    // width as every other conv here.
    for (int64_t oc = 0; oc < 3 * c; ++oc) {
      std::vector<float>& dst = oc < c ? q : (oc < 2 * c ? k : v);
      const int64_t row = oc % c;
      for (int64_t i = 0; i < n; ++i) {
        float acc = qkv_b[static_cast<size_t>(oc)];
        for (int64_t ic = 0; ic < c; ++ic) {
          acc += normed[static_cast<size_t>(ic * n + i)] * qkv_w[static_cast<size_t>(oc * c + ic)];
        }
        dst[static_cast<size_t>(row * n + i)] = acc;
      }
    }
    // f32: SDPA computes scores, softmax and the value-weighted sum in the
    // tensor dtype (attention.py:65). `attn_scale` stays f64 for the same reason
    // `norm_scale` does.
    const float attn_scale_f = static_cast<float>(attn_scale);
    for (int64_t i = 0; i < n; ++i) {
      float max_score = -std::numeric_limits<float>::infinity();
      for (int64_t j = 0; j < n; ++j) {
        float dot = 0.0f;
        for (int64_t ch = 0; ch < c; ++ch) {
          dot += q[static_cast<size_t>(ch * n + i)] * k[static_cast<size_t>(ch * n + j)];
        }
        scores[static_cast<size_t>(j)] = dot * attn_scale_f;
        max_score = std::max(max_score, scores[static_cast<size_t>(j)]);
      }
      float sum = 0.0f;
      for (int64_t j = 0; j < n; ++j) {
        scores[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(j)] - max_score);
        sum += scores[static_cast<size_t>(j)];
      }
      for (int64_t ch = 0; ch < c; ++ch) {
        float acc = 0.0f;
        for (int64_t j = 0; j < n; ++j) {
          acc += scores[static_cast<size_t>(j)] * v[static_cast<size_t>(ch * n + j)];
        }
        attended[static_cast<size_t>(ch * n + i)] = acc / sum;
      }
    }
    // f32: `proj` is a 1x1 nn.Conv2d (attention.py:56).
    for (int64_t oc = 0; oc < c; ++oc) {
      for (int64_t i = 0; i < n; ++i) {
        float acc = proj_b[static_cast<size_t>(oc)];
        for (int64_t ic = 0; ic < c; ++ic) {
          acc += attended[static_cast<size_t>(ic * n + i)] * proj_w[static_cast<size_t>(oc * c + ic)];
        }
        outh[out.At(oc, frame, i / x.w, i % x.w)] += acc;
      }
    }
  }
  if (resident) out.data.Upload(outh);
  return out;
}

}  // namespace

Ltx2VideoDecoderKind Ltx2ParseVideoDecoderKind(const std::string& vae_class_name) {
  // model_configurator.py:18-34: the conv decoder is the DEFAULT when the field is
  // absent, and is otherwise selected by the exact class name.
  if (vae_class_name.empty() || vae_class_name == "CausalVideoAutoencoder") {
    return Ltx2VideoDecoderKind::kConv;
  }
  return Ltx2VideoDecoderKind::kDiffusion;
}

Ltx2VideoFrames Ltx2ConvVideoDecode(const Ltx2ConvVideoDecoderConfig& config,
                                    const Ltx2VaeWeights& weights,
                                    const std::vector<float>& latent, int64_t latent_channels,
                                    int64_t latent_t, int64_t latent_h, int64_t latent_w,
                                    Ltx2NoiseStream* noise, const double* timestep,
                                    vt::Queue* queue) {
  VT_CHECK(latent_channels == config.in_channels,
           "ltx2 video vae: latent channel count does not match in_channels");
  VT_CHECK(static_cast<int64_t>(latent.size()) == latent_channels * latent_t * latent_h * latent_w,
           "ltx2 video vae: latent size does not match [C, T, H, W]");
  const std::string p = config.prefix;
  // ONE CACHE FOR THE WHOLE DECODE. Constructed here and handed to the spec, so
  // every convolution and every 1x1x1 linear below reaches the SAME staged copy
  // of a weight. Constructing it per call site would stage the weights per call
  // and be no cache at all.
  VaeWeightCache wcache(queue);
  VideoConvSpec spec = SpecOf(config, queue);
  spec.wcache = &wcache;

  // THE PROLOGUE RUNS ON THE HOST, ON PURPOSE, AND IT COSTS NO ROUND TRIP.
  // Both steps below touch the LATENT -- before `conv_in`, before the volume is
  // resident -- so doing them here and uploading once afterwards is one transfer,
  // exactly as many as uploading first and running two device kernels would be.
  // The noise draw has to be here in any case: `Ltx2NoiseStream` is this
  // project's reproducibility seam and a device-side generator would be a
  // different stream from the one every captured render is keyed to.
  std::vector<float> staged = latent;

  // --- noise + denormalize (conv_video_decoder.py:286-301) ---
  if (config.timestep_conditioning) {
    VT_CHECK(noise != nullptr,
             "ltx2 video vae: timestep conditioning injects noise but no noise stream was supplied");
    const std::vector<float> drawn = noise->Draw(static_cast<int64_t>(staged.size()));
    VT_CHECK(drawn.size() == staged.size(),
             "ltx2 video vae: the noise stream returned the wrong element count");
    // f32: the blend runs in the activation dtype upstream. The two scalars are
    // config values, so they are narrowed once rather than per element.
    const float noise_scale = static_cast<float>(config.decode_noise_scale);
    const float keep_scale = static_cast<float>(1.0 - config.decode_noise_scale);
    for (size_t i = 0; i < staged.size(); ++i) {
      staged[i] = drawn[i] * noise_scale + keep_scale * staged[i];
    }
  }
  {
    const std::vector<float>& std_of_means =
        weights.Get(p + "per_channel_statistics.std-of-means");
    const std::vector<float>& mean_of_means =
        weights.Get(p + "per_channel_statistics.mean-of-means");
    VT_CHECK(static_cast<int64_t>(std_of_means.size()) == latent_channels &&
                 static_cast<int64_t>(mean_of_means.size()) == latent_channels,
             "ltx2 video vae: per-channel statistics must have one value per latent channel");
    const int64_t n = latent_t * latent_h * latent_w;
    // f32: upstream's de-normalize is `latent * std + mean` on f32/bf16 tensors.
    for (int64_t c = 0; c < latent_channels; ++c) {
      const float std_c = std_of_means[static_cast<size_t>(c)];
      const float mean_c = mean_of_means[static_cast<size_t>(c)];
      for (int64_t i = 0; i < n; ++i) {
        staged[static_cast<size_t>(c * n + i)] =
            staged[static_cast<size_t>(c * n + i)] * std_c + mean_c;
      }
    }
  }

  // THE ONE UPLOAD. From here to `unpatchify` the volume never leaves the
  // queue's device, which is the whole of #1451.
  Volume x;
  x.channels = latent_channels;
  x.t = latent_t;
  x.h = latent_h;
  x.w = latent_w;
  x.data.Alloc(queue, staged.size());
  x.data.Upload(staged.data());

  const double scaled_timestep =
      config.timestep_conditioning
          ? (timestep != nullptr ? *timestep : config.decode_timestep) *
                static_cast<double>(weights.Get(p + "timestep_scale_multiplier")[0])
          : 0.0;

  // --- conv_in widens the latents to the bottleneck ---
  int64_t multiplier = 1;
  for (const Ltx2VideoDecoderBlock& block : config.decoder_blocks) {
    if (block.name == "compress_time" || block.name == "compress_space" ||
        block.name == "compress_all") {
      multiplier *= block.multiplier != 0 ? block.multiplier : 1;
    } else if (block.name == "res_x_y") {
      multiplier *= block.multiplier != 0 ? block.multiplier : 2;
    }
  }
  // TWO DIFFERENT `causal` FLAGS, and passing `config.causal` here is correct.
  // Upstream builds conv_in with `causal=True` (conv_video_decoder.py:216), but
  // that constructor argument only selects the MODULE — it is what makes conv_in a
  // CausalConv3d at all. The one-sidedness of any given call comes from the
  // separate per-call argument, `self.conv_in(sample, causal=self.causal)`
  // (conv_video_decoder.py:307), and that is `self.causal`, i.e. this config's
  // field. So `config.causal` is the value that belongs here; hardcoding `true`
  // would silently make a non-causal decoder pad one-sidedly.
  x = CausalConv3d(spec.queue, spec.wcache, x, config.base_channels * multiplier, 3, config.causal,
                   config.spatial_padding_mode, weights.Get(p + "conv_in.conv.weight"),
                   &weights.Get(p + "conv_in.conv.bias"));

  // --- the reversed block walk (conv_video_decoder.py:222-238, 315-326) ---
  int64_t index = 0;
  for (auto it = config.decoder_blocks.rbegin(); it != config.decoder_blocks.rend(); ++it, ++index) {
    const Ltx2VideoDecoderBlock& block = *it;
    const std::string bp = p + "up_blocks." + std::to_string(index);
    if (block.name == "res_x") {
      std::vector<float> embed;
      const std::vector<float>* embed_ptr = nullptr;
      if (config.timestep_conditioning) {
        embed = TimestepEmbedding(scaled_timestep, x.channels * 4, weights, bp + ".time_embedder");
        embed_ptr = &embed;
      }
      for (int64_t i = 0; i < block.num_layers; ++i) {
        x = ResnetBlock3d(spec, weights, bp + ".res_blocks." + std::to_string(i), x, x.channels,
                          block.inject_noise, config.timestep_conditioning, embed_ptr, noise);
      }
    } else if (block.name == "res_x_y") {
      const int64_t out_channels = x.channels / (block.multiplier != 0 ? block.multiplier : 2);
      // _make_decoder_block forces timestep_conditioning=False for res_x_y
      // (conv_video_decoder.py:107).
      x = ResnetBlock3d(spec, weights, bp, x, out_channels, block.inject_noise,
                        /*timestep_conditioning=*/false, nullptr, noise);
    } else if (block.name == "attn") {
      x = AttnBlock3d(weights, bp, x);
    } else if (block.name == "compress_time" || block.name == "compress_space" ||
               block.name == "compress_all") {
      const int64_t st = block.name == "compress_space" ? 1 : 2;
      const int64_t ss = block.name == "compress_time" ? 1 : 2;
      x = DepthToSpaceUpsample(spec, weights, bp, x, st, ss, ss,
                               block.multiplier != 0 ? block.multiplier : 1,
                               block.name == "compress_all" && block.residual);
    } else if (block.name == "attn_res_x") {
      VT_CHECK(false,
               "ltx2 video vae: the `attn_res_x` decoder block cannot be built — upstream passes "
               "`attention_head_dim` to UNetMidBlock3D, which does not accept it "
               "(conv_video_decoder.py:85-96 vs video_vae/resnet.py:210-222)");
    } else {
      VT_CHECK(false, "ltx2 video vae: unknown decoder block `" + block.name + "`");
    }
  }

  // --- conv_norm_out -> ada-LN -> SiLU -> conv_out ---
  if (config.norm_layer == Ltx2NormLayer::kPixelNorm) {
    PixelNorm(spec.queue, x.data.ptr(), x.channels, x.spatial(), config.pixel_norm_eps);
  } else {
    vt::Queue tail_cpu = VaeCpuQueue();
    vt::Queue& tq = spec.queue != nullptr ? *spec.queue : tail_cpu;
    VaeKernels(tq).group_norm(tq, x.data.ptr(), x.channels, x.spatial(), config.norm_num_groups,
                              spec.wcache->Get(weights.Get(p + "conv_norm_out.weight")),
                              spec.wcache->Get(weights.Get(p + "conv_norm_out.bias")),
                              config.norm_eps, vt::DType::kF32);
  }
  if (config.timestep_conditioning) {
    const std::vector<float> embed =
        TimestepEmbedding(scaled_timestep, x.channels * 2, weights, p + "last_time_embedder");
    // ada_values rows are (shift, scale) — two, not the resnet's four.
    ApplyAdaLn(spec.queue, spec.wcache, x, weights.Get(p + "last_scale_shift_table"), embed, 2, 0, 1);
  }
  Silu(spec.queue, x.data.ptr(), static_cast<int64_t>(x.data.size()));
  x = CausalConv3d(spec.queue, spec.wcache, x, config.out_channels * config.patch_size * config.patch_size, 3,
                   config.causal, config.spatial_padding_mode,
                   weights.Get(p + "conv_out.conv.weight"),
                   &weights.Get(p + "conv_out.conv.bias"));

  // --- unpatchify (ops.py:35-60): `b (c p r q) f h w -> b c (f p) (h q) (w r)`
  // with p = patch_size_t = 1. NOTE h takes q and w takes r; swapping them
  // transposes every patch.
  const int64_t q = config.patch_size;
  const int64_t r = config.patch_size;
  Ltx2VideoFrames out;
  out.channels = config.out_channels;
  out.frames = x.t;
  out.height = x.h * q;
  out.width = x.w * r;
  const size_t frame_elems =
      static_cast<size_t>(out.channels * out.frames * out.height * out.width);
  out.data.resize(frame_elems);
  // H TAKES q AND W TAKES r. Swapping them transposes every patch, and no
  // shape-valid gate can see that.
  //
  // THE UNPATCHIFY RUNS ON THE DEVICE AND THEN THE FRAMES COME BACK ONCE. Doing
  // the gather host-side would mean downloading the volume in its PACKED layout
  // and re-laying it out on the CPU, which is the round trip this whole row
  // removes, one stage before the end.
  {
    vt::Queue cpu = VaeCpuQueue();
    vt::Queue& qq = spec.queue != nullptr ? *spec.queue : cpu;
    VaeStore frames;
    frames.Like(x.data, frame_elems);
    VaeKernels(qq).unpatchify(qq, frames.ptr(), x.data.ptr(), out.channels, x.t, x.h, x.w, q, r,
                              vt::DType::kF32);
    frames.Download(out.data.data());
  }
  return out;
}

Ltx2VideoFrames Ltx2VideoDecode(Ltx2VideoDecoderKind kind,
                                const Ltx2ConvVideoDecoderConfig& config,
                                const Ltx2VaeWeights& weights, const std::vector<float>& latent,
                                int64_t latent_channels, int64_t latent_t, int64_t latent_h,
                                int64_t latent_w, Ltx2NoiseStream* noise, const double* timestep,
                                vt::Queue* queue) {
  // REFUSE, never downgrade: falling back to the conv decoder would return a
  // lower-quality render as if it were the requested one, and no gate this
  // project owns could detect that (.agents/specs/ltx-2-5.md section 0 item 2).
  VT_CHECK(kind != Ltx2VideoDecoderKind::kDiffusion,
           "ltx2 video vae: this checkpoint asks for the DIFFUSION video decoder "
           "(NADiffusionDecoder / DiffusionVideoDecoder), which is NOT implemented — it needs a "
           "neighborhood-attention kernel and has its own row. It is refused rather than "
           "downgraded to the Conv video VAE, which would silently return a worse render");
  return Ltx2ConvVideoDecode(config, weights, latent, latent_channels, latent_t, latent_h, latent_w,
                             noise, timestep, queue);
}

// ===========================================================================
// THE ENCODER HALF (video_vae.py:39-336), which phase L4 recorded as owed.
//
// It lives in this translation unit deliberately, so that `CausalConv3d`,
// `PixelNorm`, `ApplyNorm`, `ResnetBlock3d` and `AttnBlock3d` are the SAME
// functions the decoder is gated on rather than a second copy of each. The one
// primitive the decoder never needed is a STRIDE on the causal convolution, and
// that was added to the shared function above rather than forked here.
// ===========================================================================

namespace {

// patchify (ops.py:6-32), the 5-D arm with patch_size_t = 1:
//   `b c (f p) (h q) (w r) -> b (c p r q) f h w`
// r is the OUTER spatial factor and q the inner one, and h takes q while w takes
// r. That is the exact inverse of the decoder's unpatchify above; swapping r and
// q transposes every patch and still type-checks.
Volume Patchify(const Volume& in, int64_t patch) {
  if (patch == 1) return in;
  VT_CHECK(in.h % patch == 0 && in.w % patch == 0,
           "ltx2 video encoder: height and width must be whole multiples of patch_size");
  Volume out;
  out.channels = in.channels * patch * patch;
  out.t = in.t;
  out.h = in.h / patch;
  out.w = in.w / patch;
  out.data.Alloc(nullptr, static_cast<size_t>(out.channels * out.spatial()));
  for (int64_t c = 0; c < in.channels; ++c) {
    for (int64_t ri = 0; ri < patch; ++ri) {
      for (int64_t qi = 0; qi < patch; ++qi) {
        const int64_t dst_c = (c * patch + ri) * patch + qi;
        for (int64_t f = 0; f < out.t; ++f) {
          for (int64_t hi = 0; hi < out.h; ++hi) {
            for (int64_t wi = 0; wi < out.w; ++wi) {
              out.data.Host()[out.At(dst_c, f, hi, wi)] =
                  in.data.Host()[in.At(c, f, hi * patch + qi, wi * patch + ri)];
            }
          }
        }
      }
    }
  }
  return out;
}

// The space-to-depth fold both branches of SpaceToDepthDownsample share:
//   `b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w`   (sampling.py:43-49, 55-61)
Volume SpaceToDepthFold(const Volume& in, int64_t st, int64_t sh, int64_t sw) {
  VT_CHECK(in.t % st == 0 && in.h % sh == 0 && in.w % sw == 0,
           "ltx2 video encoder: space-to-depth needs each axis to be a whole multiple of its "
           "stride");
  Volume out;
  out.channels = in.channels * st * sh * sw;
  out.t = in.t / st;
  out.h = in.h / sh;
  out.w = in.w / sw;
  out.data.Alloc(nullptr, static_cast<size_t>(out.channels * out.spatial()));
  for (int64_t c = 0; c < in.channels; ++c) {
    for (int64_t p1 = 0; p1 < st; ++p1) {
      for (int64_t p2 = 0; p2 < sh; ++p2) {
        for (int64_t p3 = 0; p3 < sw; ++p3) {
          const int64_t dst_c = ((c * st + p1) * sh + p2) * sw + p3;
          for (int64_t ti = 0; ti < out.t; ++ti) {
            for (int64_t hi = 0; hi < out.h; ++hi) {
              for (int64_t wi = 0; wi < out.w; ++wi) {
                out.data.Host()[out.At(dst_c, ti, hi, wi)] =
                    in.data.Host()[in.At(c, ti * st + p1, hi * sh + p2, wi * sw + p3)];
              }
            }
          }
        }
      }
    }
  }
  return out;
}

// SpaceToDepthDownsample.forward (sampling.py:34-65). Three things that fail
// silently and are therefore spelled out:
//  * a temporal stride of 2 DUPLICATES FRAME 0 first (sampling.py:39-40), and the
//    duplication happens BEFORE both the skip fold and the convolution;
//  * the skip is a GROUP MEAN over `group_size` contiguous folded channels
//    (`b (c g) d h w -> b c g d h w` then `.mean(dim=2)`, sampling.py:50-51) —
//    c is the OUTER factor, so group g is contiguous;
//  * the convolution emits `out_channels / prod(stride)` channels and the fold
//    multiplies them back up (sampling.py:27, 55-61).
Volume SpaceToDepthDownsample(const VideoConvSpec& spec, const Ltx2VaeWeights& weights,
                              const std::string& prefix, const Volume& x, int64_t st, int64_t sh,
                              int64_t sw, int64_t out_channels) {
  const int64_t stride_product = st * sh * sw;
  VT_CHECK(out_channels % stride_product == 0,
           "ltx2 video encoder: SpaceToDepthDownsample needs out_channels divisible by the stride "
           "product (sampling.py:27)");
  const int64_t conv_out_channels = out_channels / stride_product;
  const int64_t folded = x.channels * stride_product;
  VT_CHECK(folded % out_channels == 0,
           "ltx2 video encoder: SpaceToDepthDownsample needs in_channels * prod(stride) divisible "
           "by out_channels (sampling.py:23)");
  const int64_t group_size = folded / out_channels;

  Volume grown = x;
  if (st == 2) {
    grown.t = x.t + 1;
    grown.data.Alloc(nullptr, static_cast<size_t>(grown.channels * grown.spatial()));
    for (int64_t c = 0; c < grown.channels; ++c) {
      for (int64_t ti = 0; ti < grown.t; ++ti) {
        const int64_t src_t = ti == 0 ? 0 : ti - 1;
        for (int64_t hi = 0; hi < grown.h; ++hi) {
          for (int64_t wi = 0; wi < grown.w; ++wi) {
            grown.data.Host()[grown.At(c, ti, hi, wi)] = x.data.Host()[x.At(c, src_t, hi, wi)];
          }
        }
      }
    }
  }

  // --- the skip: fold, then average each contiguous group of `group_size` ---
  const Volume folded_in = SpaceToDepthFold(grown, st, sh, sw);
  Volume skip;
  skip.channels = out_channels;
  skip.t = folded_in.t;
  skip.h = folded_in.h;
  skip.w = folded_in.w;
  skip.data.Alloc(nullptr, static_cast<size_t>(skip.channels * skip.spatial()));
  const int64_t n = skip.spatial();
  for (int64_t c = 0; c < out_channels; ++c) {
    for (int64_t i = 0; i < n; ++i) {
      // f32: upstream's group mean runs in the activation dtype.
      float acc = 0.0f;
      for (int64_t g = 0; g < group_size; ++g) {
        acc += folded_in.data.Host()[static_cast<size_t>((c * group_size + g) * n + i)];
      }
      skip.data.Host()[static_cast<size_t>(c * n + i)] = acc / static_cast<float>(group_size);
    }
  }

  // --- the conv branch, at stride 1, on the SAME duplicated input ---
  const Volume convolved =
      CausalConv3d(spec.queue, spec.wcache, grown, conv_out_channels, 3, spec.causal,
                   spec.spatial_padding_mode,
                   weights.Get(prefix + ".conv.conv.weight"),
                   &weights.Get(prefix + ".conv.conv.bias"));
  Volume out = SpaceToDepthFold(convolved, st, sh, sw);
  VT_CHECK(out.data.size() == skip.data.size(),
           "ltx2 video encoder: SpaceToDepthDownsample skip and conv shapes must match");
  for (size_t i = 0; i < out.data.size(); ++i) out.data.Host()[i] += skip.data.Host()[i];
  return out;
}

bool StartsWith(const std::string& value, const char* prefix) {
  return value.rfind(prefix, 0) == 0;
}

VideoConvSpec SpecOf(const Ltx2ConvVideoEncoderConfig& config, vt::Queue* queue = nullptr) {
  VideoConvSpec spec;
  spec.queue = queue;
  spec.norm_layer = config.norm_layer;
  spec.norm_num_groups = config.norm_num_groups;
  spec.norm_eps = config.norm_eps;
  spec.pixel_norm_eps = config.pixel_norm_eps;
  spec.spatial_padding_mode = config.spatial_padding_mode;
  // The ENCODER never passes `causal=` to anything it calls (video_vae.py:292-299),
  // so every convolution takes the `causal: bool = True` DEFAULT. There is no
  // knob, and inventing one would let a caller build a non-causal encoder upstream
  // cannot produce.
  spec.causal = true;
  return spec;
}

// `_make_encoder_block`'s out_channels arithmetic (video_vae.py:39-145). The
// plain strided convolutions keep `in_channels`; every `*_x_y` and `*_res` kind
// multiplies by `block_config.get("multiplier", 2)`.
int64_t EncoderBlockOutChannels(const Ltx2VideoEncoderBlock& block, int64_t in_channels) {
  const int64_t multiplier = block.multiplier != 0 ? block.multiplier : 2;
  if (block.name == "res_x_y" || block.name == "compress_all_x_y" ||
      block.name == "compress_all_res" || block.name == "compress_space_res" ||
      block.name == "compress_time_res") {
    return in_channels * multiplier;
  }
  return in_channels;
}

}  // namespace

int64_t Ltx2VideoTemporalScaleFactor(const std::vector<Ltx2VideoEncoderBlock>& blocks) {
  int64_t steps = 0;
  for (const Ltx2VideoEncoderBlock& block : blocks) {
    if (StartsWith(block.name, "compress_time") || StartsWith(block.name, "compress_all")) ++steps;
  }
  return int64_t{1} << steps;
}

int64_t Ltx2VideoSpatialScaleFactor(const std::vector<Ltx2VideoEncoderBlock>& blocks,
                                    int64_t patch_size) {
  int64_t steps = 0;
  for (const Ltx2VideoEncoderBlock& block : blocks) {
    if (StartsWith(block.name, "compress_space") || StartsWith(block.name, "compress_all")) ++steps;
  }
  return patch_size * (int64_t{1} << steps);
}

Ltx2LatentVolume Ltx2ConvVideoEncode(const Ltx2ConvVideoEncoderConfig& config,
                                     const Ltx2VaeWeights& weights,
                                     const std::vector<float>& frames, int64_t channels,
                                     int64_t frame_count, int64_t height, int64_t width,
                                     int64_t* out_cropped_frames) {
  VT_CHECK(channels == config.in_channels,
           "ltx2 video encoder: input channel count does not match in_channels");
  VT_CHECK(static_cast<int64_t>(frames.size()) == channels * frame_count * height * width,
           "ltx2 video encoder: input size does not match [C, F, H, W]");
  VT_CHECK(frame_count >= 1, "ltx2 video encoder: at least one frame is required");
  // `latent_log_var="none"` is REFUSED rather than reproduced. Upstream skips the
  // uniform/constant fix-ups and then still runs `torch.chunk(sample, 2, dim=1)`
  // (video_vae.py:335), so the means carry HALF of `out_channels` while
  // `per_channel_statistics` carries `out_channels` — the broadcast in
  // `normalize` (ops.py:81-84) raises. Reproducing "whatever it does" would mean
  // inventing semantics upstream does not have.
  VT_CHECK(config.latent_log_var != Ltx2LogVarianceType::kNone,
           "ltx2 video encoder: latent_log_var=`none` cannot produce a latent — upstream still "
           "chunks the conv_out into two halves (video_vae.py:335), leaving out_channels/2 mean "
           "channels against out_channels per-channel statistics, and PerChannelStatistics."
           "normalize raises on the broadcast (video_vae/ops.py:81-84)");

  const std::string p = config.prefix;
  // The encoder is HOST-ONLY in this wave: `SpecOf` is called without a queue,
  // so every volume it builds is a host allocation and the cache below is a
  // pass-through that copies nothing. Its residency is owed -- see `## Owed` in
  // .agents/specs/ltx25-vae-device-residency.md -- and it is not reachable from
  // a device queue today, because this is the only place the encoder's spec is
  // built and it never takes one.
  VaeWeightCache wcache(nullptr);
  VideoConvSpec spec = SpecOf(config);
  spec.wcache = &wcache;

  // --- the frame-count crop (video_vae.py:276-286) ---
  // Upstream WARNS and crops rather than failing, so a caller that quietly hands
  // an invalid count gets a SHORTER clip, not an error. The count is reported.
  const int64_t temporal_factor = Ltx2VideoTemporalScaleFactor(config.encoder_blocks);
  const int64_t cropped = (frame_count - 1) % temporal_factor;
  const int64_t kept = frame_count - cropped;
  if (out_cropped_frames != nullptr) *out_cropped_frames = cropped;

  Volume x;
  x.channels = channels;
  x.t = kept;
  x.h = height;
  x.w = width;
  x.data.Alloc(nullptr, static_cast<size_t>(x.channels * x.spatial()));
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t f = 0; f < kept; ++f) {
      const size_t src = static_cast<size_t>((c * frame_count + f) * height * width);
      std::copy(frames.begin() + static_cast<ptrdiff_t>(src),
                frames.begin() + static_cast<ptrdiff_t>(src + static_cast<size_t>(height * width)),
                x.data.HostBegin() + static_cast<ptrdiff_t>(x.At(c, f, 0, 0)));
    }
  }

  // --- patchify -> conv_in (video_vae.py:291-292) ---
  x = Patchify(x, config.patch_size);
  x = CausalConv3d(spec.queue, spec.wcache, x, config.out_channels, 3, spec.causal, spec.spatial_padding_mode,
                   weights.Get(p + "conv_in.conv.weight"), &weights.Get(p + "conv_in.conv.bias"));

  // --- the FORWARD block walk (video_vae.py:221-236, 294-295) ---
  int64_t index = 0;
  for (const Ltx2VideoEncoderBlock& block : config.encoder_blocks) {
    const std::string bp = p + "down_blocks." + std::to_string(index);
    const int64_t out_channels = EncoderBlockOutChannels(block, x.channels);
    if (block.name == "res_x") {
      // UNetMidBlock3D, built with neither timestep conditioning nor noise
      // injection: `_make_encoder_block` passes neither (video_vae.py:52-60), so
      // both take their `False` defaults (resnet.py:219-220).
      for (int64_t i = 0; i < block.num_layers; ++i) {
        x = ResnetBlock3d(spec, weights, bp + ".res_blocks." + std::to_string(i), x, x.channels,
                          /*inject_noise=*/false, /*timestep_conditioning=*/false, nullptr,
                          nullptr);
      }
    } else if (block.name == "res_x_y") {
      x = ResnetBlock3d(spec, weights, bp, x, out_channels, /*inject_noise=*/false,
                        /*timestep_conditioning=*/false, nullptr, nullptr);
    } else if (block.name == "attn") {
      x = AttnBlock3d(weights, bp, x);
    } else if (block.name == "compress_time" || block.name == "compress_space" ||
               block.name == "compress_all" || block.name == "compress_all_x_y") {
      // Plain strided CausalConv3d (video_vae.py:72-112). `compress_all_x_y` is
      // the only one of the four that changes the channel count.
      const int64_t st = block.name == "compress_space" ? 1 : 2;
      const int64_t ss = block.name == "compress_time" ? 1 : 2;
      x = CausalConv3d(spec.queue, spec.wcache, x, out_channels, 3, spec.causal, spec.spatial_padding_mode,
                       weights.Get(bp + ".conv.weight"), &weights.Get(bp + ".conv.bias"), st, ss,
                       ss);
    } else if (block.name == "compress_all_res" || block.name == "compress_space_res" ||
               block.name == "compress_time_res") {
      const int64_t st = block.name == "compress_space_res" ? 1 : 2;
      const int64_t ss = block.name == "compress_time_res" ? 1 : 2;
      x = SpaceToDepthDownsample(spec, weights, bp, x, st, ss, ss, out_channels);
    } else {
      VT_CHECK(false, "ltx2 video encoder: unknown encoder block `" + block.name + "`");
    }
    ++index;
  }

  // --- conv_norm_out -> SiLU -> conv_out (video_vae.py:239-262, 297-299) ---
  if (config.norm_layer == Ltx2NormLayer::kPixelNorm) {
    PixelNorm(spec.queue, x.data.ptr(), x.channels, x.spatial(), config.pixel_norm_eps);
  } else {
    vt::Queue tail_cpu = VaeCpuQueue();
    vt::Queue& tq = spec.queue != nullptr ? *spec.queue : tail_cpu;
    VaeKernels(tq).group_norm(tq, x.data.ptr(), x.channels, x.spatial(), config.norm_num_groups,
                              spec.wcache->Get(weights.Get(p + "conv_norm_out.weight")),
                              spec.wcache->Get(weights.Get(p + "conv_norm_out.bias")),
                              config.norm_eps, vt::DType::kF32);
  }
  Silu(spec.queue, x.data.ptr(), static_cast<int64_t>(x.data.size()));
  int64_t conv_out_channels = config.out_channels;
  if (config.latent_log_var == Ltx2LogVarianceType::kPerChannel) {
    conv_out_channels *= 2;
  } else if (config.latent_log_var == Ltx2LogVarianceType::kUniform ||
             config.latent_log_var == Ltx2LogVarianceType::kConstant) {
    conv_out_channels += 1;
  }
  x = CausalConv3d(spec.queue, spec.wcache, x, conv_out_channels, 3, spec.causal, spec.spatial_padding_mode,
                   weights.Get(p + "conv_out.conv.weight"), &weights.Get(p + "conv_out.conv.bias"));

  // --- the log-variance fix-ups and the mean split (video_vae.py:301-336) ---
  // Only the MEANS survive, so the fix-ups matter for exactly one reason: they
  // decide WHICH channels the split calls means. kUniform must drop the single
  // trailing logvar channel; kConstant must drop it too. Getting either wrong
  // shifts the whole latent by one channel.
  VT_CHECK(conv_out_channels >= 2,
           "ltx2 video encoder: conv_out must emit at least 2 channels (video_vae.py:308-312)");
  const int64_t latent_channels = config.out_channels;
  VT_CHECK(x.channels >= latent_channels,
           "ltx2 video encoder: conv_out emitted fewer channels than the latent width");

  const std::vector<float>& std_of_means = weights.Get(p + "per_channel_statistics.std-of-means");
  const std::vector<float>& mean_of_means = weights.Get(p + "per_channel_statistics.mean-of-means");
  VT_CHECK(static_cast<int64_t>(std_of_means.size()) == latent_channels &&
               static_cast<int64_t>(mean_of_means.size()) == latent_channels,
           "ltx2 video encoder: per-channel statistics must have one value per latent channel");

  Ltx2LatentVolume out;
  out.batch = 1;
  out.channels = latent_channels;
  out.frames = x.t;
  out.height = x.h;
  out.width = x.w;
  out.data.resize(static_cast<size_t>(out.elems()));
  const int64_t elems = x.spatial();
  // f32: the encoder's normalize is the decoder de-normalize run backwards, and
  // upstream computes it in the activation dtype on both sides.
  for (int64_t c = 0; c < latent_channels; ++c) {
    const float mean = mean_of_means[static_cast<size_t>(c)];
    const float denom = std_of_means[static_cast<size_t>(c)];
    for (int64_t i = 0; i < elems; ++i) {
      out.data[static_cast<size_t>(c * elems + i)] =
          (x.data.Host()[static_cast<size_t>(c * elems + i)] - mean) / denom;
    }
  }
  return out;
}

}  // namespace vllm

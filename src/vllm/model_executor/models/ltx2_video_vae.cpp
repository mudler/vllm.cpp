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
// ─── DTYPE: BOTH ARMS ARE LIVE, AND bf16 IS THE ONE THAT SHIPS ──────────────
//
// A24 wave 3 (row LTX25-A24-VIDEO-VAE-BF16, issue #2786) landed the bf16 arm the
// paragraphs below were written owing. The decode now runs at
// `Ltx2VaeWeights::dtype` and the render loads that bag at `kBF16`, so f32 is the
// parity REFERENCE and not the shipping path. Read what follows as the reason the
// reference exists and as the record of what an f32-only oracle cannot see; the
// six rounding rules the bf16 arm applies are stated at their own sites and
// measured in the row's spec section 4.
//
// WHAT THE bf16 ARM IS NOT BIT-EXACT AGAINST, said here rather than discovered:
// torch BLOCKS its convolution reduction and this port does not, so at the gated
// fixture's shapes the two disagree on 3 to 5 outputs of 8192 to 24576, and this
// chain amplifies one such last bit into 0.0117 at the output. The whole-decode
// bf16 golden therefore carries NO value bound -- a bound wide enough to admit
// the port would admit a real defect, and tests/vllm/models/test_ltx2_vae.cpp
// asserts that relation rather than asserting a number. The arithmetic is gated
// bit-exactly one rule per kernel instead.
//
// ─── DTYPE: THE f32 REFERENCE ARM, AND WHY IT IS NOT WHAT SHIPS ─────────────
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

#include "vllm/model_executor/models/dense_device_glue.h"  // dense_attn::Dev, dense_attn::DBuf
#include "vllm/model_executor/models/minimax_h3.h"
#include "vllm/platforms/interface.h"  // platforms::HasPlatform
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

// ─── THE POOL'S PRECONDITION (#1904) ────────────────────────────────────────
//
// `DBuf` reads the device pool's soft cap off `platforms::GetPlatform(type)`
// (dense_device_glue.h, `ResolveDevicePoolPolicy`), which THROWS for a device
// type the platform registry does not hold. Routing this file's memory through
// the shared pool therefore makes a registered platform a precondition of a
// device decode, where before a registered BACKEND alone was enough (#1904).
//
// The production chain always satisfies it, and by construction rather than by
// luck: `Ltx2VideoEngine::Load` takes the device type from
// `platforms::CurrentPlatform().device_type()` (src/vllm/multimodal/
// ltx2_video.cpp:827-828), and `CurrentPlatform()` returns a REGISTERED entry or
// throws (src/vllm/platforms/platform.cpp:91-98). There is no path from
// `include/vllm.h` to this decode that names a device type any other way.
//
// It is checked anyway, because "the only caller happens to satisfy it" is an
// audit and not a gate, and because the failure it replaces is a `GetPlatform`
// throw three headers away that names neither this decode nor the pool.
//
// ONE CALL SITE, AT THE DECODE ENTRY, and that is a mutation result rather than
// a preference. The first draft called this from BOTH `VaeStore::Alloc` and the
// `VaeWeightCache` constructor, and deleting the `VaeStore` one left the whole
// seam suite green: the cache is constructed first, so it refused first and
// neither call site was individually gated. A guard with a spare copy is a guard
// whose deletion no test can see. `Ltx2ConvVideoDecode` is the one place in this
// translation unit where a device queue arrives, so the check goes there and
// nowhere else. A `VaeStore` reached some other way still throws out of
// `GetPlatform`; what it loses is the message, not the refusal.
void RequirePooledDevice(const vt::Queue& q) {
  VT_CHECK(vllm::platforms::HasPlatform(q.device.type),
           std::string("ltx2 video vae: the decode was handed a queue on device '") +
               vt::DeviceTypeName(q.device.type) +
               "', for which no platform is registered. Its buffers are drawn from the shared "
               "device pool, and the pool's residency cap is platform data, so a device with no "
               "registered platform has no pool policy to run under. Register the platform for "
               "this device type, or decode on the CPU queue.");
}

// ─── THE ELEMENT WIDTH IS THE DELIVERABLE (A24 wave 3, #2786) ──────────────
//
// Upstream constructs the decoder in the pipeline's ONE dtype
// (distilled.py:146-149) and its forward casts the latent to the weights' dtype
// on entry and back on exit (conv_video_decoder.py:283-284, 357). Every volume
// below therefore carries a `vt::DType` and its bytes are that width -- not an
// f32 buffer holding bf16 VALUES, which would move the same bytes it moves today
// and deliver nothing but the arithmetic. `Ltx2VideoFrames::data` at the exit is
// the one f32 container that stays: it is the PUBLIC pixel return, three channels
// wide, and it holds bf16-representable values that `CountWiderThanBf16` on the
// render path gates.
//
// The two free functions are the whole of the width branch. Everything else in
// this file either dispatches through `kLtx2Vae`, which already takes the dtype,
// or reads and writes through these.
inline float LoadElem(const void* base, size_t i, vt::DType dtype) {
  return dtype == vt::DType::kBF16 ? vt::BF16ToF32(static_cast<const uint16_t*>(base)[i])
                                   : static_cast<const float*>(base)[i];
}
inline void StoreElem(void* base, size_t i, float v, vt::DType dtype) {
  if (dtype == vt::DType::kBF16) {
    static_cast<uint16_t*>(base)[i] = vt::F32ToBF16(v);
  } else {
    static_cast<float*>(base)[i] = v;
  }
}
// The two widths this decode serves, refused by name in one place so a third one
// cannot arrive by silence. FP8 and NVFP4 are A22.
void RequireVaeDType(vt::DType dtype) {
  VT_CHECK(dtype == vt::DType::kF32 || dtype == vt::DType::kBF16,
           std::string("ltx2 video vae: the decode serves f32 (the parity arm every committed "
                       "golden is measured against) and bf16 (upstream's own model dtype, "
                       "distilled.py:109) storage. The FP8 and NVFP4 arms are A22 and are not "
                       "implemented; it was handed ") +
               vt::Name(dtype));
}

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
  // this file has applied since W5. `dtype` is the ELEMENT WIDTH and it is the
  // A24 wave 3 deliverable: the bytes this allocates are that width, not an f32
  // buffer holding narrowed values.
  void Alloc(vt::Queue* queue, size_t n, vt::DType dtype) {
    RequireVaeDType(dtype);
    Release();
    n_ = n;
    dtype_ = dtype;
    queue_ = queue;
    if (!OnDevice()) {
      host_.assign(n * vt::SizeOf(dtype_), 0);
      return;
    }
    backend_ = vt::TryGetBackend(queue_->device);
    VT_CHECK(backend_ != nullptr,
             "ltx2 video vae: the decode was handed a queue on a device with no registered "
             "backend, so the volume cannot be made resident on it");
    // THE SHARED SEAM, NOT A SECOND OWNER (#1904). `dense_attn::DBuf` is this
    // tree's move-only owning device allocation and it draws from the shared
    // `DevicePool` (device_pool.h), so a volume this decode is finished with is
    // handed to the next stage that wants that size class instead of being
    // returned to the driver. Both `cudaMalloc` and `cudaFree` synchronise the
    // whole device, and a tiled render calls `Ltx2ConvVideoDecode` once per tile
    // (ltx2_video_vae_tiled.cpp:123), so the raw `Alloc`/`Free` pair this
    // replaces re-paid every one of a tile's allocations on the next tile.
    //
    // The member is a `DBuf` and not a `std::optional<DBuf>`: this store is
    // default-constructed and allocates later, which is what `DBuf()`'s empty
    // state is for. It is NOT a second buffer type -- the allocation, the
    // ownership and the return to the pool are all the seam's.
    //
    // `DBuf` itself rounds a zero-byte request up to one byte
    // (`alloc_bytes_ = bytes_ == 0 ? 1 : bytes_`), which is exactly what the
    // `n_ == 0 ? 1 : ...` here used to do.
    dev_ = vllm::dense_attn::DBuf(vllm::dense_attn::Dev{*backend_, *queue_}, dtype_,
                                  std::vector<int64_t>{static_cast<int64_t>(n_)});
  }

  // Allocate n floats ON THE SAME QUEUE as `other`. Almost every volume in this
  // file is derived from another one, and saying `Like(in.data, n)` cannot pick
  // the wrong queue the way re-deriving it at each site could -- a mismatch
  // there would put an output on the host while its input sits on a device, and
  // on a unified-memory backend that runs and produces the right pixels.
  // ...and at the SAME ELEMENT WIDTH. A derived volume that picked its own dtype
  // could put a bf16 input through an f32 output and the kernel would reinterpret
  // the bytes rather than refuse, because both are `void*` at the seam.
  void Like(const VaeStore& other, size_t n) { Alloc(other.queue_, n, other.dtype_); }

  bool OnDevice() const {
    return queue_ != nullptr && queue_->device.type != vt::DeviceType::kCPU;
  }
  size_t size() const { return n_; }
  vt::Queue* queue() const { return queue_; }
  vt::DType dtype() const { return dtype_; }
  size_t bytes() const { return n_ * vt::SizeOf(dtype_); }

  void* ptr() { return OnDevice() ? dev_.ptr() : static_cast<void*>(host_.data()); }
  const void* ptr() const {
    return OnDevice() ? dev_.ptr() : static_cast<const void*>(host_.data());
  }

  // HOST BYTES AS f32, AND AFTER A24 WAVE 4 THIS IS THE f32 HOST BRANCH OF
  // `AttnBlock3d` AND NOTHING ELSE. That block reads and writes the volume in
  // place when it is host-resident AND stored at f32; every other case takes its
  // `staged` branch, which downloads through `Download`/`Upload` and never asks
  // for these bytes. Both the encoder and the decoder reach it, so this is not an
  // encoder-only accessor and no longer names a width either path is stuck at.
  //
  // The dtype check is what stops a bf16 volume being read through a `float*`:
  // the bytes would reinterpret silently and produce a plausible clip, which no
  // shape-valid gate can see. It is a check rather than a comment for that
  // reason.
  float* Host() { return HostF32(); }
  const float* Host() const { return const_cast<VaeStore*>(this)->HostF32(); }

  // THE ONE NARROWING, AND THE ONE WIDENING. `Upload` takes f32 host values
  // because that is what the decode's prologue and `VaeScratch`'s callers hold,
  // and it narrows ONCE here on the bf16 arm -- which is what `sample.to(dtype)`
  // does on entry (conv_video_decoder.py:284). `Download` widens once on the way
  // out, which is `sample.to(output_dtype)` (`:357`) landing in the public
  // `Ltx2VideoFrames::data`. Neither is a second arithmetic path: the values are
  // already on the bf16 grid by the time they get here on that arm.
  void Upload(const float* host) {
    if (n_ == 0) return;
    if (!OnDevice()) {
      for (size_t i = 0; i < n_; ++i) StoreElem(host_.data(), i, host[i], dtype_);
      return;
    }
    if (dtype_ == vt::DType::kF32) {
      backend_->Copy(*queue_, dev_.ptr(), host, bytes());
      return;
    }
    std::vector<uint16_t> narrow(n_);
    for (size_t i = 0; i < n_; ++i) narrow[i] = vt::F32ToBF16(host[i]);
    backend_->Copy(*queue_, dev_.ptr(), narrow.data(), bytes());
  }
  void Download(float* host) const {
    if (n_ == 0) return;
    if (!OnDevice()) {
      for (size_t i = 0; i < n_; ++i) host[i] = LoadElem(host_.data(), i, dtype_);
      return;
    }
    if (dtype_ == vt::DType::kF32) {
      backend_->Copy(*queue_, host, dev_.ptr(), bytes());
      backend_->Synchronize(*queue_);
      return;
    }
    std::vector<uint16_t> narrow(n_);
    backend_->Copy(*queue_, narrow.data(), dev_.ptr(), bytes());
    backend_->Synchronize(*queue_);
    for (size_t i = 0; i < n_; ++i) host[i] = vt::BF16ToF32(narrow[i]);
  }

 private:
  // The f32 host bytes THEMSELVES, with the width checked rather than assumed --
  // the encoder's gathers write through this pointer, so it cannot be a copy. The
  // reinterpret is the tree's own idiom for a byte-backed tensor
  // (`MaterializeDitTensor`'s callers in ltx2_loader.cpp do the same), and the
  // dtype check is what makes a bf16 volume a REFUSAL here instead of a
  // reinterpretation that reads two elements as one and renders a plausible clip
  // no shape-valid gate can see.
  //
  // IT IS ALSO A STRICT-ALIASING QUESTION, and it is left open deliberately. The
  // buffer is `unsigned char` and this hands out a `float*` that callers WRITE
  // through -- `-O3 -fstrict-aliasing` is in the flags and `sanitize-cpu` does not
  // catch this class. `std::memcpy` cannot replace it without a write-back,
  // because the encoder's gathers mutate the volume in place, so the repair is a
  // refactor of that path rather than a line. Alignment holds (`std::vector`'s
  // allocator meets `max_align_t`) and every access is whole-buffer or through
  // this one accessor. Recorded in the row's `## Owed`; not repaired here.
  float* HostF32() {
    VT_CHECK(!OnDevice(),
             "ltx2 video vae: a host loop asked for the bytes of a volume that is resident on a "
             "device -- the caller must download it first, or be ported to a device arm (#1451)");
    VT_CHECK(dtype_ == vt::DType::kF32,
             "ltx2 video vae: a host f32 loop asked for the bytes of a volume stored at bf16. "
             "The only caller is AttnBlock3d's in-place host branch, which it takes exactly when "
             "the volume is host-resident and stored at f32; a bf16 volume belongs on its staged "
             "branch, which downloads and re-uploads instead");
    return reinterpret_cast<float*>(host_.data());
  }
  void Release() {
    // Assigning the EMPTY buffer returns the block to the pool it came from;
    // there is no `Free` here any more, which is the point of #1904.
    dev_ = vllm::dense_attn::DBuf();
    backend_ = nullptr;
    host_.clear();
    n_ = 0;
    dtype_ = vt::DType::kF32;
    queue_ = nullptr;
  }
  void CopyFrom(const VaeStore& other) {
    Alloc(other.queue_, other.n_, other.dtype_);
    if (n_ == 0) return;
    if (!OnDevice()) {
      host_ = other.host_;
      return;
    }
    // DEVICE TO DEVICE. A copy that went through the host would be exactly the
    // round-trip this row removes, and `Volume hidden = input;` in
    // `ResnetBlock3d` is a copy on the hot path.
    backend_->Copy(*queue_, dev_.ptr(), other.dev_.ptr(), bytes());
  }
  void Steal(VaeStore& other) {
    queue_ = other.queue_;
    backend_ = other.backend_;
    dev_ = std::move(other.dev_);
    host_ = std::move(other.host_);
    n_ = other.n_;
    dtype_ = other.dtype_;
    other.backend_ = nullptr;
    other.n_ = 0;
    other.dtype_ = vt::DType::kF32;
    other.queue_ = nullptr;
  }

  vt::Queue* queue_ = nullptr;
  vt::Backend* backend_ = nullptr;
  vllm::dense_attn::DBuf dev_;
  std::vector<uint8_t> host_;
  size_t n_ = 0;
  vt::DType dtype_ = vt::DType::kF32;
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
// ONE PARAMETER, AT WHATEVER WIDTH THE BAG HOLDS IT (A24 wave 3, #2786).
//
// `Ltx2VaeWeights` carries exactly one of its two maps and `dtype` says which
// (ltx2_audio_vae.h, wave 2). Every weight this file reads goes through here, so
// the arm is resolved from the CHECKPOINT rather than from a flag -- which is
// what upstream does: `weights_dtype = next(self.parameters()).dtype` and then
// `sample.to(weights_dtype)` (conv_video_decoder.py:283-284).
//
// It carries the ELEMENT COUNT because the size checks below are shape checks and
// must hold on both arms, and because `VaeWeightCache` keys on the host address
// and needs the byte length to stage it.
struct VaeParam {
  const void* data = nullptr;
  size_t count = 0;
  vt::DType dtype = vt::DType::kF32;
};

VaeParam Param(const Ltx2VaeWeights& weights, const std::string& name) {
  if (weights.dtype == vt::DType::kBF16) {
    const std::vector<uint16_t>& v = weights.GetBf16(name);
    return VaeParam{v.data(), v.size(), vt::DType::kBF16};
  }
  const std::vector<float>& v = weights.Get(name);
  return VaeParam{v.data(), v.size(), vt::DType::kF32};
}

// One scalar out of a parameter, widened. `timestep_scale_multiplier` is the only
// caller: it is a one-element tensor upstream reads as a tensor, not a config
// value, so it narrows with the module (conv_video_decoder.py:313).
float ParamScalar(const Ltx2VaeWeights& weights, const std::string& name) {
  const VaeParam p = Param(weights, name);
  VT_CHECK(p.count >= 1, "ltx2 video vae: '" + name + "' is empty");
  return LoadElem(p.data, 0, p.dtype);
}

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
  // NO DESTRUCTOR. Each staged weight is a `DBuf` and returns itself to the
  // shared pool (#1904), where the hand-rolled table had to walk itself calling
  // `Backend::Free`. This is what makes the per-tile restaging this cache's
  // lifetime forces (`## Owed`, this row's spec) cost a pool hit rather than a
  // synchronising driver allocation on every tile after the first.
  VaeWeightCache(const VaeWeightCache&) = delete;
  VaeWeightCache& operator=(const VaeWeightCache&) = delete;

  // On the CPU queue this is the host pointer itself and NOTHING is copied,
  // which is what keeps the host arm byte-for-byte the cost it was. It STAGES AT
  // THE PARAMETER'S OWN WIDTH: a bf16 bag sends 16-bit words, which is the whole
  // point of the bf16 arm and not only its arithmetic.
  const void* Get(const VaeParam& p) {
    if (backend_ == nullptr) return p.data;
    auto it = staged_.find(p.data);
    if (it != staged_.end()) return it->second.ptr();
    // `DBuf`'s host-pointer constructor IS the `Alloc` plus `Copy` this replaced:
    // it copies only when the tensor has bytes, and it rounds an empty tensor's
    // allocation up to one byte, both of which the hand-rolled pair did too.
    auto emplaced = staged_.emplace(
        p.data,
        vllm::dense_attn::DBuf(vllm::dense_attn::Dev{*backend_, *queue_}, p.dtype,
                               std::vector<int64_t>{static_cast<int64_t>(p.count)}, p.data));
    return emplaced.first->second.ptr();
  }

 private:
  vt::Queue* queue_ = nullptr;
  vt::Backend* backend_ = nullptr;
  std::map<const void*, vllm::dense_attn::DBuf> staged_;
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
  // `dtype` is the ACTIVATION width, because everything this stages is consumed
  // beside an activation: the spatial-noise plane is added to one and the
  // timestep embedding is added to an ada-LN table that has already narrowed with
  // it. The host values arrive f32 and are narrowed ONCE here on the bf16 arm,
  // which is the same single rounding `VaeStore::Upload` applies to the latent.
  VaeScratch(vt::Queue* queue, const std::vector<float>& host, vt::DType dtype) {
    RequireVaeDType(dtype);
    const bool on_device = queue != nullptr && queue->device.type != vt::DeviceType::kCPU;
    if (!on_device && dtype == vt::DType::kF32) {
      ptr_ = host.data();
      return;
    }
    store_.Alloc(queue, host.size(), dtype);
    store_.Upload(host.data());
    ptr_ = store_.ptr();
  }
  VaeScratch(const VaeScratch&) = delete;
  VaeScratch& operator=(const VaeScratch&) = delete;
  const void* ptr() const { return ptr_; }

 private:
  VaeStore store_;
  const void* ptr_ = nullptr;
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
// The residual add, AT THE VOLUME'S OWN WIDTH. `vt::Add` admits any float output
// (`ops.cpp`, the "add: float in, f32/bf16 out" check) and its CPU arm is
// `LoadF32At` + `StoreF32At` with ONE store rounding -- which is what upstream's
// bf16 `+` is: measured 0 of 500 against `bf16(f32 add)` and against an f64 add
// rounded once, so the width of the addition itself does not separate and only
// the STORE rounding is gated here.
void VaeAddInPlace(vt::Queue& q, void* dst, const void* src, int64_t n, vt::DType dtype) {
  vt::Tensor d = vt::Tensor::Contiguous(dst, dtype, q.device, {n});
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<void*>(src), dtype, q.device, {n});
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
// Its removal did NOT close #1904, and the note that used to stand here said it
// did. `DevBuf` was a hand-rolled second copy of `vllm::dense_attn::DBuf`, and
// deleting it left the same hand-rolled ownership one level up: `VaeStore` and
// `VaeWeightCache` each called `Backend::Alloc` and `Free` themselves, so the
// decode's buffers never reached the shared `DevicePool`. Both hold a `DBuf`
// now -- see `VaeStore::Alloc` and the pool precondition above it -- and that is
// what closed the issue.


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
// THE WIDTH IS THE VOLUME'S, AND THE WEIGHT CARRIES ITS OWN (A24 wave 3, #2786).
// `vt::Conv3d`'s CPU arm already serves bf16 storage: it widens on load, keeps an
// f32 accumulator SEEDED WITH THE BIAS and rounds once on store
// (src/vt/cpu/cpu_conv3d.cpp). That is upstream's own answer, measured -- 3 of
// 18816 words over six seeds against torch's bf16 `F.conv3d`, with `max|diff|`
// one bf16 ulp, where the same kernel with the bias added AFTER the store
// rounding is 5211 of 18816. The residue is torch's blocked reduction order and
// an f64 accumulator does not improve on it.
void Conv3dThroughSeam(vt::Queue* queue, const void* x, int64_t cin, int64_t tin, int64_t hin,
                       int64_t win, const void* weight, const void* bias, int64_t cout,
                       int64_t kernel, const vt::Conv3dArgs& args, void* out, int64_t tout,
                       int64_t hout, int64_t wout, vt::DType dtype, vt::DType weight_dtype) {
  vt::Queue cpu_queue = VaeCpuQueue();
  vt::Queue& q = queue != nullptr ? *queue : cpu_queue;
  const int64_t wrows = cout * (cin / args.groups);
  vt::Tensor tx =
      vt::Tensor::Contiguous(const_cast<void*>(x), dtype, q.device, {cin, tin, hin, win});
  vt::Tensor tw = vt::Tensor::Contiguous(const_cast<void*>(weight), weight_dtype, q.device,
                                         {wrows, kernel, kernel, kernel});
  vt::Tensor to = vt::Tensor::Contiguous(out, dtype, q.device, {cout, tout, hout, wout});
  vt::Tensor tb;
  if (bias != nullptr) {
    tb = vt::Tensor::Contiguous(const_cast<void*>(bias), weight_dtype, q.device, {cout});
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
                    const VaeParam& weight, const VaeParam& bias = VaeParam{},
                    int64_t stride_t = 1, int64_t stride_h = 1, int64_t stride_w = 1) {
  const int64_t ci = in.channels;
  VT_CHECK(stride_t >= 1 && stride_h >= 1 && stride_w >= 1, "ltx2 conv3d: stride must be positive");
  VT_CHECK(static_cast<int64_t>(in.data.size()) == ci * in.spatial(),
           "ltx2 conv3d: input size does not match [C, T, H, W]");
  VT_CHECK(static_cast<int64_t>(weight.count) == out_channels * ci * kernel * kernel * kernel,
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
                     pad_spatial, static_cast<int>(mode), in.data.dtype());

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
  // A DEFAULT-CONSTRUCTED `VaeParam` IS "NO BIAS", which is what the `const
  // std::vector<float>*` this replaces spelled as a null pointer. It is a
  // sentinel on `data` rather than a separate flag so a caller cannot pass a
  // shaped bias and forget to say it has one.
  Conv3dThroughSeam(queue, padded.ptr(), ci, pt, ph, pw, wcache->Get(weight),
                    bias.data != nullptr ? wcache->Get(bias) : nullptr, out_channels, kernel, args,
                    out.data.ptr(), out.t, out.h, out.w, in.data.dtype(), weight.dtype);
  return out;
}

// make_linear_nd for dims == 3 (convolution.py:84-85): a 1x1x1 Conv3d.
Volume Linear3d(vt::Queue* queue, VaeWeightCache* wcache, const Volume& in, int64_t out_channels,
                const VaeParam& weight, const VaeParam& bias) {
  const int64_t n = in.spatial();
  VT_CHECK(static_cast<int64_t>(weight.count) == out_channels * in.channels,
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
  // BIT-EXACT at bf16 and measured so: a 1x1x1 reduction has no association order
  // to differ in, and this kernel is 0 of 4096 against torch's own bf16 `F.conv3d`
  // over four seeds, where the 3x3x3 case leaves 3 of 18816.
  VaeKernels(q).linear_cn(q, out.data.ptr(), in.data.ptr(), wcache->Get(weight),
                          wcache->Get(bias), out_channels, in.channels, n, in.data.dtype());
  return out;
}

// f32: `F.silu` computes in the activation dtype. The kernel spells it
// x / (1 + exp(-x)) in float, which is what this loop was; the algebraically
// equivalent x * sigmoid(x) is NOT bit-identical and the goldens are held to the
// first.
void Silu(vt::Queue* queue, void* x, int64_t n, vt::DType dtype) {
  vt::Queue cpu = VaeCpuQueue();
  vt::Queue& q = queue != nullptr ? *queue : cpu;
  // NO NEW BRANCH IS NEEDED AT bf16 AND THAT IS A MEASUREMENT, not an assumption.
  // `Ltx2Silu`'s bf16 arm already widens on load, evaluates x/(1+exp(-x)) in f32
  // and rounds once on store, and upstream's `F.silu` on a bf16 tensor is exactly
  // that: 0 of 4000 against `bf16(F.silu(f32))` and 0 of 4000 against the f32
  // expression rounded once, where a FULLY bf16 chain is 1155 of 4000. The two
  // zero rows did not separate from each other, and that is recorded rather than
  // read as two confirmations.
  VaeSiluKernels(q).silu(q, x, n, dtype);
}

// PixelNorm() with its DEFAULT eps of 1e-8 (normalization.py:22, reached bare
// from video_vae/resnet.py:46 and conv_video_decoder.py:243) — NOT the 1e-6 the
// audio VAE gets through build_normalization_layer.
void PixelNorm(vt::Queue* queue, void* x, int64_t channels, int64_t spatial, double eps,
               vt::DType dtype) {
  vt::Queue cpu = VaeCpuQueue();
  vt::Queue& q = queue != nullptr ? *queue : cpu;
  // `eps` stays an f64 PARAMETER because it is a pinned config threshold
  // (Ltx2ConvVideoDecoderConfig::pixel_norm_eps). It is narrowed here, at the one
  // point it enters the arithmetic — which is where the host loop narrowed it.
  VaeKernels(q).pixel_norm(q, x, channels, spatial, static_cast<float>(eps), dtype);
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
void ApplyNorm(const VideoConvSpec& config, void* x, int64_t channels, int64_t spatial,
               const Ltx2VaeWeights& weights, const std::string& prefix, vt::DType dtype) {
  vt::Queue cpu = VaeCpuQueue();
  vt::Queue& q = config.queue != nullptr ? *config.queue : cpu;
  if (config.norm_layer == Ltx2NormLayer::kPixelNorm) {
    PixelNorm(config.queue, x, channels, spatial, config.pixel_norm_eps, dtype);
    return;
  }
  VT_CHECK(config.wcache != nullptr, "ltx2 video vae: a norm was reached with no weight cache");
  // THE AFFINE'S OWN WIDTH IS 27% OF THE OUTPUT AT bf16, and it arrives narrowed
  // because the BAG is narrowed. Upstream's `nn.GroupNorm` at bf16 holds
  // bf16 `weight` and `bias` and folds them into an f32-statistics affine:
  // against torch's own module, f64 statistics with bf16-narrowed affine is
  // 0/1600 (C=32,G=4), 0/24576 (C=128,G=32) and 0/1600 (G=1, which is `norm3`),
  // where the SAME statistics with f32 weight and bias are 463, 6664 and 461. The
  // statistics' own width does not separate at all -- f32 and f64 agree
  // everywhere -- so this kernel keeps the f64 accumulators every committed
  // golden was taken through.
  VaeKernels(q).group_norm(q, x, channels, spatial, config.norm_num_groups,
                           config.wcache->Get(Param(weights, prefix + ".weight")),
                           config.wcache->Get(Param(weights, prefix + ".bias")), config.norm_eps,
                           dtype);
}

// ---------------------------------------------------------------------------
// PixArtAlphaCombinedTimestepSizeEmbeddings (timestep_embedding.py:118-141) at
// batch 1: Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0) followed
// by TimestepEmbedding(256 -> embedding_dim) with a SiLU between its two linears.
// ---------------------------------------------------------------------------
// `dtype` IS THE ACTIVATION'S, because upstream passes `hidden_dtype=sample.dtype`
// (conv_video_decoder.py:331-334, resnet.py's UNetMidBlock3D likewise). At bf16
// this is NOT `bf16(the f32 module)`: measured on embedding_dim=32, the bf16
// module differs from the f32 one rounded once on 12 of 32 values. So the two
// linears, the SiLU between them and the frequency table's entry into the
// arithmetic all round at the activation width, and the f64 table below is
// narrowed at the point it becomes a datum rather than being carried wide through
// the accumulation.
std::vector<float> TimestepEmbedding(double timestep, int64_t embedding_dim,
                                     const Ltx2VaeWeights& weights, const std::string& prefix,
                                     vt::DType dtype) {
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

  const VaeParam w1 = Param(weights, prefix + ".timestep_embedder.linear_1.weight");
  const VaeParam b1 = Param(weights, prefix + ".timestep_embedder.linear_1.bias");
  const VaeParam w2 = Param(weights, prefix + ".timestep_embedder.linear_2.weight");
  const VaeParam b2 = Param(weights, prefix + ".timestep_embedder.linear_2.bias");
  VT_CHECK(static_cast<int64_t>(w1.count) == embedding_dim * kProjChannels,
           "ltx2 timestep embedding: linear_1 shape does not match the embedding dim");
  // `Round` is the ONE rounding point of this function, spelled once so every use
  // is visible. At f32 it is the identity, which is what keeps this arm
  // byte-identical to the committed goldens.
  const auto Round = [dtype](float v) {
    return dtype == vt::DType::kBF16 ? vt::BF16ToF32(vt::F32ToBF16(v)) : v;
  };

  // f32 for both `nn.Linear` accumulators and for the hidden activation between
  // them: upstream's TimestepEmbedder is two plain Linears with a SiLU, all in
  // the activation dtype. The frequency table above stays f64 — see its note.
  std::vector<float> hidden(static_cast<size_t>(embedding_dim));
  for (int64_t o = 0; o < embedding_dim; ++o) {
    float acc = LoadElem(b1.data, static_cast<size_t>(o), b1.dtype);
    for (int64_t i = 0; i < kProjChannels; ++i) {
      // The f64 table becomes a datum HERE, at the activation width. It is the
      // same narrowing upstream's `torch.arange`-built f32 table gets when the
      // module is `.to(bfloat16)`.
      acc += Round(static_cast<float>(proj[static_cast<size_t>(i)])) *
             LoadElem(w1.data, static_cast<size_t>(o * kProjChannels + i), w1.dtype);
    }
    acc = Round(acc);
    hidden[static_cast<size_t>(o)] = Round(acc / (1.0f + std::exp(-acc)));  // SiLU
  }
  std::vector<float> out(static_cast<size_t>(embedding_dim));
  for (int64_t o = 0; o < embedding_dim; ++o) {
    float acc = LoadElem(b2.data, static_cast<size_t>(o), b2.dtype);
    for (int64_t i = 0; i < embedding_dim; ++i) {
      acc += hidden[static_cast<size_t>(i)] *
             LoadElem(w2.data, static_cast<size_t>(o * embedding_dim + i), w2.dtype);
    }
    out[static_cast<size_t>(o)] = Round(acc);
  }
  return out;
}

// _feed_spatial_noise (resnet.py:104-119): ONE [H, W] draw, broadcast over batch,
// channels and TIME, scaled per channel. Drawing a full [C, T, H, W] block
// instead still yields a finite, plausible clip.
void FeedSpatialNoise(vt::Queue* queue, VaeWeightCache* wcache, Volume& x,
                      const VaeParam& per_channel_scale, Ltx2NoiseStream* noise) {
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
  // THE PLANE NARROWS WITH THE ACTIVATION, which is what upstream's
  // `dtype=hidden_states.dtype` does to its own draw (resnet.py:115). It is NOT
  // the bf16 DRAW: `torch.randn(dtype=bfloat16)` is a different sequence from
  // `torch.randn(dtype=float32)` at the same seed, not the f32 stream rounded,
  // and switching to it would change every render digest this repository has
  // captured. That decision is #2780 and is the developer's; this arm keeps
  // `Ltx2NoiseStream`'s f32 sequence and narrows each drawn value, which is a
  // DELIBERATE DIVERGENCE from conv_video_decoder.py:288-294 and resnet.py:115,
  // recorded here and in .agents/specs/ltx25-a24-video-vae-bf16.md section 7.
  const VaeScratch plane_dev(queue, plane, x.data.dtype());
  VaeKernels(q).spatial_noise(q, x.data.ptr(), plane_dev.ptr(),
                              wcache->Get(per_channel_scale), x.channels, x.t, x.h, x.w,
                              x.data.dtype());
}

// One ada-LN group applied in place: x * (1 + scale) + shift, with the pair taken
// from `table[row]` plus `embed[row]` (resnet.py:135-147).
void ApplyAdaLn(vt::Queue* queue, VaeWeightCache* wcache, Volume& x, const VaeParam& table,
                const std::vector<float>& embed, int64_t rows, int64_t shift_row,
                int64_t scale_row) {
  const int64_t c = x.channels;
  VT_CHECK(static_cast<int64_t>(table.count) == rows * c,
           "ltx2 video vae: scale_shift_table does not match the channel count");
  VT_CHECK(static_cast<int64_t>(embed.size()) == rows * c,
           "ltx2 video vae: timestep embedding does not match rows x channels");
  vt::Queue cpu = VaeCpuQueue();
  vt::Queue& q = queue != nullptr ? *queue : cpu;
  VT_CHECK(wcache != nullptr, "ltx2 video vae: ada-LN was reached with no weight cache");
  // The table narrows with the activation (`.to(device=..., dtype=...)`,
  // conv_video_decoder.py:336-337 and resnet.py:133-135) and so does the
  // embedding, which was already computed at that width.
  const VaeScratch embed_dev(queue, embed, x.data.dtype());
  VaeKernels(q).ada_ln(q, x.data.ptr(), wcache->Get(table), embed_dev.ptr(), c, x.spatial(), rows,
                       shift_row, scale_row, x.data.dtype());
}

// ResnetBlock3D.forward (resnet.py:121-186).
Volume ResnetBlock3d(const VideoConvSpec& config, const Ltx2VaeWeights& weights,
                     const std::string& prefix, const Volume& input, int64_t out_channels,
                     bool inject_noise, bool timestep_conditioning,
                     const std::vector<float>* timestep_embed, Ltx2NoiseStream* noise) {
  Volume hidden = input;
  const vt::DType dt = hidden.data.dtype();
  ApplyNorm(config, hidden.data.ptr(), hidden.channels, hidden.spatial(), weights,
            prefix + ".norm1", dt);
  if (timestep_conditioning) {
    VT_CHECK(timestep_embed != nullptr,
             "ltx2 video vae: a timestep-conditioned block needs a timestep embedding");
    // ada_values rows are (shift1, scale1, shift2, scale2).
    ApplyAdaLn(config.queue, config.wcache, hidden, Param(weights, prefix + ".scale_shift_table"),
               *timestep_embed, 4, 0, 1);
  }
  Silu(config.queue, hidden.data.ptr(), static_cast<int64_t>(hidden.data.size()), dt);
  hidden = CausalConv3d(config.queue, config.wcache, hidden, out_channels, 3, config.causal,
                        config.spatial_padding_mode, Param(weights, prefix + ".conv1.conv.weight"),
                        Param(weights, prefix + ".conv1.conv.bias"));
  if (inject_noise) {
    FeedSpatialNoise(config.queue, config.wcache, hidden,
                     Param(weights, prefix + ".per_channel_scale1"), noise);
  }

  ApplyNorm(config, hidden.data.ptr(), hidden.channels, hidden.spatial(), weights,
            prefix + ".norm2", dt);
  if (timestep_conditioning) {
    ApplyAdaLn(config.queue, config.wcache, hidden, Param(weights, prefix + ".scale_shift_table"),
               *timestep_embed, 4, 2, 3);
  }
  Silu(config.queue, hidden.data.ptr(), static_cast<int64_t>(hidden.data.size()), dt);
  hidden = CausalConv3d(config.queue, config.wcache, hidden, out_channels, 3, config.causal,
                        config.spatial_padding_mode, Param(weights, prefix + ".conv2.conv.weight"),
                        Param(weights, prefix + ".conv2.conv.bias"));
  if (inject_noise) {
    FeedSpatialNoise(config.queue, config.wcache, hidden,
                     Param(weights, prefix + ".per_channel_scale2"), noise);
  }

  Volume residual = input;
  if (input.channels != out_channels) {
    // norm3 is GroupNorm with ONE group — a LayerNorm over (C, T, H, W) that
    // works in the (B, C, ...) layout without a rearrange (resnet.py:91-97).
    vt::Queue norm3_cpu = VaeCpuQueue();
    vt::Queue& n3q = config.queue != nullptr ? *config.queue : norm3_cpu;
    VaeKernels(n3q).group_norm(n3q, residual.data.ptr(), residual.channels, residual.spatial(), 1,
                               config.wcache->Get(Param(weights, prefix + ".norm3.weight")),
                               config.wcache->Get(Param(weights, prefix + ".norm3.bias")),
                               config.norm_eps, residual.data.dtype());
    residual = Linear3d(config.queue, config.wcache, residual, out_channels,
                        Param(weights, prefix + ".conv_shortcut.weight"),
                        Param(weights, prefix + ".conv_shortcut.bias"));
  }
  VT_CHECK(residual.data.size() == hidden.data.size(),
           "ltx2 video vae: resnet residual and main-branch shapes must match");
  {
    vt::Queue res_cpu = VaeCpuQueue();
    vt::Queue& rq = config.queue != nullptr ? *config.queue : res_cpu;
    VaeAddInPlace(rq, hidden.data.ptr(), residual.data.ptr(),
                  static_cast<int64_t>(hidden.data.size()), dt);
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
    // SHAPE MOVEMENT, DTYPE-TRANSPARENT. One source element per destination
    // element, no reduction and no rounding, so the bf16 arm is the same gather
    // over a 2-byte element.
    VaeKernels(q).depth_to_space(q, out.data.ptr(), packed.data.ptr(), out.channels, packed.t,
                                 packed.h, packed.w, st, sh, sw, packed.data.dtype());
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
                              /*drop=*/1, v.data.dtype());
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
                                 expanded.spatial(), repeat, expanded.data.dtype());
    skip = st == 2 ? drop_first_frame(repeated) : repeated;
  }

  Volume packed = CausalConv3d(config.queue, config.wcache, x, conv_out_channels, 3, config.causal,
                               config.spatial_padding_mode,
                               Param(weights, prefix + ".conv.conv.weight"),
                               Param(weights, prefix + ".conv.conv.bias"));
  Volume out = expand(packed);
  if (st == 2) out = drop_first_frame(out);
  if (residual) {
    VT_CHECK(skip.data.size() == out.data.size(),
             "ltx2 video vae: depth-to-space residual and main-branch shapes must match");
    VaeAddInPlace(q, out.data.ptr(), skip.data.ptr(), static_cast<int64_t>(out.data.size()),
                  out.data.dtype());
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
  const VaeParam gamma = Param(weights, prefix + ".norm.gamma");
  const VaeParam qkv_w = Param(weights, prefix + ".to_qkv.weight");
  const VaeParam qkv_b = Param(weights, prefix + ".to_qkv.bias");
  const VaeParam proj_w = Param(weights, prefix + ".proj.weight");
  const VaeParam proj_b = Param(weights, prefix + ".proj.bias");
  const double norm_scale = std::sqrt(static_cast<double>(c));
  const double attn_scale = 1.0 / std::sqrt(static_cast<double>(c));
  const vt::DType dt = x.data.dtype();
  // The ONE rounding point of this block, spelled once. At f32 it is the
  // identity, which is what keeps this arm byte-identical to the committed
  // goldens.
  const auto Round = [dt](float v) {
    return dt == vt::DType::kBF16 ? vt::BF16ToF32(vt::F32ToBF16(v)) : v;
  };

  // See this function's header for why this stage is still on the host.
  //
  // THE HOST ARM STILL MOVES NO EXTRA BYTE. `Volume out = x` is the one copy
  // this block always made, and on the CPU queue the loops below read and write
  // those bytes in place exactly as they did before this row. The download and
  // the re-upload happen only when the volume is genuinely resident, which is
  // the case this stage has not been ported for. That is an ALLOCATION branch,
  // not a second arithmetic path: there is one copy of the loops below and both
  // devices run it.
  //
  // THE f32 HOST ARM STILL MOVES NO EXTRA BYTE, and the bf16 one pays exactly one
  // widen and one narrow. `staged` is taken when the volume is resident on a
  // device -- the case this stage was never ported for -- OR when its storage is
  // bf16, because a `float*` cannot alias 16-bit words. The working values are
  // f32 either way and that is NOT a width escape: the volume's own bytes stay at
  // the volume's dtype and every rounding point below is applied explicitly
  // through `Round`, which is what `LoadF32At`/`StoreF32At` do inside every
  // kernel in this tree. Upstream's own bf16 kernels widen internally too --
  // `torch.mean` on bf16 accumulates wider than bf16 on 20 of 32 values.
  Volume out = x;
  const bool staged = x.data.OnDevice() || dt != vt::DType::kF32;
  std::vector<float> staged_in, staged_out;
  const float* xh = nullptr;
  float* outh = nullptr;
  if (staged) {
    staged_in.resize(x.data.size());
    x.data.Download(staged_in.data());
    staged_out = staged_in;
    xh = staged_in.data();
    outh = staged_out.data();
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
      // `F.normalize(x, dim=1)` accumulates its L2 norm WIDER than the tensor and
      // then ROUNDS THE DENOMINATOR to the tensor dtype before dividing -- it is
      // `input / input.norm(...).clamp_min(eps)`, and `norm` on a bf16 tensor
      // returns bf16. Measured against upstream on [3,64,5,5] at two scales: f32
      // accumulate with the denominator rounded is 0 of 4800 at both, while
      // keeping the denominator in f32 is 1379 and 1469, an f64 accumulator is
      // the same 1379/1469, and a fully bf16 chain is 749/779. torch's 1e-12
      // floor stays f64 -- it is a threshold, not a datum.
      float sum_sq = 0.0f;
      for (int64_t ch = 0; ch < c; ++ch) {
        const float value = xh[x.At(ch, frame, i / x.w, i % x.w)];
        sum_sq += value * value;
      }
      const float denom = Round(static_cast<float>(
          std::max(std::sqrt(static_cast<double>(sum_sq)), kLtx2RmsNorm2dEps)));
      for (int64_t ch = 0; ch < c; ++ch) {
        // THE GAIN IS FORMED FIRST AND MULTIPLIED ONCE. `_RMSNorm2D` is
        // `F.normalize(x, dim=1) * (self.scale * self.gamma)` (attention.py:23):
        // the parenthesis binds `sqrt(C) * gamma` into ONE bf16 tensor before the
        // activation ever sees it. Our previous order -- normalize, then
        // `* sqrt(C)`, then `* gamma` -- is 1135 of 3600 words wrong at C=48.
        //
        // AND IT ONLY SEPARATES AT A CHANNEL COUNT WHOSE SQUARE ROOT IS NOT A
        // POWER OF TWO. At C=64, `sqrt(64) = 8` is exact in bf16 and all three
        // orderings agree on 4800 of 4800, so a probe built on the shipped width
        // would gate nothing. The row's generator lays a C=48 arm for exactly
        // this, and states the C=64 control as a NON-separation.
        const float gain =
            Round(static_cast<float>(norm_scale) * LoadElem(gamma.data, static_cast<size_t>(ch),
                                                            gamma.dtype));
        const float unit = Round(xh[x.At(ch, frame, i / x.w, i % x.w)] / denom);
        normed[static_cast<size_t>(ch * n + i)] = Round(unit * gain);
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
        // `to_qkv` is a 1x1 nn.Conv2d (attention.py:55): the bias SEEDS the
        // accumulator, the reduction is wider than the storage and there is one
        // rounding on store -- `vt::Conv3d`'s published contract, which the 1x1x1
        // case reproduces bit-exactly at bf16 (0 of 4096 over four seeds).
        float acc = LoadElem(qkv_b.data, static_cast<size_t>(oc), qkv_b.dtype);
        for (int64_t ic = 0; ic < c; ++ic) {
          acc += normed[static_cast<size_t>(ic * n + i)] *
                 LoadElem(qkv_w.data, static_cast<size_t>(oc * c + ic), qkv_w.dtype);
        }
        dst[static_cast<size_t>(row * n + i)] = Round(acc);
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
        // SDPA returns a tensor of the INPUT dtype, so the attention output
        // rounds once. The scores, the softmax and this weighted sum stay f32:
        // upstream's `SDPBackend.MATH` on bf16 operands is "f32 throughout with a
        // blocked reduction" -- three rounding-point hypotheses were rejected
        // against it, and MATH itself sits 3 to 9 words of 8192 to 32768 away
        // from an f32-accumulated attention rounded once, which is a reduction
        // order and not a width.
        attended[static_cast<size_t>(ch * n + i)] = Round(acc / sum);
      }
    }
    // f32: `proj` is a 1x1 nn.Conv2d (attention.py:56).
    for (int64_t oc = 0; oc < c; ++oc) {
      for (int64_t i = 0; i < n; ++i) {
        float acc = LoadElem(proj_b.data, static_cast<size_t>(oc), proj_b.dtype);
        for (int64_t ic = 0; ic < c; ++ic) {
          acc += attended[static_cast<size_t>(ic * n + i)] *
                 LoadElem(proj_w.data, static_cast<size_t>(oc * c + ic), proj_w.dtype);
        }
        // `identity + x` (attention.py:69) is a separate op from the projection,
        // so the projection rounds and then the residual add rounds.
        const size_t at = out.At(oc, frame, i / x.w, i % x.w);
        outh[at] = Round(outh[at] + Round(acc));
      }
    }
  }
  if (staged) out.data.Upload(outh);
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
  // The decode's memory comes from the shared device pool (#1904), and the pool
  // is platform data. This is the one place a device queue enters this file, so
  // it is the one place that precondition is stated.
  if (queue != nullptr && queue->device.type != vt::DeviceType::kCPU) {
    RequirePooledDevice(*queue);
  }
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
  // THE ARM IS RESOLVED FROM THE CHECKPOINT, NOT FROM A FLAG, which is what
  // upstream does: `weights_dtype = next(self.parameters()).dtype` and then
  // `sample.to(weights_dtype)` (conv_video_decoder.py:283-284). A caller that
  // loaded the bag at bf16 gets a bf16 decode; one that loaded it at f32 gets the
  // reference arm, byte for byte as before this row.
  const vt::DType dt = weights.dtype;
  RequireVaeDType(dt);
  // THE REFUSAL AND THE ROUTE PREDICATE ARE THE SAME PREDICATE, which is the trap
  // this project has paid for before. `vt::Conv3d`'s CUDA arm refuses f16/bf16
  // storage by name (src/vt/cuda/cuda_conv3d.cu, "this arm serves f32 only") and
  // EVERY convolution of this decode goes through it, so a bf16 volume on a
  // device queue cannot get past `conv_in`. Asking here, once, is what stops that
  // being discovered three headers away as a kernel-level refusal that names
  // neither this decode nor the dtype it was asked for.
  if (dt == vt::DType::kBF16 && queue != nullptr && queue->device.type != vt::DeviceType::kCPU) {
    VT_CHECK(false,
             std::string("ltx2 video vae: a bf16 decode was requested on device '") +
                 vt::DeviceTypeName(queue->device.type) +
                 "', and only the CPU arm serves it. `vt::Conv3d` has no bf16 storage arm on that "
                 "device (#1007) and every convolution here goes through it, so the bf16 arm "
                 "cannot be reached there at all. Decode on the CPU queue, or load the VAE "
                 "weights at f32. The device bf16 arm is owed -- see `## Owed` in "
                 ".agents/specs/ltx25-a24-video-vae-bf16.md (#2786)");
  }
  // `Round` is the prologue's one rounding point. At f32 it is the identity,
  // which is what keeps the reference arm byte-identical to its goldens.
  const auto Round = [dt](float v) {
    return dt == vt::DType::kBF16 ? vt::BF16ToF32(vt::F32ToBF16(v)) : v;
  };
  // `sample = sample.to(weights_dtype)` (conv_video_decoder.py:284) happens BEFORE
  // the noise blend and before `un_normalize`, so the latent is already on the
  // weights' grid when the prologue's arithmetic starts. Narrowing it only at the
  // upload -- which is where this function puts the volume on the device -- would
  // run two rounding-sensitive expressions on an f32 stream and round once at the
  // end, and that is a different number.
  std::vector<float> staged(latent.size());
  for (size_t i = 0; i < latent.size(); ++i) staged[i] = Round(latent[i]);

  // --- noise + denormalize (conv_video_decoder.py:286-301) ---
  if (config.timestep_conditioning) {
    VT_CHECK(noise != nullptr,
             "ltx2 video vae: timestep conditioning injects noise but no noise stream was supplied");
    const std::vector<float> drawn = noise->Draw(static_cast<int64_t>(staged.size()));
    VT_CHECK(drawn.size() == staged.size(),
             "ltx2 video vae: the noise stream returned the wrong element count");
    // The blend runs in the activation dtype upstream, and the two scalars are
    // config values narrowed once rather than per element.
    //
    // THE DRAW ITSELF IS NOT UPSTREAM'S AT bf16, DELIBERATELY. Upstream draws at
    // `dtype=sample.dtype` (conv_video_decoder.py:288-294), and
    // `torch.randn(dtype=bfloat16)` at a fixed seed is a DIFFERENT SEQUENCE from
    // `torch.randn(dtype=float32)` at that seed -- not the f32 stream rounded.
    // Mirroring it would change every render digest this repository has captured,
    // including the adherence campaign's byte-identical control. That decision is
    // #2780 and it is the developer's. This arm keeps `Ltx2NoiseStream`'s f32
    // sequence and narrows each drawn value, which is a recorded divergence and
    // not an oversight; the same divergence applies at `FeedSpatialNoise`.
    // THE TWO SCALARS ARE NOT NARROWED, AND THAT IS THE OPPOSITE OF THE RULE ONE
    // BLOCK BELOW. `self.decode_noise_scale` is a PYTHON FLOAT, so
    // `noise * self.decode_noise_scale` and `(1.0 - self.decode_noise_scale) *
    // sample` go through torch's scalar path, whose compute type for a bf16
    // tensor is f32: the scalar reaches the multiply at f32 and only the RESULT
    // rounds. Narrowing it first is wrong on 576 of 2000 values -- `bf16(0.975)`
    // is 0.9765625, a whole 2^-9 away from 0.975 -- and it was wrong here, which
    // is what the shallow bf16 arm caught.
    //
    // A REGISTERED BUFFER GOES THE OTHER WAY IN THIS SAME FUNCTION: the
    // per-channel statistics below are tensors and `.to(x)` narrows them
    // (ops.py:76-79), which is 1294 of 4096 values if it is skipped. And
    // `PixelNorm`'s epsilon, ALSO a Python float, IS narrowed for its add -- the
    // port is bit-exact against upstream with `bf16(1e-8)` and 7 of 144 words
    // away with `1e-8` at a row scale of 2^-14. Three scalars, three answers, in
    // two files, and this is A24 wave 1's finding in a third component: none of
    // them may be read off the source.
    const float noise_scale = static_cast<float>(config.decode_noise_scale);
    const float keep_scale = static_cast<float>(1.0 - config.decode_noise_scale);
    for (size_t i = 0; i < staged.size(); ++i) {
      staged[i] = Round(Round(Round(drawn[i]) * noise_scale) + Round(keep_scale * staged[i]));
    }
  }
  {
    const VaeParam std_of_means = Param(weights, p + "per_channel_statistics.std-of-means");
    const VaeParam mean_of_means = Param(weights, p + "per_channel_statistics.mean-of-means");
    VT_CHECK(static_cast<int64_t>(std_of_means.count) == latent_channels &&
                 static_cast<int64_t>(mean_of_means.count) == latent_channels,
             "ltx2 video vae: per-channel statistics must have one value per latent channel");
    const int64_t n = latent_t * latent_h * latent_w;
    // THE STATISTICS NARROW BEFORE THE MULTIPLY, and this is the decoder's VERY
    // FIRST arithmetic. `un_normalize` is
    // `(x * std.view(...).to(x)) + mean.view(...).to(x)` (ops.py:76-79): `.to(x)`
    // rounds both registered buffers to the activation dtype, and the multiply
    // and the add are separate ops that each round.
    //
    // Keeping the statistics in f32 -- which is what this loop did before -- is
    // 109 of 288 words wrong at C=16 and 1294 of 4096 at C=128, measured against
    // upstream's own bf16 module. A fused single-expression form is 125 and 1587.
    // No token gate can see any of it.
    for (int64_t c = 0; c < latent_channels; ++c) {
      const float std_c = LoadElem(std_of_means.data, static_cast<size_t>(c), std_of_means.dtype);
      const float mean_c =
          LoadElem(mean_of_means.data, static_cast<size_t>(c), mean_of_means.dtype);
      for (int64_t i = 0; i < n; ++i) {
        staged[static_cast<size_t>(c * n + i)] =
            Round(Round(staged[static_cast<size_t>(c * n + i)] * std_c) + mean_c);
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
  x.data.Alloc(queue, staged.size(), dt);
  x.data.Upload(staged.data());

  // BOTH OPERANDS NARROW, AND THE PRODUCT ROUNDS. Upstream builds the timestep
  // tensor at the activation dtype (`torch.full(..., dtype=sample.dtype)`,
  // conv_video_decoder.py:304-305) and multiplies it by
  // `self.timestep_scale_multiplier.to(sample)` (`:313`), which narrows the
  // registered buffer too. At `decode_timestep = 0.05` a multiplier of 1000 gives
  // the same SCALAR either way and one of 7.3 does not -- 0.365625 wide against
  // 0.365234375 through the bf16 chain -- so a multiplier that happens to be
  // exactly representable would hide the rule entirely.
  //
  // A SEPARATING SCALAR IS NOT A SEPARATING OUTPUT, and 7.3 is the counterexample
  // rather than the demonstration. The generator swept the multiplier
  // (gen-ltx2-vae-goldens.py, section 5i) and 7.3 separates the two scalars above
  // and ZERO of the arm's 144 outputs, because a 0.1% move in a 0.365-radian
  // angle is under half a bf16 ulp everywhere the sinusoid lands. 3.7, 23.7, 41.3,
  // 499.7 and the shipped 1000 also separate zero of them. The gated arm uses
  // 113.7, which separates 119 of 144. The shipped checkpoint's value is not known
  // to this row (`## Owed`), which is why the narrowing is applied rather than
  // argued away.
  const double scaled_timestep =
      config.timestep_conditioning
          ? static_cast<double>(
                Round(Round(static_cast<float>(timestep != nullptr ? *timestep
                                                                  : config.decode_timestep)) *
                      Round(ParamScalar(weights, p + "timestep_scale_multiplier"))))
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
                   config.spatial_padding_mode, Param(weights, p + "conv_in.conv.weight"),
                   Param(weights, p + "conv_in.conv.bias"));

  // --- the reversed block walk (conv_video_decoder.py:222-238, 315-326) ---
  int64_t index = 0;
  for (auto it = config.decoder_blocks.rbegin(); it != config.decoder_blocks.rend(); ++it, ++index) {
    const Ltx2VideoDecoderBlock& block = *it;
    const std::string bp = p + "up_blocks." + std::to_string(index);
    if (block.name == "res_x") {
      std::vector<float> embed;
      const std::vector<float>* embed_ptr = nullptr;
      if (config.timestep_conditioning) {
        embed = TimestepEmbedding(scaled_timestep, x.channels * 4, weights, bp + ".time_embedder",
                                  dt);
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
    PixelNorm(spec.queue, x.data.ptr(), x.channels, x.spatial(), config.pixel_norm_eps, dt);
  } else {
    vt::Queue tail_cpu = VaeCpuQueue();
    vt::Queue& tq = spec.queue != nullptr ? *spec.queue : tail_cpu;
    VaeKernels(tq).group_norm(tq, x.data.ptr(), x.channels, x.spatial(), config.norm_num_groups,
                              spec.wcache->Get(Param(weights, p + "conv_norm_out.weight")),
                              spec.wcache->Get(Param(weights, p + "conv_norm_out.bias")),
                              config.norm_eps, dt);
  }
  if (config.timestep_conditioning) {
    const std::vector<float> embed =
        TimestepEmbedding(scaled_timestep, x.channels * 2, weights, p + "last_time_embedder", dt);
    // ada_values rows are (shift, scale) — two, not the resnet's four.
    ApplyAdaLn(spec.queue, spec.wcache, x, Param(weights, p + "last_scale_shift_table"), embed, 2,
               0, 1);
  }
  Silu(spec.queue, x.data.ptr(), static_cast<int64_t>(x.data.size()), dt);
  x = CausalConv3d(spec.queue, spec.wcache, x,
                   config.out_channels * config.patch_size * config.patch_size, 3, config.causal,
                   config.spatial_padding_mode, Param(weights, p + "conv_out.conv.weight"),
                   Param(weights, p + "conv_out.conv.bias"));

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
                              x.data.dtype());
    // `sample.to(output_dtype)` (conv_video_decoder.py:357) landing in the PUBLIC
    // pixel return. `Ltx2VideoFrames::data` stays `std::vector<float>` -- it is
    // three channels wide, it is what the PPM writer and the tiled blend consume,
    // and on the bf16 arm every value in it is bf16-representable, which is what
    // the render path's `vae_decode_not_bf16` counter gates. Narrowing this
    // container too would move the width question into the public ABI for a
    // buffer whose size is a hundredth of the intermediates this row did narrow.
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
  // A24 wave 4 (#2850): the WIDTH FOLLOWS THE INPUT. `patchify` (ops.py:6-32) is
  // a `rearrange` and computes nothing, so this moves elements at whatever width
  // the volume holds and a bf16 -> f32 -> bf16 round trip through
  // `LoadElem`/`StoreElem` is exact by construction. There is no rounding rule
  // here and no probe can separate one; the row's spec §4.5 says so rather than
  // leaving the absence of a measurement to look like an omission.
  out.data.Like(in.data, static_cast<size_t>(out.channels * out.spatial()));
  const vt::DType dt = in.data.dtype();
  for (int64_t c = 0; c < in.channels; ++c) {
    for (int64_t ri = 0; ri < patch; ++ri) {
      for (int64_t qi = 0; qi < patch; ++qi) {
        const int64_t dst_c = (c * patch + ri) * patch + qi;
        for (int64_t f = 0; f < out.t; ++f) {
          for (int64_t hi = 0; hi < out.h; ++hi) {
            for (int64_t wi = 0; wi < out.w; ++wi) {
              StoreElem(out.data.ptr(), static_cast<size_t>(out.At(dst_c, f, hi, wi)),
                        LoadElem(in.data.ptr(),
                                 static_cast<size_t>(in.At(c, f, hi * patch + qi, wi * patch + ri)),
                                 dt),
                        dt);
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
  // A24 wave 4 (#2850): pure movement at the input's width, exactly as
  // `Patchify` above and for the same reason (sampling.py:43-49, 55-61 are both
  // `rearrange`).
  out.data.Like(in.data, static_cast<size_t>(out.channels * out.spatial()));
  const vt::DType dt = in.data.dtype();
  for (int64_t c = 0; c < in.channels; ++c) {
    for (int64_t p1 = 0; p1 < st; ++p1) {
      for (int64_t p2 = 0; p2 < sh; ++p2) {
        for (int64_t p3 = 0; p3 < sw; ++p3) {
          const int64_t dst_c = ((c * st + p1) * sh + p2) * sw + p3;
          for (int64_t ti = 0; ti < out.t; ++ti) {
            for (int64_t hi = 0; hi < out.h; ++hi) {
              for (int64_t wi = 0; wi < out.w; ++wi) {
                StoreElem(
                    out.data.ptr(), static_cast<size_t>(out.At(dst_c, ti, hi, wi)),
                    LoadElem(in.data.ptr(),
                             static_cast<size_t>(in.At(c, ti * st + p1, hi * sh + p2, wi * sw + p3)),
                             dt),
                    dt);
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

  const vt::DType dt = x.data.dtype();
  Volume grown = x;
  if (st == 2) {
    grown.t = x.t + 1;
    grown.data.Like(x.data, static_cast<size_t>(grown.channels * grown.spatial()));
    for (int64_t c = 0; c < grown.channels; ++c) {
      for (int64_t ti = 0; ti < grown.t; ++ti) {
        const int64_t src_t = ti == 0 ? 0 : ti - 1;
        for (int64_t hi = 0; hi < grown.h; ++hi) {
          for (int64_t wi = 0; wi < grown.w; ++wi) {
            // `torch.cat` moves elements; it does not compute (sampling.py:39-40).
            StoreElem(grown.data.ptr(), static_cast<size_t>(grown.At(c, ti, hi, wi)),
                      LoadElem(x.data.ptr(), static_cast<size_t>(x.At(c, src_t, hi, wi)), dt), dt);
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
  skip.data.Like(folded_in.data, static_cast<size_t>(skip.channels * skip.spatial()));
  const int64_t n = skip.spatial();
  for (int64_t c = 0; c < out_channels; ++c) {
    for (int64_t i = 0; i < n; ++i) {
      // A24 wave 4 (#2850): A WIDENED ACCUMULATE WITH EXACTLY ONE ROUNDING, AND
      // THE ROUNDING IS THE STORE.
      //
      // `torch.mean` on a bf16 tensor does not accumulate in bf16; it widens
      // internally and rounds only the output. MEASURED against upstream's own
      // `.mean(dim=2)` (sampling.py:50-51) at two scales: an f32 accumulator and
      // an f64 one are each 0 of 256, while a sequential bf16 accumulate is 72 to
      // 145 of 256 at group_size 4 and 8. `separating = 1` -- f32 and f64 are
      // indistinguishable here and this port claims no gate on the width.
      //
      // AT group_size == 2 NOTHING SEPARATES, because a two-element mean is exact
      // in any order. A fixture that only reaches that width is a mute switch for
      // this rule, which is why the row's bf16 golden uses a block list whose
      // group_size is 4.
      //
      // THE `StoreElem` IS NOT BOOKKEEPING. Carrying this result unrounded into
      // the add below is 45 to 61 of 256 wrong -- 18 to 24% of the block -- while
      // the add's OWN width separates nothing at any scale down to 2^-14. The
      // rounding point is the store, not the operator, and that is per-SITE.
      float acc = 0.0f;
      for (int64_t g = 0; g < group_size; ++g) {
        acc += LoadElem(folded_in.data.ptr(), static_cast<size_t>((c * group_size + g) * n + i),
                        dt);
      }
      StoreElem(skip.data.ptr(), static_cast<size_t>(c * n + i),
                acc / static_cast<float>(group_size), dt);
    }
  }

  // --- the conv branch, at stride 1, on the SAME duplicated input ---
  const Volume convolved =
      CausalConv3d(spec.queue, spec.wcache, grown, conv_out_channels, 3, spec.causal,
                   spec.spatial_padding_mode,
                   Param(weights, prefix + ".conv.conv.weight"),
                   Param(weights, prefix + ".conv.conv.bias"));
  Volume out = SpaceToDepthFold(convolved, st, sh, sw);
  VT_CHECK(out.data.size() == skip.data.size(),
           "ltx2 video encoder: SpaceToDepthDownsample skip and conv shapes must match");
  // `x = x + x_in` (sampling.py:63). Both operands are already on the arm's grid,
  // so the add has ONE rounding wherever it is evaluated -- measured, and it
  // separates NOTHING at 2^0, 2^-7 or 2^-14 against either an f32 or an f64
  // evaluation. Reported as a negative rather than as a confirmation.
  for (size_t i = 0; i < out.data.size(); ++i) {
    StoreElem(out.data.ptr(), i,
              LoadElem(out.data.ptr(), i, dt) + LoadElem(skip.data.ptr(), i, dt), dt);
  }
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
  // A24 wave 4 (#2850) TURNED THIS REFUSAL INTO AN ARM. Wave 3 wrote it knowing
  // this row would replace it: upstream resolves ONE pipeline dtype
  // (`distilled.py:109`) and hands it to `ImageConditioner` at `:120-125`, which
  // builds this encoder with it (`utils/blocks.py:985-986`). The f32 arm stays,
  // because it is the parity reference every committed golden is measured
  // against; the FP8 and NVFP4 arms are A22 and this entry still refuses a third
  // width by name so one cannot arrive by silence.
  //
  // THE CHECK BELOW CARRIES A TOKEN NO OTHER SITE EMITS, and that is the whole
  // reason it exists beside `RequireVaeDType`. `VaeStore::Alloc` calls
  // `RequireVaeDType` too, 60-odd lines downstream, so a subcase asserting the
  // shared decode message could not tell "the encoder refuses at its own entry"
  // from "the staging allocation refused later". Deleting this line has to go
  // RED, and asserting a message two sites emit cannot make it.
  VT_CHECK(weights.dtype == vt::DType::kF32 || weights.dtype == vt::DType::kBF16,
           std::string("ltx2 video encoder: the encoder was handed ") + vt::Name(weights.dtype) +
               "; it serves f32 (the parity arm every committed golden is measured against) and "
               "bf16 (upstream's own model dtype, distilled.py:109). The FP8 and NVFP4 arms are "
               "A22 and are not implemented");
  RequireVaeDType(weights.dtype);
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
  // THE ONE NARROWING, AND IT SITS WHERE UPSTREAM PUTS IT.
  //
  // The encoder's construction is the OPPOSITE of the decoder's and the
  // difference is load-bearing. `ConvVideoDecoder.forward` casts its latent to
  // the weights' dtype on entry (conv_video_decoder.py:283-284). `VideoEncoder.
  // forward` casts NOTHING (video_vae.py:264-336): the pixels are already bf16
  // when they arrive, because `load_image_and_preprocess(..., dtype=dtype, ...)`
  // builds them at the pipeline dtype and hands them straight to
  // `video_encoder(image)` (utils/helpers.py:285-294). So the rounding happens at
  // the boundary, once, before any arithmetic -- which is what `Upload` is.
  //
  // The crop is a GATHER, so the frames are staged compactly first rather than
  // written through `Host()`, which is the f32-only accessor and throws on a
  // bf16 store. Staging is a copy this function already made, at a different
  // place; it is not a second arithmetic path.
  const size_t elems_x = static_cast<size_t>(x.channels * x.spatial());
  std::vector<float> staged(elems_x);
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t f = 0; f < kept; ++f) {
      const size_t src = static_cast<size_t>((c * frame_count + f) * height * width);
      std::copy(frames.begin() + static_cast<ptrdiff_t>(src),
                frames.begin() + static_cast<ptrdiff_t>(src + static_cast<size_t>(height * width)),
                staged.begin() + static_cast<ptrdiff_t>(x.At(c, f, 0, 0)));
    }
  }
  x.data.Alloc(nullptr, elems_x, weights.dtype);
  x.data.Upload(staged.data());

  // --- patchify -> conv_in (video_vae.py:291-292) ---
  x = Patchify(x, config.patch_size);
  x = CausalConv3d(spec.queue, spec.wcache, x, config.out_channels, 3, spec.causal, spec.spatial_padding_mode,
                   Param(weights, p + "conv_in.conv.weight"),
                   Param(weights, p + "conv_in.conv.bias"));

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
                       Param(weights, bp + ".conv.weight"), Param(weights, bp + ".conv.bias"),
                       st, ss, ss);
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
    PixelNorm(spec.queue, x.data.ptr(), x.channels, x.spatial(), config.pixel_norm_eps,
              x.data.dtype());
  } else {
    vt::Queue tail_cpu = VaeCpuQueue();
    vt::Queue& tq = spec.queue != nullptr ? *spec.queue : tail_cpu;
    VaeKernels(tq).group_norm(tq, x.data.ptr(), x.channels, x.spatial(), config.norm_num_groups,
                              spec.wcache->Get(Param(weights, p + "conv_norm_out.weight")),
                              spec.wcache->Get(Param(weights, p + "conv_norm_out.bias")),
                              config.norm_eps, x.data.dtype());
  }
  Silu(spec.queue, x.data.ptr(), static_cast<int64_t>(x.data.size()), x.data.dtype());
  int64_t conv_out_channels = config.out_channels;
  if (config.latent_log_var == Ltx2LogVarianceType::kPerChannel) {
    conv_out_channels *= 2;
  } else if (config.latent_log_var == Ltx2LogVarianceType::kUniform ||
             config.latent_log_var == Ltx2LogVarianceType::kConstant) {
    conv_out_channels += 1;
  }
  x = CausalConv3d(spec.queue, spec.wcache, x, conv_out_channels, 3, spec.causal, spec.spatial_padding_mode,
                   Param(weights, p + "conv_out.conv.weight"),
                   Param(weights, p + "conv_out.conv.bias"));

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

  // READ THROUGH `Param`, WHICH IS ARM-AWARE, and that is what implements the
  // `.to(x)` upstream applies to both registered buffers (ops.py:81-84). On the
  // bf16 arm the bag already holds the narrowed words, exactly as
  // `module.to(torch.bfloat16)` narrows a registered buffer in place, so reading
  // one back IS the narrowing. `weights.Get` would have thrown here on a bf16
  // bag -- which is the accidental refusal wave 3 recorded, in a message naming
  // neither this encoder nor the dtype.
  const VaeParam std_of_means = Param(weights, p + "per_channel_statistics.std-of-means");
  const VaeParam mean_of_means = Param(weights, p + "per_channel_statistics.mean-of-means");
  VT_CHECK(static_cast<int64_t>(std_of_means.count) == latent_channels &&
               static_cast<int64_t>(mean_of_means.count) == latent_channels,
           "ltx2 video encoder: per-channel statistics must have one value per latent channel");

  Ltx2LatentVolume out;
  out.batch = 1;
  out.channels = latent_channels;
  out.frames = x.t;
  out.height = x.h;
  out.width = x.w;
  out.data.resize(static_cast<size_t>(out.elems()));
  const int64_t elems = x.spatial();
  const vt::DType dt = x.data.dtype();
  // A24 wave 4 (#2850): THE STATISTICS NARROW BEFORE THE ARITHMETIC, AND THAT IS
  // 39-47% OF THIS LOOP'S OUTPUT.
  //
  // `PerChannelStatistics.normalize` applies `.to(x)` to BOTH registered buffers
  // (ops.py:81-84), so at bf16 the two statistics round first and the subtract
  // and the divide then round in turn. Wave 3 measured `un_normalize`, which is
  // a multiply and an add; this is a subtract and a divide, and it was RE-RUN
  // rather than inherited -- five components in A24 have now produced five
  // different rules and none has yet transferred.
  //
  // MEASURED on [1, C, 2, 3, 3] at two scales, against upstream's own
  // `normalize` with the f32 buffers captured BEFORE `.to(bfloat16)`: keeping the
  // statistics f32 -- which is what this loop did until this row -- is 129 to 136
  // of 288 at C=16 and 889 to 931 of 2304 at C=128. A single fused f32 expression
  // is a third answer and is also wrong. `separating = 2`.
  //
  // NO TOKEN GATE CAN SEE THIS. `Ltx2LatentVolume::data` is a
  // `std::vector<float>` on either arm, so the widening below is the one upstream
  // does when the bf16 latent leaves the module -- not a second arithmetic path,
  // because every value is already on the arm's grid when it gets here.
  const auto Round = [dt](float v) {
    return dt == vt::DType::kBF16 ? vt::BF16ToF32(vt::F32ToBF16(v)) : v;
  };
  for (int64_t c = 0; c < latent_channels; ++c) {
    const float mean = LoadElem(mean_of_means.data, static_cast<size_t>(c), mean_of_means.dtype);
    const float denom = LoadElem(std_of_means.data, static_cast<size_t>(c), std_of_means.dtype);
    for (int64_t i = 0; i < elems; ++i) {
      const float v = LoadElem(x.data.ptr(), static_cast<size_t>(c * elems + i), dt);
      out.data[static_cast<size_t>(c * elems + i)] = Round(Round(v - mean) / denom);
    }
  }
  return out;
}

}  // namespace vllm

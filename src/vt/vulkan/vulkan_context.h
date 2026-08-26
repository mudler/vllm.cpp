// Vulkan backend — shared instance/device/queue/pipeline context
// (BACKEND-VULKAN, W0 skeleton). vllm.cpp original (vt runtime, inventory
// deviation §9.1): vLLM has no Vulkan platform anywhere, so the DESIGN is ported
// from llama.cpp's Vulkan backend (`ggml/src/ggml-vulkan/ggml-vulkan.cpp` @ pin
// 237ad9b96). Specifically:
//
//   * one process-wide VkInstance + VkPhysicalDevice + VkDevice + compute
//     VkQueue, created lazily and kept for the process — llama.cpp
//     `ggml_vk_instance_init` / `ggml_vk_device_init` and its `vk_instance`
//     singleton;
//   * host-visible storage buffers whose memory type is chosen by walking
//     VkPhysicalDeviceMemoryProperties for the required property flags with an
//     ordered fallback list — llama.cpp `ggml_vk_find_memory_properties`
//     (ggml-vulkan.cpp:2957) and `ggml_vk_create_buffer` (:2971-3100);
//   * a NAME -> compute-pipeline cache so each kernel is specialized once —
//     llama.cpp `ggml_vk_create_pipeline_func` (:2460-2560) and its per-device
//     pipeline map;
//   * push constants for the small per-dispatch parameter block — llama.cpp
//     `ggml_vk_dispatch_pipeline` (:7507-7530).
//
// This header is deliberately PLAIN C++ — it does NOT include vulkan_core.h — so
// the op TU, the platform TU and the tests can include it without pulling the
// Vulkan API into their translation units. Handles cross the boundary as void*,
// exactly as the Metal skeleton does for its ObjC types
// (src/vt/metal/metal_context.h). vulkan_context.cpp static_asserts that the
// real handle types fit.
#ifndef VT_VULKAN_VULKAN_CONTEXT_H_
#define VT_VULKAN_VULKAN_CONTEXT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vt::vulkan {

// DEVICE-MEMORY ACCOUNTING (BACKEND-VULKAN-LOADMEM). Maintained unconditionally
// -- one relaxed atomic add per vkAllocateMemory -- because on a unified-memory
// device the Vulkan heap IS system RAM, so "how many bytes does this backend
// hold" is a question a test and a diagnostic both need to be able to ask at any
// instant, not only under an env flag. `VT_VULKAN_ALLOC_STATS=1` additionally
// prints a line on every 1 GiB high-water crossing and a summary at exit.
//
// `requested` is what the caller asked for (after the 4-byte rounding
// AllocBuffer applies for the 32-bit storage view); `allocated` is what the
// driver committed, `VkMemoryRequirements::size`. They are reported separately
// so a driver-side over-allocation is distinguishable from a caller that simply
// allocates too much -- the two have completely different fixes.
struct DeviceAllocStats {
  uint64_t live_count = 0;
  uint64_t total_count = 0;
  uint64_t live_requested = 0;
  uint64_t live_allocated = 0;
  uint64_t total_requested = 0;
  uint64_t total_allocated = 0;
  uint64_t peak_allocated = 0;
  uint64_t peak_count = 0;
};

DeviceAllocStats DeviceAllocStatsSnapshot();

// Process-wide Vulkan context. Created on first use, never destroyed (the
// process outlives it; matching llama.cpp's `vk_instance` singleton lifetime and
// the Metal skeleton's MetalContext).
class VulkanContext {
 public:
  // Returns the singleton, creating instance/device/queue and the descriptor and
  // command pools on first call. Throws (VT_CHECK) if anything fails — by the
  // time this is called, Available() has already said a usable device exists, so
  // a failure here is a broken driver, not an absent one.
  static VulkanContext& Get();

  // True iff a Vulkan loader is present AND it enumerates a physical device that
  // satisfies this backend's requirements (Vulkan >= 1.1, a compute queue
  // family, VK_KHR_16bit_storage's storageBuffer16BitAccess, and a
  // HOST_VISIBLE|HOST_COHERENT memory type usable for storage buffers). Safe on
  // a machine with no GPU and no loader: it does NOT throw. This is the
  // predicate both registrars use, so a Vulkan-enabled BUILD on a machine
  // without Vulkan simply does not register kVULKAN rather than aborting during
  // static initialization.
  static bool Available();

  // Dispatch one compute kernel, SYNCHRONOUSLY (record, submit, wait). `name` is
  // a key in the committed SPIR-V table (src/vt/vulkan/vulkan_spirv.h).
  // `buffers` are the VkBuffer handles for descriptor bindings 0..n-1, in order;
  // this backend binds every buffer WHOLE (offset 0, VK_WHOLE_SIZE) and carries
  // the element offset in the push constants, so any interior tensor pointer
  // works regardless of minStorageBufferOffsetAlignment (see
  // src/vt/vulkan/shaders/vt_common.glsl § STORAGE MODEL).
  // Serialized by an internal mutex: the command buffer and each pipeline's
  // descriptor set are single instances that are re-recorded per dispatch.
  //
  // `spec_values` are SPECIALIZATION CONSTANT values, supplied in ASCENDING
  // constantID order and matching the module's declared `spec_ids` (recorded in
  // the committed SPIR-V table) exactly. They are part of the pipeline cache KEY:
  // each distinct combination becomes its own VkPipeline, specialized once and
  // reused, with the driver folding the constant and eliminating the branches it
  // kills. This is the variant mechanism that replaces llama.cpp's
  // one-module-per-#define explosion (its vulkan-shaders-gen has 242
  // `string_to_spv` call sites at pin 237ad9b96, most inside dtype/quant/coopmat
  // loops) — here one module covers the axis and the count of committed artifacts
  // tracks shader FILES instead of their cross product.
  void Dispatch(const std::string& name, const void* const* buffers, uint32_t buffer_count,
                const void* push_constants, uint32_t push_size, uint32_t group_count_x,
                const uint32_t* spec_values = nullptr, uint32_t spec_count = 0);

  // Was a pipeline for this SPIR-V module ever created? The cache key is the
  // module name plus its specialization values, so this asks "did any variant of
  // `name` get built", which is the only honest way for a test to prove that a
  // TACTIC actually ran rather than merely that the results were right. A gate
  // that checks numbers alone passes identically when the fallback served the
  // call -- the same trap the op-provider decline counters exist for.
  bool PipelineExistsFor(const std::string& name) const;

  // Every cached pipeline key, as "<module>" or "<module>|<v0>,<v1>,...".
  //
  // PipelineExistsFor above answers "did any variant of this module run", which
  // stops being enough the moment a module carries a PERFORMANCE axis rather than
  // only correctness axes: at that point every arm of the A/B is the same module,
  // the numbers are bit-identical by construction, and a test asserting only the
  // module name passes just as happily when the optimization silently stopped
  // being selected. The VALUES are the mechanism, so a gate has to be able to see
  // them (VK-G, the vt_matmul K-unroll).
  std::vector<std::string> PipelineKeys() const;

  // The FULL cache keys built for one module -- "<name>|<spec values, ascending
  // constantID>". PipelineExistsFor answers "did any variant run"; this answers
  // "WHICH variant ran", which is what a test has to assert when the variants
  // differ only in speed. vt_matmul_vec's rows-per-workgroup axis is exactly that
  // case: every value of it computes the same numbers, so a numeric tolerance
  // test cannot notice when the optimisation silently stops being selected, and
  // only the spec values in the key can.
  std::vector<std::string> PipelineKeysFor(const std::string& name) const;

  // DISPATCH ACCOUNTING (VK-E deep dive). Total submits, and a per-shader
  // histogram. This exists because a wall-clock number could not distinguish two
  // very different stories: a reasonable dispatch count each paying a large
  // fence-wait, versus dispatching far more times than the model should need.
  // Context-switch counts could not separate them either -- the driver's fence
  // wait is a poll() loop that may wake more than once per fence -- so the count
  // has to come from OUR side of the boundary.
  //
  // Always-on and lock-free-ish (guarded by the same mutex the dispatch already
  // takes), because the cost is one increment against a submit that already costs
  // milliseconds. `VT_VULKAN_DISPATCH_STATS=1` prints the histogram at exit.
  uint64_t dispatch_count() const;
  // name -> count, sorted by count descending. For the diagnostic dump.
  std::vector<std::pair<std::string, uint64_t>> DispatchHistogram() const;

  // name -> total fence-wait milliseconds, sorted descending. COUNTS NAME THE
  // SHAPE OF A RUN; ONLY TIME NAMES THE LEVER. A measured 0.046 ms per-dispatch
  // floor against a 0.357 ms observed average proved that 87% of this backend's
  // dispatch cost is real kernel execution, not submission overhead -- so the
  // question stopped being "how many dispatches" and became "which kernel", and
  // a histogram of counts cannot answer that. The two differ wildly: the most
  // FREQUENT shader is routinely not the most EXPENSIVE one.
  //
  // Submission is synchronous, so the fence wait brackets that dispatch and
  // nothing else, and these sum to the run's GPU time rather than overlapping.
  std::vector<std::pair<std::string, double>> DispatchTimeMs() const;

  // COMMAND-BUFFER BATCHING (VK-A2). Records dispatches into ONE command buffer
  // and submits once, instead of submit+fence-wait per op.
  //
  // WHY: a measured per-dispatch floor of 0.046 ms times 2,952 dispatches is
  // 135.8 ms of a 275.8 ms decode run -- 49% of GPU time, up from 13% before the
  // argmax and GEMV kernels landed. Three shaders now cost AT OR BELOW that
  // floor, meaning they do no meaningful work relative to being launched.
  //
  // FLUSH is mandatory before ANY host read of device memory. This backend's
  // Copy/Memset are plain memcpy over the persistently mapped, host-coherent
  // allocation, so a pending batch means the host reads STALE bytes -- silently,
  // with no error. Backend::Copy, Memset and Synchronize all flush.
  // `why` attributes the flush to its TRIGGER. Measured: a 27B decode does 212
  // flushes per TOKEN, most carrying only 1-2 dispatches, so batching is being
  // defeated by something other than the ring. Which trigger fires decides the
  // fix, and guessing has been wrong five times this session.
  void FlushBatch(const char* why = "explicit");
  // Drains only if `buffer` (a packed VkBuffer, or nullptr for host memory) was
  // bound by a dispatch in the currently open batch. See Backend::Copy for why
  // that is the exact condition.
  void FlushIfBatchTouches(void* buffer, const char* why);
  // Whether dispatch batching is active. Exposed so a test never has to restate
  // the default: the VK-A2 gate originally re-derived it from the environment
  // variable and silently asserted the wrong branch the moment the default
  // flipped from off to on. A predicate duplicated between an implementation and
  // its gate is a predicate that will disagree with itself.
  bool batching_enabled() const;
  // Dispatches currently recorded and not yet submitted. For the gate: batching
  // is invisible in results by construction (same kernels, same order), so a test
  // has to assert the MECHANISM rather than the numbers.
  uint32_t pending_batch() const;

  // --- PIPELINED SUBMISSION (BACKEND-VULKAN-HOSTDISPATCH). Same argument as
  // pending_batch, one level up: a pipelined flush and a blocking flush run the
  // same kernels in the same order and produce identical numbers, so if the
  // pipelining silently stopped applying every value check would still pass and
  // only the wall clock would move. These are what a gate can assert on.
  //
  // Batches submitted but not yet known-complete. > 0 after a flush is the
  // mechanism: the host returned from the flush without waiting.
  uint32_t in_flight_batches() const;
  // The configured depth (VT_VULKAN_INFLIGHT, clamped to kMaxInFlight). 1 is the
  // pre-row submit-and-wait behaviour. Asked rather than re-derived from the
  // environment, for the reason batching_enabled() exists.
  uint32_t in_flight_limit() const;
  // Descriptor sets one slot may consume before it must flush. kMaxInFlight
  // slices of the ring, never overlapping, is the correctness argument for
  // letting a submitted batch keep executing while the next one is recorded.
  uint32_t ring_slice() const;
  // Sets allocated per pipeline, i.e. the width the slices partition.
  uint32_t ring_depth() const;
  // vkQueueSubmit calls, and the subset of slot retirements that actually had to
  // wait on an unsignalled fence. A pipelined run submits more often than it
  // waits; a run that degraded back to submit-and-wait has them equal.
  uint64_t submit_count() const;
  uint64_t fence_wait_count() const;
  // First descriptor-set index the OPEN batch may use. The slices are what make
  // pipelining sound, and a timing-dependent value check cannot see them: on a
  // software rasterizer a submitted batch is effectively finished by the time the
  // next one records, so collapsing every slot onto slice 0 computes the right
  // numbers there and corrupts them on real hardware. This is the STRUCTURAL
  // assertion that closes that gap -- MEASURED: a scratch mutation forcing this
  // to 0 for every slot passed the whole file on llvmpipe until this existed.
  uint32_t ring_base() const;
  // Pipeline barriers recorded. One per dispatch, INCLUDING the first in each
  // command buffer -- that first one is what carries a dependency across a
  // command-buffer boundary now that the previous batch may still be running, and
  // dropping it is likewise invisible in llvmpipe's numbers.
  uint64_t barrier_count() const;

  // --- SMART BARRIERS (BACKEND-VULKAN-BARRIERS). Dispatches whose buffers do not
  // collide with anything recorded since the last barrier get NO barrier.
  //
  // Barriers NOT recorded because the incoming dispatch was proven independent of
  // everything since the previous barrier. barrier_count() + barrier_skip_count()
  // is the dispatch count in a batched run, which is what the always-barrier arm
  // would have recorded. This is the ONLY direct evidence that the analysis is
  // doing anything: the two arms compute identical numbers by construction, so a
  // value check cannot tell them apart, and neither can the wall clock when the
  // skip rate is small.
  uint64_t barrier_skip_count() const;
  // Whether the hazard analysis is active (VT_VULKAN_SMART_BARRIERS, or the
  // override below). Asked rather than re-derived from the environment, for the
  // reason batching_enabled() exists.
  bool smart_barriers() const;
  // A/B LEVER over that decision, inside ONE binary. -1 forces the unconditional
  // barrier before every dispatch (main's behaviour, byte for byte), +1 forces the
  // hazard analysis, 0 lets VT_VULKAN_SMART_BARRIERS decide.
  //
  // A cross-BUILD comparison of two barrier policies is exactly the shape that
  // produced a false 1.2x reading earlier in this campaign, and -- far worse here
  // -- a barrier policy that drops a REAL dependency computes wrong numbers, so
  // the correctness gate has to be able to run both arms against the same inputs
  // in one process and compare them directly.
  //
  // Drains first: the analysis state describes work already recorded, so changing
  // the policy underneath an open or in-flight batch would reason about the wrong
  // history.
  void set_smart_barriers_override(int v);
  int smart_barriers_override() const { return smart_barriers_override_; }

  // GPU-TIMELINE SPAN, milliseconds: summed over command buffers, the interval
  // from the FIRST dispatch's top-of-pipe timestamp to the LAST one's
  // bottom-of-pipe. Only collected when VT_VULKAN_DISPATCH_STATS is set, like the
  // per-dispatch timestamps it is derived from.
  //
  // WHY IT IS SEPARATE FROM DispatchTimeMs. Those sum each dispatch's OWN
  // interval; this measures the wall the GPU spent on the whole command buffer.
  // The difference between them is the time the GPU spent BETWEEN dispatches --
  // barrier drains and launch setup -- which is the quantity this row exists to
  // move and which no per-shader number can show. It is also the honest metric
  // once barriers are skipped: without a barrier between them two dispatches may
  // OVERLAP, so their individual intervals double-count and only the span stays
  // meaningful.
  double gpu_span_ms() const;
  // Command buffers the span above was accumulated over.
  uint64_t gpu_span_batches() const;

  // Number of distinct pipelines currently cached. Exposed for the unit gate: it
  // is how a test proves a new specialization produced a NEW pipeline rather than
  // silently reusing an existing one — which would look identical in the results.
  size_t PipelineCacheSize() const;

  // Allocate one host-visible, host-coherent storage buffer of `bytes` and keep
  // it PERSISTENTLY MAPPED. Returns the mapped host pointer — which is what
  // vt::Tensor::data carries — and hands back the VkBuffer / VkDeviceMemory
  // handles for the allocation registry. Persistent mapping is what makes
  // Backend::Copy/Memset plain memcpy/memset and keeps them BIT-EXACT.
  void* AllocBuffer(size_t bytes, void** out_buffer, void** out_memory);
  void FreeBuffer(void* buffer, void* memory);

  // A small device-visible scratch buffer for per-dispatch data that is too big
  // for push constants (the fused-chain recipe step list). Returns the VkBuffer
  // handle; `Data()` is its persistently mapped host pointer. Reused across
  // dispatches, which is safe because dispatch is synchronous.
  void* ScratchBuffer() const { return scratch_buffer_; }
  void* ScratchData() const { return scratch_mapped_; }
  static constexpr size_t kScratchBytes = 1024;

  // --- The VULKAN API VERSION. {1, 4} on GB10 (API 1.4.312). It is reached
  // through vt::Backend::DeviceCapabilityMajor/Minor
  // (src/vt/vulkan/vulkan_backend.cpp:144-145), which forwards to the
  // api_major()/api_minor() accessors below; those are public and the suite
  // reads them directly (tests/vt/test_vulkan_backend.cpp).
  //
  // It is NOT mirrored onto vllm::platforms::Platform, and #1823 is why. That
  // seam's get_device_capability() is an NVIDIA SM version by contract
  // (interface.py:420-431, "Stateless version of torch.cuda.get_device_capability"),
  // so `has_device_capability(1, 1)` there does NOT read as "Vulkan >= 1.1" — on
  // kVULKAN it is false for every argument, because the platform correctly
  // reports ABSENT. This header used to claim the opposite, and the claim cost
  // FLASH_ATTN on this backend: an SM-8.0 bar was applied to `major == 1`.
  // The shared VkQueue, as the opaque handle vt::Queue carries.
  void* queue_handle() const { return queue_; }

  int api_major() const { return api_major_; }
  int api_minor() const { return api_minor_; }
  bool unified_memory() const { return unified_memory_; }
  const std::string& device_name() const { return device_name_; }
  uint32_t max_workgroup_count_x() const { return max_workgroup_count_x_; }
  // The two float-controls properties that decide whether our f32 arithmetic is
  // IEEE as written. Probed, recorded, and asserted by the unit gate; see
  // vulkan_context.cpp § RELAXED PRECISION.
  bool denorm_preserve_f32() const { return denorm_preserve_f32_; }
  bool signed_zero_inf_nan_preserve_f32() const { return sz_inf_nan_preserve_f32_; }

  // --- COOPERATIVE MATRIX (VK-C). True iff the device exposes
  // VK_KHR_cooperative_matrix AND reports a bf16 x bf16 -> f32 configuration at
  // 16x16x16 with SUBGROUP scope AND has the subgroup size the committed SPIR-V
  // assumes -- all four, because any one of them missing makes the coopmat
  // pipeline unusable and the scalar tactic is always correct.
  //
  // MEASURED 2026-08-07: GB10 reports 11 configurations, all SUBGROUP scope,
  // among them 16x16x16 bf16/bf16/f32/f32; llvmpipe exposes the extension NOT AT
  // ALL. So this predicate is genuinely false on the only device CI can reach,
  // which is why the scalar fallback is the tested-everywhere path and the
  // coopmat path is dgx-gated.
  bool coopmat_bf16_f32() const { return coopmat_bf16_f32_; }
  uint32_t subgroup_size() const { return subgroup_size_; }

  // --- WIDE REDUCTION SHADERS (VK-RMSNORM). Every shader in this backend is
  // compiled for the Vulkan-GUARANTEED 128 invocations, which is the right floor
  // for a flat kernel but leaves a per-ROW reducing kernel on four warps of one
  // SM at batch 1. `vt_rms_norm_wide` is a second module at 1024 invocations with
  // a subgroup reduction; this predicate is what decides whether it may be used.
  //
  // All four conditions, because any one missing makes the module unusable:
  // enough invocations per workgroup, enough of them on the X axis, and the
  // subgroup BASIC (gl_NumSubgroups / gl_SubgroupID / subgroupElect) and
  // ARITHMETIC (subgroupAdd) feature bits present IN THE COMPUTE STAGE. A module
  // whose capabilities the device lacks is UNDEFINED BEHAVIOUR at pipeline
  // creation rather than a VkResult, so this is probed rather than hoped for.
  bool wide_reduce() const { return wide_reduce_; }

  // A/B LEVER over that decision, and the ONLY way the fallback is reachable on a
  // device that has the capability. -1 forces the portable 128-wide module, +1
  // forces the wide one, 0 (the default, and what VT_VULKAN_RMSNORM=base|wide
  // overrides) lets wide_reduce() decide.
  //
  // Two jobs, both load-bearing. (1) A cross-BUILD comparison of two shader
  // widths is exactly the shape that produced a false 1.2x reading for the
  // subgroup tactic earlier in this campaign; switching arms inside ONE binary is
  // the fix. (2) `test_vulkan_backend` uses it to prove the FALLBACK path still
  // computes the same numbers on hardware that would otherwise never take it --
  // without a setter that path is dead code on every box we own.
  //
  // Set it before dispatching. It is read on the (mutex-serialized) dispatch
  // path and is not meant to change while work is in flight.
  int rms_norm_override() const { return rms_norm_override_; }
  void set_rms_norm_override(int v) { rms_norm_override_ = v; }
  uint32_t max_workgroup_invocations() const { return max_workgroup_invocations_; }
  uint32_t max_workgroup_size_x() const { return max_workgroup_size_x_; }
  bool subgroup_arithmetic_compute() const { return subgroup_arithmetic_compute_; }

  // Invocations the wide reducing modules are compiled for. Mirrors VT_TG in
  // src/vt/vulkan/shaders/vt_rms_norm_wide.comp; the host never launches with it
  // (a reducing kernel dispatches one workgroup per row) but the probe above
  // compares against it, so the two must not drift.
  static constexpr uint32_t kWideWorkgroupSize = 1024;

  // Upper bound on batches in flight at once (BACKEND-VULKAN-HOSTDISPATCH). The
  // per-slot resources below are sized by it, and VT_VULKAN_INFLIGHT is clamped
  // to it. Public because the A/B lever is read at namespace scope in the
  // implementation, and because the unit gate asserts against it.
  static constexpr uint32_t kMaxInFlight = 4;

 private:
  VulkanContext();
  struct Pipeline;
  Pipeline& GetPipeline(const std::string& name, uint32_t buffer_count, uint32_t push_size,
                        const uint32_t* spec_values, uint32_t spec_count);

  void* instance_ = nullptr;         // VkInstance
  void* physical_device_ = nullptr;  // VkPhysicalDevice
  void* device_ = nullptr;           // VkDevice
  void* queue_ = nullptr;            // VkQueue
  void* command_pool_ = nullptr;     // VkCommandPool
  void* command_buffer_ = nullptr;   // VkCommandBuffer
  void* descriptor_pool_ = nullptr;  // VkDescriptorPool
  void* fence_ = nullptr;            // VkFence
  void* scratch_buffer_ = nullptr;   // VkBuffer
  void* scratch_memory_ = nullptr;   // VkDeviceMemory
  void* scratch_mapped_ = nullptr;   // host pointer
  // Submits the open batch. Does NOT wait unless in-flight slots are exhausted.
  void FlushBatchLocked(const char* why = "explicit");  // caller holds mutex_
  // Submits the open batch AND waits for every submitted batch to complete, so
  // that device memory is safe for the host to read. This is what the three host
  // read paths (Copy, Memset, Synchronize/FlushPending) need; FlushBatchLocked
  // alone is NOT sufficient once submission is pipelined.
  void DrainLocked(const char* why);                    // caller holds mutex_
  // Waits for slot `s` if it is in flight, reads back its timestamps, and resets
  // its command pool. After this returns, slot `s`'s command buffer, its slice of
  // every pipeline's descriptor ring, and its query range are all free to reuse.
  void RetireSlotLocked(uint32_t s);                    // caller holds mutex_
  // GPU TIMESTAMP PROFILING. Batching submits many dispatches under ONE fence,
  // so the per-dispatch fence wait that used to attribute time to a shader no
  // longer exists. Timestamps written into the command buffer are the only way to
  // recover per-kernel GPU time once submissions are batched -- and they measure
  // the GPU directly rather than a host-side wait, so they are strictly better
  // evidence than what they replace.
  //
  // Allocated and written ONLY when VT_VULKAN_DISPATCH_STATS is set, so a
  // production dispatch pays nothing.
  void* query_pool_ = nullptr;       // VkQueryPool
  double timestamp_period_ns_ = 0.0; // 0 => device cannot timestamp; profiling off
  bool batch_open_ = false;          // a command buffer is recording
  uint32_t batch_count_ = 0;         // dispatches recorded into it
  void* batch_buffers_ = nullptr;    // BufferSet*, buffers any UNRETIRED batch bound

  // --- SMART BARRIERS (BACKEND-VULKAN-BARRIERS). The two halves of the access
  // history the hazard test consults, both BufferSet*.
  //
  // THE INVARIANT, which is the whole correctness argument. At every point,
  // hazard_written_ contains every buffer WRITTEN, and hazard_read_ every buffer
  // READ, by any command recorded AFTER the most recent vkCmdPipelineBarrier this
  // context emitted -- across command-buffer and submission boundaries, because
  // they are cleared ONLY when a barrier is recorded and by nothing else.
  //
  // A dispatch is independent of that history iff none of its reads is in
  // hazard_written_ (read-after-write), none of its writes is in hazard_written_
  // (write-after-write) and none of its writes is in hazard_read_
  // (write-after-read). If any of the three holds, a barrier is recorded, which
  // by Vulkan's submission-order scope orders the dispatch after EVERY command
  // submitted earlier on this queue -- so the sets may then be emptied and the
  // invariant restarts. That scope is also what makes the analysis sound across
  // the command-buffer boundary a pipelined submission creates.
  //
  // Both are keyed on the whole VkBuffer, never on the tensor's byte range: two
  // tensors that share an allocation are reported as colliding even when their
  // ranges do not, which costs a barrier and never misses one. An operand that is
  // both read and written by the same shader lands in the WRITE set only, which is
  // strictly stronger -- a write collides with everything a read collides with.
  void* hazard_written_ = nullptr;
  void* hazard_read_ = nullptr;
  uint64_t barrier_skipped_ = 0;     // barriers the analysis proved unnecessary
  int smart_barriers_override_ = 0;  // -1 always barrier, +1 analyse, 0 = env
  // "The access history is not trustworthy, barrier unconditionally once." Set at
  // construction and whenever the barrier policy changes, because the
  // always-barrier arm deliberately does not maintain the history and an empty
  // history must never be mistaken for an absence of hazards.
  bool force_barrier_next_ = true;
  uint64_t gpu_span_ns_ = 0;         // summed per-command-buffer GPU spans
  uint64_t gpu_span_batches_ = 0;

  // PIPELINED SUBMISSION (BACKEND-VULKAN-HOSTDISPATCH). Ported from llama.cpp
  // `ggml_backend_vk_graph_compute` (ggml-vulkan.cpp:16192-16195 and
  // :16417-16423 @ pin 237ad9b96), whose own comment states the mechanism:
  // "Submit after enough work has accumulated, to overlap CPU cmdbuffer
  // generation with GPU execution." It submits several times per graph and waits
  // exactly ONCE, in `ggml_vk_wait_for_fence` (:2298).
  //
  // Before this row, every flush was submit-AND-wait, so the GPU was idle for the
  // whole of the host's recording of the next batch. MEASURED on 27B decode: 900
  // dispatches and 4 flushes per token, 3.04 ms/token of host time, every
  // nanosecond of it serialized against a 227 ms GPU step.
  //
  // Each slot owns a command buffer, its own command POOL (a pool can only be
  // reset as a whole, so one shared pool cannot serve an in-flight buffer), its
  // own fence, its own query range, and a DISJOINT SLICE of every pipeline's
  // descriptor ring. That last one is the correctness argument: a descriptor set
  // is read at execution time, so rewriting one while a submitted command buffer
  // still references it is silent corruption. A set can only be rewritten when
  // its slot is re-entered, and a slot is re-entered only through
  // RetireSlotLocked, which waits on that slot's fence first.
  void* slot_cmd_[kMaxInFlight] = {};    // VkCommandBuffer
  void* slot_pool_[kMaxInFlight] = {};   // VkCommandPool
  void* slot_fence_[kMaxInFlight] = {};  // VkFence
  void* slot_names_[kMaxInFlight] = {};  // std::vector<std::string>*
  bool slot_in_flight_[kMaxInFlight] = {};
  uint32_t slot_ = 0;                // slot the open batch records into
  uint64_t submit_count_ = 0;        // vkQueueSubmit calls
  uint64_t fence_wait_count_ = 0;    // retirements that had to wait
  uint64_t barrier_count_ = 0;       // vkCmdPipelineBarrier calls
  uint32_t slot_ring_base_ = 0;      // slot_ * ring slice width, cached
  void* dispatch_ms_ = nullptr;      // std::map<std::string, double>*
  uint64_t dispatch_total_ = 0;
  void* pipelines_ = nullptr;        // std::map<std::string, Pipeline>*
  void* mutex_ = nullptr;            // std::mutex*
  uint32_t queue_family_ = 0;
  uint32_t memory_type_index_ = 0;
  int api_major_ = 0;
  int api_minor_ = 0;
  bool unified_memory_ = false;
  bool denorm_preserve_f32_ = false;
  bool sz_inf_nan_preserve_f32_ = false;
  bool coopmat_bf16_f32_ = false;
  uint32_t subgroup_size_ = 0;
  uint32_t max_workgroup_count_x_ = 0;
  uint32_t max_workgroup_invocations_ = 0;
  uint32_t max_workgroup_size_x_ = 0;
  bool subgroup_arithmetic_compute_ = false;
  bool wide_reduce_ = false;
  int rms_norm_override_ = 0;
  std::string device_name_;

  friend class VulkanAllocator;
};

// Plain-C++ spelling of VulkanContext::Available(), so the engine-side platform
// TU (src/vllm/platforms/vulkan.cpp) can ask "is there a Vulkan device?" without
// depending on static-initialization ORDER — asking "did the backend registrar
// already run?" from another TU's initializer is unspecified-order and would
// intermittently skip platform registration. Same reasoning, same shape, as
// vt::metal::MetalDeviceAvailable().
bool VulkanDeviceAvailable();

// Workgroup size every kernel in this backend is compiled with. Mirrors VT_TG in
// src/vt/vulkan/shaders/vt_common.glsl; the host must agree with the SPIR-V
// because the flat kernels compute their workgroup COUNT from it.
inline constexpr uint32_t kWorkgroupSize = 128;

// Number of workgroups needed to cover `n` elements at kWorkgroupSize threads
// each.
uint32_t FlatGroupCount(int64_t n);

// OUTPUT COLUMNS PER LANE for the scalar matmul tactic (VK-G), 1, 4 or 8. Defined
// in vulkan_ops.cpp; declared here because it is the one axis of that kernel a
// test has to be able to move.
//
// WHY A RUNTIME SETTER AND NOT JUST THE ENVIRONMENT VARIABLE. The claim this
// optimization rests on is that the column-blocked body is BIT-IDENTICAL to the
// flat one -- each accumulator owns one output element and runs the whole K
// reduction sequentially, so nothing is reassociated and the byte-exact tier of
// the general matmul path is untouched. Bit-identity between two SEPARATE
// processes is not something a test can assert; both arms have to run in ONE
// binary against the same inputs so the comparison can be a memcmp. The
// environment variable stays the production lever; this is how the gate proves the
// lever is numerically free.
uint32_t MatmulColumnsPerLane();
void SetMatmulColumnsPerLane(uint32_t ncols);

}  // namespace vt::vulkan

#endif  // VT_VULKAN_VULKAN_CONTEXT_H_

// vllm.cpp original (vt runtime, inventory deviation §9.1) — implementation of
// the acceleration-provider seam declared in include/vt/op_provider.h
// (BACKEND-ACCEL-PROVIDER, .agents/specs/metal-mlx-reuse-study.md §6).
//
// Structure is ported from src/vt/cuda/cuda_arch_tactics.cu:64-170, which proved
// the shape one layer down: fixed-capacity static storage so registration is
// safe from any static-init order, table-fill-only registration that never
// throws, a linear capability scan at selection, and atomic instrumentation.
//
// The ONE thing changed in the port is the property that made the flat op table
// a bug: selection is `(priority DESC, name ASC)`, NOT registration order. The
// tactic registry could afford registration order because exactly one tactic is
// registered per family; the op table cannot, because the whole point is two
// providers coexisting, and registration order across TUs is unspecified.
//
// These three symbols (`RegisterOp`, `GetOp`, `OpRegistered`) were previously
// defined in src/vt/ops.cpp over a flat `void* [OpId][DeviceType]` array. They
// moved HERE rather than growing that file — it is the single hottest shared TU
// in the tree and every op wrapper in it is untouched by this change.
#include "vt/op_provider.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "vt/backend.h"
#include "vt/ops.h"

namespace vt {
namespace {

// Enough for the native kernel plus three accelerator providers on one op. A
// higher count is a design smell (four libraries claiming one op), and a bounded
// array is what keeps registration allocation-free and static-init-order-safe.
constexpr int kMaxProvidersPerOp = 4;
constexpr size_t kOpCount = static_cast<size_t>(OpId::kCount);

struct Slot {
  OpProvider providers[kMaxProvidersPerOp];
  int count = 0;
  // Resolved selection cache. GetOp is on the hot path of every op in the tree,
  // so the steady state must stay one relaxed load, exactly like the array load
  // it replaces. Registration and SetDeviceProviderCaps() clear it.
  std::atomic<void*> selected{nullptr};
  // Memoized NEGATIVE resolution. `OpRegistered` is consulted per call by the
  // fused-recipe fast-realization ladder (src/vt/ops.cpp) for ops a backend does
  // NOT have, so the "nothing supports this" answer must be as cheap as the
  // positive one — otherwise the seam would put a capability scan on the 35B
  // decode path. Cleared by registration and by SetDeviceProviderCaps().
  std::atomic<bool> resolved_none{false};
  // Cached NATIVE-availability for OpRegistered: -1 unknown, 0 none, 1 present.
  // OpRegistered means "is there a NATIVE kernel" (the fused-recipe ladder and
  // the cross-device harness depend on that meaning), so it deliberately IGNORES
  // any reference-tier fallback provider in this slot. Cleared exactly like the
  // selection cache. This is separate from `selected` because `selected` CAN
  // hold a reference-tier fn (a real GetOp answer) while native stays absent.
  std::atomic<int8_t> native_registered{-1};
  // One-time loud reference-tier warning per (op, device) — see Resolve.
  std::atomic<bool> ref_announced{false};
  std::atomic<const char*> last_selected{nullptr};
  std::atomic<unsigned long long> selections{0};
  std::atomic<unsigned long long> declines{0};
  std::atomic<unsigned long long> fallbacks{0};
  std::atomic<bool> announced{false};
};

// OBSERVABILITY: total reference-tier selections across all (op, device). A
// process that ran a model entirely on native kernels leaves this at 0; any
// value > 0 is proof the portable CPU path fired and the run is not a
// performance measurement. Ported in spirit from the ArchTacticStats counters.
std::atomic<unsigned long long>& RefTierHits() {
  static std::atomic<unsigned long long> hits{0};
  return hits;
}

// Zero-initialized static storage: usable from any static-init order, no
// dynamic allocation, no dependence on another TU's constructor. ~66 KiB of BSS.
Slot* Table() {
  static Slot table[kOpCount * kNumDeviceTypes];
  return table;
}

Slot& At(OpId op, DeviceType device) {
  const size_t o = static_cast<size_t>(op);
  const size_t d = static_cast<size_t>(device);
  VT_CHECK(o < kOpCount, "invalid op id");
  VT_CHECK(d < kNumDeviceTypes, "invalid device type");
  return Table()[o * kNumDeviceTypes + d];
}

// --- device capability records ---------------------------------------------
struct CapsTable {
  std::mutex mu;
  ProviderCaps caps[kNumDeviceTypes];
};
CapsTable& Caps() {
  static CapsTable t;
  return t;
}

// --- runtime provider disable list (VT_OP_PROVIDER_DISABLE / DisableOpProvider)
constexpr int kMaxDisabled = 8;
constexpr size_t kMaxNameLen = 63;
struct DisableList {
  std::mutex mu;
  char names[kMaxDisabled][kMaxNameLen + 1] = {};
  int count = 0;
  bool env_parsed = false;
  // Lock-free fast path. `Supported()` runs inside the capability scan, which
  // `OpRegistered` reaches on a cold slot; taking a mutex there would be a hot
  // path regression. The overwhelmingly common state is "nothing disabled", and
  // this atomic answers that without touching `mu`.
  std::atomic<int> live_count{-1};  // -1 == env not parsed yet
};
DisableList& Disabled() {
  static DisableList d;
  return d;
}

void AddDisabledLocked(DisableList& d, const char* begin, const char* end) {
  size_t n = static_cast<size_t>(end - begin);
  if (n == 0 || n > kMaxNameLen || d.count >= kMaxDisabled) return;
  for (int i = 0; i < d.count; ++i) {
    if (std::strncmp(d.names[i], begin, n) == 0 && d.names[i][n] == '\0') return;
  }
  std::memcpy(d.names[d.count], begin, n);
  d.names[d.count][n] = '\0';
  ++d.count;
  d.live_count.store(d.count, std::memory_order_relaxed);
}

void EnsureEnvParsedLocked(DisableList& d) {
  if (d.env_parsed) return;
  d.env_parsed = true;
  const char* e = std::getenv("VT_OP_PROVIDER_DISABLE");
  if (e != nullptr) {
    const char* p = e;
    while (*p != '\0') {
      const char* q = p;
      while (*q != '\0' && *q != ',') ++q;
      AddDisabledLocked(d, p, q);
      p = (*q == '\0') ? q : q + 1;
    }
  }
  d.live_count.store(d.count, std::memory_order_relaxed);
}

std::atomic<bool>& CallStatsFlag() {
  static std::atomic<bool> on{[] {
    const char* e = std::getenv("VT_OP_PROVIDER_STATS");
    return e != nullptr && e[0] == '1';
  }()};
  return on;
}

bool AnnounceEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_OP_PROVIDER_STATS");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// The DETERMINISTIC total order: higher priority first, then name ascending by
// strcmp. Both keys are compile-time constants of the registering TU, so the
// winner does not depend on which static initializer ran first.
bool Better(const OpProvider& a, const OpProvider& b) {
  if (a.priority != b.priority) return a.priority > b.priority;
  return std::strcmp(a.name, b.name) < 0;
}

bool Supported(const OpProvider& p, const ProviderCaps& caps) {
  if (p.fn == nullptr) return false;
  if (OpProviderDisabled(p.name)) return false;
  return p.supports == nullptr || p.supports(caps);
}

// Best supported provider strictly below `floor` in the order, or the best
// overall when `floor` is null. One linear scan; kMaxProvidersPerOp is 4.
const OpProvider* Choose(Slot& slot, const ProviderCaps& caps, const OpProvider* floor) {
  const OpProvider* best = nullptr;
  for (int i = 0; i < slot.count; ++i) {
    const OpProvider& p = slot.providers[i];
    if (!Supported(p, caps)) continue;
    if (floor != nullptr && !Better(*floor, p)) continue;
    if (best == nullptr || Better(p, *best)) best = &p;
  }
  return best;
}

// The portable reference tier (S5). When (op, device) has no provider and the
// device is a UNIFIED-MEMORY accelerator, install the CPU kernel as a
// negative-priority fallback so GetOp returns a working (if slow) kernel instead
// of throwing — our equivalent of vLLM CustomOp.forward_native. Returns true iff
// a fallback was installed. The unified-memory gate lives in
// ReferenceTierEligible; a discrete device NEVER reaches RegisterOpProvider here.
bool MaybeInstallReferenceTier(OpId op, DeviceType device) {
  if (!ReferenceTierEligible(device)) return false;
  // Already installed for this (op, device)? Report no-install so an eager
  // second pass (RegisterReferenceTier) is exactly idempotent — RegisterOpProvider
  // rejects the duplicate name, so counting it would over-report.
  Slot& target = At(op, device);
  for (int i = 0; i < target.count; ++i) {
    if (std::strcmp(target.providers[i].name, kReferenceProviderName) == 0) return false;
  }
  Slot& cpu = At(op, DeviceType::kCPU);
  if (cpu.count == 0) return false;  // no CPU reference either — a genuine throw
  const ProviderCaps cpu_caps = GetDeviceProviderCaps(DeviceType::kCPU);
  const OpProvider* src = Choose(cpu, cpu_caps, nullptr);
  if (src == nullptr) return false;
  OpProvider ref;
  ref.name = kReferenceProviderName;
  ref.priority = kReferenceTierPriority;
  ref.supports = nullptr;  // portable: runs wherever host==device memory
  ref.fn = src->fn;        // the SAME host kernel the CPU device dispatches
  RegisterOpProvider(op, device, ref);
  return true;
}

void Announce(OpId op, DeviceType device, Slot& slot, const OpProvider* chosen,
              const ProviderCaps& caps) {
  if (!AnnounceEnabled()) return;
  if (slot.announced.exchange(true, std::memory_order_relaxed)) return;
  std::fprintf(stderr,
               "[vt op-provider] op=%d device=%d selected=%s priority=%d "
               "registered=%d caps=%d.%d/%s\n",
               static_cast<int>(op), static_cast<int>(device),
               chosen != nullptr ? chosen->name : "<none>",
               chosen != nullptr ? chosen->priority : 0, slot.count, caps.compute_major,
               caps.compute_minor, caps.valid ? "valid" : "unprobed");
}

void* Resolve(OpId op, DeviceType device, Slot& slot) {
  ProviderCaps caps = GetDeviceProviderCaps(device);
  const OpProvider* chosen = Choose(slot, caps, nullptr);
  // MISS: on a unified-memory accelerator, install the CPU reference tier and
  // re-select. The native kernel — if one existed — would already have been
  // chosen above (priority >= 0 beats the tier's negative priority), so this
  // path is reached ONLY when the device genuinely lacks a native kernel.
  if (chosen == nullptr && MaybeInstallReferenceTier(op, device)) {
    chosen = Choose(slot, caps, nullptr);
  }
  Announce(op, device, slot, chosen, caps);
  if (chosen == nullptr) {
    slot.fallbacks.fetch_add(1, std::memory_order_relaxed);
    slot.resolved_none.store(true, std::memory_order_relaxed);
    VT_CHECK(false, std::string("no kernel for op ") +
                        std::to_string(static_cast<int>(op)) + " on device type " +
                        std::to_string(static_cast<int>(device)));
    return nullptr;
  }
  // Reference-tier accounting: count it, and warn LOUDLY exactly once per
  // (op, device) so "this backend ran op X on the portable tier" is never silent.
  if (std::strcmp(chosen->name, kReferenceProviderName) == 0) {
    RefTierHits().fetch_add(1, std::memory_order_relaxed);
    if (!slot.ref_announced.exchange(true, std::memory_order_relaxed)) {
      std::fprintf(stderr,
                   "[vt reference-tier] op=%d device=%d has NO native kernel; "
                   "running the PORTABLE CPU fallback (correct but slow)\n",
                   static_cast<int>(op), static_cast<int>(device));
    }
  }
  slot.last_selected.store(chosen->name, std::memory_order_relaxed);
  slot.selected.store(chosen->fn, std::memory_order_relaxed);
  return chosen->fn;
}

void InvalidateAll() {
  Slot* t = Table();
  for (size_t i = 0; i < kOpCount * kNumDeviceTypes; ++i) {
    t[i].selected.store(nullptr, std::memory_order_relaxed);
    t[i].resolved_none.store(false, std::memory_order_relaxed);
    t[i].native_registered.store(-1, std::memory_order_relaxed);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
void SetDeviceProviderCaps(DeviceType device, const ProviderCaps& caps) {
  VT_CHECK(static_cast<size_t>(device) < kNumDeviceTypes, "invalid device type");
  {
    CapsTable& t = Caps();
    std::lock_guard<std::mutex> lock(t.mu);
    t.caps[static_cast<size_t>(device)] = caps;
    t.caps[static_cast<size_t>(device)].device = device;
  }
  // A predicate may have declined against unprobed capabilities; drop the cache
  // so the next dispatch re-selects against the real record.
  InvalidateAll();
}

ProviderCaps GetDeviceProviderCaps(DeviceType device) {
  VT_CHECK(static_cast<size_t>(device) < kNumDeviceTypes, "invalid device type");
  CapsTable& t = Caps();
  std::lock_guard<std::mutex> lock(t.mu);
  ProviderCaps c = t.caps[static_cast<size_t>(device)];
  c.device = device;
  return c;
}

void RegisterOpProvider(OpId op, DeviceType device, const OpProvider& provider) {
  // Table fill only; NEVER throws (registrars run before main, where a throw has
  // no receiver) — the RegisterArchTactic contract, cuda_arch_tactics.cu:126-131.
  if (static_cast<size_t>(op) >= kOpCount) return;
  if (static_cast<size_t>(device) >= kNumDeviceTypes) return;
  if (provider.fn == nullptr || provider.name == nullptr) return;
  Slot& slot = At(op, device);
  if (slot.count >= kMaxProvidersPerOp) return;
  for (int i = 0; i < slot.count; ++i) {
    // A duplicate name would make the order non-total. First registration wins,
    // which is deterministic for the only case that can legitimately occur (a
    // single provider registering twice); two DIFFERENT providers sharing a name
    // is a naming bug the test suite catches.
    if (std::strcmp(slot.providers[i].name, provider.name) == 0) return;
  }
  slot.providers[slot.count++] = provider;
  slot.selected.store(nullptr, std::memory_order_relaxed);
  slot.resolved_none.store(false, std::memory_order_relaxed);
  slot.native_registered.store(-1, std::memory_order_relaxed);
}

void RegisterOp(OpId op, DeviceType device, void* fn) {
  VT_CHECK(static_cast<size_t>(op) < kOpCount, "invalid op id");
  VT_CHECK(static_cast<size_t>(device) < kNumDeviceTypes, "invalid device type");
  OpProvider p;
  p.name = kNativeProviderName;
  p.priority = 0;
  p.supports = nullptr;
  p.fn = fn;
  RegisterOpProvider(op, device, p);
}

void* GetOp(OpId op, DeviceType device) {
  Slot& slot = At(op, device);
  void* fn = slot.selected.load(std::memory_order_relaxed);
  if (fn == nullptr) fn = Resolve(op, device, slot);
  if (CallStatsFlag().load(std::memory_order_relaxed)) {
    slot.selections.fetch_add(1, std::memory_order_relaxed);
  }
  return fn;
}

void* GetOpFallback(OpId op, DeviceType device, const char* declining_provider) {
  VT_CHECK(declining_provider != nullptr, "op provider fallback requires a provider name");
  Slot& slot = At(op, device);
  const OpProvider* floor = nullptr;
  for (int i = 0; i < slot.count; ++i) {
    if (std::strcmp(slot.providers[i].name, declining_provider) == 0) {
      floor = &slot.providers[i];
      break;
    }
  }
  VT_CHECK(floor != nullptr, std::string("declining provider '") + declining_provider +
                                 "' is not registered for this op/device");
  const ProviderCaps caps = GetDeviceProviderCaps(device);
  const OpProvider* next = Choose(slot, caps, floor);
  slot.declines.fetch_add(1, std::memory_order_relaxed);
  VT_CHECK(next != nullptr,
           std::string("provider '") + declining_provider + "' declined op " +
               std::to_string(static_cast<int>(op)) + " on device type " +
               std::to_string(static_cast<int>(device)) + " and no provider is below it");
  return next->fn;
}

bool OpRegistered(OpId op, DeviceType device) {
  // Meaning (unchanged, and load-bearing for the fused-recipe fast-realization
  // ladder and the cross-device harness): is there a NATIVE kernel for
  // (op, device)? The portable reference tier is a FALLBACK, not a native
  // kernel, so it is deliberately EXCLUDED here — otherwise a unified accelerator
  // would report every op as "registered" the moment its fallback installed, and
  // the ladder would stop choosing its portable composite path. It cannot use the
  // `selected` cache because `selected` can legitimately hold a reference-tier fn
  // (a real GetOp answer); it keeps its own native-only memo instead.
  Slot& slot = At(op, device);
  const int8_t cached = slot.native_registered.load(std::memory_order_relaxed);
  if (cached >= 0) return cached == 1;
  const ProviderCaps caps = GetDeviceProviderCaps(device);
  bool has_native = false;
  for (int i = 0; i < slot.count; ++i) {
    const OpProvider& p = slot.providers[i];
    if (std::strcmp(p.name, kReferenceProviderName) == 0) continue;  // fallback, not native
    if (Supported(p, caps)) {
      has_native = true;
      break;
    }
  }
  slot.native_registered.store(has_native ? 1 : 0, std::memory_order_relaxed);
  return has_native;
}

int OpProviderCount(OpId op, DeviceType device) { return At(op, device).count; }

const char* OpProviderNameAt(OpId op, DeviceType device, int i) {
  Slot& slot = At(op, device);
  if (i < 0 || i >= slot.count) return nullptr;
  // Report in SELECTION order, not storage order — the observable order is the
  // one the seam promises, so a test can assert it directly.
  const OpProvider* prev = nullptr;
  for (int rank = 0; rank <= i; ++rank) {
    const OpProvider* best = nullptr;
    for (int j = 0; j < slot.count; ++j) {
      const OpProvider& p = slot.providers[j];
      if (prev != nullptr && !Better(*prev, p)) continue;
      if (best == nullptr || Better(p, *best)) best = &p;
    }
    if (best == nullptr) return nullptr;
    prev = best;
  }
  return prev->name;
}

OpProviderStats GetOpProviderStats(OpId op, DeviceType device) {
  Slot& slot = At(op, device);
  OpProviderStats s;
  s.last_selected = slot.last_selected.load(std::memory_order_relaxed);
  s.selections = slot.selections.load(std::memory_order_relaxed);
  s.declines = slot.declines.load(std::memory_order_relaxed);
  s.fallbacks = slot.fallbacks.load(std::memory_order_relaxed);
  return s;
}

void EnableOpProviderCallStats(bool on) {
  CallStatsFlag().store(on, std::memory_order_relaxed);
}

void ResetOpProviderStats(OpId op, DeviceType device) {
  Slot& slot = At(op, device);
  slot.selections.store(0, std::memory_order_relaxed);
  slot.declines.store(0, std::memory_order_relaxed);
  slot.fallbacks.store(0, std::memory_order_relaxed);
}

void DisableOpProvider(const char* name, bool disabled) {
  if (name == nullptr) return;
  DisableList& d = Disabled();
  {
    std::lock_guard<std::mutex> lock(d.mu);
    EnsureEnvParsedLocked(d);
    const size_t n = std::strlen(name);
    if (disabled) {
      AddDisabledLocked(d, name, name + n);
    } else {
      for (int i = 0; i < d.count; ++i) {
        if (std::strcmp(d.names[i], name) == 0) {
          for (int j = i; j + 1 < d.count; ++j) {
            std::memcpy(d.names[j], d.names[j + 1], kMaxNameLen + 1);
          }
          --d.count;
          break;
        }
      }
    }
  }
  InvalidateAll();
}

bool OpProviderDisabled(const char* name) {
  if (name == nullptr) return false;
  DisableList& d = Disabled();
  if (d.live_count.load(std::memory_order_relaxed) == 0) return false;  // lock-free
  std::lock_guard<std::mutex> lock(d.mu);
  EnsureEnvParsedLocked(d);
  for (int i = 0; i < d.count; ++i) {
    if (std::strcmp(d.names[i], name) == 0) return true;
  }
  return false;
}

// --- Portable reference tier (S5) ------------------------------------------
bool ReferenceTierEligible(DeviceType device) {
  // The CPU is the SOURCE of the reference kernels, never a fallback target
  // (falling back to itself is a no-op at best and self-reference at worst).
  if (device == DeviceType::kCPU) return false;
  // THE SAFETY GATE. A CPU kernel dereferences host pointers, which is correct
  // ONLY where host and device memory alias. Gate on the unified-memory property
  // of the ACTUAL registered backend, not on DeviceType: a discrete GPU (CUDA or
  // Vulkan) answers false and never receives a CPU fallback. A device with no
  // backend in this build is trivially ineligible.
  Backend* b = TryGetBackend(device);
  return b != nullptr && b->UnifiedMemory();
}

int RegisterReferenceTier(DeviceType target) {
  if (!ReferenceTierEligible(target)) return 0;
  int installed = 0;
  for (size_t o = 0; o < kOpCount; ++o) {
    const OpId op = static_cast<OpId>(o);
    // Skip ops the target already serves natively — the tier is a fallback for
    // MISSING kernels only, and installing under a native provider is wasted
    // capacity (the native one wins by priority regardless).
    if (OpRegistered(op, target)) continue;
    if (MaybeInstallReferenceTier(op, target)) ++installed;
  }
  return installed;
}

unsigned long long GetReferenceTierHits() {
  return RefTierHits().load(std::memory_order_relaxed);
}

}  // namespace vt

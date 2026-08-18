// #839 shared prefill-peer lifetime. Product HIP and host tests execute these
// transitions. Failed retirement quarantines the pin (sole zero-pin exception).
#pragma once

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace vt::rocm {

inline int PrefillDequantCacheSlots() { return 1; }
inline constexpr int kPrefillDequantCacheMaxSlots = 128;

enum class PrefillRetireOutcome { None, Unpinned, Quarantined };
enum class PrefillRetireTarget { None, ComputeStream, ExpertStream, RecordedEvent };

// Finish output copy on a specific compute stream. Reuse/reconfigure of tls.y
// is forbidden until that stream is host-observed retired.
struct OutputCopyGate {
  int copy_stream = -1;
  bool pending = false;
  void Enqueue(int sid) {
    copy_stream = sid;
    pending = true;
  }
  bool Retire(int sid) {
    if (!pending) return true;
    if (sid != copy_stream) return false;
    pending = false;
    copy_stream = -1;
    return true;
  }
  bool CanReuseScratch() const { return !pending; }
};

struct PrefillPeerLife {
  int cache_pin = -1;
  int cache_dev = -1;
  int pending_M = 0;
  bool ev_e_recorded = false;
  bool this_gen_ev_e = false;
  bool work_enqueued = false;
  bool work_on_compute = false;
  bool work_on_expert = false;
  bool rollback_armed = false;
  bool quarantined = false;
  bool compute_restored = false;
  bool fill_lease = false;
  OutputCopyGate output_copy{};

  // Arm after the first successful current-generation enqueue.
  void ArmRollback() {
    rollback_armed = true;
    work_enqueued = true;
    this_gen_ev_e = false;
  }
  void MarkComputeWork() { work_on_compute = true; }
  void MarkExpertWork() { work_on_expert = true; }
  void MarkThisGenEvent() {
    this_gen_ev_e = true;
    ev_e_recorded = true;
    rollback_armed = false;
  }
  void OnSuccessfulRetire() {
    cache_pin = -1;
    cache_dev = -1;
    ev_e_recorded = false;
    this_gen_ev_e = false;
    rollback_armed = false;
    work_on_compute = false;
    work_on_expert = false;
    fill_lease = false;
  }
};

// Map any slot-like object that carries the product flags.
template <typename Slot>
PrefillPeerLife LifeFromSlot(const Slot& tls) {
  PrefillPeerLife life;
  life.cache_pin = tls.cache_pin;
  life.cache_dev = tls.cache_dev;
  life.this_gen_ev_e = tls.this_gen_ev_e;
  life.ev_e_recorded = tls.ev_e_recorded;
  life.rollback_armed = tls.rollback_armed;
  life.quarantined = tls.quarantined;
  life.work_on_compute = tls.work_on_compute;
  life.work_on_expert = tls.work_on_expert;
  life.fill_lease = tls.fill_lease;
  return life;
}

inline bool SlotReusable(bool pending, bool rollback_armed, bool quarantined) {
  return !pending && !rollback_armed && !quarantined;
}

// this_gen ev_e wins. Else expert-stream work, else compute-stream work.
// Never treat a leftover ev_e_recorded as the current rollback target.
inline PrefillRetireTarget ChoosePrefillRetire(const PrefillPeerLife& life) {
  if (life.this_gen_ev_e) return PrefillRetireTarget::RecordedEvent;
  if (life.work_on_expert) return PrefillRetireTarget::ExpertStream;
  if (life.work_on_compute || life.rollback_armed || life.cache_pin >= 0)
    return PrefillRetireTarget::ComputeStream;
  return PrefillRetireTarget::None;
}

struct RestoreFailed : std::runtime_error {
  RestoreFailed() : std::runtime_error("prefill-peer compute device restore failed") {}
};

template <typename SetDev>
struct ComputeDevGuard {
  int dev = -1;
  SetDev set{};
  bool done = false;

  ComputeDevGuard(int d, SetDev s) : dev(d), set(std::move(s)) {}
  ComputeDevGuard(const ComputeDevGuard&) = delete;
  ComputeDevGuard& operator=(const ComputeDevGuard&) = delete;

  void RestoreOrThrow() {
    if (done) return;
    const bool ok = set(dev);
    done = true;  // failure is thrown; dtor must not terminate on the same attempt
    if (!ok) throw RestoreFailed{};
  }

  // Failed restore is fatal even during exception unwind (c2ae).
  ~ComputeDevGuard() noexcept {
    if (done) return;
    if (!set(dev)) std::terminate();
    done = true;
  }
};

struct SameDevLife {
  int cache_pin = -1;
  int cache_dev = -1;
  bool quarantined = false;
  bool CanEnter() const { return !quarantined; }
  bool CanReconfigure() const { return !quarantined && cache_pin < 0; }
  void PersistPin(int pin, int dev) {
    cache_pin = pin;
    cache_dev = dev;
  }
  void Quarantine(int pin, int dev) {
    cache_pin = pin;
    cache_dev = dev;
    quarantined = true;
  }
  void ClearPin() {
    cache_pin = -1;
    cache_dev = -1;
    quarantined = false;
  }
};

template <typename Hooks>
struct PrefillDequantCacheT {
  struct Slot {
    const void* key = nullptr;
    void* gu = nullptr;
    void* dn = nullptr;
    uint64_t lru = 0;
    int pins = 0;
    bool ready = false;
    bool filling = false;
    bool fill_failed = false;
    typename Hooks::Event ready_ev{};
  };

  int dev = -1;
  int I = 0, H = 0;
  int nslots = 0;
  uint64_t clock = 0;
  uint64_t hits = 0, misses = 0;
  Slot slots[kPrefillDequantCacheMaxSlots]{};
  std::mutex mu;
  Hooks hooks{};

  void FreeAll() {
    for (int i = 0; i < kPrefillDequantCacheMaxSlots; ++i) {
      if (slots[i].gu) {
        hooks.Free(slots[i].gu);
        slots[i].gu = nullptr;
      }
      if (slots[i].dn) {
        hooks.Free(slots[i].dn);
        slots[i].dn = nullptr;
      }
      hooks.DestroyEvent(slots[i].ready_ev);
      slots[i] = Slot{};
    }
    nslots = 0;
    dev = -1;
    I = H = 0;
  }

  int LivePins() const {
    int n = 0;
    for (int i = 0; i < kPrefillDequantCacheMaxSlots; ++i) n += slots[i].pins;
    return n;
  }

  int LiveAllocs() const {
    int n = 0;
    for (int i = 0; i < kPrefillDequantCacheMaxSlots; ++i) {
      if (slots[i].gu) ++n;
      if (slots[i].dn) ++n;
    }
    return n;
  }

  bool SlotNonReusable(int i) const {
    return slots[i].filling || slots[i].fill_failed || slots[i].pins > 0;
  }

  bool Ensure(int device, int i_dim, int h_dim) {
    const int want = PrefillDequantCacheSlots();
    if (dev == device && I == i_dim && H == h_dim && nslots == want) return true;
    if (LivePins() > 0) return false;
    FreeAll();
    if (device < 0 || i_dim <= 0 || h_dim <= 0 || want <= 0) return false;
    if (!hooks.SetDevice(device)) return false;
    const size_t gu_b = static_cast<size_t>(2 * i_dim) * static_cast<size_t>(h_dim) * 2;
    const size_t dn_b = static_cast<size_t>(h_dim) * static_cast<size_t>(i_dim) * 2;
    for (int i = 0; i < want; ++i) {
      slots[i].gu = hooks.Malloc(gu_b);
      if (!slots[i].gu) {
        FreeAll();
        return false;
      }
      slots[i].dn = hooks.Malloc(dn_b);
      if (!slots[i].dn) {
        FreeAll();
        return false;
      }
    }
    dev = device;
    I = i_dim;
    H = h_dim;
    nslots = want;
    clock = 0;
    return true;
  }

  // pin_out is written as soon as a fill lease or hit pin is taken, including
  // failed Fill/RecordReady — caller must persist/retire that pin.
  bool GetLocked(const void* key, void** gu_out, void** dn_out, int* pin_out) {
    if (!key || !gu_out || !dn_out || nslots <= 0) return false;
    int hit = -1;
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < nslots; ++i) {
      if (slots[i].key == key && slots[i].ready && !slots[i].filling && !slots[i].fill_failed) {
        hit = i;
        break;
      }
      if (SlotNonReusable(i)) continue;
      if (slots[i].key == nullptr) {
        if (victim < 0) victim = i;
      } else if (slots[i].lru < oldest) {
        oldest = slots[i].lru;
        if (victim < 0 || slots[victim].key != nullptr) victim = i;
      }
    }
    if (hit >= 0) {
      if (!hooks.WaitReady(slots[hit].ready_ev)) return false;
      ++hits;
      slots[hit].lru = ++clock;
      slots[hit].pins++;
      *gu_out = slots[hit].gu;
      *dn_out = slots[hit].dn;
      if (pin_out) *pin_out = hit;
      return true;
    }
    if (victim < 0) {
      for (int i = 0; i < nslots; ++i) {
        if (!SlotNonReusable(i)) {
          victim = i;
          break;
        }
      }
      if (victim < 0) return false;
    }
    Slot& s = slots[victim];
    s.key = nullptr;  // unpublish before fill
    s.ready = false;
    s.filling = true;
    s.fill_failed = false;
    s.pins++;  // fill lease from first enqueue
    if (pin_out) *pin_out = victim;
    if (!hooks.Fill(s.gu, s.dn)) {
      s.fill_failed = true;
      return false;
    }
    if (!hooks.RecordReady(s.ready_ev)) {
      s.fill_failed = true;
      return false;
    }
    s.ready = true;
    s.key = key;
    s.filling = false;
    ++misses;
    s.lru = ++clock;
    *gu_out = s.gu;
    *dn_out = s.dn;
    return true;
  }

  void UnpinLocked(int idx) {
    if (idx < 0 || idx >= nslots) return;
    if (slots[idx].pins > 0) slots[idx].pins--;
  }

  bool IsFillLease(int idx) const {
    if (idx < 0 || idx >= nslots) return false;
    return slots[idx].filling || slots[idx].fill_failed;
  }

  // After a successful owning-stream retire: fill-lease uses RetireFillLocked,
  // ready pin uses UnpinLocked. Caller must not hold mu around the stream sync.
  bool ReleaseObservedPinLocked(int idx, bool fill_lease) {
    if (idx < 0 || idx >= nslots) return true;
    if (fill_lease || IsFillLease(idx)) return RetireFillLocked(idx, true);
    UnpinLocked(idx);
    return true;
  }

  // Producer-stream retirement of a failed fill lease.
  bool RetireFillLocked(int idx, bool retire_ok) {
    if (idx < 0 || idx >= nslots) return false;
    if (!retire_ok) {
      slots[idx].fill_failed = true;
      return false;
    }
    slots[idx].filling = false;
    slots[idx].fill_failed = false;
    if (slots[idx].pins > 0) slots[idx].pins--;
    return true;
  }
};

struct HostAlloc {
  int next = 1;
  int mallocs = 0;
  int frees = 0;
  int fill_calls = 0;
  int fail_malloc_at = -1;
  bool fail_record = false;
  bool fail_wait = false;
  bool fail_set_device = false;
  bool fail_fill = false;
  struct Event {
    bool recorded = false;
  };
  bool SetDevice(int) { return !fail_set_device; }
  void* Malloc(size_t) {
    ++mallocs;
    if (fail_malloc_at >= 0 && mallocs == fail_malloc_at) return nullptr;
    return reinterpret_cast<void*>(static_cast<intptr_t>(++next));
  }
  void Free(void* p) {
    if (p) ++frees;
  }
  void DestroyEvent(Event& e) { e.recorded = false; }
  bool Fill(void*, void*) {
    ++fill_calls;
    return !fail_fill;
  }
  bool RecordReady(Event& e) {
    if (fail_record) return false;
    e.recorded = true;
    return true;
  }
  bool WaitReady(Event& e) { return e.recorded && !fail_wait; }
};

using PrefillDequantCacheHost = PrefillDequantCacheT<HostAlloc>;

inline PrefillRetireOutcome RetirePinIfObserved(PrefillDequantCacheHost& cache, int& cache_pin,
                                                bool retire_ok, bool fill_lease = false) {
  if (cache_pin < 0) return PrefillRetireOutcome::None;
  if (!retire_ok) return PrefillRetireOutcome::Quarantined;
  std::lock_guard<std::mutex> lk(cache.mu);
  if (!cache.ReleaseObservedPinLocked(cache_pin, fill_lease)) return PrefillRetireOutcome::Quarantined;
  cache_pin = -1;
  return PrefillRetireOutcome::Unpinned;
}

// Product publish-then-restore: retire/quarantine before rethrowing RestoreFailed.
template <typename Slot, typename Restore, typename Retire>
void PublishThenRestoreOrThrow(Slot& slot, int M, Restore&& restore, Retire&& retire) {
  slot.pending_M = M;
  slot.rollback_armed = false;
  try {
    restore();
  } catch (const RestoreFailed&) {
    (void)retire();
    slot.pending_M = 0;
    throw;
  }
}

enum class PrefillPeerFailAt {
  None,
  AfterFirstEnqueue,
  AfterPinEnqueue,
  RecordEvE,
  AfterRecord,
  AfterWait,
  AfterCopy,
  AfterSameDevAcquire,
  AfterSameDevGemm,
};

using PrefillPeerSlotHost = PrefillPeerLife;

inline bool HostRetireThenUnpin(PrefillDequantCacheHost& cache, PrefillPeerLife& slot,
                                bool retire_ok) {
  const auto target = ChoosePrefillRetire(slot);
  if (target == PrefillRetireTarget::None && slot.cache_pin < 0) return true;
  if (!retire_ok) {
    slot.quarantined = true;
    return false;
  }
  const auto out = RetirePinIfObserved(cache, slot.cache_pin, /*retire_ok=*/true, slot.fill_lease);
  if (out == PrefillRetireOutcome::Unpinned || out == PrefillRetireOutcome::None) {
    slot.OnSuccessfulRetire();
    return true;
  }
  slot.quarantined = true;
  return false;
}

// Product-shaped Launch: arm rollback only after the first enqueue, then every
// later failure retires the streams that actually received work.
inline bool HostLaunch(PrefillDequantCacheHost& cache, PrefillPeerLife& slot, int device,
                       const void* key, int M, PrefillPeerFailAt fail, bool retire_ok = true,
                       bool restore_ok = true) {
  if (!SlotReusable(slot.pending_M > 0, slot.rollback_armed, slot.quarantined) ||
      !slot.output_copy.CanReuseScratch())
    return false;
  auto set = [restore_ok](int) { return restore_ok; };
  ComputeDevGuard<decltype(set)> guard(device, set);
  slot.compute_restored = false;

  // First current-generation enqueue (compute-stream ev_c analogue).
  slot.ArmRollback();
  slot.MarkComputeWork();
  if (fail == PrefillPeerFailAt::AfterFirstEnqueue) {
    if (!HostRetireThenUnpin(cache, slot, retire_ok)) {
      slot.compute_restored = true;
      guard.RestoreOrThrow();
      return false;
    }
    guard.RestoreOrThrow();
    slot.compute_restored = true;
    return false;
  }
  slot.MarkExpertWork();

  int pin = -1;
  void* gu = nullptr;
  void* dn = nullptr;
  bool got = false;
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    got = cache.Ensure(device, 4, 8) && cache.GetLocked(key, &gu, &dn, &pin);
    if (!got && pin >= 0) {
      slot.cache_pin = pin;
      slot.cache_dev = device;
      slot.fill_lease = true;
    }
  }
  if (!got) {
    if (!HostRetireThenUnpin(cache, slot, retire_ok)) {
      slot.compute_restored = true;
      guard.RestoreOrThrow();
      return false;
    }
    guard.RestoreOrThrow();
    slot.compute_restored = true;
    return false;
  }
  slot.cache_pin = pin;  // persist immediately on acquisition
  slot.cache_dev = device;
  slot.fill_lease = false;
  if (fail == PrefillPeerFailAt::AfterPinEnqueue || fail == PrefillPeerFailAt::RecordEvE) {
    if (!HostRetireThenUnpin(cache, slot, retire_ok)) {
      slot.compute_restored = true;
      guard.RestoreOrThrow();
      return false;
    }
    guard.RestoreOrThrow();
    slot.compute_restored = true;
    return false;
  }
  slot.MarkThisGenEvent();
  if (fail == PrefillPeerFailAt::AfterRecord) {
    if (!HostRetireThenUnpin(cache, slot, retire_ok)) {
      slot.compute_restored = true;
      guard.RestoreOrThrow();
      return false;
    }
    guard.RestoreOrThrow();
    slot.compute_restored = true;
    return false;
  }
  PublishThenRestoreOrThrow(
      slot, M, [&] { guard.RestoreOrThrow(); },
      [&] { return HostRetireThenUnpin(cache, slot, retire_ok); });
  slot.compute_restored = true;
  return true;
}

inline bool HostFinish(PrefillDequantCacheHost& cache, PrefillPeerLife& slot, int M,
                       PrefillPeerFailAt fail, bool retire_ok = true, bool restore_ok = true,
                       int copy_stream = 0, int retire_stream = 0) {
  if (slot.pending_M <= 0 || M > slot.pending_M) return false;
  auto set = [restore_ok](int) { return restore_ok; };
  ComputeDevGuard<decltype(set)> guard(0, set);
  if (fail == PrefillPeerFailAt::AfterWait) {
    if (!HostRetireThenUnpin(cache, slot, retire_ok)) {
      slot.compute_restored = true;
      guard.RestoreOrThrow();
      return false;
    }
    slot.pending_M = 0;
    guard.RestoreOrThrow();
    slot.compute_restored = true;
    return false;
  }
  slot.output_copy.Enqueue(copy_stream);
  if (fail == PrefillPeerFailAt::AfterCopy) {
    if (!HostRetireThenUnpin(cache, slot, retire_ok)) {
      slot.compute_restored = true;
      guard.RestoreOrThrow();
      return false;
    }
    (void)slot.output_copy.Retire(copy_stream);
    slot.pending_M = 0;
    guard.RestoreOrThrow();
    slot.compute_restored = true;
    return false;
  }
  if (!slot.output_copy.Retire(retire_stream)) {
    slot.compute_restored = true;
    guard.RestoreOrThrow();
    return false;
  }
  if (!HostRetireThenUnpin(cache, slot, retire_ok)) {
    slot.compute_restored = true;
    guard.RestoreOrThrow();
    return false;
  }
  slot.pending_M = 0;
  guard.RestoreOrThrow();
  slot.compute_restored = true;
  return true;
}

// Same-device product seam: persist pin on acquire; GEMM readers run while
// pinned; unpin only after observed retire. Early UnpinLocked is a RED mutation.
struct SameDevSession {
  PrefillDequantCacheHost* cache = nullptr;
  SameDevLife* life = nullptr;
  int pin = -1;
  bool gemm_readers_ran = false;
  bool unpinned_before_gemm = false;

  bool Acquire(const void* key, int device) {
    if (!cache || !life || !life->CanEnter()) return false;
    void* gu = nullptr;
    void* dn = nullptr;
    bool got = false;
    {
      std::lock_guard<std::mutex> lk(cache->mu);
      got = cache->Ensure(device, 4, 8) && cache->GetLocked(key, &gu, &dn, &pin);
      if (!got && pin >= 0) life->PersistPin(pin, device);
    }
    if (!got) {
      if (pin >= 0) {
        std::lock_guard<std::mutex> lk(cache->mu);
        (void)cache->ReleaseObservedPinLocked(pin, /*fill_lease=*/true);
        life->ClearPin();
        pin = -1;
      }
      return false;
    }
    life->PersistPin(pin, device);
    return true;
  }

  bool RunGemmReaders() {
    if (!cache || pin < 0) return false;
    if (cache->slots[pin].pins <= 0) {
      unpinned_before_gemm = true;
      return false;
    }
    gemm_readers_ran = true;
    return true;
  }

  bool Retire(bool sync_ok) {
    if (!cache || !life) return false;
    if (pin < 0) return true;
    if (!sync_ok) {
      life->Quarantine(pin, life->cache_dev);
      return false;
    }
    {
      std::lock_guard<std::mutex> lk(cache->mu);
      cache->UnpinLocked(pin);
    }
    life->ClearPin();
    pin = -1;
    return true;
  }
};

}  // namespace vt::rocm

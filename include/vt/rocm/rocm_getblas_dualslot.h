// #837 GetBlas dual-slot TLS — slot index + hookable lifetime engine.
// Product GetBlas and host tests both execute RocmProductGetBlasOn
// (research c24b: substring wiring is not a load-bearing seam).
#pragma once

namespace vt::rocm {

// Device 1 owns slot 1. Every other device id (0, 2, ...) shares slot 0.
// Two-GPU lab assumption; do not invent a map (spec #837).
inline int GetBlasSlotIndex(int device) { return (device == 1) ? 1 : 0; }

// Dual-slot GetBlas lifetime. Hooks must provide:
//   handle_t, stream_t
//   NullHandle(), NullStream(), IsNull(handle)
//   StreamIsCapturing(stream) -> bool
//   GetDevice() -> int
//   SetDevice(int)
//   Create() -> handle_t
//   Destroy(handle_t)
//   SetStream(handle_t, stream_t)
template <class Hooks, int (*SlotIndex)(int) = GetBlasSlotIndex>
struct GetBlasDualSlotEngine {
  using handle_t = typename Hooks::handle_t;
  using stream_t = typename Hooks::stream_t;

  struct Tls {
    int dev = -1;
    stream_t stream{};
    handle_t handle{};
  };

  Tls tls_slots[2]{};

  handle_t Get(int device, stream_t stream, Hooks& hooks) {
    Tls& tls = tls_slots[SlotIndex(device)];
    if (!hooks.StreamIsCapturing(stream)) {
      const int cur = hooks.GetDevice();
      if (cur != device) hooks.SetDevice(device);
    }
    if (hooks.IsNull(tls.handle) || tls.dev != device) {
      if (!hooks.IsNull(tls.handle)) {
        hooks.Destroy(tls.handle);
        tls.handle = hooks.NullHandle();
      }
      if (!hooks.StreamIsCapturing(stream)) hooks.SetDevice(device);
      tls.handle = hooks.Create();
      tls.dev = device;
      tls.stream = hooks.NullStream();
    }
    if (tls.stream != stream) {
      hooks.SetStream(tls.handle, stream);
      tls.stream = stream;
    }
    return tls.handle;
  }
};

// Product-call seam: production GetBlas and host tests both execute this.
// Forwards device + stream unchanged. Mutating either argument is RED.
struct RocmGetBlasForward {
  template <class Engine, class Hooks>
  static typename Engine::handle_t apply(Engine& engine, int device,
                                         typename Engine::stream_t stream,
                                         Hooks& hooks) {
    return engine.Get(device, stream, hooks);
  }
};

template <class Engine, class Hooks, class Forward = RocmGetBlasForward>
inline typename Engine::handle_t RocmProductGetBlasOn(
    Engine& engine, int device, typename Engine::stream_t stream, Hooks& hooks) {
  return Forward::apply(engine, device, stream, hooks);
}

}  // namespace vt::rocm

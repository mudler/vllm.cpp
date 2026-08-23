// Syntax-check stub for mlx/stream.h. See ../README.md.
//
// The provider asks for one stream, `default_stream(Device::gpu)`, and passes
// it to matmul/transpose/astype. `Device::gpu` is an enumerator of a nested
// enum in the real header, which is why it is spelled that way here.
#ifndef VT_METAL_STUBS_MLX_STREAM_H_
#define VT_METAL_STUBS_MLX_STREAM_H_

namespace mlx::core {

struct Device {
  enum DeviceEnum { cpu, gpu };
  // Non-explicit on purpose: `default_stream(mx::Device::gpu)` passes the bare
  // enumerator and relies on this conversion, exactly as the real header does.
  constexpr Device(DeviceEnum t) : type(t) {}  // NOLINT(google-explicit-constructor)
  DeviceEnum type;
};

struct Stream {
  int index = 0;
  Device device{Device::gpu};
};

Stream default_stream(Device device);

}  // namespace mlx::core

#endif  // VT_METAL_STUBS_MLX_STREAM_H_

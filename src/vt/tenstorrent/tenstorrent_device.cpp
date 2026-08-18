// vllm.cpp original (BACKEND-TENSTORRENT W0); no upstream mirror.
#include "vt/tenstorrent/tenstorrent_device.h"

#include <memory>

#include <ttnn/device.hpp>
#include <tt-metalium/host_api.hpp>

namespace vt::tenstorrent {

bool DeviceAvailable() { return tt::tt_metal::GetNumAvailableDevices() > 0; }

MeshDevice& SharedMeshDevice() {
  // DELIBERATE LEAK, not an oversight: a plain `static std::shared_ptr`
  // reproducibly SEGFAULTS on process exit (tests/vt/test_tenstorrent_backend.cpp,
  // 2026-08-09) inside MeshDevice's own destructor chain
  // (GraphTracker::track_deallocate_cb <- ProgramImpl::deallocate_circular_buffers
  // <- ~MeshWorkloadImpl), i.e. C++ static destruction order across the
  // libtt_metal.so boundary tears down some tt_metal-internal global this
  // teardown path depends on before our function-local static's destructor
  // runs. The standalone spike this backend is modeled on (spec's "Resolved:
  // hands-on spike result") did NOT hit this: it held the device in a
  // `main()`-local variable, destroyed deterministically before any static
  // teardown begins, not during it. Allocating on the heap and never
  // `delete`-ing means MeshDevice's destructor never runs at process exit at
  // all, sidestepping the ordering problem entirely — safe in practice (the
  // OS/kernel driver reclaim the PCIe device's file descriptors and hardware
  // state on process exit regardless of a userspace close() call, the same
  // assumption CUDA processes routinely rely on), if not textbook-clean.
  // ITEM 5: reserve a dedicated DRAM trace region (50 MB, the tt-metal vLLM
  // plugin's value — worker.py:710) so allocations during trace capture can
  // never overlap the trace buffer. With the default (0), the trace buffer
  // is carved from the general pool and ANY capture-time allocation fatals
  // with "Trace buffer ... overlaps with DRAM activity".
  static std::shared_ptr<MeshDevice>* device = new std::shared_ptr<MeshDevice>(
      ttnn::open_mesh_device(
          /*device_id=*/0, /*l1_small_size=*/DEFAULT_L1_SMALL_SIZE,
          /*trace_region_size=*/50 * 1024 * 1024));
  return **device;
}

}  // namespace vt::tenstorrent

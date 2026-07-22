# Vendored third-party (headers only)

| Library | Version | Source | License |
|---|---|---|---|
| doctest | v2.5.2 | github.com/doctest/doctest | MIT |
| nlohmann/json | v3.12.0 | github.com/nlohmann/json | MIT |
| cpp-httplib | v0.49.0 | github.com/yhirose/cpp-httplib | MIT |
| Vulkan-Headers (`vulkan_core.h`, `vk_platform.h`) | vulkan-sdk-1.4.328.1 (`VK_HEADER_VERSION` 328) | github.com/KhronosGroup/Vulkan-Headers | Apache-2.0 |

Update procedure: re-download the pinned header(s) at a newer tag, update this
table, note it in .agents/parity-ledger.md. No compiled deps allowed
(.agents/discipline.md).

**Vulkan-Headers is the one entry that is two headers rather than one**
(`vulkan_core.h` includes `vk_platform.h`), and it is worth stating why it is
vendored at all rather than found with `find_package(Vulkan)`: neither of our
boxes has the Vulkan development package (no `/usr/include/vulkan`, no
`libvulkan.so` link name, no sudo — measured 2026-07-22), only the runtime
loader. These are the Khronos-generated API TYPE declarations and nothing else:
they are compiled with `VK_NO_PROTOTYPES`, so no symbol is imported from them and
nothing is linked; the entry points are resolved with `dlopen`/`vkGetInstanceProcAddr`
at run time (`src/vt/vulkan/vulkan_loader.cpp`). Hand-declaring the subset we use
was the alternative and was rejected: a single wrong struct field would be silent
memory corruption rather than a compile error. The pin matches the loader version
on `dgx.casa` (1.4.328). Version bumps are a mechanical re-download; the backend
uses only Vulkan 1.1 core.

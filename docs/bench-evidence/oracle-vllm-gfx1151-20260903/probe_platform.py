import amdsmi
amdsmi.amdsmi_init()
print("AMDSMI_HANDLES =", len(amdsmi.amdsmi_get_processor_handles()))
amdsmi.amdsmi_shut_down()
from vllm.platforms import current_platform
print("CURRENT_PLATFORM =", current_platform.__class__.__name__)
print("DEVICE_NAME      =", current_platform.get_device_name(0))
print("DEVICE_CAP       =", current_platform.get_device_capability(0))
assert current_platform.__class__.__name__ == "RocmPlatform", current_platform
from vllm.platforms.rocm import on_gfx1151, on_gfx1x
print("ON_GFX1151 =", on_gfx1151(), " ON_GFX1X =", on_gfx1x())
assert on_gfx1151()
print("PLATFORM_RESOLVES=RocmPlatform")

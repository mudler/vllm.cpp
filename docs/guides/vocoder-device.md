# Select the vocoder device

Use this workflow to compare CPU and accelerator vocoder execution.

`VLLM_CPP_VOCODER_DEVICE=cuda` routes `vt::Conv1d` and `vt::ConvTranspose1d` to
their CUDA providers for every model that decodes through the shared vocoder
core. It needs a CUDA build; asking for it without one throws by name rather than
falling back silently, because a silent fallback means an operator who asked for
a device never learns they did not get one.

The knob is not CUDA-specific. It accepts any device name `vt` knows (`cpu`,
`cuda`, `metal`, `vulkan`, `xpu`, `rocm`, `tenstorrent`) and refuses one whose
device carries no registered provider in the build in front of it, so a Metal or
Vulkan provider becomes reachable here by being registered and nothing else.


The default is `cpu`. The CPU and CUDA providers produce byte-identical output.
The device arm remains opt-in until each audio model completes its device gate.
See [the MiniMax-Music3 spec](../../.agents/specs/minimax-music3.md) for the
validation evidence.

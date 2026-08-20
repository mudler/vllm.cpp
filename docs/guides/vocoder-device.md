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

The default is `cpu`, and deliberately so, not because the device arm is
approximate. The two providers are **byte-identical**: one f64 accumulator per
output element walked in the same order on both, with the host pinned
`-ffp-contract=off` and the device kernel pinned with `__dmul_rn`/`__dadd_rn`, so
`tests/vt/test_ops_conv1d_general.cpp` gates them with `memcmp` rather than a
tolerance (8 cases / 385 assertions on Jetson Thor sm_110, against 8 / 347 on a
CPU-only box, the 38-assertion difference IS the device arm). It stays opt-in
because flipping four shipped audio models onto a device arm needs its own
re-gate against each one's committed goldens, which is owed to the row that
wires it ([#672](https://github.com/mudler/vllm.cpp/issues/672),
[.agents/specs/minimax-music3.md](../../.agents/specs/minimax-music3.md) §13).

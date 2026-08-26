# Syntax-check stubs for `metal_mlx_provider.mm`

These headers exist so `src/vt/metal/metal_mlx_provider.mm` can be COMPILED on a
host with no Mac, no Metal and no MLX, by the never-linked
`vllm_metal_mlx_provider_syntax_check` OBJECT library in `CMakeLists.txt`. They
are the same idea as `src/vt/cuda/flash_attn/stubs/{ATen,c10}`, which let a
vendored CUDA translation unit compile without torch.

**What the stubs buy.** The provider's real dependency is the `vt::OpProvider`
seam (`include/vt/op_provider.h`, `include/vt/ops.h`), and that is ordinary
portable C++. Stubbing only the FOREIGN dependencies leaves the seam real, so a
rename or a signature change in our own header fails a Linux build instead of
being discovered when a release is cut. Issue #1584 is the case: its edit to
`MlxFallback` landed unbuilt because nothing here compiled the file.

**What they do NOT buy, and cannot.** They are written FROM the provider's call
sites, so they can never disagree with it. This gate is therefore blind to every
MLX API change; only the `mlx_arm64` job in `.github/workflows/release.yml`,
which builds against the real `mlx` wheel on `macos-15`, can see one. Do not
read a green syntax check as "the MLX provider builds".

Keep them minimal. A declaration the provider does not name does not belong
here, because an unused declaration is one more thing that can be wrong without
anything noticing.

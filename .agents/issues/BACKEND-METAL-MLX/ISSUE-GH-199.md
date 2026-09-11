ID: ISSUE-GH-199
Title: [Bug] macOS MLX build fails because warnings in MLX headers are treated as errors (-Werror)
Row: BACKEND-METAL-MLX
State: CLOSED
Kind: bug
GitHub: 199
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-09
Updated: 2026-08-09
Closed: 2026-08-09

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Hi,
>
> Building vllm.cpp on macOS fails while compiling metal_mlx_provider.mm because compiler warnings originating from MLX headers are promoted to errors by -Werror.
>
> I reproduced the issue consistently across multiple macOS 26.6 releases and multiple MLX versions, including older MLX releases.
>
> Reproduced configurations
>
> macOS:
>
> * 26.6.1
> * 26.6.0
>
> MLX:
>
> * 0.32.0
> * 0.31.2
> * 0.30.6
>
> The same compilation failure occurs with all of these combinations, on both master and latest release of vllm.cpp
>
> # Build error
> `
> In file included from /Users/angusyoung/dev/vllm.cpp/src/vt/metal/metal_mlx_provider.mm:59:
> /usr/local/include/mlx/allocator.h:39:36: error: unused parameter 'ptr' [-Werror,-Wunused-parameter]
>    39 |   virtual Buffer make_buffer(void* ptr, size_t size) {
>       |                                    ^
> /usr/local/include/mlx/allocator.h:39:48: error: unused parameter 'size' [-Werror,-Wunused-parameter]
>    39 |   virtual Buffer make_buffer(void* ptr, size_t size) {
>       |                                                ^
> /usr/local/include/mlx/allocator.h:42:31: error: unused parameter 'buffer' [-Werror,-Wunused-parameter]
>    42 |   virtual void release(Buffer buffer) {}
>       |                               ^
> In file included from /Users/angusyoung/dev/vllm.cpp/src/vt/metal/metal_mlx_provider.mm:60:
> In file included from /usr/local/include/mlx/array.h:12:
> In file included from /usr/local/include/mlx/dtype.h:9:
> In file included from /usr/local/include/mlx/types/complex.h:5:
> In file included from /usr/local/include/mlx/types/half_types.h:32:
> /usr/local/include/mlx/types/bf16.h:29:3: error: definition of implicit copy assignment operator for '_MLX_BFloat16' is deprecated
>       because it has a user-declared copy constructor [-Werror,-Wdeprecated-copy]
>    29 |   _MLX_BFloat16(_MLX_BFloat16 const&) = default;
>       |   ^
> /usr/local/include/mlx/types/bf16.h:38:19: note: in implicit copy assignment operator for 'mlx::core::_MLX_BFloat16' first required
>       here
>    38 |     return (*this = _MLX_BFloat16(x));
>       |                   ^
> 4 errors generated.
> make[2]: *** [CMakeFiles/vllm.dir/src/vt/metal/metal_mlx_provider.mm.o] Error 1
> make[2]: *** Waiting for unfinished jobs....
> make[1]: *** [CMakeFiles/vllm.dir/all] Error 2
> make: *** [all] Error 2
> `
>
> # Analysis
>
> The failure appears to be caused by MLX headers being compiled under the same warning policy as vllm.cpp itself.
>
> In particular, -Werror turns warnings originating from third-party MLX headers into fatal compilation errors:
>
> -Werror,-Wunused-parameter
> -Werror,-Wdeprecated-copy
>
> The warnings originate from:
>
> /usr/local/include/mlx/allocator.h
> /usr/local/include/mlx/types/bf16.h
>
> while compiling:
>
> src/vt/metal/metal_mlx_provider.mm
>
> Since the issue is reproducible with MLX 0.30.6, 0.31.2 and 0.32.0, it does not appear to be a regression specific to the latest MLX release.
>
> Likewise, it is reproducible on both macOS 26.6.0 and 26.6.1.
>
> Environment
>
> * macOS: 26.6.1, also reproduced on 26.6.0
> * Architecture: Apple Silicon
> * Compiler: Apple Clang
> * Backend: Metal / MLX
> * MLX versions tested: 0.32.0, 0.31.2, 0.30.6
> * MLX headers: /usr/local/include/mlx
> * Failing translation unit: src/vt/metal/metal_mlx_provider.mm
>
> Thanks

## Resolution

GitHub's timeline records closing commit `785e29783561c5ca852c7973df0a9139d3141cb5` on 2026-08-09. The commit keeps MLX dependency header warnings out of this project's `-Werror` build.

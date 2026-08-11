{
  description = "Local development shells for vllm.cpp on NixOS";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      systems = [ "x86_64-linux" ];
      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system:
          f (import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
          }));
    in {
      devShells = forAllSystems (pkgs:
        let
          cuda = pkgs.cudaPackages;
          commonPackages = with pkgs; [
            bashInteractive
            cmake
            git
            ninja
            pkg-config
            python312
            which
          ];
          cudaPackages = [
            cuda.cuda_cccl
            cuda.cuda_cudart
            cuda.cuda_nvcc
            cuda.cuda_nvtx
            cuda.libcublas
            cuda.libcurand
          ];
          cudaRuntimePackages = cudaPackages ++ [
            pkgs.stdenv.cc.cc.lib
            pkgs.gcc14.cc.lib
            pkgs.zlib
          ];
          cudaLibPath = pkgs.lib.makeLibraryPath cudaRuntimePackages;
          cudaIncludePath = pkgs.lib.concatStringsSep ":" [
            (pkgs.lib.makeSearchPathOutput "dev" "include" cudaPackages)
            (pkgs.lib.makeSearchPathOutput "include" "include" cudaPackages)
          ];
          # Nsight's CLI is self-contained. Drop nixpkgs' optional UCX runtime:
          # that build enables DOCA GDA without providing its headers here, and
          # CUDA tracing does not use UCX.
          nsightSystemsCli = cuda.nsight_systems.overrideAttrs (old: {
            buildInputs = builtins.filter
              (dep: (dep.pname or "") != "ucx") old.buildInputs;
          });

          rocm = pkgs.rocmPackages;
          # nixpkgs ships each ROCm component as its own store path instead of
          # one /opt/rocm-shaped prefix, but CMake's HIP language support
          # (CMakeDetermineHIPCompiler.cmake) hard-requires a SINGLE root
          # containing lib/cmake/hip-lang/hip-lang-config.cmake. clr's own
          # output already has that (plus hip/hip_runtime.h, hipcc,
          # libamdhip64.so), so it alone can serve as ROCM_PATH. hipBLAS/
          # hipBLASLt/hipblas-common are separate outputs the project's
          # CMakeLists finds via ${ROCM_PATH}/include and ${ROCM_PATH}/lib,
          # so those three are merged into a small writable overlay at shell
          # entry (idempotent — skipped if already populated). This is the
          # one Nix-specific step; a standard /opt/rocm install needs none of
          # it.
          rocmOverlayInputs =
            [ rocm.hipblas rocm.hipblaslt rocm.hipblas-common ];
        in {
          default = pkgs.mkShell {
            packages = commonPackages ++ [ pkgs.gcc ];
            shellHook = ''
              echo "vllm.cpp CPU dev shell"
              echo "  cmake -S . -B build-nix-cpu -G Ninja -DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo"
              echo "  cmake --build build-nix-cpu -j\''${JOBS:-4}"
            '';
          };

          cuda = pkgs.mkShell {
            packages = commonPackages ++ [
              pkgs.bpftrace
              pkgs.gcc14
              pkgs.gdb
              nsightSystemsCli
            ] ++ cudaPackages;

            CUDA_PATH = "${cuda.cuda_nvcc}";
            CUDAToolkit_ROOT = "${cuda.cuda_nvcc}";
            CMAKE_CUDA_COMPILER = "${cuda.cuda_nvcc}/bin/nvcc";
            CMAKE_CUDA_HOST_COMPILER = "${pkgs.gcc14}/bin/g++";
            CUDAHOSTCXX = "${pkgs.gcc14}/bin/g++";
            # Triton's NixOS-safe driver discovery path; avoids relying on
            # /sbin/ldconfig, which is not present on this host.
            TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";

            # The live NVIDIA driver must precede toolkit stubs. This is also
            # the lookup order Triton and PyTorch use for their CUDA probe.
            LD_LIBRARY_PATH = "/run/opengl-driver/lib:${cudaLibPath}";
            LIBRARY_PATH = "/run/opengl-driver/lib:${cudaLibPath}";
            CPATH = "${cudaIncludePath}";
            C_INCLUDE_PATH = "${cudaIncludePath}";
            CPLUS_INCLUDE_PATH = "${cudaIncludePath}";

            shellHook = ''
              export CMAKE_CUDA_COMPILER="${cuda.cuda_nvcc}/bin/nvcc"
              export CMAKE_CUDA_HOST_COMPILER="${pkgs.gcc14}/bin/g++"
              export TRITON_LIBCUDA_PATH="/run/opengl-driver/lib"
              echo "vllm.cpp CUDA dev shell"
              echo "  nvidia-smi"
              echo "  cmake -S . -B build-nix-cuda -G Ninja -DVLLM_CPP_CUDA=ON -DCMAKE_CUDA_COMPILER=$CMAKE_CUDA_COMPILER -DCMAKE_CUDA_HOST_COMPILER=$CMAKE_CUDA_HOST_COMPILER -DVLLM_CPP_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=RelWithDebInfo"
              echo "For RTX 50-series Blackwell, use native first; if CMake cannot detect it, try 120a."
            '';
          };

          rocm-shell = pkgs.mkShell {
            packages = commonPackages ++ [
              pkgs.gcc
              rocm.clr
              rocm.hipblas
              rocm.hipblaslt
              rocm.hipblas-common
              rocm.rocminfo
            ];

            shellHook = ''
              export ROCM_OVERLAY="''${XDG_CACHE_HOME:-$HOME/.cache}/vllm-cpp-rocm-overlay"
              if [ ! -e "$ROCM_OVERLAY/.complete" ]; then
                echo "vllm.cpp ROCm dev shell: assembling one-time hipBLAS/hipBLASLt overlay at $ROCM_OVERLAY ..."
                mkdir -p "$ROCM_OVERLAY/include" "$ROCM_OVERLAY/lib"
                for pkg in ${
                  pkgs.lib.concatStringsSep " " rocmOverlayInputs
                }; do
                  [ -d "$pkg/include" ] && cp -rns "$pkg"/include/* "$ROCM_OVERLAY/include/" 2>/dev/null
                  [ -d "$pkg/lib" ] && cp -rns "$pkg"/lib/* "$ROCM_OVERLAY/lib/" 2>/dev/null
                done
                chmod -R u+w "$ROCM_OVERLAY"
                touch "$ROCM_OVERLAY/.complete"
              fi
              # clr is the ROCM_PATH: it alone has lib/cmake/hip-lang, which
              # CMake's enable_language(HIP) requires at a single fixed root.
              export ROCM_PATH="${rocm.clr}"
              # hipBLAS/hipBLASLt headers+libs, plus the hipblas-common header
              # hipblas.h includes but does not itself ship (see the
              # rocmOverlayInputs comment above) — clang's --rocm-path probe
              # does not reach it, so it must ride CPATH explicitly.
              export CPATH="$ROCM_OVERLAY/include:''${CPATH:-}"
              export LIBRARY_PATH="$ROCM_OVERLAY/lib:''${LIBRARY_PATH:-}"
              export LD_LIBRARY_PATH="$ROCM_OVERLAY/lib:${rocm.clr}/lib:''${LD_LIBRARY_PATH:-}"
              export CMAKE_PREFIX_PATH="$ROCM_OVERLAY:''${CMAKE_PREFIX_PATH:-}"
              echo "vllm.cpp ROCm dev shell"
              echo "  rocminfo | grep -A2 Marketing   # confirm the gfx target"
              echo "  cmake -S . -B build-hip -G Ninja -DVLLM_CPP_HIP=ON -DVLLM_CPP_HIP_ARCHITECTURES=gfx1200 -DROCM_PATH=\$ROCM_PATH -DCMAKE_BUILD_TYPE=Release"
              echo "  cmake --build build-hip -j\''${JOBS:-4}"
              echo "  ctest --test-dir build-hip -R 'rocm|cross_device' --output-on-failure"
              echo "Swap gfx1200 for your board's arch (rocminfo Name: field)."
            '';
          };
        });
    };
}

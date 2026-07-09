{
  description = "Local development shells for vllm.cpp on NixOS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

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
    in
    {
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
            pkgs.gcc14.cc.lib
          ];
          cudaLibPath = pkgs.lib.makeLibraryPath cudaRuntimePackages;
          cudaIncludePath = pkgs.lib.concatStringsSep ":" [
            (pkgs.lib.makeSearchPathOutput "dev" "include" cudaPackages)
            (pkgs.lib.makeSearchPathOutput "include" "include" cudaPackages)
          ];
        in
        {
          default = pkgs.mkShell {
            packages = commonPackages ++ [ pkgs.gcc ];
            shellHook = ''
              echo "vllm.cpp CPU dev shell"
              echo "  cmake -S . -B build-nix-cpu -G Ninja -DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo"
              echo "  cmake --build build-nix-cpu -j\''${JOBS:-4}"
            '';
          };

          cuda = pkgs.mkShell {
            packages = commonPackages ++ [ pkgs.gcc14 ] ++ cudaPackages;

            CUDA_PATH = "${cuda.cuda_nvcc}";
            CUDAToolkit_ROOT = "${cuda.cuda_nvcc}";
            CMAKE_CUDA_COMPILER = "${cuda.cuda_nvcc}/bin/nvcc";
            CMAKE_CUDA_HOST_COMPILER = "${pkgs.gcc14}/bin/g++";
            CUDAHOSTCXX = "${pkgs.gcc14}/bin/g++";

            # On NixOS the proprietary driver libraries come from the host driver
            # profile, not the CUDA toolkit package. Keep that path in the shell
            # so nvidia-smi/CUDA runtime probing can find libcuda and NVML.
            LD_LIBRARY_PATH = "${cudaLibPath}:/run/opengl-driver/lib";
            LIBRARY_PATH = "${cudaLibPath}:/run/opengl-driver/lib";
            CPATH = "${cudaIncludePath}";
            C_INCLUDE_PATH = "${cudaIncludePath}";
            CPLUS_INCLUDE_PATH = "${cudaIncludePath}";

            shellHook = ''
              export LD_LIBRARY_PATH="${cudaLibPath}:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              export LIBRARY_PATH="${cudaLibPath}:/run/opengl-driver/lib''${LIBRARY_PATH:+:$LIBRARY_PATH}"
              export CPATH="${cudaIncludePath}''${CPATH:+:$CPATH}"
              export C_INCLUDE_PATH="${cudaIncludePath}''${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
              export CPLUS_INCLUDE_PATH="${cudaIncludePath}''${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
              export CMAKE_CUDA_COMPILER="${cuda.cuda_nvcc}/bin/nvcc"
              export CMAKE_CUDA_HOST_COMPILER="${pkgs.gcc14}/bin/g++"
              echo "vllm.cpp CUDA dev shell"
              echo "  nvidia-smi"
              echo "  cmake -S . -B build-nix-cuda -G Ninja -DVLLM_CPP_CUDA=ON -DCMAKE_CUDA_COMPILER=$CMAKE_CUDA_COMPILER -DCMAKE_CUDA_HOST_COMPILER=$CMAKE_CUDA_HOST_COMPILER -DVLLM_CPP_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=RelWithDebInfo"
              echo "For RTX 50-series Blackwell, use native first; if CMake cannot detect it, try 120a."
            '';
          };
        });
    };
}

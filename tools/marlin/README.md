# Marlin NVFP4 W4A16 MoE — kernel-lift correctness harness

Proves the **vendored** Marlin MoE kernel (`src/vt/cuda/marlin/`, a torch-free
1:1 lift of vLLM's `moe/marlin_moe_wna16` @ `e24d1b24`) is numerically identical
to vLLM's own `moe_wna16_marlin_gemm`, isolating the kernel from the load-time
repack. Run on GB10 (sm_121a, CUDA 13).

1. **Golden** (oracle venv) — build a small NVFP4 MoE GEMM, repack via vLLM's
   exact recipe (`gptq_marlin_repack` + `marlin_permute_scales` +
   `nvfp4_marlin_process_scales`/`_global_scale`, mirroring
   `prepare_nvfp4_moe_layer_for_marlin`), run `moe_wna16_marlin_gemm`, dump every
   input + the golden C to `/tmp/moe_dump/`:

   ```
   ~/venvs/vllm-oracle/bin/python tools/marlin/moe_golden.py
   ```

2. **Ours** — feed the *identical* dumped inputs into the vendored `marlin_mm`
   and dump `C_ours.bin`:

   ```
   export PATH=/usr/local/cuda/bin:$PATH
   R=src/vt/cuda/marlin; F="-std=c++17 -arch=sm_121a -O2 --expt-relaxed-constexpr -static-global-template-stub=false -I$R"
   nvcc $F -c $R/libtorch_stable/moe/marlin_moe_wna16/marlin_mm_moe.cu -o /tmp/mm.o
   nvcc $F -c $R/libtorch_stable/moe/marlin_moe_wna16/sm80_kernel_bfloat16_fe2m1f_bfloat16.cu -o /tmp/k.o
   nvcc $F tools/marlin/harness_main.cu /tmp/mm.o /tmp/k.o -o /tmp/harness && /tmp/harness
   ```

## Verified result (2026-07-05, GB10 sm_121a, CUDA 13.0)

```
cosine=1.00000238  rel_err=0.000e+00  bitexact_frac=1.0000  max_abs_diff=0.000e+00
VERDICT: BIT-EXACT
```

The vendored kernel reproduces vLLM's `moe_wna16_marlin_gemm` **bit-for-bit**.
The remaining integration work (a C++ port of the load-time repack, the
`moe_align_block_size` port, and the 35B forward wiring) is thereby de-risked to
"lay out the same repacked buffers" — the kernel itself is proven.

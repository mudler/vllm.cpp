# The CUDA dequantizing gather — wave KGATHER of `MODEL-MM-QWEN4-EXP`

Row: `MODEL-MM-QWEN4-EXP` · sibling spec:
[`qwen4-exp-flash-next.md`](qwen4-exp-flash-next.md) · the debt this discharges
was recorded by [#2083](https://github.com/mudler/vllm.cpp/issues/2083) under
that row's `## Owed

- **sm_121a.** The device gate ran on sm_110. `dgx:gpu0` job `38a9b799` is
  queued and will confirm the target architecture and close M2's device
  measurement. No sm_121a claim is made here.
- **M2 on device.** Measured on the host harness (457,498 of 968,199 elements,
  max|diff| 4.0295e+06) and owed on a GPU; the mutation is fixed and staged.
- **METAL, VULKAN, ROCM, TENSTORRENT.** Each gather kernel asserts a float table
  by name and none registers `kEmbeddingQuant`, so each answers false to the
  residency gate and keeps expand-bf16. The `qwen4_exp` loader still refuses them
  by name. Their arms are owed per device.
- **Performance.** This wave gates CORRECTNESS only. The decoders read byte-wise
  for alignment safety and one thread decodes a whole block. No throughput number
  is claimed, measured or implied, and no benchmark ID moves.
- **The end-to-end `qwen4_exp` forward on CUDA.** The LOAD-side blocker is gone;
  the forward is not. `ModelRegistry::Forward` is all-or-nothing and several
  `qwen4_exp` ops have no CUDA arm, so this produces no token on a GPU.

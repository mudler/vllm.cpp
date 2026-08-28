# QUANT-EXL3 — EXL3 as a quantization scheme every architecture can reach

Row: `QUANT-EXL3`
Issues: [#2181](https://github.com/mudler/vllm.cpp/issues/2181) (primary)
Base SHA: `bca11d03d`
Matrix: [`.agents/quantization-matrix.md`](../quantization-matrix.md)

Upstream pin: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` — **vLLM implements
no EXL3** (`layers/quantization/` registers no exl3/exllamav3/trellis method at
the pin), so the format is mirrored from the registered secondary oracle
[`exllamav3`](../oracles/exllamav3.md) @ `2398c05635fbbad01a0a51dce63c85c6c8a8450e`
(tag `v1.4.3`, MIT). **The SEAM is vLLM's**, and that half is not a fallback
case: `layers/quantization/base_config.py:87-180` (`QuantizationConfig` +
`get_quant_method(layer, prefix)`) and `layers/linear.py:141-181`
(`LinearMethodBase.create_weights` + `apply`) define where a scheme plugs in,
and this row mirrors them. Where vLLM defines structure, vLLM wins; exllamav3
supplies only the trellis format and its kernels.

## Now

`SPIKE`. Nothing has landed. The row exists because EXL3 is implemented in this
tree as a DeepSeek-V4-private arm and no other checkpoint can reach it.

## The gap, measured

`grep -rl Exl3 src/vllm include/vllm` returns three files, all DeepSeek-V4
(`models/deepseek_v4.cpp`, `models/deepseek_v4_weights.cpp`,
`include/vllm/model_executor/models/deepseek_v4.h`).
`.agents/quantization-matrix.md` carries no EXL3 row while registering 20+ other
schemes. `IsExl3Checkpoint` (`deepseek_v4_weights.cpp:229-233`) reads the same
`quantization_config.quant_method == "exl3"` marker every EXL3 checkpoint
carries, and is consulted only from the DeepSeek-V4 loader.

**The kernels are ready and device-proven.** `vt::Exl3HadR128`, `vt::Exl3Gemm`,
`vt::Exl3MoeMlp` and `src/vt/cpu/cpu_exl3_dequant.cpp` exist and passed their
device gates on `dgx:gpu0` (GB10 `sm_121a`, driver 580.173.02, nvcc 13.0.88,
tree `525d2b991`, 2026-08-28, worker `rc-worker-4b8lj`): `had_r_128` CUDA-vs-CPU
`mismatches == 0`; `exl3_gemm` vs f64 `rel_rms 5.538e-4` (bound `1.0e-3`, worst
`0.0334` against 8·ulp `0.0625`); GEMV tier 3c `rel_rms 5.160e-4` (bound
`6.0e-3`). `cuda_exl3.cu.o` carried one `sm_121a` cubin. Everything missing is
ABOVE the kernels.

## The stock layout, measured rather than assumed

Range-read from the safetensors header of `turboderp/Llama-3.2-1B-Instruct-exl3`
revision `3.0bpw` (`f8f438c290680b15622270eff03bef23a458b1cf`) on 2026-08-28 —
373 tensors, one 1.09 GB file, header 40,368 bytes:

- HF-standard keys with EXL3 fields appended:
  `model.layers.N.self_attn.{q,k,v,o}_proj.{trellis,suh,svh}`,
  `model.layers.N.mlp.{gate,up,down}_proj.{trellis,suh,svh}`,
  `lm_head.{trellis,suh,svh}`.
- `trellis` `I16 [k/16, n/16, 16*bits]`, exactly what `Exl3ReconstructInner`
  reads: `mlp.gate_proj.trellis [128, 512, 48]` is k=2048, n=8192, bits=3.
- `suh` `F16 [k]`, `svh` `F16 [n]` — `mlp.down_proj.suh [8192]` /
  `.svh [2048]` confirms `suh` is the INPUT side and `svh` the OUTPUT side on
  every projection, which is what `Exl3DequantLinear` assumes.
- Norms and `model.embed_tokens.weight` stay `F16`, unquantized.
- **No `.rank{r}` segments.** The rank-sliced `rank-sliced-deepseek-v4-v1`
  schema `MODEL-DSV4-EXL3` W1b implements is SparkInfer's variant, not the
  format's ordinary shape; this row's reader is the simpler one.
- `quantization_config = {quant_method: "exl3", version: "0.0.0", bits: 3.0,
  calibration: {rows: 100, cols: 2048}}`.

**`bits` is PER TENSOR, and the config scalar is not it.** `lm_head.trellis` is
`[128, 8016, 96]` — 96 = 16*6, so the head is 6-bit while the body is 3-bit and
`quantization_config.bits` says `3.0`. Every consumer derives `bits` from the
`trellis` last dimension divided by 16, and the config value is used only to
cross-check the modal case. A reader that trusts the config scalar decodes the
head at the wrong width and produces garbage that no shape check catches, since
the tensor's shape is self-consistent at either reading.

**No `mcg` tensor ships in this checkpoint.** The DSV4 artifact carries a per
linear `mcg` int32 codebook marker; this one does not, and upstream's
`Linear.is_exl3_storage` requires only `{key}.trellis` with `suh|su` and
`svh|sv` (`modules/linear.py:385-389`). The codebook therefore defaults to MCG
(`cb == 1`), which is `LinearEXL3`'s own default, and a checkpoint that ships a
marker naming anything else REFUSES BY NAME rather than being decoded as MCG.

## Scope, in waves

**W1 — the scheme, the reader, and one model end to end (this spec's first
dispatch).**
- `layers/quantization/exl3.{h,cpp}`: an `Exl3LinearMethod : LinearMethodBase`
  whose `Apply` is `vt::Exl3HadR128` in, `vt::Exl3Gemm`, `had_r_128` out, and an
  `Exl3Config` recognized from `quantization_config.quant_method == "exl3"`,
  mirroring `get_quant_method`.
- A native-layout reader keyed on the three sibling tensors, beside the
  rank-sliced arm rather than replacing it.
- The shared dense container and forward (`Qwen3DenseWeights`, the Qwen3-dense
  `AttnBlock` path that `LlamaForCausalLM` reuses verbatim) gain the EXL3 arm,
  so Llama and Qwen3-dense both reach it from one change.
- E2E greedy generation from a production entry point on
  `turboderp/Llama-3.2-1B-Instruct-exl3`.

**W2 — residency.** A device-resident destination for the trellis tower.
`MODEL-DSV4-EXL3` `## Owed` already needs this for its host-residency refusal;
a 1.09 GB checkpoint makes it testable without the 100 GB artifact, and it is
the precondition for `vt::Exl3MoeMlp`'s device arm, which skips today because
`CudaBackend::DeviceMemoryIsHostAddressable()` is false by design
(`cuda_backend.cu:330-366`, #1635).

**W3 — width coverage.** The CUDA arm instantiates `bits == 3, codebook == 1`
only. A stock checkpoint's 6-bit head has no device arm. Either widen the
instantiation set the way upstream splits it
(`comp_units/exl3_comp_unit_K_cbX.cu`) or route the head to the generic CPU arm
and say so at the refusal.

**W4 — DeepSeek-V4 routes through this seam.** `MODEL-DSV4-EXL3`'s private
`Exl3Linear` becomes a caller of the shared method. Sequenced last because
#1875's blockers are its DSA composition and its residency, neither of which
this row's seam changes.

## Design

**Why a `LinearMethodBase` rather than a fourth field on the dense container.**
`include/vllm/model_executor/layers/linear.h:43` already mirrors
`LinearMethodBase`, and `base_config.h` records why it exists: scheme selection
used to be a per-model tensor-name probe with device gates scattered through
forwards, and this seam is the removal of that tangle. Adding EXL3 as another
inline branch in a model forward would rebuild exactly what that row deleted.

**Why the dense hot path stays byte-identical.** Only Gemma and dots3 consult
the seam today; the dense Qwen3/Llama forward calls `vt::MatmulBT` inline. W1
does NOT migrate that path onto method dispatch. It adds a branch taken only
when the loaded weights carry an EXL3 arm, so a bf16 checkpoint executes the
same instructions it does today. Migrating the dense path onto the seam is a
separate decision belonging to whoever owns that forward, and this spec does not
make it.

**Output dtype.** The dequant reference carries fp16-valued data in `float`
(`MODEL-DSV4-EXL3` risk 5). This row's destination is the model dtype the
checkpoint declares (`torch_dtype: bfloat16` for the Llama artifact), never f32
inherited from a reference signature. `AGENTS.md` §"Inherit vLLM defaults": a
token gate cannot see a dtype that is too wide.

## Risks

1. **The gate is the hard part, not the code.** vLLM has no EXL3 and the
   secondary oracle does not build on aarch64 (#1901), so a token-exact
   comparison against an oracle EXL3 run is unavailable on this fleet. See
   `## Gates`; the bound is chosen BEFORE the gate runs, never widened after a
   red.
2. Per-tensor `bits` (above). A config-scalar reader is silently wrong on the
   head.
3. The 6-bit head has no device arm, so W1's e2e run may be part-host. That is
   recorded as a measurement, never hidden by falling back silently.
4. `tie_word_embeddings` is FALSE in this artifact while the bf16 Llama-3.2-1B
   ties them — the EXL3 repo ships a real quantized `lm_head`. The loader must
   not apply the bf16 path's `skip_prefixes(["lm_head."])` to an EXL3 load.

## Tests

Red first, in this order:

1. `tests/vllm/layers/test_exl3_linear_method.cpp` — the method's `Apply`
   against `vt::Exl3DequantLinear` + a dense GEMM on the same fixture, and
   `bits` resolved from the tensor rather than the config (a fixture whose
   config says 3 and whose tensor says 6 must decode at 6).
2. `tests/vllm/models/test_exl3_native_loader.cpp` — a hermetic native-layout
   checkpoint (no rank segments) loads into the dense container; a missing
   `svh` REFUSES BY NAME; a non-MCG marker REFUSES BY NAME.
3. Reachability: the e2e case drives `LoadLlamaForCausalLMWeights` and
   `ModelRegistry::Forward`, never a hand-built struct. Deleting the production
   call site must go RED — the #1923 failure in its exact shape is what that
   guards against.

## Gates

| Gate | Owner |
|---|---|
| W1: method vs the W1a dequant reference on one fixture, within a stated bound | implementer |
| W1: per-tensor `bits` — a 6-bit tensor under a 3-bit config decodes at 6; mutating the reader to trust the config goes RED | implementer |
| W1: the native reader refuses a missing sibling and a foreign codebook BY NAME | implementer |
| W1: a real EXL3 Llama checkpoint GENERATES from a production entry point | operator |
| W1: deleting the production call site goes RED | implementer/reviewer |
| W1: a bf16 Llama load is BYTE-IDENTICAL to its pre-change logits | implementer |
| W2: the trellis tower is device-resident; the MoE device arm stops skipping | implementer |
| W3: the 6-bit head has a device arm, or refuses by name | implementer |

**What the correctness gate CAN bind, since an oracle token match cannot.**
Stated here before code, per risk 1:

- **Reference-model agreement.** The same prompts through the BF16
  `Llama-3.2-1B-Instruct` we already gate token-exact 16/16 vs vLLM, and through
  the EXL3 3.0bpw quant of that same model. Quantization changes tokens, so this
  is NOT a token gate: it is a bounded divergence gate on the logit
  distribution, with the bound stated before the run.
- **Self-consistency.** Our EXL3 dequant-to-dense reconstruction vs our EXL3
  native compute on the same weights — a real gate on the compute path, and the
  one place a token-exact bound IS available.
- **Coherence.** A greedy continuation that is readable English is a weak gate
  and is recorded as weak, never as a pass.
- The oracle gate is OWED and blocked on #1901; when exllamav3 builds on
  aarch64 it becomes the token oracle and this section is replaced, not
  supplemented.

## Owed

- The oracle token gate (#1901).
- W4: `MODEL-DSV4-EXL3`'s private `Exl3Linear` still exists after W1.
- `docs/FEATURES.md` and `docs/USAGE.md` rows, including the checkpoint's file
  name, size, repo and REVISION — a bpw branch name, since this repo publishes
  one revision per bit width and `main` carries no weights at all.

## Stop conditions

- The native reader disagrees with the W1a reference on a fixture → stop and
  re-derive from `exl3.py:227-237`; never tune a constant to green.
- A bf16 load stops being byte-identical → the change is not additive; stop.

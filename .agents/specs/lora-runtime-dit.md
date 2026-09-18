# Runtime prompt-activated LoRA for diffusion DiT models

**Row:** `ROAD-V1-LORA-RUNTIME` (roadmap portfolio).
**Issue:** `ISSUE-LOCAL-01M2RECPA70PBQN14E1VCMHJ5T`.
**Upstream:** vLLM-Omni `DiffusionLoRAManager`
(`vllm_omni/diffusion/lora/manager.py`, `vllm_omni/diffusion/lora/layers/base_linear.py`);
LocalAI stable-diffusion.cpp `parse_loras_from_prompt`
(`backend/go/stablediffusion-ggml/cpp/gosd.cpp:174-330`).
**Sibling:** `dit-lora-generic.md` (DONE) — load-time fusion seam `dit_lora.{h,cpp}`.
This row adds per-request activation on top of that seam without baking into base weights.

## Now

**SPEC.** Row claimed, spec committed, no implementation yet.

## Scope

Add prompt-activated runtime LoRA to diffusion DiT models (MiniMax-H3, LTX2.5) so
different LoRAs apply to different generation requests without reloading the model.
Users write `<lora:name:strength>` tags in the prompt; the engine strips the tags,
resolves each name to a safetensors adapter file, loads the A/B factors, and applies
the delta `(x @ A^T) @ B^T * strength` at each linear projection during the denoise
loop. The base weights are never modified.

This mirrors vLLM-Omni's `DiffusionLoRAManager`, which applies a single-active-adapter
per-layer additive delta at forward time using plain matmul (not punica). It also
mirrors LocalAI stable-diffusion.cpp's `parse_loras_from_prompt`, which extracts
`<lora:name:strength>` tags from the prompt and returns a cleaned prompt.

Load-time fusion (ROAD-V1-DIT-LORA, merged) and runtime activation coexist:
config-defined LoRAs fuse into base weights at load; prompt-activated LoRAs apply
per-request at forward time.

## 1. What exists

### Load-time LoRA fusion (DONE, merged)

`dit_lora.{h,cpp}` provides a model-agnostic seam:

- `DitLoraSpec` — path + strength (float).
- `DitLoraAdapter` — opens a safetensors adapter, resolves keys onto the contract
  via `DitLoraContractName` (strips ComfyUI prefixes, rewrites `.lora_{A,B}.weight`
  to `.weight`).
- `DitLoraFactorPair` — A/B factors in bf16.
- `DitFuseLoraIntoTensor` — bakes `B*strength @ A` into a weight buffer (load-time).
- `IsDitLoraExtra` / `IsDitLoraIndexedExtra` — filter LoRA extras in the extras map.
- `ResolveDitLoraSpecs` — parses indexed `lora_path`/`lora_strength` from extras.

H3 wires this in `minimax_h3_video.cpp:380` (`ResolveDitLoraSpecs`) and fuses during
`StreamMiniMaxH3DitToDeviceBf16`. LTX2 wires it in `ltx2_video.cpp:692`
(`ResolveLoraSpecs`) and fuses during `MaterializeDitTensor`.

### Linear projection seam

All linear projections funnel through one function per model:

- **H3 device:** `LinearDev` (`minimax_h3_device.cpp:158-179`) → `vt::MatmulBT`
  (or `dense_nvfp4::MatmulNvfp4W4A16D` for fp4).
- **H3 CPU:** `Linear` (`minimax_h3.cpp:95-112`) → `vt::MatmulBT` (f32 only).
- **LTX2 device:** `LinearDev` (`ltx2_device.cpp:190-200`) → `vt::MatmulBT`.
- **LTX2 CPU:** inline `vt::MatmulBT` calls (no wrapper).

None of these take LoRA-related parameters. A runtime delta must be added to the
output after the base GEMM.

### Forward and denoise path

- **H3 device:** `MiniMaxH3DitForwardDevice` (`minimax_h3_device.cpp:558-974`) is
  called per sigma step by `MiniMaxH3DenoiseLoop` (`minimax_h3.cpp:760`). Weights
  are staged once; the loop calls forward repeatedly.
- **H3 CPU:** `MiniMaxH3DitForward` (`minimax_h3.cpp:526`).
- **LTX2 device:** `Ltx2DitForwardDevice` calls `Ltx2TransformerBlockForward` per
  block, which calls `LinearDev` for QKV, output proj, MLP, adaln, patchify.
- **LTX2 CPU:** `Ltx2DitForward` (`ltx2_dit.cpp:762`).

### Per-request parameter transport

- `VideoGenParams` (`video_engine.h:75`) has `prompt` and `extras` (a string map).
  LTX2 uses `extras` extensively for per-generation knobs.
- `MiniMaxH3VideoGenParams` (`minimax_h3_video.h:102`) has `prompt` but NO `extras`
  field. `MiniMaxH3VideoGenParamsFromGeneric` (`minimax_h3_video.cpp:856`) throws on
  unknown per-generation extras: "this family defines none."
- `MiniMaxH3VideoModelParams` (`minimax_h3_video.h:73`) has `extras` for load-time
  knobs (including LoRA paths).

### LocalAI prompt-activated LoRA pattern

`parse_loras_from_prompt` (`gosd.cpp:174-330`):
- Regex `<lora:([^:>]+):([^>]+)>` extracts name and strength.
- `discover_lora_files` (`gosd.cpp:110-158`) scans `lora_dir` for safetensors/ckpt/pt/gguf.
- Name resolution: exact filename in `lora_dir`, absolute path, case-insensitive match,
  relative path under `lora_dir`, extension probing.
- Multipliers accumulate for duplicate LoRAs (same normalized path sums strengths).
- Returns `{vector<sd_lora_t>, cleaned_prompt}` with tags stripped, whitespace collapsed.

### vLLM-Omni runtime LoRA pattern

`DiffusionLoRAManager` (`manager.py:36`) and `DiffusionBaseLinearLayerWithLoRA`
(`base_linear.py:10`):
- Single active adapter at a time. Per-layer additive delta at forward time.
- `apply()` (`base_linear.py:69-134`): `output = base_linear(x)`;
  `delta = (x_flat @ A.T) @ B.T`; `y_flat[:, offset:offset+slice] += delta`.
  Plain `torch.matmul`, NOT punica.
- Packed projections (QKV) handled via `output_slices` — per-slice delta.
- Weights loaded to CPU, copied to pre-allocated GPU stacked buffers on activation.
- Two-layer scaling: internal `alpha/rank` baked into `lora_b` at load
  (`manager.py:320`, `optimize()`); external `lora_scale` multiplied at activation
  (`_activate_adapter:563`, `lora_b * scale`).
- Layer wrapping: `_replace_layers_with_lora` wraps base `LinearBase` — original
  becomes `.base_layer`.

## 2. Design

### 2.1 Prompt tag parsing

New function `DitParseLoraTags`:

```cpp
struct DitRuntimeLoraSpec {
  std::string path;     // resolved file path
  float strength = 1.0; // from the tag
};

struct DitParseLoraResult {
  std::vector<DitRuntimeLoraSpec> loras;
  std::string clean_prompt; // tags stripped, whitespace collapsed
};

DitParseLoraResult DitParseLoraTags(const std::string& prompt,
                                     const std::string& lora_dir);
```

- Regex `<lora:([^:>]+):([^>]+)>` — same as LocalAI sd.cpp.
- Name resolution mirrors `discover_lora_files` + the sd.cpp fallback chain:
  exact filename in `lora_dir`, absolute path, case-insensitive, relative under
  `lora_dir`, extension probing (`.safetensors`).
- Duplicate LoRAs (same normalized path): strengths sum. Mirrors sd.cpp accumulation.
- Returns the cleaned prompt (tags removed, whitespace collapsed) and the specs.

Lives in `dit_lora.{h,cpp}` (shared seam). Model-agnostic.

### 2.2 Runtime LoRA state

New struct `DitRuntimeLoraState` holds loaded adapter factors, keyed by contract
target name, already on the compute device:

```cpp
struct DitRuntimeLoraLayer {
  Tensor lora_a;  // [rank, in_features] on device
  Tensor lora_b;  // [out_features, rank] on device
  float strength = 1.0;
};

struct DitRuntimeLoraState {
  bool empty() const;
  const DitRuntimeLoraLayer* Find(const std::string& target) const;
  // loaded layers, keyed by contract target name
  std::map<std::string, DitRuntimeLoraLayer> layers;
};
```

Loading: inside `Generate`, before the prompt is encoded:
1. `DitParseLoraTags(prompt, lora_dir)` → specs + clean prompt.
2. For each spec, `DitLoraAdapter::Open` opens the safetensors and resolves keys
   via `DitLoraContractName` (same prefix set as load-time fusion).
3. For each target the adapter binds, extract the A/B factors, fold `alpha/rank`
   into B if the adapter carries alpha metadata (mirrors vLLM-Omni `optimize()`),
   multiply by strength, and upload to the device.
4. Build the `DitRuntimeLoraState` map.

The alpha/rank fold is computed once at load: if metadata has `lora_alpha`,
`effective_b = b * (alpha / rank) * strength`; otherwise `effective_b = b * strength`.
This keeps the per-forward delta a simple `(x @ A^T) @ B_eff^T` with no scaling.

### 2.3 Delta application at the Linear seam

The delta is applied to the OUTPUT of the base linear, after `vt::MatmulBT`:

```
base_out = x @ W^T           // the existing LinearDev/Linear
delta = (x @ A^T) @ B_eff^T  // shrink @ expand
out = base_out + delta        // element-wise add
```

This is the vLLM-Omni pattern (`base_linear.py:130-131`). The base weights are
never touched, so no copy-on-write is needed.

Add an optional `const DitRuntimeLoraLayer*` parameter to `LinearDev` and `Linear`:

```cpp
void LinearDev(Dev d, const Tensor& in, int64_t rows, int64_t in_features,
                const Tensor& weight, const Tensor* bias, Tensor& out,
                const Nvfp4Weight* fp4 = nullptr,
                const DitRuntimeLoraLayer* lora = nullptr);
```

When `lora != nullptr`:
1. After the base GEMM (and bias add), compute `delta = (a @ lora_a^T) @ lora_b^T`
   via two `vt::MatmulBT` calls (a is already reshaped to `[rows, in_features]`).
2. `vt::Add(d.q, o, o, delta_tensor)` — in-place add to the output.

The two matmuls use the stream dtype (bf16 on device, f32 on CPU). The
intermediate `[rows, rank]` buffer is a temporary `DBuf` (device) or
`std::vector<float>` (CPU).

For LTX2, `LinearDev` (`ltx2_device.cpp:190`) gets the same optional parameter.
The inline `vt::MatmulBT` calls in `ltx2_dit.cpp` (CPU path) get a helper wrapper
or inline delta, matching the `Linear()` approach.

### 2.4 Threading through the forward path

The `DitRuntimeLoraState` is threaded from `Generate` through the denoise loop
into the forward function:

- **H3:** `MiniMaxH3DenoiseLoop` → `MiniMaxH3DitForwardDevice` / `MiniMaxH3DitForward`.
  Each block's `AttentionDev`, `MlpDev`, `AdalnProjectDev` call `LinearDev` with
  the layer's target name, looked up in the state:
  `state.Find("blocks.0.attn.qkv_proj")`.
- **LTX2:** `Ltx2DitForwardDevice` → per-block forward → `LinearDev` calls with
  target lookup: `state.Find("to_q")`, `state.Find("to_out")`, etc.

The target name is the same contract name used by load-time fusion
(`DitLoraContractName` output), so the adapter's resolved keys match the
forward path's weight names directly.

### 2.5 lora_dir transport

A `lora_dir` path is needed so the engine can resolve prompt tag names to files.
It rides as a **model-load extra** (not per-generation), mirroring sd.cpp's `lora_dir`:

- `MiniMaxH3VideoModelParams.extras["lora_dir"]` — the H3 engine reads it at load
  and stores it for use during `Generate`.
- `VideoModelParams.extras["lora_dir"]` — the generic seam; LTX2 reads it at load.
- C ABI: `vllm_video_model_params.extra_keys`/`extra_values` carry `lora_dir`.
- CLI: `--lora-dir <path>` flag on `minimax_h3_gen` and `ltx2_gen`.
- Server: `--video-extra lora_dir=<path>` (model params).
- LocalAI: `lora_dir` in the model config YAML; the backend passes it as a model
  extra. The prompt flows through with tags intact.

### 2.6 H3 per-generation extras

`MiniMaxH3VideoGenParamsFromGeneric` (`minimax_h3_video.cpp:856`) throws on any
unknown per-generation extra. The runtime LoRA path does NOT need a per-generation
extra: the LoRA specs come from parsing the prompt, not from the extras map. The
`lora_dir` is a model-load extra. So no change to the H3 per-generation extras
filter is needed for the basic prompt-tag path.

If a future caller wants to pass runtime LoRA specs via extras instead of prompt
tags, that would require allowing `lora_runtime_*` keys through the filter. That
is out of scope for this row.

### 2.7 Packed projections

H3's QKV is a packed `qkv_proj` weight `[3*dim, dim]`. A LoRA adapter targeting
`qkv_proj` has A `[rank, dim]` and B `[3*dim, rank]`. The delta
`(x @ A^T) @ B^T` is `[rows, 3*dim]`, matching the packed output. No per-slice
split is needed — the delta covers the full packed output in one matmul pair.

This matches vLLM-Omni's fallback when `output_slices` is None
(`base_linear.py:102-104`): infer the slice from the B tensor shape.

If an adapter targets only `q_proj` (not the packed name), the contract name
rewrite must map it to the packed name's sub-slice. This is the same resolution
load-time fusion already handles via `DitLoraContractName`. If the adapter key
does not match any contract name, the load refuses with the existing "does not
bind" error.

### 2.8 Surfaces

| Surface | Change |
|---|---|
| C ABI | `lora_dir` in `vllm_video_model_params.extras`. Prompt tags in `vllm_video_params.prompt`. No new functions. |
| CLI (`minimax_h3_gen`, `ltx2_gen`) | `--lora-dir <path>` flag. Prompt contains tags. |
| OpenAI server | `--video-extra lora_dir=<path>` (model). Prompt in request body. |
| video_studio | `--lora-dir <path>` flag. |
| LocalAI | `lora_dir` in model config YAML. Backend passes it as model extra. Prompt flows through. Existing `buildLoraExtras` (load-time) stays for config-defined LoRAs. |

### 2.9 Interaction with load-time fusion

Both paths coexist:
1. Load-time: config-defined `lora_path`/`lora_strength` extras → fused into base
   weights at load. Always active.
2. Runtime: prompt-tag `<lora:name:strength>` → delta applied per-forward. Per-request.

The runtime delta is computed against the FUSED base weights (load-time fusion
already modified them). This is correct: the runtime LoRA is an additional delta on
top of whatever the base weights already are.

## 3. Phases

| Phase | What | Files |
|---|---|---|
| 1 | `DitParseLoraTags` + `DitRuntimeLoraState` + `DitRuntimeLoraLayer` in `dit_lora.{h,cpp}`. Name resolution. Unit tests for parsing and resolution. | `dit_lora.{h,cpp}`, new test |
| 2 | Add `const DitRuntimeLoraLayer*` to `LinearDev`/`Linear`. Delta computation via two `vt::MatmulBT` + `vt::Add`. | `minimax_h3_device.cpp`, `minimax_h3.cpp`, `ltx2_device.cpp`, `ltx2_dit.cpp` |
| 3 | Thread `DitRuntimeLoraState` through H3 forward: `Generate` → `DenoiseLoop` → `DitForwardDevice`. Load adapters, build state, look up per layer. | `minimax_h3_video.{h,cpp}`, `minimax_h3_device.{h,cpp}`, `minimax_h3.cpp` |
| 4 | Thread `DitRuntimeLoraState` through LTX2 forward: `Generate` → `DitForwardDevice`. | `ltx2_video.cpp`, `ltx2_device.cpp` |
| 5 | `lora_dir` model extra + CLI `--lora-dir` + server `--video-extra`. | `include/vllm.h`, `examples/minimax_h3_gen/main.cpp`, `examples/ltx2_gen/main.cpp`, `examples/video_studio/main.cpp`, `server_main.cpp` |
| 6 | LocalAI: `lora_dir` in model config, pass as model extra. Prompt tags flow through. | LocalAI `video.go` |
| 7 | Build + E2E on `dgx:gpu0` via `rc run`. | — |

## 4. Upstream anchors

- vLLM-Omni `vllm_omni/diffusion/lora/layers/base_linear.py:69-134` — `apply()`:
  `output = base_linear(x)`, `delta = (x_flat @ A.T) @ B.T`, `y += delta`.
- vLLM-Omni `vllm_omni/diffusion/lora/manager.py:518-573` — `_activate_adapter`:
  copies weights to GPU buffers, applies `lora_b * scale`.
- vLLM-Omni `vllm_omni/diffusion/lora/manager.py:340-436` — `_replace_layers_with_lora`:
  wraps base linear, original becomes `.base_layer`.
- vLLM-Omni `vllm_omni/diffusion/lora/manager.py:213-253` — `set_active_adapter`:
  single active adapter, `lora_scale` parameter.
- LocalAI `gosd.cpp:174-330` — `parse_loras_from_prompt`: regex, name resolution,
  multiplier accumulation, cleaned prompt.
- LocalAI `gosd.cpp:110-158` — `discover_lora_files`: scans `lora_dir`.
- Existing seam: `dit_lora.h` — `DitLoraAdapter`, `DitLoraContractName`,
  `DitLoraFactorPair`, `DitLoraSpec` (reused for adapter loading and key resolution).

## 5. Tests

- **Prompt tag parsing**: unit test `DitParseLoraTags` with various inputs:
  single tag, multiple tags, duplicate name (strengths sum), no tags (clean prompt
  returned as-is), tag with whitespace, tag with path separator in name (refused).
- **Name resolution**: unit test the resolution chain: exact filename in `lora_dir`,
  absolute path, case-insensitive, extension probing. Nonexistent name → error.
- **Delta math**: unit test that `LinearDev` with a known LoRA layer produces
  `base_out + (x @ A^T) @ B^T * strength` for a small matrix. Compare against a
  reference f32 computation.
- **Empty state**: `LinearDev` with `lora == nullptr` produces byte-identical output
  to the existing function (no behavioral change when no runtime LoRA is active).
- **H3 E2E**: generate a short video with and without a prompt-activated LoRA on
  `dgx:gpu0` and verify the output differs. The prompt contains
  `<lora:test_adapter:1.0>`.
- **LTX2 E2E**: same, on LTX2.5 if a suitable LoRA adapter is available; otherwise
  the H3 E2E is the gate.

## 6. Gates

- Build: the tree compiles clean with the new parameters on all Linear/LinearDev
  signatures.
- Prompt parsing unit tests: all pass.
- Delta math unit test: passes against reference.
- Empty-state regression: existing golden tests pass unchanged (no runtime LoRA
  active → identical output).
- H3 E2E: video generated with prompt-activated LoRA differs from baseline.
- Reachability: the prompt-tag path is reachable from `vllm_video_generate` (C ABI),
  `minimax_h3_gen` (CLI), the server, video_studio, and LocalAI.

## 7. Risks

- **Performance**: two extra matmuls per linear per sigma step. For rank-8 LoRA
  on H3 with ~40 layers and ~5 projections each, that is ~400 extra small GEMMs
  per step. The intermediate `[rows, rank]` buffer is small. The cost is measurable
  but secondary to correctness for this row. Performance optimization (fusing the
  delta into the base GEMM, or batching the shrink across layers) is a future row.
- **Device memory**: runtime LoRA factors are small (rank * dim per layer). For
  rank 8 and dim 4096, that is ~65 KB per factor pair. Negligible.
- **nvfp4 arm**: the fp4 path in `LinearDev` uses `MatmulNvfp4W4A16D`, which
  produces a different output dtype. The runtime delta must match the output dtype.
  If the fp4 path is active, the delta is computed in the stream dtype (bf16) and
  added. If dtype mismatch makes this infeasible, the fp4 arm refuses runtime LoRA
  with a named message.
- **Adapter key mismatch**: if a prompt-tagged adapter's keys do not match the
  contract (wrong prefix, wrong tensor names), `DitLoraAdapter::Open` refuses with
  the existing "does not bind" error. The user sees which adapter and which key
  failed.
- **Multiple LoRAs in one prompt**: the spec supports N adapters. The current design
  applies them as a single combined delta (strengths accumulate for duplicates).
  If two different adapters target the same layer, the deltas sum. This matches
  vLLM-Omni's single-active-adapter constraint: we do NOT support per-token
  multi-LoRA batching. All adapters in one prompt apply to all layers for all tokens.
  This is the correct behavior for diffusion (one generation = one set of adapters).

## 8. Git integration

One pull request (developer preference, 2026-09-17). The spec commit precedes the
implementation commits in the same pull request.

## 9. Stop conditions

- A real H3 LoRA adapter reveals a different key naming convention than the
  `model.diffusion_model.` prefix — adjust the prefix set in the existing
  `DitLoraContractName`, do not rearchitect.
- The two-matmul delta on the device path is too slow to complete an E2E generation
  within the lease budget — record the measurement, reduce steps to 2 (the minimum
  that produces a sigma schedule), and complete the gate. Performance is a future row.
- The fp4 device path cannot accept a runtime delta due to dtype constraints —
  refuse runtime LoRA on fp4 with a named message, record as owed, and gate on the
  bf16 dequant path only.

## Owed

- Performance optimization of the runtime delta (fuse into base GEMM, batch shrink
  across layers) — future row.
- Per-token multi-LoRA batching (different adapters for different tokens in the same
  batch) — not needed for diffusion (one generation = one adapter set), would be
  needed for LLM-style multi-LoRA serving.
- Runtime LoRA on the nvfp4 fp4-resident arm — refused if dtype constraints prevent
  the delta add.

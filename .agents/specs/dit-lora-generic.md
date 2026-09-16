# DiT LoRA — shared load-time fusion seam for diffusion transformers

**Row:** `ROAD-V1-DIT-LORA` (roadmap portfolio).
**Issue:** `ISSUE-LOCAL-01M2KNH9FDSRM0PEWDJFTGJF9X`.
**Upstream:** ltx-core `fuse_loras.py` (LTX2.5, secondary oracle `ltx-2`); vLLM-Omni
`DiffusionLoRAManager` (H3, primary oracle `vllm-omni`).
**Sibling:** `ltx25-lora-fuse-seam.md` (DONE) routed the LTX2 fusion product through
`vt::Matmul`. This row generalizes the fusion API it optimized.

## Now

**SPEC.** Row claimed, spec committed, no implementation yet.

## Scope

Generalize the LTX2.5 LoRA load-time-fusion implementation
(`Ltx2LoraSpec`/`Ltx2LoraAdapter`/`Ltx2FuseLoraIntoTensor` in
`ltx2_lora.{h,cpp}`) into a model-agnostic `dit_lora` seam that both LTX2.5 and
MiniMax-H3 use. Wire H3 to call the seam during weight materialization. Expose
both models through LocalAI config-based `lora_adapters`/`lora_scales`.

Load-time fusion means LoRA deltas are baked into the base weights at model
load. The LoRA is always active. Per-request prompt-activated LoRA
(`<lora:name:strength>`) requires the runtime punica path (the W3-W7 LoRA
engine, `LORA-RUNTIME`) and is a separate future row.

## 1. What exists

LTX2.5 has a complete, battle-tested LoRA load-time-fusion implementation:

- `include/vllm/model_executor/models/ltx2_lora.h` — `Ltx2LoraSpec` (path +
  strength), `Ltx2LoraFactorPair` (A/B factors in bf16), `Ltx2LoraAdapter`
  (opens a safetensors adapter, resolves keys onto the contract), and
  `Ltx2FuseLoraIntoTensor` (fuses the delta into a materialized tensor).
- `src/vllm/model_executor/models/ltx2_lora.cpp` — the implementation.
  `Ltx2LoraContractName` (line 106) is the ONE model-specific function: it
  strips the `diffusion_model.` ComfyUI prefix and rewrites
  `.lora_{A,B}.weight` to `.weight`.
- `src/vllm/model_executor/models/ltx2_loader.cpp` — `OpenDitLoras` (line 556)
  opens adapters against the contract; `FuseLorasInto` (line 574) calls
  `Ltx2FuseLoraIntoTensor` during `MaterializeDitTensor`.
- `include/vllm/model_executor/models/ltx2_loader.h` — `Ltx2DitLoadOptions::loras`
  (line 410) carries the `std::vector<Ltx2LoraSpec>`.
- `src/vllm/multimodal/ltx2_video.cpp` — `ResolveLoraSpecs` (line 692) reads
  `lora_path`/`lora_strength` from `VideoModelParams::extras` and populates
  the load options.

The arithmetic is an exact mirror of ltx-core `fuse_loras.py:119-150`: B is
scaled by strength, the `(B*strength) @ A` product is formed in bf16 via
`vt::Matmul`, the delta is added to the weight in f32, and the result is stored
in the weight's dtype. `ltx25-lora-fuse-seam.md` routed the product through
`vt::Matmul` for a 143x speedup.

MiniMax-H3 has no LoRA support at all. It is the same architecture class — a
diffusion DiT that materializes weights to host buffers then stages to device —
so load-time fusion is the natural mechanism. H3's weight materialization
happens in:

- `src/vllm/model_executor/models/minimax_h3_device.cpp:1116` —
  `StreamMiniMaxH3DitToDeviceBf16` dequantizes GGUF tensors to bf16/f32 on host
  (line 1163) and uploads to device (line 1171). The gap between dequant and
  upload is the fusion point.
- `src/vllm/model_executor/models/minimax_h3_gguf.cpp:299,339` —
  `LoadMiniMaxH3DitFromGguf` and `LoadMiniMaxH3DitFromGgufBf16` materialize all
  weights to host.
- `src/vllm/model_executor/models/minimax_h3_nvfp4.cpp:63` —
  `LoadMiniMaxH3DitFromNvfp4` materializes from safetensors.

H3's tensor contract (`EnumerateMiniMaxH3DitTensors`, `minimax_h3.cpp:372`)
uses bare module names like `blocks.0.attn.qkv_proj.weight`,
`blocks.0.mlp.fc1.weight`. ComfyUI LoRA adapters for H3 carry the
`model.diffusion_model.` prefix, so the contract name rewrite strips it.

## 2. Design

### 2.1 Generalize into `dit_lora`

Rename the LTX2-specific types and functions to model-agnostic names in new
files `include/vllm/model_execution/models/dit_lora.h` and
`src/vllm/model_execution/models/dit_lora.cpp`:

| Current (LTX2-specific) | New (shared) |
|---|---|
| `Ltx2LoraSpec` | `DitLoraSpec` |
| `Ltx2LoraFactorPair` | `DitLoraFactorPair` |
| `Ltx2LoraAdapter` | `DitLoraAdapter` |
| `Ltx2LoraReferenceFactors` | `DitLoraReferenceFactors` |
| `Ltx2FuseLoraIntoTensor` | `DitFuseLoraIntoTensor` |
| `Ltx2ReadLoraMetadataFactor` | `DitReadLoraMetadataFactor` |
| `Ltx2ResolveLoraReferenceFactors` | `DitResolveLoraReferenceFactors` |
| `Ltx2LoraContractName` | (removed — injected per model, see 2.2) |

The fusion arithmetic, adapter loading, spec parsing, and metadata factor
reading are generic. Only the contract name rewrite is model-specific.

### 2.2 Contract name rewrite as a per-model injection

The one model-specific function is the key rewrite that maps a LoRA checkpoint
key to a contract tensor name:

```cpp
bool DitLoraContractName(const std::string& key,
                         const std::vector<std::string>& prefixes,
                         std::string* out_target, bool* out_is_a);
```

`prefixes` is the set of ComfyUI prefixes to strip before the
`.lora_{A,B}.weight` -> `.weight` rewrite. For LTX2 it is `{"diffusion_model."}`.
For H3 it is `{"model.diffusion_model.", "diffusion_model."}`.

The `Ltx2LoraAdapter::Open` call site passes its model's prefixes. The adapter
class stores them and uses them during key resolution.

### 2.3 Migrate LTX2 to the shared seam

`ltx2_lora.{h,cpp}` become thin aliases or are replaced by `#include` of
`dit_lora.h` with `Ltx2LoraSpec = DitLoraSpec` typedefs. The LTX2 loader's
`OpenDitLoras`/`FuseLorasInto` pass `{"diffusion_model."}` as the prefix set.
No behavioral change: the existing LTX2 LoRA golden tests must pass unchanged.

### 2.4 Wire H3 LoRA fusion

Add `std::vector<DitLoraSpec> loras` to H3's load options (mirror
`Ltx2DitLoadOptions::loras`). In `StreamMiniMaxH3DitToDeviceBf16`
(`minimax_h3_device.cpp:1116`), call `DitFuseLoraIntoTensor` on the host buffer
after dequant and before device upload (between lines 1168 and 1171).

The non-streaming host loaders (`LoadMiniMaxH3DitFromGguf`,
`LoadMiniMaxH3DitFromGgufBf16`) get the same fusion call during materialization.

For the nvfp4 arm, fusion applies to the bf16-dequantized path only
(`StreamMiniMaxH3Nvfp4ToDeviceBf16`). Fusing into fp4-packed weights is out of
scope; the arm refuses if a LoRA is requested with an fp4 target and names the
owed arm.

### 2.5 Extras seam for H3

Add `lora_path`/`lora_strength` support to H3's `MiniMaxH3VideoModelParams`
through `VideoModelParams::extras`, mirroring LTX2's
`ResolveLoraSpecs` (`ltx2_video.cpp:692`). The extras are already the shared
route through `VideoModelParams::extras` (`video_engine.h`).

### 2.6 LocalAI wiring

LocalAI's vllm-cpp backend reads `lora_adapters` (list of paths) and
`lora_scales` (list of floats) from the model config YAML, mirroring the
diffusers backend. The backend maps these into indexed `lora_path`/
`lora_path_2`/`lora_strength`/`lora_strength_2` extras on the C ABI call.

## 3. Phases

| Phase | What | Files |
|---|---|---|
| 1 | Extract `dit_lora.{h,cpp}` from `ltx2_lora.{h,cpp}` | new files |
| 2 | Migrate LTX2 to use `dit_lora` (typedefs or direct) | `ltx2_lora.*`, `ltx2_loader.cpp`, `ltx2_video.cpp` |
| 3 | Wire H3: `DitLoraSpec` in load options, fusion in `StreamMiniMaxH3DitToDeviceBf16` + host loaders, contract name prefix for H3, extras resolution | `minimax_h3_device.cpp`, `minimax_h3_gguf.cpp`, `minimax_h3_video.h`, `minimax_h3_video.cpp` |
| 4 | Wire LocalAI vllm-cpp backend `lora_adapters`/`lora_scales` -> extras | LocalAI `backends/vllm.cpp` |
| 5 | Build + E2E test on `dgx:gpu0` via `rc run` | — |

## 4. Upstream anchors

- ltx-core `fuse_loras.py:119-150` — `fuse_lora_weights` (the fusion arithmetic)
- ltx-core `fuse_loras.py:183-204` — key shape and A/B product
- ltx-core `iclora_utils.py:30-49` — `__metadata__` reference factors
- vLLM-Omni `vllm_omni/diffusion/loras/` — `DiffusionLoRAManager` (runtime path,
  not the mirror for load-time fusion, but the LoRA key naming and delta
  arithmetic are the same)
- LTX2 extras resolution: `ltx2_video.cpp:692` (`ResolveLoraSpecs`)

## 5. Tests

- **LTX2 regression**: existing LTX2 LoRA golden tests pass unchanged after the
  rename. The fusion output is byte-identical (same FNV-1a digest).
- **H3 LoRA fusion**: a focused test that loads an H3 checkpoint with a LoRA
  adapter and verifies the fused weights differ from the unfused weights by
  exactly the expected delta. Use a small test adapter (rank 2) against a
  subset of H3 blocks.
- **H3 contract name rewrite**: unit test that `model.diffusion_model.blocks.0
  .attn.qkv_proj.weight.lora_A.weight` resolves to
  `blocks.0.attn.qkv_proj.weight`.
- **H3 end-to-end**: generate a short video with and without a LoRA on
  `dgx:gpu0` and verify the output differs.

## 6. Gates

- LTX2 LoRA golden tests: byte-identical output (existing gate, unchanged).
- H3 LoRA: the fused-weight test and the E2E generation test.
- Build: the tree compiles clean with the new files.

## 7. Risks

- **H3 contract name prefix**: the exact ComfyUI prefix for H3 LoRA adapters
  needs verification against a real adapter. The spec assumes
  `model.diffusion_model.` based on the ComfyUI format; the implementation
  confirms against a real adapter file.
- **nvfp4 arm**: fusing into fp4-packed weights is out of scope. The arm refuses
  with a named message if a LoRA targets an fp4 path. Recorded as owed.
- **Multiple LoRAs on H3**: the LTX2 implementation already supports N adapters
  with conflict resolution. H3 inherits this through the shared seam.

## 8. Git integration

One pull request (default repository policy). The spec and implementation
land together.

## 9. Stop conditions

- A real H3 LoRA adapter reveals a different key naming convention than
  `model.diffusion_model.` prefix — adjust the prefix set, do not rearchitect.
- The H3 streaming path's memory invariant (one host buffer at a time) is
  violated by the fusion — fuse into the pre-upload host buffer, which is
  already the single live buffer.

## Owed

- nvfp4-arm LoRA fusion (fuse into fp4-packed weights, or dequant-fuse-requant)
- Runtime prompt-activated LoRA (`<lora:name:strength>`) — separate row
  `ROAD-V1-LORA-RUNTIME`

## Outcome

All five phases landed and were verified on GPU.

### What was measured

- Unit tests: 11 H3 LoRA tests pass (contract rewriting, fusion math,
  check-was-applied refusal, extras resolution, gap refusal, `IsDitLoraExtra`).
  17 LTX2 LoRA tests pass (unchanged behavior after migration to shared seam).
  6 LocalAI `buildLoraExtras` tests pass. 92/92 vllm-cpp backend tests pass.
- E2E on dgx:gpu0 (Grace Blackwell, sm_121a, CUDA 13.0, Q3_K_M GGUF ~15 GB,
  bf16 dequant path):
  - Positive: valid rank-8 bf16 LoRA targeting `blocks.0.attn.qkv_proj` fused
    silently during 535/535 tensor stream. Hit VAE check (expected — no VAEs
    supplied). Fusion succeeded.
  - Negative: LoRA targeting `nonexistent.tensor` refused with "does not bind"
    error during `DitOpenLoras`. LoRA was loaded and rejected.

### What was rejected and why

- Runtime punica-style LoRA (per-request, not load-time): vLLM-Omni uses
  `DiffusionLoRAManager` for runtime matmul-delta fusion. Rejected for v1
  because the tree's H3 uses load-time weight materialization (like LTX2),
  not runtime matmul. Load-time fusion is the natural mirror. Runtime
  activation is a separate row (`ROAD-V1-LORA-RUNTIME`).
- Fusing into nvfp4-packed weights: out of scope. The fp4 arm refuses with a
  named message if a LoRA targets an fp4 path. The streaming paths fuse on
  the dequantized host buffer before repacking.
- torch for synthetic LoRA creation in E2E: torch is not installed on the DGX.
  Used safetensors + numpy with manual BF16 conversion instead.

### Bug found and fixed during E2E

`MiniMaxH3VideoModelParamsFromGeneric` filtered LoRA extras with
`IsDitLoraIndexedExtra`, which only recognizes indexed keys (`lora_path_2`).
The first adapter's base keys (`lora_path`, `lora_strength`) were silently
dropped, so `--lora` fused nothing. Fixed by adding `IsDitLoraExtra` (commit
`cc8ca06`).

### Why each default has its value

- Load-time fusion (not runtime): matches LTX2's proven pattern and the tree's
  materialize-then-stream architecture.
- `model.diffusion_model.` + `diffusion_model.` prefix set: ComfyUI adapters
  carry the full prefix; bare-name adapters also work.
- bf16 dequant path for E2E: the f32 path would use ~128 GB, exceeding 128 GB
  unified memory. bf16 fits.
- Strength defaults to 1.0: mirrors vLLM and LTX2.

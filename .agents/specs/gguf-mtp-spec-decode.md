# Spike: MTP speculative decoding from a GGUF target (`SPEC-MTP-GGUF`)

Stable row: `SPEC-MTP-GGUF` (`.agents/engine-matrix.md`).
Depends on: `SPEC-MTP` (`DONE`, safetensors), [mtp-spec-decode.md](mtp-spec-decode.md).

## Scope

Make `--speculative-config '{"method":"mtp"}'` work when the TARGET is a `.gguf`
file, not only a safetensors directory.

Today `LoadedEngine::FromModelDir` refuses the combination outright
(`src/vllm/entrypoints/model_loader.cpp:717-723`):

```
speculative decoding requires a safetensors target checkpoint: GGUF exports
lack the mtp.* draft tensors / bf16 target embed+lm_head the DFlash draft shares
```

That message encodes an assumption from the original MTP spike
([mtp-spec-decode.md](mtp-spec-decode.md):979-980, "GGUF checkpoints lack
`mtp.*` -> document as safetensors-only feature **until we re-export GGUFs with
the head**"). The assumption is now stale in one direction: llama.cpp's Qwen3.5
converter DOES emit the MTP head into GGUF, under its own layer-indexed `nextn`
naming, and our own `HfConfigFromGguf` ALREADY reads the metadata key that
announces it. So the head is present and detectable in third-party GGUFs we do
not produce; what is missing is our loader mapping.

IN SCOPE: the GGUF weight resolver for the `mtp.*` block, the config plumbing so
`ResolveSpecConfig` learns the head depth from GGUF metadata, removing the
rejection for `method == "mtp"`, and the token-identity gate.

OUT OF SCOPE: `dflash` over GGUF (separate row `SPEC-DFLASH-GGUF`,
[gguf-dflash-draft.md](gguf-dflash-draft.md)); a standalone MTP-only drafter
GGUF as a SECOND model handle (noted under Risks); non-Qwen3.5/3.6
architectures, which the widened spec KV path does not serve at all
(`MakeKVCacheMaybeSpec`, `src/vllm/entrypoints/model_loader.cpp`).

## Upstream chain

The producer side is llama.cpp, not vLLM (vLLM has no GGUF MTP path), so the
"upstream" to mirror here is the GGUF CONTRACT llama.cpp writes.

- `gguf-py/gguf/constants.py:129` — `{arch}.nextn_predict_layers`, the head-depth
  metadata key (the GGUF spelling of HF `mtp_num_hidden_layers`).
- `gguf-py/gguf/constants.py:910-917` — the `NEXTN_*` tensor enums;
  `:1494-1501` — their GGUF names: `blk.{bid}.nextn.eh_proj`,
  `.nextn.embed_tokens`, `.nextn.enorm`, `.nextn.hnorm`,
  `.nextn.shared_head_head`, `.nextn.shared_head_norm`, plus the unindexed
  `nextn.pre_projection` / `nextn.post_projection`.
- `gguf-py/gguf/tensor_mapping.py` — `NEXTN_*` map to HF
  `model.layers.{bid}.eh_proj` / `.enorm` / `.hnorm` / `.shared_head.head` /
  `.shared_head.norm` / `.embed_tokens`, i.e. the DeepSeek-V3 MTP spelling.
- `conversion/qwen.py:535-604` — `_Qwen35MtpMixin`, the authoritative bridge.
  It extends `block_count` by `mtp_num_hidden_layers`, emits
  `add_nextn_predict_layers(n)`, and in `filter_tensors` remaps Qwen3.5's
  `mtp.*` onto the layer-indexed DeepSeek spelling with exactly:

  | Qwen3.5 HF | remapped to |
  |---|---|
  | `mtp.fc` | `model.layers.{L}.eh_proj` |
  | `mtp.pre_fc_norm_embedding` | `model.layers.{L}.enorm` |
  | `mtp.pre_fc_norm_hidden` | `model.layers.{L}.hnorm` |
  | `mtp.norm` | `model.layers.{L}.shared_head.norm` |
  | `mtp.layers.{i}.<rest>` | `model.layers.{L+i}.<rest>` |

  where `L` = the ORIGINAL `num_hidden_layers` (`_original_block_count`). Note
  the asymmetry: the four scalar tensors land on block `L` while
  `mtp.layers.{i}.*` land on `L+i`. At `mtp_num_hidden_layers == 1` (both gate
  checkpoints) these coincide.
- Same mixin: `no_mtp` and `mtp_only` conversion flags, so a Qwen3.5 GGUF in the
  wild may carry the head, omit it, or BE only the head.

Also relevant, and the reason `llama.cpp` archs matter beyond Qwen: the same
`nextn` tensor set is declared for `QWEN35`, `QWEN35MOE`, `GEMMA4_ASSISTANT`,
`COHERE2MOE`, `DEEPSEEK32`, `GLM4_MOE`, `GLM_DSA`, `EXAONE4`, `EXAONE_MOE`,
`BAILINGMOE2`. Only the two Qwen35 archs are reachable for us at this pin.

## Our baseline

What already exists, and is the reason this row is small:

- **The head-depth key is already read.** `HfConfigFromGguf`
  (`src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:568`) does
  `const int64_t nextn = OptInt(gguf, p + "nextn_predict_layers", 0);` and uses
  it as `c.num_hidden_layers = block_count - nextn`. The trunk layer count is
  therefore ALREADY correct on a head-carrying GGUF; the head blocks are simply
  never loaded. The value is then discarded.
- **The MTP weight loader is already resolver-shaped.**
  `LoadQwen3_5MTP(const TensorResolver& get, const HfConfig&, Qwen3_5MTPKind)`
  (`src/vllm/model_executor/models/qwen3_5_mtp.cpp:272`) is source-agnostic; the
  safetensors entry point (`:337`) only builds a name -> shard map and delegates.
  `TensorResolver` is `std::function<const StTensor&(const std::string&)>`
  (`include/vllm/model_executor/models/qwen3_5_weights.h:348`).
- **`StTensor` is a plain view**, not a safetensors-owned type:
  `{dtype, shape, data, nbytes}`
  (`include/vllm/model_executor/model_loader/safetensors_reader.h:17-22`). A
  GGUF-backed resolver can hand out views over dequantized buffers it owns.
- **GGUF -> bf16 dequant already exists** and is already used by the trunk GGUF
  loader: `DequantGgufRowToBf16` via
  `vllm/model_executor/model_loader/gguf_dequant.h`, wrapped locally in
  `qwen3_5_gguf_weights.cpp`.
- **The spec engine itself is architecture-complete and gated** for these
  checkpoints (`SPEC-MTP` `DONE`: three-way token-exact at c1, c2-c8 on-par with
  vLLM). Nothing about the propose/verify loop changes here.

So the gap is exactly: a `TensorResolver` over `GgufFile`, a config field, and
one deleted `throw`.

## Port map

| # | File | Change |
|---|---|---|
| 1 | `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:568` (`HfConfigFromGguf`) | Stop discarding `nextn`: set `c.raw["mtp_num_hidden_layers"] = nextn` so `ResolveSpecConfig` (`src/vllm/entrypoints/model_loader.cpp:458-470`) resolves the real depth instead of its `int64_t{1}` fallback. Keep `num_hidden_layers = block_count - nextn` unchanged. |
| 2 | `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp` (`LoadQwen3_5MTPFromGguf`) | **CORRECTED at implementation time.** The head loader lives in the EXISTING GGUF weights TU, NOT as a `TensorResolver` in a new one. **The resolver seam is the wrong seam**: it hands out raw tensor views, but the GGUF conventions live one level up in that file's helpers, and there are three of them, all silent if missed. (a) Norm weights are stored **(w + 1)** and must be un-shifted (`OwnNormMinus1`) - a plain read leaves every norm off by one and poisons every proposal without failing anything. (b) Matmul weights carry the file's quantization + residency routing (`OwnMatmulWeight`, `RequireExpand`). (c) `GgufTensorInfo::shape` is already torch `[N, K]`, so `fc` is the VERBATIM path, not the transposing one - the transposing helper would yield `[2H, H]` and only fail deep inside the draft forward. Going through the resolver means re-deriving all three in a second place; reusing the trunk helpers is what makes the head obey the same conventions as the trunk it speculates for. |
| 3 | same | Scalars resolve at block `L` (`nextn.eh_proj`/`enorm`/`hnorm`/`shared_head_norm`); the head's transformer block(s) at `L+i` reuse `LoadAttnGguf` / `OwnMatmulWeight` / `LoadMoeGguf`. The head block is ALWAYS full-attention and is deliberately NOT looked up in `config.layer_types`, which covers the trunk only (a lookup there would run off the end). |
| 4 | `src/vllm/entrypoints/model_loader.cpp:717-723` | Narrow the rejection: keep it for `method == "dflash"` (until `SPEC-DFLASH-GGUF`), drop it for `"mtp"`. `"ngram"` is already unaffected. |
| 5 | `src/vllm/entrypoints/model_loader.cpp` (GGUF branch, ~`:710-730`) | After `ModelRegistry::Load`, when a spec config is present call `LoadQwen3_5MTP` with the GGUF resolver and `AttachMtpDraftWeights`, mirroring the safetensors branch at `:770`. |
| 6 | `include/vllm.h` / `docs` | No ABI change. The capability is reachable through the existing `speculative_config` field. |

Decision to make in step 2 (record the outcome in this file): whether the MTP
head uses the GGUF's own `blk.{L}.nextn.shared_head_head` or the target's
`output.weight`. Qwen3.5 SHARES the target lm_head
(`LoadQwen3_5MTP` asserts `!UsesDedicatedEmbeddings(config)`), and the converter
emits `shared_head.head` only because the DeepSeek layout has that slot, so the
two should be the same tensor. Verify byte-equality on a real GGUF; if they
differ, the target's is authoritative.

## Tests to port

There is no upstream test to port (vLLM has no GGUF MTP path), so the gates are
ours, built to the same shape as the safetensors MTP gates.

| Test | Location | What it pins |
|---|---|---|
| Resolver name mapping | new `tests/vllm/models/test_qwen3_5_gguf_mtp.cpp` | Every name `LoadQwen3_5MTP` requests resolves; the four scalars land on block `L`, `mtp.layers.{i}` on `L+i`. Synthetic in-memory `GgufFile`, no weights. RED-first by asserting against a deliberately off-by-one block index. |
| Dequant equivalence | same | A GGUF-backed resolver over an F32/F16 head produces bit-identical bf16 to the safetensors resolver over the same values. |
| Config depth | extend `tests/vllm/models/test_qwen3_5_gguf_weights.cpp` (or the owning GGUF config test) | `nextn_predict_layers=N` yields `raw["mtp_num_hidden_layers"] == N` AND `num_hidden_layers == block_count - N`. Currently only the second holds. |
| Rejection narrowing | `tests/capi/test_capi.cpp` or the model-loader test | `mtp` + GGUF no longer throws; `dflash` + GGUF still does, with the message naming dflash. |
| Three-way token gate | new `tests/parity/test_qwen35_gguf_spec_decode.cpp`, modelled on `tests/parity/test_qwen27_spec_decode.cpp` | GGUF spec-ON == GGUF spec-OFF token-for-token at c1, plus acceptance > 0. |

## Gates

1. **Spec-OFF byte-identical (SACRED).** The GGUF trunk path must not move:
   27B 235/235, 35B 315/315, Coder 138/138. Step 1 touches `HfConfigFromGguf`,
   which every GGUF load runs, so this is the gate that matters most.
2. **Token identity, c1.** GGUF spec-ON == GGUF spec-OFF, token-for-token,
   greedy, single request. This is the `SPEC-MTP` I5e bar restated for GGUF.
3. **Cross-format agreement (the strong gate, if the checkpoint allows it).**
   For a checkpoint available BOTH as safetensors and as an F16 GGUF, the GGUF
   spec-ON tokens should equal the safetensors spec-ON tokens. Only meaningful
   at F16/F32 (a quantized GGUF legitimately diverges), so record it as
   NOT APPLICABLE when only a quantized export exists.
4. **Acceptance parity.** GGUF acceptance rate within noise of the safetensors
   rate on the same prompts; a near-zero rate means the head loaded but is
   wired wrong (the exact failure mode I5e hit and RCA'd).
5. **Speed: PENDING, not owed by this row.** Spec-decode throughput is already
   recorded under `SPEC-MTP` (I6/I7). This row owes a `docs/BENCHMARKS.md`
   entry, but it may legitimately be `PENDING` with the reproduction command
   until a GGUF A/B is run.

### Gate dispositions as of 2026-07-29 (re-anchored to a production-configured build)

Every GPU row below was originally measured on a build configured WITHOUT
`-DVLLM_CPP_CUTLASS_DIR` and WITHOUT `-DVLLM_CPP_TRITON=ON`, the defect
`CLAIM-27B-GATE-RCA` proved. All of them were re-run on a build carrying both,
proven correct three ways (clean configure log, `cuobjdump -lelf` 40 cubins all
`sm_121a`, SACRED 27B gate 235/235 exit 0). **Every disposition survived
unchanged.** The one thing that did NOT survive is the `G6` CPU-vs-GPU narrative,
retracted below.

| Gate | Disposition | Evidence |
|---|---|---|
| 1 spec-OFF SACRED | **MET** | The trunk regression sweep recorded under `G4` is unchanged; the only later change is an env-gated test case, which touches no engine code. Re-confirmed 2026-07-29: SACRED `test_qwen27_paged_engine` 235/235 exit 0 on the production build |
| 2 token identity c1 | **MET on BOTH devices** | CPU dense 2B token-exact; GPU sm_121a 35B A3B MoE token-exact, `G5`. Re-run 2026-07-29 on the production build: 3/3 cases, 10/10 assertions, exit 0, spec-ON token-identical to spec-OFF, 90.26 GiB, 7m13.59s |
| 3 cross-format agreement | **NOT APPLICABLE**, for two reasons | the gate's own precondition is an F16/F32 GGUF sibling and every head-carrying export on hand is quantized (2B Q8_0, 35B NVFP4); and per `G7` the same-weights NVFP4 safetensors sibling is not token-stable even against itself, so it could not serve as a reference either |
| 4 acceptance parity | **MET** | GGUF 13 proposed / 11 accepted (0.846) vs safetensors 12 proposed / 11 accepted (0.917) on the SAME prompt, params and box, `G7`. The GGUF side re-measured 2026-07-29 on the production build at exactly 13 / 11; the safetensors side was not re-run |
| 5 speed | **PENDING by design** | not owed by this row; `docs/BENCHMARKS.md` carries the disposition and the repro command |

## Dependencies

- `SPEC-MTP` `DONE` (the propose/verify loop, widened KV, GDN spec routing).
  Nothing here modifies it.
- The GGUF trunk loader and `GgufLoadPolicy` residency routing
  (`qwen3_5_gguf_weights.cpp`), unchanged but depended on for stem naming.
- `gguf_dequant.h` (`DequantGgufRowToBf16`), unchanged.
- **A test asset.** A Qwen3.5/3.6 GGUF that actually carries the head, i.e.
  converted WITHOUT `--no-mtp`. Sourcing or producing one is the first task and
  can block everything after step 1 — see Risks.
- No new third-party dependency. No ABI change. No CUDA/GPU dependency for
  steps 1-3 (CPU build suffices for the resolver tests).

## Work breakdown

| Row | Work | Gate | Blocked by |
|---|---|---|---|
| `G0` | **DONE 2026-07-28.** Confirmed against a real llama.cpp-converted Qwen3.5-2B (Q8_0 body) already on disk: `qwen35.block_count=25`, `qwen35.nextn_predict_layers=1` => trunk `L=24`, head at `blk.24`, carrying `nextn.eh_proj [2048,4096]` (torch `[N,K]`), `nextn.enorm/hnorm/shared_head_norm [2048]`, plus the block's own `attn_*`/`ffn_*`/`attn_norm`/`post_attention_norm`. **NO `nextn.embed_tokens` and NO `nextn.shared_head_head`** - Qwen3.5's head shares the target's embedding and lm_head, so the converter emits neither and the loader must not ask. Mapping table above is evidence-backed. | Evidence-backed, not inferred | - |
| `G1` | `HfConfigFromGguf` publishes `mtp_num_hidden_layers` into `c.raw` | Config unit test; SACRED spec-OFF byte-identical | - |
| `G2` | `MakeGgufMtpResolver` + name-mapping and dequant-equivalence tests | Resolver unit tests, RED-first | `G0`, `G1` |
| `G3` | Narrow the rejection; wire `LoadQwen3_5MTP` + `AttachMtpDraftWeights` into the GGUF branch | Rejection test; engine loads spec-ON on GGUF | `G2` |
| `G4` | **RUNS and FAILS 2026-07-28 - gate landed, defect OPEN.** `tests/parity/test_qwen35_gguf_spec_decode.cpp` drives the real 2B GGUF twice (spec-OFF then spec-ON) on CPU. It found and fixed one real loader defect (`fc.nk` unset: shape checks passed, the draft forward refused with "fc must be raw bf16 [H,2H]"), then surfaced a SECOND, still-open failure: **spec-ON diverges from spec-OFF while acceptance is HIGH (13 proposed / 10 accepted)**. Greedy MTP is exactness-preserving, so this is a genuine correctness defect. NOT yet attributed - see `G4a`. Needs `VT_GDN_STATE_BF16=0` on CPU (`causal_conv1d_spec_update` requires f32 conv state off CUDA) | Gates 2 and 4 | `G3` |
| `G4a` | **ANSWERED 2026-07-28 by bisect - the defect is NOT in this row.** Killing the drafter outright (zeroing `fc`, so every proposal is garbage) gives **23 proposed / 0 accepted** and *byte-identical output to the live-drafter run*, still diverging from spec-OFF. With ZERO drafts accepted the emitted sequence must be the target's own greedy sequence; it is not. So enabling speculation changes the TARGET's forward on CPU, independently of the head. This row's loader is EXONERATED (and, with a live head, earns 10/13 acceptance). Tracked below as `CPU-SPEC-DIVERGENCE`, which is a pre-existing engine defect this row merely surfaced | Attribution complete, no safetensors download needed | `G4` |
| `G5` | **DONE 2026-07-28.** The GPU end-to-end gate re-run on a from-scratch RELEASE-TARGET build, `-DVLLM_CPP_CUDA_ARCHITECTURES=121a`, verified by `flags.make` (`--generate-code=arch=compute_121a,code=[compute_121a,sm_121a]`) and by `cuobjdump -lelf` (20 cubins, ALL sm_121a, zero sm_75) rather than by `CMakeCache.txt`, whose `CMAKE_CUDA_ARCHITECTURES:STRING=75` line is the compiler probe default and is shadowed by `CMakeLists.txt:186`. dgx.casa GB10 under `flock`, 35B A3B NVFP4 GGUF: 2/2 cases, 10/10 assertions, exit 0, spec-ON token-identical to spec-OFF, 13 proposed / 11 accepted, 90.2 GiB peak RSS, 8m01s | Gates 2 and 4 on the release target | `G4` |
| `G6` | **ANSWERED 2026-07-28 by measurement - the CPU/GPU delta is a NEAR-TIE, not a defect.** The spec-OFF-only, double-gated probe case requests 20 alternatives per position and runs GPU then `CUDA_VISIBLE_DEVICES=`, same binary and file. Both arms pick `11751` at position 0 and fork at position 1 on a BIT-IDENTICAL prefix: GPU rank1 `13` -0.773180 vs rank2 `11` -0.847055 (margin 0.0739 nats); CPU rank1 `11` -0.765499 vs rank2 `13` -0.830374 (margin 0.0649 nats). Each device's pick is the other's rank 2, both margins ~7x inside the ratified 0.5-nat band, and the cross-device disagreement on the SAME token (0.057 and 0.082 nats) EXCEEDS the margin being decided, so the choice is settled by rounding. The 24-token texts diverge visibly only because positions 2+ condition on different prefixes | Attribution complete; no bisect owed | `G5` |
| `G7` | **MEASURED 2026-07-28 - gate 4 MET, and the reference arm turned out to be the interesting result.** The same test case pointed at the same-quantization-run safetensors sibling (`FromModelDir` takes either) gives acceptance 12 proposed / 11 accepted vs the GGUF's 13 / 11, so gate 4 is met. But that arm FAILS spec-ON == spec-OFF at concurrency 1 (one inserted token at position 10) and does not reproduce its own spec-OFF sequence run to run (position 16). The probe attributes both: it carries THREE exact ties (positions 7, 10, 16) with BIT-IDENTICAL logprobs, and both divergences land on two of them, whereas BOTH GGUF arms have ZERO exact ties and a 0.048-nat minimum margin. Mechanism: the safetensors NVFP4 path's logprobs fall on a 1/16 grid (quantized GEMM), the GGUF path expands to bf16 and does not. This EXONERATES the GGUF result and OPENS a `SPEC-MTP` item that is recorded, not fixed here | Gate 4, plus the concrete reason gate 3 is NOT APPLICABLE | `G6` |

`G0` is deliberately a work row, not an assumption. `G1` is independently
landable and inert.

## Defect surfaced AND FIXED by this row: `CPU-SPEC-DIVERGENCE`

**FIXED 2026-07-28.** Speculative decoding changed the target's own greedy output
on CPU. Root cause, in one line
(`src/vllm/model_executor/models/qwen3_5.cpp:3616`):

    const int64_t conv_row_elems = conv_dim * (Kw - 1);

Under speculation the persistent conv row is widened to
`conv_dim * ((Kw - 1) + num_spec)` so a rejected step can roll back, but the
prefill working copy is legitimately narrow (prefill produces only the K-1
leading taps). `GatherRows`/`ScatterRows` use ONE size for BOTH the slot stride
and the row contents, so once those differ they mis-stride the slot AND
mis-address every channel past the first. Post-prefill recurrent state was
therefore garbage, which is exactly the symptom: prefill token correct, first
decode token wrong.

**Fix:** `CopyStateRowsStrided` in the same TU, used by `GatherStateF32` /
`ScatterStateF32` when `cache.shape[2] != work.shape[2]`. It copies `taps`
elements per CHANNEL out of a `cache_taps`-strided row. When the two widths agree
- every non-speculative path - the contiguous helpers are used unchanged, so
non-spec behaviour is byte-identical by construction AND by gate.

**Evidence (CPU, 2B GGUF):**

    before   spec-OFF " Paris.\nA. True..."   spec-ON " Paris is the capital..."   13/10
    after    spec-OFF " Paris."               spec-ON " Paris."  IDENTICAL         13/11

Regression sweep, all unchanged: `test_ops_gdn` 1825, `test_gdn_metadata_builder`
483, `test_gdn_prefill_conv` 28, `test_qwen3_5_gdn_spec_routing` 12, `test_gguf`
103, `test_gguf_qwen36_loader` 99, `test_gguf_keep_quant` 5958,
`test_gguf_dequant` 215, `test_llm_engine` 196, `test_input_batch` 163,
`test_runner` 257, `test_capi` 232.

**Scope of the bug.** CPU-only in effect: the compressed-cache (fp16/bf16) arm
routes through `GdnStateGather`/`GdnStateScatter` ops instead, which take tensors
and carry their own geometry, so CUDA was never exposed. No GPU result is
affected or retracted. It was invisible until now because it requires
speculation ON (widened row) AND the f32 row-copy path AND an actual generation -
and there was no CPU spec-decode gate anywhere in the tree.

**How it was found.** Three-step bisect, each step removing a variable: (1) kill
the drafter -> identical divergence at 0 acceptance, so not the head; (2) `ngram`
-> token-exact while never calling the spec conv update, so not the widening
itself; (3) instrument both conv kernels -> 3-vs-4 stride split, pointing at the
state movement between them.

## GPU close-out 2026-07-28: the sm_121a arm PASSES, the CPU/GPU delta is a near-tie

Two things held the GPU arm open. NVFP4 GGUF dequant, which `QUANT-GGUF-NVFP4`
removed, is the superseded first one and its chronology now lives in
`.agents/state.md`. The two that remained are answered here as `G5` and `G6`.

### `G5`: the architecture claim was a MISREAD, and the trap is the keeper

The build the prior GPU run used was recorded as
`CMAKE_CUDA_ARCHITECTURES=75`, i.e. PTX-JIT'd Turing on an sm_121 GB10, and the
row was correctly held open on that basis. Re-configuring from scratch (build
directory DELETED first, so no stale cache) with the documented invocation
`-DVLLM_CPP_CUDA_ARCHITECTURES=121a` reproduces that cache line exactly:

    build-cuda/CMakeCache.txt
      CMAKE_CUDA_ARCHITECTURES:STRING=75
      VLLM_CPP_CUDA_ARCHITECTURES:STRING=121a

**That cache entry is not evidence of the compiled architecture.**
`enable_language(CUDA)` records the compiler's own probe default in the cache,
and [CMakeLists.txt:186](../../CMakeLists.txt) then sets a NORMAL variable of the
same name, which shadows the cache entry for every target created afterwards.
The compiled arch is one level down, and says 121a on both counts:

    build-cuda/CMakeFiles/vllm.dir/flags.make
      CUDA_FLAGS = -O3 -DNDEBUG -std=c++20
        "--generate-code=arch=compute_121a,code=[compute_121a,sm_121a]"
        -Xcompiler=-fPIC -Werror=all-warnings

    cuobjdump -lelf tests/test_qwen35_gguf_spec_decode
      -> 20 cubins, EVERY one sm_121a, zero sm_75 (native SASS, nothing JIT-only)

So: ask `flags.make` or `cuobjdump`, never `CMakeCache.txt`, which architecture a
build really targets. It also follows that the earlier run was probably ALREADY
121a and was held open on a decoy; that cannot be proven retroactively (the build
directory was deleted to guarantee a clean configure), which is why the run was
simply repeated rather than argued about.

**The repeated run PASSES.** dgx.casa GB10, CUDA 13, under `flock $HOME/gpu.lock`,
`VLLM_MTP_GGUF_MODEL=~/bench/q36-35b-a3b-nvfp4.gguf ./tests/test_qwen35_gguf_spec_decode`:
**2/2 cases, 10/10 assertions, exit 0**, spec-ON token-identical to spec-OFF,
**13 drafts proposed / 11 accepted**, peak RSS 90.2 GiB, wall 8m01s. The
sm_121a arm emits the same 24 tokens the earlier arm did, so the two builds agree
token-for-token as well.

### `G6`: RETRACTED 2026-07-29 - the CPU-vs-GPU token difference was the BUILD, not a device near-tie

**Read this before the section below.** `G6` measured a CPU-vs-GPU token
divergence on a build configured without CUTLASS and without the vendored
Triton-AOT GDN kernels. Re-measured on a production build, **there is no
divergence at all**: both devices emit the same 24 tokens
(`11751 11 264 3177 34756 364 1141 8807 3712 11 7431 11 321 25438 57902 13 27480
12484 303 279 9897 81183 919 314`). The probe shows the GPU now picking `11`
(-0.763897) over `13` (-0.824083) at position 1, where the defective build picked
`13` (-0.773180) over `11` (-0.847055); the CPU arm's logprobs are BIT-IDENTICAL
across the two builds, exactly as expected since CUTLASS and Triton are CUDA-only.
Zero exact ties in either arm, minimum margins 0.060186 nats (GPU) and 0.064875
(CPU). So the near-tie at position 1 is real and about the same size, but the
"each device's pick is the other's rank 2, one coin flip cascading" conclusion
described a build defect, not a property of this row or of the devices. The
framing paragraph immediately below is still correct as a general statement of
the bar; the measurement it introduces is superseded.

State the bar first, because the previous framing invited the wrong chase:
**CPU-vs-GPU token equality is NOT this row's gate and is not generally
expected.** Different kernels accumulate bf16 in different orders, which is
exactly why several rows in this repo gate on a near-tie-robust form rather than
a strict one. The binding bar is spec-ON == spec-OFF WITHIN one device, and that
is what `G5` passed.

The probe (third, double-gated case in
`tests/parity/test_qwen35_gguf_spec_decode.cpp`) runs spec-OFF ONLY, so nothing
it shows can be blamed on or credited to the draft head, and requests 20
alternatives per position. Same binary, same file, GPU arm then
`CUDA_VISIBLE_DEVICES=` arm, one `flock` series.

Both arms agree on position 0 (`11751` = `" Paris"`, same rank-2 `264`) and fork
at position 1, where they still share a BIT-IDENTICAL prefix, so their rows are
directly comparable and the comparison is teacher-forced by construction:

| arm | rank 1 | rank 2 | margin |
|---|---|---|---|
| GPU (sm_121a) | `13` `"."` -0.773180 | `11` `","` -0.847055 | **0.0739 nats** |
| CPU | `11` `","` -0.765499 | `13` `"."` -0.830374 | **0.0649 nats** |

**Each device's pick is the other device's rank 2**, and both margins sit roughly
7x inside the repo's ratified near-tie band (`kNearTieMnats = 500`, 0.5 nats).
The decisive number is the comparison of those margins against the cross-device
noise on the SAME token: the two arms disagree by 0.057 nats on `13` and 0.082
nats on `11`, i.e. **the numerical disagreement between the two forwards is the
same size as, or larger than, the gap it is being asked to decide**. Which of
`"."` and `","` wins is therefore settled by rounding, not by the model. That is
the textbook definition of an approximate tie, and it is disposition (a).

The reason the two 24-token texts look nothing alike (`" Paris.\nA. True\nB.
False..."` vs `" Paris, a city renowned for its rich history..."`) is that from
position 2 on the two arms are conditioning on DIFFERENT prefixes: the entire
visible difference is the cascade of one 0.07-nat coin flip. **The size of a text
difference is not evidence of the size of its numerical cause**, which is the
trap this row records.

Refuted alternatives, so the reading is not merely plausible:

- Not materialization: the loader hands both arms identical bf16 buffers (the
  NVFP4 expansion is host-side and device-independent), and
  `QUANT-GGUF-NVFP4` separately proved the containers byte-equal on 192 tensors.
- Not the draft head: the probe never enables speculation.
- Not a GDN state-dtype mismatch: `MakeQwen3_5KVCacheSpec`
  (`src/vllm/model_executor/models/qwen3_5_common.cpp:52-63`) defaults the conv
  and SSM state to bf16 on BOTH devices, and the probe left `VT_GDN_STATE_BF16`
  unset on both arms, so the two arms carried the same state precision. The
  known CPU f32 restriction is specific to `causal_conv1d_spec_update`, which
  spec-OFF never calls, and indeed the CPU probe arm ran clean without it.
- Not a structural forward divergence: the arms agree on the argmax at position
  0 and on the rank-2 candidate there, and at position 1 they agree on WHICH two
  tokens are in contention and on their logprobs to within 0.08 nats. A broken
  forward does not agree to two decimal places.

### `G7`: the safetensors sibling is NOT a token-exact reference, and why

Gate 4 wants the GGUF acceptance rate compared against the safetensors rate on
the same prompts. The 35B A3B NVFP4 GGUF has a safetensors sibling from the SAME
quantization run (`~/bench/q36-35b-a3b-nvfp4-vllm`, 19 `mtp.*` tensors), and
because `LoadedEngine::FromModelDir` takes a directory or a `.gguf`
interchangeably, the SAME test case runs on it unchanged. That measurement was
taken, and it produced a finding worth more than the rate it was after.

**Acceptance parity: MET.** GGUF 13 proposed / 11 accepted (0.846) vs safetensors
12 proposed / 11 accepted (0.917), same prompt, same engine params, same box.
Both are far from the collapsed-acceptance failure mode gate 4 exists to catch.

**But the safetensors arm FAILS the same-device token-identity bar** that the
GGUF arm passes: spec-ON emitted `31883` (`" vibrant"`) at position 10 where
spec-OFF emitted `7431` (`" culture"`), inserting one token and shifting the
tail. It also did not reproduce its OWN spec-OFF sequence between two runs of the
same binary, diverging at position 16.

The probe explains both, and the explanation is quantitative. Running it on all
three arms and reading the rank-1-minus-rank-2 margin at every one of the 24
positions:

| arm | exact ties (0.000 nats) | minimum margin | positions under 0.1 nats |
|---|---|---|---|
| GGUF, GPU sm_121a | **none** | 0.0482 nats (pos 23) | 1, 23 |
| GGUF, CPU | **none** | 0.0649 nats (pos 1) | 1 |
| safetensors, GPU | **three: pos 7, 10, 16** | 0.0000 nats | 7, 10, 16 |

Both safetensors divergences land EXACTLY on two of its three exact ties. At
position 10 the top THREE candidates (`7431`, `12387`, `31883`) carry
BIT-IDENTICAL logprobs of -1.563910; at position 16 the top two (`11751`,
`27480`) are bit-identical at -1.837616. Nothing decides those but tie-break
order, and the batched verify forward and the sequential spec-OFF forward do not
have to break them the same way (nor, evidently, does the same forward across
runs).

**Why the ties exist there and not in the GGUF arm.** The safetensors arm's
logprobs fall on a coarse grid: the gaps between its ranked candidates are
multiples of 1/16 (0.0625), e.g. -1.563910, -2.188910 (0.625 = 10/16), -2.313910
(0.75 = 12/16). The GGUF arm's are not (-0.773180, -0.847055). The safetensors
NVFP4 path keeps the weights packed and runs the quantized GEMM, whose logits
land on that grid, so distinct tokens genuinely collide; the GGUF path expands
NVFP4 to bf16 (`QUANT-GGUF-NVFP4` has no `vec_dot`, so keep-quant expands) and
runs the bf16 GEMM, which resolves them.

**Consequences, split by owner:**

- FOR THIS ROW: the GGUF arm's token identity is a real property, not luck. It
  has ZERO exact ties in the window and a 0.048-nat minimum margin, and it
  reproduced its sequence across every run and across the logprobs-on and
  logprobs-off variants. Gate 4 is met and gate 3 is NOT APPLICABLE for a second,
  now concrete reason: the only same-weights sibling cannot serve as a token
  reference because its own tokens are not stable.
- NOT FOR THIS ROW, and newly OPEN: `docs/SPECULATIVE-DECODING.md` states
  spec-ON == spec-OFF as a concurrency-1 property, and on the 35B A3B NVFP4
  SAFETENSORS target on this build it does not hold. That belongs to `SPEC-MTP`
  and the NVFP4 quantized-GEMM logit path, not to the GGUF loader. It is recorded
  rather than fixed here, and it is not root-caused beyond the tie measurement
  above (two runs, one prompt).

## Risks/decisions

- **RISK (highest): the test asset may not exist publicly.** Most published
  Qwen3.5/3.6 GGUFs may have been converted before the MTP mixin landed, or with
  `--no-mtp`. If no head-carrying GGUF can be obtained, `G0` must produce one
  locally from the safetensors gate checkpoint, which needs a llama.cpp
  converter run and disk for a second full export. If that is not affordable,
  the row stays `BLOCKED` at `G0` rather than shipping an ungated mapping. Do
  NOT infer the layout from names alone and call it done.
- **RISK: two MTP layouts, one loader.** Our `LoadQwen3_5MTP` is written to the
  Qwen3.5 `mtp.*` layout; the GGUF spelling is DeepSeek's. The remapper is
  purely nominal per `conversion/qwen.py`, but that is a claim about the
  CONVERTER, not about every producer. A GGUF written by another tool under the
  same `nextn` names could carry genuinely DeepSeek-shaped tensors. Mitigation:
  `RequireShape` on every resolved tensor (the safetensors path already does
  this at `qwen3_5_mtp.cpp:317`), so a layout mismatch fails loudly at load.
- **RISK: quantized head accuracy.** The head's tensors are quantized at
  whatever the export used. Dequant-to-bf16 keeps the compute path identical to
  safetensors but costs memory; computing on blocks would be faster and is what
  the trunk does for matmul weights via `GgufLoadPolicy`. DECISION for the first
  cut: dequant to bf16, because it makes gate 3 (cross-format agreement)
  meaningful at F16 and keeps the diff to the resolver. Revisit under a
  follow-on row if the head shows up in a profile.
- **RISK: a low-bit head may not be worth speculating with.** Acceptance rate is
  the whole value of MTP; a Q4 head could accept so rarely that spec-ON is
  SLOWER than spec-OFF. Gate 4 measures this. If acceptance collapses at low
  bit-width, the honest outcome is a documented minimum quantization, not a
  silent capability claim.
- **DECISION: `mtp_only` drafter GGUFs are out of scope.** llama.cpp can emit a
  GGUF containing only the head (`conversion/qwen.py`, `mtp_only`). Consuming
  one means a SECOND model handle plus a target/draft pairing contract, which is
  the `dflash` problem shape, not this one. Deliberately deferred; if wanted, it
  is a follow-on row that reuses `G2`'s resolver.
- **DECISION: no ABI change.** `speculative_config` already carries this. A
  caller that works on safetensors works unchanged on GGUF once this lands,
  which is the point.
- **NON-RISK, stated to close it:** `ngram` over GGUF already works and is
  untouched; the current rejection never covered it.

## Gate repair 2026-08-21: the file's own gate reported a pass it never ran ([#1454](https://github.com/mudler/vllm.cpp/issues/1454))

This row is `DONE` and its production code is unchanged. What follows is a
repair of the row's own gate file, `tests/vllm/models/test_qwen3_5_gguf_mtp.cpp`.

### What was wrong

Both cases opened `if (path == nullptr) return;` on `VLLM_MTP_GGUF_MODEL`, and a
bare `return` from a doctest case is a PASS. Re-derived on a clean Release
CPU build at base `947e5f648`, variable unset:

```text
[doctest] test cases: 2 | 2 passed | 0 failed | 0 skipped
[doctest] assertions: 0 | 0 passed | 0 failed |
[doctest] Status: SUCCESS!            rc 0
```

Nothing printed. `VLLM_MTP_GGUF_MODEL` is set nowhere in `.github/workflows/`,
so that was the state of every CI run this row has ever had, and the
`2 cases / 18 assertions` this row recorded in `.agents/engine-matrix.md` was the
LIVE count, reachable only by a person who had the asset.

The second defect is one line. `:52-53` read:

```cpp
// them into block_count, so num_hidden_layers + depth == block_count.
CHECK(c.num_hidden_layers > 0);
```

The comment names the invariant; the assertion is true of every valid model.

### The invariant is now HERMETIC, and that is the substantive change

The old file could not check the arithmetic without a 17 GB asset, so it never
did. `HfConfigFromGguf` needs only scalars to derive the depth, so a KV-only
GGUF carrying no tensor bytes at all drives the PRODUCTION config builder in
CI. Three head-carrying arms and one head-less arm:

| `block_count` | `nextn_predict_layers` | why this arm |
|---:|---:|---|
| 65 | 1 | the shipped `Qwen3.8-27B-Q4_K_M.gguf` pair |
| 25 | 1 | the Qwen3.5-2B reference file this suite was developed against |
| 28 | 3 | separates `- nextn` from `- 1`, which the two 1-block arms cannot |
| 24 | absent | the head-less export: nothing published, whole count is the trunk |

The head-less arm is the half `NumMtpLayers` cannot express. That helper
answers 1 for an absent key on purpose (`qwen3_5_mtp.cpp:264` - it answers "how
deep is the head we are running", not "does one exist"), so
`num_hidden_layers + NumMtpLayers(c) == block_count` is FALSE on a head-less
file and the correct statement there is that `mtp_num_hidden_layers` is absent
and `num_hidden_layers == block_count`. Writing the invariant with that helper
alone would have been a second tautology in a different disguise.

The two env-gated cases stay, because no synthetic fixture can honestly stand in
for what llama.cpp's converter really emits, and they now skip with a `MESSAGE`
naming the variable in the shape `tests/vllm/entrypoints/test_gguf_mmproj_reach.cpp`
landed. The live depth case re-derives the invariant from the file's OWN
`block_count` kv rather than from a number written in the test.

### Evidence

| Run | Result |
|---|---|
| unset, repaired | 4 cases / 18 assertions / `Status: SUCCESS!` / rc 0 |
| `VLLM_MTP_GGUF_MODEL=/mnt/nas_share/checkpoints/qwen3.8-27b-gguf/Qwen3.8-27B-Q4_K_M.gguf` | 4 cases / 38 assertions / `Status: SUCCESS!` / rc 0 |

Mutation on `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:889`, each
compiled clean (`compile_rc=0`, `git diff --stat` 1 file / 1 insertion / 1
deletion) and each restored against a sha256 taken before the first mutation:

| Mutant | Repaired file | PREVIOUS file |
|---|---|---|
| `c.num_hidden_layers = block_count;` | 3/4 cases, 9/18 assertions red, exit 1 | 2/2 cases, 0 assertions, `SUCCESS!`, exit 0 |
| `c.num_hidden_layers = block_count - 1;` | 2/4 cases, 5/18 assertions red, exit 1 | not run; the first mutant already showed the file is blind |

### What this repair does NOT claim

The invariant was not globally unpinned. `tests/vllm/models/test_qwen38_27b_gguf_manifest.cpp:161`
([#821](https://github.com/mudler/vllm.cpp/issues/821) W2, `0adeb8b0e`,
2026-08-20) pins the same arithmetic hermetically for the 27B artifact on a
committed manifest, and it catches mutant A (6/6 cases and 464 assertions green
on the restored tree; 4/6 cases and 6 assertions red under the mutant). That
file belongs to a different row. What #1454 found is that THIS row's gate,
which the row's own records cite as its evidence, could not.

`src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp` is untouched.
`git blame` puts `block_count - nextn` at `1a4db5c3c` (2026-07-04) and the
`mtp_num_hidden_layers` republication at `493327b4e` (2026-07-28); both are
correct. This was a test defect, and no gate result recorded above this section
is invalidated by it.

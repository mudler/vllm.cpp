# Qwen3.8-27B: the quantized arms (Q4_K_M + `clip` mmproj, and the NVFP4 artifact)

**Rows:** `LOAD-GGUF-MMPROJ` ([`engine-matrix.md`](../engine-matrix.md)),
`QUANT-QWEN38-27B-GGUF-ARM`, `QUANT-QWEN38-27B-NVFP4-ARM`
(both [`quantization-matrix.md`](../quantization-matrix.md))
**Issue:** [#821](https://github.com/mudler/vllm.cpp/issues/821)
**Related:** [#915](https://github.com/mudler/vllm.cpp/issues/915) gated the bf16
arm of the same model and explicitly excluded these two;
[#979](https://github.com/mudler/vllm.cpp/issues/979) established which oracle
runs which arm; [#857](https://github.com/mudler/vllm.cpp/issues/857) owes the
llama.cpp gateability measurement this spec's GGUF gate depends on;
[#1185](https://github.com/mudler/vllm.cpp/issues/1185) owes the vLLM
model-run-in-a-lease that this spec's NVFP4 gate depends on;
[#809](https://github.com/mudler/vllm.cpp/issues/809) / PR
[#876](https://github.com/mudler/vllm.cpp/pull/876) owns the GGUF architecture
dispatch this spec builds on and which is still OPEN.
**Lifecycle:** `LOAD-GGUF-MMPROJ` is `PARTIAL` (W1 landed; see
[W1 outcome](#w1-outcome)). `QUANT-QWEN38-27B-GGUF-ARM` and
`QUANT-QWEN38-27B-NVFP4-ARM` are `READY`.
**Owner:** unassigned

## Why this is not optional

`AGENTS.md` §Shared seams:

> A model port includes the **quantized arms, not only bf16**. GGUF k-quants are
> a standing requirement. They are not a choice for each model. Most users run
> the quantized arms, and a quant-matched llama.cpp comparison needs them.

The bf16 arm of this exact checkpoint is gated (#915, landed `0f58cbdb5`,
4/7 prompts STRICT 16/16 with all three divergences adjudicated as exact fp32
ties). The quantized arms are what remains, and they are the arms a user can
actually run: 17.1 GB for the Q4_K_M language file against 53.8 GB for the bf16
GGUF and ~54 GB for the safetensors set. `BACKEND-GATE-CUDA-LLAMACPP`
([`backend-matrix.md`](../backend-matrix.md)) is already recorded as **blocked on
this issue** for our own Q4_K_M arm, so the debt is not only ours to notice —
another row is already waiting on it.

This is also the arm on which a llama.cpp comparison is even possible. Per #979,
vLLM at the pin `555967922` has **no in-tree GGUF reader at all** (`6635279d8`
moved it to an unpinned out-of-tree `vllm-gguf-plugin`), and SGLang's alias table
(`loader.py:2129-2142`) does not reach `qwen3_5`. So for the GGUF arm llama.cpp
is not a secondary bar beside vLLM — it is the **only** comparator, and therefore
that arm's oracle. On NVFP4 both vLLM and SGLang run the model, so there vLLM is
the mirror and the primary oracle and llama.cpp is not consulted.

## Scope

Three separable units of work. The split is argued in
[Work breakdown](#work-breakdown).

1. **`LOAD-GGUF-MMPROJ`** — teach the loader to accept a second, `clip`-architecture
   GGUF projector file beside the language file, and load the Qwen3-VL vision
   tower out of it. Model-agnostic seam work; Qwen3.8-27B is its first consumer
   and MuseGlimmer is the second.
2. **`QUANT-QWEN38-27B-GGUF-ARM`** — `Qwen3.8-27B-Q4_K_M.gguf` end to end: full
   tensor accounting, text load and greedy decode against the pinned llama.cpp,
   the MTP/`nextn` block the file ships, the multimodal legs once (1) exists, and
   this artifact's own tokenizer and chat template.
3. **`QUANT-QWEN38-27B-NVFP4-ARM`** — the `unsloth/Qwen3.8-27B-NVFP4` artifact,
   which is **not** what its name says: a compressed-tensors `mixed-precision`
   checkpoint whose FP8 group uses per-channel weight scales and dynamic
   per-token activations, a spelling no arm in this tree loads.

Out of scope: the bf16 arm (#915, done), advancing the vLLM pin, re-pinning
llama.cpp (#1003), fixing the GGUF architecture dispatch (#809 / PR #876 — this
spec depends on it and does not re-litigate it), the other 21 GGUF encodings in
the same repo, and any speed claim before the declared correctness gate passes.

## What I inspected, and what I took on trust

`AGENTS.md` requires a checkpoint claim to be verified semantically rather than
from a repo id or a remote hash, because an unauthenticated HuggingFace tree call
on a gated repo returns a fabricated `lfs.oid` (one character x64, identical for
every file). Both repos here are public, so the oids are real, but the binding
verification below is the **header parse plus data-end == file size** in every
case, not the oid.

**Inspected directly, 2026-08-18, by HTTP range read of the file's own header:**

| Artifact | Bytes | Verification |
|---|---:|---|
| `unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23d973adb629709749dc4f6756df66ef10` → `Qwen3.8-27B-Q4_K_M.gguf` | 17,106,775,008 | GGUF v3, `general.architecture = qwen35`, 866 tensors, 51 KV, align 32, header ends 10,996,704; **computed data end == 17,106,775,008 == file size** |
| same revision → `mmproj-BF16.gguf` | 931,146,432 | GGUF v3, `general.architecture = clip`, `general.type = mmproj`, 334 tensors, 35 KV; **computed data end == 931,146,432 == file size** |
| `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` → `model.safetensors` | 22,568,192,096 | safetensors header 251,128 B; **8 + header_len + max(data_offsets[1]) == 22,568,192,096 == file size**; 1953 tensors |

**Inspected directly, from bytes already on the NAS:**

| Path | Bytes | Verification |
|---|---:|---|
| `/mnt/nas_share/checkpoints/qwen3.8-27b-bf16/Qwen3.8-27B-BF16.gguf` | 53,808,281,952 | GGUF v3, `qwen35`, 851 tensors, data end == file size. The control for the Q4_K_M tensor set |
| `/mnt/nas_share/rc/ckpt/qwen3.8-27b-hf/config.json` | 4,312 | The official bf16 config, read in full |

**Taken on trust, and named as such:** `model_mtp.safetensors` (849,400,392 B) —
its 15 tensor names come from `model.safetensors.index.json`, which I read, but I
did not range-read that file's own header, so its data-end is unverified. Every
other statement below rests on a header I parsed.

**Not inspected because it no longer exists:** see the next section.

## The first finding: #821's NVFP4 revision is gone

`unsloth/Qwen3.8-27B-NVFP4` @ `a767244d27bd76589a3e3b2ab4e64032c4ebc7af`, the
revision #821 pins, **does not resolve**. The tree API answers
`{"error":"Invalid rev id"}` and
`GET /unsloth/Qwen3.8-27B-NVFP4/resolve/a767244d.../config.json` answers **HTTP
404**. `git ls-remote https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4` reports exactly
one branch, `refs/heads/main` = `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`; it
prints two lines, because `HEAD` resolves to the same commit.

This is the failure mode this tree has already recorded once, for the sibling
repo: [`porting-a-model.md`](../porting-a-model.md) §2.1 says
"`unsloth/Qwen3.6-27B-NVFP4` was silently re-quantized in place under an
unchanged name". The same publisher has now done it again on the 27B 3.8 repo,
and this time the old revision was not merely superseded — it was removed.

Two consequences, both binding:

- **The user report on #821 is not reproducible from its stated artifact.** A
  reporter hit `qwen3_5 dense: tensor not found:
  model.language_model.layers.0.linear_attn.in_proj_qkv.input_scale` at repo
  commit `07457f87c` against `a767244d...`. That checkpoint cannot be fetched
  again. The finding below re-derives the same failure from `7d6f8d4d...`, which
  can, so the report is **corroborated at a different revision** rather than
  reproduced. Do not record it as reproduced.
- **This row re-pins to `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`** and records
  the sha256 of the bytes we hold once we hold them. A repo id is not a pin here,
  and now neither is a revision unless it is also mirrored to the NAS.

## Our baseline

What exists in this tree today, measured against `origin/main` `836c13c35` and
against the artifacts' own headers rather than against any record of them. The
first reading was taken at `1dac4f9a7`. `4ee5f4a69` landed #1258 (via #1267) and
#1259, and `836c13c35` then landed #1277, which between them edited four of the
files this section anchors, so every citation below was re-derived against the
merged tree rather than carried forward. `origin/main` moved twice during this
one repair, which is why the re-anchor obligation under `## Owed` is stated as a
standing pre-landing step rather than a thing this change finished.

### The GGUF arm

#### What the file actually is (and what #821 did not know)

#821 recorded "GGUF v3, architecture `qwen35`, 866 tensors, Q4_K/Q5_K/Q6_K/Q8_0/F32".
All of that is confirmed exactly: F32 456, Q4_K 294, Q6_K 67, Q5_K 48, Q8_0 1.

What #821 did not record, and what changes the scope:

- **`qwen35.block_count = 65`, and `qwen35.nextn_predict_layers = 1`.** The BF16
  GGUF of the same model carries `block_count = 64` and 851 tensors. The
  difference is exactly 15 tensors, and they are all block 64: an entire
  full-attention block and FFN, plus `blk.64.nextn.eh_proj.weight`,
  `blk.64.nextn.enorm.weight`, `blk.64.nextn.hnorm.weight`,
  `blk.64.nextn.shared_head_norm.weight`. **The Q4_K_M file ships the MTP
  drafter.** A loader that reads `block_count` as the number of decoder layers
  will build a 65-layer model out of a 64-layer checkpoint plus a drafter, and
  every gate downstream of that is measuring the wrong graph.
- **`tokenizer.ggml.padding_token_id = 248055`**, against `248044` in the BF16
  GGUF of the same model and `pad_token_id: null` in the official HF
  `config.json`. Three artifacts of one model, three answers. That is why the
  tokenizer gate belongs to the arm rather than to the model.

#### What the mmproj actually is — and why it is loadable where MuseGlimmer's was not

`mmproj-BF16.gguf` is `clip` / `mmproj`, 334 tensors, BF16 110 + F32 224. Its KV
block matches the official `vision_config` field for field:
`clip.vision.block_count 27` = `depth 27`, `embedding_length 1152` =
`hidden_size 1152`, `feed_forward_length 4304` = `intermediate_size 4304`,
`attention.head_count 16` = `num_heads 16`, `patch_size 16`,
`projection_dim 5120` = `out_hidden_size 5120`. `clip.projector_type` is
`qwen3vl_merger` — the projector family we already implement for Qwen3-VL. There
are no deepstack tensors, which agrees with `deepstack_visual_indexes: []` in the
config: this checkpoint has no DeepStack, so that leg is **not applicable**
rather than owed.

The load-bearing detail is the patch embedding. It ships as **two** tensors,
`v.patch_embd.weight` and `v.patch_embd.weight.1` — llama.cpp's split of a
`conv3d` with `temporal_patch_size = 2` into two `conv2d` halves. That is
precisely the thing whose absence made MuseGlimmer's mmproj unloadable:
`muse_glimmer_gguf_weights.h:198-214` records that its `v.patch_embd.weight` is
ggml `ne [14,14,3,1536]`, i.e. only 588 of the 1176 input features the temporal
patch needs, and `muse_glimmer_gguf_weights.cpp:695-706` refuses it by name. The
reason — "loading it would mean inventing the temporal half of a weight" — is in
the header at `muse_glimmer_gguf_weights.h:212`, not beside the throw. **Both
halves are present here.** So this projector is loadable, and the MuseGlimmer
refusal is not precedent for refusing it — it is precedent for exactly the check
that distinguishes the two.

#### What the loader does today, and where the second file attaches

There is one production route from a `.gguf` path to a model, and it takes one
file:

- `src/vllm/entrypoints/model_loader.cpp:1678` — the GGUF branch, `dir.extension() == ".gguf"`.
- `src/vllm/entrypoints/model_loader.cpp:1679` — `vllm::GgufFile gguf = vllm::GgufFile::Open(model_dir);`, the only `GgufFile` opened for a text engine.
- `src/vllm/entrypoints/model_loader.cpp:794-804` — `HfConfigFromGgufDispatch`, an
  if-ladder with **no default**, falling through to `HfConfigFromGguf`, which
  asserts the architecture at `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:874-875`.
- `src/vllm/entrypoints/model_loader.cpp:1742` — `ModelRegistry::Load(config, ModelSource::FromGguf(gguf))`.
- `src/vllm/model_executor/models/qwen3_5_dense.cpp:89` → `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:1474` `LoadQwen3_5DenseFromGguf`.

The single-file assumption is structural, not incidental:

- `include/vllm/model_executor/models/model_registry.h:98` carries `const GgufFile* gguf = nullptr;` — **one pointer**. Safetensors gets a *vector* at `:95`. GGUF does not.
- `include/vllm/entrypoints/model_loader.h:78` `EngineParams` has no projector or mmproj field; `:300` `FromModelDir` takes one path string.
- `include/vllm/model_executor/model_loader/gguf_reader.h:122` `GgufFile::Open(const std::string&)` opens one logical file. It *does* handle sharding — `src/vllm/model_executor/model_loader/gguf_reader.cpp:519-546` `DetectSplit` parses `-NNNNN-of-MMMMM.gguf` and merges shard tensor tables — but shards of one split are not a second, differently-architected file.
- Every qwen3_5 GGUF entry point in `include/vllm/model_executor/models/qwen3_5_gguf_weights.h` takes `const GgufFile&` singular.

So a second file attaches at, minimally: `EngineParams` and `FromModelDir`
(`include/vllm/entrypoints/model_loader.h:78,300`); the open site and the model
construction (`src/vllm/entrypoints/model_loader.cpp:1679,1742`); `ModelSource`
plus `ModelSource::FromGguf` (`include/vllm/model_executor/models/model_registry.h:79-102`,
`src/vllm/model_executor/models/model_registry.cpp:222`); the two arch call sites
(`src/vllm/model_executor/models/qwen3_5_dense.cpp:89`,
`src/vllm/model_executor/models/qwen3_5_moe.cpp:86`); and the loader signatures in
`include/vllm/model_executor/models/qwen3_5_gguf_weights.h`.

Nothing in this tree loads a `clip`-architecture projector today. The two things
that look like counterexamples are not:

- `src/vllm/model_executor/models/minimax_h3_vision_gguf.cpp:72-73`
  `LoadQwen3VLVisionFromGguf` reads `visual.*` out of the **same** encoder GGUF —
  not a `clip` file, not a second file — and its only caller is
  `tests/vllm/models/test_minimax_h3.cpp:5070`. Production video load
  (`src/vllm/multimodal/minimax_h3_video.cpp:458-459`) never calls it.
- MuseGlimmer's mmproj path is a refusal (`src/vllm/model_executor/models/muse_glimmer_gguf_weights.cpp:695-706`)
  whose only caller is `tests/vllm/models/test_muse_glimmer_gguf.cpp:849`. There
  is no production call site *because there is no production path that accepts a
  second GGUF path at all*.

`src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp` has no vision handling
whatsoever: grepping it and its header case-insensitively for
`vision|visual|patch_embd|mmproj|clip|mrope|image` returns one hit, and it is
unrelated prose at `:1336`.

#### What already works, and therefore is not in scope

Every tensor dtype a Q4_K_M carries is already computed natively, on both tiers.
The two tiers differ in whether they branch on `M`, and only one of them does:

- CUDA: `src/vt/cuda/cuda_quant_dot.cu:700-713` enumerates
  IQ2_XXS/IQ3_XXS/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ2_S/IQ1_S/IQ1_XXXS, gated by
  `IsCudaKeepQuantSupported` at `:1588-1607`; Q8_0 has its own dedicated path at
  `:1659`. Here there really is **no prefill/decode split**: `LaunchGemm`
  (`:1609-1626`) sizes its grid as `m * n` warps and reads `M` nowhere else, and
  the dispatch switch that selects the encoding (`:1864-1874`) does not see `M`
  at all, so `M=1` and `M>1` enter the identical kernel.
- CPU: `src/vt/cpu/cpu_quant_dot.cpp:787-809` carries the same set plus Q4_0 and
  MXFP4. This tier **does** branch on `M`:
  `src/vt/cpu/cpu_quant_gemm.cpp:190` takes the Arm i8mm `mmla` 2x2 register tile
  only when `mmla != nullptr && m % 2 == 0 && n % 2 == 0`, and its comment
  (`:183-187`) names decode (`M=1`) as the case that falls to the portable
  `nrc == 1` path, mirroring ggml's own `num_rows_per_vec_dot` guard.

**That CPU branch is a kernel-TIER split, not a coverage split, and the
conclusion is unchanged.** No dtype gains or loses support at any `M`: both arms
end in the same `BlockVecDot` table, and the odd-`M` arm is the general one, so
every encoding this file carries is computed at every shape. So the Q4_K_M arm is
**not** blocked on kernels. What the branch does change is the speed a Q4_K_M
decode step runs at on Arm, which is a benchmarking fact for W3 rather than a
gap for W2.

So the Q4_K_M arm is **not** blocked on kernels. Q4_0 and MXFP4 would cost a
per-GEMM `cudaStreamSynchronize` (`src/vt/cuda/cuda_quant_dot.cu:1830-1836`);
neither appears in this file, so that is not a risk here either.

### The NVFP4 arm

#### The artifact is `mixed-precision`, not NVFP4

At `7d6f8d4d...`, `config.json` declares `quantization_config.format =
"mixed-precision"`, `quant_method = "compressed-tensors"`, version
`0.17.2.a20260716`, with two groups:

| Group | Format | Targets | Weights | Input activations |
|---|---|---|---|---|
| `group_0` | `float-quantized` (FP8 W8A8) | `self_attn.(q\|k\|v\|o)_proj`, `linear_attn.(in_proj_qkv\|in_proj_z\|out_proj)`, `lm_head`, `layers.(56..63).mlp.(gate\|up\|down)_proj` | 8-bit, **`strategy: channel`**, static | 8-bit, **`dynamic: true`**, `strategy: token` |
| `group_1` | `nvfp4-pack-quantized` (W4A4) | `mlp.(gate\|up\|down)_proj` | 4-bit, `group_size: 16`, `strategy: tensor_group`, `actorder: static` | 4-bit, `dynamic: "local"`, `group_size: 16` |

plus `kv_cache_scheme` (8-bit, static, per-tensor) and an `ignore` list of
**303 entries**, which is not merely "the vision tower" and must be honoured
entry for entry when a W4 implementer resolves group membership. Counted from
the same `config.json`, the 303 are:

| Count | Entry shape |
|---:|---|
| 48 | `model.language_model.layers.<i>.linear_attn` |
| 48 | `model.language_model.layers.<i>.linear_attn.norm` |
| 48 | `model.language_model.layers.<i>.linear_attn.in_proj_b` |
| 48 | `model.language_model.layers.<i>.linear_attn.in_proj_a` |
| 27 x 4 | `model.visual.blocks.<i>.attn.{qkv,proj}`, `model.visual.blocks.<i>.mlp.{linear_fc1,linear_fc2}` |
| 2 | `model.visual.merger.{linear_fc1,linear_fc2}` |
| 1 | `re:^mtp.*` |

The 48 is the GDN layer count (the other 16 of 64 are full-attention), and the
`ignore` list is **why the `IsQwen27QuantizedLinear` claim below holds**:
`in_proj_a` and `in_proj_b` are ignored while `in_proj_qkv`, `in_proj_z` and
`out_proj` are not — they are `group_0` targets. A resolver that reads the
groups but not the `ignore` list, or that treats `linear_attn.*` as one unit,
gets the GDN block exactly wrong in both directions.

The header confirms every one of those claims at the byte level:

| Tensor | dtype | shape |
|---|---|---|
| `model.language_model.layers.0.linear_attn.in_proj_qkv.weight` | F8_E4M3 | `[10240, 5120]` |
| `model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale` | **BF16** | **`[10240, 1]`** |
| `model.language_model.layers.3.self_attn.q_proj.weight_scale` | BF16 | `[12288, 1]` |
| `model.language_model.layers.3.self_attn.k_scale` | BF16 | `[1]` |
| `model.language_model.layers.0.mlp.gate_proj.weight_packed` | U8 | `[17408, 2560]` |
| `model.language_model.layers.0.mlp.gate_proj.weight_scale` | F8_E4M3 | `[17408, 320]` |
| `model.language_model.layers.0.mlp.gate_proj.weight_global_scale` | F32 | `[1]` |
| `model.language_model.layers.0.mlp.gate_proj.input_global_scale` | F32 | `[1]` |
| `model.language_model.layers.60.mlp.gate_proj.weight` | F8_E4M3 | `[17408, 5120]` |
| `lm_head.weight_scale` | BF16 | `[248320, 1]` |
| `model.visual.patch_embed.proj.weight` | BF16 | `[1152, 3, 2, 16, 16]` |

Whole-checkpoint dtype histogram: F32 336, BF16 1048, F8_E4M3 401, U8 168. The
group-size arithmetic checks out (5120 / 16 = 320 scale columns; 5120 / 2 = 2560
packed bytes). The MTP head in `model_mtp.safetensors` is 15 tensors and
unquantized.

**And the decisive count: `*.input_scale` appears ZERO times in the checkpoint.**

#### Four independent blockers, each anchored

1. **The missing `input_scale` — the reported fatal.**
   `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:503-505` routes an
   `F8_E4M3` GDN `in_proj_qkv` to `LoadFp8RawShared`
   (`src/vllm/model_executor/models/qwen3_5_weights.cpp:1196-1198`), which reaches
   `LoadFp8Raw` at `src/vllm/model_executor/models/qwen3_5_weights.cpp:447,456-458`:

   ```cpp
   r.weight_scale = ReadF32Scalar(get, proj + ".weight_scale");
   r.input_scale = ReadF32Scalar(get, proj + ".input_scale");
   r.alpha = r.input_scale * r.weight_scale;
   ```

   `ReadF32Scalar` calls the resolver immediately
   (`include/vllm/model_executor/models/dense_weight_loaders.h:164-165`), and the
   resolver throws at
   `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:828`. This is the
   **only** FP8 arm with no `has()` guard: the block-wise arm refuses an
   `input_scale` (`:467-473`), the ModelOpt NVFP4 arm presence-guards it
   (`:388-395`). Verified still present at `origin/main` `836c13c35`.

2. **The per-channel BF16 `weight_scale` is refused independently.**
   `include/vllm/model_executor/models/dense_weight_loaders.h:168` asserts
   `numel == 1` and `:172` asserts `dtype == "F32"`, and the comment above them
   (`:147-163`) names this exact case: "A per-output-channel `[out] BF16` scale passed at two bytes an element
   and was read as one float built from the first two entries." So even after (1)
   is fixed, `weight_scale` BF16 `[10240,1]` fails the count check first. `Fp8Weight`
   (`include/vllm/model_executor/models/qwen3_5_weights.h:342-355`) is three host
   floats with **no tensor-valued scale slot**, so this is a type change, not a
   read fix.

3. **A dynamic per-token activation scheme has no representation.**
   `src/vllm/model_executor/models/qwen3_5.cpp:3629` quantizes the activation with
   one static scalar (`vt::QuantFp8Static(..., w.in_proj_qkv_fp8.input_scale)`),
   and `:3548-3549` asserts the two GDN shards share it. There is no dynamic-scale
   path on this arm at all.

4. **The scheme is never read from the config.** The only `quantization_config`
   keys this arm consults are the block-wise FP8 ones —
   `src/vllm/model_executor/layers/quantization/fp8_block_quant.cpp:19-28,49-61,77-91,106-116,131-142`
   (`weight_block_size`, `quant_method`, `activation_scheme`, `ignored_layers`).
   Nothing reads `format`, `config_groups`, `targets`, `strategy`, or the
   compressed-tensors `ignore`. Detection is by tensor presence and dtype, per
   projection. A `mixed-precision` checkpoint whose group membership is a *regex
   over layer indices* (layers 56-63 FP8, 0-55 NVFP4, same module name) cannot be
   resolved that way without at least reading the groups. A generic resolver
   exists — `src/vllm/model_executor/layers/quantization/modelopt_mixed_precision.h`
   — **and no production file includes it.** `grep -rn modelopt_mixed_precision
   src/ include/ tests/` returns exactly two includes, both tests
   (`tests/vllm/model_executor/layers/quantization/test_modelopt_mixed_precision.cpp:32`
   and `test_modelopt_mixed_precision_checkpoint.cpp:25`); the only other mention
   under `src/` is a comment at
   `src/vllm/model_executor/models/voxtral_loader_internal.h:15`. Nemotron-H is
   not a counterexample and was the one this spec previously named:
   `src/vllm/model_executor/models/nemotron_h_weights.cpp` includes
   `nemotron_h.h`, `nemotron_h_loader.h`, `nvfp4_dequant.h` and `vt/unaligned.h`,
   and reads its quantization config inline.

   **This is an `AGENTS.md` §"Nothing lands dead" fact, and it has to be stated
   as one:** a 33,575-byte header whose only reachable entry points are two unit
   tests. It is a resolver that has been proven to work and never proven to be
   reached. That is exactly the failure that section names — the tests measure a
   class, not a capability — and it changes what a W4 implementer may assume,
   because "reuse the existing resolver" and "be the first production caller of
   an untried one" are different jobs with different evidence burdens.

Two adjacent facts that will bite an implementer:

- `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:698,702`
  (`IsQwen27QuantizedLinear`) returns **false** for any name containing
  `.linear_attn.in_proj_`, i.e. it declares the GDN input projections never
  quantized. That is correct for the *3.6* unsloth artifact —
  `tests/parity/hf_snapshot.h:287-299` records that one lists
  `linear_attn.in_proj_{qkv,z,a,b}` in `ignore` and ships zero `*.input_scale` —
  and **false for this one**, whose `ignore` list stops at `in_proj_a` and
  `in_proj_b` while `group_0` claims `in_proj_qkv`, `in_proj_z` and `out_proj`.
  The two unsloth 27B artifacts differ, and reasoning from the 3.6 shape is what
  produced this line.
- The GDN path never probes NVFP4 at all: `IsNvfp4Projection` is applied only to
  `out_proj` (`src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:514`),
  while `self_attn` and `mlp` do probe it (`:561`, `:594`). Correct for
  this artifact, worth stating so nobody "fixes" it.
- **A doc comment on the NVFP4 loader is stale in the direction that matters
  here, and it is NOT this row's to repair.**
  `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:162-163` says the
  on-disk `input_global_scale` "is not read", and `LoadCtNvfp4Raw` reads it
  unconditionally sixteen lines later at `:190`, refusing a zero at `:191-192`.
  It matters to W4 because this checkpoint's `group_1` ships
  `input_global_scale` on every NVFP4 projection (see the byte table above), so
  an implementer who trusts the comment will mis-plan the one half of this
  artifact that already works. The defect predates this branch and belongs to
  whoever owns that loader; naming it here keeps it from being rediscovered as
  a surprise, and this spec deliberately does not edit that file.

The NVFP4 MLP group, by contrast, is the compressed-tensors spelling the loader
already handles (`weight_packed` + `weight_scale` F8 + `weight_global_scale` +
`input_global_scale` → `LoadCtNvfp4Raw`,
`src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:164`). **The NVFP4
half of this "NVFP4" checkpoint is the half that is closest to working.** The FP8
tower is the blocker.

#### Also unread, and owed by name

`self_attn.k_scale` / `v_scale` (16 each, BF16 scalars) are the `kv_cache_scheme`
FP8 KV-cache scales. Nothing reads them. They must be either consumed or refused
by name; silently ignoring a KV-cache quantization scheme is the defect class a
token gate cannot see.

## Work breakdown

Three units, in dependency order. Each is one row, one branch and one pull
request; none of them is this change, which is the spec and the records.

| # | Unit | Row | Needs a lease? | Blocked by |
|---|---|---|---|---|
| W1 | Second-`GgufFile` plumbing + the `clip` vision loader + the two-half patch-embedding join and its refusal | `LOAD-GGUF-MMPROJ` | no | PR #876 |
| W2 | Q4_K_M manifest, 866-tensor accounting, the `nextn` block-count correction, the artifact's tokenizer and chat template | `QUANT-QWEN38-27B-GGUF-ARM` | no | — |
| W3 | Q4_K_M text / image / video token gates vs pinned llama.cpp, then the speed axes | `QUANT-QWEN38-27B-GGUF-ARM` | yes | W1, W2, #857 |
| W4 | NVFP4 re-pin, 1968-name accounting, `config_groups` resolution, per-channel FP8 scale, dynamic-activation FP8, `k_scale`/`v_scale` | `QUANT-QWEN38-27B-NVFP4-ARM` | no | — |
| W5 | NVFP4 text / image / video / MTP token gates vs pinned vLLM, then the speed axes | `QUANT-QWEN38-27B-NVFP4-ARM` | yes | W4, #1185 |

W1, W2 and W4 are the majority of the work and need no GPU, no lease and no
oracle. W3 and W5 are the only units that do.

### Why this is three rows and not one

- **`LOAD-GGUF-MMPROJ` is its own row** because it is a loader-seam change with a
  second consumer already waiting. It touches `EngineParams`, `ModelSource`, the
  registry and the CLI; it is not Qwen-specific; and MuseGlimmer's refusal
  (`muse_glimmer_gguf_weights.cpp:695-706`) becomes reachable production code the
  moment it lands, which is a change to MuseGlimmer's behaviour that a Qwen row
  must not be making. It also gates only the *multimodal* half of the GGUF arm:
  text decode from `Qwen3.8-27B-Q4_K_M.gguf` needs none of it.
- **`QUANT-QWEN38-27B-GGUF-ARM` and `QUANT-QWEN38-27B-NVFP4-ARM` are separate
  rows** because they share nothing but a model name. Different file format,
  different loader translation unit, different oracle (llama.cpp vs vLLM),
  different blockers (#857 vs #1185), different hardware needs, and the evidence
  above shows they even ship different tokenizers. Merging them would make one
  external blocker hold the other's work.
- **The tokenizer / chat-template / reasoning-parser / tool-parser gates are NOT a
  fourth row.** They are per-artifact, not per-model: `padding_token_id` is 248055
  in the Q4_K_M and 248044 in the BF16 GGUF, and the NVFP4 repo ships a
  `tokenizer.json` of 19,989,325 B and a `tokenizer_config.json` of 1,047 B
  against the official repo's 12,809,320 B and 17,928 B, with no `merges.txt`. A
  shared "surface" row would have to load both artifacts to say anything, and
  would have nothing left to assert once it did, because the two artifacts
  disagree on the answer. Each arm gates its own surface.

## Port map

### `LOAD-GGUF-MMPROJ`

Mirror the shape the safetensors path already has: `ModelSource` carries a
*vector* of safetensors shards at `model_registry.h:95` and one GGUF pointer at
`:98`. Add a second, explicitly-named optional projector pointer rather than
turning the GGUF field into a vector — a projector is a different architecture,
not another shard, and `DetectSplit` already owns the shard concept.

Discovery follows llama.cpp's user-facing convention (`--mmproj`), with an
explicit `EngineParams` field. Auto-discovery of a sibling `mmproj*.gguf` is
**deliberately not** in this row: a directory holding two unrelated models must
not silently fuse them, and the failure would be a wrong-shaped model rather than
an error.

The vision loader reads `clip.*` KV and `v.*` / `mm.*` tensors and builds the same
`multimodal::Qwen3VLVisionConfig` that
`src/vllm/model_executor/models/minimax_h3_vision_gguf.cpp:32-56` builds from
`visual.*`; that mapping is the port target, not a new design. The two-tensor
patch embedding (`v.patch_embd.weight` + `v.patch_embd.weight.1`) is joined into
the `[out, temporal*3*p*p]` operand our `conv1_linear` needs, and a checkpoint
carrying only the first half is refused by name — the MuseGlimmer condition,
enforced rather than assumed.

This row depends on PR #876 landing, because the architecture dispatch must be
able to say "this file is a `clip` projector" without falling through to
qwen3_5's assert (`qwen3_5_gguf_weights.cpp:874-875`).

### `QUANT-QWEN38-27B-GGUF-ARM`

1. A committed header-only manifest of all 866 tensors, generated the way
   `tests/vllm/models/muse_glimmer_gguf_manifest.inc` and
   `scripts/gen-muse-glimmer-gguf-manifest.py` already do it (names, dims, type
   ids; no bytes), plus a 334-tensor manifest for the projector. An accounting
   test asserts enumerated == present, zero unaccounted in both directions,
   against the manifest in CI and against the live file under an env gate. This
   is the gate this family has never had: no Qwen3.5 checkpoint-index accounting
   test exists anywhere in `tests/`.
2. **Block 64 is the drafter, not layer 64.** The loader must read
   `qwen35.nextn_predict_layers` and take `block_count - nextn_predict_layers`
   as the decoder depth, routing `blk.64.*` to the MTP head the way
   `LoadQwen3_5MTPFromGguf` already expects. A test on the manifest alone pins
   this without the weights.
3. Text greedy decode against pinned llama.cpp on the identical file.
4. Image and video legs, once `LOAD-GGUF-MMPROJ` exists.
5. This artifact's tokenizer and chat template, from its own KV block.

### `QUANT-QWEN38-27B-NVFP4-ARM`

1. Re-pin to `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`, mirror the bytes to the
   NAS, and record the locally-computed sha256. The old pin is unfetchable and a
   third re-quantization is to be expected.
2. A tensor-accounting gate over the 1968-name index, per scheme —
   `tests/vllm/models/test_nemotron_h_loader.cpp` is the template (it does exactly
   this for 18,487 tensors with a per-scheme composition assertion), and
   `tests/vllm/models/minimax_h3_nvfp4_manifest.inc` is the precedent for capturing
   the manifest by HTTP range on the header, which is how the numbers in this spec
   were obtained.
3. Read the compressed-tensors `config_groups` / `targets` / `format` rather than
   inferring the scheme from tensor dtypes, because a regex over layer indices is
   not inferable from a per-projection probe. `modelopt_mixed_precision.h` is the
   candidate to build on, and it is **test-only code today** (see the fourth
   blocker above), so treat it as a design to evaluate rather than as a
   production-proven component: read it against this checkpoint's
   `config_groups`, and if it fits, the change that adopts it is also the change
   that gives it its first production call site and the reachability evidence
   `AGENTS.md` §"Nothing lands dead" requires. If it does not fit, extend it.
   Either way the tree ends with ONE mixed-precision resolver.
4. Widen the FP8 weight scale from a host float to a resident per-channel vector.
   `qwen3_5.cpp:3557,3568-3576,3589-3590` already carries a per-column
   folded-alpha vector for the merged GDN GEMM, so the *consumer* shape exists;
   what is missing is loading one from disk.
5. Dynamic per-token activation FP8: mirror vLLM's own path for
   `activation_scheme: dynamic`, do not invent a static substitute.
6. `k_scale` / `v_scale`: consume, or refuse by name with a message naming the
   missing piece. Not silence.
7. The vision tower is bf16 and in `ignore`; the MTP head is bf16. Both load
   through existing paths and are exercised, not re-ported.

## Upstream chain

To be filled at implementation time, per `AGENTS.md` ("Cite the `file:line` that
you ported"), at the pinned revisions:

- vLLM `555967922`: `compressed_tensors` scheme resolution for a
  `mixed-precision` format, the `float-quantized` channel-weight / dynamic-token
  path, and the `nvfp4-pack-quantized` path. These are the mirror for
  `QUANT-QWEN38-27B-NVFP4-ARM` and nothing here may diverge from them.
- llama.cpp `b10451` = `10bf611e533d81f739128304991c5e133c6aebd8`: `LLM_ARCH_QWEN35`
  and `PROJECTOR_TYPE_QWEN3VL`. `backend-matrix.md` records the previous anchors
  (`src/llama-arch.cpp:41`, `tools/mtmd/clip-impl.h:330`) as read at the
  superseded local fork `237ad9b96`, so they are **owed re-anchoring** against
  `b10451` (#1003) and must not be copied forward as verified positions.

## Risks

- **A third silent re-quantization.** Already happened twice in this family. Mitigated
  by mirroring bytes to the NAS and recording a locally-computed sha256; a repo id
  is not a pin and, as this spec found, neither is a revision id.
- **The 65th block loads as a decoder layer.** The single most likely way to get a
  fluent, wrong model out of the Q4_K_M file. Mitigated by pinning
  `block_count - nextn_predict_layers` in a manifest-only test that needs no weights.
- **A tokens-only gate cannot see a dtype that is too wide.** `AGENTS.md` says so
  explicitly, and this spec is entirely about arms whose whole point is byte
  width. Every arm needs a resident-bytes assertion beside its token gate: a
  Q4_K_M arm that silently dequantizes to bf16 passes every token gate and defeats
  the purpose of the row.
- **The GGUF arm's only comparator is not gateable.** `.agents/oracles/llama-cpp.md`
  records `gateable = no` for pin `b10451`, with #857 owing the measurement. This
  is not a reason to weaken the gate; it is a dependency, recorded under `## Owed`.
- **A per-channel scale read as a scalar produces plausible wrong numbers.**
  `dense_weight_loaders.h:147-163` records this having happened, and names the
  `[out] BF16` scale read as one float built from its first two entries at
  `:151-152`. Any widening of `Fp8Weight` must keep that refusal for the arms
  that really are per-tensor.
- **Merging the FP8 tower and NVFP4 MLP work.** Layers 56-63 are FP8 and 0-55 are
  NVFP4 under the same module names. A per-projection probe that gets the boundary
  wrong is silent.

## Tests to port

Every one RED first, and each mutation-proven per `AGENTS.md` §How work gets done.

Upstream's own tests come with the change that ports the behaviour, preserving
parameters, modes, fixtures, tolerances, failure cases and the revision anchor:
vLLM's compressed-tensors scheme-resolution tests for the `mixed-precision`
format and the `float-quantized` / `nvfp4-pack-quantized` groups (W4), and its
reasoning- and tool-parser tests for this model family (W2, W4). llama.cpp is the
GGUF arm's oracle, not its mirror, so nothing is ported from it — its role is to
produce the reference tokens, and the anchors it needs are owed re-anchoring
against `b10451` (#1003).

| Test | Needs a checkpoint? | Needs a GPU? |
|---|---|---|
| Q4_K_M 866-tensor manifest accounting, both directions | committed manifest; live file env-gated | no |
| mmproj 334-tensor manifest accounting | committed manifest | no |
| `block_count - nextn_predict_layers` = decoder depth, from the manifest | committed manifest | no |
| Two-half patch-embedding join; single-half refused by name | synthetic fixture | no |
| Second-`GgufFile` plumbing through `EngineParams` → `ModelSource` → loader | synthetic fixture | no |
| NVFP4 1968-name index accounting, per scheme | committed manifest | no |
| `config_groups` / `targets` resolution, incl. the 56-63 boundary | committed `config.json` fixture | no |
| Per-channel BF16 `weight_scale` loads; per-tensor arm still refuses a vector | synthetic fixture | no |
| Dynamic-activation FP8 GDN projection loads without an `input_scale` | synthetic fixture | no |
| `k_scale` / `v_scale` consumed or refused by name | synthetic fixture | no |
| Resident-bytes assertion per arm (no silent dequant) | real checkpoint | yes |
| Q4_K_M text greedy decode vs pinned llama.cpp | real checkpoint | yes |
| Q4_K_M image + video vs pinned llama.cpp | real checkpoint | yes |
| NVFP4 text / image / video / MTP vs pinned vLLM | real checkpoint | yes |
| Tokenizer + chat template per artifact | real artifact files | no |
| Reasoning + tool parser, upstream's own tests ported | fixtures | no |

**Ten of the sixteen need neither a GPU nor a lease.** Everything that is a
header parse, a manifest comparison, a config resolution, a name-mapping refusal
or a plumbing test runs on any CPU host — which is how every number in this spec
was obtained. Only the resident-bytes assertion and the four token gates need a
leased GPU.

## Gates

**Correctness before speed, on every arm.** No throughput, latency or memory
number is accepted before that arm's declared token gate passes.

| Arm | Oracle | Why that oracle |
|---|---|---|
| Q4_K_M GGUF | llama.cpp, pin `10bf611e533d81f739128304991c5e133c6aebd8` (`b10451`) | vLLM has no in-tree GGUF at `555967922` and SGLang's alias table does not reach `qwen3_5` (#979). It is the only comparator, so it is the oracle |
| NVFP4 | vLLM, pin `555967922` | vLLM runs this format. It is the mirror and the primary oracle; llama.cpp is not consulted |

Greedy, identical artifacts, prompts, token counts, batching, concurrency and
sampling on both sides. vLLM's **production** configuration is the denominator;
never `--enforce-eager`. The ratified near-tie band applies only where the
oracle's own greedy decode is non-deterministic, and #910's lower-token-id
tie-break is expected to recur here exactly as it did on the bf16 arm.

Speed axes are recorded with values and ratios once correctness passes; an axis
below its floor stays an open gap and no ceiling is declared.

**Both gates are externally blocked today, and neither blocker is this row's to
clear:**

- llama.cpp `b10451` records `gateable = no` (`.agents/oracles/llama-cpp.md`);
  #857 owes the measurement that would make the GGUF gate runnable.
- The pinned vLLM builds, installs and imports inside an `rc` lease on `dgx:gpu0`
  (#1185, measured 2026-08-18), but **running a model is untested**, and the last
  oracle that reached that point consumed the host and rebooted the box. #1185
  names this row among the five it still blocks.

Neither blocker stops the CPU-side work in the table above. Both stop the token
gates, and until they clear those cells are `PENDING` on a named external
authority — not waived, and not silent.

## Evidence required

- The header parse of every artifact, with data-end == file size, and a
  locally-computed sha256 of the bytes we hold. Never a remote-reported hash.
- The committed manifests, and the generator that produced them.
- Red-first output for every test above, and a mutation per claimed guarantee
  with the tree restored byte-for-byte after each.
- For each token gate: the exact build and run recipe, both revisions, model
  hashes, environment, contention state, and a same-binary A/B on an idle host.
- Resident bytes per arm, so a dequant-to-bf16 fallback cannot pass as a
  quantized arm.
- `docs/USAGE.md` rows for every artifact, per `porting-a-model.md` §2.1: file
  name, size, repo **and revision**, sha256 for each quantized artifact, arm
  grouping, refused arms named, total resident size, and the fact that these are
  **third-party** quantizations by Unsloth rather than first-party releases.

## Stop conditions

- Stop if the Q4_K_M or NVFP4 artifact is re-published again under the same name:
  re-verify the header before any further work, and do not carry forward a
  measurement taken against the previous bytes.
- Stop and escalate if closing an arm would need a divergence from vLLM's
  compressed-tensors semantics. vLLM is the mirror on the NVFP4 arm; llama.cpp
  never becomes one.
- Stop before writing a second mixed-precision resolver. Adopt or extend
  `modelopt_mixed_precision.h`, or record one exact tracked exception. The
  intent is unchanged — the tree must not carry two resolvers for one format —
  but do not read this as "the existing one is proven": it is reached only from
  two tests, so adopting it is a first production wiring and owes reachability
  evidence, and finding it unfit is a legitimate outcome that the spec of the
  adopting row records rather than a reason to fork it.
- Stop before auto-discovering an mmproj beside a language file. That is a
  wrong-shaped model with no error, and it is out of scope by design.
- Do not report a token gate as passing on an oracle recorded `gateable = no`.

## The #1168 rider this branch carries, and why that is an exception

`9a1f57348` is on this branch and is **not this row's work**. It moves
`VT_GDN_OUT_BF16` from `scripts/env-doc-allowlist.txt` into
`docs/ENVIRONMENT.md`, a `GDN-MOE-BF16-OUT` ([#1168](https://github.com/mudler/vllm.cpp/issues/1168))
record repair. `AGENTS.md` §"Work happens in a worktree" narrows what counts as
a unit of work and then says the narrowing "never licenses bundling unrelated
work into one branch", so carrying it here needs an argument rather than a
silence, and the commit's own body argues the *reclassification* — why the
variable stopped being kernel-internal — and never the *bundling*.

The argument for the bundling is this. It is a separate issue, carried on this
branch by **explicit developer direction** rather than by an inference this
session made. It is a two-line record move, `+1` in `docs/ENVIRONMENT.md` and
`-1` in `scripts/env-doc-allowlist.txt`, and it touches **no surface this row
touches** — this row writes the spec, the two `quantization-matrix.md` rows, the
`engine-matrix.md` row, `.agents/issue-index.md`,
`scripts/check-agent-record.py`, `docs/FEATURES.md`, `docs/STATUS.md` and
`docs/BENCHMARKS.md`, and the intersection with those two files is empty. So the
usual cost of bundling — a reviewer who cannot tell which change a finding
belongs to, and a revert that takes the innocent half with it — is not paid
here. It was kept as **its own commit** for exactly that reason: the two remain
separately revertible by `git revert 9a1f57348`, which is the property bundling
normally destroys.

Recorded here rather than in that commit's message because the commit is now an
ancestor of three merge commits on this branch, so amending it would rewrite
published history, which this repair is not permitted to do. The commit that
adds this section carries the same argument in its own message, so the reason is
in Git history with a diff, an author and a date, as
`AGENTS.md` §"Changing the rules or a checker" requires of an exception. This is
visible debt, not a precedent: the next unrelated rider gets its own branch.

## W1 outcome

`LOAD-GGUF-MMPROJ` is implemented and `PARTIAL`. What follows is what W1
actually delivered, what it deliberately did NOT deliver, and the one place it
departs from the [Port map](#load-gguf-mmproj) above.

### What landed

- **The flag, on all three surfaces.** `EngineParams::mmproj_path`
  (`include/vllm/entrypoints/model_loader.h`), `vllm_model_params.mmproj_path`
  (`include/vllm.h`, **C ABI v22**, appended so a zero-initialised v21 struct is
  byte-identical), and `--mmproj` on the OpenAI server. The spelling is
  llama.cpp's, which is the flag a holder of these artifacts already types.
- **The reader.** `include/vllm/model_executor/models/clip_mmproj_gguf.h` +
  `src/vllm/model_executor/models/clip_mmproj_gguf.cpp`. It reads the
  projector's own `clip.*` metadata into the SAME
  `multimodal::Qwen3VLVisionConfig` that
  `minimax_h3_vision_gguf.cpp::MiniMaxH3EncoderVisionConfig` builds from
  `visual.*`, and the `v.*` / `mm.*` tensors into the SAME
  `multimodal::Qwen3VLVisionWeights`. No second tower, no second config type.
- **The temporal-patch join, and its refusal.** llama.cpp stores the `conv3d`
  patch embedding as two `conv2d` halves summed over the two frames
  (`qwen2vl.cpp::clip_graph_qwen2vl::build_inp_with_temporal_merge` at
  `b10451`), so the join INTERLEAVES them per channel into the
  `[out, C * T * p * p]` operand the tower reads. A concatenation has the same
  size and the same multiset of values, so the gate checks every position rather
  than the length. A file carrying only `v.patch_embd.weight` is refused by
  name, with both feature counts in the message.
- **MuseGlimmer's refusal, reached.** `clip.projector_type == "muse-glimmer"`
  routes to `MuseGlimmerRefuseMmproj()`, whose only caller before this row was
  `tests/vllm/models/test_muse_glimmer_gguf.cpp`. **This is a change to
  MuseGlimmer's behaviour**, made deliberately and stated here: a user who
  passes `mmproj-kquant.gguf` now gets that recorded message from the loader
  instead of getting no way to name the file at all.
- **Placement.** The projector is opened, validated and READ after the
  architecture resolve and the device-fit refusal and **before the tokenizer**,
  so a projector this build cannot load costs a message rather than a 17 GB map
  followed by one.

### Where W1's llama.cpp anchors were read

At the pin, and this is stated because the rest of this spec cannot say the
same. `backend-matrix.md`'s `LLM_ARCH_QWEN35` / `PROJECTOR_TYPE_QWEN3VL`
positions were read at the superseded local fork `237ad9b96` and are owed
re-anchoring under [#1003](https://github.com/mudler/vllm.cpp/issues/1003);
W1's are NOT those. Every `file:line` in `clip_mmproj_gguf.h`'s UPSTREAM block
was read with `git show 10bf611e5:<path>` — the commit the tag `b10451` names,
contained in `ggml-org/llama.cpp` `origin/master`, so the bytes are upstream's
at the pin and not a fork's. The commit is present in the local checkout
`/home/mudler/_git/llama.cpp-mtp-imatrix`, whose HEAD is a fork commit and was
therefore NOT the read position.

What that confirmed at the pin rather than near it:
`PROJECTOR_TYPE_NAMES` at `clip-impl.h:499` maps `PROJECTOR_TYPE_QWEN3VL`
(`:444`) to `qwen3vl_merger` (`:507`) and `PROJECTOR_TYPE_MUSE_GLIMMER`
(`:495`) to `muse-glimmer` (`:557`); the NINE `clip.*` key spellings this
reader uses are `:33,40-44,47,58,65` — the contiguous-looking `40-47` sweeps up
`KEY_N_HEAD_KV` (`:45`) and `KEY_N_EMBD_HEAD` (`:46`), which
`clip_mmproj_gguf.cpp:22-30` does NOT read, so the range is written open;
the tensor-name macros are `:104,106-108,131-132,153-155`;
the per-block reads are `clip.cpp:2021`; and
`qwen2vl.cpp:3::build_inp_with_temporal_merge` is two `ggml_conv_2d` halves
combined by `ggml_add` at `:12-26`, refusing `n_batch > 2` outright at `:28` —
which is the SUM this row's interleave join mirrors, read at the pin rather
than inferred from a fork.

### The C ABI append, and what a v21 caller does with it

`mmproj_path` is APPENDED to `vllm_model_params`, so a zero-initialised v21
struct is byte-identical to what it was and every existing caller keeps its
behaviour. The other direction is the ordinary struct-append shape and is
**not** something this row introduces: a caller COMPILED against v21, passing
its shorter struct to a v22 library, has the library read `mmproj_path` past
that allocation. The v21 `offload_config` append has exactly the same shape, as
does every append before it, because `vllm_model_params` carries no size or
version field for the library to check. It is stated here once rather than left
unstated; changing it is an ABI-wide decision about the struct, not a decision
this row may make on its own.

### What did NOT land, and is owed

- **Nothing runs the tower.** `LoadedEngine::vision_tower()` holds it; no
  forward reads it. There is no multimodal request path on the C ABI (the ABI
  v19 note in `include/vllm.h` already records this for the input limits) and no
  GGUF image/video driver. Owed by `QUANT-QWEN38-27B-GGUF-ARM` (#821 W3), and
  listed under `## Owed` below.
- **No COMMITTED manifest, and no CI accounting against one.** The reader is now
  confirmed against the real `mmproj-BF16.gguf` (see
  [The live confirmation](#the-live-confirmation) below), but that confirmation
  is env-gated and reads a file on the NAS. The committed 334-name manifest and
  the accounting test that runs in CI without the bytes belong to
  `QUANT-QWEN38-27B-GGUF-ARM`, which is where the manifest lives.
- **Deepstack is carried, and no real weights EXERCISE it.** The discovery is
  exercised synthetically. Qwen3.8-27B's projector taps nothing — its
  `clip.vision.is_deepstack_layers` is 27 `false` values and it ships no
  `v.deepstack.*` tensor, agreeing with the safetensors side's
  `deepstack_visual_indexes: []` — so the live case can only confirm the EMPTY
  answer. A projector that actually taps is owed to the first row that holds
  one.

### The one departure from the Port map, and why

The Port map says to add "a second, explicitly-named optional projector pointer"
to `ModelSource`. **W1 does not add it.** The tower is loaded by the loader and
held by `LoadedEngine`, not passed through `ModelSource` into an architecture's
weight loader.

The reason is `AGENTS.md` §"Nothing lands dead". No `Qwen3_5*Weights` has a
vision member, and no architecture's registry loader reads one: on the
safetensors side the tower is a SEPARATE reader (`LoadQwen3_5MoeVision`) over the
same shards, and it too has no production caller today. A `ModelSource::mmproj`
pointer in W1 would therefore be the "unpassed parameter" shape
[`reachability.md`](../reachability.md) names — a field the loader fills and no
loader reads — which is worse than not adding it. `LoadedEngine` is also where a
multimodal request path will look for the tower, and it is where the file
actually belongs: a projector is a separate FILE the engine was handed, not part
of the model checkpoint.

The seam is not lost. When W3 gives the tower a consumer, the consumer decides
whether it wants the tower through `ModelSource` or off the engine, and it can
add the field in the same change that reads it.

**RATIFIED by the operator on 2026-08-19.** This is a design change from the
committed spec and it is recorded as granted rather than left open. The grounds,
each re-verified in the tree at the ratifying head rather than carried over:

- No `Qwen3_5*Weights` carries a vision member —
  `Qwen3_5MoeWeights` (`qwen3_5_weights.h:635-656`) and `Qwen3_5DenseWeights`
  (`qwen3_5_dense.h:125+`) both stop at `embed_tokens` / `final_norm` /
  `lm_head` / layers, and `ModelSource` (`model_registry.h:79-102`) has fields
  for safetensors shards, one `GgufFile*` and a load queue, and no projector.
- No registry loader reads one. `LoadQwen3_5MoeVision`, the safetensors side's
  separate tower reader, has exactly FIVE call sites in the tree and every one
  of them is a test (`test_qwen3_5_moe_vision.cpp:430,459`,
  `test_qwen3_5_moe_vl_hw.cpp:124,188,221`). This bullet said "three" before the
  count was re-derived, and it named three of the five. The ratification rests
  on the absence of a PRODUCTION caller, which is what re-deriving confirmed;
  the count beside it was the part that was wrong, and it is corrected here
  rather than carried.
- So a `ModelSource::mmproj` pointer in W1 would be the "unpassed parameter"
  shape [`reachability.md`](../reachability.md) names — a field the loader fills
  and no loader reads.
- [#1358](https://github.com/mudler/vllm.cpp/issues/1358) is an OPEN bug in this
  repository for exactly that shape: Qwen3-VL loads its whole vision tower on
  the production path (`qwen3_vl.cpp:418`) into `Qwen3VLWeights::vision` and
  nothing in `src/` reads it back. Adding the field now would deliberately
  manufacture a second instance of a filed defect.
- The seam is not lost, because W3's consumer adds the field in the same change
  that reads it, which is the only way it lands reached.

### The live confirmation

W1's CI gate is synthetic and stays synthetic: CI must not depend on a 931 MB
file on a NAS share. Beside it, the same reader now runs over the artifact a
user actually holds, behind `VLLM_CPP_QWEN38_27B_MMPROJ`. Unset, the case skips
loudly; set, it adds 43 assertions to `test_clip_mmproj_gguf`.

The file is `unsloth/Qwen3.8-27B-GGUF` @
`fe1e2a23d973adb629709749dc4f6756df66ef10`, `mmproj-BF16.gguf`, 931 146 432 B,
sha256 `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53`, GGUF
v3, 334 tensors (110 BF16 + 224 F32), 35 metadata keys. Its companion language
file is `Qwen3.8-27B-Q4_K_M.gguf`, 17 106 775 008 B, sha256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`, same repo
and revision. Both sha256 values were computed locally on the mirrored copy.
Both are third-party Unsloth quantizations, not first-party releases.

**What the real bytes CONFIRMED**, none of it contradicted:

- Every one of the nine `clip.*` keys the reader spells is present under the
  spelling it reads, and so are the two `general.*` keys beside them, which the
  header read above records as `general.architecture = clip` and
  `general.type = mmproj`. `clip_mmproj_gguf.cpp:20-30` is nine `clip.*` plus
  those two `general.*`, which is where the earlier count of eleven came from,
  and `clip.*` was the wrong prefix for two of them. Eight of the nine are
  REQUIRED: `clip.projector_type`, whose absence `RefuseUnsupportedClipMmproj`
  reports as `<absent>` rather than defaulting, plus the six `ReqInt` geometry
  keys and the `ReqFloat` epsilon. Only `clip.vision.spatial_merge_size` is
  OPTIONAL, read as `OptInt` with a default of 2, so this file states it rather
  than the reader assuming it. The geometry is hidden 1152, 16 heads, 27
  blocks, feed-forward 4304, projection 5120, patch 16, spatial merge 2,
  layer-norm epsilon 1e-6.
- `in_channels = 3` and `num_position_embeddings = 2304` come off the tensor
  shapes, which is the only place the file states them, and the shapes are the
  ones the reader's checks demand: `v.patch_embd.weight` is torch
  `[1152, 3, 16, 16]` and `v.position_embd.weight` is torch `[2304, 1152]`.
- **Both patch-embedding halves ship.** `v.patch_embd.weight.1` is present with
  the identical shape, so the real artifact takes the ACCEPTING arm and the
  refusal is for a file that is not this one. Both halves are F32 on this
  artifact, so the live case reads them straight out of the mmap and checks the
  join at all 1 769 472 positions without going through the same dequant helper
  the reader uses.
- The name map closes in both directions and nothing is left over. The 334
  names the reader consumes are exactly the 334 the file ships: 27 blocks x 12,
  plus `mm.0`/`mm.2` weight and bias, `v.post_ln` weight and bias,
  `v.patch_embd` weight, `.weight.1` and bias, and `v.position_embd.weight`.
- `v.post_ln` is 1152 wide — the PRE-shuffle hidden size, not the merged 4608 —
  which is the FILE confirming `use_postshuffle_norm = false` rather than the
  reader assuming it.
- DeepStack discovery agrees with the file's own declaration: the reader finds
  no tap from the tensor names, and `clip.vision.is_deepstack_layers` is 27
  `false` values.

**What it did NOT confirm.** The tower is loaded, never run, so nothing here
says the weights produce correct activations. `general.file_type = 32` and the
BF16 tensor half are dequantized through `DequantGgufRowToF32`, which this row
does not gate. And the discovery still reads the TENSORS rather than
`clip.vision.is_deepstack_layers`; the two agree on this file and only on this
file, which is why the live case asserts both.

### Evidence

- Red first, captured by building the tree with the reader stubbed and the
  loader hook absent: `test_clip_mmproj_gguf` 8/8 cases failed (0 passed),
  `test_gguf_mmproj_reach` 4/5 failed. Green after: 8/8 and 5/5, 272 and 19
  assertions, both `Status: SUCCESS!`. Those CASE counts are the counts of that
  head, and they are 9 and 6 now: each target has since gained one env-gated
  live case, and a skipped live case contributes zero assertions, which is why
  the ASSERTION counts below are still 272 and 19. Every figure in this section
  after this bullet was measured on the tree this repair commits, over the merge
  `b1088d317` of `origin/main` `307273764`.
- The live case re-established its own red the same way, and its second
  mutation is the one that says what the live case BUYS. Turning the join into
  a concatenation reds it at `wrong == 0` (2 of 9 cases, 185 assertions).
  Renaming `ffn_up` to `ffn_gate` in the reader AND in the synthetic fixture
  together — a name-map defect the fixture agrees with — leaves the hermetic
  gate **fully green at 9/9, 272 assertions, `Status: SUCCESS!`**, and reds only
  the live case, by name: `missing tensor v.blk.0.ffn_gate.weight`. A fixture
  cannot catch a name its own author got wrong; the shipped bytes can.
  Green after both restores, verified by sha256: 9/9 at 272 assertions hermetic
  and 9/9 at 315 assertions live, `test_gguf_mmproj_reach` 5/5 at 19.
- Reachability mutation (the one `AGENTS.md` requires), RE-RUN here rather than
  quoted. Deleting the two-line `LoadQwen3VLVisionFromClipMmproj` call site at
  `model_loader.cpp:1974-1975` — `git diff --stat` `1 file changed, 2
  deletions(-)`, `COMPILE_RC=0`, so neither a mutation that never applied nor
  one that failed to build is being read as a pass — reds
  `test_gguf_mmproj_reach` and leaves `test_clip_mmproj_gguf` green, on the same
  binary pair:

  | target | env | rc | cases | assertions | `Status:` |
  |---|---|---:|---|---|---|
  | `test_gguf_mmproj_reach` | unset | 1 | 6, 5 passed / 1 failed | 19, 16 / 3 | `FAILURE!` |
  | `test_clip_mmproj_gguf` | unset | 0 | 9, 9 passed | 272 | `SUCCESS!` |
  | `test_clip_mmproj_gguf` | live | 0 | 9, 9 passed | 315 | `SUCCESS!` |

  That contrast is the evidence, not the red alone: the reader's own gate cannot
  tell the difference, on the bytes or without them. Restored with
  `git checkout --`, `src/vllm/entrypoints/model_loader.cpp` sha256
  `fb0789127a61615be31e865108d0904a7b76b6f21ac1c0ae589fedafe822cef4` before and
  after, and the rebuilt target green again at 6/6, 19.

  **A correction, and where it came from.** `647f3f194`'s body reported this
  contrast as `test_clip_mmproj_gguf` "fully green at 8/8", inside a sentence
  saying the mutation "was re-run on this head rather than inherited". The
  substance was right and the figure was the EARLIER head's: the live case had
  since made that target nine cases. The number was not re-derived when the
  sentence claiming it had been was written, which is the failure worth naming —
  a figure quoted often enough starts reading as one somebody measured. The
  table above is the current measurement and it lands in the pull-request body,
  which is what `squash_merge_commit_message = PR_BODY` makes the commit message
  on `main`.
- Guard mutations: deleting the `RefuseUnsupportedClipMmproj` call reds 2 cases;
  deleting the non-`.gguf` `--mmproj` refusal reds 1; turning the patch-embedding
  interleave into a concatenation reds 128 of the join's assertions. The tree was
  restored byte-for-byte after each, verified by sha256.
- **The tower is HELD, and that is now measured.** Every case listed above stops
  at a MESSAGE: the permitting reach case throws at the tokenizer one step past
  the projector, so no `LoadedEngine` was ever constructed and nothing in the
  tree observed the positive claim four records make. That was the finding, and
  it was measured rather than argued. Dropping the constructor's
  `vision_tower_(std::move(vision_tower))` to
  `vision_tower_(((void)vision_tower, std::nullopt))` — `git diff --stat` `1
  file changed, 1 insertion(+), 1 deletion(-)`, `COMPILE_RC=0`, because a
  mutation that fails to build and a mutation that never applied both read as a
  pass — left `test_gguf_mmproj_reach` hermetic at rc 0, 6/6, 19 assertions,
  `Status: SUCCESS!` and `test_clip_mmproj_gguf` at rc 0, 9/9, 272 assertions,
  `Status: SUCCESS!`. Every gate in this tree stayed green with the tower
  thrown away.

  So W1's repair adds
  `test_gguf_mmproj_reach.cpp::"mmproj reach: a load that COMPLETES leaves the
  tower ON THE ENGINE"`, which drives `LoadedEngine::FromModelDir` over the
  REAL pair — both files pinned by bytes and sha256 in `docs/USAGE.md` — and
  reads `vision_tower()` back off the constructed engine, then checks the
  published `vision_config()` against the projector's own header geometry and
  every weight extent against numbers derived from it (`patch_proj_w` at
  1152x1536, 27 blocks, the merger's `mm.0` at 4608x4608 and `mm.2` at
  5120x4608, no DeepStack merger) plus a non-zero energy check, so a tower of
  the right SHAPE and the wrong content fails too. It is env-gated on BOTH
  paths and skips loudly, so CI still reads no NAS file.

  | tree | env | rc | cases | assertions | `Status:` |
  |---|---|---:|---|---|---|
  | restored | unset | 0 | 6, 6 passed | 19 | `SUCCESS!` |
  | restored | live | 0 | 6, 6 passed | 232 | `SUCCESS!` |
  | tower dropped | unset | 0 | 6, 6 passed | 19 | `SUCCESS!` |
  | tower dropped | live | 1 | 6, 5 passed / 1 failed | 21, 20 / 1 | `FAILURE!` |

  The new case's own contribution is 232 - 19 = 213 assertions, and under the
  mutation it reaches only 2 of them before dying at the claim itself:
  `test_gguf_mmproj_reach.cpp:221: FATAL ERROR: REQUIRE( tower != nullptr ) is
  NOT correct!  values: REQUIRE( nullptr != nullptr )`. The mutation was
  restored with `git checkout --` and `model_loader.cpp` re-hashed to
  `fb0789127a61615be31e865108d0904a7b76b6f21ac1c0ae589fedafe822cef4`, its
  pre-mutation value.

  **What it costs to run, so the next reader can decide before starting one.**
  The load is CPU-only on this host and reads 17,106,775,008 bytes over CIFS.
  Two runs, `/usr/bin/time -v`, `Exit status: 0` both times: 6 min 21.84 s and
  5 min 36.92 s wall, 33,062,564 and 33,062,612 kB peak resident. The wall time
  is CIFS-bound and is NOT a constant, so it is given as the two values measured
  rather than as one; the peak is stable to 48 kB across them. A box with less
  than about 40 GB of available memory should not start it.

## Dependencies and blockers

Named here rather than under `## Owed`, because `## Owed` means this spec owns
the issue and each of these is owned by another row.

- [#857](https://github.com/mudler/vllm.cpp/issues/857) — the llama.cpp
  gateability measurement at pin `b10451`. The Q4_K_M token gate cannot run until
  it lands, and this row does not clear it.
- [#1185](https://github.com/mudler/vllm.cpp/issues/1185) — a demonstrated vLLM
  **model run** inside an `rc` lease. Build, install and import are measured;
  running a model is not. The NVFP4 token gate cannot run until it does.
- [#1003](https://github.com/mudler/vllm.cpp/issues/1003) — re-anchoring the
  llama.cpp `file:line` citations from the superseded local fork `237ad9b96` to
  `b10451`. This spec deliberately does not copy those anchors forward.
- [#809](https://github.com/mudler/vllm.cpp/issues/809) / PR
  [#876](https://github.com/mudler/vllm.cpp/pull/876) — the GGUF architecture
  dispatch. `LOAD-GGUF-MMPROJ` depends on it and does not duplicate it.

## Owed

Owned by this spec's three rows, and unpaid until an implementation change pays
them:

- [#821](https://github.com/mudler/vllm.cpp/issues/821) itself — its GGUF-arm
  and NVFP4-arm acceptance bullets are still open. W1 closes only the
  second-projector-file half of it.
- **A consumer for the loaded vision tower.** W1 loads it and
  `LoadedEngine::vision_tower()` holds it; no forward reads it and no C-ABI or
  server request can feed it an image. Owned by `QUANT-QWEN38-27B-GGUF-ARM`
  (W3 in the [Work breakdown](#work-breakdown)), tracked by
  [#821](https://github.com/mudler/vllm.cpp/issues/821). Named here rather than
  left to be discovered: a tower that loads and never runs is exactly the shape
  [`reachability.md`](../reachability.md) exists to keep visible.
- **The COMMITTED 334-name manifest for `mmproj-BF16.gguf`, and the CI
  accounting against it.** W1's CI gate is synthetic and its live comparison is
  env-gated on a NAS file, so nothing in CI accounts for that artifact's tensor
  set. The manifest, generated the way
  `scripts/gen-muse-glimmer-gguf-manifest.py` generates one, belongs to
  `QUANT-QWEN38-27B-GGUF-ARM`.
- **The merger and attention widths the reader never checks.** W1's reader
  validates the patch embedding's shape (`clip_mmproj_gguf.cpp:241-251`: both
  halves the same shape, `[out, in_channels, patch, patch]`) and nothing else.
  It does NOT check `mm.0` against `hidden_size * spatial_merge_size^2`, `mm.2`
  against `out_hidden_size`, or that `hidden_size % num_heads == 0`. Only the
  env-gated live case in `test_gguf_mmproj_reach.cpp` compares those, so a
  projector whose metadata and tensors disagree on any of them builds a
  wrong-shaped tower in CI silence and fails later, inside a forward, wearing
  somebody else's stack. Owned by `QUANT-QWEN38-27B-GGUF-ARM` (W3), tracked by
  [#821](https://github.com/mudler/vllm.cpp/issues/821): W3 is the row that
  gives the tower a consumer, and a refusal is worth writing where a forward
  exists to be protected.
- **The `backend-matrix.md` re-anchor at the llama.cpp pin.** W1's own
  `file:line` citations are read at `b10451` = `10bf611e5` and the reading
  position is stated in
  [Where W1's llama.cpp anchors were read](#where-w1s-llamacpp-anchors-were-read),
  so this bullet is NOT about them. `backend-matrix.md`'s
  `LLM_ARCH_QWEN35` / `PROJECTOR_TYPE_QWEN3VL` positions were read at the
  superseded local fork `237ad9b96` and are still owed re-anchoring under
  [#1003](https://github.com/mudler/vllm.cpp/issues/1003). Named here because
  W1 cites the same upstream file and a reader who finds one anchor sound will
  assume the other is.
- The mirrored bytes and a **locally computed** sha256 for `model.safetensors`
  and `model_mtp.safetensors`. Nothing in this spec records a hash it did not
  compute, and no remote-reported hash may become one.
  `Qwen3.8-27B-Q4_K_M.gguf` and `mmproj-BF16.gguf` are PAID: both are mirrored
  and both hashes are recorded in
  [The live confirmation](#the-live-confirmation) and in `docs/USAGE.md`.
- `model_mtp.safetensors`'s own header parse. Its 15 tensor names came from
  `model.safetensors.index.json`, not from its header, and this spec says so
  rather than implying otherwise.
- The `docs/USAGE.md` rows for the two SAFETENSORS artifacts, per
  [`porting-a-model.md`](../porting-a-model.md) §2.1. Owed by whichever row first
  makes an arm reachable, not by this spec. The two GGUF rows are PAID: W1 made
  `--mmproj` reachable, so it published the projector and its companion language
  file under `docs/USAGE.md` §"The exact files this was gated against", named as
  the third-party Unsloth quantizations they are.
- A re-anchor pass over the `file:line` citations on **all three** surfaces this
  change publishes them on — this spec, the two `quantization-matrix.md` rows
  (which publish clickable `#L` permalinks), and the justifying comment in
  `scripts/check-agent-record.py` — before any implementation lands. They were
  re-derived at `origin/main` `836c13c35` and a line number is stale the moment
  the file above it moves, including inside the same pull request. This bullet
  previously promised only the spec's citations, which is the narrower promise
  that let the matrix permalinks go stale while the spec was being repaired.
  `scripts/check-symbol-anchors.py` validates only `path::Symbol` citations, so
  a bare line number is checked by a reader or not at all.

## Now

`LOAD-GGUF-MMPROJ` is `PARTIAL`: W1 landed, and what it did and did not deliver
is [W1 outcome](#w1-outcome). Its blocker cleared before it started — PR #876
merged as `250db75a2`, so the GGUF architecture dispatch names an unsupported
file by its own architecture instead of falling through to qwen3_5's assert.
A user can now pass `--mmproj mmproj-BF16.gguf` beside a `.gguf` model, and the
projector is read, refused by name, or accepted with its tower held on the
engine. That is confirmed on the shipped 334-tensor artifact and not only on a
fixture ([The live confirmation](#the-live-confirmation)); CI still reads the
fixture alone. Nothing runs that tower yet, and that is `## Owed`.

`QUANT-QWEN38-27B-GGUF-ARM` and `QUANT-QWEN38-27B-NVFP4-ARM` stay `READY`, both
verified against the artifacts' own headers.

Next action is W2, the Q4_K_M manifest and accounting gate, because it needs no
lease, no oracle and no GPU, and because the `block_count - nextn_predict_layers`
correction it pins is the single most likely way to get a fluent, wrong model out
of that file. Both token gates wait on #857 and #1185 respectively, and are not
scheduled here.

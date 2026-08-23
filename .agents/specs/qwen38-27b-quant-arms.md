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
[#1632](https://github.com/mudler/vllm.cpp/issues/1632) owes the
denominator-configuration run in a lease that this spec's NVFP4 gate depends on
(it supersedes [#1185](https://github.com/mudler/vllm.cpp/issues/1185), closed
2026-08-18 as local-only);
[#809](https://github.com/mudler/vllm.cpp/issues/809) / PR
[#876](https://github.com/mudler/vllm.cpp/pull/876) owns the GGUF architecture
dispatch this spec builds on and which is still OPEN.
**Lifecycle:** `LOAD-GGUF-MMPROJ` is `PARTIAL` (W1 landed; see
[W1 outcome](#w1-outcome)). `QUANT-QWEN38-27B-GGUF-ARM` is `PARTIAL` (W2 and W3
landed; see [W2 outcome](#w2-outcome) and [W3 outcome](#w3-outcome) — W3's token
gate RAN and FAILED). `QUANT-QWEN38-27B-NVFP4-ARM` is
`PARTIAL` (W4 and W5 landed; see [W4 outcome](#w4-outcome) and
[W5 outcome](#w5-outcome)).
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
| same revision → `model_mtp.safetensors` | 849,400,392 | safetensors header 1,600 B; **8 + header_len + max(data_offsets[1]) == 849,400,392 == `x-linked-size`**; 15 tensors, all BF16. Re-read from the file's OWN header 2026-08-20 by W4, which is what the `## Owed` bullet asked for; it is no longer taken on trust from the index |
| same revision → `model.safetensors.index.json` | — | 1968 `weight_map` names = 1953 `model.safetensors` + 15 `model_mtp.safetensors`; `metadata.total_size` 23,417,592,488 == 22,568,192,096 + 849,400,392 |

**Inspected directly, from bytes already on the NAS:**

| Path | Bytes | Verification |
|---|---:|---|
| `/mnt/nas_share/checkpoints/qwen3.8-27b-bf16/Qwen3.8-27B-BF16.gguf` | 53,808,281,952 | GGUF v3, `qwen35`, 851 tensors, data end == file size. The control for the Q4_K_M tensor set |
| `/mnt/nas_share/rc/ckpt/qwen3.8-27b-hf/config.json` | 4,312 | The official bf16 config, read in full |

**Taken on trust when this spec was written, and PAID by W4:**
`model_mtp.safetensors` (849,400,392 B). Its 15 tensor names came from
`model.safetensors.index.json`; W4 range-read the file's own header on
2026-08-20 and confirms all fifteen names, dtypes and shapes and a data-end
equal to the size the hub reports. Every other statement below rests on a header
that was parsed.

**1968 AND 1953 ARE BOTH RIGHT, and they count different things.** 1953 is the
tensor count of `model.safetensors`'s own header. 1968 is the name count of the
shipped `model.safetensors.index.json` weight map, which is the 1953 plus the
MTP shard's 15. This spec's [Work breakdown](#work-breakdown) says "1968-name
accounting" and means the index; the row above says 1953 and means the one
shard. W4's gate accounts for **both**, from two committed manifests whose
counts are asserted to sum to 1968, so neither number can be quoted as the other
one's contradiction again.

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
   `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:504-506` routes an
   `F8_E4M3` GDN `in_proj_qkv` to `LoadFp8RawShared`
   (`src/vllm/model_executor/models/qwen3_5_weights.cpp:1381-1382`), which reaches
   `LoadFp8Raw` at `src/vllm/model_executor/models/qwen3_5_weights.cpp:632,641-643`.
   The three positions in this bullet were **wrong** as this spec first wrote
   them — `qwen3_5_weights.cpp:447` and `:456-458` are `OwnedBytes::Borrow`
   code, and `:1196-1198` is not `LoadFp8RawShared` — and W4 re-derived them at
   its own head. They are the `## Owed` re-anchor bullet firing exactly where it
   said it would:

   ```cpp
   r.weight_scale = ReadF32Scalar(get, proj + ".weight_scale");
   r.input_scale = ReadF32Scalar(get, proj + ".input_scale");
   r.alpha = r.input_scale * r.weight_scale;
   ```

   `ReadF32Scalar` calls the resolver immediately
   (`include/vllm/model_executor/models/dense_weight_loaders.h:164-165`), and the
   resolver throws. **W4 confirms this reproduces against the re-pinned bytes,
   from the artifact rather than from the report:** `*.input_scale` occurs
   **zero** times in `model.safetensors`'s header at `7d6f8d4d...` (168
   `*.input_global_scale` do, which is a different operand), and
   `layers.0.linear_attn.in_proj_qkv.weight` is `F8_E4M3 [10240, 5120]` with a
   `BF16 [10240, 1]` `weight_scale`, so the F8 branch is taken and asks for a
   tensor the checkpoint does not have. W4 does not let the load get that far
   any more; see [W4 outcome](#w4-outcome). This is the
   **only** FP8 arm with no `has()` guard: the block-wise arm refuses an
   `input_scale` (`:467-473`), the ModelOpt NVFP4 arm presence-guards it
   (`:388-395`). Verified still present at `origin/main` `836c13c35`.

2. **The per-channel BF16 `weight_scale` is refused independently.**
   `include/vllm/model_executor/models/dense_weight_loaders.h:168` asserts
   `numel == 1` and `:172` asserts `dtype == "F32"`, and the comment above them
   (`:147-163`) names this exact case: "A per-output-channel `[out] BF16` scale passed at two bytes an element
   and was read as one float built from the first two entries." So even after (1)
   is fixed, `weight_scale` BF16 `[10240,1]` fails the count check first. `Fp8Weight`
   (`include/vllm/model_executor/models/qwen3_5_weights.h:563-576`) is three host
   floats with **no tensor-valued scale slot**, so this is a type change, not a
   read fix.

3. **A dynamic per-token activation scheme has no representation.**
   `src/vllm/model_executor/models/qwen3_5.cpp:3629` quantizes the activation with
   one static scalar (`vt::QuantFp8Static(..., w.in_proj_qkv_fp8.input_scale)`),
   and `:3548-3549` asserts the two GDN shards share it. There is no dynamic-scale
   path on this arm at all.

4. **The scheme is never read from the config.** The only `quantization_config`
   keys this arm consults are the block-wise FP8 ones —
   `src/vllm/model_executor/layers/quantization/fp8_block_quant.cpp:27-29,50-62,78-92,107-117,132-143`
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

- `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:699,703`
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
  `out_proj` (`src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:515`),
  while `self_attn` and `mlp` do probe it (`:561`, `:594`). Correct for
  this artifact, worth stating so nobody "fixes" it.
- **A doc comment on the NVFP4 loader is stale in the direction that matters
  here, and it is NOT this row's to repair.**
  `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:163-164` says the
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
`src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:165`). **The NVFP4
half of this "NVFP4" checkpoint is the half that is closest to working.** The FP8
tower is the blocker.

#### Also unread, and owed by name

`self_attn.k_scale` / `v_scale` (16 each, BF16 scalars) are the `kv_cache_scheme`
FP8 KV-cache scales. Nothing reads them. They must be either consumed or refused
by name; silently ignoring a KV-cache quantization scheme is the defect class a
token gate cannot see.

### The second NVFP4 artifact: a ModelOpt checkpoint, not a compressed-tensors one

`r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @
`36f717a22990e82c54c1d48ee77c491b87825680` is a second published NVFP4 quant of
the same model, and the campaign
[#1574](https://github.com/mudler/vllm.cpp/issues/1574) needs it. It is not a
second revision of the artifact above and it is not the same format. Read
2026-08-21 by HTTP range request over each of its four shards' own safetensors
headers and over the three JSON documents it ships; every number below is from
those bytes.

| | `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d` | `r0b0tlab/...-MTP-sm121` @ `36f717a2` |
|---|---|---|
| `quant_method` | `compressed-tensors` | **`modelopt`** |
| declared by | `format: "mixed-precision"` | **`quant_algo: "MIXED_PRECISION"`** |
| membership | `config_groups`, regex `targets` | **`quantized_layers`, 401 EXACT module names** |
| `ignore` | 303 entries | **EMPTY** |
| FP8 weight scale | per-CHANNEL, `BF16 [out, 1]` | **per-TENSOR, `F32 []`** |
| FP8 activations | DYNAMIC per-token, zero `input_scale` | **STATIC, 208 `input_scale` `F32 []`** |
| NVFP4 half | W4A4 `nvfp4-pack-quantized`, layers 0-55 | **W4A16_NVFP4 g16, weight-only, ALL 64 layers** |
| NVFP4 spelling | `weight_packed` + `weight_scale` + `weight_global_scale` | **`weight` U8 + `weight_scale` F8_E4M3 + `weight_scale_2` F32 `[]`** |
| `lm_head` | FP8 | **W4A16_NVFP4** |
| index | 1968 names, 2 files | **2001 names, 4 files** |
| KV scales shipped | 32 (`k_scale`/`v_scale`) | **0** |

Shard verification, the same semantic check `AGENTS.md` requires in place of a
remote hash — `8 + header_len + max(data_offsets[1])` against the size the hub
reports:

| Shard | Bytes | Header | Tensors |
|---|---:|---:|---:|
| `model-00001-of-00004.safetensors` | 9,965,644,108 | 118,392 | 970 |
| `model-00002-of-00004.safetensors` | 9,985,743,924 | 122,832 | 976 |
| `model-00003-of-00004.safetensors` | 1,120,886,516 | 4,784 | 40 |
| `model-00004-of-00004.safetensors` | 849,400,592 | 1,800 | 15 |

All four match. Their sizes sum to 21,921,675,140 and the index's
`metadata.total_size` is 21,921,427,300; the difference, 247,840, is exactly the
four headers plus their four 8-byte length prefixes, so the index and the files
agree about the payload as well as about the names. The fourth shard's
`__metadata__` carries `"merge": "mtp-head-bf16-from-source"`, and its 15
tensors are the MTP head, all BF16 and none of them named by
`quantized_layers`.

**Every one of W4's four blockers is a property of the OTHER artifact, not of
the format.** This one ships the per-tensor static FP8 that `LoadFp8Raw`
already reads and the ModelOpt NVFP4 that `LoadNvfp4AnyNaming` already reads, so
its projections load. What it exposed instead is a fifth blocker W4 could not
see, because W4's artifact is refused before it: **nothing in this tree read
this config at all.** `ct::Config::FromQuantizationConfig` stops at
`quant_method != "compressed-tensors"`, so W4's whole-checkpoint read answers ""
for a ModelOpt checkpoint and every routing decision falls to the loader's
per-projection tensor-NAME probe. That probe cannot be wrong about the bytes and
it can be wrong about the checkpoint, in both directions, and both directions
load silently. W5 closes that; see [W5 outcome](#w5-outcome).

**The vLLM loader log says MXFP8, and the checkpoint says otherwise.** Running
this artifact under a vLLM image prints three `Detected ModelOpt ...` warnings,
and one of them is `Detected ModelOpt MXFP8 checkpoint`. It is not a statement
about the checkpoint. At the pin, `ModelOptMixedPrecisionConfig._from_config`
(`modelopt.py:2371-2400`) UNCONDITIONALLY constructs all four candidate
sub-configs — `ModelOptFp8Config`, two `ModelOptNvFp4Config` (`NVFP4` and
`W4A16_NVFP4`) and `ModelOptMxFp8Config` — and each warns from its own
`__init__` (`:386`, `:1035` twice, `:1708`). The lines therefore report what was
CONSTRUCTED, not what was selected; a run that prints the MXFP8 line and not the
`fp8` line at `:386` would falsify this reading, so the fp8 line is the cheap
cross-check. SELECTION is
`ModelOptMxFp8Config.override_quantization_method` (`:1724-1731`), which returns
`modelopt_mxfp8` only when the extracted algo string CONTAINS `"MXFP8"`, and
per-module dispatch is `ModelOptMixedPrecisionConfig.get_quant_method`
(`:2525-2535`), which hands out `mxfp8_config` only for a module whose
`quantized_layers` entry says MXFP8. This artifact's `quant_algo` is
`MIXED_PRECISION` and its 401 entries are 208 `FP8` and 193 `W4A16_NVFP4`, with
zero `MXFP8`. The checkpoint's own half of the argument is stronger still: MXFP8
carries an E8M0 block scale per 32 elements and this file ships no E8M0 tensor
at all — its dtype histogram is F32 609, BF16 798, F8_E4M3 401, U8 193, and its
only non-scalar scales are the 193 NVFP4 group scales at `F8_E4M3 [out, in/16]`.
Both halves are asserted in
`tests/vllm/models/test_qwen38_27b_modelopt_mtp_arm.cpp` so that a later reader
does not build an MXFP8 arm on the strength of a log line. **No MXFP8 arm is
implemented here**; W5 REFUSES a module whose declared algorithm is MXFP8, and
on this artifact that refusal never fires because no entry declares it.

**It is not the only ModelOpt `MIXED_PRECISION` checkpoint this loader sees.**
`nvidia/Qwen3.6-27B-NVFP4` @ `0893e1606ff3d5f97a441f405d5fc541a6bdf404` — the
FP8-tower gate model of #466, recorded at
[`hf_snapshot.h:287-315`](../../tests/parity/hf_snapshot.h) — declares the same
`quant_method`, the same `quant_algo`, and the same 401-entry split of 208 FP8
and 193 `W4A16_NVFP4`. Its index is 2194 names because its NVFP4 projections
also ship an `input_scale`, its `exclude_modules` is `["mtp*", "mtp.layers.0*"]`
rather than empty, and its `config.json` declares a `kv_cache_scheme` for which
it ships ZERO scales. Anything W5 adds to the load path runs on that gate model
too, which is why W5 refuses a DISAGREEMENT rather than routing by the
declaration.

## Work breakdown

Three units, in dependency order. Each is one row, one branch and one pull
request; none of them is this change, which is the spec and the records.

| # | Unit | Row | Needs a lease? | Blocked by |
|---|---|---|---|---|
| W1 | Second-`GgufFile` plumbing + the `clip` vision loader + the two-half patch-embedding join and its refusal | `LOAD-GGUF-MMPROJ` | no | PR #876 |
| W2 | Q4_K_M manifest, 866-tensor accounting, the `nextn` block-count correction, the artifact's tokenizer and chat template | `QUANT-QWEN38-27B-GGUF-ARM` | no | — |
| W3 | Q4_K_M text / image / video token gates vs pinned llama.cpp, then the speed axes | `QUANT-QWEN38-27B-GGUF-ARM` | yes | W1, W2, #857 |
| W4 | NVFP4 re-pin, 1968-name accounting, `config_groups` resolution, per-channel FP8 scale, dynamic-activation FP8, `k_scale`/`v_scale` | `QUANT-QWEN38-27B-NVFP4-ARM` | no | — |
| W5 | The LOAD side of the SECOND NVFP4 artifact, `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`: 2001-name per-scheme accounting, and the ModelOpt `MIXED_PRECISION` config read that cross-checks the loader's name probe | `QUANT-QWEN38-27B-NVFP4-ARM` | no | W4 |
| W6 | NVFP4 text / image / video / MTP token gates vs pinned vLLM, then the speed axes | `QUANT-QWEN38-27B-NVFP4-ARM` | yes | W4, W5, #1632 |

W1, W2, W4 and W5 are the majority of the work and need no GPU, no lease and no
oracle. W3 and W6 are the only units that do.

**W5 IS NEW AND THE TOKEN-GATE UNIT MOVED TO W6.** When this table was written
the NVFP4 arm was one artifact, so W4 was its whole CPU-side and the next unit
was the gate. A second artifact of the same model then arrived —
[the second NVFP4 artifact](#the-second-nvfp4-artifact-a-modelopt-checkpoint-not-a-compressed-tensors-one) —
and its load side is neither W4's work (a different format, resolved by a
different upstream module) nor the gate's (no GPU, no oracle). Renumbering was
preferred to appending a W6 that runs before W5, because a dependency order a
reader has to reconstruct from the "Blocked by" column is the thing this table
exists to spare them. The three in-file references to the old W5 moved with it;
nothing outside this file names either number.

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
  different blockers (#857 vs #1632), different hardware needs, and the evidence
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

**One of the two gates has RUN and FAILED; the other is still externally
blocked:**

- llama.cpp `b10451` records **`gateable = yes`** (`.agents/oracles/llama-cpp.md`)
  since [#857](https://github.com/mudler/vllm.cpp/issues/857) landed on
  2026-08-22, so the GGUF gate became runnable. W3 ran it on 2026-08-23 and it
  **FAILED**: tokenizer exact 6/6, generation divergent 5/6, every divergence a
  rank-2 loss under 0.18 logits. That cell is a measured open gap, not
  `PENDING`. See [W3 outcome](#w3-outcome).
- The pinned vLLM builds, installs, imports AND generates tokens inside an `rc`
  lease on `dgx:gpu0` (measured 2026-08-18 under #1185, which closed as
  local-only), but it survived at `max_num_batched_tokens` 512 and
  `max_model_len` 512 on a ~20 GiB model, where the recorded denominator for
  this family is 8192 and 2048. `AGENTS.md` §Gates requires vLLM's PRODUCTION
  configuration as the denominator, so a reduced-`mnbt` arm is a different
  engine setup rather than a smaller measurement. #1632 owns the demonstration
  that the recorded configuration survives, and the staging of this artifact's
  ~20.4 GiB.

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

### The line-anchor rot this change caused, and repaired in the same commit

Adding one `#include` to `qwen3_5_dense_weights.cpp` shifted every line below it
by one, and rewriting `QuantizationConfigOf` in `fp8_block_quant.cpp` shifted
that file. Between them, **31 files across `.agents/`, `include/`, `src/` and
`tests/` carried a bare `file:line` citation into those two files**, and every
one of them silently became wrong. They are repaired here rather than left,
because `AGENTS.md` says a record edit rides in the pull request whose change
made the record stale, and because `scripts/check-symbol-anchors.py` validates
only `path::Symbol` citations — a bare line number is checked by a reader or not
at all.

The repair is a per-line remap derived from a `difflib` opcode diff between the
merge base and this head, so it is **content-preserving by construction**: for
every remapped line the old file's line N and the new file's line M were
asserted byte-equal before the citation was rewritten. Only citations that exist
in the merge base were rewritten, so the anchors this change itself authored —
which are already head-based — were left alone. Ranges and comma lists were
remapped endpoint by endpoint.

**One surface is deliberately NOT repaired.**
[`issue-index.md`](../issue-index.md) carries three of those citations inside
issue #1411's row, and `AGENTS.md` §"Every change starts from an issue" says the
index is append-only and that a row is never edited. A stale line number inside
a closed-or-open issue row is the cost of that rule, and paying it by editing
the row would trade a wrong number for a merge conflict on every concurrent
branch. Left as is, on purpose, rather than missed.

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

## W2 outcome

`QUANT-QWEN38-27B-GGUF-ARM` is `PARTIAL`. W2 delivered the accounting for both
artifacts; W3, the token gates, is untouched and still blocked on
[#857](https://github.com/mudler/vllm.cpp/issues/857).

### What landed

- **Two committed header-only manifests, and the generator that made them.**
  `scripts/gen-qwen38-27b-gguf-manifest.py` reads a GGUF **header** — tensor
  names, ggml dims and type ids, plus the SCALAR kvs — and freezes it as a C++
  fixture. `tests/vllm/models/qwen38_27b_q4km_gguf_manifest.inc` is 866 names
  and 51 keys; `tests/vllm/models/qwen38_27b_mmproj_gguf_manifest.inc` is 334
  and 35. No weight bytes, so CI carries them and reads no file on the share.
  ARRAY kvs are deliberately not emitted: `tokenizer.ggml.tokens`, `.merges`
  and `.token_type` are 743,494 entries between them and none of them is a
  tensor-accounting fact, so freezing them would make a tensor manifest the
  place a tokenizer change has to be edited.
- **Both enumerations, beside the loaders that own them.**
  `Qwen3_5GgufExpectedTensors` / `Qwen3_5GgufAccountTensors` in
  `qwen3_5_gguf_weights.cpp` enumerate what the qwen3_5-family loaders read for
  a config — embedding and head, the trunk under its `layer_types`, and the MTP
  head blocks at `blk.{L+i}` when the checkpoint declares one — and
  `Qwen3VLClipMmprojExpectedTensors` in `clip_mmproj_gguf.cpp` does the same for
  the projector. An NVFP4 sidecar (`<stem>.scale`, `<stem>.input_scale`) is
  accounted by its stem rather than enumerated, because whether it exists is a
  property of that weight's ENCODING and not of the architecture.
- **The refusal, on the production load path.** `LoadedEngine::FromModelDir`
  refuses a qwen3_5-family GGUF, and a `--mmproj` projector, that carries
  tensors nothing reads, naming them — after the architecture resolve and the
  device-fit refusal, before the tokenizer and before any weight byte, so on the
  real artifact that is a message rather than a 17 GB map.
- **ONE direction, deliberately.** A tensor the loaders ask for and the file
  lacks already refuses by name at `GgufFile::Get`, on every arm. A tensor the
  file carries and no loader reads had no detector at all, and it is the silent
  one. Both directions are still gated, on the committed manifest, where zero
  unaccounted is asserted each way with no asset.

### The `nextn` correction: a gap that does not exist

The [Port map](#quant-qwen38-27b-gguf-arm) says the loader "must read
`qwen35.nextn_predict_layers` and take `block_count - nextn_predict_layers` as
the decoder depth". **It already does, and it has since before this spec was
written.** `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:889` is
`c.num_hidden_layers = block_count - nextn;`, landed in `1a4db5c3c`
(2026-07-04); `:897` republishes the depth as `mtp_num_hidden_layers` for the
spec resolver, landed in `493327b4e` (2026-07-28). Re-derived by `git blame`
rather than assumed, and confirmed end to end by the live case, which builds an
`HfConfig` off the real 51-key metadata and reads back `num_hidden_layers == 64`
and `NumMtpLayers == 1` against `block_count == 65`.

What was missing was a **gate**. `tests/vllm/models/test_qwen3_5_gguf_mtp.cpp`
is the only test in the tree that touches the key, and it cannot serve: it is
asset-gated on `VLLM_MTP_GGUF_MODEL`, it skips with a bare `return` rather than
loudly, so in CI it contributes nothing and says so to nobody; and where its own
comment says "num_hidden_layers + depth == block_count" its assertion is
`CHECK(c.num_hidden_layers > 0)`. A comment is not a gate. W2's hermetic case
pins the arithmetic itself, off the manifest's own kvs, with no asset:
`c.num_hidden_layers + NumMtpLayers(c) == qwen35.block_count`, plus the trunk's
composition (16 full-attention layers at every 4th index, 48 GDN) cross-checked
against the manifest's tensor NAMES, because a 64 that is 64 of the wrong kind
is the same defect one level down.

That weak sibling test is a finding this row did not repair and does not own.
It is named in `## Owed`.

### What did NOT land, and is owed

- **No token gate, and none faked.** `.agents/oracles/llama-cpp.md` records
  `gateable = no` at pin `b10451`, so the Q4_K_M text gate stays `PENDING` on a
  named external authority ([#857](https://github.com/mudler/vllm.cpp/issues/857)).
  W2 measured no throughput, latency or memory axis, so `docs/BENCHMARKS.md`
  gains nothing.
- **No resident-bytes assertion.** `## Risks` requires one per arm so a Q4_K_M
  that silently dequantizes to bf16 cannot pass a token gate. It needs the
  weights loaded, which is W3.
- **The artifact's tokenizer and chat template** (Port map item 5) are
  untouched. The manifest freezes `tokenizer.ggml.padding_token_id = 248055`
  and the gate asserts it, which pins the DIFFERENCE from the BF16 GGUF's
  248044; nothing yet loads that tokenizer.
- **The merger and attention widths the reader never checks** stay owed to W3,
  unchanged by this row: the accounting compares NAMES, not extents.

### Evidence

Every figure below was measured on the tree this change commits, on this host,
CPU-only build, and re-derived rather than carried from the session that started
this branch.

- **The artifacts, re-parsed from the mirrored bytes** rather than from any
  record of them. `Qwen3.8-27B-Q4_K_M.gguf`: GGUF v3, 866 tensors, 51 kv, header
  ends at 10,996,700, `general.architecture = qwen35`,
  `qwen35.block_count = 65`, `qwen35.nextn_predict_layers = 1`,
  `tokenizer.ggml.padding_token_id = 248055`; encodings F32 456, Q4_K 294,
  Q6_K 67, Q5_K 48, Q8_0 1; highest `blk` index 64 carrying 15 tensors, and the
  block-size histogram is 48 blocks of 14 (GDN), 16 of 11 (full attention), one
  of 15 (the drafter), plus `token_embd.weight`, `output.weight` and
  `output_norm.weight` — 48x14 + 16x11 + 15 + 3 = 866; computed data end
  17,106,775,008 == file size. `mmproj-BF16.gguf`: GGUF v3, 334 tensors, 35 kv,
  `clip` / `mmproj`, `clip.projector_type = qwen3vl_merger`,
  `clip.vision.block_count = 27`, BF16 110 + F32 224, computed data end
  931,146,432 == file size. Both files' sha256 are in `docs/USAGE.md`, computed
  locally by W1 on the same mirrored copies these numbers came off.
- **The committed manifests regenerate byte-identically** from those bytes:
  `gen-qwen38-27b-gguf-manifest.py` run over each file's header prefix produces
  output that `diff` reports as identical to the committed `.inc`. That is what
  binds the manifests to the artifact rather than to their author.
- **Red first, by mutation, because the enumerations are the change.** Stubbing
  both `*ExpectedTensors` to `return {}` — `COMPILE_RC=0`, and the stubbed
  files measure one added line each against their pre-mutation selves, so
  neither a mutation that failed to build nor one that never applied is being
  read as a pass — reds `test_qwen38_27b_gguf_manifest` at rc 1, 6 cases / 4
  passed / 2 failed, 464 assertions / 4 failed, `Status: FAILURE!`
  (`CHECK( 0 == 866 )` and `CHECK( 0 == 334 )` on the two enumerated counts),
  and `test_gguf_accounting_reach` at rc 1, 6 / 1 passed / 5 failed, 22
  assertions / 8 failed. Restored by removing the two inserted lines and
  verified by sha256 against the pre-mutation values
  (`qwen3_5_gguf_weights.cpp` `6ebf76453872ce6cb754b8809284a4c75e2a68abd8ef196d324e6f01e6c08018`,
  `clip_mmproj_gguf.cpp` `3c88eb4f0f1c92e1a4b84034a1d8cf4e77112c71df8bd433a2f27a8f5c5a2801`).
- **Green after**, same binaries: `test_qwen38_27b_gguf_manifest` rc 0, 6/6,
  464 assertions, `Status: SUCCESS!`; `test_gguf_accounting_reach` rc 0, 6/6,
  22 assertions, `Status: SUCCESS!`.
- **The live arm**, over the shipped bytes, with
  `VLLM_CPP_QWEN38_27B_GGUF` and `VLLM_CPP_QWEN38_27B_MMPROJ` both set: rc 0,
  6/6, **4745 assertions**, `Status: SUCCESS!` — 4281 more than the hermetic run,
  which is the two live cases comparing every one of the 1200 frozen names,
  ggml dims and type ids against the files' own headers. Unset, both live cases
  print a `SKIPPED:` message naming the variable and the file, so a skip is
  visible rather than a silent pass. It reads only the two headers, so it costs
  seconds and no weight byte.
- **Reachability, the mutation `AGENTS.md` requires.** Deleting BOTH refusal
  call sites in `src/vllm/entrypoints/model_loader.cpp` — 5 deleted lines,
  `COMPILE_RC=0`:

  | target | rc | cases | assertions | `Status:` |
  |---|---:|---|---|---|
  | `test_gguf_accounting_reach` | 1 | 6, 3 passed / 3 failed | 22, 13 / 9 | `FAILURE!` |
  | `test_qwen38_27b_gguf_manifest` | 0 | 6, 6 passed | 464 | `SUCCESS!` |

  The three that red are the two refusal cases and the MTP-misread case; the
  manifest target cannot tell the difference, which is the point. That contrast
  is the evidence that the accounting is a capability and not a class. Restored
  from a pre-taken copy and verified by sha256,
  `model_loader.cpp` `b10d2487f63b4e994456cc310c03556b623f23bafbe08640f624247a4e9ac7b5`
  before and after, and the rebuilt target green again at 6/6, 22.
## W3 outcome

`QUANT-QWEN38-27B-GGUF-ARM` stays `PARTIAL`. **W3 ran the Q4_K_M text token gate
against the pinned llama.cpp and the gate FAILED.** The tokenizer is exact on
6 of 6 prompts and the generation diverges on 5 of 6. No speed axis was run and
none may be quoted, because `AGENTS.md` §Gates admits a performance result only
after that arm's declared token gate passes. Full evidence:
[`docs/bench-evidence/qwen38-27b-q4km-token-gate-20260823.md`](../../docs/bench-evidence/qwen38-27b-q4km-token-gate-20260823.md).

### The blocker cleared, and the two records it left stale

[#857](https://github.com/mudler/vllm.cpp/issues/857) landed (`1a4d18b`-era,
recorded in `.agents/oracles/llama-cpp.md`), so `llama-cpp` is
**`gateable = yes`** at pin `b10451` = `10bf611e5` and this gate became
runnable. Two places in this repository still said otherwise and W3 repaired
them in the same change, because its own diff is what made them wrong to read:
this spec's [`## Gates`](#gates) blocker paragraph, and the
`QUANT-QWEN38-27B-GGUF-ARM` row in
[`quantization-matrix.md`](../quantization-matrix.md), which published
`gateable = no` beside the oracle link.

### What ran

Two `rc run` jobs on `thor:gpu0`, worker `rc-worker-kk96r`, 2026-08-23:
`64f66cda-48be-445a-85d1-49bd689306f6` built both engines and ran the gate, and
`8e0d8e54-594f-45d2-bf94-1270401bab49` ran the margin diagnosis. A third,
`0aba5d29-5b8b-4bdd-b5d6-f8fc9b5d8d1e`, removed the worker-local tree and
reclaimed 17 G. `rc devices` reported `thor:gpu0` `ready` after each, so the
resource was verified returned rather than assumed, and nothing reached the box
by `ssh`.

`dgx:gpu0` was held by another session throughout and queueing behind it is
correct rather than contending with it. `thor:gpu0` is where #857 demonstrated
this oracle at this pin, its `/workspace` is the same NAS folder holding the
staged artifact, and 14 cores with 122 GB plus 30.7 G of zram leave real headroom
over a 15.93 GiB model. `MemAvailable` was asserted at or above 40 GiB before
every model run and read 117-118 GiB each time; the two engines never ran
concurrently.

### The `blk.64` asymmetry, and how W3 handled it

`.agents/oracles/llama-cpp.md` records that `b10451` loads 851 of the 866 tensors
and ignores all 15 of `blk.64` (289,527,808 B), four of them the `nextn.*` MTP
head, so "quant-matched against the same weights" is not automatically matched
WORK. W3 re-observed the fact rather than citing it — the stock control run
emitted exactly 15 `unused tensor blk.64.*` warnings — and then removed the
asymmetry instead of annotating it.

**Our arm ran with MTP OFF, so both engines decoded the same 851 tensors.**
Neither `vllm-cli` nor `vllm-bench` was given a `--speculative-config`, and
`model_loader.cpp` calls `AttachMtpDraftWeights(LoadQwen3_5MTPFromGguf(...))`
only when `speculative_config.has_value() && method == "mtp"`, so `blk.64` is
never read on this path. The trunk is `block_count - nextn_predict_layers` =
65 - 1 = 64 layers on both sides. The [W2 outcome](#w2-outcome) gated that
arithmetic hermetically; W3 is where it is exercised on the real weights.

### The oracle side needed a harness, and here is its chain of custody

`llama-completion` at `b10451` prints token PIECES and never token ids
(`tools/completion/completion.cpp:707-710`), and `--verbose-prompt` prints the
PROMPT ids only (`:379-389`). A token-exact gate needs the generated ids, so
`oracle_tokens.cpp` is an unavoidable harness adaptation, per `AGENTS.md`
"Document only an unavoidable adaptation of the harness".

**It is not a second oracle.** It links the stock libllama built at the pin from
a byte-clean tree, calls the public `llama.h` API only, and mirrors
`completion.cpp`'s own choices: `llama_tokenize(add_special=true,
parse_special=true)` as `common_tokenize(ctx, prompt, true, true)` does at
`:279`, `llama_token_to_piece(..., special=false)` as `:707` does, and plain
argmax with first-maximum-wins, which is `llama_sampler_init_greedy`'s own tie
rule. The pinned tree's porcelain was asserted EMPTY before and after building
it.

Two independent checks bind the harness to the stock binary. The stock control
run reproduced #857's recorded six capitals **byte for byte from a different
build**, and `CHAIN_OF_CUSTODY=EXACT`: the harness's detokenized output for
prompt 0 is byte-identical to that stock stdout. So the ids the gate compares
against belong to a sequence the stock binary demonstrably produced.

### The verdict

Six raw completion prompts, no chat template, greedy on both sides, 48 tokens
each, concurrency 1, `ignore_eos` so both emit exactly 48.

`TOKENIZER_DIVERGENCES=0/6`. `examples/tokenize` on the GGUF's own vocab
produced the oracle's `PROMPT_IDS` line for line at lengths 6, 5, 6, 7, 11, 7,
and `vllm-cli` reported the same six counts through the C ABI independently.
**The #1355 prompt-token undercount does not appear on this path.**

`GENERATION_DIVERGENCES=5/6`, `TOKEN_GATE=FAIL`:

| Prompt | Verdict | First differing index | vllm.cpp | llama.cpp |
|---|---|---:|---:|---:|
| `The capital city of France is` | DIVERGE | 7 | 9338 | 9564 |
| `The three primary colors are` | DIVERGE | 34 | 198 | 3095 |
| `Water boils at a temperature of` | DIVERGE | 20 | 13 | 539 |
| `The Pythagorean theorem states that` | **TOKEN-EXACT 48/48** | — | — | — |
| `In 1969, humans first walked on` | DIVERGE | 14 | 4593 | 22486 |
| `A prime number is a natural number` | DIVERGE | 32 | 15 | 16 |

Both of our frontends give the same answer, so this is the engine and not the
harness: `vllm-cli` is a thin `include/vllm.h` client, `vllm-bench` drives the V1
`AsyncLLM`, and `vllm-cli`'s text for prompt 0 detokenizes exactly the ids
`vllm-bench --output-token-ids` recorded.

### Every divergence is a rank-2 loss under 0.18 logits

A failing gate that says only "different" names no hypothesis, so W3 measured
the shape of the difference. `oracle_margin.cpp` teacher-forces the oracle along
**vllm.cpp's own** token sequence and reports where our token ranked in the
oracle's distribution at every step. It produces no gate result.

Over all 288 steps: **`our_rank=1` on 282, `our_rank=2` on 6, and not one step
ranked 3 or worse.**

```text
MARGIN 0  7  9338 rank2 15.933463  top1  9564 15.991513  gap 0.058050  our=" France"   top1=" Germany"
MARGIN 1 34   198 rank2 19.653543  top1  3095 19.738977  gap 0.085434  our="\n"        top1=" When"
MARGIN 1 35  4350 rank2 18.953596  top1  5844 19.077843  gap 0.124247  our="When"      top1="Red"
MARGIN 2 20    13 rank2 22.465479  top1   539 22.643715  gap 0.178236  our="."         top1=" by"
MARGIN 4 14  4593 rank2 15.919413  top1 22486 16.034895  gap 0.115482  our=" heart"    top1=" satellite"
MARGIN 5 32    15 rank2 20.796000  top1    16 20.823185  gap 0.027185  our="0"         top1="1"
```

The absolute logits are 15.9 to 22.6, so the losing margins are **0.12% to
0.79%** of the logit magnitude. Prompt 3 was rank-1 on all 48 steps, which is
exactly why it is token-exact.

**This is a precision difference in the quantized compute path, not a wiring or
structural defect.** A wrong graph, a mis-routed layer or a dequant fallback
would put our token far down the oracle's ranking, repeatedly, and would not
leave 282 of 288 steps at rank 1.

### The near-tie band was NOT reached for

[`## Gates`](#gates) admits the ratified band only where the ORACLE's own greedy
decode is non-deterministic. This oracle's greedy decode reproduced #857's output
byte for byte from a different build, so it is deterministic and the premise does
not hold. A 0.058-logit loss is a small margin, not the exact fp32 tie #910
adjudicated on the bf16 arm. **The gate FAILS, and W3 did not tune, re-prompt,
shorten, or restate the comparison to make it pass.**

### The resident bytes, which answer one thing and are not the assertion

[`## Risks`](#risks) wants a resident-bytes assertion per arm so a Q4_K_M that
silently dequantizes to bf16 cannot pass a token gate. Measured on the same box
and the same file: **ours 24.997 GiB (`vllm-bench`) and 29.443 GiB (`vllm-cli`)
against llama.cpp's 30.917 GiB.** Ours is LOWER than the oracle's, and both sit
near twice the 15.93 GiB file because both repack quantized weights into a second
buffer (llama.cpp's own capability line reads `REPACK = 1`). **So there is no
dequant-to-bf16 blow-up on our side.**

That is an observation, not the assertion the spec owes. The assertion belongs
beside a passing gate, and it stays owed.

### What did NOT land, and is owed

- **The gate itself.** The Q4_K_M text arm is `FAILED`, not `PENDING`: the
  external blocker is gone, the gate ran, and it did not pass. That is a
  measured open gap and it is recorded as one.
- **No speed axis, and no memory axis.** Both were timed as a by-product and both
  are quotable as nothing. `docs/BENCHMARKS.md` gains an open gap and no number.
  The decode gap that by-product shows is large and is deliberately NOT
  attributed, because an ungated arm's throughput ranks nothing.
- **The image and video legs were not attempted.** They need a consumer for the
  vision tower W1 loads, which does not exist, and the text gate had to land
  first. `## Owed` already carries both that consumer and the merger and
  attention widths the projector reader never checks; W3 pays neither.
- **This artifact's tokenizer and chat template** (Port map item 5) remain
  untouched as a LOADED surface. W3 proves the GGUF vocab tokenizes identically
  to llama.cpp's, which is the encode half; nothing loads the artifact's chat
  template, reasoning parser or tool parser.
- **No logit vector is observable on any production path.** `vllm-bench` exposes
  generated ids and nothing exposes a distribution, so our logits and the
  oracle's cannot be diffed today. That missing instrument is what the repair
  needs first, and it is named below.

### The next traceable hypothesis

No ceiling is declared. The margin data says where to look:

1. **Compare the final logits, not the tokens.** The instrument does not exist
   and is the first thing the repair owes.
2. **Suspect the quantized dot product and the activation width first.** The
   file is Q4_K/Q5_K/Q6_K/Q8_0 and our CPU tier branches on `M`
   ([cpu_quant_gemm.cpp:190](../../src/vt/cpu/cpu_quant_gemm.cpp#L190)), taking
   the Arm i8mm `mmla` 2x2 tile only for even `M` and `N` and sending decode at
   `M=1` to the portable `nrc==1` path, while ggml uses its own repacked
   kernels. A different accumulation order or intermediate width produces
   exactly this signature: identical ordering, sub-1% logit offsets.
3. **Bisect by layer on a single forward.** One prompt, one position, our hidden
   state against llama.cpp's at each of the 64 layers, to see whether the offset
   accumulates smoothly or appears at one block. The 48 GDN (`ssm_*`) layers and
   the 16 full-attention layers are different code and must be separated.

That work is not W3's, needs its own row, issue and spec, and is listed under
`## Owed`.

## W4 outcome

`QUANT-QWEN38-27B-NVFP4-ARM` is `PARTIAL`. W4 delivered the re-pin record, the
accounting gate and the scheme resolution, and it did NOT deliver a loadable FP8
tower. The FP8 group is now **refused by name** instead of dying on a tensor the
checkpoint does not ship, which is the outcome the [Port map](#quant-qwen38-27b-nvfp4-arm)
§6 wording ("consume, or refuse by name with a message naming the missing
piece") allows and the [Stop conditions](#stop-conditions) require.

### What landed

- **The re-pin, verified from bytes this project holds.**
  `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`,
  `model.safetensors`, 22,568,192,096 B, sha256
  `c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05`, computed
  locally over the mirrored file and never read back from the hub. Header 251,128
  B, 1953 tensors, `8 + header_len + max(data_offsets[1])` == the file size.
  `model_mtp.safetensors` re-read from its own header: 15 BF16 tensors, data-end
  849,400,392. Published in `docs/USAGE.md` with the refused arm named.
- **1968 and 1953 reconciled**, above. Both counts are real and they count
  different things; the gate asserts they sum.
- **A tensor-accounting gate over the real name index, per scheme**, in
  `tests/vllm/models/test_qwen38_27b_nvfp4_arm.cpp` against two committed
  header-only manifests generated by the committed
  `scripts/gen-minimax-h3-safetensors-manifest.py` (names, dtypes, shapes; no
  weight byte). Enumerated == present with zero unaccounted in both directions,
  and a per-scheme composition that is the checkpoint's own: **466 tensors over
  233 modules in `group_0`** (FP8 W8A8), **672 over 168 in `group_1`** (NVFP4
  W4A4), **475 over 317 ignored**, **323 over 267 matched by no target**, and
  **32 over 16** `k_scale`/`v_scale`, summing to 1968 with zero unclassified.
  Hermetic in CI; a live arm behind `VLLM_CPP_QWEN38_27B_NVFP4_DIR` re-reads the
  mirrored header and compares it name by name, and SKIPS loudly when unset.
- **`config_groups` resolution, mirroring vLLM rather than probing dtypes**, in
  `src/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_config.h`.
  `ignore` first and outright, then the FIRST matching `targets` entry in
  declaration order, with `re:` meaning Python `re.match` (anchored at the start,
  not required to reach the end) and every other target an exact string compare.
  That order is load-bearing here: both groups' targets match layer 60's
  `gate_proj`, and only `group_0` being declared first puts layers 56-63 on the
  FP8 side of the boundary.
- **Upstream's own `*Attention`-only group drop, mirrored.**
  `from_config` (`compressed_tensors.py:230-246`) removes a config group whose
  targets are exactly one `*Attention` entry, because attention quantization on
  its own is coupled to the KV-cache scheme rather than to a linear method. This
  artifact declares no such group, so the branch is ported for the mirror and
  not for the artifact: keeping the group would refuse a checkpoint upstream
  loads, and no shipped tensor name would change, so the divergence would be
  silent. RED first, and the same case pins the negative half — a MULTI-target
  group whose first target ends in `Attention` is NOT dropped, because upstream's
  condition is `len(targets) == 1 AND targets[0].endswith("Attention")`.
- **The per-channel FP8 scale, the dynamic activation scheme and
  `k_scale`/`v_scale`, each refused by name through the production loader.**
  `LoadQwen3_5Dense` reads the compressed-tensors config once for the whole
  checkpoint and refuses before the first projection, naming the count of
  affected modules, the group, the group's `format`, and both missing pieces.

### The one exact tracked exception, and its argument

[Stop conditions](#stop-conditions) says to stop before writing a second
mixed-precision resolver, to adopt or extend `modelopt_mixed_precision.h`, or to
record ONE EXACT TRACKED EXCEPTION, and says finding it unfit is a legitimate
outcome. **W4 records the exception.** The two headers resolve two DIFFERENT
upstream formats that share only the English word "mixed":

| | `modelopt_mixed_precision.h` | `compressed_tensors_config.h` |
|---|---|---|
| `quant_method` | `modelopt` | `compressed-tensors` |
| Upstream module | `quantization/modelopt.py` | `quantization/compressed_tensors/` |
| Membership is declared by | a `quantized_layers` map, module prefix → algorithm string | `config_groups`, each with a `targets` list |
| Matching rule | five PREFIX strategies with `fnmatch` (`modelopt.py:2412-2505`) | ignore-first, then ORDERED FIRST REGEX MATCH (`utils.py:113-193`) |
| Exclusions | `exclude_modules` / `ignore`, prefix-matched | `ignore`, exact or `re:`, consulted BEFORE any target |

They share no key, no matching rule and no upstream module, and vLLM itself keeps
them in two files. Reading this checkpoint with the ModelOpt resolver mis-resolves
every module. So the tree does not end up with two resolvers for one format; it
ends with one resolver per format, which is upstream's structure and the one
[`porting.md`](../porting.md) requires us to mirror. What the tree does NOT get
is a second copy of "where does the quantization config live":
`fp8_block_quant.cpp` carried its own `QuantizationConfigOf` and now calls the
shared one, because that lookup really is one function.

### Upstream chain, read at the pin

Every position below was read in `/home/mudler/_git/vllm` at
`5559679229bc961848b121ccdeaa8fa5d79bec98`, the
[`upstream-sync.md`](../upstream-sync.md) pin, and each is cited in the header
that ports it:

| Upstream `file:line` | What was ported |
|---|---|
| `compressed_tensors/utils.py:50-102` | `should_ignore_layer` — `ignore` is consulted first and wins outright |
| `compressed_tensors/utils.py:105-110` | `check_equal_or_regex_match` — `any()`, so ignore-entry order is irrelevant |
| `compressed_tensors/utils.py:155-172` | `_find_first_match` — FIRST match in iteration order |
| `compressed_tensors/utils.py:175-193` | `_is_equal_or_regex_match` — `re:` means `re.match`, otherwise an exact compare |
| `compressed_tensors/compressed_tensors.py:230-265` | `from_config`, including the `*Attention`-only group drop at `:230-246` |
| `compressed_tensors/compressed_tensors.py:300-369` | `_quantization_scheme_map_from_config` — flattened in dict INSERTION order |
| `compressed_tensors/compressed_tensors.py:404-421` | `_is_nvfp4_format`, mirrored predicate for predicate |
| `compressed_tensors/compressed_tensors.py:526-560` | `_is_fp8_w8a8` |
| `compressed_tensors/compressed_tensors.py:722-743` | `_get_scheme_from_parts`, incl. the `input_quant is None` → W4A16 split and the "NVFP4 weights need NVFP4 activations" raise at `:738-742` |
| `compressed_tensors/compressed_tensors.py:810-836` | the W8A8 FP8 branch |
| `compressed_tensors/compressed_tensors.py:1034-1073` | `validate_kv_cache_scheme` |
| `compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py:53-56,127-139,152-165` | the weight STRATEGY picks the scale parameter type, and `input_scale` registers ONLY for a static input scheme |

### Reachability

The refusal is reached from `LoadQwen3_5Dense`, the loader every consumer of a
Qwen3.5-family safetensors checkpoint arrives through, and the gate enters
through that function rather than constructing the resolver by hand. Proven by
mutation: deleting the call site in a scratch copy compiles (`rc 0`, so this is
not a build failure wearing a pass), `git diff --stat` shows the file changed,
and two cases with six assertions go RED. Restored byte-for-byte against a
pre-taken sha256, green again. Two further mutations inside the resolver — group
iteration reversed, and `ignore` widened from an exact match to a prefix match —
each turn two cases red (64 and 19 assertions), and each was reverted to the same
sha256.

### What did NOT land, and is owed

- **A loadable FP8 W8A8 tower with a per-channel weight scale and dynamic
  per-token activations.** [Port map](#quant-qwen38-27b-nvfp4-arm) §4 and §5 ask
  for the scale widened to a resident per-channel vector and for vLLM's dynamic
  activation path. W4 refuses both by name instead, which §6's wording permits
  for `k_scale`/`v_scale` and which this section records explicitly for §4 and
  §5 rather than letting the refusal read as completion. It is a kernel and
  weight-representation change (`Fp8Weight` is three host floats with no
  tensor-valued scale slot), it needs a GPU to gate, and W4 was scoped to the
  CPU-side units. Owned by `QUANT-QWEN38-27B-NVFP4-ARM`, tracked by
  [#821](https://github.com/mudler/vllm.cpp/issues/821).
- **A consumed `kv_cache_scheme`.** There is no quantized KV cache on this arm to
  apply `k_scale`/`v_scale` to. Refused by name; owed by the same row and issue.
- **The resident-bytes assertion per arm**, and every token gate. Both need a
  leased GPU and W6, and the NVFP4 gate additionally waits on
  [#1632](https://github.com/mudler/vllm.cpp/issues/1632).
- **`model_mtp.safetensors` has no locally-computed sha256.** Its header was
  re-read by range request and its bytes are NOT mirrored to the NAS, so the hash
  bullet under `## Owed` is paid for `model.safetensors` only.

## W5 outcome

`QUANT-QWEN38-27B-NVFP4-ARM` stays `PARTIAL`. W5 delivered the load side of the
SECOND NVFP4 artifact and the first production consumer of the ModelOpt
resolver, and it delivered no token and no byte measurement, both of which need
a leased GPU and are W6's.

### What landed

- **The artifact, verified from its own bytes**, in
  [the second NVFP4 artifact](#the-second-nvfp4-artifact-a-modelopt-checkpoint-not-a-compressed-tensors-one):
  four shards, four header parses, four data-end-equals-file-size checks, and an
  index whose `metadata.total_size` reconciles with the four sizes to the byte.
  Published in `docs/USAGE.md` with the arms named. **No sha256 of ours**, and
  that is under `## Owed`: the bytes are ~20.4 GiB and were not mirrored where
  this unit could reach them, so a remote-reported hash was not written down as
  one.
- **A 2001-name per-scheme accounting gate, this artifact's own**, in
  `tests/vllm/models/test_qwen38_27b_modelopt_mtp_arm.cpp` against four
  committed header-only manifests generated by the committed
  `scripts/gen-minimax-h3-safetensors-manifest.py` (names, dtypes, shapes; no
  weight byte). Every name classifies, nothing is unaccounted in either
  direction, and the buckets sum to 2001: **720 tensors over 256 modules
  resolve to `FP8`**, **579 over 193 to `W4A16_NVFP4`**, **702 over 536 to
  UNLISTED**, zero to a KV-cache scale and zero unclassified. The 256 is
  separated from the 208 the map NAMES: 48 of them are
  `...layers.<i>.linear_attn` CONTAINERS that own no Linear weight and resolve
  through upstream's strategy-3 prefix scan, and the gate asserts 401 `kDirect`
  against 48 `kPrefix` so the count cannot be right for the wrong reason. The
  937 weight-bearing modules — one per `.weight` name, and exactly the ones the
  loader cross-checks — split 208 / 193 / 536. 22 cases / 1687 assertions,
  hermetic; the live re-read arm is env-gated on
  `VLLM_CPP_QWEN38_27B_MODELOPT_MTP_DIR` and skips loudly.
  **W4's 1968-name gate is untouched**, and re-ran green at 9 cases / 194
  assertions on the same tree.
- **A guard that the refusal stays SILENT on the sibling gate model's shape.**
  `nvidia/Qwen3.6-27B-NVFP4` differs from this artifact in exactly three ways
  that could each have made the refusal fire: `exclude_modules` is
  `["mtp*", "mtp.layers.0*"]` and matched by `fnmatch` rather than exactly, its
  193 NVFP4 modules ALSO ship an `input_scale`, and its `config.json` declares a
  `kv_cache_scheme` for which it ships zero scales. One case rebuilds that shape
  and asserts the answer is "". Its own mutation — `ModeloptNvfp4()` narrowed to
  reject a module that also ships an `input_scale` — turns that case red and
  leaves every other one green, so the case discriminates rather than
  decorates.
- **The first production wiring of `modelopt_mixed_precision.h`.**
  `LoadQwen3_5Dense` reads the ModelOpt `quantization_config` once for the whole
  checkpoint, through the shared `QuantizationConfigOf` lookup, and refuses by
  name when the DECLARED algorithm and the SHIPPED operand names disagree, in
  both directions, or when the declared algorithm has no loader here. Before
  this the header's only reachable entry points were its own two unit tests —
  the `AGENTS.md` §"Nothing lands dead" state W4's record named and did not
  change.
- **A second operand-suffix list, for the second format.**
  `modelopt::OperandSuffixes` carries `weight_scale_2` and `input_scale`, which
  the compressed-tensors list does not and whose absence would leave 401 of this
  artifact's 2001 names unsplittable and therefore silently unclassified.
  Widening the compressed-tensors list instead would have falsified its
  "verified complete against `unsloth/Qwen3.8-27B-NVFP4`" claim without
  measuring anything.

### Why a CROSS-CHECK and not a router

The obvious change is to route each projection by the declared algorithm instead
of by the tensor-name probe. W5 deliberately did not, and the reason is a gate
model. `nvidia/Qwen3.6-27B-NVFP4` @ `0893e160` declares 193 modules
`W4A16_NVFP4` and ALSO ships an `input_scale` on each of them, and this tree's
`VT_MODELOPT_W4A4=1` arm reads that presence to select the fp4-ACTIVATION GEMM.
Routing by the declaration would change which GEMM that A/B lever reaches on a
checkpoint every recorded 27B NVFP4 ratio was taken on. That is a measurement
change wearing a correctness change, and it needs its own row and its own
before/after. The cross-check moves no arm: a checkpoint whose config and
tensors agree loads exactly as it did.

### The refusal's blast radius, measured rather than assumed

Two checkpoints in this tree reach the new call site, and both were read from
the hub before it was written: `r0b0tlab/...-MTP-sm121` @ `36f717a2` and
`nvidia/Qwen3.6-27B-NVFP4` @ `0893e160`. On both, every one of the 937
weight-bearing modules' declaration matches its shipped spelling, so both answer
"". Every other checkpoint — compressed-tensors, `fp8`, GGUF, plain bf16 —
fails `IsMixedPrecision` at `quant_method` and the refusal reads nothing at all.

### Upstream chain, read at the pin

The resolver W5 consumes was ported by row
`MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` W1 and cites
`vllm/model_executor/layers/quantization/modelopt.py:145-181,282-367,2279-2505`
and `quantization/utils/quant_utils.py:510-572` at
`5559679229bc961848b121ccdeaa8fa5d79bec98` in its own header. W5 adds no new
upstream port: it adds the CONSUMER, and the one behaviour it adds beyond
upstream is the refusal, which is divergence 1 of that header applied to the
operand names as well as to the algorithm string. Upstream has no equivalent,
because upstream builds a `LinearMethodBase` from the declaration and never asks
what the checkpoint spells.

### Reachability

The refusal is reached from `LoadQwen3_5Dense`, and the gate enters through that
function rather than constructing the resolver by hand. Proven by mutation:
deleting the call site in a scratch copy compiles (`rc 0`, so this is not a
build failure wearing a pass), `git diff --stat` shows the file changed, and
EIGHT cases go red; restored byte-for-byte against a pre-taken sha256, green
again. Ten further mutations inside the resolver and the manifests were each
detected, and every one printed its applied diff, its compile status and its
restore hash:

| Mutation | Cases red |
|---|---|
| the production call site in `LoadQwen3_5Dense` is deleted | 8 |
| `OperandSuffixes` drops `.weight_scale_2` | 7 |
| one manifest row removed, the count literal left at 40 | 5 |
| the NVFP4 refusal branch never fires | 2 |
| an unseen operand family is skipped again | 2 |
| `StaticFp8()` accepts any spelling | 3 |
| the container skip is removed | 3 |
| the unquantized-direction branch never fires | 1 |
| the MXFP8 arm carries no reason | 1 |
| a KV-cache scale counts as a quantized weight spelling | 1 |
| `IsKvCacheScaleSuffix` stops recognising the two suffixes | 1 |
| `ModeloptNvfp4()` rejects a module that also ships `input_scale` | 1 (the sibling guard, and only it) |

The restored tree is green at 22 cases / 1687 assertions.

### What W5 did NOT deliver, and is owed

- **Any token or byte measurement on this artifact.** W6, and blocked on
  [#1632](https://github.com/mudler/vllm.cpp/issues/1632).
- **A locally computed sha256, and mirrored bytes.** Named under `## Owed`.
- **Routing by the declared algorithm.** Named under `## Owed`.
- **The FP8 KV arm.** `hf_quant_config.json` asks for `kv_cache_quant_algo:
  "FP8"` and the checkpoint ships zero `k_scale`/`v_scale`; no production path
  in this tree reads `hf_quant_config.json` at all, so the declaration is
  invisible to the loader rather than ignored by it. Owned by `KV-FP8`
  ([#1593](https://github.com/mudler/vllm.cpp/issues/1593)) and named under
  `## Owed`. W5 does not refuse it, because the identical declaration in
  `nvidia/Qwen3.6-27B-NVFP4`'s `config.json` would then refuse a gate model.

### What the fresh review found, and what the repair changed

The review mutated each claimed guarantee. Six findings, all repaired in the
same branch. None of them changes what the loader DOES for a checkpoint whose
config and tensors agree; what changed is what the gate can SEE.

- **Half the guarantee could be deleted green.** `if (false && ...)` on the
  NVFP4 refusal branch left the suite fully green at 15 cases / 1649
  assertions. That branch is the entire cross-check for the 193 `W4A16_NVFP4`
  modules — 48% of this artifact's 401 and 48% of the sibling's, `lm_head`
  included — and nothing asserted it: every case exercised the FP8, the
  unquantized, the MXFP8 or the unimplemented-algo arm, and the sibling-shape
  case pins only the direction in which the branch must NOT fire. Two cases now
  assert the refusal, one per shape a checkpoint can ship under that
  declaration, both through `LoadQwen3_5Dense`. The review's exact mutation now
  reds both.
- **A tensor family the splitter has never seen was skipped, not refused.**
  `SplitOperand` documents that `false` means "this resolver has never seen the
  family, and the caller must NOT read that as unquantized", and `Refusal` did
  read it as exactly that: the name belonged to no module, so nothing
  cross-checked it in either direction. A declared-FP8 module shipping
  `qweight`/`qzeros`/`scales` LOADED through the production entry point,
  silently. It is refused by name now. The blast radius is measured rather than
  assumed: both artifacts that reach this resolver classify EVERY name they
  ship — 2001 of 2001 and 2194 of 2194 — so the arm cannot fire on either, and
  a case asserts the first of those.
- **The KV-cache skip asserted nothing.** `Refusal` skipped `k_scale`/`v_scale`
  before any module was built, and `ModuleOperands::Add` had no branch for
  either, so the skip changed no verdict and deleting it left the suite green —
  while the sibling case carried the comment "the SUFFIXES must stay out of the
  cross-check whatever a checkpoint does with them", an assertion no assertion
  could read. The skip is gone and the decision now lives in one branch: `Add`
  RECORDS a KV scale and `AnyQuantOperand` deliberately leaves it out, because
  `kv_cache_quant_algo` is a sibling of `quantized_layers` and a bf16 tower
  shipping a KV scale must not be refused for "shipping a quantized spelling".
  Two cases hold the two directions and two mutations red them.
- **The manifest count was a hand-maintained literal that `Append` iterated**,
  so a row deleted without its count was an out-of-bounds read: a SIGSEGV
  inside case 1 that aborted the binary with 14 cases never run, and a crash is
  neither a pass nor a fail. The row count is derived with `std::size` and the
  literal is asserted against the array rather than trusted by it. The same
  mutation now reds 5 cases with all 22 having run.
- **The file header's item (2) contradicted the case below it**, giving the
  per-scheme composition as 208 / 624, 193 / 579 and 584 / 798 where the
  assertions read 256 / 720, 193 / 579 and 536 / 702 — a difference of exactly
  the 48 `linear_attn` containers and their 96 tensors, which resolve to FP8
  through the strategy-3 prefix scan rather than staying unlisted. The asserted
  set is the measured one and the prose now says so.
- **The W6 blocker citation named a closed issue.** #1185 closed on 2026-08-18
  as local-only. [#1632](https://github.com/mudler/vllm.cpp/issues/1632) files
  what actually blocks W6 — the recorded DENOMINATOR configuration surviving a
  lease, and this artifact's ~20.4 GiB being staged — and the five citations
  name it.

Two smaller things ride along. The branch was 24 commits behind `origin/main`,
which is why preflight SKIPPED its `commit-trailers` and `commit-style` gates
and reported nothing about this tree; it is merged up to `c020347a7` and both
gates now run. And new cases are one per shape rather than grouped under
`SUBCASE`, because a `REQUIRE` aborts its whole TEST_CASE and a subcase that
reds hides every subcase after it — which is how the first red-before run
reported one failure where there were three.

## Dependencies and blockers

Named here rather than under `## Owed`, because `## Owed` means this spec owns
the issue and each of these is owned by another row.

- ~~[#857](https://github.com/mudler/vllm.cpp/issues/857) — the llama.cpp
  gateability measurement at pin `b10451`.~~ **DISCHARGED 2026-08-22**;
  `.agents/oracles/llama-cpp.md` records `gateable = yes` with evidence at
  `docs/bench-evidence/oracle-llamacpp-b10451-gateable-20260822.md`. W3 then ran
  the Q4_K_M token gate and it failed on its own merits rather than on a blocker
  ([W3 outcome](#w3-outcome)).
- [#1632](https://github.com/mudler/vllm.cpp/issues/1632) — a demonstrated vLLM
  model run inside an `rc` lease **at the recorded denominator configuration**,
  and this artifact's ~20.4 GiB staged where a lease can read them. A model run
  itself is measured (2026-08-18, under the now-closed #1185) at
  `max_num_batched_tokens` 512 against a denominator of 8192. The NVFP4 token
  gate cannot run until both clear.
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
- **THE Q4_K_M TEXT GATE FAILS, and closing it needs its own row, issue and
  spec.** Measured 2026-08-23 by W3 ([W3 outcome](#w3-outcome)): tokenizer exact
  on 6/6 prompts, generation divergent on 5/6, and over 288 teacher-forced steps
  our token is the oracle's rank 1 on 282 and rank 2 on 6, never rank 3 or worse,
  losing by 0.027 to 0.178 logits against absolute logits of 15.9 to 22.6. That
  is a precision difference in the quantized compute path, not a wiring defect,
  and diagnosing it is a kernel investigation rather than a gate re-run. **No
  issue is filed for it yet**, because W3 had no authority to open one; the
  operator dispatching the repair owns filing it and linking it in
  [`issue-index.md`](../issue-index.md). Until then it is owned by
  `QUANT-QWEN38-27B-GGUF-ARM` and tracked by
  [#821](https://github.com/mudler/vllm.cpp/issues/821), whose GGUF-arm
  acceptance bullet it is.
- **No production path exposes a logit vector, so the two engines' distributions
  cannot be diffed.** `vllm-bench --output-token-ids` gives generated ids and
  nothing gives a distribution, so W3 had to teacher-force the ORACLE along our
  ids to learn anything about the margin — which measures the oracle's opinion of
  our tokens and never our own numbers. The repair above needs our logits first.
  Owned by `QUANT-QWEN38-27B-GGUF-ARM`, tracked by
  [#821](https://github.com/mudler/vllm.cpp/issues/821).
- **The resident-bytes ASSERTION per arm is still owed**, and W3 did not pay it.
  It measured the bytes — ours 24.997 GiB against the oracle's 30.917 GiB on the
  same box and the same file, so there is no dequant-to-bf16 blow-up — but an
  assertion belongs beside a passing gate, and this arm has none.
- **A consumer for the loaded vision tower.** W1 loads it and
  `LoadedEngine::vision_tower()` holds it; no forward reads it and no C-ABI or
  server request can feed it an image. Owned by `QUANT-QWEN38-27B-GGUF-ARM`
  (W3 in the [Work breakdown](#work-breakdown)), tracked by
  [#821](https://github.com/mudler/vllm.cpp/issues/821). Named here rather than
  left to be discovered: a tower that loads and never runs is exactly the shape
  [`reachability.md`](../reachability.md) exists to keep visible.
- ~~**The COMMITTED 334-name manifest for `mmproj-BF16.gguf`, and the CI
  accounting against it.**~~ **PAID by W2**, together with the 866-name manifest
  for the language file: `tests/vllm/models/qwen38_27b_mmproj_gguf_manifest.inc`
  and `qwen38_27b_q4km_gguf_manifest.inc`, generated by
  `scripts/gen-qwen38-27b-gguf-manifest.py` the way
  `scripts/gen-muse-glimmer-gguf-manifest.py` generates one, and accounted in CI
  with no asset by `tests/vllm/models/test_qwen38_27b_gguf_manifest.cpp`. See
  [W2 outcome](#w2-outcome).
- **`test_qwen3_5_gguf_mtp.cpp` claims a guarantee it does not assert, and skips
  silently.** Found by W2 and NOT repaired by it, because it belongs to
  `SPEC-MTP-GGUF` rather than to this row and repairing another row's gate is
  not a record edit this change's diff made stale. Two defects in one file: the
  asset gate returns bare (`if (path == nullptr) return;`) so an unset
  `VLLM_MTP_GGUF_MODEL` is indistinguishable from a pass, against the loud
  `MESSAGE("SKIPPED: ...")` shape W1 landed and W2 follows; and the comment
  "the trunk count must EXCLUDE the head blocks: ... num_hidden_layers + depth
  == block_count" sits above `CHECK(c.num_hidden_layers > 0)`, which is true of
  every config ever built. W2's hermetic case pins that arithmetic for the
  qwen35 GGUF path, so the family is no longer ungated, but the sibling test
  still reads as evidence it is not. **No issue is filed for it yet**, because
  W2 had no authority to open one; the operator dispatching the repair owns
  filing it and linking it in `.agents/issue-index.md`.
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
  `model.safetensors` is PAID by W4: mirrored, and sha256
  `c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05` computed
  over the bytes we hold. `model_mtp.safetensors` is still UNPAID: its header is
  now read but its bytes are not mirrored, so no hash of ours exists.
- ~~`model_mtp.safetensors`'s own header parse.~~ PAID by W4, 2026-08-20: read
  by range request over the file's own header, 15 BF16 tensors, data-end
  849,400,392 == the size the hub reports. Its 15 names, dtypes and shapes are
  the committed `tests/vllm/models/qwen38_27b_nvfp4_mtp_manifest.inc`.
- ~~The `docs/USAGE.md` rows for the two SAFETENSORS artifacts~~, per
  [`porting-a-model.md`](../porting-a-model.md) §2.1. PAID by W4, which made the
  refusal reachable and therefore published both artifacts with their repo,
  revision, bytes, the one sha256 it computed, the arm split, and the FP8 arm
  named as refused. Was owed by whichever row first made an arm reachable, not by
  this spec. The two GGUF rows are PAID: W1 made
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
- **A locally computed sha256 and mirrored bytes for
  `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a2`.** W5 verified all four
  shards semantically — header parse plus data-end equal to the size the hub
  reports, and an index `metadata.total_size` that reconciles with the four
  sizes to the byte — and computed no hash, because it never held the ~20.4 GiB.
  Nothing in this spec records a hash it did not compute, and no
  remote-reported hash may become one. Owned by
  `QUANT-QWEN38-27B-NVFP4-ARM`, tracked by
  [#821](https://github.com/mudler/vllm.cpp/issues/821).
- **The Qwen3.5 dense load path routes by TENSOR PRESENCE, and the declared
  algorithm is only cross-checked.** W5 made the declaration checkable and
  deliberately did not make it authoritative; see
  [Why a CROSS-CHECK and not a router](#why-a-cross-check-and-not-a-router).
  Two consequences are live rather than hypothetical. `VT_MODELOPT_W4A4=1`
  selects the fp4-ACTIVATION GEMM for any NVFP4 projection that happens to ship
  an `input_scale`, so on `nvidia/Qwen3.6-27B-NVFP4` @ `0893e160` it flips 193
  modules the config declares `W4A16_NVFP4` to W4A4. And
  `IsQwen27QuantizedLinear`
  (`src/vllm/model_executor/models/qwen3_5_dense_weights.cpp`, declared in
  `include/vllm/model_executor/models/qwen3_5_dense.h`) hard-codes ONE
  artifact's answer — it returns false for every `.linear_attn.in_proj_*` and
  for `lm_head`, which is right for `unsloth/Qwen3.6-27B-NVFP4` and wrong for
  both ModelOpt 27B artifacts — and has no production caller, only
  `tests/vllm/models/test_qwen27_dense_forward.cpp`. Owned by
  `QUANT-QWEN38-27B-NVFP4-ARM`, tracked by
  [#1597](https://github.com/mudler/vllm.cpp/issues/1597).
- **The FP8 KV-cache arm on the ModelOpt artifacts.**
  `r0b0tlab/...-MTP-sm121` sets `kv_cache_quant_algo: "FP8"` in
  `hf_quant_config.json` and ships ZERO `k_scale`/`v_scale`;
  `nvidia/Qwen3.6-27B-NVFP4` declares an equivalent `kv_cache_scheme` in its
  `config.json` and also ships zero. No production path in this tree reads
  `hf_quant_config.json`, and this tree has no quantized KV cache to apply
  either declaration to. Owned by `KV-FP8`, tracked by
  [#1593](https://github.com/mudler/vllm.cpp/issues/1593) — named here rather
  than refused, because refusing it would refuse a gate model this tree loads
  and measures today.

## Now

`LOAD-GGUF-MMPROJ` is `PARTIAL`: W1 landed, and what it did and did not deliver
is [W1 outcome](#w1-outcome). A user can pass `--mmproj mmproj-BF16.gguf` beside
a `.gguf` model, and the projector is read, refused by name, or accepted with
its tower held on the engine. Nothing runs that tower yet, and that is
`## Owed`.

`QUANT-QWEN38-27B-GGUF-ARM` is `PARTIAL`: W2 landed the accounting and W3 ran
the gate, and what each did and did not deliver is [W2 outcome](#w2-outcome) and
[W3 outcome](#w3-outcome). Both artifacts have a committed header-only manifest,
CI accounts each against the loaders' own enumeration with zero unaccounted in
both directions and reads no file on the share, and a load refuses either file by
name when it carries a tensor nothing reads. The `nextn` correction the
[Port map](#quant-qwen38-27b-gguf-arm) asks for was already in the loader and was
ungated; it is gated now.

**The Q4_K_M text token gate has RUN and it FAILED.** The engine loads the real
17.1 GB artifact on CPU, decodes fluent text, and tokenizes byte-identically to
llama.cpp on 6 of 6 prompts through three independent paths; the generation
diverges on 5 of 6. Every divergence is a rank-2 loss to the oracle's own top-1
by 0.027 to 0.178 logits, and over 288 teacher-forced steps our token is the
oracle's rank 1 on 282 and never ranks 3 or worse — a precision difference in the
quantized compute path rather than a wiring defect. The near-tie band was not
reached for, because the oracle's greedy decode is deterministic and the band's
premise does not hold. That cell is a measured OPEN GAP, not `PENDING` on
anybody, and no speed or memory number from this arm is admissible until it
closes.

`QUANT-QWEN38-27B-NVFP4-ARM` is `PARTIAL`: W4 and W5 landed, and what each did
and did not deliver is [W4 outcome](#w4-outcome) and
[W5 outcome](#w5-outcome). The row now covers TWO published NVFP4 artifacts of
the same model, and they are different formats.

`unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d` is re-pinned and mirrored with a
locally computed sha256, its 1968 index names are accounted per scheme in CI
from committed header-only manifests, and its FP8 group — 233 modules,
including every attention and GDN projection and `lm_head` — is refused by name
at load with both missing pieces stated, instead of dying on
`tensor not found: ...input_scale` for a tensor the checkpoint correctly does
not ship. Its NVFP4 half loads. Its FP8 tower still does not.

`r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a2` LOADS, both halves. Its FP8
is per-tensor and static and its NVFP4 is ModelOpt-spelled W4A16, which are the
two spellings this loader already reads, so none of W4's four blockers applies
to it — they were properties of the other artifact. Its 2001 index names are
accounted per scheme in CI from four committed header-only manifests, and
`LoadQwen3_5Dense` now READS its ModelOpt `quantization_config` and refuses by
name when the declared algorithm and the shipped tensor names disagree in either
direction. That is the first production consumer the resolver at
`src/vllm/model_executor/layers/quantization/modelopt_mixed_precision.h` has
ever had.

**Every CPU-side unit of this spec has landed, and W3 has now run.** What
remains is closing W3's failure and running W6. They are not the same kind of
work: **W3 is no longer blocked on anybody** — #857 discharged, the gate ran, and
it failed on the engine's own arithmetic, so the next step is a kernel
investigation that needs its own row, issue and spec (see `## Owed`). W6 is still
blocked on a named external authority rather than on work this spec can
schedule:
[#1632](https://github.com/mudler/vllm.cpp/issues/1632) (the pinned vLLM runs a
model inside an `rc` lease, and the recorded DENOMINATOR configuration is what
has not been shown to survive there).

**The next action on this spec is W3's divergence**, because it is the only
remaining unit with no external blocker and it holds an acceptance bullet of
[#821](https://github.com/mudler/vllm.cpp/issues/821) shut. It needs an
instrument this tree does not have — a logit vector off a production path — and
then a layer bisection against llama.cpp, both listed under `## Owed`. Behind it
sit the FP8 tower of the unsloth artifact, which W4 refuses by name and lists
under
[What did NOT land, and is owed](#what-did-not-land-and-is-owed-2), and W6.
# Nemotron-H — the first hybrid Mamba2 model, and the first MIXED_PRECISION checkpoint

**Claim:** `CLAIM-MODEL-NEMOTRON-H`. **Model row:**
`MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` (existing, stays `INVENTORIED`
at this spec commit). **Issue:** [#517](https://github.com/mudler/vllm.cpp/issues/517).

**Hard blocker:** `KERNEL-SSM-MAMBA` [#496](https://github.com/mudler/vllm.cpp/issues/496)
([spec](mamba2-ssd.md)). W1 (CPU host references) is in fresh review on
`row/KERNEL-SSM-MAMBA-SSD-W1`; the CUDA arm is W2 of that row. **No forward
path in this spec can be gated before that lands**, and W3 below says so
explicitly rather than pretending otherwise.

**Base:** `main` HEAD `66deca15`.
**Pinned oracle:** `${VLLM_SOURCE}` = `/home/mudler/_git/vllm` @
`5559679229bc961848b121ccdeaa8fa5d79bec98` (vLLM 0.26.0.dev0).
**Driver checkpoint:** `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`
at pinned revision `29f2d1746d8f41e316523194b19018707749b1b1`, staged on the
NAS at `$CHECKPOINT_ROOT/nemotron-3.5-lightning-30b-nvfp4` (20.1 GiB, fits one
GB10).

---

## 0. Scope (headline verdict)

Two things make this arch more than "another model file", and both are the
first of their kind in this tree:

1. **It is the first true hybrid Mamba2 model.** 52 layers: 23 Mamba2, 6 GQA
   attention, 23 MoE. Our GDN hybrids (Qwen3.5, Kimi-Linear) supply the het-KV
   machinery, but not the recurrence — see [mamba2-ssd.md](mamba2-ssd.md).
2. **It is the first `MIXED_PRECISION` checkpoint.** One file carries NVFP4
   W4A16 group-16 experts, FP8 W8A8 static mamba projections, bf16 attention,
   and an fp8 KV scheme. We have never resolved a quant algorithm *per module*;
   `quantization_config` is read ad-hoc in exactly two weight files today
   (`kimi_k3_weights.cpp:171`, `deepseek_v2_weights.cpp:365`).

A third thing is smaller but has no local precedent either: the MoE is
**non-gated**. There is no `gate_proj` anywhere in the checkpoint — the expert
is `up_proj -> relu² -> down_proj`. Every grouped-MoE op we have is
SwiGLU-shaped.

**Out of scope, explicitly:** `NemotronHPuzzleForCausalLM` heterogeneous
per-layer configs (`get_nemotron_h_config_for_layer`), `moe_latent_size`
(null in this checkpoint, so `fc1_latent_proj`/`fc2_latent_proj` are absent),
TP sharding of `n_groups`, ReplaySSM, and any speed claim. Correctness first,
always; no ratio is recorded by this row until the token gate passes.

## 1. The checkpoint, exactly

`architectures: ["NemotronHForCausalLM"]`, `model_type: nemotron_h`,
`hidden_size=2688`, `vocab_size=131072`, `max_position_embeddings=1048576`,
`tie_word_embeddings=false`, weight prefix `backbone.`.

`layers_block_type` (52 entries) gives 23 `mamba`, 23 `moe`, 6 `attention`
at indices **5, 12, 19, 26, 33, 42**.

| Block | Parameters |
|---|---|
| Mamba2 | `mamba_num_heads=64`, `mamba_head_dim=64`, `n_groups=8`, `ssm_state_size=128`, `conv_kernel=4`, `chunk_size=128`, `mamba_hidden_act=silu`, `use_conv_bias=true`, `use_bias=false`, `mamba_proj_bias=false`, `mamba_ssm_cache_dtype=float32`, `time_step_min/max/floor = 1e-3 / 1e-1 / 1e-4` |
| Attention | 32 q / 2 kv heads, `head_dim=128`, `rope_theta=10000`, `partial_rotary_factor=1.0`, `attention_bias=false`, `sliding_window=null` |
| MoE | `n_routed_experts=128`, `num_experts_per_tok=6`, `moe_intermediate_size=1856`, `n_shared_experts=1`, `moe_shared_expert_intermediate_size=3712`, `mlp_hidden_act=relu2`, `norm_topk_prob=true`, `n_group=1`, `topk_group=1`, `routed_scaling_factor=2.5`, `moe_shared_expert_overlap=true` |
| MTP | `num_nextn_predict_layers=1`, `mtp_layers_block_type=["attention","moe"]`, weights `mtp.layers.0.{eh_proj,enorm,hnorm,final_layernorm,norm,mixer.*}`, unquantized |

**Quantization** — `quant_method: modelopt`, `quant_algo: MIXED_PRECISION`,
`producer: modelopt 0.44.0rc5`, two `config_groups` plus a 5981-entry
`quantized_layers` map and an `ignore` list:

| Target | Scheme | Tensors |
|---|---|---|
| routed experts, shared experts, `lm_head` | `W4A16_NVFP4`, **`group_size=16`** | `weight`, `weight_scale` (e4m3 per-16-block), `weight_scale_2` (fp32 global) |
| mamba `in_proj` / `out_proj` (46 targets) | FP8 W8A8 static | `weight`, `weight_scale`, `input_scale` |
| attention `q/k/v/o_proj`, `conv1d`, gates, norms, embeddings | unquantized bf16 | — |
| KV cache | `kv_cache_scheme` fp8 | `k_proj.k_scale`, `v_proj.v_scale` |

Note the polarity trap: the repo name says NVFP4, and most of the *parameters*
are, but the mamba projections are FP8 and the attention tower is bf16. Reading
it as uniform NVFP4 gets the loader wrong in a way a token gate can still pass
while moving the wrong bytes — see [porting.md](../porting.md) on checking the
memory format against the oracle explicitly.

## 2. Upstream chain (`file:line` @ `555967922`)

| What | Anchor |
|---|---|
| registry | `registry.py:179` -> `models/nemotron_h.py::NemotronHForCausalLM` |
| layer dispatch (`ALL_DECODER_LAYER_TYPES`, `M`/`*`/`E`/`-`) | `nemotron_h.py:531-536`, model `:546-600` |
| Mamba2 layer | `nemotron_h.py:373-389` (`MambaMixer2`, fed `mamba_num_heads * mamba_head_dim`) |
| MoE | `nemotron_h.py:126-256` (`NemotronHMoE`), decoder layer `:317` |
| non-gated activation | `activation_without_mul(config.mlp_hidden_act)` -> `ReLUSquaredActivation` (`layers/activation.py`) |
| expert ckpt naming | `ckpt_names=("up_proj", "down_proj", "")` (`nemotron_h.py:220`) |
| routed scale applied to OUTPUT | `apply_routed_scale_to_output=True` (`nemotron_h.py:246`) |
| router dtype | `GateLinear(..., out_dtype=torch.float32, force_fp32_compute=True)` (`nemotron_h.py:150-156`) |
| state shape / dtype | `mamba_utils.py:174-199`, `:73-81` |
| MTP | `models/nemotron_h_mtp.py::NemotronHMTP` (`registry.py:638`) |
| MIXED_PRECISION resolution | `layers/quantization/modelopt.py:2280-2450`, per-layer lookup `:2416-2445` |

**Correction (W1, 2026-08-12).** This section previously said the
`quantized_layers` lookup is "direct name first, then shard prefix
(`modelopt.py:2426`, `:2437`)". That understated upstream and was a defect in
this spec, not in upstream. `_resolve_quant_algo` (`modelopt.py:2412-2487`) has
**five** strategies and tries them in this order:

1. **Direct lookup** (`:2424-2427`) over `_quantized_layer_prefix_candidates`.
2. **Packed/fused lookup** (`:2429-2447`): unfuse via the model's
   `packed_modules_mapping`, collect each shard's algo **across all base
   candidates**, and **raise `ValueError`** if the shards of one fused layer
   disagree.
3. **Prefix lookup** (`:2449-2453`): any `quantized_layers` key starting with
   `prefix + "."`, returning the first in map order.
4. **The `.experts` special case** (`:2455-2461`): a `FusedMoE` layer's prefix
   is `...moe.experts` while ModelOpt lists `...moe.up_proj` / `...moe.down_proj`,
   so the container falls back to its parent.
5. **`fused_projection_shards` fallback** (`:2463-2486`): `qkv_proj ->
   (q_proj, k_proj, v_proj)` and `gate_up_proj -> (gate_proj, up_proj)` for
   configs that list shard names with no `packed_modules_mapping` registered.
   The algo set is rebuilt **per candidate** here, unlike strategy 2, and it
   raises on disagreement too.

`_quantized_layer_prefix_candidates` (`:2489-2505`) itself yields the prefix, a
bare `lm_head` when the prefix ends in `.lm_head` (the real checkpoint stores a
BARE `lm_head` key), and the `language_model.model.` <-> `model.language_model.`
swap, de-duplicated in order.

Exclusion is separate and is checked **first**, in `get_quant_method`
(`:2515-2522`): `is_layer_excluded` (`:145-181`) runs `is_layer_skipped`
(`quant_utils.py:510-572`, which raises on a partially-excluded fused layer),
then a legacy substring rule kept for pre-0.39 exports, then `fnmatch`
wildcards — which is how the real `ignore` entry `mtp*` covers the whole MTP
head.

**Measured on the real config** (all 5981 `quantized_layers` entries and all 72
`ignore` entries): the histogram is exactly `{W4A16_NVFP4: 5935, FP8: 46}`,
every entry resolves by strategy **1** alone, and no entry collides with the
`ignore` list. Strategies 2-5 are therefore not reachable from this checkpoint
and are covered by synthetic fixture entries instead — see W1 below.

Config note: `nemotron_h.py` reads `config.hybrid_override_pattern`, which
newer transformers exposes as a property derived from `layers_block_type`
(`_list_to_pattern`, mapping `mamba->M`, `moe->E`, `attention->*`). Our loader
reads `layers_block_type` directly and does not reconstruct the char pattern.

## 3. Our baseline — reuse vs new

### REUSE (landed)

- Het-KV: `MambaSpec` + `HybridKVCacheCoordinator` + per-group managers
  (porting-inventory.md:78-79,109). Only 6 of 52 layers hold a paged KV group.
- Causal conv1d, all three arms (`kCausalConv1dFwd/Update/SpecUpdate`).
- Grouped-topk sigmoid routing with `e_score_correction_bias` and
  `routed_scaling_factor` — the DeepSeek-V2/V4 path (`kMoeRouterTopK`).
- Shared experts (`kSharedExpertGate`, `kMoeCombineGate`).
- NVFP4 W4A16 grouped MoE Marlin (`kMoeGroupedGemmNvfp4Marlin`), FP8 W8A8
  linear, fp8 KV (`kReshapeAndCacheFp8`).
- MTP spec-decode machinery (`qwen3_5_mtp.cpp`, `SPEC-MTP`).
- Dense attention + rope (`dense_attn::AttnBlock`, `kAttnQkNormRope`).

### NEW

- `vt::Mamba2ChunkScan` / `Mamba2StateUpdate` / `RmsNormGatedGroup` — **owned
  by #496, not by this row.**
- A non-gated `relu²` grouped-MoE arm on the merged-GEMM seam.
- A ModelOpt `MIXED_PRECISION` per-module quant resolver.
- `nemotron_h.cpp` / `nemotron_h_weights.cpp` / `nemotron_h_registry.cpp`,
  and the MTP head.

## 4. W-breakdown

Each W is one delegated task with its own fresh implementer and fresh reviewer.
W1 and W2 are independent of #496 and can run in parallel with it; W3 onward
cannot.

| W | Content | Gate | Depends on |
|---|---|---|---|
| **W1** | ModelOpt `MIXED_PRECISION` resolver: parse `quantization_config`, resolve per-module `quant_algo` (direct then shard-prefix), expose it to weight loading. Refuse an unknown algo by name | unit tests on the REAL `config.json` (committed as a fixture, weights not needed): every one of the 5981 entries resolves, the `ignore` list resolves to unquantized, an unknown algo refuses | — |
| **W2** | Non-gated `relu²` grouped MoE through `MlpGateUpMethodBase` / `vt::MergedGemmGroup`; bf16 arm then NVFP4 W4A16 g16 | byte/tolerance tests vs a host reference; `relu²` mutation caught; routed scale applied to the OUTPUT, not the logits | — |
| **W3** | `nemotron_h_weights.cpp` + `_registry.cpp`: `layers_block_type` dispatch, `backbone.` prefix, het-KV group construction (1 Mamba group + 1 full-attn group over 6 layers), enumeration gate vs the released index | enumeration: every tensor in `model.safetensors.index.json` is claimed or explicitly refused; KV spec shapes match `mamba2_state_shape` | #496 W1 |
| **W4** | `nemotron_h.cpp` forward: hybrid layer loop, Mamba2 mixer wiring, 6 attention layers, MoE layers | CPU forward runs; per-layer activations vs a dumped oracle reference | #496 W1, W2, W3 |
| **W5** | MTP head (`mtp.layers.0`, `eh_proj`/`enorm`/`hnorm`) on the existing spec-decode seam | draft acceptance non-zero; spec-off and spec-on token-identical | W4 |
| **W6** | **GB10 e2e token gate vs the pinned oracle** | token-exact greedy, identical prompts/counts/batching/sampling; oracle identity asserted | #496 W2 (CUDA), W4, W5 |
| **W7** | GGUF k-quant / i-quant arm through the shared GGUF loader (see §5b); refused by name until it lands | quant-matched load + token gate | W4 |

### W1 progress — the resolver has LANDED (2026-08-12)

`src/vllm/model_executor/layers/quantization/modelopt_mixed_precision.h`,
header-only, mirroring `modelopt.py` at the pin. It sits under `src/` rather
than beside its two `include/vllm/...` siblings on purpose:
`check-doc-checkpoint` classifies the whole `include/vllm/` prefix as a
user-facing surface and demands `docs/USAGE.md` move with it, and nothing this
header does is user-facing yet — it is not on the `include/vllm.h` ABI and no
loader calls it, so that edit would have documented nothing. **W3 promotes it to
`include/` when it becomes part of the consumed surface**, and pays the public
document obligation that genuinely applies then. Parses either config shape,
resolves a module prefix through all five strategies plus the exclusion pass,
and returns a TYPED `ModuleQuant` (algo, which strategy answered, group size)
rather than a raw string.

Two gate arms, deliberately two binaries so the opt-in one cannot mask the
always-on one:

- `test_modelopt_mixed_precision` — always-on, curated fixture at
  `tests/fixtures/modelopt_mixed_precision/curated_config.json`, whose real
  entries are copied verbatim from the checkpoint and whose synthetic entries
  are each annotated with the strategy they exist to reach. **26 cases, 167
  assertions** (22 / 137 as first landed; see the W1 repair note below).
- `test_modelopt_mixed_precision_checkpoint` — exhaustive, reads the real
  1.3 MB `config.json` from `$CHECKPOINT_ROOT/nemotron-3.5-lightning-30b-nvfp4`,
  asserts all 5981 + 72 entries and the exact histogram, plus the standalone
  `hf_quant_config.json` the repair below added. **3 cases, 12181 assertions,
  GREEN** (2 / 12145 as first landed; the checkpoint is staged on the NAS and
  reachable from the CPU box, so this ran rather than skipping). Exits 77 —
  CTest *Skipped*, with a loud banner naming the exact export — when the
  checkpoint is absent.

**One deliberate divergence from upstream**, recorded here because it is a
policy choice and not a port: an algorithm that resolves to something this
consumer does not implement is **refused by name**. Upstream's
`get_quant_method` falls through to `UnquantizedLinearMethod()` for anything
outside {FP8, NVFP4, W4A16_NVFP4, MXFP8} — including `FP8_PB_WO` and
`FP8_PER_CHANNEL_PER_TOKEN`, which are entries of its own `QUANT_ALGOS`. Silent
dequantization is numerically correct and therefore invisible to a token gate,
which is precisely the stop condition §0 names. A prefix simply ABSENT from
`quantized_layers` is not this case and stays unquantized, as upstream.

**Still owed by later W's:** nothing consumes the resolver yet — no loader, no
`get_quant_method` equivalent, no kernel selection. W3 wires it.

### W1 repair — the fresh review returned FAIL (2026-08-12)

Reviewed at `f5c901ce`; repaired on `row/MODEL-NEMOTRON-H-W1-FIX`. The review
confirmed all five resolution strategies against a verbatim upstream
transcription (63,787 + 6,493 prefixes, **0 diffs**) and found one real defect
plus four guarantees the suite did not actually hold.

**The defect: `Parse` refused the driver checkpoint's own `hf_quant_config.json`
(HIGH).** `Parse` gated on `ExtractQuantAlgo`, which mirrors
`_extract_modelopt_quant_algo` (`modelopt.py:245-263`) and requires a top-level
`quant_method` starting with `"modelopt"`. That precondition belongs to the
SELECTION hook — `override_quantization_method` uses it to tell a ModelOpt
`quantization_config` apart from a compressed-tensors one in the same field of
the same `config.json`. Upstream's PARSER, `from_config` (`:282-367`), never
inspects `quant_method` at all: it dispatches on the SHAPE. The real file has
top-level keys exactly `{producer, quantization}` and names no `quant_method`
anywhere, so the header threw
`std::invalid_argument: not a MIXED_PRECISION config (quant_algo="", quant_method="")`
on the one config `get_config_filenames()` (`:265-267`) actually points at,
while upstream parsed it and resolved all 5981 entries. Detection and parsing
are now separate: `ShapeQuantAlgo` for `Parse`, `ExtractQuantAlgo` for
`IsMixedPrecision`.

**Four guarantees the tests did not hold.** Each survived the reviewer's
mutation with the suite green; each now has a case that catches it, proven by
re-applying that exact mutation:

| Guarantee | Mutation that used to survive | Now caught by |
|---|---|---|
| Scans run in INSERTION order (`ordered_json`, header divergence 3) | load the fixture with plain `nlohmann::json` | `synthetic.layers.2.self_attn` resolves FP8, not W4A16_NVFP4 — sorted, `k_proj` comes first |
| Strategies 3 and 4 return the FIRST matching child | return the LAST match | new `synthetic.layers.8.moe.{a,b}_proj`, the first parent whose first and last child DISAGREE |
| Strategy 2's algo set spans ALL base candidates, unlike strategy 5 | rebuild the set per candidate | new `language_model.model...q_proj` / `model.language_model...k_proj` pair — the union raises, a per-candidate rebuild returns FP8 and never sees the second spelling |
| `is_layer_skipped`'s `experts` branch (`quant_utils.py:559-565`) | invert `e.find(prefix)`, or delete the branch | an `ignore` naming ONE expert child must exclude the whole container |

**`FnMatch` vs CPython, measured not assumed.** Two differential sweeps against
`fnmatch.fnmatchcase`: 3,183,165 pairs (names ≤3, patterns ≤5) with **0**
mismatches, and 6,291,453 pairs (names ≤2, patterns ≤6) with **108** across
**27** patterns — all one class, all CPython-True/ours-False, all a bracket
whose contents reduce to a bare `!` after `translate` drops a REVERSED range
(`[?-.!]` → `(?s:.)\Z`). Recorded in the header and NOT fixed: matching it means
reproducing `translate`'s rewriting, and ModelOpt emits no bracket expressions
at all, so no reachable input touches it.

**LOW-1, corrected 2026-08-13.** That count read **20** here and at
`modelopt_mixed_precision.h:253` when the branch went to review. It was wrong,
and wrong in a way the numbers refute on their own: every one of those patterns
translates to `(?s:.)\Z`, which matches a length-1 name and nothing else, so a
pattern can contribute at most the **4** one-character names in the sweep
alphabet `{a,b,.,0}` and 20 patterns cap out at 80 — below the 108 the same
sentence reports. Re-measured independently against CPython 3.12
`fnmatch.fnmatchcase` on this branch: sweep 1 reproduces `3,183,165 pairs,
0 mismatches`; sweep 2 reproduces `6,291,453 pairs, 108 mismatches` and yields
`27 distinct patterns`, 27 x 4 = 108, direction `{(True, False): 108}`,
`distinct translations: {'(?s:.)\Z': 108}`, `mismatching name lengths: {1: 108}`.
All 27 have the shape `[X-Y!]` with X > Y. `FnMatch`'s behavior is unchanged —
only the recorded number was.

**Why a plain `ctest` skips the exhaustive arm.** `CHECKPOINT_ROOT` is a `.env`
key and `.env` is not exported by anything; the repo's documented loader is
`set -a; . ./.env; set +a` (`.env.example:8`). There is no CTest-side `.env`
reader, so a shell that has not sourced it skips the arm. The skip banner names
the exact export rather than pretending the arm ran. (This paragraph also
credited `.agents/environment.md` with that loader line and claimed
`hf_snapshot.h` does not apply here; both were wrong — see LOW-3 below.)

Gate after repair: **26 cases / 167 assertions** curated (was 22 / 137),
**3 cases / 12181 assertions** exhaustive (was 2 / 12145), clean Release
`-Werror`, ASan and UBSan clean, full `ctest` 396/396.

### W1 land-prep — round-2 review PASS, its LOWs, and the two open dispositions (2026-08-13)

Round 2 reviewed `e734fe9e` and returned **PASS** with three LOW findings. All
three are repaired on `row/MODEL-NEMOTRON-H-W1-LAND`; each was re-verified
against the file or the measurement rather than taken on report.

**LOW-1 — the `27` above.** Recorded as `20` in two places. See the correction
paragraph in the block above for the arithmetic and the re-measurement.

**LOW-2 — `CAPTURE` on a `const char*` prints `1`.** doctest 2.5.2 has no
stringifier for a `const char*` lvalue, and two loops in
`test_modelopt_mixed_precision_checkpoint.cpp` (`:172`, `:248`) each compared
FIVE prefixes through one, so a failure named none of them. Demonstrated RED by
inverting both CHECKs in a scratch copy — `logged: p := 1`, ten times — and
GREEN under the identical inversions after switching both loops to
`const std::string`, which is what `:149` and `:161` already used:
`logged: p := backbone.embeddings`, `... := lm_head`, and so on for all nine
distinct prefixes.

**LOW-3 — two env vars gated one checkpoint, and neither carried the pin.**
Two parts.

*(a) A half-true citation.* The test claimed `.env.example` **and**
`.agents/environment.md` document the loader as `set -a; . ./.env; set +a`.
`.env.example:8` does, verbatim; `.agents/environment.md` does not contain the
string at all (`grep -n 'set -a'` exits 1) and instead points at `.env.example`
at `:16`. Corrected to cite `.env.example:8` alone.

*(b) The pin had no teeth.* `CHECKPOINT_ROOT` (this test, joining the staging
directory name by hand) and `VT_NEMOTRON35_SNAPSHOT`
(`parity::Nemotron35LightningSnapshot()`, landed on main with §5a) resolved the
same NAS `local_dir`, and `kNemotron35LightningNvfP4Revision` could refuse
neither: the accessor's HF-cache spelling is unreachable for a `local_dir` tree,
because there is no `snapshots/<rev>/` whose NAME carries the revision, and an
env override is deliberately never revision-checked. A re-download of the same
repo name lands a different revision under the identical path — exactly the
substitution `kQwen27NvfP4Revision` exists because of.

`hf download --local-dir` does record the revision, just not in the path: it
writes a per-revision file manifest at
`<dir>/.cache/huggingface/trees/<revision>.json`, confirmed present on the NAS
tree for `29f2d174…`. `Nemotron35LightningSnapshot()` is now the SINGLE resolver
for both spellings and gates the `CHECKPOINT_ROOT` path on that manifest. Proven
by construction, the manifest being the only thing that differs between A and B:

| Run | Setup | Result |
|---|---|---|
| A | staged dir carries only `deadbeef….json` | `EXIT=77`, loud skip |
| B | add `29f2d174….json`, nothing else changed | `EXIT=0`, 3 / 12181 |
| C | `VT_NEMOTRON35_SNAPSHOT` at the real NAS dir | `EXIT=0`, 3 / 12181 |
| D | neither env var set | `EXIT=77`, banner names the export AND the manifest |
| E | `VT_` set to a nonexistent dir, `CHECKPOINT_ROOT` valid | `EXIT=77` — refuses, never falls back |

`VT_NEMOTRON35_SNAPSHOT` keeps `HfSnapshot`'s documented escape semantics
unchanged and is checked first. That asymmetry is deliberate: naming ONE
directory outright is the deliberate different-checkpoint run the override
exists for, while naming a ROOT is not, so only the root path is revision-gated.

**Two round-1 dispositions that existed nowhere.** Round 1 returned findings 1-8
and 7 and 8 were answered verbally only.

- **Finding 7 — the header lives under `src/`, not `include/`.**
  **ACCEPTED, by operator decision.** `check-doc-checkpoint` classifies the
  whole `include/vllm/` prefix as user-facing and demands `docs/USAGE.md` move
  with it; nothing consumes the resolver yet and it is not on the
  `include/vllm.h` ABI, so that edit would have documented nothing. The checker
  behavior is tracked as **#515**. **W3 must promote the header to `include/`
  when it becomes consumed surface** and pay the public-document obligation that
  genuinely applies then. Recorded here so the debt outlives the conversation.
- **Finding 8 — the exhaustive arm is opt-in because `CHECKPOINT_ROOT` is not
  exported.** **ANSWERED.** The repo convention is `set -a; . ./.env; set +a`
  (`.env.example:8`); no new mechanism was invented, and the skip banner now
  names the exact export. LOW-3(b) supersedes the part of this that claimed
  `hf_snapshot.h` did not apply — it does, and it is now the resolver.

**Reported, outside this row's authority to fix.**
`.agents/environment.md:29-30` states that `CHECKPOINT_ROOT` "states an INTENT
and nothing more: no code in the tree reads `CHECKPOINT_ROOT`". The exhaustive
arm reads it — before this branch directly, now through
`parity::Nemotron35LightningSnapshot()` — so that sentence is false as written
and belongs to whoever owns `.agents/environment.md`.

**Land-prep gate, re-run on the re-merged tree.** Baselines are unchanged, which
is the point: the merge moved the spec and nothing else this row owns.

| Arm | Result |
|---|---|
| curated `test_modelopt_mixed_precision` | **26 cases / 167 assertions**, `Status: SUCCESS!` |
| exhaustive `test_modelopt_mixed_precision_checkpoint` | **3 cases / 12181 assertions**, `Status: SUCCESS!` (`CHECKPOINT_ROOT=/mnt/nas_share/checkpoints`) |
| clean Release, CUDA off, `-Werror` | 1209/1209 built, **0 warnings** |
| full `ctest` (Release) | **402 tests, 401 passed**; `test_engine_core_proc` failed under `-j` and passed on a serial re-run (the known starvation set); `test_modelopt_mixed_precision_checkpoint` and `test_voxtral_e2e` correctly *Skipped* with no `CHECKPOINT_ROOT` |
| Debug + `VLLM_CPP_SANITIZE=address,undefined`, the four targets this change touches | **4/4 Passed**, including under CI's own `ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1`, `UBSAN_OPTIONS=print_stacktrace=1`, `VT_POOL_BYPASS=1` |
| `scripts/agent-preflight.sh` | green |
| `scripts/check-doc-checkpoint.py` | `OK: public documents match the claims this change makes.` |
| `scripts/check-commit-trailers.py --range origin/main..HEAD` | `OK: commit trailer contract` |

The whole-tree sanitizer run reports 39 LeakSanitizer failures across unrelated
model/serving binaries. Those are the lane's known pre-existing state, not this
change: `.github/workflows/ci.yml:775-793` marks `sanitize-cpu`
`continue-on-error: true` precisely so "a pre-existing finding cannot block
unrelated work" and cites a live run whose conclusion is `success` with both
sanitizer lanes `failure`. This change cannot reach them either — the only `src/`
file it touches is `modelopt_mixed_precision.h`, which `grep -rln` finds included
by exactly two files, both of them its own tests, so every object in `libvllm.a`
is byte-identical to one built from `origin/main`.

**The six mutations, re-run against the re-merged tree.** All six RED on the
curated arm; the baseline is `26 | 26 passed | 0 failed` before each.

| Mutation | Curated result |
|---|---|
| `Parse` back on `ExtractQuantAlgo` | `FAILURE!` 25 passed / 1 failed — and REDs the exhaustive arm too, where the thrown case drops the assertion count to 12152 |
| fixture loaded with plain `nlohmann::json` | `FAILURE!` 163 passed / 4 failed |
| strategies 3+4 return the LAST match | `FAILURE!` 163 passed / 4 failed |
| strategy 2 rebuilds the algo set per candidate | `FAILURE!` 164 passed / 3 failed |
| `e.find(prefix)` → `prefix.find(e)` | `FAILURE!` 163 passed / 4 failed |
| the `experts` branch deleted | `FAILURE!` 163 passed / 4 failed |

Read `Status:`, not `assertions:` — mutation 1 prints `0 failed` on the
exhaustive arm while failing, because the case THREW and its remaining
assertions were never reached.

## 5. Gates

**Correctness first, always.** No throughput number is recorded by this row
until W6 passes. When speed is measured later, the denominator is vLLM's
production configuration, never `--enforce-eager`.

**Oracle identity is asserted, not assumed.** `$HOME/venvs/vllm-oracle` on the
dgx host symlinks to `vllm-oracle-v0.25.0-stage` — vLLM **0.25.0**, transformers
5.13.1 — which predates `NemotronHMoEDecoderLayer` entirely. A run through that
venv fails on this checkpoint and reads as "the model is unsupported". The pin
is `vllm-oracle-next`: `0.23.1rc1.dev1511+g555967922`, transformers 5.14.1,
flashinfer 0.6.15.post1. Every oracle run asserts all three and ABORTS on
mismatch before producing a number.

**The fixture must be the checkpoint the changed path loads.** Pin the
revision (`29f2d174`) explicitly; a repo silently re-quantized under the same
name has cost this project a full campaign before.

**A token gate cannot see a dtype that is too wide.** Every f32 buffer on this
path owes a one-line reason, and the loaded memory format is checked against
the oracle explicitly, not inferred from matching tokens.

**GPU discipline on dgx:** `flock $HOME/gpu.lock`, `local-ai-worker` parked,
never a large oracle alongside `ctest` — `gpu_memory_utilization` reserves HOST
RAM on GB10 and has OOM-rebooted the box.

## 5a. Gateability: CLOSED — the pinned oracle loads and runs it (2026-08-12)

`.agents/porting-a-model.md` §4 requires saying plainly whether the pinned
oracle can load the model. **It can.** On GB10, from
`$HOME/venvs/vllm-oracle-next` (vLLM `0.23.1rc1.dev1511+g555967922`,
transformers 5.14.1, flashinfer 0.6.15.post1 — all three asserted, the run
aborting on mismatch), against
`$CHECKPOINT_ROOT/nemotron-3.5-lightning-30b-nvfp4` at pinned revision
`29f2d1746d8f41e316523194b19018707749b1b1`:

- `ORACLE_IDENTITY_OK`
- `CONFIG arch = ['NemotronHForCausalLM']`, `nlayers = 52`
- `CONFIG pattern = MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME` —
  confirming on the real checkpoint that newer transformers derives
  `hybrid_override_pattern` from `layers_block_type`, which `nemotron_h.py`
  reads. That was previously a source-level assumption.
- `MODEL_LOADED_OK`, greedy decode, `EXIT=0`.

Constructing a config proves nothing; this ran the model. Three greedy goldens
(`temperature=0.0`, `max_tokens=32`) are committed at
[`tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json`](../../tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json),
with the revision pinned as `parity::kNemotron35LightningNvfP4Revision`
(`tests/parity/hf_snapshot.h`) so the gate names the revision its golden belongs
to. Output is coherent — Fibonacci is `0, 1, 1, 2, 3, 5, 8, 13, 21, 34` — so
this is a healthy denominator, not a degraded one.

**Two traps recorded here because they cost time.** (1) `$HOME/venvs/vllm-oracle`
on dgx symlinks to `vllm-oracle-v0.25.0-stage`, which predates
`NemotronHMoEDecoderLayer`; a run through it fails and reads as "the model is
unsupported". Use `vllm-oracle-next`. (2) A vLLM v1 driver script MUST guard its
body with `if __name__ == "__main__":` — EngineCore is spawned, the module
re-imports, and the failure surfaces as a `multiprocessing` traceback naming
neither vLLM nor the caller. The tell is the banner printing twice.

W6 consumes this golden; nothing else in this spec is unblocked by it.

## 5b. Quantized arms owed (`porting-a-model.md` §2)

`AGENTS.md` makes the quantized arms part of a model port, not a follow-up, and
names **GGUF k-quants** a standing requirement rather than a per-model choice.
For this row that means:

| Arm | State |
|---|---|
| ModelOpt NVFP4 W4A16 g16 + FP8 W8A8 (the shipped checkpoint) | W1-W6, the critical path |
| Safetensors bf16 | reachable via the same loader; owed a fixture |
| **GGUF k-quants / i-quants** through the shared GGUF loader | **OWED.** No NemotronH GGUF arm exists |

Until the GGUF arm lands it is **refused by name** at load, naming the missing
piece, never silently dequantized to a supported path — a silent fallback is
exactly what a token gate cannot see. Tracked as W7.

## 6. Risks / decisions

- **Non-gated MoE must not become a parallel path.** If
  `MlpGateUpMethodBase` / `vt::MergedGemmGroup` cannot represent a
  gate-half-absent expert, extend the seam or record one exact tracked
  exception. Never hand-roll a sibling.
- **`group_size=16` NVFP4.** Confirm our Marlin grouped path actually supports
  16 and does not silently assume another group size. Prove it on the real
  tensors, not on a synthetic fixture.
- **Router in f32.** Upstream forces fp32 router compute
  (`force_fp32_compute=True`). Mirror the polarity; do not inherit the model
  dtype here.
- **`routed_scaling_factor` position.** Applied to the OUTPUT
  (`apply_routed_scale_to_output=True`), not folded into the router weights. A
  mis-placed scale is exactly the class of error a token gate catches late and
  a unit test catches immediately.
- **6 attention layers out of 52** means KV is small and the 1M context is
  cheap — but it also means an attention-side defect is diluted across 46
  non-attention layers and may not move tokens on short prompts. Gate with a
  long-prompt arm, not only a 6-token one.

## 7. Now

**State at this commit:** spec committed, implementation **not started**. The
row stays `INVENTORIED`; this commit changes no lifecycle state. The checkpoint
is staged on the NAS and the oracle smoke run is queued behind the GPU lock.

**Next action:** dispatch fresh implementers for **W1** and **W2** (both
independent of #496) as soon as `row/KERNEL-SSM-MAMBA-SSD-W1` clears review,
so the ops-header churn does not collide.

## 8. Stop conditions

- The pinned oracle cannot be made to load and run this checkpoint on GB10 →
  stop and report; without a running oracle there is no gateable denominator
  and the row does not proceed on source inspection alone.
- A `quantized_layers` entry names an algorithm we do not implement → refuse by
  name and record it as owed. Never silently dequantize to a supported path;
  that is invisible to a token gate.
- The non-gated expert cannot be expressed on the shared merged-GEMM seam →
  `NEEDS_DECISION`, do not fork a parallel MoE path.

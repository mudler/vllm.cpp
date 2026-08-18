# Nemotron-H — the first hybrid Mamba2 model, and the first MIXED_PRECISION checkpoint

**Claim:** `CLAIM-MODEL-NEMOTRON-H`. **Model row:**
`MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` (existing, stays `INVENTORIED`
at this spec commit). **Issue:** [#517](https://github.com/mudler/vllm.cpp/issues/517).
**Gate-infrastructure issues this row depends on:**
[#569](https://github.com/mudler/vllm.cpp/issues/569) (checkpoint pinned by
CONTENT — **closed**, `751325460`; W6 must run with `VT_NEMOTRON35_SNAPSHOT`
UNSET and record the resolved directory, because that override is deliberately
never revision-checked),
[#547](https://github.com/mudler/vllm.cpp/issues/547) (GB10 reference-tier runs
the CPU kernel over device pointers).

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
| routed scale applied to OUTPUT | `apply_routed_scale_to_output=True` (`nemotron_h.py:234`), factor `:233` |
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
| **W2** | Non-gated `relu²` grouped MoE; bf16 arm then NVFP4 W4A16 g16. **As built it is NOT a merged pair** — the planned `MlpGateUpMethodBase` / `vt::MergedGemmGroup` routing was refuted during implementation and the arm is the EXISTING grouped GEMM plus a new `vt::MoeRelu2`; §6a is the authority on the seam | byte/tolerance tests vs a host reference; `relu²` mutation caught; routed scale applied to the OUTPUT, not the logits | — |
| **W3** | `nemotron_h_weights.cpp` + `_registry.cpp`: `layers_block_type` dispatch, `backbone.` prefix, het-KV group construction (1 Mamba group + 1 full-attn group over 6 layers), enumeration gate vs the released index | enumeration: every tensor in `model.safetensors.index.json` is claimed or explicitly refused; KV spec shapes match `mamba2_state_shape` | #496 W1 |
| **W4** | `nemotron_h.cpp` forward: hybrid layer loop, Mamba2 mixer wiring, 6 attention layers, MoE layers. **DONE — §6b is the authority on what was gated and how** | CPU forward runs; every block compared ELEMENTWISE against a reference written independently in `double` from the upstream formulas, both dtype arms, short AND long prompts. NOT against a dumped oracle activation reference as this row originally planned: that needs the WEIGHT LOADER, which does not exist yet, so it is owed to W6 alongside the token gate | #496 W1, W2, W3 |
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

`hf download --local-dir` does record the revision, just not in the path.
`Nemotron35LightningSnapshot()` is the SINGLE resolver for both spellings.

**SUPERSEDED 2026-08-13 by #569 (`751325460`).** The manifest at
`<dir>/.cache/huggingface/trees/<revision>.json` records *"this revision was
downloaded here once"*, **not** *"these bytes are that revision"*, so gating on
it was existence-only: a directory holding `LlamaForCausalLM` with every sidecar
naming a different revision, an **empty** `touch`ed manifest and a decoy beside
it, still RESOLVED. `huggingface_hub`'s `_tree_cache.py` never invalidates — its
docstring says a tree listing "can be cached forever without any invalidation
logic" — so manifests **accumulate** and an old revision's manifest vouches for
new bytes.

The resolver now sweeps the per-file `.cache/huggingface/download/<file>.metadata`
sidecars, whose `commit_hash` tracks the bytes. Row B below **no longer
reproduces**: adding the manifest changes nothing. Retained as the record of what
was tried and why it was insufficient.

| Run | Setup | Result (at the time; B is now superseded) |
|---|---|---|
| A | staged dir carries only `deadbeef….json` | `EXIT=77`, loud skip |
| B | add `29f2d174….json`, nothing else changed | `EXIT=0`, 3 / 12181 — **no longer true; the sidecars decide** |
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
| Safetensors bf16 | reachable via the same loader; owed a fixture. Its ENUMERATION is now correct (§5d finding 1): a producer with no `quantization_config` claims the bare weights and none of the 92 FP8/NVFP4 scale companions |
| **GGUF k-quants / i-quants** through the shared GGUF loader | **OWED.** No NemotronH GGUF arm exists |

Until the GGUF arm lands it is **refused by name** at load, naming the missing
piece, never silently dequantized to a supported path — a silent fallback is
exactly what a token gate cannot see. Tracked as W7.

## 5c. W3 result — registered, parsed, enumerated, KV-shaped (2026-08-13)

W3 landed on `row/MODEL-NEMOTRON-H-W3B` (base `fafa16f0`). It makes the
architecture KNOWN; it runs nothing. The forward and GGUF arms REFUSE BY NAME.

**Enumeration, the hard numbers.** `EnumerateNemotronHTensors` claims **18487 of
18487** released tensors; **0 unaccounted, 0 invented, 0 refused** — every tensor
has a named consumer, so no refusal was needed. Composition: 5 root (embeddings,
`norm_f`, the NVFP4 `lm_head` triple), 23 mamba layers x 13, 6 attention x 7,
23 MoE x 777 (2 router + 128x6 expert + 6 shared + 1 norm), and 270 MTP
(`mtp.layers.0` 8, `mtp.layers.1` 262). Both directions are gated against a
committed headers-only projection of the index, and a second case re-verifies
that projection against the LIVE checkpoint when `VT_NEMOTRON35_SNAPSHOT` names
it.

**Four things the reconnaissance had wrong or unstated, settled against source
and disk:**

1. **`layers_block_type` really is the source of truth, but for a subtler
   reason than "the config says so".** vLLM VENDORS its own `NemotronHConfig`
   (`transformers_utils/configs/nemotron_h.py:277-287`) in which the polarity is
   REVERSED — `hybrid_override_pattern` is the ctor argument and
   `layers_block_type` the derived property. That class is imported by
   `nemotron_h.py:83` for TYPE ANNOTATION only; the object that reaches the
   model comes from transformers `AutoConfig`, where `num_hidden_layers` is a
   property over `layers_block_type` whose setter discards the checkpoint's
   value (`configuration_nemotron_h.py:225-238`). §5a's live oracle run is what
   settles it on the real checkpoint. Both spellings are accepted here, the
   modern one winning — **for the SCHEDULE pair only**; see §5d, which corrects
   the generalization this sentence originally made.
2. **`moe_latent_size`: absent and `null` are the SAME state.** Upstream's
   predicate is `getattr(config, "moe_latent_size", None) is not None`
   (`nemotron_h.py:143`), so a missing key and an explicit `null` both mean "no
   latent MoE". A three-state representation would have been inventing a
   distinction upstream does not make. `std::optional` covers both; a real value
   REFUSES (§0).
3. **The shared `detail::ResolveMambaSsmCacheDType` is the WRONG reader here,
   and using it silently halves the recurrent state.** It reads
   `HfConfig::mamba_ssm_dtype`, which `hf_config.cpp:439` parses from the key
   **`mamba_ssm_dtype`** — Qwen3.5/3.6's spelling. NemotronH ships
   **`mamba_ssm_cache_dtype`** (`configuration_nemotron_h.py:121`), so the
   shared helper saw an empty string and returned the CONVOLUTION dtype. Caught
   by the KV gate as `page_size_bytes() == 1085440` against an expected
   `2134016` — the SSM state at bf16 instead of f32. This is not a Qwen bug;
   the two families genuinely use different config keys. Resolved locally by
   `NemotronHSsmCacheDType`, with the reason recorded at the call site.
4. **The conv-state layout discrepancy is real and deliberate.** Upstream's
   default is `"SD"` = `(state_len, dim)` (`mamba_utils.py:27-48`,
   `VLLM_SSM_CONV_STATE_LAYOUT` unset); our local convention across
   `qwen3_5_common.cpp:85` and `kimi_linear_registry.cpp:156` is
   `(dim, state_len)`. Same bytes, same page size; the local convention is kept
   so the shared runner sees one orientation, and the divergence is commented
   rather than left for W4.

**KV topology.** Two groups carrying their REAL per-layer names — 6
`backbone.layers.{5,12,19,26,33,42}.mixer` on a `FullAttentionSpec(2 kv heads,
head_size 128)`, and 23 mamba layers on a `MambaSpec` with shapes
`{{6144, 3}, {64, 64, 128}}` and dtypes `{bf16, f32}`. `conv_dim == 6144` is
falsified straight off disk by `mixer.conv1d.weight` BF16 `[6144, 1, 4]`, and
`in_proj` `[10304, 2688]` confirms `z + xBC + dt`. The names are load-bearing:
`kv_cache_utils.cpp:979` multiplies a mamba group's page by
`layer_names.size()`, and `kv_cache_interface.cpp:151-158` does the same for an
attention group, so a one-element tag would under-count both by 23x and 6x.

**Scope boundary held.** No per-module quant algorithm is resolved — that is W1,
which is not on `main`. W3 reads four coarse, individually falsifiable keys
(`quant_method`, `quant_algo`, `kv_cache_scheme`, and the `mtp*` entry in
`ignore`) and derives the scale companions STRUCTURALLY; the enumeration gate is
what proves that derivation against all 18487 tensors. A non-ModelOpt producer
refuses by name. One piece of honest debt is recorded in the code: the quantized
companion layout of a dense `mlp` block is DERIVED from the shared linear
layout, because no in-scope released NemotronH checkpoint ships one.

**Fixture.** `tests/vllm/models/fixtures/nemotron_h_35_lightning/` holds the
released `config.json` minus exactly `quantization_config.{config_groups,
quantized_layers}` (865 KB of 1.34 MB, the 5981-entry maps W1 owns) and a
707-family projection of the index. `ignore` is KEPT, unlike the original plan:
at 2.4 KB it is small, and its `mtp*` wildcard is what makes the MTP tower
unquantized — eliding it would have forced a guess about 270 tensors.

**Mutation proof (IMP-MUTATE).** Each defect applied alone to the restored tree,
rebuilt, the gate run, then `git checkout` and `git status --porcelain` verified
empty. All five turn it RED:

| Mutation | Result |
|---|---|
| `layers_block_type` `"moe"` mapped to `kAttention` | 4 cases / 9 assertions FAILURE |
| `LayerIndices` shifts the FIRST attention index by +1 | 2 cases / 2 assertions FAILURE |
| `conv_dim` drops the `2*n_groups*state_size` term | 4 cases / 8 assertions FAILURE |
| SSM cache dtype collapsed to the conv/activation dtype | 1 case / 2 assertions FAILURE |
| mamba `dt_bias` left UNCLAIMED (23 tensors) | 1 case / 2 assertions FAILURE |

**Gate evidence.** Release `-Werror` CPU: `test_nemotron_h_scaffold` 10/10 cases,
38245/38245 assertions, `Status: SUCCESS!`; with `VT_NEMOTRON35_SNAPSHOT` set,
10/10 and 39113/39113. Debug arm (asserts unmasked): identical. Full `ctest`:
`100% tests passed, 0 tests failed out of 401` (`test_voxtral_e2e` skipped, no
asset). `test_model_registry` 24/24 and `test_model_loader_gguf` 3/3 after their
pinned 37-architecture ledgers were reconciled to 38.

One defect was found this way rather than by inspection: the live re-verification
first died with `[json.exception.type_error.304] cannot use at() with null` from
inside a loop nowhere near its cause. `nlohmann::json::parse(x).items()` binds a
range to a TEMPORARY that is destroyed before the body runs. It reads as a clean
one-liner and it is undefined behaviour; the materialized form is what the
muse-glimmer precedent already used.

**Not done here:** the forward (W4), the MTP head (W5), the e2e token gate (W6),
the GGUF arm (W7). The row stays `INVENTORIED`.

## 5d. W3 repair — the review's four findings, and two residuals (2026-08-13)

The fresh review of `row/MODEL-NEMOTRON-H-W3C` @ `3295d0c1a` (PR #565) returned
**PASS** with four MINOR/NIT findings and two report-only items. Repaired on
`row/MODEL-NEMOTRON-H-W3-FIX` (base `3295d0c1a` + `origin/main` re-merged, W1's
`MIXED_PRECISION` resolver having landed as `1bc5ef82c`).

**1. `ClaimMamba` ignored `quantized` — FIXED.** Every other claimer
(`ClaimNvfp4`, `ClaimMoe`, `ClaimMlp`, and `ClaimAttention`'s `fp8_kv`) gates on
`quantized`; `ClaimFp8` did not. The released config MINUS `quantization_config`
parsed without refusal and enumerated **92 tensors an unquantized checkpoint
does not ship** — `backbone.layers.{mamba}.mixer.{in,out}_proj.{weight_scale,
input_scale}`, 23x4, first `backbone.layers.0.mixer.in_proj.weight_scale`. That
is exactly the shape §5b's owed **safetensors bf16 arm** will present. Nothing
consumes the map yet, so no gated claim was wrong — but an unimplemented arm is
refused BY NAME, never silently mis-enumerated. `quantized` now threads through
`ClaimFp8`/`ClaimMamba` and both call sites (backbone `quantized`, MTP
`mtp_quantized`). The 18487-tensor gate is unchanged: the released checkpoint is
quantized, and its MTP schedule is `{attention, moe}` with no mamba block.

**2. Legacy-alias precedence was INVERTED — FIXED, and the RECORD corrected.**
`Get{Int,Double,Bool}Aliased` preferred the MODERN key. Upstream does the
opposite for the `mamba_*` SCALARS: `configuration_nemotron_h.py:145-155` is
`self.n_groups = kwargs.pop("mamba_n_groups") if "mamba_n_groups" in kwargs else
self.n_groups`, which OVERWRITES an already-populated dataclass field, so
**legacy wins**. Re-derived by RUNNING transformers @ `7d06b1a5` (the pin this
port names), not by reading it:

```
NemotronHConfig(n_groups=8, mamba_n_groups=4, conv_kernel=4, mamba_d_conv=7)
  -> n_groups=4, conv_kernel=7
NemotronHConfig(chunk_size=128, mamba_chunk_size=77, expand=2, mamba_expand=9,
                use_conv_bias=True, mamba_conv_bias=False,
                time_step_min=1e-3, mamba_dt_min=0.5)
  -> chunk=77 expand=9 conv_bias=False dt_min=0.5
NemotronHConfig(layer_types=['mamba','mamba'], hybrid_override_pattern='*-')
  -> ['mamba', 'mamba']
NemotronHConfig(mtp_layers_block_type=['mamba'], mtp_hybrid_override_pattern='*E')
  -> ['mamba']
```

So the two families **genuinely disagree** and each is mirrored on its own
terms: legacy-wins for the `mamba_*` scalars (`:145-155`), modern-wins for both
SCHEDULE pairs (`:158-165`, `:176-184`, where the legacy pattern is consulted
only when the modern list is `None`). Worse than the behavior was the record:
the code comment and §5c above asserted "the modern one wins" as if it were
upstream's rule, which is what would mislead the next porter. Both polarities
are now stated where they are implemented, each with its own upstream anchor and
an explicit "do not unify these" note. **No released checkpoint ships both
spellings of one field**, so this was a mirroring defect and a record defect,
never a live one — which is precisely why it needed a test.

**3. The forward refusal was claimed but never exercised — FIXED.** The case
titled "the unported ARMS refuse by name" had one SUBCASE (GGUF).
`ForwardNemotronHForCausalLM` is an unconditional `VT_CHECK`, which throws
`std::runtime_error` (`vt/dtype.h:11-17`), so it is directly callable with a stub
`LoadedModel`. Now asserted, including that the message NAMES the missing piece
(W4 and this spec).

**4. `NemotronHBlockName` had zero call sites — FIXED by using it.**
`BlockFromName` now maps both directions through it and builds its refusal's
expected-list from the enum, so a fifth block kind cannot arrive alongside a
message that still lists four. A round-trip assertion pins every spelling the
refusal offers to one that actually parses.

**5. RESIDUAL, pre-existing, NOT fixed here — `docs/FEATURES.md:171` is off by
two.** It says "27 of the 32 registered text-generation architectures" while
`:173` implies 38 − 3 Parakeet − 1 `LlamaModel` = **34**, and
`tests/vllm/models/test_model_registry.cpp:47` now says "34 text archs". W3's
`+1` increment was correct; the BASE number was already stale before this row
touched it, and no checker validates it. Left for the operator to file — the
repair branch has no `docs/` authority, and fixing a pre-existing doc drift
inside a scoped repair would hide it.

**6. RESIDUAL, accepted by design — fixture DTYPE drift is invisible offline.**
The committed index projection pins tensor NAMES and SHAPES offline; DTYPES are
only re-verified by the live case
(`test_nemotron_h_scaffold.cpp`, `VT_NEMOTRON35_SNAPSHOT`). CI has no
checkpoint, so a re-quantization that changed dtypes while preserving names and
shapes — which has happened before to an `unsloth` repo under an unchanged name
— would pass CI and fail only where the checkpoint is staged. That is the
declared design (the alternative is committing dtype metadata that nothing
offline can falsify), and it is named here so W4/W6 do not rediscover it.

**Mutation proof, re-run in full on the repaired tree.** Each defect applied
alone to the restored tree, rebuilt, the gate run, then the file restored and
its SHA-256 re-verified byte-for-byte (`git status --porcelain` empty
afterwards). All fourteen turn it RED — the ten from the W3 review plus four
this pass adds:

| Mutation | Result |
|---|---|
| `layers_block_type` `"moe"` mapped to `kAttention` (at `BlockFromName`) | 7 cases / 12 assertions FAILURE |
| `kMoe` claims attention tensors (at the enumeration switch) | 2 cases / 4 assertions FAILURE |
| `LayerIndices` shifts every index by +1 | 2 cases / 2 assertions FAILURE |
| `conv_dim` drops the `2*n_groups*state_size` term | 4 cases / 8 assertions FAILURE |
| SSM cache dtype collapsed to the conv/activation dtype | 1 case / 2 assertions FAILURE |
| mamba `dt_bias` left UNCLAIMED (23 tensors) | 2 cases / 3 assertions FAILURE |
| MTP `final_layernorm` dropped | 2 cases / 3 assertions FAILURE |
| attention KV group collapsed to ONE layer tag | 1 case / 1 assertion FAILURE |
| the `mtp*` `ignore` entry not honored | 3 cases / 4 assertions FAILURE (assertions 38284 -> **39320**) |
| mamba KV group collapsed to ONE layer tag | 1 case / 1 assertion FAILURE |
| MTP `enorm` dropped | 2 cases / 3 assertions FAILURE |
| **NEW** the forward returns `{}` instead of refusing (finding 3) | 1 case / 2 assertions FAILURE |
| **NEW** `NemotronHBlockName` mislabels `kMoe` (finding 4) | 8 cases / **0 assertions** FAILURE |
| **NEW** `ClaimFp8` ignores `quantized` again (finding 1) | 1 case / 1 assertion FAILURE |
| **NEW** the aliased getters prefer MODERN again (finding 2) | 1 case / 4 assertions FAILURE |

Two of those rows are worth keeping in view. The `mtp*` mutation makes the
assertion COUNT go **up** by 1036 while the gate goes red — a changed count is
itself the signal. And the `NemotronHBlockName` mutation reports
**`assertions: 28 | 28 passed | 0 failed`** next to `8 failed` test cases: the
cases THREW, so `grep 'assertions:'` alone would have read that mutation as
clean. Read `Status:`.

**Gate evidence (this repair).** Local CPU-only host (`VLLM_CPP_CUDA=OFF`), disk
**68G free / 85% used** at every measurement below.

| Arm | Result |
|---|---|
| Release `-Werror`, full build | 0 warnings, 0 errors |
| Release `test_nemotron_h_scaffold`, offline | **12/12 cases, 38284/38284 assertions, `Status: SUCCESS!`** |
| Release `test_nemotron_h_scaffold`, `VT_NEMOTRON35_SNAPSHOT` live | **12/12, 39152/39152, `Status: SUCCESS!`** |
| Debug (`-g0`, asserts unmasked), offline | 12/12, 38284/38284, `Status: SUCCESS!` |
| Debug (`-g0`), live | 12/12, 39152/39152, `Status: SUCCESS!` |
| Full `ctest -j4` | **100% tests passed, 0 tests failed out of 403** (skipped: `test_modelopt_mixed_precision_checkpoint`, `test_voxtral_e2e` — neither has its asset here) |

The W3 baselines were 10/38245 offline and 10/39113 live; the deltas (+2 cases,
+39 offline / +39 live assertions) are this pass's three new cases and the
`NemotronHBlockName` round-trip. RED-before on the pre-fix tree was
12 cases / 9 failed assertions, reporting `92` companions with first
`backbone.layers.0.mixer.in_proj.weight_scale`, and `8 == 4` / `4 == 7` /
`2 == 9` / `128 == 77` / `0.001 == 0.5` / `0.1 == 0.6` / `0.0001 == 0.7` for the
alias precedence.

## 5e. W3 landing — the `doc-checkpoint range` gate, and how it was cleared (2026-08-13)

`row/MODEL-NEMOTRON-H-W3-FIX` @ `cb37239d4` (PR #572) was green on every gate
except one, and could not clear it itself:

```
Committed range vs origin/main:
  FAIL doc-checkpoint range
     ERROR: commit 3981de6a4: changed feature_surface but did not update
     docs/FEATURES.md.
```

The rule is unconditional and PER-COMMIT: `check-doc-checkpoint.py:79`
classifies any path under `src/vllm/model_executor/models/` as
`feature_surface`, `:311-316` requires `docs/FEATURES.md` in the SAME commit,
and `commits_in_range` (`:369-375`) walks
`rev-list --reverse --no-merges origin/main..HEAD` — so W3C's own
`docs/FEATURES.md` edit in `3295d0c1a` does not cover a later commit. `ci.yml:343`
runs the identical invocation, so CI was red for the same reason. The checker
has no exemption mechanism, deliberately, and weakening it was never on the
table.

`main` is never force-pushed, so `3981de6a4` could not be amended in place.
The history was instead rebuilt on `row/MODEL-NEMOTRON-H-W3-LAND` as ONE
squashed commit off current `origin/main`, carrying the whole W3 + W3-FIX
content with the `docs/FEATURES.md` clause included — which is what the
squash-merge would have produced anyway, and which satisfies the per-commit rule
trivially. This mirrors the precedent set on this row when `W3B` was superseded
by `W3C`.

**Tree equivalence, proven not asserted.** `git diff cb37239d4` against the
squashed tree lists exactly five files: the four that `72e661ae4` (the one
`origin/main` commit the fix branch lacked, #442 DSPARK) touches, plus the one
`docs/FEATURES.md` line. Every `src/`, `tests/`, `CMakeLists.txt`,
`docs/USAGE.md` and spec-body path is byte-identical.

**The `docs/FEATURES.md` wording, and why it is not the one proposed here.** The
draft above read "claims the bare weights and none of the **92** FP8/NVFP4 scale
companions". 92 is the count of the companions `ClaimMamba` was leaking
(23 mamba blocks × 2 projections × 2), not the number of scale companions in the
model — the released checkpoint carries thousands, across `ClaimNvfp4`,
`ClaimMoe`, `ClaimMlp` and the fp8-KV pair. Naming 92 would have implied the
architecture has 92 companions in total. What the gate actually asserts is
stronger and simpler: an unquantized producer claims a strict SUBSET of the
quantized arm's names with ZERO companions of any kind
(`.weight_scale`, `.weight_scale_2`, `.input_scale`, `.k_scale`, `.v_scale`).
The landed clause is:

> 18487/18487 tensors claimed; **a bf16 config claims that set minus its scale
> companions.** Nothing runs yet (spec #517, blocked on #496)

`check-public-doc-tables.py` caps a table CELL at 220 chars and the row was
already at 211, so the clause was paid for out of this row's own budget, as
that checker's `MAX_ROW_CHARS` comment directs ("shorten THIS row and move its
forensics"). What moved out is `het-KV shapes match \`mamba2_state_shape\`` — a
forensic anchor already carried by §5c above and by "KV-shape gated", which
stays in the cell. `0 unaccounted` and `released` went too, both implied by
"18487/18487". Finding 5's `:171`/`:173` count drift was NOT touched; it stays
open for its own issue.

**Every gate re-run on the squashed tree**, because a squash is a new tree and
inherited numbers are void. Local CPU-only host (`VLLM_CPP_CUDA=OFF`, GNU 13.3,
Ninja), disk **68-69G free / 85% used** at every measurement:

| Arm | Result |
|---|---|
| Release `-Werror`, clean full build | 1213/1213 targets, **0 `warning:` lines in the captured log**, 0 errors |
| Release `test_nemotron_h_scaffold`, offline | **12/12 cases, 38284/38284 assertions, `Status: SUCCESS!`** |
| Release, `VT_NEMOTRON35_SNAPSHOT` live | **12/12, 39152/39152, `Status: SUCCESS!`** |
| Debug (`-g0`, asserts unmasked), offline | 12/12, 38284/38284, `Status: SUCCESS!` |
| Debug (`-g0`), live | 12/12, 39152/39152, `Status: SUCCESS!` |
| Full `ctest -j4` | **100% tests passed, 0 failed out of 403** (skipped: `test_modelopt_mixed_precision_checkpoint`, `test_voxtral_e2e` — neither has its asset here) |
| `doc-checkpoint` over `origin/main..HEAD` | **ok** (was the one FAIL) |

**The fifteen mutations, re-applied to the squashed tree.** Each alone, rebuilt,
run, restored, SHA-256 re-verified and the working-tree-vs-index diff proven
empty. All fifteen RED:

| Mutation | Result on the squashed tree |
|---|---|
| `layers_block_type` `"moe"` → `kAttention` at `BlockFromName` | 7 cases / 12 assertions FAILURE |
| `kMoe` claims attention tensors (enumeration switch) | 2 cases / 4 assertions FAILURE |
| `LayerIndices` shifts every index by +1 | 2 cases / 2 assertions FAILURE |
| `conv_dim` drops the `2*n_groups*state_size` term | 4 cases / 8 assertions FAILURE |
| SSM cache dtype collapsed to the conv dtype | 1 case / 2 assertions FAILURE |
| mamba `dt_bias` left UNCLAIMED | 2 cases / 3 assertions FAILURE (38284 → 38238) |
| MTP `final_layernorm` dropped | 2 cases / 3 assertions FAILURE |
| attention KV group collapsed to ONE layer tag | 1 case / 3 assertions FAILURE |
| the `mtp*` `ignore` entry not honored | 3 cases / 4 assertions FAILURE (38284 → **39320**) |
| mamba KV group collapsed to ONE layer tag | 1 case / 2 assertions FAILURE |
| MTP `enorm` dropped | 2 cases / 3 assertions FAILURE |
| the forward returns `{}` instead of refusing | 1 case / 2 assertions FAILURE |
| `NemotronHBlockName` mislabels `kMoe` | 8 cases / **0 assertions** FAILURE |
| `ClaimFp8` ignores `quantized` again | 1 case / 1 assertion FAILURE |
| the aliased getters prefer MODERN again | 1 case / 8 assertions FAILURE |

The two instructive rows survive the re-run unchanged. The `mtp*` mutation moves
the assertion COUNT **up** by 1036 while the gate goes red — a changed count is
itself the signal. And `NemotronHBlockName` prints
**`assertions: 28 | 28 passed | 0 failed`** beside `8 failed` test cases,
because the cases THREW: `grep 'assertions:'` reads that red gate as clean.
Read `Status:`.

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

## 6a. W2 note — the non-gated `relu²` expert, as built

**Seam verdict: the non-gated expert is NOT a merged pair, and does not get a
`MergedGemmGroup` descriptor.** `MergedGemmGroup` describes N GEMMs *sharing
operand A* collapsed into one launch (`merged_gemm.h:1-22`). NemotronH's expert
has exactly one projection — `ckpt_names=("up_proj", "down_proj", "")`
(`nemotron_h.py:220`, the empty third entry being the absent gate) — so with
N == 1 there is nothing to merge and no launch to save; an arity-1 descriptor
would name a fusion that does not exist. `MlpGateUpMethodBase`
(`linear.h:82-86`) is likewise a *merged `[2I,H]` gate_up* seam and has no pair
to hold either.

The arm is therefore the **existing** grouped projection plus the activation we
did not have — exactly the shape the gated bf16 archs had before their pair was
folded (`kMoeGroupedGemmBf16` + `kMoeSiluMul`):

```
up   : kMoeGroupedGemmBf16   (bf16)  |  kMoeGroupedGemmNvfp4Marlin (W4A16 g16)
act  : kMoeRelu2                          <- NEW, the only new kernel
down : kMoeGroupedGemmBf16   (bf16)  |  kMoeGroupedGemmNvfp4Marlin (W4A16 g16)
comb : kMoeCombine(..., routed_scale)     <- routed scale on the OUTPUT
```

No parallel MoE path was added. The reasoning is recorded next to the seam it
excludes (`merged_gemm.h`, the note after the bf16-sibling block).

**`vt::MoeRelu2` (`OpId::kMoeRelu2`, CPU + CUDA).** Mirrors
`ReLUSquaredActivation` (`layers/activation.py:609-628`) as the fused-MoE path
reaches it: `activation_without_mul("relu2")` → `MoEActivation.RELU2_NO_MUL`
(`layers/fused_moe/activation.py:34`; `activation_without_mul` is `:98`, the
enumerator at `:33` is `GELU_TANH_NO_MUL`) → `apply_moe_activation`'s (`:184`)
`F.relu(input, inplace=True); torch.square(input, out=output)`. The **dtype
order is the mirrored part**: upstream's kernel
(`csrc/libtorch_stable/activation_kernels.cu:673-678`) widens to f32, clamps at
zero in f32, squares in f32 and rounds ONCE on the store. No new f32 buffer is
introduced — the op reads and writes the caller's dtype and only its arithmetic
is f32, which is what `LoadF32`/`StoreF32` already are elsewhere in `vt`.

**`routed_scaling_factor` is applied to the OUTPUT**
(`apply_routed_scale_to_output=True`, `nemotron_h.py:234`). `vt::MoeCombine`
gained a trailing `routed_scale` (default `1.0f`, so every landed caller is
byte-identical) which multiplies the routed sum *before* the shared term is
added — literally `moe_runner.py:390-407` (`:402-406` `fused_output *= routed_scaling_factor`,
`shared_output` untouched) followed by `:722-725` (`shared_output + fused_output`).
Upstream forces the ROUTER's factor to `1.0` in exactly this case
(`layer.py:291-300`), so `MoeRouterTopKArgs::routed_scaling_factor` stays 1.0 on
this path. Note this is the *opposite* polarity from Laguna, which folds the same
factor into the router weights by linearity (`laguna_ops.h:48`); NemotronH takes
the literal upstream form.

**`group_size=16` NVFP4 — SUPPORTED, risk closed by source.** `MoeMarlinArgs`
already defaults to `group_size = 16` with `mxfp4 = false` (`ops.h`), and
`cuda_moe_marlin.cu:7,115-129` documents and consumes exactly that
(`group_blocks=1`, `s_type = kFE4M3fn`, `num_groups = size_k / group_size`); 32
is reachable only via the MXFP4 branch. It is the configuration the landed
NVFP4 MoE archs (Laguna, Qwen3.5) already run. A unit test pins the default so a
later widening cannot silently re-point these experts.

**CUDA arms — what was actually run, and by whom.** The implementer did NOT
compile them: their worktree had no `nvcc`, so at `e2d68404` the CUDA arms were
*written and reviewed*, never built, and the earlier wording here
("compiled-and-reviewed") overstated it. They have since been compiled and
GPU-verified **by the fresh reviewer**, on `dgx.casa` (GB10, nvcc 13.0.88), from
a `git archive` of `e2d68404`:

- Release `-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=121a
  -DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0 -DVLLM_CPP_TRITON=ON` exited 0 with
  **671/671 targets and zero warnings**; `cuda_moe.cu.o` compiled under
  `-Werror=all-warnings`.
- A reviewer-authored GPU parity test proved `MoeRelu2` CUDA == CPU
  **bit-for-bit** over 4097 elements in all four dtype arms; that CUDA
  `routed_scale` scales the routed sum only; and that the `1.0f` default is
  byte-identical to the landed 4-arg call across all 8 dtype combinations.
- Branch tests on the GPU box: `test_ops_moe_nongated_relu2` 10/10,
  `test_ops_moe` 9/9 with 33451 assertions.

**That GPU build is NOT a proof about this head, and the merge messages that
implied otherwise were wrong (corrected 2026-08-13).** Four of this branch's
land-prep merge commits carry a sentence of the form "`src/vt/cuda/cuda_moe.cu`
and `vt/ops.h` are untouched by this merge, so the CUDA arm the fresh reviewer
compiled and GPU-verified on GB10 at `e2d68404` is unchanged." Each half is
individually true — `git diff --shortstat <merge>^1 <merge> -- include/vt/ops.h`
is empty for `57e4301489`, `40908db153`, `6ac8eed85e`, `721f44d38d` and
`f6a7f87090` — but the CONCLUSION does not follow, because the compile inputs
moved in commits those merges do not cover. Measured on this branch:

| Since `e2d68404` (the GPU-verified tree) | Delta |
|---|---|
| `include/vt/ops.h` | **+180 / −1** (`git diff --numstat e2d68404 HEAD`) |
| `src/vt/cuda/cuda_moe.cu` | **+7 / −1** |

The header movement is `d3cd442e6`, the merge that brought Mamba2 SSD W1
(`fe81bd3b5`, #496) — a merge that DID touch `include/vt/ops.h`, by +179 lines
against its first parent (`git show --stat d3cd442e6`) — and the `.cu` movement
is this row's own `dd7a6477d` repair. So `cuda_moe.cu` has been compiled at
`e2d68404` and nowhere since, and no local GPU host is reachable to redo it
(`dgx.casa` is down). The standing evidence for the CUDA arm at THIS head is
therefore the PR's `cuda-fat-build` CI job, which compiles `cuda_moe.cu`
against the merged `include/vt/ops.h` for 10 architectures under
`-Werror=all-warnings`; the GB10 run remains the evidence for the arm's
RUNTIME behaviour at `e2d68404`, which the eight-line `.cu` delta since is a
comment change plus the `routed_scale` parameter the reviewer's own GPU parity
test exercised.

**Scope of the "bit-for-bit" claim.** The reviewer's GPU parity result is a
MEASUREMENT on GB10, not a guarantee the build flags provide: `-ffp-contract=off`
is pinned for CXX, HIP and OBJCXX (`CMakeLists.txt:55`, `:393`, `:467`) but
nothing passes nvcc `--fmad=false`, and `CMakeLists.txt:41-56` carves CUDA out
deliberately ("GPU parity tests compare GPU-vs-GPU"). `MoeRelu2` is safe by
construction — a compare-select and one multiply, with no multiply-add to
contract — and the `routed_scale` step is one standalone multiply on the
finished accumulator, which is why §6a's evidence is stated for exactly those
two. `MoeCombine`'s `acc += w * Load(...)` reduction is the contractable one
and is NOT covered; the comment in `cuda_moe.cu` claiming it was has been
narrowed to what holds. The flag gap itself is repo-wide and tracked as **#591**
(found independently in `cuda_mamba2_ssd.cuh`); it is not repaired here.

**Still OWED** (no GPU in the implementer/repair worktrees, and not covered by
the above): `kMoeGroupedGemmNvfp4Marlin` exercised on the real NemotronH g16
tensors, and the end-to-end NemotronH MoE block on GB10. Both remain owed to W6
or an earlier GPU-host spot check. The `group_size` unit test pins the default
only — it is not a run of the Marlin arm.

**Evidence.** `tests/vt/test_ops_moe_nongated_relu2.cpp` (**12 cases / 81
assertions**): the activation against hand-computed exact values, the
`relu`/`silu` mis-ports, a bf16-in/f32-out arm that catches narrowing the square,
a bf16-out raw-bit arm, the shape/dtype/**device** contract refusals, the routed
scale on the routed sum only, the routed scale on the **assembled sum rather than
each router weight** (bitwise), the f16-out refusal that makes upstream's fp16
arm unreachable, the 1.0 default being byte-identical to the landed call, and the
whole expert `up → relu² → down → scaled combine` against an
independently-written scalar reference.

Mutations executed and caught (Release, `-ffp-contract=off`; every one restored
and md5-verified afterwards):

| # | Mutation | Target | Result |
|---|---|---|---|
| M1 | `relu` (square dropped) | `test_ops_moe_nongated_relu2` | RED 5 cases / 27 assertions |
| M2 | `silu` (the gated family's activation) | same | RED 5 / 35 |
| M3 | square narrowed through bf16 | same | RED 2 / 20 |
| M4 | `routed_scale` dropped | same | RED 3 / 32 |
| M5 | `routed_scale` applied to routed **+ shared** | same | RED 2 / 29 |
| M6 | `routed_scale` **folded into each router weight** | same | RED 1 / 4 |
| M7 | routed scale folded into the router **logits** | `test_ops_moe_router_grouped` | RED 3 / **525** (was recorded 498 — see below) |
| M8 | NVFP4 `group_size` default 16 → 32 | `test_ops_moe_nongated_relu2` | RED 1 / 1 |
| M9 | `MoeRelu2` device `VT_CHECK` dropped | same | RED 1 / 2 |
| M10 | `IsOutFloat` widened to admit `kF16` | same | RED 1 / 2 |

M6 is the one this repair added. At `e2d68404` it **survived green** (10/10
cases, 71/71 assertions): the landed cases all compared with a tolerance, and the
fold is exact-arithmetic-equal, so nothing could see it. It is also the most
likely W4 mistake, because Laguna performs exactly that fold
(`laguna_ops.h:48`) — legally, since Laguna passes no `shared`. The new case
pins it bitwise on decimal-grid data whose f32 products carry full mantissas
(rows separate by 10 and 4 ULP), with a `REQUIRE` that the data separates the two
forms so the green cannot be vacuous. M7 does NOT red the NemotronH file by
design — this path forces the router factor to 1.0 (`layer.py:291-300`), so the
router's own suite is where that defect is visible.

**M7's assertion count was recorded wrong (repaired 2026-08-13).** The table and
the `dd7a6477d` commit message both said `RED 3 / 498`. Three independent
re-measurements — the fresh review's, the operator's, and this repair's — all
read **3 cases / 525 assertions** failing, out of the unchanged 14 / 941
baseline. The CASE count was right; only the assertion count was wrong, and 498
is reproducible by nobody. `tests/vt/test_ops_moe_router_grouped.cpp` and the
grouped-router kernel it exercises have not moved since `e2d68404`
(`git log --oneline e2d68404..HEAD -- tests/vt/test_ops_moe_router_grouped.cpp`
is empty), so the count is stable and the original entry was a transcription
slip, not drift. Re-measured here by folding
`args.routed_scaling_factor` into the logits fed to sigmoid/softmax in
`MoeRouterGroupedTopKKernel` (`cpu_ops.cpp:2325-2339`) and disabling the
post-renormalize weight scale (`:2430-2434`); Release, `-ffp-contract=off`;
tree restored and md5-verified (`cd409b9465c00834be373cf3ecfb4c1d`), rebuilt,
back to 14/941 SUCCESS.

## 6b. W4 result — the forward COMPUTES, and the gate around it did not (2026-08-14)

W4 was built on `row/MODEL-NEMOTRON-H-W4B` from the rescued WIP commit
`d8c0d13f2` — 2411 lines committed by a session that died mid-build with no gate
ever executed — re-merged onto `origin/main`.

**The inherited forward compiles and is numerically RIGHT.** First clean Release
`-Werror` build: 416/416 targets, **0 `warning:` lines**, ninja exit 0. First run
of its gate: **13 cases, 10 passed, 3 failed, 161 assertions, `Status: FAILURE!`,
exit 1** — and every one of the POSITIVE comparisons against the independently
written `double` references passed on that first run: the Mamba2 mixer, SSD
chunk-size invariance, the carried two-leg state, GQA attention, the non-gated
relu² MoE, the dense MLP, the whole hybrid stack (short + long, f32 + bf16), the
single-branch residual structure, the refusals, and greedy determinism.

### The three failures were all anti-vacuity guards, and chasing them found the real defect

Each `AnyDiffers(...)` guard asserts "the mis-port is a DIFFERENT answer". All
three returned false. Measured separations:

| Guard | max_abs | max_rel | bitwise differing | Verdict |
|---|---|---|---|---|
| bf16 SSM state vs f32 | 1.42e-6 | 2.38e-2 | 192/192 | REAL, but judged by an ABSOLUTE band 140x larger than the largest difference the defect can produce |
| RoPE-rotated attention | 9.41e-6 | 3.18 | 383/384 | attention was DEGENERATE in the fixture (below) |
| routed scale folded into router weights | 2.09e-8 | 3.68e-5 | 245/288 | the fold is EXACT-ARITHMETIC-EQUAL; the test's claim was FALSE |
| routed scale folded into logits | 8.55e-2 | 25.3 | 288/288 | genuinely different — this guard was sound |

RopeNeox demonstrably DID rotate q (360 of 384 elements moved), so no guard
failed because its own instrument was dead
([[absent-hook-looks-like-armed-instrument]]).

**What that exposed: the bf16 arms could not fail at all.** `TolFor(bf16)` was a
flat `{atol 6e-2, rtol 6e-2}` applied to references whose own magnitude nobody
had measured:

| Comparison | max abs(want) | mean abs(want) | atol | atol/mean | all-zeros answer |
|---|---|---|---|---|---|
| attention T=48 bf16 | 0.0109 | 3.55e-4 | 0.06 | **169** | PASSES |
| attention T=6 bf16 | 0.0109 | 1.85e-3 | 0.06 | **32.4** | PASSES |
| mamba2 mixer bf16 | 0.0447 | 1.69e-2 | 0.06 | **3.55** | PASSES |
| logits T=6 bf16 | 0.950 | 0.377 | 0.6 | **1.59** | PASSES |
| logits T=40 bf16 | 1.03 | 0.425 | 0.6 | **1.41** | PASSES |
| attention T=48 f32 | 0.0106 | 3.60e-4 | 2e-4 | 0.556 | weak |

bf16 is the RELEASED checkpoint's model dtype. On that arm a mixer returning all
zeros passed. The f32 arms carried all the gating there ever was. This is
[[gate-comparing-shared-helper-proves-consistency-not-correctness]] in a new
shape: not a shared helper, but a band larger than the signal.

### Five repairs

1. **Bands are RELATIVE to the reference's own peak**, not flat absolutes.
2. **Every comparison SELF-CERTIFIES.** `ExpectCloseRel` REQUIREs that its own
   band REJECTS an all-zeros answer before it accepts the real one. A gate that
   cannot fail is now itself a test failure, asserted at run time rather than
   left for the next reader to re-derive. M15 below proves it works.
3. **The attention fixture was degenerate.** At q/k weight scale 0.2 the tiny
   model's logits were ~0.09, so the softmax was near-UNIFORM and the block was
   an unweighted mean of v — tiny by cancellation, and nearly blind to anything
   that only moves attention WEIGHTS (RoPE, the scale factor, causality). The
   real checkpoint is not in that regime: `head_dim=128` and `hidden_size=2688`
   put it in the selective one by construction. Raised to 0.95 to restore it.
4. **The routed-scale case asserted something false.** `routed_scaling_factor`
   is applied AFTER the top-k renormalisation, so folding it into the router
   weights and scaling the assembled routed sum are the SAME expression; the
   separation is floating-point association only. The bitwise instrument for it
   is at the op level (§6a M6). The case now gates the two things that ARE
   observable here — scaling the SHARED term (which
   `apply_routed_scale_to_output=True` exists to prevent, and which a
   shared-expert-free architecture like Laguna cannot expose) and scaling the
   LOGITS — and pins the fold as arithmetically indistinguishable.
5. **The SSM-cache-dtype guard asserted in the wrong place.** Downstream, the
   separation is 3.16e-5 of the signal peak, BELOW the f32 arm's own 2e-4 band,
   because `A = -exp(A_log)` decays a carried state within a few tokens. It now
   gates the STORED STATE, where the dtype actually lives: the dtype itself, the
   exact 2x byte-count ratio (the assertion that caught the shared-resolver
   defect in §5c), and a relative separation above bf16's resolution.

### Seams

- **`vt::FusedChain`** — the two residual add+RMSNorm sites now route through
  `kFusedAddRmsNormStd` behind `VT_FUSED_CHAIN_ADOPT`, per AGENTS.md. The
  inherited code hand-called `vt::RmsNorm(..., &residual)` and
  `scripts/check-fusion-consistency.py` was RED on it. Both arms are green and
  the Tier-0 composite dispatches to the same primitive.
- **`ModelRegistry::Forward`** — `ForwardNemotronHForCausalLM` reaches the
  forward through the shared seam (inherited from the WIP, kept).
- **`dense_attn::AttnBlock` — NOT APPLICABLE AT W4, recorded rather than
  skipped.** That seam is the DEVICE/PAGED full-attention block: it requires a
  `PagedKvCache`, a `slot_mapping`, a `block_table`, `StepInputs`, and a
  `Qwen3DenseAttnWeights` built from `OwnedTensor`s a loader produced. W4 is the
  HOST reference forward and owns none of those. This is the same boundary
  Kimi-Linear's `kimi_linear_forward.cpp` and DeepSeek-V4's
  `DeepseekV4ForwardHost` sit on, and W6 is where `AttnBlock` applies.
- **`layers::MlpGateUpMethodBase` / `vt::MergedGemmGroup`** — not applicable;
  §6a already settled that a non-gated expert has no pair to merge.
- **`scripts/runner-routing-allowlist.txt`** — a DEVIATION from the W4 task's
  stated authority, argued in the commit that makes it. Wiring the registry
  forward to the host reference makes the model visible to
  `check-runner-routing-consistency` as off-framework host logits. It cannot
  return device-resident logits because there is no NemotronH weight loader at
  all; the entry names W6 as what removes it.

### RED-first

With the five forward entry points reverted to W3's refusal, the suite is
**12 of 13 cases FAILING, `Status: FAILURE!`, exit 1**. Note the instrument
trap: the assertion count collapses **254 -> 16** and prints
`14 passed | 2 failed`, because the cases THREW — `grep 'assertions:'` reads a
fully red gate as nearly clean ([[doctest-assertions-line-hides-thrown-cases]]).
Read `Status:`. Tree restored, SHA-256 re-verified, worktree proven clean,
rebuilt back to green.

### Mutation proof (IMP-MUTATE) — 15 applied alone, rebuilt, run, restored

Every one RED; after each, the file was restored, its SHA-256 re-verified against
the baseline, and the whole worktree proven clean before the next.

| Mutation | Result |
|---|---|
| M1 `routed_scale` dropped from `MoeCombine` | FAILURE 3 cases / 14 assertions |
| M2 routed factor ALSO folded into the router weights (double-scale) | FAILURE 3 / 14 |
| M3 router weight dtype inherited from the model instead of f32 | FAILURE 3 cases, assertions 254 -> **188** (threw) |
| M4 attention scale `1/Dh` instead of `1/sqrt(Dh)` | FAILURE 3 / 9 |
| M5 attention causality dropped | FAILURE 3 / 11 |
| M6 conv silu activation dropped | FAILURE 3 / 13 |
| M7 `dt_softplus` dropped | FAILURE 3 / 11 |
| M8 `D` skip-connection dropped from the SSD scan | FAILURE 3 / 13 |
| M9 `dt_bias` dropped from the SSD scan | FAILURE 3 / 8 |
| M10 gated group norm collapsed to `n_groups=1` | FAILURE 3 / 8 |
| M11 `A = +exp(A_log)` (decay sign inverted) | FAILURE 6 cases, assertions 254 -> **87** (threw) |
| M12 relu² bypassed in the non-gated expert | FAILURE 4 / 15 |
| M13 B and C swapped in the conv-output split | FAILURE 3 / 10 |
| M14 final `norm_f` loses its residual fold | FAILURE 1 / 8 |
| M15 **self-certification**: bf16 band widened to 3.0 (makes the gate vacuous) | FAILURE 4 cases / 4 assertions — the non-vacuity REQUIREs fire |

M15 is the one that keeps the others honest: it proves the self-certification
added above actually detects a band that can no longer fail. M3 and M11 are the
instrument trap again — their assertion COUNTS fall because cases threw.

M8, M9 and M14 additionally could not be COMPILED in their first form: each left
a `Tensor` unused and `-Werror` rejected it. That is a real, if accidental,
second gate; the mutations were re-run with the variable consumed.

### Gate evidence

Local x86_64 CPU-only Release `-Werror` (GNU 13.3, Ninja), plus a Debug arm with
asserts unmasked. Disk is recorded beside every number because this box hit
**100% mid-run**: the first full build reported `FULL_BUILD_EXIT=1` with
`fatal error: error writing to /tmp/ccvKMX4g.s: No space left on device`, while
the harness notification for that same job read "exit code 0" — the wrapper's
status, not ninja's ([[unit-success-is-not-script-success]]). Space was reclaimed
and every number below comes from a re-run with headroom.

| Arm | Result | disk free |
|---|---|---|
| Release `-Werror`, clean full build | **exit 0, 0 `warning:` lines, 0 ENOSPC lines** | 8.9G / 98% |
| `test_nemotron_h_forward` (Release) | **13/13 cases, 254/254 assertions, `Status: SUCCESS!`** | 8.9G |
| same, `VT_FUSED_CHAIN_ADOPT=0` A/B | **13/13, 254/254, `Status: SUCCESS!`** | 8.9G |
| `test_nemotron_h_scaffold` (Release) | **12/12, 38285/38285, `Status: SUCCESS!`** | 9.1G |
| Debug (`-g0`, asserts unmasked) forward | **13/13, 254/254, `Status: SUCCESS!`** | 5.0G |
| Debug (`-g0`) scaffold | **12/12, 38285/38285, `Status: SUCCESS!`** | 5.0G |
| full `ctest -j4` | **100% tests passed, 0 failed out of 430** (skipped: `test_modelopt_mixed_precision_checkpoint`, `test_voxtral_e2e` — neither asset present) | 5.6G / 99% |

**The CUDA arm, on Jetson Thor (`kairos-4db2`, aarch64, sm_110).** Transferred by
`git archive` and md5-verified on both ends. Container `vllmcpp-build:aarch64`,
`--runtime=nvidia`, `NVIDIA_DISABLE_REQUIRE=1`; GPU visible inside as **NVIDIA
Thor, compute_cap 11.0**, nvcc **13.0.88**. Configured `-DVLLM_CPP_CUDA=ON
-DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF`, no CUTLASS — the log's
`CUTLASS not found ... NVFP4 GEMM + FA2 disabled` and three `DISABLED (no
requested arch in [110] provides it)` lines are CORRECT for sm_110, not a silent
fallback. Disk 472-476G free / 46% throughout.

| Thor arm | Result |
|---|---|
| build, 455 targets | **`BUILD_EXIT=0`, 0 `warning:` lines** (CUDA TUs incl. `cuda_ops.cu`, `cuda_paged_attn.cu`, `cuda_gdn.cu`, Marlin) |
| `test_nemotron_h_forward` | **13/13, 254/254, `Status: SUCCESS!`** — IDENTICAL counts to x86_64 |
| `test_nemotron_h_scaffold` | **12/12, 38285/38285, `Status: SUCCESS!`** — IDENTICAL to x86_64 |
| `test_ops_mamba2_ssd` | 12/12, **2095** assertions, SUCCESS |
| `test_ops_mamba2_gated_norm` | 12/12, **3723**, SUCCESS |
| `test_ops_mamba2_state_update` | 10/10, **5965**, SUCCESS |
| `test_ops_moe_nongated_relu2` | 12/12, 81, SUCCESS |

The three Mamba2 counts reproduce the GB10 precedent (12/2095, 10/5965, 12/3723)
exactly, on a different architecture and toolchain.

**What Thor does NOT prove, stated plainly.** `NemotronHForward` asserts a CPU
queue by design — the device/paged path is W6 — so Thor's value here is a
CUDA-ENABLED BUILD and a second architecture executing the suite, not GPU
execution of the forward. The vt primitives the forward composes
(`Mamba2ChunkScan`, `RmsNormGatedGroup`, `Mamba2StateUpdate`, `MoeRelu2`) DO have
CUDA arms and are gated above.

### Still owed after W4

The **weight loader** — nothing materializes the 18487 enumerated tensors, so no
checkpoint can be run and the forward refuses by name on every load. Then the MTP
head (W5), the e2e token gate against the committed goldens (W6), and the GGUF
arm (W7). The committed `nemotron_35_lightning_greedy/oracle.json` goldens are
W6's gate and were deliberately NOT consumed here. No speed claim is made or
implied by this W.

### 6c. Fresh-review residuals carried forward (2026-08-14)

W4's fresh review returned **PASS** and proved its central claim by experiment:
it reconstructed the inherited gate, made an attention block return all zeros,
and watched **both bf16 arms accept it** — the released checkpoint's dtype. The
repair holds; the same mutant now fails on all four arms. Four residuals are
recorded rather than left in a reviewer's report:

**R1 — the bf16 band is coarser than the defect class this file targets, and
the f32 arm must never be dropped as redundant.** `test_nemotron_h_forward.cpp`
compares bf16 at 3e-2 of peak. Demonstrated: a 2% attention-scale error
(`args.scale *= 1.02`) fails both f32 arms and **passes both bf16 arms**.
Corroborating: this file's own no-RoPE separation is **0.0210891**, *below* the
bf16 band, so a RoPE mis-port would pass the bf16 comparison. The f32 arm
(2e-4) and the dedicated no-RoPE guard catch both, which is why this is a
residual and not a defect — but the asymmetry is now on the record so nobody
prunes the f32 arm as duplicated coverage.

**R2 — the self-certification is a structural identity, not a tightness
measure.** Each comparison REQUIREs that its band reject all-zeros, which with
a peak-relative band reduces exactly to `rel < 0.5`. It eliminates the precise
defect it was written for and nothing more; at `rel = 0.49` it still passes
everywhere. Honestly scoped, not a general guarantee of band quality.

**R3 — the routed-scale CALL SITE is genuinely ungated.** Folding
`routed_scaling_factor` into the router instead of `MoeCombine`'s
`routed_scale` gives **13/13 cases, 254/254 assertions, SUCCESS** — the
model-level gate cannot see it (peak-relative separation 1.91e-07). It is
arithmetically equal post-renormalisation, which `layer.py:291-300` states
outright by forcing the router factor to 1.0 "so it ends up being a nop". The
bitwise instrument at `tests/vt/test_ops_moe_nongated_relu2.cpp:270` gates
`vt::MoeCombine`'s own semantics, **not** this call site's choice — a
distinction the earlier write-up blurred. Unavoidable here (the model-level
reference is `double`, so no bitwise comparison exists), and recorded as
uncovered rather than implied to be covered.

**R4 — two comment magnitudes state no denominator.** "3.7e-5 relative" for the
fold measures 1.91e-07 peak-relative; "25.3x the signal" for `scale_logits`
measures 0.568. The qualitative claims are right and were independently
verified; the numbers appear to be mean-relative or from an earlier fixture.

## 6d. The WEIGHT LOADER — the checkpoint runs (2026-08-14)

§6b closed with "the **weight loader** — nothing materializes the 18487
enumerated tensors, so no checkpoint can be run and the forward refuses by name
on every load", and §7 said the loader "§4's table does not name as a W of its
own and should". This is that brick, built on `row/MODEL-NEMOTRON-H-LOADER` off
`22367c551`.

**RED first, off the real checkpoint.** Resolved through
`parity::Nemotron35LightningSnapshot()` (content-pinned, #569) to
`/mnt/nas_share/checkpoints/nemotron-3.5-lightning-30b-nvfp4`: `Status:
FAILURE!`, exit 1, `THREW: NemotronHForCausalLM forward: host weights are not
materialized`, peak RSS after "load" **29 MiB**. Note the instrument trap in the
same output — `assertions: 3 | 3 passed | 0 failed` beside a red gate, because
the case THREW ([[doctest-assertions-line-hides-thrown-cases]]).

### The design decision, and why it is arithmetic rather than taste

**Every weight is held in the memory format the checkpoint SHIPS it in.** Not
because widening is untidy, but because a dequantize-at-load loader does not fit
on any box this project owns: the 5888 routed-expert projections alone are
29.4e9 parameters, **16.5 GB packed against 58.7 GB at bf16**. On a
unified-memory box that is not a failed load, it is a reboot
([[gb10-unified-memory-oom-reboots-box]]).

Both of those are **decimal GB**, and the unit was wrong here and in two headers
until the fresh review of this row caught it (`16.5 GiB`). The packed figure is
14.7e9 bytes of nibbles (0.5 B/param) plus 1.84e9 bytes of group scales (1 B per
16 params) = 16.54e9 bytes, which is **15.4 GiB**. A number quoted three times
starts being treated as measured ([[a-number-quoted-often-becomes-treated-as-measured]]),
so it is written out rather than left to be re-derived.

Measured host mirror: **18013 MiB**, against 18013 MiB read out of the shards —
the mirror is the checkpoint, not a widened copy of it. Peak RSS **17.70 GiB**
after the load and **18.38 GiB** at the end of a forward.

The consequence is that the HOST reference forward, which composes
`vt::MatmulBT` and has no NVFP4 and no FP8 entry point, widens a quantized
operand TRANSIENTLY at the GEMM call site through the shared
`model_loader/nvfp4_dequant.h` seam. That is a **declared** arm, named in
`nemotron_h_loader.h` and `nemotron_h.cpp` and reported by the load report — not
a silent fallback of the kind §8's stop condition forbids. The quantized GEMMs
remain `kMoeGroupedGemmNvfp4Marlin` and the fp8-linear registration, which W6
selects on the device path. `NemotronHOwned::View` now REFUSES a non-dense
weight by name, so a packed buffer cannot be reinterpreted as the model dtype by
a caller that did not think about it.

One consequential wiring change followed: the routed-expert loop in
`NemotronHMoeMixer` now visits its (token, slot) pairs **expert-major** rather
than token-major, so one dequant of an NVFP4 expert serves all of that expert's
rows. Each pair's own `NonGatedExpert` call is unchanged (one row against one
expert), and each pair writes a disjoint `expert_out` slot and reads nothing
another wrote, so the ORDER cannot change the result; `test_nemotron_h_forward`
stays 13/13 and 254/254.

### The structural gate, as hard numbers

`tests/vllm/models/test_nemotron_h_loader.cpp`. It is gated STRUCTURALLY and not
only by tokens, because a checkpoint read as uniform NVFP4 stays numerically
plausible and still matches tokens while moving the wrong bytes.

| Row | Measured |
|---|---|
| enumerated / in `model.safetensors.index.json` | **18487 / 18487** |
| materialized + deferred | **18217 + 270 = 18487** |
| deferred BY NAME (MTP tower, W5) | **270**, every tag naming W5 |
| NVFP4 W4A16 g16 | **5935 projections / 17805 tensors** |
| FP8 W8A8 static | **46 projections / 138 tensors** |
| fp8 KV scales (`k_scale`/`v_scale`) | **12** |
| unquantized bf16 / f32 on disk | **216 / 46** |
| the five scheme rows sum to `materialized` | 18217 == 18217 |
| widenings (bf16 on disk, f32 in memory) | **69**, and no more |
| host bytes | **18,888,922,112** (17.59 GiB) |

The `{W4A16_NVFP4: 5935, FP8: 46}` split is exactly the histogram W1 measured
over all 5981 `quantized_layers` entries, now confirmed against the TENSORS
rather than the config. The 69 widenings are the three f32-by-contract SSM
scalars (`A_log`, `D`, `dt_bias`) on 23 mamba layers — upstream's own polarity
(`-torch.exp(self.A_log.float())`) and what `vt::Mamba2ChunkScan` validates.

**One finding worth carrying: `A_log`, `D` and `dt_bias` ship BF16 on disk.**
The forward requires them f32, so the loader widens; a loader that inherited the
model dtype instead produces a numerically plausible model. M3 below is the
instrument.

### EVIDENCE, not the W6 token gate

For each of the three committed oracle prompts, ONE forward over its
`prompt_token_ids` and the argmax of the last position against the golden's
FIRST generated token:

| Prompt | argmax | oracle |
|---|---|---|
| `The capital of France is` | 6993 | 6993 |
| `Write the first five Fibonacci numbers:` | 1032 | 1032 |
| `Explain what a state space model is, in one sentence:` | 1349 | 1349 |

**3/3.** W6 still owns the token gate (identical prompts, counts, batching and
sampling, oracle identity asserted, full 32-token greedy decode); this consumes
one token per prompt and makes no speed claim. It is here because it is the only
check that can fail for a reason the structural gate cannot see — every count can
be right while a group scale is transposed or a nibble order is flipped.

The doctest 2.5.2 `const char*` trap this row already repaired once (§"W1
land-prep" LOW-2) bit again in the first run of exactly this line: the verdict
printed as `oracle 69931`, the `1` being doctest rendering a `const char*`
lvalue. Bound to a `std::string`.

### Mutation proof (IMP-MUTATE)

Each applied ALONE to the restored tree, rebuilt, run against the real
checkpoint, then restored and the file's SHA-256 re-verified.

| Mutation | Result |
|---|---|
| M1 mamba `in_proj` read as NVFP4 instead of FP8 (the "uniform NVFP4" defect) | **FAILURE!** — THREW `'backbone.layers.0.mixer.in_proj.weight' ships dtype F8_E4M3, not the U8 its scheme declares`; 1 case / 0 passed, assertions 2 |
| M2 the 12 fp8-KV scales silently dropped | **FAILURE!** — THREW `'backbone.layers.5.mixer.k_proj.k_scale' (consumer 'attn.k_scale[fp8-kv]') is enumerated but no host slot claimed it`; assertions 2 |
| M3 `A_log`/`D`/`dt_bias` inherit the model dtype (no widening) | **FAILURE!** — `rep.widened_tensors == 69` red, then THREW `weight 'mixer.A_log' has the wrong dtype for this arm`; assertions 35, 1 failed |
| M4 NVFP4 nibble order flipped to `kHighFirst` | **FAILURE!** — **0/3** goldens (argmax 66822 / 60300 / 31645 vs 6993 / 1032 / 1349); assertions 46, exactly **1** failed |
| M5 `weight_scale_2` read (so the accounting stays right) and then ignored | **FAILURE!** — **0/3** goldens (argmax 1321 / 2142 / 1321); assertions 46, exactly **1** failed |

M1's first form — BOTH mamba projections switched — could not be COMPILED:
`-Werror=unused-function` rejected the now-unreferenced `LoadFp8`. That is a
real, if accidental, second gate, the same one W4's M8/M9/M14 hit; it was re-run
with `out_proj` left on `LoadFp8`.

**M4 and M5 are the pair that keeps the golden arm honest.** Both leave EVERY
structural count correct — 45 of 46 assertions still pass, and the one that
fails is `matched == total`. A loader gate built only out of counts would have
called both of them clean, and both are exactly the "still numerically
plausible, still the right shape, wrong bytes" class §1 warns about. M1-M3
conversely are invisible to the golden arm, because they refuse before a token
exists.

**Restoration.** After each, the file was rewritten from the captured original
and its SHA-256 re-verified; `git status --porcelain` showed only the spec and
`docs/FEATURES.md` edits this section is part of.

The filenames in that record were STALE, and the fresh review of this row was
right to refuse a proof naming a file that does not exist. Re-anchored at HEAD:

| Mutated file, as the record named it | What it is now | SHA-256 |
|---|---|---|
| `nemotron_h_loader.cpp` (M1–M3), the TU as it stood at `c4029deed` | `nemotron_h_weights.cpp` | `186ae4cea4d7…` **pre-rename** |
| `nemotron_h_weights.cpp` at the landed head `9bffa2b60` | same file | `f8a87b15fb…` |
| `nemotron_h.cpp` (M4, M5), unchanged since `cf95f59b8` | same file | `554d1c7fc65b…` |

M1–M3 ARE anchored — `186ae4cea4d7…` is a real, reachable SHA
(`git show c4029deed:src/vllm/model_executor/models/nemotron_h_loader.cpp | sha256sum`).
What happened is that `08a65696d` then folded that TU into
`nemotron_h_weights.cpp` as a documented **pure move** (its only edit being
`Refuse` → `RefuseLoad` at 27 call sites), so the record kept a filename the
tree no longer has. The mutations were NOT re-run after the move, and this
section says so rather than implying they were. M4/M5's anchor never moved and
the reviewer reproduced both argmax triples exactly.

`f8a87b15fb…` is the sha of the reviewed head. The F1 repair below changes that
file, so it is recorded as the anchor of the mutation campaign, not of `main`.

### Gate evidence

Local x86_64 CPU-only host (GNU 13.3, Ninja, `VLLM_CPP_CUDA=OFF`), disk recorded
beside every number because this box has hit 100% mid-run before and a build
that ENOSPCs leaves the PREVIOUS binary in place
([[stale-binary-prints-green-status]]).

| Arm | Result | disk free |
|---|---|---|
| Release `-Werror`, clean full build | **exit 0, 0 `warning:` lines, 0 `No space left` lines**, 891/891 targets | 53G / 88% |
| `test_nemotron_h_loader` (Release, live checkpoint) | **1/1 cases, 46/46 assertions, `Status: SUCCESS!`**, 10:07 wall, VmHWM 19,270,444 KiB | 36G / 92% |
| `test_nemotron_h_forward` (Release) | 13/13, 254/254, `Status: SUCCESS!` | 53G |
| `test_nemotron_h_scaffold` (Release) | 12/12, 38285/38285, `Status: SUCCESS!` | 53G |
| Debug (`-g0`, asserts unmasked) forward | **13/13, 254/254, `Status: SUCCESS!`** | 36G |
| Debug (`-g0`) scaffold | **12/12, 38285/38285, `Status: SUCCESS!`** | 36G |
| Debug (`-g0`) **loader, live checkpoint** | **1/1, 46/46, `Status: SUCCESS!`**, 3/3 goldens, 15:21 wall, VmHWM 19,277,028 KiB | 26G / 95% |
| full `ctest -j4` | **449 of 451 passed**, then **450 of 451**: `test_engine_core_proc` failed under `-j4` and passes ALONE (14/14, 113/113, exit 0) — the known starvation set, and this box was running the Debug loader arm at the time. `test_op_parity` FAILED, **pre-existing on the base** (below). Skipped: `test_modelopt_mixed_precision_checkpoint`, `test_voxtral_e2e`; `test_nemotron_h_loader` passed vacuously with no `CHECKPOINT_ROOT` in that shell | 26G / 94% |

The forward and scaffold counts are IDENTICAL to W4's (13/254, 12/38285), which
is the evidence that the expert-major reorder is result-neutral rather than an
assertion that it is.

**Jetson Thor (`kairos-4db2`, aarch64, sm_110) — a CUDA build AND the real
checkpoint.** Transferred by `git archive` and md5-verified on both ends
(`1cdc92214540646cc1fb7c9e87c87b23`); the 20.1 GiB checkpoint staged to
`/home/mudler/nemo-loader/ckpt` with its `.cache/huggingface/download/*.metadata`
sidecars intact, so the CONTENT pin resolves there too. Container
`vllmcpp-build:aarch64`, `--runtime=nvidia`, `NVIDIA_DISABLE_REQUIRE=1`, nvcc
13.0.88, `-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=110
-DVLLM_CPP_TRITON=OFF`, no CUTLASS — the six `DISABLED (no requested arch in
[110] provides it)` lines are CORRECT for sm_110, not a silent fallback. Disk
408-439G free / 50-54% throughout; RAM 122 GiB, and the box was loaded ALONE.

| Thor arm | Result |
|---|---|
| build | **`BUILD_EXIT=0`, 0 `warning:` lines, 0 `No space left` lines** |
| `test_nemotron_h_forward` | **13/13, 254/254, `Status: SUCCESS!`** — identical to x86_64 |
| `test_nemotron_h_scaffold` | **12/12, 38285/38285, `Status: SUCCESS!`** — identical to x86_64 |
| `test_nemotron_h_loader`, **live checkpoint at `/w/ckpt`** | **1/1, 46/46, `Status: SUCCESS!`**, `LOADER_EXIT=0`, **3/3** goldens, 3:00 wall |
| peak RSS on Thor | **17.85 GiB** after load, **18.53 GiB** at the end |

Two cross-architecture agreements worth naming, because neither was arranged.
`host bytes: 18013 MiB, source 18013 MiB` is byte-for-byte what x86_64 reported,
and the logits range is **`[-9.4375, 8.75]` on both**. The 3:00 wall against
x86_64's 10:07 is the storage, not the CPU: Thor reads the checkpoint off local
NVMe and the x86 box reads it over SMB from the NAS.

`/usr/bin/time` is NOT in that container — the first attempt exited **127** and
would have read as a failed run to anyone grepping only for `SUCCESS`. The RSS
above is the test's own `/proc/self/status` `VmHWM`, which is why it has one.

**One honestly weak property, recorded rather than discovered — and since
REPAIRED (§6e F2b).** With no checkpoint the gate emitted a `MESSAGE` naming the
missing export and RETURNED, so CTest recorded `Passed 0.00 sec` with **zero
assertions** — indistinguishable from a real pass without reading the log. The
convention was inherited from `test_nemotron_h_scaffold`'s live case, and
matching it beat inventing a second one; the alternative is
`test_modelopt_mixed_precision_checkpoint`'s exit-77, which needs a custom
`main`. The fresh review was right that recording it is not the same as fixing
it, and §6e replaces it with a named, counted skip plus a closing accounting
case.

**`test_op_parity` is RED on the base and not this row's.** `RunGoldenPass`
(`tests/parity/test_op_parity.cpp:1852-1860`) walks every subdirectory of the
goldens root and does `std::string op = m["op"];` on any `manifest.json` it
finds. `tests/parity/goldens/minimax_music3_oracle/manifest.json` has no `op`
key — its keys are `captured_on, checkpoint, environment, generated_by, issue,
model, oracle, request, result, spec, spec_disagreements, spec_facts` — so the
pass throws `[json.exception.type_error.302] type must be string, but is null`.
That manifest landed at `34dc57876` (`oracle(MODEL-MUSIC-MUSIC3)`, #672 / #708),
which `git merge-base --is-ancestor 34dc57876 22367c551` confirms is an ancestor
of this branch's base, and this branch's diff touches no file under
`tests/parity/` or `src/vt/`. Reported rather than repaired: `tests/parity/` is
outside this task's authority, the fix belongs to MODEL-MUSIC-MUSIC3 (either an
`op` key or the `manifest.json` renamed so the op walker skips it as the
tokenizer golden dirs already are), and repairing a pre-existing break inside a
scoped loader change would hide it — the same call §5d finding 5 made.

## 6e. The fresh review of §6d, and the repairs (2026-08-14)

PR #752 was reviewed at the immutable head `9bffa2b60` by an agent that neither
wrote nor gated it. The review **confirmed the substance** — 18217 of 18487
enumerated tensors materialized, 270 `mtp.*` deferred by name, every weight in
its packed form (host bytes `18,888,922,112`, re-derived bit-exact from the 52
safetensors headers), 5935 NVFP4 W4A16 g16 / 46 FP8 W8A8 / 12 fp8 KV scales /
216 BF16 + 46 F32 / 69 widenings, and 3/3 first-token agreement on x86_64 and
Thor — and returned two MEDIUM findings, three LOW and two NIT. All seven are
repaired below; none was declined.

### F1 — a pointer the loader may not form

`CopyDense` read the safetensors mapping through
`reinterpret_cast<const uint16_t*>(t.data)` and
`reinterpret_cast<const float*>(t.data)`. `StTensor::data` is
`8 + <JSON header length> + <sum of the preceding tensors' sizes>` and **none of
those three terms is required to be even**, so forming either pointer is
undefined whether or not the access faults. The same file's scalar read already
used `memcpy`, so it was internally inconsistent.

This is the class [#627](https://github.com/mudler/vllm.cpp/issues/627) tracks,
whose THIRD recurrence closed on `main` the same day at `fc903b8dd` (#674) after
`main` had sat RED on `sanitize-cpu (address,undefined)` since #641. That repair
is mirrored exactly: reads go through `vt::LoadUnaligned` — the seam #301 left
behind, already used by eight sibling loaders — and the two bulk `memcpy`s drop
their typed pointer entirely (`memcpy` never needed one). No sanitizer
configuration was touched and no scope was widened.

Anchors re-derived at HEAD and asserted unique (count == 1), because recorded
line numbers go stale within a PR: `nemotron_h_weights.cpp:455` (BF16 arm) and
`:473` (F32 arm) at the reviewed head.

**Evidence.** A standalone probe (`-fsanitize=address,undefined
-fno-sanitize-recover=all`, GNU 13.3) over a deliberately ODD address, running
both read forms of this file:

| Form | Result |
|---|---|
| the shipped `reinterpret_cast<const uint16_t*>` | **exit 1**, `runtime error: load of misaligned address 0x521000000101 for type 'const short unsigned int', which requires 2 byte alignment` |
| `vt::LoadUnaligned<uint16_t>` | **exit 0**, `data % 2 = 1`, sum `2006464` **matching the memcpy oracle exactly** |

The reviewer measured this as LATENT rather than live — 0 of 216 BF16 and 0 of
6085 F32 tensors land misaligned on this particular checkpoint — and that is
recorded rather than used as a reason to leave it: the loader builds for
`build-test-cpu-arm64`, and UBSan could never have caught it in CI because of
F2. A project-configured `-DVLLM_CPP_SANITIZE='address,undefined'` build of the
three Nemotron-H targets is clean (below).

**Not changed, and why:** the remaining `reinterpret_cast`s in `nemotron_h.cpp`
and the two write-side casts in `nemotron_h_weights.cpp` are over
`std::vector<uint8_t>::data()`, which the default allocator returns suitably
aligned for any scalar. They are not views into a mapping and are not in F1's
class.

**No overlap with PR #815 / `row/FIX-UNALIGNED-LOADERS-772`, checked rather than
assumed.** That branch closes the same CLASS at four other sites —
`voxtral.cpp:51`, `voxtral.cpp:344`, `qwen3_vl.cpp:78`, `qwen3_5_mtp.cpp:71`,
plus `minimax_h3_vae_loader.cpp`. Its complete file list contains no
`nemotron_h_*` file at all, and `git log -S'nemotron_h_weights'` over
`origin/main..origin/row/FIX-UNALIGNED-LOADERS-772` is empty. The reason is
structural, not luck: #772's sweep was taken over `main`, and this site does not
exist on `main` — it arrives with PR #752 itself. Established with a POSITIVE
CONTROL rather than from a silent grep ([[never-assert-absence-from-a-failed-grep]]):
the same regex that is silent on the repaired `nemotron_h_weights.cpp` FIRES on
`voxtral.cpp:51`, `voxtral.cpp:344` and `qwen3_vl.cpp:78` — exactly #815's set —
so the instrument works. The two changes are complementary and together close
the class. Whichever lands second must re-resolve `docs/FEATURES.md` and
`tests/CMakeLists.txt` BY KEY, the only two files both touch.

### F2 — the guard was correct but UNARMED, and none of §6d's new code ran in CI

The review proved the seams were unreachable without the 20.1 GiB checkpoint:
deleting the `VT_CHECK` in `NemotronHOwned::View` left the live gate at 46/46
with 3/3 goldens and both offline suites green, and a `ctest` run with no
`CHECKPOINT_ROOT` recorded `test_nemotron_h_loader ... Passed 0.00 sec` with
**zero assertions** — byte-for-byte what a real pass looks like from outside.
The guard is not decorative: with it removed, `View` hands out a 128-element
bf16 tensor over a 64-byte NVFP4 buffer, **192 bytes out of bounds**.

**(a) `tests/vllm/models/test_nemotron_h_quantized_forms.cpp`** — a new OFFLINE
gate needing no checkpoint, registered unconditionally in `tests/CMakeLists.txt`
so it runs on every CI arm. It asserts that `View` refuses a non-dense weight BY
NAME (message names the form and points at `DenseBf16`), that a dense weight is
still viewed unchanged, and that `DenseBf16` reproduces a dequant derived
**independently from the upstream formula** — the E2M1 table, an fp8-e4m3
decoder and a bf16 RNE round all written out in the test from their format
definitions rather than called out of `vt`/`vllm`, and anchored by hand
(0x38 → 1.0, 0x40 → 2.0, nibble 0x7 → 6.0, bf16(1.5) = 0x3FC0). Every fixture
value is dyadic, so every expected result is EXACT in bf16 and the comparison is
on the BIT PATTERN — `doctest::Approx`'s ~1.19e-5 absolute floor
([[doctest-approx-scale-term-floor]]) never enters. The three properties a
counts-only gate cannot see are asserted individually: the nibble order (element
2j is the LOW nibble), the per-16 group scale (two groups of one row, different
fp8 scales), and `weight_scale_2` MULTIPLIED (not reciprocated, not ignored). A
fifth case reaches the file-private `DenseFor`/`DenseCopy` through
`NemotronHMlpMixer` and proves a quantized weight is actually widened at the
GEMM call site — against a dense arm built from the INDEPENDENT reference, so a
dequant defect cannot cancel itself out, and with the weight still packed
afterwards.

**RED-before, by mutation.** Every result carries the compile exit status beside
it, because a mutation that fails to BUILD reads as a passing test
([[mutation-build-failure-reads-as-a-passing-test]]), and `Status:` is grepped
as well as `assertions:`, because a thrown case prints "N passed | 0 failed"
beside a red status ([[doctest-assertions-line-hides-thrown-cases]]). The binary
SHA is printed too, so a stale binary cannot print a green status
([[stale-binary-prints-green-status]]) — all four differ.

| Mutation (applied alone to `nemotron_h.cpp`, restored + sha-verified after) | BUILD_EXIT | binary sha | Result |
|---|---|---|---|
| baseline (unmutated) | 0 | — | 5 cases / **130 assertions**, `Status: SUCCESS!`, exit 0 |
| **M4′** nibble order → `kHighFirst` | **0** | `f7e75e690a62` | **`Status: FAILURE!`**, 3 passed / **2 failed**, 4 assertions failed, exit 1 — **96 of 96** elements differ, and the probe pair reads exactly swapped (`16576` vs `16128`) |
| **M5′** `weight_scale_2` ignored (passed as `1.0F`) | **0** | `c797cc44ec4b` | **`Status: FAILURE!`**, 3 passed / 2 failed, **36** assertions failed, exit 1 |
| `View`'s `VT_CHECK` neutralized to `VT_CHECK(true, …)` | **0** | `0303eeaeff7d` | **`Status: FAILURE!`**, 4 passed / 1 failed, 5 assertions failed, exit 1 |
| FP8 `input_scale` APPLIED (it must be carried, not applied) | **0** | `630495efa1a9` | **`Status: FAILURE!`**, 4 passed / 1 failed, 3 assertions failed, exit 1 |

M4′ and M5′ are §6d's M4 and M5, the pair that a counts-only gate calls clean.
They previously needed the 20.1 GiB checkpoint and three oracle prompts to
detect; they are now caught in **0.01 s with no checkpoint at all**.

**(b) the checkpoint-gated skip is now LOUD, NAMED and COUNTED.**
`test_nemotron_h_loader` declares its checkpoint-gated cases in a
`CheckpointGatedCases()` registry — the `PendingRunnerOps()` idiom
(`tests/parity/test_op_parity.cpp:1834`) — records a verdict for each, asserts
in the skip path that the case is declared and that its reason is non-empty, and
closes with an accounting case that REQUIREs every declared case reached exactly
one verdict and MESSAGEs the tally.

With `CHECKPOINT_ROOT` unset it now reports **2 cases / 7 assertions,
`Status: SUCCESS!`** and logs `checkpoint-gated cases: 0 ran, 1 skipped, of 1`,
instead of `Passed 0.00 sec` with nothing on the record. The accounting is
itself armed:

| Mutation to `test_nemotron_h_loader.cpp` | BUILD_EXIT | binary sha | Result |
|---|---|---|---|
| the skip is announced but not RECORDED | 0 | `0fc726deb80f` | **`Status: FAILURE!`**, 0 of 2 cases passed, 4 of 6 assertions failed |
| the pre-repair SILENT early return restored | 0 | `89cb559244b8` | **`Status: FAILURE!`**, 1 of 2 cases passed, **2 of 2 assertions failed** |

### The LOWs and NITs

**L1 — `LoadMamba` did not branch on `quantized`.** `ClaimMamba` deliberately
supports the unquantized case (hard-coding the FP8 companions there enumerated
92 tensors a released bf16 checkpoint does not ship), but the loader called
`LoadFp8` unconditionally, so such a checkpoint refused with `'…in_proj.weight'
ships dtype BF16, not the F8_E4M3 its scheme declares` — a DTYPE message for
what is really the declared scheme. Repaired by branching, exactly as
`LoadExpert`/`LoadMlp` and the enumeration do.

**That arm is REACHABLE and consistent with its enumeration, and it is NOT
GATED — recorded here rather than left to be found.** No released bf16
NemotronH checkpoint is within this project's reach and no synthetic one exists,
so the branch is argued from the enumeration (`ClaimFp8`'s own `quantized`
gate) rather than measured. Two things are owed with it: a checkpoint or
synthetic fixture that executes it, and `mamba_proj_bias`, which is enumerated
(`in_proj.bias`/`out_proj.bias`) with no host slot, so a checkpoint that sets it
still refuses through the accounting path. The released checkpoint sets it
false. `docs/FEATURES.md` deliberately does NOT claim the unquantized arm: an
ungated branch is visible debt, not a supported surface.

**L2 — stale scope statements.** Both
`tests/vllm/models/test_nemotron_h_loader.cpp:36` ("It consumes no golden…") and
`tests/CMakeLists.txt:555` ("no golden is consumed") contradicted the file,
which consumes `oracle.json` at :268-316 while the CMake block defines
`NEMOTRON_H_GOLDENS_DIR`. Both now say what the code does: ONE forward per
prompt and the argmax of the last position against that prompt's FIRST token;
the full 32-token greedy decode stays W6's.

**L3 — the restoration proof named a nonexistent file.** Repaired in §6d's
Mutation-proof table above; `186ae4cea4d7…` turned out to be a REAL sha, of the
pre-rename TU at `c4029deed`.

**N1 — "16.5 GiB" is 16.5 GB.** Repaired in `nemotron_h_loader.h:39`,
`nemotron_h_forward.h:92` and §6d above, with the arithmetic written out.

**N2 — "counted … by the loader's report".** The report counts quantized
WEIGHTS (5935 + 46), which is the population the widening applies to, not
dequant EVENTS, which are per GEMM call and a property of the workload.
`nemotron_h.cpp:205-206` now says that.

### Carried forward — recorded, not repaired

Two limits the review established, which belong in the record because neither is
visible from the code or the counts:

1. **The FP8 arm is weight-only on the host path.** `input_scale` is carried and
   never applied, so it is NOT a bit-mirror of vLLM's W8A8 *static* GEMM, which
   quantizes the activation with it. Nothing on the host path quantizes an
   activation, so applying it here would scale the product by a factor upstream
   applies to the OTHER operand. W6's device path is where the activation scale
   becomes live, and where the mirror claim can be made.
2. **The golden arm's reach is 3 prompts x 1 token.** That touches at most 78 of
   the 128 experts per layer, never the dense `mlp` block (no released in-scope
   checkpoint ships one), and never the MTP tower (deferred to W5). It is
   evidence, not the token gate; W6 owns the token gate.
3. **THE WEIGHTS LOAD; THE MODEL IS NOT REACHABLE FROM THE PUBLIC ABI (#810).**
   An independent investigation established that NemotronH does not run end to
   end through `include/vllm.h` — not on `main`, and not with this branch
   merged. Both refuse at `src/vllm/v1/worker/gpu/runner.cpp:525`,
   `runner: Qwen3.5 MambaSpec shapes disagree with model config`, a line this
   branch does not touch. Proven on the real 21 GiB checkpoint on GB10: the
   weights load fine (17.7 GiB RSS) and then ENGINE CONSTRUCTION refuses. So
   "the real checkpoint runs" in §6d means the HOST REFERENCE FORWARD runs on
   real weights, and it must not be read as "the server runs it".

   That single `VT_CHECK` is the ONLY load-time blocker — neutering it alone
   made `vllm_engine_load` succeed and reach the forward. **That is not a fix
   and must not be done**: `ForwardNemotronHForCausalLM` ignores `attn_kv`,
   `gdn_state`, `gdn_meta` and `num_reqs`, so a server past that check emits
   silently wrong tokens from decode step 2 — the failure mode a token gate on
   a 1-token prompt cannot see. The refusal is currently the only thing making
   the gap visible. #810 owns it; W6 owns the paged/device runner. Explicitly
   OUT OF SCOPE here.

### A merge that was CLEAN and did not BUILD the behaviour either side had (#818)

Found by re-running the full gate after merging `origin/main` — not by reading
the diff, which showed no conflict at all because the two changes touch
different files.

`#784` (`b1cd4d8f6`, part of #730) rewrote `test_nemotron_h_scaffold`'s refusal
subcase to call the REAL `reg.factory->load_weights(reg, config, source)`
instead of downcasting a fabricated `struct StubModel : vllm::LoadedModel`.
UBSan was right about the stub, and the rewrite is correct **on `main`**, where
`LoadNemotronHForCausalLM` reads only `source.kind` and never touches
`source.safetensors`. §6d then gave NemotronH a loader that refuses an empty
source BY NAME. Merged, #784's `ModelSource source; source.kind =
kSafetensors;` hits exactly that refusal and the subcase THREW before ever
reaching the forward:

```
test_nemotron_h_scaffold.cpp:666: ERROR: test case THREW exception:
  Model architecture NemotronHForCausalLM: the safetensors source carries no shards
```

Neither parent is red. This is [[merge-tree-clean-is-not-builds]], and it is
repaired here because here is where the two sides meet. The subcase is SPLIT
rather than either side's guarantee deleted:

- an empty safetensors source REFUSES AT LOAD, by name (`NemotronHForCausalLM`,
  `carries no shards`). The guarantee moved EARLIER and is asserted where it now
  lives, which is a stronger claim than the one it replaces;
- the forward still refuses on unmaterialized weights, reached through the
  exported `vllm::NemotronHForward` on a default-constructed
  `NemotronHHostWeights`. That state is no longer reachable through the factory
  at all now that the loader exists — `load_weights` either materializes or
  refuses — so it is asserted on a REAL `NemotronHHostWeights`. #784's substance
  is kept in full: no `StubModel`, no downcast onto an object that never was a
  `NemotronHLoadedModel`, no UB.

Both halves are ARMED, not decorative. Compile exit and binary sha printed
beside each, and the two binaries differ:

| Mutation | BUILD_EXIT | binary sha | Result |
|---|---|---|---|
| the `carries no shards` refusal replaced by `return model;` | 0 | `4e6cdbd93478` | **`Status: FAILURE!`**, 11 of 12 cases, 2 of 38289 assertions failed |
| `VT_CHECK(host.materialized, …)` neutralized to `VT_CHECK(true, …)` | 0 | `84179116c90c` | **`Status: FAILURE!`**, 11 of 12 cases, 2 of 38289 assertions failed |

The suite goes from 38285 assertions to **38289** — the split adds four, and the
case count stays 12, which is the number to watch: a changed CASE count is
signal ([[doctest-assertions-line-hides-thrown-cases]]).

### Re-gate evidence (the repair head)

Local x86_64 CPU-only host (GNU 13.3, Ninja, `VLLM_CPP_CUDA=OFF`), disk recorded
beside every number ([[enospc-makes-checkers-emit-false-policy-refusals]]).

FINAL, at the merged head over `origin/main` @ `ca01719e6` -- re-run in full
because `origin/main` moved again before the push AND landed a checker this
branch had never been measured against (`scripts/check-commit-style.py`,
POLICY-SINGLE-PR-AND-STYLE `ddff09093`):

| Arm | Result | disk free |
|---|---|---|
| Release `-Werror`, **clean full build from an empty tree** | **`BUILD_EXIT=0`, 0 `warning:` lines, 0 `error:` lines, 0 `No space left` lines**, 1395/1395 targets | 100G / 77% |
| full `ctest -j4` | **471 of 471 PASSED, `CTEST_EXIT=0`**, 111 s. Skipped: `test_modelopt_mixed_precision_checkpoint`, `test_voxtral_e2e`. `test_op_parity` PASSES (the `main`-inherited red #755/#672 is gone at this base); `test_cpu_threadpool` passes on an idle box | 100G / 77% |
| the four Nemotron-H suites in that run | scaffold **12/12**, forward **13/13**, loader **2/2**, quantized_forms **5/5**, all Passed | 100G |
| `scripts/check-commit-style.py --range ca01719e6..HEAD` (NEW gate) | **`OK: commit writing style`** | — |
| `scripts/check-commit-trailers.py --range origin/main..HEAD` | **`OK: commit trailer contract`** | — |
| `scripts/agent-preflight.sh` | **All gates green** — including `test_cpu_x86_llamacpp_floor`, which had RED earlier in this session at load average 91-130 and passes on the idle box, confirming it environmental rather than this row's ([[cpu-x86-floor-test-reds-under-box-load]]) | 103G |

The earlier run of this same gate, on a contended box, is kept because a
negative result is a result: `ctest` 468 of 469 with `test_nemotron_h_scaffold`
red — that is how #818 was found, and it was found by RE-RUNNING the gate rather
than by reading the merge diff, which showed no conflict at all.
| `test_nemotron_h_scaffold` (after the #818 repair) | **12/12 cases, 38289/38289 assertions, `Status: SUCCESS!`** | 23G |
| `test_nemotron_h_forward` | **13/13, 254/254, `Status: SUCCESS!`** — identical to W4's | 23G |
| `test_nemotron_h_loader`, `CHECKPOINT_ROOT` UNSET | **2/2, 7/7, `Status: SUCCESS!`**, logs "0 ran, 1 skipped, of 1" | 23G |
| `test_nemotron_h_quantized_forms` (NEW) | **5/5, 134/134, `Status: SUCCESS!`** | 23G |
| `ctest -R '^test_nemotron_h'` after the repair | **4 of 4 passed** | 23G |
| `-DVLLM_CPP_SANITIZE='address,undefined'`, the three Nemotron-H targets | build exit 0 / 0 warnings; `quantized_forms` 5/5 130/130, `loader` 2/2 7/7, `forward` 13/13 254/254, all `Status: SUCCESS!`, exit 0, **no sanitizer finding** | 41G |
| `scripts/agent-preflight.sh --staged` | all gates OK except `test_cpu_x86_llamacpp_floor`, which is ENVIRONMENTAL and base-inherited: it reproduces on the SHARED CHECKOUT at `main` with `NO_QUIET_WINDOW after 30s (busy=102% load=62.28 88.03 93.73)` on a box at load average 91-130 ([[cpu-x86-floor-test-reds-under-box-load]]) | — |

The three test TUs edited after the clean full build (`test_nemotron_h_scaffold`,
`test_nemotron_h_loader`, `test_nemotron_h_quantized_forms`) are leaf `.cpp`
files with no dependents, and each was recompiled from source afterwards at
`-Werror` with 0 warnings — a TU is compiled whole or not at all, so that is a
clean compile of each, not an incremental one. No header changed after the clean
build.

The live checkpoint gate was deliberately NOT re-run here: the operator runs it
on GB10, and this box finished the run at **95% full**, where an ENOSPC leaves
the PREVIOUS binary in place and prints a green status
([[stale-binary-prints-green-status]], [[enospc-makes-checkers-emit-false-policy-refusals]]).

## 6d. The forward's downcast was a promise, not a check (#775, 2026-08-14)

**Issue:** [#775](https://github.com/mudler/vllm.cpp/issues/775). **Class
residue:** [#847](https://github.com/mudler/vllm.cpp/issues/847). **Related, and
NOT the same fix:** [#730](https://github.com/mudler/vllm.cpp/issues/730) / PR
#784.

`ForwardNemotronHForCausalLM` opened its handle with an unconditional
`static_cast<NemotronHLoadedModel&>(model)` — the seam shape every other arch's
forward shares. A `static_cast` down a hierarchy is a promise the compiler is
entitled to act on, so on an object whose dynamic type is not that model, every
`nh.` member call is type confusion. UBSan's vptr check named
`nemotron_h_registry.cpp:112:30` (`nh.params()`, the member call THROUGH the
reference; the cast itself is `:102`) and `-fno-sanitize-recover=all` aborted
`test_nemotron_h_scaffold` on the `sanitize-cpu (address,undefined)` lane.

**Why the earlier repair did not close it.** PR #784 fixed #730 by removing the
`StubModel` the refusal subcase was handing in, so the subcase now forwards
through the real `NemotronHLoadedModel` that `factory->load_weights` returns.
That is a genuine improvement and it stands — but it cleared the *symptom* on
the lane while leaving the cast unchecked. The UB happened on the way to a
refusal that throws anyway, so nothing outside a sanitizer could ever see it,
and it would have stayed invisible after the weight loader lands. Baseline
measured at `0e8b15d56`: `test_nemotron_h_scaffold` is GREEN under
address+undefined, 12 cases / 38286 assertions — i.e. the reported diagnostic is
no longer reproducible verbatim, which is exactly why the cast needed fixing on
its own evidence rather than on the old run's.

**What was ported vs. written.** Nothing here mirrors an upstream path: vLLM's
registry hands a Python object to a Python `forward`, so there is no C++ downcast
to mirror and no upstream anchor to cite. `ModelAs` is written from scratch
against the seam this project already has, and it mirrors the local established
idiom rather than inventing one — `dynamic_cast`-with-refusal is how
`kv_cache_spec_registry.cpp:162-193`, `single_type_kv_cache_manager.cpp:370-949`
and `attention/backend.cpp:155` already establish a spec's or metadata's dynamic
type before using it.

**The fix.**
`include/vllm/model_executor/models/model_registry.h` gains
`vllm::ModelAs<Model>(model, architecture)` (and a `const` overload): a
`dynamic_cast` and a branch, with the refusal authored ONCE, out of line, in
`RaiseModelTypeMismatch` (`src/vllm/model_executor/models/model_registry.cpp`).
The message names the entry point that refused and the architecture the passed
model's own registration claims. `nemotron_h_registry.cpp:102` calls it.

Cost: one `dynamic_cast` per forward *step*. `ModelRegistry::Forward`
(`model_registry.cpp:326`) is the seam's single call site and is entered once per
step, not per layer, against a step that is milliseconds of GEMMs.

**Rejected alternatives.**

1. *Compare `model.registration().architecture` against the expected name.* It
   needs no RTTI, but it answers the wrong question: it establishes what the
   registration claims, not what the object IS. The realistic defect — a caller
   that resolves a registration and hands `factory->forward` a model another path
   produced — carries the RIGHT architecture string on the WRONG object, so this
   check passes and the UB proceeds. The RED test in this row is precisely that
   shape.
2. *Hand the forward a correctly-typed reference — make `forward` a virtual on
   `LoadedModel`.* This removes the cast class entirely and is arguably the
   better design. It is also a change to the `ModelFactory` seam and 32 model
   TUs, against a deliberate type-erasure contract (`model_registry.h:300`,
   "forward remains type-erased over LoadedModel"). Out of scope for a red-lane
   repair; recorded in #847 as the option a sweep should weigh.
3. *Fix the test again.* Refused by the issue and by this spec: the by-name
   refusal on a missing weight loader is load-bearing, and the defect is in the
   product path.

**RED-first.** `tests/vllm/models/test_nemotron_h_scaffold.cpp` gains
`NemotronH: the forward entry point REFUSES a foreign LoadedModel by name`,
which hands `reg.factory->forward` a complete `ForeignLoadedModel` — a
well-formed `LoadedModel` that simply is not a `NemotronHLoadedModel`. Against
the unfixed cast it reproduced the issue's diagnostic exactly:

```
src/vllm/model_executor/models/nemotron_h_registry.cpp:112:30: runtime error:
member call on address 0x7fffcab04070 which does not point to an object of type
'NemotronHLoadedModel'
0x7fffcab04070: note: object is of type '*N12_GLOBAL__N_118ForeignLoadedModelE'
    #0 ForwardNemotronHForCausalLM nemotron_h_registry.cpp:112
```

process exit 1, no doctest summary reached. After the fix: 13 cases / 38289
assertions, `Status: SUCCESS`.

This test is NOT a revival of the stub #784 removed, and must not be "repaired"
back into a real NemotronH model. That stub existed because the old subcase had
no other way to reach the refusal, and it was wrong because the forward
DEREFERENCED it. This one asserts the opposite guarantee — that the entry point
establishes the dynamic type BEFORE any member call — and after the fix it
performs no UB at all.

**Mutation proof.** Applied alone, rebuilt, run, restored, checksums verified.

| # | Mutation | Result |
|---|---|---|
| M1 | `ModelAs<NemotronHLoadedModel>(...)` → `static_cast<NemotronHLoadedModel&>(model)` | RED: same UBSan vptr report at `nemotron_h_registry.cpp:118:30`, process aborted before any doctest summary |
| M2 | `RaiseModelTypeMismatch` message replaced by `"model type mismatch"` | RED: 2 failed assertions, 12/13 cases, `Status: FAILURE!` — both by-name CHECKs fired |

M2 is the one that matters for the *guarantee* rather than the crash: it proves
the test is asserting a NAMED refusal, not merely the absence of an abort.

**The class is 34 more sites, and it is NOT swept here.** `grep -rn
"static_cast<[A-Za-z0-9_:]*LoadedModel&>" src/` returns 34 remaining
`prepare`/`forward` entry points across 32 model TUs, all the same shape; there
is no `const`-reference or pointer spelling, so that grep is the whole class.
They are not swept in this change because the sweep is not mechanical:
`llama_registry.cpp`, `qwen3_5_dense.cpp` and `gemma4_registry.cpp` each register
THREE architectures against ONE forward and `mistral_registry.cpp` registers two,
so those sites have no single architecture name to refuse under. That is a design
question with three defensible answers, enumerated in #847, and answering it
inside a red-lane repair would have made the repair unreviewable. This follows
how the unaligned-read class was handled — named site in #674/PR #688, residue
filed as #772 — rather than widening silently.

## 7. Now

**State at this commit:** **W1 and W3 have LANDED on `main`; W2 is in
re-review.** The `MIXED_PRECISION` resolver landed at `1bc5ef82c` (#561) and the
W3 scaffold at `c6b240edd` (#576); both are merged into this branch, and their
spec sections (§4's three W1 subsections, §5c/§5d/§5e) are main's, carried here
byte for byte. The Mamba2 SSD kernel work W1 landed earlier at `47960a009`
(#496), so `include/vt/ops.h` carries main's
`kMamba2ChunkScan`/`kMamba2StateUpdate`/`kRmsNormGatedGroup` first and appends
`kMoeRelu2` after them; no existing op id shifted. W2 (the non-gated `relu²`
expert, §6a) was reviewed PASS at `e2d68404`, repaired for that review's six
findings at `dd7a6477d`, and this branch is its land-prep: re-merged onto
`origin/main` and fully re-gated. A second fresh review (PR #586 @ `f6a7f8709`)
returned **PASS** — it established that the repair delta changes zero executable
lines — with four RECORD findings, all repaired on
`row/MODEL-NEMOTRON-H-W2-RECORDS`: two stale `activation.py:33` anchors
(`include/vt/ops.h`, this spec), M7's assertion count (§6a), §4's W2 seam text
contradicting §6a, and an overstated bit-identity comment in `cuda_moe.cu`
(#591). That repair touches comments and this spec only; no executable line
moved. `tests/vt/test_ops_moe_nongated_relu2.cpp:12` already carried the
corrected anchors and needed no change.

The row stays `INVENTORIED`; this commit changes no lifecycle state, so it owes
no `STATUS`/`BENCHMARKS` write. **Oracle gateability is CLOSED** — §5a records
the pinned oracle loading and running the checkpoint on GB10 with three greedy
goldens committed, so W6 has a denominator whenever it is reached.

**W4 has LANDED** (`ce8c8bf67`, #718), and the WEIGHT LOADER §7 named as "the
next brick" is built and gated on `row/MODEL-NEMOTRON-H-LOADER` — **§6d is the
authority on it**. The forward no longer refuses on a checkpoint load: 18487 of
18487 tensors are accounted (18217 materialized in their shipped formats, 270
deferred by name to W5), the real 20.1 GiB checkpoint runs at **17.70 GiB peak
RSS**, and its first greedy token matches the pinned oracle's committed golden on
**3 of 3** prompts.

**Next action:** the loader needs a **FRESH REVIEW** — never the agent that wrote
it. The two claims it changes rather than adds, and which a review should mutate,
are (a) `NemotronHOwned::View`'s refusal of a non-dense weight, which is the only
thing standing between a packed NVFP4 buffer and a plausible-garbage GEMM operand,
and (b) the expert-major reorder in `NemotronHMoeMixer`, whose result-neutrality
is claimed from the disjointness of the output slots rather than measured.

Then W5 (the MTP head, whose 270 tensors the loader already names as owed), W6
(the e2e token gate against the committed goldens, now unblocked — it has weights),
and W7 (GGUF). Carried forward, not resolved: the two OWED GPU items in §6a
(`kMoeGroupedGemmNvfp4Marlin` on the real g16 tensors, and the end-to-end
NemotronH MoE block on GB10) — the loader now produces exactly those g16 tensors,
so the first of them is reachable — and the OWED GGUF k-quant arm (§5b).

**Reported, outside this task's authority to fix.** `test_op_parity` is RED on
this row's base for a reason belonging to MODEL-MUSIC-MUSIC3 (#672); §6d's gate
evidence has the diagnosis.

The row stays `INVENTORIED`: the loader changes no lifecycle state, because the
forward is still the HOST reference and nothing runs on the paged runner (W6).

## 8. Stop conditions

- The pinned oracle cannot be made to load and run this checkpoint on GB10 →
  stop and report; without a running oracle there is no gateable denominator
  and the row does not proceed on source inspection alone.
- A `quantized_layers` entry names an algorithm we do not implement → refuse by
  name and record it as owed. Never silently dequantize to a supported path;
  that is invisible to a token gate.
- The non-gated expert cannot be expressed on the shared merged-GEMM seam →
  `NEEDS_DECISION`, do not fork a parallel MoE path.

## Owed

- [#847](https://github.com/mudler/vllm.cpp/issues/847) — **SWEPT, no longer
  owed here.** Row `FIX-REGISTRY-DOWNCAST-SWEEP` claimed the registry
  type-confusion class this row's §6d fix named but did not sweep, and closed it:
  all 34 remaining `prepare`/`forward` entry points now open their handle through
  `ModelAs`, and the class residue is zero. Spec
  [`registry-downcast-sweep.md`](registry-downcast-sweep.md), which also answers
  the blocking question (a shared forward refuses under the FAMILY PRIMARY — the
  architecture whose `load_weights` produces the type it opens) and corrects this
  entry's arithmetic: 30 model TUs, not 32, and the affected registries carry two
  architectures each, not three.

  The entry stays rather than being deleted because #847's row in the
  append-only `.agents/issue-index.md` names no owning row, so a spec must keep
  claiming it; GitHub holds the closed state. Do not read it as open work.

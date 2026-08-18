# DSpark draft routing for `architectures=["DSparkDraftModel"]` (`SPEC-DSPARK-QWEN3-ROUTING`)

| Field | Value |
|---|---|
| Row | `SPEC-DSPARK-QWEN3-ROUTING` (engine-matrix, Speculative decoding) |
| Issue | [#1193](https://github.com/mudler/vllm.cpp/issues/1193) |
| Scope | Give a DSpark draft whose config declares `architectures=["DSparkDraftModel"]` with `model_type == "qwen3"` an explicit route to the landed Qwen3 DSpark lane, and give the routing predicate a production caller. Three parts: (A) extend `SpeculativeConfig::IsDsparkDraft` with the `DSparkDraftModel` + `model_type == "qwen3"` pair; (B) port the architecture normalization upstream performs before the draft loads, including the DeepSeek-V4 fallback as a NAMED refusal rather than a silent rewrite; (C) reach both from `ResolveSpecConfig` so the loader classifies the draft instead of trusting the CLI method string alone. **Excluded:** the DeepSeek-V4 DSpark draft runtime (hardware-blocked, needs two Sparks), the DSpark speculator itself (landed under `SPEC-DSPARK`), the `--speculative-config` unknown-key drop ([#1160](https://github.com/mudler/vllm.cpp/issues/1160)), and the W7/W8 residuals ([#436](https://github.com/mudler/vllm.cpp/issues/436), [#442](https://github.com/mudler/vllm.cpp/issues/442)). |
| Upstream chain | At the pin `555967922`: `vllm/config/speculative.py:882-887` (the `method = "dspark"` auto-detection) → `:934-944` (the forced `model_type = "deepseek_v4"` + `architectures = ["DSparkDraftModel"]` rewrite) → `:945-961` (the Gemma4 key normalization) → `:973-994` (`n_predict` and k) → `:1003-1027` (the `k >= dspark_block_size` hard error). Beyond the pin: PR [vllm#52197](https://github.com/vllm-project/vllm/pull/52197), merged 2026-08-17T16:48:30Z at `7075ddac28c25d4fd2b84bc2a9a6c5ffde0345c8`, one file, +14/-1. |
| Our baseline | `include/vllm/config/speculative.h:114-136` (`IsDsparkDraft`) and `:138-200` (`ResolveDspark`), landed under `SPEC-DSPARK` W1. The loader path is `src/vllm/entrypoints/model_loader.cpp:875-885` (`ResolveSpecConfig` dspark branch), `:418` (`MakeDsparkDraftConfig`), `:456-500` (`LoadDsparkDraft`), `:1670-1678` (the second `ResolveDspark` call site). Tests: `tests/vllm/config/test_speculative_dspark.cpp`. |
| Port map | §3. |
| Tests to port | §5. |
| Gates | §6. |
| Dependencies | Landed: `SPEC-DSPARK` W1-W8 ([dspark-spec-decode.md](dspark-spec-decode.md), `ACTIVE`). External, PENDING developer authority: the 2.53 GiB draft download and GPU time for the run gate. No hardware blocker for the 27B lane. |
| Work breakdown | §4. |
| Risks/decisions | §7. |
| Pin policy | Mirror ahead of the pin, marked `BEYOND-PIN`. §2 gives the reasoning and the precedents. |
| Role / claim | helper, branch `row/SPEC-DSPARK-QWEN3-ROUTING` |
| Base | `fd64c76ee45ba49b070ea83024f6678ddd7f64a6` (origin/main, 2026-08-18) |
| Parity pin | vLLM `555967922` (0.26.0.dev0) at `$VLLM_SOURCE` |

## 0. Verdict

The brief this row started from said we misroute the checkpoint into the
DeepSeek-V4 lane. We do not. We never ported the branch that would misroute it.
That is a different defect and it is worth stating first, because it changes
what the fix has to do.

Three findings, each verified in a clean worktree at the base above.

**1. The forced route is absent.** `grep -rn DSparkDraftModel src include`
returns exactly one hit, a comment in
`src/vllm/model_executor/models/deepseek_v4_registry.cpp:9-10`. No code reads a
draft config's `architectures` key. The three reads in
`src/vllm/entrypoints/model_loader.cpp` at `:1247`, `:1249` and `:1692` are the
TARGET model's architectures, consumed by the live architecture dispatch.
`LoadDsparkDraft` (`:456`) passes the draft JSON to `MakeDsparkDraftConfig`
(`:418`), which reads dimensions and copies the object into `cfg.raw`, then
calls `LoadQwen3DSpark` (`:494`) unconditionally.

So a `DSparkDraftModel` draft loads as a Qwen3 DSpark draft today. That is the
post-#52197 answer, reached by never having ported the pinned behavior rather
than by decision. The divergence from the pin is real, silent, and unrecorded.

**2. `IsDsparkDraft` is unreached.** Its architecture list at
`include/vllm/config/speculative.h:131` names `Qwen3DSparkModel` and
`Gemma4DSparkModel`, mirroring `speculative.py:884-885`. Every reference to the
function outside its own header is in
`tests/vllm/config/test_speculative_dspark.cpp:132-140`. `ResolveSpecConfig`
branches on `cli.method` alone (`model_loader.cpp:844`, `:862`, `:875`, `:886`),
so nothing shipped classifies a draft by its architecture. The predicate is a
class that works, not a capability anything reaches. `AGENTS.md` "Nothing lands
dead" names this shape exactly, and adding the `DSparkDraftModel` case without
a caller would add a second dead branch to a dead function.

**3. The arch string now has two destinations and the record holds one.**
`.agents/model-matrix.md:549` maps `DSparkDraftModel` to
`vllm/models/deepseek_v4/__init__.py::DSparkDeepseekV4ForCausalLM`, state
`INVENTORIED`. `deepseek_v4_registry.cpp:9-10` repeats it. After #52197 the same
string names a Qwen3 draft whenever `model_type == "qwen3"`. The record is not
wrong about the DeepSeek-V4 destination. It is incomplete about the string.

Note for anyone reading the brief that opened this row: it also said no
`DSparkDraftModel` row exists in the engine matrix. That is true of the engine
matrix and beside the point, because the registry comment's referent is a MODEL
matrix row and it exists at `.agents/model-matrix.md:549`. An architecture is a
model-matrix key. Adding a second row under an engine key would duplicate a
keyed record across two matrices, which `AGENTS.md` "Records" forbids. §8 lists
the model-matrix amendment this row owes instead.

## 1. What #52197 changed

The PR is one file and two hunks. Both are quoted from the merge commit
`7075ddac28c25d4fd2b84bc2a9a6c5ffde0345c8`. Its line numbers are ahead of our
pin, so each hunk is given with the pin's own anchor beside it.

**Hunk 1, detection.** Upstream `:950` post-merge, our pin `:882-887`. The
`method = "dspark"` condition gains a fourth arm:

```python
or (
    "DSparkDraftModel" in self.draft_model_config.architectures
    and self.draft_model_config.hf_config.model_type == "qwen3"
)
```

**Hunk 2, routing.** Upstream `:1011` post-merge, our pin `:934-944`. A new
branch runs BEFORE the DeepSeek-V4 fallback, and the fallback becomes its
`elif`:

```python
if (
    self.method == "dspark"
    and "DSparkDraftModel" in self.draft_model_config.architectures
    and self.draft_model_config.hf_config.model_type == "qwen3"
):
    self.draft_model_config.hf_config.architectures = ["Qwen3DSparkModel"]
    self.update_arch_()
elif self.method == "dspark" and (
    ...
```

The order matters and is the whole fix. The new branch is a guard in front of a
catch-all, so the specific case is claimed before the general one rewrites it.

The upstream head has drifted well past our pin around this code. The PR's own
`elif` context carries a `K3DSparkModel` arm that does not exist at
`555967922`. §2 uses that drift.

## 2. The pin decision

`AGENTS.md` advances the pin only after every affected row and gate is
reconciled. #52197 merged after the pin, so this row must choose. The two
options are (a) mirror the post-pin change ahead of the pin and mark it
`BEYOND-PIN`, or (b) scope a pin-advance row and take the change with it.

**Decision: (a), mirror ahead of the pin, marked `BEYOND-PIN`.**

Four reasons.

**The precedent is established and it is the same shape.**
`.agents/model-matrix.md:318-319` carries `Qwen3_5ForCausalLM` and its MoE
sibling as **BEYOND-PIN — NOT in `555967922`**, because the text-only Qwen3.5
arms landed upstream after the pin in PR [vllm#50210](https://github.com/vllm-project/vllm/pull/50210)
at `ad5d29db7`. That row names the PR and its commit and mirrors from them. This
row is the same case with a smaller diff: a merged upstream PR, a named commit,
a behavior the pin gets wrong. `.agents/model-matrix.md:100` carries the same
mark for `ParakeetForCTC`, and `:211` for `BailingMoeV3ForCausalLM` as
pin-lag. The mark is a working convention here, not an exception being invented.

**A pin advance is out of proportion and it is not scoped.** The pin is a
whole-repository fact recorded in `.agents/upstream-sync.md` and repeated in
`.agents/engine-matrix.md:5-9`. Advancing it obliges reconciling 161 engine
rows, 377 model rows, and every gate bound to `555967922`. The drift is not
hypothetical: the `K3DSparkModel` arm visible in the PR's context is upstream
work absent from our pin, so an advance imports unrelated speculative-decoding
scope into a routing fix.

**Waiting costs correctness, not just currency.** The pinned behavior is wrong
for a checkpoint that exists and is published. Mirroring the pin faithfully here
would mean porting the DeepSeek-V4 rewrite and thereby BREAKING a load that
works today. That is the one outcome no reading of the policy wants.

**The change is separable and reversible.** It is one predicate and one branch
at one seam. When the pin advances, the row's `BEYOND-PIN` mark is removed and
the anchors are re-pointed at the new pin's line numbers. Nothing else moves.

Recorded under `## Now`. §8 lists what a later pin advance inherits from this
row.

## 3. Port map

| Upstream, at the pin unless marked | Lands in | Shape |
|---|---|---|
| `speculative.py:882-887` + #52197 hunk 1 (BEYOND-PIN) | `include/vllm/config/speculative.h`, `IsDsparkDraft` | Extend the signature with the draft's `model_type`, and add the `DSparkDraftModel` + `qwen3` pair beside the two existing architecture names at `:131`. |
| #52197 hunk 2, the new leading branch (BEYOND-PIN) | `include/vllm/config/speculative.h`, a new `ResolveDsparkArchitecture` | Return the normalized architecture for a DSpark draft. `Qwen3DSparkModel` for the native name, for the Gemma4 name, and for `DSparkDraftModel` + `qwen3`. |
| `speculative.py:934-944`, the DeepSeek-V4 fallback | the same function | NOT a silent rewrite. Return a refusal that names DeepSeek-V4 DSpark as unimplemented. §7 R2 gives the reason. |
| the classification call | `src/vllm/entrypoints/model_loader.cpp`, `ResolveSpecConfig` dspark branch at `:875-885` | Read the draft `config.json` before `ResolveDspark`, call `IsDsparkDraft` and `ResolveDsparkArchitecture` on it, and refuse by name when the draft resolves to a lane we do not implement. This is the production caller item 2 of §0 says the predicate lacks. |

The draft config is already read once inside `LoadDsparkDraft`
(`model_loader.cpp:471-474`). The implementer decides whether to read it twice
or to hoist the read, and records the choice under `## Outcome`. Correctness
does not depend on which, and a second small JSON read of a five-layer draft
config is not a measured cost.

## 4. Work breakdown

| Step | Content | Gateable on this host |
|---|---|---|
| W1 | The red test. A `DSparkDraftModel` + `qwen3` draft config that today gets no classification, and a DeepSeek-V4-shaped one that today gets no refusal. Capture both reds. | yes, CPU |
| W2 | `IsDsparkDraft` gains the pair; `ResolveDsparkArchitecture` lands with the three accept cases and the one refusal. Focused green. | yes, CPU |
| W3 | The `ResolveSpecConfig` call site, so the predicate is reached from the loader. The reachability mutation of §6 belongs here. | yes, CPU |
| W4 | The records: the model-matrix amendment of §8, `docs/FEATURES.md` if the surface statement changes, and `docs/USAGE.md` weights for the 27B draft. | yes |
| W5 | The run gate of §6 against the pinned oracle with #52197 applied, on `Qwen3.8-27B` + `RadixArk/Qwen3.8-27B-DSpark`. | needs a GPU lease |

W1-W4 need no GPU. W5 does. They are separable and W5 does not block W1-W4.

## 5. Tests to port

Upstream ships no test for #52197. Its "Test Plan" section is empty and its
"Test Result" is a pasted gsm8k serve log. So there is nothing to port
one-for-one, and this row writes the tests it needs. `.agents/porting.md`
requires the adaptation to be recorded, and this paragraph is that record.

The existing local harness is `tests/vllm/config/test_speculative_dspark.cpp`.
Its `IsDsparkDraft` cases at `:132-140` are the pattern to extend.

**Red-first, and each red must fail for its own reason.**

| Test | Today | After |
|---|---|---|
| A config carrying `architectures=["DSparkDraftModel"]` and `model_type="qwen3"`, with a model id that does NOT contain the substring `dspark` | `IsDsparkDraft` returns false, because `:127` misses and `:131` names neither | returns true |
| The same config through `ResolveDsparkArchitecture` | the function does not exist | returns `Qwen3DSparkModel` |
| A config carrying `architectures=["DSparkDraftModel"]` and `model_type="deepseek_v4"` | nothing refuses; `LoadDsparkDraft` would read it as a Qwen3 draft and fail later on a missing key | refuses by name, and the message says DeepSeek-V4 DSpark is unimplemented |
| The three pinned cases at `:132-140` | pass | pass, byte-identical |

The model id in the first case must avoid the substring `dspark`, or `:127`
answers true before the architecture list is consulted and the test passes
without the change. That is the mute-switch shape, and it is the one trap this
test set has.

**The reachability test.** A loader-level case that drives the classification
through `ResolveSpecConfig` on a draft directory, not through a direct call to
the predicate. `AGENTS.md` "Nothing lands dead" requires the smallest failing
test to enter the new code through a production entry point, and a unit test on
`IsDsparkDraft` is exactly the unit test that rule excludes.

## 6. Gates

| Gate | Content | State |
|---|---|---|
| G1, focused | The §5 cases green, the pinned `:132-140` cases unchanged. | owed |
| G2, mutation | Delete the `ResolveSpecConfig` call site in a scratch copy and rerun G1. A green G1 without the call site measures a class, not a capability, and fails this row. | owed |
| G3, mutation | Revert the `:131` architecture list to the two pinned names and rerun G1. The first §5 case must go red. | owed |
| G4, full | `scripts/agent-preflight.sh` and the repository gate on the branch. | owed |
| G5, run | Token-exact decode against the pinned oracle with #52197 applied, same target, same draft, same k, greedy. `Qwen/Qwen3.8-27B` + `RadixArk/Qwen3.8-27B-DSpark` at revision `85ef153be924f17ce4bf62726954eeaa4a73e854`. | owed, GPU-blocked, NOT hardware-blocked |
| G6, spec-off | Decode with speculation off byte-identical before and after. The change touches classification only, so a difference here is a defect in the change. | owed |

G5 is reachable on this hardware. §7 R1 records the checkpoint evidence.

The oracle for G5 is the pinned vLLM with #52197 applied as a patch, not vLLM
`main`. Patching one merged commit onto the pin keeps the comparison at the pin
for everything this row does not touch. That is the same discipline the
`BEYOND-PIN` model rows use.

## 7. Risks and decisions

**R1. Does a gateable checkpoint exist on this hardware?** Yes, and this is the
reality check the row was told not to duck.

The checkpoint the upstream PR names,
`RadixArk/Qwen3.8-2.4T-A95B-DSpark`, drafts for a 2.4T base. At bf16 that base
is roughly 4.8 TB against 128 GB of unified memory on this box. It is not
gateable here and never will be.

`RadixArk/Qwen3.8-27B-DSpark` exists, and it is the same config shape. Read
live on 2026-08-18 at revision `85ef153be924f17ce4bf62726954eeaa4a73e854`:

```json
"architectures": ["DSparkDraftModel"],
"model_type": "qwen3",
"block_size": 7,
"num_hidden_layers": 5,
"hidden_size": 5120,
"num_attention_heads": 40,
"num_key_value_heads": 8,
"head_dim": 128,
"num_target_layers": 64,
"dflash_config": { "target_layer_ids": [4, 16, 28, 40, 52], "markov_rank": 256, "projector_type": "dspark" }
```

One shard, `model.safetensors`, 2718576122 bytes, sha256
`9d26d5e637551c244d543c67c790bd0947f360e005c569e5851a185ffe692786`. Five layers
against a 64-layer target. Qwen3.8-27B is already a gate target here.

So this row is **gateable on this host**, not unreachable-but-correct. The
2.4T lane stays memory-infeasible and the row does not claim it.

The repository id alone is not a pin, so the revision and the sha256 above are
the pin, and `docs/USAGE.md` takes them in W4.

**R2. Should the DeepSeek-V4 fallback be a rewrite or a refusal?** Upstream
rewrites the config and lets the DeepSeek-V4 model path handle it. We do not
implement that path: `deepseek_v4_registry.cpp:19-25` records that
`DeepseekV4Model` is a stub that fails a `VT_CHECK`, and the DeepSeek-V4 lane
needs two Sparks. Mirroring the rewrite would send the draft toward a stub and
produce a failure that names an internal check instead of the missing arm.
`AGENTS.md` requires an unimplemented arm to refuse with a message that names
the missing part. **Decision: refuse by name.** Record it as one tracked
divergence from `speculative.py:940-944`, with the reason, in the same commit
that lands it.

**R3. Two draft-config reads.** §3 leaves the choice to the implementer and
requires the reason under `## Outcome`.

**R4. `block_size` is read by nobody.** Both `ResolveDspark` call sites
(`model_loader.cpp:881-883` and `:1675-1677`) pass `std::nullopt` for
`n_predict` and for `dspark_block_size`, so the `k >= dspark_block_size` hard
error at `speculative.py:1003-1027` cannot fire. The 27B draft carries
`block_size: 7`, which is the key upstream's Gemma4 normalization at `:945-961`
maps onto `n_predict`. A k below 7 would therefore be accepted here and would
produce garbled output, which `dspark-spec-decode.md` R4 already names as the
trap. This is a real gap, it is adjacent, and it is NOT this row's scope. §8
owes it an issue.

**R5. Mirroring ahead of the pin drifts if upstream changes again.** #52197 is
merged, not open, so the risk is a later upstream revision of the same lines,
not a rebase of this one. The `BEYOND-PIN` mark plus the merge commit
`7075ddac` make the exact source re-checkable, which is what the precedent rows
in §2 do.

## 8. Owed

- [#1193](https://github.com/mudler/vllm.cpp/issues/1193) — this row. Open,
  owned here.
- The model-matrix amendment for `.agents/model-matrix.md:549`: after #52197
  the `DSparkDraftModel` string names a Qwen3 draft when `model_type` is
  `qwen3`, and that row records only the DeepSeek-V4 destination. Owed by W4 of
  this row, in the same change that lands the routing.
- The `deepseek_v4_registry.cpp:9-10` comment says `DSparkDraftModel` is "a
  separate row (INVENTORIED)". The referent exists at
  `.agents/model-matrix.md:549`. The comment stays correct once the amendment
  above lands. No engine-matrix arch row is created, for the keyed-record reason
  in §0.
- R4's `block_size` gap needs its own issue and row. It is a correctness hole in
  the landed `SPEC-DSPARK` lane, not in this routing change, and folding it in
  here would bundle unrelated work into one branch.
- A pin advance past `555967922` inherits from this row: remove the
  `BEYOND-PIN` marks, re-point the `speculative.py` anchors at the new line
  numbers, and re-check whether the `K3DSparkModel` arm visible in #52197's
  context needs its own row.

## 9. Stop conditions

Stop and report `NEEDS_DECISION` when any of these holds.

- The developer rejects the §2 pin decision and directs a pin-advance row
  instead. The spec is then superseded, not edited into a different shape.
- The DeepSeek-V4 refusal of R2 is rejected in favour of mirroring the rewrite.
  That reverses a stated divergence and changes the gate.
- W5 needs a GPU lease and none is granted. W1-W4 still land; G5 stays `PENDING`
  against the named resource, and the row does not reach `DONE`.
- The 27B draft's revision moves under its name. Checkpoints get re-quantized in
  place, so a sha256 that stops matching means the gate's denominator changed,
  not that the code regressed.
- The row's change grows past classification into the DSpark speculator, the
  loader's weight path, or the `--speculative-config` parser.

Stop and report `NEEDS_CONTEXT` when the draft download authority or the GPU
authority is unrecorded. Do not assume either.

## Now

`READY`. The spec is committed, [#1193](https://github.com/mudler/vllm.cpp/issues/1193)
is open, and no implementation has landed.

The pin decision is made and recorded: **mirror #52197 ahead of the pin, marked
`BEYOND-PIN`**, on the precedent of `.agents/model-matrix.md:318-319`,
`:100` and `:211`. §2 carries the reasoning. A pin advance is not scoped by this
row and is not blocked by it.

The reality check is answered: `RadixArk/Qwen3.8-27B-DSpark` exists at revision
`85ef153be924f17ce4bf62726954eeaa4a73e854` with the exact config shape, so this
row is gateable on this host. The 2.4T lane named by the upstream PR stays
memory-infeasible here and is not claimed.

Next: dispatch a fresh implementer for W1-W3 against this spec. W1 captures the
two reds of §5 before any production edit.

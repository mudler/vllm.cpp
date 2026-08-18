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
| Role / claim | helper. The implementation branch is `row/DSPARK-QWEN3-ROUTING-IMPL`, which does NOT match the `row/<ID>` form this row's ID (`SPEC-DSPARK-QWEN3-ROUTING`) requires. Recorded rather than renamed: renaming mid-flight strands the review that is already anchored to the branch, and the deviation costs a reader one lookup, not a wrong answer. The next branch for this row uses `row/SPEC-DSPARK-QWEN3-ROUTING` |
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

**Each refusal is asserted by its OWN wording.** Both refusals on this path name
DeepSeek-V4 and say "not implemented", so a case that asserts only those two
substrings cannot tell the identity check from the architecture normalization —
and a test that cannot tell them apart cannot see either one deleted. The
identity case asserts "does not identify as"; the normalization case asserts
"routes to the DeepSeek-V4 DSpark lane". Both mutations in §6 G2 depend on this.

**Branch 3 needs a loader case too.** Added on the `b626be75a` merge rather
than at the original landing. `ResolveDsparkArchitecture`'s Gemma4 arm — the
COLLAPSE onto `Qwen3DSparkModel` that `## Outcome` records as the second
divergence — was exercised only by a hand-called case in
`tests/vllm/config/test_speculative_dspark.cpp`. That proves the function
answers; it does not prove the loader reaches the arm, which is exactly the
distinction `AGENTS.md` `## Nothing lands dead` draws. The loader case asserts
the Gemma4 name is ADMITTED, from a directory whose name carries no `dspark`, so
only the architecture arm can answer for it.

**Both published draft layouts, not one.** The Speculators layout declares no
top-level `architectures` and is translated to `["Qwen3DSparkModel"]` before the
classification reads it, so it is the one shape where the no-architecture
narrowing and the classification disagree and the ORDER decides. It gets its own
loader case.

## 6. Gates

| Gate | Content | State |
|---|---|---|
| G1, focused | The §5 cases green, the pinned `IsDsparkDraft` cases unchanged. | GREEN 2026-08-18 on the `b626be75a` merge: `test_speculative_dspark` 12 cases / 40 assertions, `test_dspark_draft_routing` 7 cases / 19 assertions, both `Status: SUCCESS!`, rc=0. The routing suite grew from 6/17 at the merge: the Gemma4 collapse (branch 3 of `ResolveDsparkArchitecture`) had only a hand-called unit case, which is the shape `AGENTS.md` `## Nothing lands dead` excludes, so it gained a loader-level case beside the other `DraftDir` fixtures |
| G2, mutation | Delete the `ResolveSpecConfig` call site in a scratch copy and rerun G1. A green G1 without the call site measures a class, not a capability, and fails this row. Three mutations, because the call site has three separable parts and a refusal that names only the LANE cannot tell them apart. | RED as required 2026-08-18, in a scratch copy at `$SCRATCH/mut`, each with `compile_rc=0`: (a) delete the whole `IsDsparkDraft` call → routing rc=1; (b) make `IsDsparkDraft` return constant `true` → routing rc=1 and the predicate suite rc=1; (c) delete the `ResolveDsparkArchitecture` call → routing rc=1. The scratch tree was restored byte-for-byte and re-verified green after each. A FOURTH mutation was added on the `b626be75a` merge, for the Gemma4 case above: narrowing branch 3's guard from `!has_qwen3 && !has_gemma4` to `!has_qwen3` reds the routing suite at rc=1, 7 cases run, 6 passed, 1 failed, and the one failure is the new case. Its FIRST writing failed to build — `-Werror` on the now-unused `has_gemma4` — and the stale binary reported 7/19 SUCCESS, so the mutation was rewritten with a `(void)has_gemma4;` to compile cleanly and only then measured (`compile_rc=0`). The header was restored from a byte copy and all three suites re-verified green |
| G3, mutation | Revert the architecture list in `IsDsparkDraft` to the two pinned names and rerun G1. The first §5 case must go red. | proven by G2(b), which is the stronger form: a predicate that admits EVERYTHING already fails the case that only the `DSparkDraftModel` + `qwen3` arm can pass |
| G4, full | `scripts/agent-preflight.sh` and the repository gate on the branch. | GREEN 2026-08-18 on the tree merged up to `origin/main` `aba8d5ffb`, with ONE environmental red: `test_cpu_x86_llamacpp_floor` refuses to measure while the box is loaded (`NO_QUIET_WINDOW`, exit 4 where the case expects 2, at load average 41-107 driven by other sessions). It is not this row's: the file's last change is `0305b909f` on `main`, this branch touches no benchmark harness, and the same suite reported `ok` on the same sources in the preceding preflight run at a quieter moment. Every other gate `ok`, including `check-agent-record`, `check-symbol-anchors`, `check-doc-checkpoint`, `check-public-doc-tables`, and both trailer gates |
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

**R4. `block_size` is read by nobody.** CLOSED by
[#1225](https://github.com/mudler/vllm.cpp/issues/1225)
(`SPEC-DSPARK-BLOCK-SIZE-GUARD`, landed `b626be75a`), which took the issue and
the row this risk asked for. The paragraph below records the state at the time
this spec was written; on the merged tree there is ONE `ResolveDspark` call
site and it is passed the draft's real `n_predict` and block floor. Both
`ResolveDspark` call sites
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
- ~~R4's `block_size` gap needs its own issue and row.~~ DISCHARGED: it became
  [#1225](https://github.com/mudler/vllm.cpp/issues/1225) and row
  `SPEC-DSPARK-BLOCK-SIZE-GUARD`, which landed on `main` as `b626be75a` and is
  merged into this branch. Keeping it out of this branch was the right call for
  the reason given — it is a correctness hole in the landed `SPEC-DSPARK` lane,
  not in this routing change — and the two rows met at the merge rather than in
  one branch.
- **The empty `architectures` list is NOT classified.** Upstream reads the key
  off a HuggingFace `ModelConfig`, where an absent key is `[]`, so its catch-all
  sends that list to DeepSeek-V4. The loader here skips classification instead,
  because refusing on the ABSENCE of evidence would refuse the native
  `deepseek-ai/dspark_qwen3_*_block7` drafts if they declare no architecture, and
  no copy of one has been read on this host to settle it. Owed: read a published
  native draft's `config.json`, and tighten the guard to upstream's catch-all if
  it declares the key. Gated today by
  `test_dspark_draft_routing.cpp`'s no-architecture case, which pins the
  narrowing so it cannot change silently.
- **The refusal LEADS, as of the merge of `SPEC-DSPARK-BLOCK-SIZE-GUARD`
  ([#1225](https://github.com/mudler/vllm.cpp/issues/1225), `b626be75a`).** The
  two entries this replaces were both true when they were written and are both
  false now, so they are re-derived here rather than deleted. `AGENTS.md`
  `## Nothing lands dead` judges reachability at the MERGE commit, which is why
  this could not be left to the next reader.

  #1225 hoists the DSpark resolution to the top of
  `src/vllm/entrypoints/model_loader.cpp::LoadedEngine::FromModelDir`: a
  `dspark` method now calls `LoadedEngine::ResolveSpecConfig(params,
  vllm::HfConfig{})` there and hands the result down to the draft load. That
  call is ABOVE the target-directory existence check and far above
  `maybe_load_dflash`, so `ResolveSpecConfig` — and therefore
  `ReadDsparkDraftIdentity`, `IsDsparkDraft` and `ResolveDsparkArchitecture` —
  runs BEFORE `LoadDsparkDraft` rather than after it. Re-derived from the merged
  control flow:

  - `FromModelDir` is the one production engine constructor, reached from
    `include/vllm.h` through `src/capi/vllm_c.cpp` and from
    `src/vllm/entrypoints/openai/server_main.cpp`. A user arriving through the C
    ABI or the server at a DeepSeek-V4 DSpark draft now gets the NAMED refusal.
    `LoadDsparkDraft`'s "the draft config must carry target_layer_ids and
    mask_token_id" no longer wins, because the load it comes from no longer
    runs first.
  - Two of the resolution's OWN messages still precede the classification, and
    both are correct where they are. #1225's "requires num_speculative_tokens"
    check sits ahead of the classification inside the same `cli.method ==
    "dspark"` branch, so a run that names no `k` against a draft carrying no
    `n_predict` is refused for the missing `k`. And a `.gguf` target takes the
    GGUF branch above the hoist, whose own named refusal ("needs a safetensors
    target at this pin") fires first.
  - The in-memory constructors
    `LoadedEngine(HfConfig, Qwen3_5DenseWeights, Tokenizer, EngineParams)` and
    its `Qwen3_5MoeWeights` sibling — the ones whose member-init runs
    `ResolveSpecConfig`, and the entry point the reachability suite uses — have
    exactly ONE caller across `src/`, `include/` and `examples/`, and the
    earlier writing of this entry said there was none. Re-run on the merge, a
    grep for `LoadedEngine` over those three trees hits 23 files; the three that
    CONSTRUCT one are `src/capi/vllm_c.cpp:776` and
    `src/vllm/entrypoints/openai/server_main.cpp:974,1155`, both through
    `LoadedEngine::FromModelDir`, and `examples/bench/bench_core.h:570`, which
    calls the in-memory `Qwen3_5MoeWeights` constructor directly. The
    CONCLUSION is unchanged and now rests on the right reason: an example's
    internals are not a production entry point under `AGENTS.md`
    `## Nothing lands dead`, and that call sets no `params.speculative_config`
    at all (`bench_core.h` assigns it only in its `else` arm at `:587-589`,
    which loads through `FromModelDir` at `:591`), so `ResolveSpecConfig`
    returns `std::nullopt` at its first line and never reaches the dspark
    branch.

  So the second classification call site inside `LoadDsparkDraft`, which the
  earlier entry owed, is NOT owed: the ordering it existed to fix is fixed, and
  adding it would put a second copy of the classification behind the one that
  already leads. What remains owed on this item is nothing. The row's
  remaining debt is §6's G5 run gate and G6 spec-off gate, both above.

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

## Outcome

Recorded at the W1-W4 landing, ahead of `DONE`, because §3 and §7 R3 name a
decision this row had to make and neither the code nor the Git history states
the reason on its own.

**R3, THREE draft-config reads, not one.** #1225 has now landed
(`b626be75a`), so the count this entry anticipated is the count that exists. The
dspark path reads the draft's `config.json` three times: in
`src/vllm/entrypoints/model_loader.cpp::ReadDsparkDraftIdentity` for the two
classification keys, in
`src/vllm/entrypoints/model_loader.cpp::ReadDsparkDraftKeys` for `n_predict` and
the block floor, and in
`src/vllm/entrypoints/model_loader.cpp::LoadDsparkDraft` for the weights. All
three bodies open the file, parse it, and run the same Speculators translation.
**Decision: unchanged — do not hoist here.**

The REASON first given for it has expired and is corrected. It was that the
reads "sit on opposite sides of the engine constructor", so a shared read had
nowhere to live that both callers reach. That was true while `LoadDsparkDraft`
ran inside `FromModelDir`'s `maybe_load_dflash` and `ResolveSpecConfig` ran in
the constructor's member-init. #1225's hoist moved the resolution to the top of
`FromModelDir`, so `ReadDsparkDraftIdentity` and `ReadDsparkDraftKeys` are now
ADJACENT — consecutive statements in the same branch of the same function — and
a shared read between those two would need no new member and no new parameter on
any public seam. The structural obstacle is gone.

What survives is the cost argument and the scope argument, and they still carry
the decision. The cost is a five-layer draft's `config.json` parsed three times
at load time, on a path that then reads a 2.53 GiB shard; it is not measurable,
and this row measured nothing that would justify changing it. The three readers
want different keys and answer to different rows, and each names the keys it
wants, so neither can silently change what the others see. And the third reader,
`LoadDsparkDraft`, is still on the far side of the whole target load, so a hoist
that unified only the two adjacent readers would leave the duplication in a
shape harder to reason about than three symmetric readers. ONE hoist serving all
three callers is worth its own row; it is not this row's, whose §9 lists growth
into the loader's weight path as a stop condition.

**The DeepSeek-V4 refusal, and what it reaches.** §7 R2's decision stands and is
argued in three places (the predicate's header, the loader, and this row). Its
REACH has been restated twice. The first writing overclaimed it; a review
corrected it to "leads on no production path today", which was right while
`FromModelDir` ran `LoadDsparkDraft` before `ResolveSpecConfig`; and #1225's
hoist (`b626be75a`) made that correction false in turn. The refusal now LEADS
from `include/vllm.h` and from the server, which is this row's own goal being
met by a change that landed beside it. The re-derivation, the two resolution
messages that still precede it, and the second call site that is consequently NO
LONGER OWED are under `## Owed`.

**TWO divergences from upstream, not one.** The first writing named only the
DeepSeek-V4 refusal. The second is at branch 3: upstream leaves a Gemma4 draft's
`Gemma4DSparkModel` architecture in place and normalizes only its keys, because
upstream has a Gemma4 DSpark class to dispatch to. This engine has one DSpark
draft lane, so branch 3 COLLAPSES onto `Qwen3DSparkModel`, and that collapse is
what makes `ResolveDsparkArchitecture` total over a single lane.

**Why the loader dispatches on nothing.** The first shape of the call site
guarded on `lane != "Qwen3DSparkModel"` and threw. Because of the collapse
above, no input can enter that branch: the function answers `Qwen3DSparkModel`
or throws. A reviewer's mutation deleted the branch and both suites stayed
green, which is the definition of dead code under `AGENTS.md` `## Nothing lands
dead`. It was deleted rather than disclosed, because the staged-slice exception
is for work a named row will wire, and no row will wire this one: the branch
becomes live only when a SECOND lane exists, and the change that adds that lane
is the change that should add its dispatch. `ResolveDsparkArchitecture` is still
called, for its refusal — a mutation deleting the call turns the routing suite
red.

**Why an empty `architectures` list is not classified.** Under `## Owed`.
Upstream can send an absent key to its DeepSeek-V4 path; refusing here would
refuse a checkpoint whose contents nobody on this host has read, and the native
`deepseek-ai/dspark_qwen3_*_block7` drafts load today.

## Now

`ACTIVE`. W1-W4 have landed on the row's branch, been reviewed once and repaired
once. [#1193](https://github.com/mudler/vllm.cpp/issues/1193) is open and stays
open until G5 runs.

Landed: `IsDsparkDraft` carries the `DSparkDraftModel` + `qwen3` pair;
`ResolveDsparkArchitecture` normalizes the three accepted shapes and refuses the
DeepSeek-V4 lane by name; and the dspark branch of `ResolveSpecConfig` reaches
both from the draft's own `config.json`. G1 green (`test_speculative_dspark`
12 cases / 40 assertions, `test_dspark_draft_routing` 7 cases / 19 assertions),
G2 and G3 proven by mutation, G4 by the branch gate.

The pin decision stands as recorded in §2: **mirror #52197 ahead of the pin,
marked `BEYOND-PIN`**, on the precedent of the three existing `BEYOND-PIN` rows
in `.agents/model-matrix.md`. A pin advance is not scoped by this row and is not
blocked by it.

Next, in order: G5, the token-exact run gate against
`Qwen/Qwen3.8-27B` + `RadixArk/Qwen3.8-27B-DSpark` @
`85ef153be924f17ce4bf62726954eeaa4a73e854`, which needs the 2.53 GiB draft
download and a GPU lease and has neither; then G6, the spec-off byte-identity
check. Both are `PENDING` against a named external authority, not failing, and
the row does not reach `DONE` until they run. They are the row's only remaining
debt.

Merged `origin/main` `b626be75a` on 2026-08-18. `SPEC-DSPARK-BLOCK-SIZE-GUARD`
([#1225](https://github.com/mudler/vllm.cpp/issues/1225)) landed there and
hoisted the DSpark resolution to the top of `LoadedEngine::FromModelDir`, which
makes this row's classification and its named DeepSeek-V4 refusal RUN FIRST on
the production path instead of behind `LoadDsparkDraft`. Two statements this
spec carried became false at that merge and are re-derived under `## Owed`, and
`## Outcome`'s R3 reason expired with them and is corrected there. The focused
suites on the merge: `test_speculative_dspark` 12 cases / 40 assertions,
`test_dspark_draft_routing` 7 cases / 19 assertions, and #1225's own
`test_dspark_block_size_guard` 14 cases / 39 assertions, all `Status: SUCCESS!`,
rc=0.

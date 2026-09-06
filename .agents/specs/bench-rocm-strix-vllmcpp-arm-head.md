# Spec — the vllm.cpp arm of the gfx1151 survey, on a head that is asserted

Row `BACKEND-GATE-ROCM-LLAMACPP`. Issue
[#2944](https://github.com/mudler/vllm.cpp/issues/2944).

Sibling records:
[#2933](https://github.com/mudler/vllm.cpp/issues/2933) (the provenance defect
this measurement answers),
[#2921](https://github.com/mudler/vllm.cpp/issues/2921) (the survey that was
directed), [#2497](https://github.com/mudler/vllm.cpp/issues/2497) (the campaign
that already carries one retraction for quoting a ratio as a gated result),
[#2511](https://github.com/mudler/vllm.cpp/issues/2511) (the allocator defect
whose fix the surveyed binary predates), and PR
[#2940](https://github.com/mudler/vllm.cpp/issues/2940) (the published survey).

Predecessor evidence:
`docs/benchmarks/qwen38-27b-q4km-gfx1151.md`, which is NOT yet on `main` --
it is added by PR [#2940](https://github.com/mudler/vllm.cpp/pull/2940) and is
deliberately cited here by name rather than by link, because a link to a path
this base does not carry is a dangling link and the record gate reds on it; and
[`docs/bench-evidence/rocm-strix-llamacpp-denominator-20260902.md`](../../docs/bench-evidence/rocm-strix-llamacpp-denominator-20260902.md).

## Now

`ACTIVE`. The design below is committed before the lease runs, so the run cannot
choose its own N or its own comparison after seeing the numbers.

## 1. The question

The published survey reports the vllm.cpp arm as `FAULTED, 0 of 4, rc 139`.
#2933 established that this measured `11fed3ba5`, an ancestor of `6b97a6800`,
the commit that fixed exactly that hang. `survey.sh` asserted the llama.cpp
source manifest and refused to run when it moved; for our own arm it checked
`[ -x "$VC" ]` and **printed** the revision. It never built, never fetched and
never asserted, so it inherited #2511's staging directory at
`/tmp/rocm-strix-q4k` and measured that campaign's pre-fix build.

**So the published fault rate is #2511 reproducing.** What is unmeasured is
whether the current head completes this workload on `gfx1151`. #2511's own
"0 faults in 21 legs" was a different workload against a branch build in a
different lease, so it does not answer this either.

One question, then: **on a binary built inside the lease from the declared
revision, with that revision asserted rather than printed, what is the fault
rate, and if any leg completes, what is the throughput?**

The declared revision is `c796fea41`, which was `main`'s head when this row
claimed its lease. `main` moves faster than a lease can be scheduled, so the
figure names that revision and never "main", which is not a reproducible
identifier. §3 records why chasing the newer head would measure the same bytes.

## 2. Scope

In scope: one lease on `strix:gpu0`; a build from the declared head inside that
lease; the survey's own workload; the fault rate; and, if legs complete, a
throughput figure with per-leg spread and clock windows.

Out of scope: re-measuring llama.cpp or vLLM, whose figures stand. Repairing
`survey.sh`, which is #2933's own acceptance. Any parity claim.

## 3. The pin, and why it is asserted three ways

| declared | value |
|---|---|
| revision | `c796fea41f74fe90b8cf78190eeb2a5b4c977449` |
| tree | `018178f3bf1fca3983d9bc0cfd42f5ea4bf130b1` |
| bundle sha256 | `044dec44ebd8f153f25e13bfbeffd45ca7ba33d6342bdc99fdc63e348b6ca2b8` |

The job asserts all three and **exits non-zero** on any disagreement. A revision
alone is a label; the tree hash is the content behind it, and git computes it
the same way on both sides. The worktree must also be unmodified, because a
clean label over a dirty tree is the same lie in a different place.

**The built bytes are asserted separately from the source**, because #2933's
failure was not a wrong source tree — it was a right-looking source tree beside
an inherited build:

- The build directory is deleted before configure. A directory that does not
  exist cannot be stale.
- Both binaries must post-date the moment the directory was emptied.
- `libvllm.so` must carry the literal `cannot take a recoverable page fault`,
  which only #2511's narrowing introduces.
- The `vllm-cli` sha256 must **not** be
  `a703b83dd8954ba6dd3cbe82efcd38083c1d55492bbbaecf5c406f7c6efd646f`, the exact
  pre-fix binary #2933 names.

**`main` moved while the lease queued, and the pin deliberately did not.** By
the time the board was free, `main` had advanced past `c796fea41` and this branch
has since merged it. That does not change what is measured, and the claim is
checked rather than asserted:

```sh
git diff --name-only c796fea41..HEAD \
  -- src/vt/rocm include/vt/rocm src/vt/op_provider.cpp examples/cli   # EMPTY
git diff --stat c796fea41..HEAD -- src/ include/ CMakeLists.txt examples/cli
#   CMakeLists.txt | 5 +++++
```

That single hunk is `if(TARGET test_bench_eos_chat_template AND TARGET
vllm-bench) add_dependencies(...)`. This build configures
`-DVLLM_CPP_BUILD_TESTS=OFF`, so the target does not exist and the block never
fires; it orders two test targets and touches neither `vllm-cli` nor
`libvllm.so`. The two revisions therefore produce the same measured bytes, and
re-pinning would have cost an 88 MB bundle rebuild and the queue slot to measure
them again. **This was re-checked after the merge**, because a merge can falsify
prose written before it.

`ccache` may serve objects on the podman route. A ccache hit is keyed on
preprocessed source, compiler and flags, so it is byte-identical to compiling
that source; it does not weaken the assertion, and it is recorded rather than
hidden.

## 4. The design, declared before the run

| | |
|---|---|
| rounds | 4 |
| token counts | 64 (the survey's own) and 128 (for the slope) |
| repeat | 4 completions per leg after one load |
| cold runs | 1, discarded, matching the survey |
| legs by design | 8 |
| prompt | `The capital of France is`, greedy, `--temperature 0` |
| other flags | `--max-num-seqs 1`, `VT_OP_PROVIDER_STATS=1` |
| clock | sampled at 4 Hz beside every leg, to worker-local disk |

N is passed to the fold. **The fold counts nothing.** A previous comment on
#2511 reported `0/10 vs 8/10` because it counted log lines rather than legs.

The order of the two token counts rotates by round, so drift along the session
cannot be read as a difference between them.

## 5. Which quantity, and the ratio rule

`vllm-cli` reports `tok_s` as `completion_tokens` over the wall time of the
whole `vllm_complete()` call. That is **whole completion**, prompt included —
the same quantity as the survey's vLLM `generate()` leg, and **not** the
quantity `llama-bench -p 0` reports, which is pure decode.

PR #2940 records what happens when those are divided anyway: a meaningless
`1.814x`. So:

- `vLLM 6.734` against our whole-completion figure is **like for like**.
- `llama.cpp 12.219` may only sit beside a **derived** decode figure, obtained
  from the 64/128 slope exactly as the survey derived vLLM's `11.056`, and
  labelled derived wherever it appears.
- Both absolutes accompany any ratio.

The reference figures are constants carried from published evidence. No leg of
this run re-measures them.

## 6. Risks

- **The build may fail on the head.** Then no leg is an engine verdict, and the
  job says so and exits. A build failure is not a fault rate.
- **The legs may still fault.** That is a result, not a retry condition. The
  designed rounds still run so the fault rate has a denominator, under a
  deadline; nothing is re-run hoping for green.
- **A leg that refuses is not a leg that faults.** The reference tier is
  withdrawn on this board since `6b97a6800`, so `no kernel for op` is the
  specific failure this configuration could introduce. The classifier orders
  `OPREFUSED` before `MEMFAULT` before `GPUHANG` so a refusal is never reported
  as a board fault.

## 7. Gates

- The revision, tree, cleanliness and built-bytes assertions all pass, and their
  own output is in the evidence.
- The artifact sha256 is recomputed on the worker before any timing.
- `HSA_OVERRIDE_GFX_VERSION` reads `UNSET` and no `HSA_`/`ROCR_`/`PYTORCH_`/
  `HIP_`/`GGML_`/`VT_` variable is inherited.
- The fault rate is reported over the legs the design declares.
- The correctness caveat appears on the face of any table this produces:
  `TOKEN_GATE=FAIL`, every divergence a near-tie at about 0.125 nats, no
  deterministic denominator on this path, and those divergence counts named as
  constants rather than as fresh measurements.

## 8. Stop conditions

- The build fails: report the failure, publish no leg.
- The revision, tree or built-bytes assertion fails: the job exits non-zero and
  nothing is published. This is the assertion working, not a blocker.
- The lease is lost: report what completed and what did not.

# Spec — `BUILD-MAIN-TIP-TESTFIX`: main tip builds again — the DSA schedule test follows W9's signature, and glm5_next_weights stops shadowing under MSVC

Issue: [#2372](https://github.com/mudler/vllm.cpp/issues/2372). Owner row:
`BUILD-MAIN-TIP-TESTFIX` (user-directed take-over of the two facets the issue
named: MODEL-TEXT-GLM-MOE-DSA's test and MODEL-MM-GLM53-FLASH's warning).
Branch `row/BUILD-MAIN-TIP-TESTFIX`, base `1e65b8364` (origin/main, pinned at
worktree creation).

## Scope

Two build breaks on main's tip, neither introduced by any open PR, each
blocking every PR's CI at the merge ref:

1. `tests/vllm/models/test_glm_moe_dsa_schedule.cpp:304` — the call to
   `mla::ForwardMlaAttentionBlock` passes 11 arguments. W9 (`11f34effb`,
   MODEL-TEXT-GLM-MOE-DSA) inserted `Tensor* attn_pre_o_proj` before
   `MlaSharedSelection* shared`
   (`src/vllm/model_executor/layers/attention/mla_attention.cpp:423-428`), so
   the test's `MlaSharedSelection*` lands in the `Tensor*` slot:
   `error: cannot convert 'vllm::mla::MlaSharedSelection*' to 'vt::Tensor*'`.
   Breaks `build-test-cpu`, `build-newest-gcc`, and both `sanitize-cpu` jobs.
2. `src/vllm/model_executor/models/glm5_next_weights.cpp:223,226` — MSVC
   `warning C4456: declaration of 'v' hides previous local declaration`
   (twice), escalated by `error C2220` under the Windows -Werror regime. An
   if-condition declaration is visible in the else branch, so the second and
   third `const GgufValue* v` of `KdaHeadCount`'s else-if chain shadow the
   first. gcc/clang do not warn by default, which is why Linux CI stayed
   green. Breaks both `windows-msvc-*` jobs.

## Design

1. Test: mirror the production call. `glm_moe_dsa_forward.cpp:455-460` passes
   `/*attn_pre_o_proj=*/nullptr, shared` — "`attn_pre_o_proj` is DeepSeek-V4's
   early return (#2323); this model applies `o_proj` in the block." The test
   drives that same schedule, so its call gains the same explicit `nullptr`
   slot. The selection stays where W9 put it: the parameter it names.
2. Loader: rename the shadowing declarations only — branch-local `v` at line
   223 becomes `v_group` (use at 225), line 226 becomes `v_inner` (use at
   228). The first branch's `v` and everything else in the function stay
   byte-identical. No logic, ordering, or refusal behavior changes; both
   `KvInt` calls keep their exact arguments.

## Upstream anchors

vLLM `deepseek_v2.py:1372-1377` (`topk_indices_buffer`), cited by the
production comment the test call now mirrors (`glm_moe_dsa_forward.cpp:450-454`).
The test fix carries no behavior change; its anchor is the production call it
copies. The shadow rename has no upstream anchor — it is a platform-compiler
constraint of the Windows -Werror regime, not a mirrored behavior.

## Tests

- Red-first (gcc facet): configure and build the `test_glm_moe_dsa_schedule`
  target on this worktree BEFORE the fix and capture the type error in the
  build log.
- Red-first (MSVC facet): impossible on this host — no MSVC. The red evidence
  is the CI log at PR #2370's head (`windows-msvc-cpu`, C4456 at 223,31 and
  226,31). Unavoidable adaptation, named here; the arm's green proof is CI on
  this PR's head.
- Green: the same target builds and the test binary passes.
- Mutation: (a) revert the test call to the 11-arg form — the focused build
  reds again; (b) reintroduce a shadow and compile the TU with `-Wshadow` —
  gcc flags it, proving a local detection mechanism exists for what only MSVC
  flags by default.
- Full gate: `scripts/agent-preflight.sh` on the final head, chained to the
  exact-SHA push.

## Gates

- Focused: `cmake --build build --target test_glm_moe_dsa_schedule`, then run
  the test binary.
- Full: preflight rc=0 on the final head.
- CI (external authority): the six #2372 job names are expected green on this
  PR's head — that is the MSVC arm's green evidence. The `agent-record` red
  (#2371, fixed by #2373) and `documentation-checkpoint` (#2375, main-only)
  are unrelated and out of scope.

## Stop conditions

- If the `nullptr` slot turns out wrong for this schedule (the test's shared
  semantics change behavior, not just compilation), STOP and return
  `NEEDS_DECISION`: the schedule's intent belongs to MODEL-TEXT-GLM-MOE-DSA's
  W9 owner, not a build repair.
- If the shadow rename needs to touch anything beyond the two branch
  declarations and their immediate uses, STOP: the defect is bigger than the
  CI log shows.

## Git integration

One pull request, spec + fixes together (the repository default; the user
ordered this row mid-session with no split request). The squash body carries
`Closes #2372`.

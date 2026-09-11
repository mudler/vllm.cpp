ID: ISSUE-GH-2371
Title: ci.yml runs the retired issue-index test; the agent-record job reds every run since the retirement
Row: -
State: CLOSED
Kind: UNKNOWN
GitHub: 2371
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Observed
>
> Every CI run since the issue-index retirement lands: the "Canonical roadmap tables and links are consistent" step finishes its real work (checker `OK`, 113 tests `OK`) and then dies with exit 2:
>
> ```
> python3: can't open file '/home/runner/work/vllm.cpp/vllm.cpp/tests/scripts/test_check_issue_index_append_only.py': [Errno 2] No such file or directory
> ```
>
> Main's head `da2291339` reds the same job; PRs #2370 and #2368 inherit it.
>
> ## Root cause
>
> `.github/workflows/ci.yml:189` still invokes `tests/scripts/test_check_issue_index_append_only.py`, which the retirement (`7dc2ef1ea`) deleted together with the index. The replacement renderer's suite `tests/scripts/test_agent_issue_index.py` exists on main and is not wired into the step.
>
> ## Fix
>
> Re-point line 189 at `tests/scripts/test_agent_issue_index.py`. No checker semantics change; the step's intent (the record/index suites run in CI) is preserved.
>
> FOLLOWING_AGENTS_PROTOCOL
>
> Following-Agents-Protocol: true
> AI-Assisted: true
> Assisted-by: AGENT:zai-glm-5.3-flash [maki]

## Resolution

GitHub records closing pull request #2373 (https://github.com/mudler/vllm.cpp/pull/2373) merged on 2026-08-31 as commit `df2ad6d847f1e6b2ccac601ba77ba0dff055b2a1`. GitHub closed issue #2371 on 2026-08-31.

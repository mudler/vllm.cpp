ID: ISSUE-GH-3108
Title: fix: reap the Strix adapter resource tracker before exit
Row: BACKEND-GATE-ROCM-SGLANG
State: OPEN
Kind: UNKNOWN
GitHub: 3108
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-09
Updated: 2026-09-09
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-GATE-ROCM-SGLANG`
>
> Parent and owning scope: #3053. Owner: current Strix campaign operator. Performance campaign: #3076.
>
> Diagnostic job 5a78caa8-ee66-4ccf-b9e1-6572464af54e completed Qwen3-4B c4 warmup and the actual controller preflight, but the unchanged Adapter.close process-group gate failed. Instrumented snapshots before and after the real killpg(group,0) show only resource-tracker PID83808/starttime8526239, already Z with PPID1 and no device descriptors. EngineCore83809 was absent by shutdown acknowledgement. Cleanup SIGKILL did not change the zombie. This classifies this observed run only; snapshots perturb timing.
>
> Evidence: campaign NAS teardown-observation-3053.qBJJHR/shutdown-timeline.json, SHA256 5e6103e4fbc2e61cead0b04ba762615bd0dce4b60696745664b8b4a56610efc1. Exact recovered venv source provenance is retained in venv-lifecycle-sources.RdbAqT, Python3.12.3, base_prefix /usr. resource_tracker.py SHA256 b0099f5e2285fe2462b77c47bbf29220b57c778741a34f386c10c1e9940884c8; util.py SHA256 6752c4515ec69f82e9df64e017da490c3754e51d818c270ea1ad2d64e09268be.
>
> Required fix: establish terminal adapter-parent ownership and reap its resource tracker before successful acknowledgement/exit. Complete engine cleanup and tracker-using finalizers first. Preserve the existing absent-process-group gate, real live-survivor failures, bounded controller timeout, and original exceptions. Do not ignore zombies, reap foreign children, blindly call a blocking private method before its pipe users finish, or change model/runtime numerics.
>
> Commit a dedicated lifecycle spec before fresh test-first implementation. Enter the real adapter CLI using public engine doubles and a real multiprocessing tracker. Cover owned tracker, no tracker, repeated terminal cleanup, shutdown errors, live pipe holders, late finalizers, foreign/inherited ownership and real survivor refusal. Independent mutation review must delete the production reaping call and retain the failing original group gate. Operator repeats full verification and real Strix shutdown validation. No benchmark acceptance is implied.

## Resolution

-

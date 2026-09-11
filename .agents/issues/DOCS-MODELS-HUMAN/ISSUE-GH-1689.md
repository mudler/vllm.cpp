ID: ISSUE-GH-1689
Title: **All 13 pages under `docs/models/` are written for a reviewer of the change that produced them, not for a reader who wants to run the model.** Three shapes recur. Twelve of the thirteen open with the identical line `Use this page for <model> checkpoints, commands, supported arms, and current limitations`, which says nothing the title did not. Defect archaeology precedes the command: `qwen3-8-2-4t.md` was 593 lines and spent its first 100 on [#1123](https://github.com/mudler/vllm.cpp/issues/1123), [#1124](https://github.com/mudler/vllm.cpp/issues/1124) and [#1299](https://github.com/mudler/vllm.cpp/issues/1299) before saying what to type, reaching the serve command at line 434. And spec content sits in a public projection: rationale for a default and the history of a fixed defect belong in the row spec. FIXED IN FLOW: every page now leads with what the model is, what it costs, the exact command, and the honest limits, in that order. **No measured value changed** — a string-level retention check asserts that every checkpoint revision, sha256, byte count, decode figure, refusal message and issue link on the old `qwen3-8-2-4t.md` still resolves in the tree. The mechanism content removed from that page was NOT deleted: `docs/guides/expert-streaming.md` was a 17-line stub that the model page already claimed owned the config schema, precedence rule, statistics line and per-device conditions, so that material moved there and the stub became the owner it was described as. `docs/models/README.md` becomes a real index naming what each page answers. Related: [#1691](https://github.com/mudler/vllm.cpp/issues/1691), the stale container claim found by the same work
Row: DOCS-MODELS-HUMAN
State: UNKNOWN
Kind: record
GitHub: 1689
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:629`

### Frozen archive evidence

> | [#1689](https://github.com/mudler/vllm.cpp/issues/1689) | `DOCS-MODELS-HUMAN` | **All 13 pages under `docs/models/` are written for a reviewer of the change that produced them, not for a reader who wants to run the model.** Three shapes recur. Twelve of the thirteen open with the identical line `Use this page for <model> checkpoints, commands, supported arms, and current limitations`, which says nothing the title did not. Defect archaeology precedes the command: `qwen3-8-2-4t.md` was 593 lines and spent its first 100 on [#1123](https://github.com/mudler/vllm.cpp/issues/1123), [#1124](https://github.com/mudler/vllm.cpp/issues/1124) and [#1299](https://github.com/mudler/vllm.cpp/issues/1299) before saying what to type, reaching the serve command at line 434. And spec content sits in a public projection: rationale for a default and the history of a fixed defect belong in the row spec. FIXED IN FLOW: every page now leads with what the model is, what it costs, the exact command, and the honest limits, in that order. **No measured value changed** — a string-level retention check asserts that every checkpoint revision, sha256, byte count, decode figure, refusal message and issue link on the old `qwen3-8-2-4t.md` still resolves in the tree. The mechanism content removed from that page was NOT deleted: `docs/guides/expert-streaming.md` was a 17-line stub that the model page already claimed owned the config schema, precedence rule, statistics line and per-device conditions, so that material moved there and the stub became the owner it was described as. `docs/models/README.md` becomes a real index naming what each page answers. Related: [#1691](https://github.com/mudler/vllm.cpp/issues/1691), the stale container claim found by the same work | record |

## Resolution

-

ID: ISSUE-GH-1562
Title: **W6's oracle capture harness exists only as PROSE, and the FLASH_ATTN label is a post-hoc relabel of an uncommitted log.** `## Owed` O22 and O23 were written so the next agent would not pay three 51.75 GiB loads again; they describe the hook on `DFlashSpeculator.propose` below the `cg_mode == FULL` branch, the `torch.cuda.is_current_stream_capturing()` delegation, the resolved-backend read-back and the abort-on-zero -- and none of it is in the tree, nor is `w6-relabel.py`, nor any run log. Three consequences visible in the committed evidence: `dflash2_27b_spec_on_flash_attn.json` carries `attention_backend_source: "corrected from the run log by w6-relabel.py; the capture's original value came from VLLM_ATTENTION_BACKEND, which does not exist in this wheel and selected nothing"`, which does not meet the read-back rule O22 itself lays down and cannot be re-derived; the TRITON_ATTN golden's `hook_stats` reads `{propose_calls: 59, skipped_dummy: 1, skipped_capture: 0}` against 55 recorded blocks with contiguous `call` ids 3..57, so `59-1-0 = 58 != 55` and THREE propose calls are unaccounted for (now bounded one-sidedly by the gate and pinned at 3, still unexplained); and `## Owed` O17's discharge has no committed runner and no log, so its peak-RSS pair and its `[SPECTRACE]` lines cannot be re-derived either. Owed: commit the harness, the relabel script and the log, or record that they were lost with the lease
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1562
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:589`

### Frozen archive evidence

> | [#1562](https://github.com/mudler/vllm.cpp/issues/1562) | `SPEC-DFLASH2` | **W6's oracle capture harness exists only as PROSE, and the FLASH_ATTN label is a post-hoc relabel of an uncommitted log.** `## Owed` O22 and O23 were written so the next agent would not pay three 51.75 GiB loads again; they describe the hook on `DFlashSpeculator.propose` below the `cg_mode == FULL` branch, the `torch.cuda.is_current_stream_capturing()` delegation, the resolved-backend read-back and the abort-on-zero -- and none of it is in the tree, nor is `w6-relabel.py`, nor any run log. Three consequences visible in the committed evidence: `dflash2_27b_spec_on_flash_attn.json` carries `attention_backend_source: "corrected from the run log by w6-relabel.py; the capture's original value came from VLLM_ATTENTION_BACKEND, which does not exist in this wheel and selected nothing"`, which does not meet the read-back rule O22 itself lays down and cannot be re-derived; the TRITON_ATTN golden's `hook_stats` reads `{propose_calls: 59, skipped_dummy: 1, skipped_capture: 0}` against 55 recorded blocks with contiguous `call` ids 3..57, so `59-1-0 = 58 != 55` and THREE propose calls are unaccounted for (now bounded one-sidedly by the gate and pinned at 3, still unexplained); and `## Owed` O17's discharge has no committed runner and no log, so its peak-RSS pair and its `[SPECTRACE]` lines cannot be re-derived either. Owed: commit the harness, the relabel script and the log, or record that they were lost with the lease | bug |

## Resolution

-

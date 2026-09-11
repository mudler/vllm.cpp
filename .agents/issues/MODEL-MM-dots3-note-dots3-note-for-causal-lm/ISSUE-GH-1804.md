ID: ISSUE-GH-1804
Title: **`.agents/specs/dots3-note.md` §4 item 6 read `is_neox_style=False` as belonging to the dots3-note SLIDING rope ONLY, and it belongs to both MLA ropes** — the sentence would have sent a W3 implementer to rotate the 13 full-attention layers split-half NeoX. `Dots3NoteSlidingAttention` does pass `is_neox_style=False` literally (`model.py:408` @ vLLM `origin/main` `c205726108df54bb6fbf15b19e725a4a3add2b18`), which is the half W0 read; `Dots3NoteFullAttention` (`model.py:219`) inherits the SAME hard-coded value from `deepseek_v2.py`::`DeepseekV2MLAAttention.__init__` (`:1093-1098`). So the two geometries do NOT differ on the RoPE layout at all — they differ on the THETA, `swa_rope_theta` 5e4 on 33 layers against `rope_theta` 8e7 on 13. The polarity that DOES flip is the INDEXER's, and that is §4 trap 2's point rather than item 6's: `deepseek_v2.py:1148` sets the indexer rope to `is_neox_style = not indexer_rope_interleave`, so at DeepSeek-V3.2's absent-key default the indexer runs NeoX beside an MLA rope that is GPT-J, and `indexer_rope_interleave = True` (`configs/dots3_note.py:23`) is what makes dots3-note's two agree. Being wrong in this direction is SILENT: the same 64 coordinates are rotated either way, so nothing changes shape and nothing throws — the §4 defect class exactly, on a row that spec §6.4 says has no oracle anywhere to catch it. FIXED IN FLOW on `row/MODEL-MM-dots3-note-W1`: §4 item 6 corrected in place and it says what it used to say (`main` is never rewritten), `ParseDots3NoteParams` resolves `rope_is_neox_style = false` on BOTH geometries with the two citations beside it, and `tests/vllm/models/test_dots3_note_scaffold.cpp` asserts both plus the indexer's agreement with them. The assertion was captured RED against the NeoX reading, on an arm that compiled and ran, before the corrected value existed
Row: MODEL-MM-dots3-note-dots3-note-for-causal-lm
State: UNKNOWN
Kind: bug
GitHub: 1804
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:673`

### Frozen archive evidence

> | [#1804](https://github.com/mudler/vllm.cpp/issues/1804) | `MODEL-MM-dots3-note-dots3-note-for-causal-lm` | **`.agents/specs/dots3-note.md` §4 item 6 read `is_neox_style=False` as belonging to the dots3-note SLIDING rope ONLY, and it belongs to both MLA ropes** — the sentence would have sent a W3 implementer to rotate the 13 full-attention layers split-half NeoX. `Dots3NoteSlidingAttention` does pass `is_neox_style=False` literally (`model.py:408` @ vLLM `origin/main` `c205726108df54bb6fbf15b19e725a4a3add2b18`), which is the half W0 read; `Dots3NoteFullAttention` (`model.py:219`) inherits the SAME hard-coded value from `deepseek_v2.py`::`DeepseekV2MLAAttention.__init__` (`:1093-1098`). So the two geometries do NOT differ on the RoPE layout at all — they differ on the THETA, `swa_rope_theta` 5e4 on 33 layers against `rope_theta` 8e7 on 13. The polarity that DOES flip is the INDEXER's, and that is §4 trap 2's point rather than item 6's: `deepseek_v2.py:1148` sets the indexer rope to `is_neox_style = not indexer_rope_interleave`, so at DeepSeek-V3.2's absent-key default the indexer runs NeoX beside an MLA rope that is GPT-J, and `indexer_rope_interleave = True` (`configs/dots3_note.py:23`) is what makes dots3-note's two agree. Being wrong in this direction is SILENT: the same 64 coordinates are rotated either way, so nothing changes shape and nothing throws — the §4 defect class exactly, on a row that spec §6.4 says has no oracle anywhere to catch it. FIXED IN FLOW on `row/MODEL-MM-dots3-note-W1`: §4 item 6 corrected in place and it says what it used to say (`main` is never rewritten), `ParseDots3NoteParams` resolves `rope_is_neox_style = false` on BOTH geometries with the two citations beside it, and `tests/vllm/models/test_dots3_note_scaffold.cpp` asserts both plus the indexer's agreement with them. The assertion was captured RED against the NeoX reading, on an arm that compiled and ran, before the corrected value existed | bug |

## Resolution

-

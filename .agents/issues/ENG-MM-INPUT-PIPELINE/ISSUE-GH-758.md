ID: ISSUE-GH-758
Title: A multimodal refusal cannot distinguish a configured limit from an UNIMPLEMENTED arm. `Qwen3VLChatSupportedMmLimits()` declares the seam's honest ceiling `{"image": 1}` with video/audio absent, but the message a client gets is upstream's generic `At most 0 video(s) may be provided in one prompt.` — identical to what `--limit-mm-per-prompt '{"video": 0}'` produces. AGENTS.md requires an unimplemented arm be refused "with a message naming the missing piece"; #749 claimed the ceiling satisfies that and its review found it does not. The only present signal is by OMISSION (`ValidateNumItems` withholds the `--limit-mm-per-prompt` hint when raising the limit would not help). Not fixed in flow: naming the arm diverges from a verbatim-ported message three suites assert byte-for-byte, so it needs its own spec and fresh review. Found in the #749 review round (#607 wave L2, #686)
Row: ENG-MM-INPUT-PIPELINE
State: UNKNOWN
Kind: bug
GitHub: 758
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:185`

### Frozen archive evidence

> | [#758](https://github.com/mudler/vllm.cpp/issues/758) | `ENG-MM-INPUT-PIPELINE` | A multimodal refusal cannot distinguish a configured limit from an UNIMPLEMENTED arm. `Qwen3VLChatSupportedMmLimits()` declares the seam's honest ceiling `{"image": 1}` with video/audio absent, but the message a client gets is upstream's generic `At most 0 video(s) may be provided in one prompt.` — identical to what `--limit-mm-per-prompt '{"video": 0}'` produces. AGENTS.md requires an unimplemented arm be refused "with a message naming the missing piece"; #749 claimed the ceiling satisfies that and its review found it does not. The only present signal is by OMISSION (`ValidateNumItems` withholds the `--limit-mm-per-prompt` hint when raising the limit would not help). Not fixed in flow: naming the arm diverges from a verbatim-ported message three suites assert byte-for-byte, so it needs its own spec and fresh review. Found in the #749 review round (#607 wave L2, #686) | bug |

## Resolution

-

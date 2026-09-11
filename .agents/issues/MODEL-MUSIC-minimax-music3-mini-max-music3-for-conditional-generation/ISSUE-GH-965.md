ID: ISSUE-GH-965
Title: `windows-msvc-cpu`/`windows-msvc-vulkan` fail on EVERY open pull request with `C4456: declaration of 'loaded' hides previous local declaration` at `server_main.cpp:1315` — the speech engine's `loaded` nested inside the text engine's `loaded` at `:1025`, both already on `main`, and the ONLY warning in the job. **It is NOT [#645](https://github.com/mudler/vllm.cpp/issues/645)**, which is the `M_PI` regression in three LTX2 sources: a second cause hiding behind a known-red name, which is why "known-red" needs a MATCHED-ARM check and not a label. Confirmed pre-existing against three unrelated PRs (#956, #950, #939) that all fail identically. Invisible on `main` because `windows-msvc-*` are PR-only ([#584](https://github.com/mudler/vllm.cpp/issues/584)), so it presents to each author in turn as a red their own diff caused. FIXED IN FLOW while landing [#672](https://github.com/mudler/vllm.cpp/issues/672): the inner declaration is renamed, no detector weakened and no warning suppressed
Row: MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 965
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:266`

### Frozen archive evidence

> | [#965](https://github.com/mudler/vllm.cpp/issues/965) | `MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation` | `windows-msvc-cpu`/`windows-msvc-vulkan` fail on EVERY open pull request with `C4456: declaration of 'loaded' hides previous local declaration` at `server_main.cpp:1315` — the speech engine's `loaded` nested inside the text engine's `loaded` at `:1025`, both already on `main`, and the ONLY warning in the job. **It is NOT [#645](https://github.com/mudler/vllm.cpp/issues/645)**, which is the `M_PI` regression in three LTX2 sources: a second cause hiding behind a known-red name, which is why "known-red" needs a MATCHED-ARM check and not a label. Confirmed pre-existing against three unrelated PRs (#956, #950, #939) that all fail identically. Invisible on `main` because `windows-msvc-*` are PR-only ([#584](https://github.com/mudler/vllm.cpp/issues/584)), so it presents to each author in turn as a red their own diff caused. FIXED IN FLOW while landing [#672](https://github.com/mudler/vllm.cpp/issues/672): the inner declaration is renamed, no detector weakened and no warning suppressed | bug |

## Resolution

-

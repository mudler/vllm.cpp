ID: ISSUE-GH-1512
Title: **Five changes landed on MiniMax-Music3's two largest buckets in two days and not one of them has an end-to-end number on the shipped binary.** The depth incremental schedule (#1238), the depth **device arm** (#1309), the vocoder `Conv1d` tiling (#1334), `vt::SiluAndMul`'s rounding polarity (#1322) and the vocoder conv **f32 accumulator** (#1474). The tiling was measured at KERNEL level on synthetic weights with no checkpoint read; the device arm was gated for correctness and never timed; and `.agents/specs/minimax-music3.md` §13.10 retains the whole vocoder CUDA-vs-host axis as **VOID** — 0.37-0.40x with the f64 accumulator named as the leading suspect — because the samples were taken over a bypassed lease while a second session held the same box. #1474 removes that suspect and nothing re-measured it. The recorded baselines are 163.00 s at 4 s / 4 steps (§16.6b, `4568c6e71`) and 3269.789 s at 20 s / 30 steps (§15.7, CIFS checkpoint).
Row: MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation
State: UNKNOWN
Kind: UNKNOWN
GitHub: 1512
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:539`

### Frozen archive evidence

> | [#1512](https://github.com/mudler/vllm.cpp/issues/1512) | `MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation` | **Five changes landed on MiniMax-Music3's two largest buckets in two days and not one of them has an end-to-end number on the shipped binary.** The depth incremental schedule (#1238), the depth **device arm** (#1309), the vocoder `Conv1d` tiling (#1334), `vt::SiluAndMul`'s rounding polarity (#1322) and the vocoder conv **f32 accumulator** (#1474). The tiling was measured at KERNEL level on synthetic weights with no checkpoint read; the device arm was gated for correctness and never timed; and `.agents/specs/minimax-music3.md` §13.10 retains the whole vocoder CUDA-vs-host axis as **VOID** — 0.37-0.40x with the f64 accumulator named as the leading suspect — because the samples were taken over a bypassed lease while a second session held the same box. #1474 removes that suspect and nothing re-measured it. The recorded baselines are 163.00 s at 4 s / 4 steps (§16.6b, `4568c6e71`) and 3269.789 s at 20 s / 30 steps (§15.7, CIFS checkpoint). | — |

## Resolution

-

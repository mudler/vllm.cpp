# SPEC: W4d W5 — decode-shaped capture; mixed prefill steps run eager

Row: `BACKEND-TENSTORRENT-KEEPQUANT`. Parent:
[tenstorrent-27b-gdn-keepquant.md](tenstorrent-27b-gdn-keepquant.md) (W4),
[tenstorrent-keepquant.md](tenstorrent-keepquant.md).

## Problem (measured 2026-09-12 20:49, /tmp/w4-run.log)

After W4's packed GDN projections (slot residency 23.8 -> 7.7 GiB, all 64
blocks warm — validated on device), the 27B still OOMs in generation: a
635,699,200 B allocation (a [~640, 248320] f32 logits plane) with banks at
99.5 percent (4.2/4.27 GiB per bank, largest free block 29.8 MB).

Root cause: the single-shape graph driver (qwen3_5.cpp:11155-11244)
captures ONE shape — the mixed warm step, B ~= 640 tokens — and replays it
every step. The head's f32 output ([B, 248320]) is therefore 606 MiB on
EVERY step, though a decode step samples num_seqs (~10) rows (~16 MiB).
The gather-before-lm_head (qwen3_5.cpp:9517-9530) exists but only fires
when the driver passes non-empty `logits_indices`; the capture path passes
`{}` by design (the captured region covers ForwardLayers only).

## Design

Mirror vLLM's logits semantics (logits_processor: generation needs logits
at the last token per sequence; prompt logprobs only when requested):

1. The driver captures the decode graph at the DECODE shape
   (B = num_seqs), not the mixed warm shape. The mixed prefill step runs
   EAGERLY (the existing cold/eager path — no capture), passing
   per-sequence last-token indices so the head gathers.
2. The eager mixed step and the captured decode both consume the SAME
   resident inputs (word shadows, state shadows, persistent step buffers),
   which the pre-warm already populated — nothing new is baked.
3. Keep `VT_GDN_*` output-dtype rollbacks and the existing capture-safety
   guards exactly as they are.

Non-goals: prompt-logprobs support (the gate never requests it — when a
caller does, the eager path passes the full identity indices, which is
today's behaviour); dual-shape capture (deferred until a workload needs
it).

## Tests (red-first)

1. RED: the 27B gate case dies at the 606 MiB mixed-step ask on the
   pre-fix tree (/tmp/w4-run.log). GREEN: generation completes.
2. Focused driver test: a mixed step (prefill rows + decode rows) with
   `logits_indices` = last-token-per-seq allocates a head output of
   [num_seqs, vocab], never [T, vocab] — assert via the alloc-trace
   max-delta probe or the census.
3. No-regression net: the 0.8B vehicle gate (its capture is decode-shaped
   already), backend suite 71/71, bit-exact sweeps 4/4.

## Gates

The 27B gate verdict without the chunk knob, then the capture campaign
(device bootstrap -> reader+gap -> grading) and the W3+W4 landing
(checklist in memory: issue files, debug-print revert, per-piece
keep/revert, goldens quarantine).

## Stop conditions

- If the eager mixed step cannot reuse the persistent decode inputs (the
  capture baked pointers the eager step cannot rebuild), stop and scope a
  resident-input rebinding instead of forcing the eager arm.
- If decode-shaped capture changes ANY captured-region operand (the graph
  must re-warm), the re-warm cost is measured and recorded before
  proceeding — never silently absorbed.

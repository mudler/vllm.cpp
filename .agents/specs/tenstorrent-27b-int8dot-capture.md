# SPEC: W4d W6 — int8-dot as the captured keep-quant decode (no chunk chain in capture)

Row: `BACKEND-TENSTORRENT-KEEPQUANT`. Parent:
[tenstorrent-keepquant.md](tenstorrent-keepquant.md). Siblings:
[tenstorrent-27b-gdn-keepquant.md](tenstorrent-27b-gdn-keepquant.md) (W4,
landed), [tenstorrent-27b-decode-shape-capture.md](tenstorrent-27b-decode-shape-capture.md)
(W5 — superseded by this spec; see §Relationship).

## Problem (measured 2026-09-12/13)

The 27B Q4_K_M decode cannot run captured, and cannot run eager in
reasonable time:

- CAPTURED: the keep-quant matmuls decode their weight chunks through the
  E=1 chain (DecodeKeepQuantBlocksF32 -> DecodeKeepQuantWordsF32), and
  tt-metal pins EVERY capture-time allocation into the trace region for
  replay. The pinned sum reaches 3.8 GiB (mesh_trace.cpp:82: "Creating
  trace buffers of size 3817046016B") against <1 GiB free — the capture
  cannot fit, at any chunk size (the chunk size changes granularity, not
  the pinned volume: every chunk plane is pinned).
- EAGER (capture off): correct but 40-70+ s/token — the whole-model word
  decode re-runs per token through the chunk chain. An 8-hour bootstrap
  run died at SIGTERM mid-generation (/tmp/eager-boot4.log, RC=124,
  29/35 assertions, no ids dumped). The eager arm is not a system.

## The designed answer, already in the tree, default OFF

W4b (#3031) landed the int8-dot device kernel with this exact rationale
(tenstorrent_ops.cpp:2665): "One captured launch replaces the per-chunk
E=1 chain (the capture-demand gate this row owes)". Today it covers the
F32-OUT dense arm only, is default OFF (VT_TT_KEEPQUANT_INT8DOT), and its
e2e anchor failed the 0.8B band on ONE quantized-domain non-tie flip
(vehicle p5 tok7, 1125 mnats, determinism-proven by byte-identical
capture dumps x2).

## W6 scope

1. **int8-dot for every captured keep-quant matmul**: the bf16-out arms
   (in_proj_z, the down/o sinks) get the kernel with an explicit f32->
   bf16 store cast (the W4b comment names the bug: committing f32
   dev_out into a bf16 slot left f32 bytes at an f32 page geometry).
   The GDN split-packed arm (W4) and the FFN route through it. The head
   keeps f32-out (the #2534 divergence) — the int8-dot computes f32
   cells, so the head is a native fit.
2. **Zero E=1 chain inside the captured region, asserted**: a red-first
   CHECK (or alloc-trace assertion) that no `kq-decode` label fires
   between GraphCaptureScope open and close. RED today: the trace-gated
   ledger shows kq-decode/rows=2620 + repair deltas inside the capture
   (/tmp/w5-run3.log:27946-27948, fatal in end_trace_capture :27958).
3. **Per-arm golden re-anchoring, not a band change**: the int8-dot arm
   captures its OWN device pair (the 0.8B precedent: ambient vs captured
   pairs per arm). The known 1125-mnat deviation is recorded in the arm's
   golden header as a ratified property of the quantized-domain
   activation encoding — the EXISTING arms and bands are untouched. The
   oracle denomination stays llama.cpp b10451: the int8-dot arm's gaps
   are measured against the oracle gap golden like any arm.
4. **The trace region then fits**: one captured launch per matmul has no
   weight planes to pin; the 3.8 GiB collapses to the resident set.

## Relationship to W5

W5 (decode-shaped capture) solved the SHAPE half of the problem (the
[B=640, V] head plane). With W6, the per-matmul transients vanish, so
the capture economics hold at decode shapes; W5's decode-shaped capture
remains correct practice and its logits-gather half is ALREADY
implemented (the LastTokenLogitsIndices work). This spec supersedes W5's
trace-region framing, not its gather.

## Tests (red-first)

1. The capture-scope assertion (2) REDs today: kq-decode labels inside
   the capture (/tmp/w5-run3.log evidence).
2. The bf16-out int8-dot store: op-level test, f32 dev_out cast to the
   bf16 slot geometry (the W4b-named bug) — RED until the cast lands.
3. The 0.8B int8-dot arm's own captured pair: re-anchored, gated at the
   same 500-mnat band against its own anchor.
4. The 27B: generation completes CAPTURED; the gate verdict against the
   device pair (re-captured on the int8-dot arm).

## Gates

0.8B vehicle 16/16 on the default (E=1) arms; backend suite 71/71;
sweeps 4/4; the 27B captured gate as the e2e proof; APEX-I-Nano remains
owed to the IQ row (QUANT-GGUF-IQ-TENSTORRENT).

## Stop conditions

- If the int8-dot kernel cannot serve an arm without a numerics cliff
  beyond the recorded one-flip deviation, that arm stays E=1 and the
  capture restructures around it (split capture) — never a silent band
  move.
- If the trace region still overflows with zero chunk-chain transients,
  the remaining pinned set is attributed per-op with the census before
  any further lever.

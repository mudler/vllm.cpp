# `BENCH-QWEN38-EXL3-LONGCTX` — the head-to-head at the context length the server actually serves

Row: `BENCH-QWEN38-EXL3-LONGCTX`.
Issue: `ISSUE-LOCAL-01M2TNH3A9DKGWCRADM4Q8J30C`.
Base SHA: `7fa861392`.
Predecessors: [`bench-qwen38-exl3-variadic.md`](bench-qwen38-exl3-variadic.md) (the
harness and the mixed load), [`quant-exl3-recon-scratch.md`](quant-exl3-recon-scratch.md)
(W7, which this run must carry to reach c = 32).
Comparator: `MiaAI-Lab/exllamav3` @ `63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed`
([oracle](../oracles/exllamav3.md)). vLLM implements no EXL3.

## Now

`ACTIVE`. Spec committed before the harness change.

2026-09-18: the harness part landed on `row/BENCH-QWEN38-EXL3-LONGCTX`.
`build_corpus.py` carries the `XXL` band at `XXL_TARGET_CHARS = 21000`, and
`--weights` replaces the four `--weight-*` flags.

The band is sized to FIT, not to hit the 7000 tokens §2 asks for. The ratio is
the `XL` band's own, because `XXL` copies `XL`'s composition rule: over the same
144 prompts, `corpus-manifest.json` and `corpus-token-histogram.md` in
`docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/` give 9011 characters
against 2804 prompt tokens at the mean, and 10048 against 3233 at the top, or
**3.21 and 3.11 characters per prompt token**. An earlier draft of this section
used 3.4, which is a figure blended across the whole corpus; the English bands
render higher (`L` is 3.83 on the same pairing) and `XXL` is entirely Python.

At 3.21, 21000 characters is about **6540 prompt tokens**, about **2.2 times**
the `XL` band's realised median of 2928 tokens. The band's own overshoot at
`k_mid = 47` is 12.54% (4.26% k jitter plus 8.28% sampling spread, scaled
from XL's measured 22.7% at `k_mid = 20`), so the expected ceiling is about
23600 characters, or about **7600 tokens** at the adverse 3.11 pairing and about
7360 at the mean. The served configuration leaves 8192 - 192 - 57 = 7943 tokens
for the prompt body, so the headroom is about **346 tokens** at the adverse
pairing and about 583 at the mean, not the 1000 an earlier draft claimed. Every
published pairing leaves the band inside the budget. (An earlier draft displayed
the overshoot terms rounded, as 4.3% + 8.3% = 12.5%; those rounded terms add to
12.6%. The 1.125 constant comes from the unrounded terms and does not move.)

The 57 tokens of chat template is a MEASURED allowance.
[`qwen38-27b-exl3-variadic-gb10`](../../docs/benchmarks/qwen38-27b-exl3-variadic-gb10.md)
ran this corpus through both engines, each rendering its own chat template, and
read the counts back from every server's own `usage.prompt_tokens`: the corpus
measures 26 to 3233 tokens with the target's own tokenizer and the served
histogram runs 78 to 3290, so the template adds +52 at the corpus minimum and
+57 at the `XL` top. The larger delta is the one pinned, so that the refusal is
conservative at its own boundary; at 50 it admits a target of 21977 characters
whose ceiling is 7950 served tokens and overruns. That page also found the two
engines rendered the same corpus to the SAME counts, so the two-tokenizer
caution earlier drafts asserted is not something this tree has measured. One
agreeing run is not a guarantee for a band no published run has built, so
`G-FITS` reads both histograms back and decides anyway.

The first draft targeted 7000 tokens at 23800 characters, which is about 7900
tokens at the ceiling and leaves nothing. That margin was rejected: a band that
overruns the context is voided by `G-FITS` and costs the dgx lease it was
measured on.

`XXL` carries no default weight, so the four-band default is byte-identical to
the predecessor: the pre-change and post-change generators produced the same
corpus sha256 on five `(count, seed)` pairs over the same fixture sources, and
`tests/scripts/test_variadic_harness.py` pins that sha256 as a golden. The run
itself, `job.sh`, and the publication are not in this change.

## Owed

- **The `XXL` band lands unreached.** `benchmarks/variadic/job.sh` calls
  `build_corpus.py` without `--weights`, and the default distribution gives
  `XXL` zero items, so no default path builds a single prompt in the band. The
  generator, the band and its controls land ahead of the wiring deliberately:
  the run is the next step of this same row, and a corpus band has to exist and
  be pinned before a job can be pointed at it.
- **What reaches it, and who owns that.** `job.sh` gains the `XXL` weights, as a
  variable beside the rungs and the corpus count it already carries, in the wave
  of this row that runs the sweep. The row is `BENCH-QWEN38-EXL3-LONGCTX` and
  the issue is `ISSUE-LOCAL-01M2TNH3A9DKGWCRADM4Q8J30C`.
- Until then the band is reachable only by naming `--weights` on the command
  line, which is what `tests/scripts/test_variadic_harness.py`
  `LongBandIsRequestable` exercises.

## 1. The question

Does our prefill lead over exllamav3 hold at the context this server actually
serves? Every published band stops at about 3.3k prompt tokens, which is 40% of
the `--max-model-len 8192` both engines run.

The prediction is that the lead widens, because #3150's reconstruct + cuBLAS path
amortizes its one-time dequantization over more rows. The counter-pressure is on
our side too: draft acceptance falls with context (0.49 at 324 tokens, 0.37 at
8159), so decode throughput and TTFT move in opposite directions with length.
Both effects are ours; neither has been measured against their engine.

## 2. Scope

- One new corpus band, `XXL`, sized to FIT rather than to hit a round number:
  21000 characters, about 6540 prompt tokens at the 3.21 characters per token the
  published `XL` band realised (`## Now` carries the derivation and its
  provenance), with an expected ceiling near 7600 at the adverse 3.11 pairing and
  about 346 tokens of headroom under `8192 - 192 output - 57 template`, where the
  57-token allowance is the larger of the two template deltas
  `docs/benchmarks/qwen38-27b-exl3-variadic-gb10.md` measured on this corpus
  (+52 at the corpus minimum, 26 to 78 tokens; +57 at the `XL` top, 3233 to
  3290). A first draft targeted 7000 tokens (23800 characters), which leaves no
  headroom at all once the band's own 12.5% overshoot is applied; that was
  rejected, because a band that
  overshoots is voided by G-FITS and costs a lease. 6540 tokens is about 2.2x the
  `XL` band's realised median and well past the 3.3k where every published band
  stops.
- The band is built from the same HumanEval source as `XL`, by the same
  concatenation rule, so length is the only variable that changes.
- The band weights become a command-line knob, because a run that wants long
  prompts must be able to ask for them without editing the generator.
- The publication is a NEW benchmark id, `qwen38-27b-exl3-longctx-gb10`. The
  existing `qwen38-27b-exl3-variadic-gb10` page keeps its numbers and its
  disposition; it measured a different corpus on a tree that predates #3150, and
  nothing here supersedes it.

Out of scope: a matched KV configuration (theirs is `-cs 262144`, ours auto-fits
8192; that is `#2620`), and any context above 8192, which needs a different
served configuration.

## 3. Design

- `build_corpus.py` gains `XXL_TARGET_CHARS` and a `--weights` argument. The
  default weights keep the existing four-band distribution byte-for-byte, so an
  unchanged invocation still produces the predecessor's corpus.
- The run uses weights that put real mass in `XXL` while keeping the shorter
  bands for continuity.
- Both engines, c = 1 and c = 16, two rounds, arms and rung order reversed in
  round 2, exactly as the predecessor harness does.
- c = 32 only if W7 has landed and its G-MEM gate passed; otherwise the rung is
  refused by name rather than run, because the host OOM is what W7 fixes.

## 4. Gates

- `G-BYTES`: the three checkpoint shard sha256 values recomputed on the device
  match the pins.
- `G-FITS`: every realised `usage.prompt_tokens` plus `max_tokens` is below the
  served context on both engines, read back from each server's own usage. A
  truncated or refused prompt voids the band.
- `G-RESOLVED`: our server reports the configured `max_num_seqs` against the KV
  pool, so no leg runs below its rung.
- `G-SPREAD`: two rounds per cell, and the round-to-round spread is published
  beside every value.

## 5. Risks

- **The band does not fit.** `XXL_TARGET_CHARS = 21000` is a target in
  CHARACTERS, and the about 6540 prompt tokens it converts to rests on one
  checkpoint's tokenizer and on the 57-token chat-template allowance
  `docs/benchmarks/qwen38-27b-exl3-variadic-gb10.md` measured on this corpus.
  That page found both engines rendered the corpus to the same counts, but it
  never built this band, and neither delta it measured was taken on a prompt of
  this length.
  `tests/scripts/test_variadic_harness.py`
  `test_the_xxl_target_fits_the_served_context` refuses a target whose expected
  ceiling does not fit, with no GPU; G-FITS then reads the realised counts back
  from both servers and voids the band rather than publishing a truncation.
- **Their engine refuses the length.** Their card runs `-cs 262144`, so it should
  not, but a refusal is recorded as their result, not worked around.
- **Acceptance collapse confounds decode.** Report acceptance per band from both
  engines' own counters, so a TPOT change is attributable.
- **dgx:gpu0 loses its rc stream under load.** Every leg is written to the share
  as it completes, and the job is resumable, as the predecessor's is.

## 6. Evidence

`docs/bench-evidence/qwen38-27b-exl3-longctx-<date>/`: the job as run, the corpus
manifest with its sha256, the per-leg JSON, the realised token histogram, both
engines' acceptance, and the lease ids and boot ids of every leg.

## 7. Stop conditions

- G-FITS fails: publish the refusal, not the numbers.
- W7 has not landed with G-MEM green: run c = 1 and c = 16 only, and record c = 32
  as refused with the issue that owns it.

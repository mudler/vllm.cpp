# Spec — `tt_clock_state` W2: the firmware claimed-max pin, the pyluwen sampler, and the #2003 resolution

Issues: [#2005](https://github.com/mudler/vllm.cpp/issues/2005) (the `## Owed`
residuals of [tt-clock-state.md](tt-clock-state.md)) and its first consumer
[#2003](https://github.com/mudler/vllm.cpp/issues/2003). Owner rows:
`BACKEND-TENSTORRENT` (the tool) and `BACKEND-TENSTORRENT-HOST-FREE-FORWARD`
(the #2003 decision). Branch `bench/tt-clock-state-w2`, base `a1d04bbc9`.

## Scope

Three closes, one PR:

1. **Verified claimed-max pin (#2005 owed).** The W1 tool carried
   `--claimed-max-aiclk-mhz` as folklore with an UNVERIFIED provenance
   string. The board verifies its own cap: every `tt-smi -s` snapshot
   already carries `device_info[0].limits.asic_fmax`, read by tt-smi's
   `get_bh_chip_limits` from the smbus-telemetry field `AICLK_LIMIT_MAX`
   (`tt-smi/tt_smi/backend.py:830`). Measured live on thalia (P150a, board
   `000004033191406b`, fw bundle 19.7.1.0, 2026-08-30):
   `AICLK_LIMIT_MAX = 0x546` = **1350**, and `AICLK_ARB_MAX` low word
   `0x546` with `AICLK_ARB_MIN` low word `0x320` = 800 — the exact
   two-state range the W1 windows observed. The tool stops treating the
   cap as folklore: `sample_once` records the firmware limit from the
   snapshot it already parses, `fold` carries
   `firmware_aiclk_limit_max` + a telemetry provenance string, and the
   claimed-max resolution order becomes CLI override > firmware readout >
   None (never guessed). The judge reports the busy-slice median against
   the resolved cap, so "pegged at cap" becomes a stated, checkable fact.
2. **In-process pyluwen sampling (#2005 owed).** `--sampler pyluwen` runs
   the same snapshot loop in-process through pyluwen (the library tt-smi
   itself binds), removing the ~430 ms subprocess startup that caps
   practical cadence near 2 Hz. The subprocess path stays the default;
   pyluwen is a same-record-shape alternative, skipped with a stated
   reason when the import is unavailable. No device is needed for the
   suite: the sampler is exercised through a fake chip object.
3. **The #2003 resolution.** The opt-out (host-hybrid) arm beat the
   shipped host-free DEFAULT 1.24–1.25× clock-attributed (W1 wired leg).
   This unit attributes the movement and lands the polarity the
   measurement supports:
   - Per-op delta of the host-hybrid path over
     `b86e3705f..21fe11cf1` (offline archaeology; prime candidates are
     the `EnsureDevice2D` interior-slice and stale-host-byte fixes
     `353511e72`/`101b415d7`, which sit exactly on the staging primitive
     the hybrid arm exercises per op and the host-free arm mostly
     avoids — consistent with "default unchanged, opt-out moved ~2.5×").
   - One P150 confirming A/B under the sampler: order-alternated pairs,
     `--repeat 5` with run 1 discarded, `--leg-pid` armed, judge PASS on
     busy slices, firmware cap recorded.
   - The polarity record: if the confirming A/B reproduces the
     inversion, the default polarity for the affected shape is decided
     BY the measurement and landed in the same change (env default flip
     or a documented stand-pat with the attribution as the reason),
     never by editing thresholds to make an arm look better.

## Policy decision (recorded, closes a W1 Owed bullet)

The two-state-governor spread rule STAYS an offline refold
(`tools/bench/tt_refold_busy.py`). The tool keeps NVIDIA-parity
rule-for-rule semantics; no `--spread-scope busy` flag. Decision:
developer, 2026-08-30, recorded in
`.agents/developer-preferences.md` alongside the PR-shape answer for
this row. A future change to this needs a new red-before mutation, per
the W1 spec.

## Upstream anchors

No upstream implements TT host-side clock gating; the reference is our
own W1 helper (`tools/bench/tt_clock_state.py`) and its NVIDIA twin
(`tools/bench/gpu_clock_state.py`). The limit field is tt-smi's
`get_bh_chip_limits` (`tt_smi/backend.py:830-833`) over the Blackhole
smbus telemetry struct (`AICLK_LIMIT_MAX`); pyluwen is the binding
tt-smi itself uses (`tt-smi` venv, luwen crates under `~/Sources/tt/luwen`).

## Tests

`tests/tools/test_tt_clock_state.py` grows:

- Firmware-limit readout: a snapshot fixture with
  `limits.asic_fmax = "0x546"` produces a sample carrying
  `fw_aiclk_limit_max == 1350`; a snapshot without the limits block
  produces `None` (NOT-APPLICABLE stated, never guessed); fold surfaces
  the field + provenance; CLI override still wins over firmware; judge
  output states busy-median-vs-cap when a cap resolved.
- pyluwen sampler: fake chip fixtures produce the identical sample dict
  shape; unavailable-import path returns a stated skip reason; the
  subprocess default is unchanged when the flag is absent.

Each new gate constant mutated red (the resolution-order rule is a
gate).

## Gates

Suite green; preflight; the confirming A/B leg on thalia (one
`$HOME/gpu.lock` hold, `tt-smi -r` first, sampler armed); records
appended to `.agents/benchmark-record.md`; `docs/benchmarks/open-gaps.md`
host-free row updated; comments left on #2005 and #2003.

## Stop conditions

- pyluwen import fails in the serving venv: the subprocess sampler
  stays default, the flag ships with the stated skip reason, and the
  cadence claim is retracted in the record (the pin and #2003 are
  unaffected).
- The confirming A/B does NOT reproduce the inversion: the polarity
  stands as shipped, #2003 is re-scoped onto the discrepancy with the
  new measurements, and nothing flips on one session's numbers.
- The attribution contradicts the EnsureDevice2D hypothesis: the
  contradiction is the finding; the polarity decision waits for the
  corrected mechanism and the PR carries records only.

## Git integration

One PR for spec and implementation (row claim answer 2026-08-30,
recorded in developer-preferences; spread-scope stays offline refold;
pyluwen in scope — same answer session).

## Outcome

All three closes landed (2026-08-30):

1. **Firmware claimed-max verified (#2005).** `limits.asic_fmax` on the
   P150 is the DECIMAL string "1350" (0x546 is the raw smbus word), so
   `_limit_int` accepts either and guesses nothing; provenance ships per
   window as "tt-smi smbus telemetry AICLK_LIMIT_MAX
   (get_bh_chip_limits, tt_smi backend.py:830)". The W1 "UNVERIFIED pin
   owed" is retired by the confirming A/B, where all six leg windows and
   both cadence windows resolved the cap.
2. **pyluwen in-process sampler (#2005).** `--sampler pyluwen`, subprocess
   stays default. On the real board it holds the requested 0.25 s cadence
   (81 samples/20 s vs the subprocess sampler's 30, ~2.7x) with an
   identical sample shape; run 1's windows at 1 s cadence fell under the
   30-busy-sample floor on the faster arm, which is the concrete case the
   flag exists for. System python3 has no pyluwen, so the stated-skip
   path is real and tested.
3. **#2003 decided: documented stand-pat.** The confirming A/B reproduced
   the inversion at clock parity — default eager median 10.998 tok/s vs
   14.299 for `VT_TT_HOST_FREE_DECODE=0`, ratio 1.300 median / 1.294 mean,
   judge PASS, every busy slice of both arms exactly {1350} MHz
   (`benchmark-record.md`, 2026-08-30 entry). The EnsureDevice2D
   hypothesis was REJECTED by the per-op delta (`353511e72`/`101b415d7`
   only ADD staging work), so stop condition 3 fired: the PR carries
   records only, the default stays host-free, and the corrected-mechanism
   residual is owned in the host-free spec `## Owed`.

Rejected and why: a `--spread-scope busy` flag (policy decision above —
the refold stays offline); flipping the `VT_TT_HOST_FREE_DECODE` default
on the opt-out's win (a faster number is not a corrected mechanism);
widening the spread rule inside the tool (NVIDIA parity stays
rule-for-rule).

Process note, disclosed: the fresh-review step ran as a
coordinator-executed 7/7 mutation matrix after both the implementer and
reviewer subagents died of context overflow; each mutation's catching
test is asserted unique and the head was byte-verified after restore
(first commit body carries the detail).

## Now

`DONE` at the merge of the W2 pull request (implementation `4fe8a5eaf`,
records + this Outcome in the second commit). Follow-ups owned elsewhere:
the corrected mechanism for the opt-out arm lives in the host-free spec
`## Owed`; #2107 is the next lever candidate and needs its own spec and
P150 session.

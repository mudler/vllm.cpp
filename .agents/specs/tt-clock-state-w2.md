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

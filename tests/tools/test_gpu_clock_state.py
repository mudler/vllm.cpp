"""Clock-state recording and assertion (`BENCH-ASSERT-CLOCK-STATE`, #543).

Every case here is synthetic: no GPU, no `nvidia-smi`, no driver. The one real
input is the CSV row captured read-only from `dgx.casa` on 2026-08-12, which
fixes the parser's format contract rather than a value.

The numbers the thresholds are argued from live in
`.agents/specs/bench-assert-clock-state.md` and in #543:

    boot f6bbbfc6  n=61  min 2398 / med 2470 / max 2489   82.1664 ms/step
    boot 2fca2b02  n=50  flat 2190                        88.1000 ms/step

and the byte-identical `marlin::Marlin` control moved 45.2845 -> 49.6544
ms/step across that pair.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from tools.bench.gpu_clock_state import (
    BENIGN_THROTTLE_MASK,
    CLOCK_TIME_TRANSFER,
    MAX_CROSS_ARM_OFFSET_PCT,
    MAX_WITHIN_RUN_SPREAD_PCT,
    MIN_BUSY_FRACTION,
    MIN_BUSY_SAMPLES,
    build_clock_record,
    clock_reasons,
    compare_clock_records,
    merge_clock_records,
    parse_query_row,
    read_boot_id,
    run_sampler,
    summarize_sm_clocks,
    validate_clock_record,
)
from tools.bench.serve_low_common import HarnessError

# Verbatim `nvidia-smi --query-gpu=... --format=csv,noheader` output, dgx.casa,
# driver 580.159.03, 2026-08-12. The trailing `utilization.gpu` column is the
# one field the live probe did not request; its `%` unit form is asserted below.
LIVE_ROW = (
    "0, NVIDIA GB10, 580.159.03, 2190 MHz, 3003 MHz, 2418 MHz, "
    "0x0000000000000000, Enabled, 97 %"
)
LIVE_ROW_NOUNITS = "0, NVIDIA GB10, 580.159.03, 2190, 3003, 2418, 0x0000000000000000, Enabled, 97"

BOOT_GOOD = "f6bbbfc6-0000-4000-8000-000000000000"
BOOT_BAD = "2fca2b02-0000-4000-8000-000000000000"


def _samples(values, *, utilization=97, throttle="0x0000000000000000"):
    return [
        {
            "clocks_applications_graphics_mhz": 2418,
            "clocks_max_sm_mhz": 3003,
            "driver_version": "580.159.03",
            "gpu_name": "NVIDIA GB10",
            "persistence_mode": "Enabled",
            "sm_clock_mhz": value,
            "throttle_reasons_active": throttle,
            "utilization_gpu_pct": utilization,
        }
        for value in values
    ]


def _window(values):
    """Repeat a clock pattern until it clears ``MIN_BUSY_SAMPLES``.

    Whole-list repetition preserves min, median, max and therefore
    ``spread_pct`` EXACTLY, so every threshold case below still asserts what it
    asserted before the coverage floor existed; only the sample COUNT changes,
    which is the floor's entire subject. A fixture leg of three samples was
    never a leg anyone could have measured -- the driver samples at 1 Hz across
    a bench loop of minutes.
    """

    values = list(values)
    repeats = -(-MIN_BUSY_SAMPLES // len(values))
    return values * repeats


def _record(values, *, boot_id=BOOT_GOOD, **kwargs):
    return build_clock_record(_samples(_window(values), **kwargs), boot_id=boot_id)


class ParseTests(unittest.TestCase):
    def test_live_row_with_units(self) -> None:
        sample = parse_query_row(LIVE_ROW)
        self.assertEqual(sample["gpu_name"], "NVIDIA GB10")
        self.assertEqual(sample["driver_version"], "580.159.03")
        self.assertEqual(sample["sm_clock_mhz"], 2190)
        self.assertEqual(sample["clocks_max_sm_mhz"], 3003)
        self.assertEqual(sample["clocks_applications_graphics_mhz"], 2418)
        self.assertEqual(sample["throttle_reasons_active"], "0x0000000000000000")
        self.assertEqual(sample["persistence_mode"], "Enabled")
        self.assertEqual(sample["utilization_gpu_pct"], 97)

    def test_nounits_row_parses_identically(self) -> None:
        self.assertEqual(parse_query_row(LIVE_ROW), parse_query_row(LIVE_ROW_NOUNITS))

    def test_not_available_is_a_refusal_never_a_zero(self) -> None:
        row = LIVE_ROW.replace("2190 MHz", "[N/A]")
        with self.assertRaises(HarnessError) as caught:
            parse_query_row(row)
        self.assertIn("clocks.sm", str(caught.exception))

    def test_short_row_is_refused(self) -> None:
        with self.assertRaises(HarnessError) as caught:
            parse_query_row("0, NVIDIA GB10, 580.159.03")
        self.assertIn("expected", str(caught.exception))


class BootIdTests(unittest.TestCase):
    def test_boot_id_is_read_and_stripped(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = pathlib.Path(raw) / "boot_id"
            path.write_text("13dc5579-455c-45c8-8e4d-d09c457fa826\n", encoding="utf-8")
            self.assertEqual(read_boot_id(path), "13dc5579-455c-45c8-8e4d-d09c457fa826")

    def test_absent_boot_id_refuses_rather_than_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(HarnessError) as caught:
                read_boot_id(pathlib.Path(raw) / "absent")
            self.assertIn("boot id", str(caught.exception))

    def test_empty_boot_id_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = pathlib.Path(raw) / "boot_id"
            path.write_text("\n", encoding="utf-8")
            with self.assertRaises(HarnessError):
                read_boot_id(path)


class SummaryTests(unittest.TestCase):
    def test_min_median_max_and_n_over_the_window(self) -> None:
        summary = summarize_sm_clocks([2398, 2470, 2489, 2470])
        self.assertEqual(summary["n"], 4)
        self.assertEqual(summary["min"], 2398)
        self.assertEqual(summary["max"], 2489)
        self.assertEqual(summary["median"], 2470.0)
        self.assertAlmostEqual(summary["spread_pct"], (2489 - 2398) / 2470.0 * 100.0)

    def test_the_clean_boot_window_is_368_percent(self) -> None:
        # The observed clean window: this is the value the threshold has to
        # accept, so it is pinned here rather than left implicit.
        summary = summarize_sm_clocks([2398, 2470, 2489])
        self.assertAlmostEqual(summary["spread_pct"], 3.6842, places=3)
        self.assertLess(summary["spread_pct"], MAX_WITHIN_RUN_SPREAD_PCT)

    def test_empty_window_refuses(self) -> None:
        with self.assertRaises(HarnessError):
            summarize_sm_clocks([])

    def test_idle_samples_are_excluded_and_counted_not_dropped(self) -> None:
        record = build_clock_record(
            [
                *_samples([2470, 2470]),
                *_samples([300], utilization=0),
            ],
            boot_id=BOOT_GOOD,
        )
        self.assertEqual(record["idle_samples_excluded"], 1)
        self.assertEqual(record["sm_clock_mhz"]["n"], 2)
        self.assertEqual(record["sm_clock_mhz"]["min"], 2470)

    def test_an_entirely_idle_window_is_refused(self) -> None:
        with self.assertRaises(HarnessError) as caught:
            build_clock_record(_samples([300, 300], utilization=0), boot_id=BOOT_GOOD)
        self.assertIn("idle", str(caught.exception))

    def test_static_fields_must_not_change_mid_window(self) -> None:
        samples = [*_samples([2470]), *_samples([2470])]
        samples[1]["clocks_max_sm_mhz"] = 2000
        with self.assertRaises(HarnessError) as caught:
            build_clock_record(samples, boot_id=BOOT_GOOD)
        self.assertIn("clocks_max_sm_mhz", str(caught.exception))

    def test_throttle_reasons_are_a_sorted_union(self) -> None:
        samples = [*_samples([2470]), *_samples([2470], throttle="0x0000000000000001")]
        record = build_clock_record(samples, boot_id=BOOT_GOOD)
        self.assertEqual(
            record["throttle_reasons_active"],
            ["0x0000000000000000", "0x0000000000000001"],
        )


class ValidationTests(unittest.TestCase):
    def test_a_complete_record_validates(self) -> None:
        validate_clock_record(_record([2398, 2470, 2489]), label="ours")

    def test_every_required_field_is_required(self) -> None:
        for field in (
            "boot_id",
            "clocks_applications_graphics_mhz",
            "clocks_max_sm_mhz",
            "driver_version",
            "gpu_name",
            "idle_samples_excluded",
            "persistence_mode",
            "sm_clock_mhz",
            "throttle_reasons_active",
        ):
            with self.subTest(field=field):
                record = _record([2470])
                record.pop(field)
                with self.assertRaises(HarnessError) as caught:
                    validate_clock_record(record, label="ours")
                self.assertIn(field, str(caught.exception))

    def test_non_finite_clock_is_refused(self) -> None:
        record = _record([2470])
        record["sm_clock_mhz"]["median"] = float("inf")
        with self.assertRaises(HarnessError):
            validate_clock_record(record, label="ours")

    def test_non_positive_clock_is_refused(self) -> None:
        record = _record([2470])
        record["sm_clock_mhz"]["median"] = 0.0
        with self.assertRaises(HarnessError) as caught:
            validate_clock_record(record, label="ours")
        self.assertIn("positive", str(caught.exception))

    def test_a_non_integer_idle_count_is_refused(self) -> None:
        # The floors divide by `busy + idle`, so an idle count that is not a
        # non-negative integer has to fail validation rather than reach the
        # arithmetic: `int("many")` is an uncaught ValueError, not a reason.
        for value in ("many", 2.5, -1, True, None):
            with self.subTest(value=value):
                record = _record([2470])
                record["idle_samples_excluded"] = value
                with self.assertRaises(HarnessError) as caught:
                    validate_clock_record(record, label="ours")
                self.assertIn("idle_samples_excluded", str(caught.exception))

    def test_a_non_integer_idle_count_becomes_a_reason_not_a_crash(self) -> None:
        record = _record([2470])
        record["idle_samples_excluded"] = "many"
        reasons = clock_reasons(record, label="ours")
        self.assertEqual(len(reasons), 1)
        self.assertIn("idle_samples_excluded", reasons[0])

    def test_empty_throttle_list_is_refused(self) -> None:
        record = _record([2470])
        record["throttle_reasons_active"] = []
        with self.assertRaises(HarnessError):
            validate_clock_record(record, label="ours")

    def test_a_json_round_trip_still_validates(self) -> None:
        record = json.loads(json.dumps(_record([2398, 2470, 2489])))
        validate_clock_record(record, label="ours")


class WithinRunTests(unittest.TestCase):
    def test_the_clean_window_is_established(self) -> None:
        self.assertEqual(clock_reasons(_record([2398, 2470, 2489]), label="ours"), [])

    def test_the_flat_degraded_window_is_also_established(self) -> None:
        # 2190 flat is the WRONG clock, not an unstable one. The within-run rule
        # cannot see that; only the cross-arm rule can. Pinned so a future edit
        # cannot quietly repurpose the spread rule into an absolute-clock rule.
        self.assertEqual(clock_reasons(_record([2190] * 50), label="vllm"), [])

    def test_over_spread_window_is_not_established(self) -> None:
        # The two probes eight minutes apart inside ONE boot: 2398 against 1781.
        reasons = clock_reasons(_record([1781, 2100, 2398]), label="ours")
        self.assertEqual(len(reasons), 1)
        self.assertIn("spread", reasons[0])
        self.assertIn(str(MAX_WITHIN_RUN_SPREAD_PCT), reasons[0])

    def test_the_threshold_is_inclusive_on_both_sides(self) -> None:
        # median 2000 -> a 100 MHz range is exactly 5.0%.
        self.assertEqual(clock_reasons(_record([1950, 2000, 2050]), label="ours"), [])
        self.assertEqual(len(clock_reasons(_record([1949, 2000, 2050]), label="ours")), 1)

    def test_a_non_benign_throttle_bit_is_not_established(self) -> None:
        for bit, name in (
            (0x4, "SwPowerCap"),
            (0x8, "HwSlowdown"),
            (0x20, "SwThermalSlowdown"),
            (0x40, "HwThermalSlowdown"),
            (0x80, "HwPowerBrakeSlowdown"),
        ):
            with self.subTest(name=name):
                record = _record([2470], throttle=f"0x{bit:016x}")
                reasons = clock_reasons(record, label="ours")
                self.assertEqual(len(reasons), 1)
                self.assertIn("throttl", reasons[0])

    def test_benign_bits_are_accepted(self) -> None:
        for bit in (0x0, 0x1, 0x2, 0x100):
            with self.subTest(bit=bit):
                self.assertTrue(bit & ~BENIGN_THROTTLE_MASK == 0)
                record = _record([2470], throttle=f"0x{bit:016x}")
                self.assertEqual(clock_reasons(record, label="ours"), [])

    def test_persistence_off_is_not_established(self) -> None:
        record = _record([2470])
        record["persistence_mode"] = "Disabled"
        reasons = clock_reasons(record, label="ours")
        self.assertEqual(len(reasons), 1)
        self.assertIn("persistence", reasons[0])

    def test_one_busy_sample_is_not_the_cleanest_possible_record(self) -> None:
        """The incentive without a floor is INVERTED, so the floor is pinned.

        `spread_pct` over n == 1 is definitionally 0.00% -- the best score the
        gate can award -- so a window the sampler barely observed outscores one
        it actually watched. The record below is exactly that shape and must be
        refused on the COUNT, not on the spread it has no right to report.
        """

        record = build_clock_record(
            [*_samples([2470]), *_samples([300] * 300, utilization=0)],
            boot_id=BOOT_GOOD,
        )
        self.assertEqual(record["sm_clock_mhz"]["n"], 1)
        self.assertEqual(record["sm_clock_mhz"]["spread_pct"], 0.0)
        reasons = clock_reasons(record, label="ours")
        self.assertTrue(any("retained only 1 busy" in reason for reason in reasons))
        self.assertTrue(any(str(MIN_BUSY_SAMPLES) in reason for reason in reasons))

    def test_the_busy_sample_floor_is_inclusive(self) -> None:
        exactly = build_clock_record(
            _samples([2470] * MIN_BUSY_SAMPLES), boot_id=BOOT_GOOD
        )
        self.assertEqual(clock_reasons(exactly, label="ours"), [])
        one_short = build_clock_record(
            _samples([2470] * (MIN_BUSY_SAMPLES - 1)), boot_id=BOOT_GOOD
        )
        self.assertEqual(len(clock_reasons(one_short, label="ours")), 1)
        self.assertIn("retained only", clock_reasons(one_short, label="ours")[0])

    def test_a_diluted_window_is_refused_even_with_enough_busy_samples(self) -> None:
        """The count floor alone does not catch dilution; the fraction does.

        30 busy samples among 3000 idle clears the count and still reports a
        spread over 1% of the window.
        """

        record = build_clock_record(
            [
                *_samples([2470] * MIN_BUSY_SAMPLES),
                *_samples([300] * (MIN_BUSY_SAMPLES * 100), utilization=0),
            ],
            boot_id=BOOT_GOOD,
        )
        self.assertEqual(record["sm_clock_mhz"]["n"], MIN_BUSY_SAMPLES)
        reasons = clock_reasons(record, label="ours")
        self.assertFalse(any("retained only" in reason for reason in reasons))
        self.assertTrue(any("was idle for" in reason for reason in reasons))

    def test_the_busy_fraction_floor_is_inclusive(self) -> None:
        busy = MIN_BUSY_SAMPLES * 2
        idle = int(busy * (1.0 - MIN_BUSY_FRACTION) / MIN_BUSY_FRACTION)
        at_floor = build_clock_record(
            [*_samples([2470] * busy), *_samples([300] * idle, utilization=0)],
            boot_id=BOOT_GOOD,
        )
        self.assertEqual(clock_reasons(at_floor, label="ours"), [])
        below = build_clock_record(
            [*_samples([2470] * busy), *_samples([300] * (idle + 1), utilization=0)],
            boot_id=BOOT_GOOD,
        )
        self.assertTrue(any("was idle for" in reason for reason in clock_reasons(below, label="ours")))

    def test_a_malformed_record_becomes_a_reason_not_a_crash(self) -> None:
        record = _record([2470])
        record.pop("boot_id")
        reasons = clock_reasons(record, label="ours")
        self.assertEqual(len(reasons), 1)
        self.assertIn("boot_id", reasons[0])

    def test_the_reason_names_its_arm(self) -> None:
        reasons = clock_reasons(_record([1781, 2100, 2398]), label="vllm")
        self.assertIn("vllm", reasons[0])


class CrossArmTests(unittest.TestCase):
    def test_same_boot_matched_clocks_compare(self) -> None:
        comparison = compare_clock_records(
            _record([2470, 2470, 2470]), _record([2470, 2470, 2470])
        )
        self.assertEqual(comparison["reasons"], [])
        self.assertTrue(comparison["same_boot"])
        self.assertAlmostEqual(comparison["median_offset_pct"], 0.0)
        self.assertAlmostEqual(comparison["estimated_effect_pct"], 0.0)

    def test_a_cross_boot_pair_is_refused(self) -> None:
        comparison = compare_clock_records(
            _record([2470] * 3, boot_id=BOOT_GOOD),
            _record([2470] * 3, boot_id=BOOT_BAD),
        )
        self.assertFalse(comparison["same_boot"])
        self.assertFalse(comparison["cross_boot_override"])
        self.assertTrue(any("boot" in reason for reason in comparison["reasons"]))
        self.assertIn(BOOT_GOOD, comparison["ours_boot_id"])
        self.assertIn(BOOT_BAD, comparison["vllm_boot_id"])

    def test_the_override_records_a_caveat_rather_than_silence(self) -> None:
        comparison = compare_clock_records(
            _record([2470] * 3, boot_id=BOOT_GOOD),
            _record([2470] * 3, boot_id=BOOT_BAD),
            allow_cross_boot=True,
        )
        self.assertTrue(comparison["cross_boot_override"])
        self.assertFalse(comparison["same_boot"])
        self.assertEqual(comparison["reasons"], [])
        self.assertTrue(any("boot" in note for note in comparison["caveats"]))

    def test_the_override_does_not_waive_state_only_identity(self) -> None:
        comparison = compare_clock_records(
            _record([2470] * 3, boot_id=BOOT_GOOD),
            _record([2190] * 3, boot_id=BOOT_BAD),
            allow_cross_boot=True,
        )
        self.assertTrue(comparison["cross_boot_override"])
        self.assertTrue(any("offset" in reason for reason in comparison["reasons"]))

    def test_the_override_waives_the_boot_and_not_the_hardware(self) -> None:
        """A GB10 against an H100 compared CLEAN under the override.

        Same-boot equality was the implicit proxy for "same machine". #545 makes
        the override the normal path, so the proxy is gone and the identity has
        to be asserted in its own right -- unconditionally, because no override
        can make two different GPUs two arms of one comparison.
        """

        ours = _record([2470] * 3, boot_id=BOOT_GOOD)
        theirs = _record([2470] * 3, boot_id=BOOT_BAD)
        theirs["gpu_name"] = "NVIDIA H100 80GB HBM3"
        theirs["driver_version"] = "550.54.15"
        theirs["clocks_max_sm_mhz"] = 1980
        comparison = compare_clock_records(ours, theirs, allow_cross_boot=True)
        self.assertTrue(comparison["cross_boot_override"])
        for field in ("gpu_name", "driver_version", "clocks_max_sm_mhz"):
            with self.subTest(field=field):
                self.assertTrue(
                    any(field in reason for reason in comparison["reasons"]),
                    comparison["reasons"],
                )

    def test_every_static_field_is_compared_across_the_arms(self) -> None:
        for field, other in (
            ("gpu_name", "NVIDIA H100 80GB HBM3"),
            ("driver_version", "550.54.15"),
            ("clocks_max_sm_mhz", 1980),
            ("clocks_applications_graphics_mhz", 1980),
            ("persistence_mode", "Disabled"),
        ):
            for override in (False, True):
                with self.subTest(field=field, allow_cross_boot=override):
                    theirs = _record([2470] * 3)
                    theirs[field] = other
                    comparison = compare_clock_records(
                        _record([2470] * 3), theirs, allow_cross_boot=override
                    )
                    self.assertTrue(
                        any(
                            field in reason and "not two arms" in reason
                            for reason in comparison["reasons"]
                        ),
                        comparison["reasons"],
                    )

    def test_matched_hardware_adds_no_reason(self) -> None:
        comparison = compare_clock_records(_record([2470] * 3), _record([2470] * 3))
        self.assertEqual(comparison["reasons"], [])

    def test_the_observed_window_is_surfaced_next_to_the_ratio(self) -> None:
        ours = build_clock_record(
            [*_samples([2470] * 40), *_samples([300] * 5, utilization=0)],
            boot_id=BOOT_GOOD,
        )
        comparison = compare_clock_records(ours, _record([2470] * 3))
        self.assertEqual(comparison["ours_busy_samples"], 40)
        self.assertEqual(comparison["ours_idle_samples_excluded"], 5)
        self.assertEqual(comparison["vllm_busy_samples"], len(_window([2470] * 3)))
        self.assertEqual(comparison["vllm_idle_samples_excluded"], 0)

    def test_the_measured_pair_is_refused_and_its_effect_reported(self) -> None:
        # 2470 against 2190: the exact pair behind #543.
        comparison = compare_clock_records(
            _record([2470] * 3, boot_id=BOOT_GOOD),
            _record([2190] * 3, boot_id=BOOT_BAD),
        )
        self.assertAlmostEqual(comparison["median_offset_pct"], 12.7854, places=3)
        self.assertAlmostEqual(
            comparison["estimated_effect_pct"], 12.7854 * CLOCK_TIME_TRANSFER, places=3
        )
        # The marlin control actually moved +9.65%; the estimate must land on it.
        self.assertAlmostEqual(comparison["estimated_effect_pct"], 9.65, places=1)
        self.assertGreaterEqual(len(comparison["reasons"]), 2)

    def test_the_offset_threshold_is_inclusive(self) -> None:
        inside = compare_clock_records(_record([2020] * 3), _record([2000] * 3))
        self.assertAlmostEqual(inside["median_offset_pct"], 1.0)
        self.assertEqual(inside["reasons"], [])
        outside = compare_clock_records(_record([2021] * 3), _record([2000] * 3))
        self.assertGreater(abs(outside["median_offset_pct"]), MAX_CROSS_ARM_OFFSET_PCT)
        self.assertTrue(any("offset" in reason for reason in outside["reasons"]))

    def test_the_offset_is_signed_and_the_gate_is_not(self) -> None:
        slower = compare_clock_records(_record([1900] * 3), _record([2000] * 3))
        self.assertLess(slower["median_offset_pct"], 0.0)
        self.assertTrue(any("offset" in reason for reason in slower["reasons"]))

    def test_a_within_run_defect_on_either_arm_propagates(self) -> None:
        comparison = compare_clock_records(
            _record([1781, 2100, 2398], boot_id=BOOT_GOOD),
            _record([2100] * 3, boot_id=BOOT_GOOD),
        )
        self.assertTrue(any("spread" in reason for reason in comparison["reasons"]))

    def test_both_arms_spread_pct_are_surfaced_next_to_the_ratio(self) -> None:
        comparison = compare_clock_records(
            _record([2398, 2470, 2489]), _record([2470] * 3)
        )
        self.assertAlmostEqual(comparison["ours_spread_pct"], 3.6842, places=3)
        self.assertAlmostEqual(comparison["vllm_spread_pct"], 0.0)
        self.assertEqual(comparison["ours_median_sm_mhz"], 2470.0)
        self.assertEqual(comparison["vllm_median_sm_mhz"], 2470.0)


class MergeTests(unittest.TestCase):
    def test_three_steady_legs_fold_into_one_arm(self) -> None:
        merged = merge_clock_records([_record([2470] * 3) for _ in range(3)])
        self.assertEqual(merged["legs"], 3)
        self.assertEqual(merged["sm_clock_mhz"]["n"], 3 * len(_window([2470] * 3)))
        self.assertEqual(merged["sm_clock_mhz"]["median"], 2470.0)
        self.assertAlmostEqual(merged["sm_clock_mhz"]["spread_pct"], 0.0)
        self.assertEqual(clock_reasons(merged, label="ours"), [])

    def test_an_arm_may_not_straddle_two_boots(self) -> None:
        with self.assertRaises(HarnessError) as caught:
            merge_clock_records(
                [_record([2470] * 3, boot_id=BOOT_GOOD), _record([2470] * 3, boot_id=BOOT_BAD)]
            )
        self.assertIn("straddles two boots", str(caught.exception))

    def test_legs_at_different_clocks_widen_the_merged_spread(self) -> None:
        # Each leg is individually steady; the ARM is not. The within-run rule
        # has to catch that, which is why the fold spans every leg.
        merged = merge_clock_records([_record([2470] * 3), _record([2190] * 3)])
        self.assertGreater(merged["sm_clock_mhz"]["spread_pct"], MAX_WITHIN_RUN_SPREAD_PCT)
        self.assertTrue(clock_reasons(merged, label="ours"))

    def test_merging_nothing_refuses(self) -> None:
        with self.assertRaises(HarnessError):
            merge_clock_records([])

    def test_a_defective_leg_refuses_the_fold(self) -> None:
        bad = _record([2470])
        bad.pop("persistence_mode")
        with self.assertRaises(HarnessError) as caught:
            merge_clock_records([_record([2470]), bad])
        self.assertIn("leg 2", str(caught.exception))

    def test_idle_exclusions_are_summed_not_lost(self) -> None:
        leg = build_clock_record(
            [*_samples([2470, 2470]), *_samples([300], utilization=0)], boot_id=BOOT_GOOD
        )
        merged = merge_clock_records([leg, leg])
        self.assertEqual(merged["idle_samples_excluded"], 2)


class SamplerTests(unittest.TestCase):
    """The sampler is what the leg harness actually calls, so it is exercised.

    A stub stands in for `nvidia-smi`: the point is that a window of probes
    becomes the two artifacts `online_gate_summary` reads, not that the driver
    works.
    """

    def _stub(self, directory: pathlib.Path, body: str) -> pathlib.Path:
        stub = directory / "nvidia-smi-stub"
        stub.write_text(body, encoding="utf-8")
        stub.chmod(0o755)
        return stub

    def test_a_window_becomes_a_stream_and_a_record(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = pathlib.Path(raw)
            stub = self._stub(
                directory,
                "#!/bin/sh\necho '0, NVIDIA GB10, 580.159.03, 2470, 3003, 2418, "
                "0x0000000000000000, Enabled, 97'\n",
            )
            boot = directory / "boot_id"
            boot.write_text(BOOT_GOOD + "\n", encoding="utf-8")
            # Long enough to clear MIN_BUSY_SAMPLES with ~20x margin: the stub
            # loop runs ~590 probes in this window on an idle box and would have
            # to be 20 times slower to fall under the floor. The count is
            # asserted directly below so a shortfall reads as a shortfall rather
            # than as an unexplained clock reason.
            record = run_sampler(
                samples_output=directory / "r1.samples.jsonl",
                summary_output=directory / "r1.summary.json",
                interval_s=0.001,
                max_duration_s=1.0,
                smi=str(stub),
                boot_id_path=boot,
            )
            stream = [
                json.loads(line)
                for line in (directory / "r1.samples.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(record["sm_clock_mhz"]["n"], MIN_BUSY_SAMPLES)
            self.assertGreaterEqual(len(stream), 1)
            self.assertEqual(record["boot_id"], BOOT_GOOD)
            self.assertEqual(record["sm_clock_mhz"]["median"], 2470.0)
            self.assertEqual(
                record["sm_clock_mhz"]["n"] + record["idle_samples_excluded"],
                len(stream),
            )
            on_disk = json.loads((directory / "r1.summary.json").read_text())
            self.assertEqual(on_disk, record)
            self.assertEqual(clock_reasons(on_disk, label="ours"), [])

    def test_a_failed_probe_refuses_rather_than_recording_a_guess(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = pathlib.Path(raw)
            stub = self._stub(directory, "#!/bin/sh\nexit 3\n")
            boot = directory / "boot_id"
            boot.write_text(BOOT_GOOD + "\n", encoding="utf-8")
            with self.assertRaises(HarnessError) as caught:
                run_sampler(
                    samples_output=directory / "r1.samples.jsonl",
                    summary_output=directory / "r1.summary.json",
                    interval_s=0.01,
                    max_duration_s=0.05,
                    smi=str(stub),
                    boot_id_path=boot,
                )
            self.assertIn("exited 3", str(caught.exception))
            self.assertFalse((directory / "r1.summary.json").exists())

    def test_existing_evidence_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = pathlib.Path(raw)
            stub = self._stub(directory, "#!/bin/sh\nexit 0\n")
            (directory / "r1.summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(HarnessError) as caught:
                run_sampler(
                    samples_output=directory / "r1.samples.jsonl",
                    summary_output=directory / "r1.summary.json",
                    smi=str(stub),
                )
            self.assertIn("refusing to overwrite", str(caught.exception))


class ThresholdProvenanceTests(unittest.TestCase):
    """The thresholds are arguments from data, so the data is asserted here."""

    def test_the_spread_threshold_accepts_the_clean_window_and_rejects_the_disagreement(
        self,
    ) -> None:
        clean = summarize_sm_clocks([2398, 2470, 2489])["spread_pct"]
        disagreement = summarize_sm_clocks([1781, 2398])["spread_pct"]
        self.assertLess(clean, MAX_WITHIN_RUN_SPREAD_PCT)
        self.assertGreater(disagreement, MAX_WITHIN_RUN_SPREAD_PCT)

    def test_the_offset_threshold_stays_under_the_smallest_ranked_deficit(self) -> None:
        # in_proj +2.97% is the smallest deficit this harness has been used to
        # rank. A pair inside the offset threshold may not be able to explain it.
        self.assertLess(MAX_CROSS_ARM_OFFSET_PCT * CLOCK_TIME_TRANSFER, 2.97)

    def test_the_offset_threshold_holds_on_physics_not_on_n_equals_one(self) -> None:
        """The threshold must not DEPEND on the n = 1 coefficient.

        For a kernel whose time scales with clock the transfer is bounded above
        by 1.0 -- a 12.79% clock deficit can cost at most 12.79% of time. So the
        1.0% offset implies at most a 1.0% effect with no appeal to any
        measurement, and still lands under the 2.97% smallest ranked deficit.
        The measured coefficient is corroboration: it sits BELOW that ceiling,
        exactly as a partly memory-bound kernel should.
        """

        self.assertLess(MAX_CROSS_ARM_OFFSET_PCT * 1.0, 2.97)
        self.assertLess(CLOCK_TIME_TRANSFER, 1.0)

    def test_the_spread_ceiling_is_not_held_to_the_offsets_criterion(self) -> None:
        """Stated, not hidden: at the ceiling a leg can carry a 3.77% artifact.

        The forward criterion the offset was chosen by would demand <=3.93% of
        spread, which sits 0.25 points above the only clean window we have
        (3.68%) and would void it on a noisier-but-healthy day. The residual is
        recorded here so nobody re-derives it as a defect: passing spread
        establishes that an arm was ONE state, not that a sub-4% deficit is
        established. The offset rule is what qualifies the ratio.
        """

        self.assertGreater(MAX_WITHIN_RUN_SPREAD_PCT * CLOCK_TIME_TRANSFER, 2.97)
        self.assertLess(summarize_sm_clocks([2398, 2470, 2489])["spread_pct"], 2.97 / CLOCK_TIME_TRANSFER)

    def test_the_transfer_coefficient_is_the_measured_one(self) -> None:
        measured = (49.6544 / 45.2845 - 1.0) / (2470.0 / 2190.0 - 1.0)
        self.assertAlmostEqual(CLOCK_TIME_TRANSFER, measured, places=3)

    def test_the_busy_floors_accept_the_two_real_windows(self) -> None:
        """Both floors are bounded on the accepting side by real captures.

        #543's two windows are n=61 and n=50 busy samples; a floor above either
        would void the only real data the row is built from, which is the same
        both-sides bound the spread threshold was chosen under.
        """

        for observed in (61, 50):
            with self.subTest(n=observed):
                self.assertLessEqual(MIN_BUSY_SAMPLES, observed)
        # And it is far above the degenerate window that scores a perfect 0.00%.
        self.assertGreaterEqual(MIN_BUSY_SAMPLES, 30 * 1)
        self.assertGreater(MIN_BUSY_FRACTION, 0.0)
        self.assertLessEqual(MIN_BUSY_FRACTION, 1.0)


if __name__ == "__main__":
    unittest.main()

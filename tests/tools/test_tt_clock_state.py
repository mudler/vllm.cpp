"""Tests for tools/bench/tt_clock_state.py (#2005).

Synthetic fixtures, no device required. Every gate constant is mutated red
by the mutation driver: change a threshold here and its paired case must
fail. The busy proxy's /proc walk runs against THIS process, not a mock.
"""

import json
import os
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools" / "bench"))

import tt_clock_state as ttc  # noqa: E402


def mk_sample(aiclk=800, busy=True):
    return {
        "t": 0.0,
        "aiclk": aiclk,
        "busy": busy,
        "arcclk": 800,
        "axiclk": 960,
        "vcore": 726,
        "tdp": 19,
        "tdc": 27,
        "asic_temp_raw": "0x3474da",
        "fan_rpm": 2105,
        "board_id": "0x403:0x3191406b",
        "flash_bundle": "6.7.1",
        "kmd_driver": "TT-KMD 2.10.1-pre",
        "tt_smi_version": "6.2.1",
        "umd_version": "0.9.9",
    }


def mk_window(aiclks, busy=None, **identity_over):
    samples = [
        mk_sample(clk, True if busy is None else busy[i])
        for i, clk in enumerate(aiclks)
    ]
    rec = ttc.fold(samples)
    base = mk_sample()
    for key in ("board_id", "flash_bundle", "kmd_driver", "tt_smi_version",
                "umd_version"):
        rec[key] = identity_over.pop(key, base[key])
    assert not identity_over
    return rec


class FoldMath(unittest.TestCase):
    def test_fold_fields_match_the_record_shape(self):
        w = mk_window([100, 200, 300, 400])
        self.assertEqual(w["n"], 4)
        self.assertEqual((w["min"], w["max"]), (100, 400))
        self.assertEqual(w["median"], 250.0)
        self.assertAlmostEqual(w["spread_pct"], 300 / 250 * 100)
        self.assertEqual((w["busy_n"], round(w["busy_fraction"], 9)), (4, 1.0))

    def test_empty_window_folds_to_refusable_record(self):
        empty = ttc.fold([])
        self.assertIsNone(empty["median"])
        rs = ttc.clock_state_reasons(empty)
        self.assertEqual(len(rs), 1)
        self.assertIn("idle-or-empty", rs[0])

    def test_not_applicable_fields_are_stated_never_dropped(self):
        rec = ttc.fold([mk_sample()])
        na = rec["not_applicable"]
        for key in ("applications_clocks_setting", "persistence_mode",
                    "throttle_reasons_live"):
            self.assertIn(key, na)
            self.assertTrue(str(na[key]))


class WithinRunRules(unittest.TestCase):
    def test_spread_boundary_is_five_pct(self):
        self.assertEqual(ttc.MAX_WITHIN_RUN_SPREAD_PCT, 5.0)
        ok = mk_window([800] * 35 + [820, 780])  # exactly 5.00%
        self.assertNotIn(
            "spread", "".join(ttc.clock_state_reasons(ok))
        )
        bad = mk_window([800] * 35 + [821, 779])  # 5.25%
        spread_rs = [r for r in ttc.clock_state_reasons(bad) if "spread" in r]
        self.assertEqual(len(spread_rs), 1)

    def test_busy_floor_thirty_retained(self):
        self.assertEqual(ttc.MIN_BUSY_SAMPLES, 30)
        thin = mk_window([800] * 29)
        rs = ttc.clock_state_reasons(thin)
        self.assertTrue(all("MIN_BUSY_SAMPLES" in r or r for r in rs))
        self.assertTrue(any("MIN_BUSY_SAMPLES" in r for r in rs))
        good = mk_window([800] * 70, busy=[True] * 45 + [False] * 25)
        self.assertEqual(ttc.clock_state_reasons(good), [])

    def test_busy_fraction_fifty_pct_independent_of_count_floor(self):
        self.assertEqual(ttc.MIN_BUSY_FRACTION, 0.5)
        diluted = mk_window(
            [800] * 150, busy=[True] * 30 + [False] * 120
        )
        rs = ttc.clock_state_reasons(diluted)
        self.assertFalse(any("MIN_BUSY_SAMPLES" in r for r in rs))  # 30 >= 30
        self.assertTrue(any("busy fraction" in r for r in rs))


class CrossArmRules(unittest.TestCase):
    def test_median_rule_boundary_one_pct(self):
        self.assertEqual(ttc.MAX_CROSS_ARM_OFFSET_PCT, 1.0)
        wa = mk_window([800.0] * 64)
        inside = mk_window([807.0] * 64)  # 0.875%
        self.assertEqual(ttc.clock_state_reasons(inside, wa), [])
        outside = mk_window([810.0] * 64)  # 1.25%
        rs = ttc.clock_state_reasons(outside, wa)
        self.assertTrue(any("median offset" in r for r in rs))

    def test_mean_rule_is_independent_of_median_rule(self):
        # Median pinned, mean dragged: 31 samples at -3%, 33 AT the median.
        # Sorted positions 32/33 land on 800 so BOTH medians are 800; the
        # skew pulls arm B's mean off by ~1.47% (>1%) while arm B alone
        # stays inside the 5% within-run spread band.
        wa = mk_window([800.0] * 64)
        wb = mk_window([776.0] * 31 + [800.0] * 33)
        self.assertEqual(wb["median"], wa["median"])
        rs = ttc.clock_state_reasons(wb, wa)
        self.assertTrue(any("mean offset" in r for r in rs), rs)
        self.assertFalse(any("median offset" in r for r in rs), rs)
        self.assertFalse(any("within-run spread" in r for r in rs), rs)

    def test_boot_mismatch_refuses_and_waiver_requires_caveat(self):
        wa = mk_window([800.0] * 64)
        wb = mk_window([800.0] * 64)
        wb["boot_id"] = "00000000-0000-0000-0000-000000000001"
        rs = ttc.clock_state_reasons(wb, wa)
        self.assertTrue(any("different boot ids" in r for r in rs))
        waived = dict(wb)
        waived["_allow_cross_boot"] = True
        waived["_waiver_caveat"] = "stamped"
        self.assertEqual(ttc.clock_state_reasons(waived, wa), [])
        bare = {k: v for k, v in wb.items()}
        bare["_allow_cross_boot"] = True  # waiver without caveat
        rs = ttc.clock_state_reasons(bare, wa)
        self.assertTrue(any("WITHOUT the stamped waiver caveat" in r for r in rs))

    def test_machine_identity_refuses_unconditionally_under_waiver(self):
        wa = mk_window([800.0] * 64)
        for key, label in (
            ("board_id", "board id"),
            ("flash_bundle", "firmware bundle"),
            ("kmd_driver", "KMD/driver"),
            ("tt_smi_version", "tt-smi version"),
            ("umd_version", "UMD version"),
        ):
            wb = mk_window([800.0] * 64, **{key: "OTHER"})
            wb["_allow_cross_boot"] = True
            wb["_waiver_caveat"] = "stamped"
            wb["boot_id"] = wa["boot_id"]
            rs = ttc.clock_state_reasons(wb, wa)
            self.assertTrue(
                any(f"unconditional mismatch {label}" in r for r in rs),
                (key, rs),
            )
            self.assertFalse(any("different boot ids" in r for r in rs))


class JudgeContract(unittest.TestCase):
    def test_clean_pair_passes_and_carries_the_throttle_caveat(self):
        res = ttc.judge([mk_window([800.0] * 64), mk_window([800.0] * 64)])
        self.assertEqual(res["reasons"], [])
        for win in res["windows"]:
            self.assertIn("throttle_unobservability_caveat", win)
            self.assertTrue(win["not_applicable"]["persistence_mode"].startswith("no TT"))

    def test_per_window_and_pairwise_reasons_both_surface(self):
        good = mk_window([800.0] * 64)
        thin = mk_window([800.0] * 20)
        res = ttc.judge([good, thin])
        self.assertTrue(any("MIN_BUSY_SAMPLES" in r for r in res["reasons"]))
        drifted = mk_window([900.0] * 64)
        res2 = ttc.judge([good, drifted])
        self.assertTrue(any("median offset" in r for r in res2["reasons"]))


class BusyProxy(unittest.TestCase):
    def test_this_process_holds_no_tt_device(self):
        self.assertIs(ttc.pid_holds_tt_device(os.getpid()), False)

    def test_dead_pid_is_false_without_crash(self):
        self.assertFalse(ttc.pid_holds_tt_device(2_097_151))


class SummaryRoundtrip(unittest.TestCase):
    def test_waiver_terms_survive_json(self):
        rec = mk_window([800.0] * 64)
        rec["_allow_cross_boot"] = True
        rec["_waiver_caveat"] = "stamped"
        blob = json.loads(json.dumps(rec))
        self.assertTrue(blob["_allow_cross_boot"])
        self.assertEqual(blob["_waiver_caveat"], "stamped")

    def test_cli_judge_exit_semantics(self):
        here = pathlib.Path(__file__).resolve().parent
        tmp = pathlib.Path(os.environ.get("TT_TEST_TMP", "/tmp"))
        pa, pb = tmp / "j_a.json", tmp / "j_b.json"
        for name, clks in ((pa, [800.0] * 64), (pb, [800.0] * 64)):
            pathlib.Path(name).write_text(json.dumps(mk_window(clks)))
        rc_ok = ttc.main(["judge", str(pa), str(pb)])
        self.assertEqual(rc_ok, 0)
        pathlib.Path(pb).write_text(json.dumps(mk_window([900.0] * 64)))
        rc_bad = ttc.main(["judge", str(pa), str(pb)])
        self.assertEqual(rc_bad, 1)


if __name__ == "__main__":
    unittest.main()

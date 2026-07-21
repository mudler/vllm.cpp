import json
import tempfile
import unittest
from pathlib import Path

from tools.bench.vllm_closed_loop_metrics import load_prompt_text, percentile, summarize


class VllmClosedLoopMetricsTest(unittest.TestCase):
    def test_numpy_style_percentile_and_summary(self):
        self.assertAlmostEqual(percentile([0.0, 10.0], 99), 9.9)
        result = summarize([1.0, 2.0, 3.0])
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["median"], 2.0)
        self.assertAlmostEqual(result["p99"], 2.98)

    def test_sharegpt_first_turn_filter_and_exact_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sharegpt.json"
            path.write_text(
                json.dumps(
                    [
                        {"bad": "row"},
                        {
                            "conversations": [
                                {"from": "human", "value": "first"}
                            ]
                        },
                        {
                            "conversations": [
                                {"from": "human", "value": "second"}
                            ]
                        },
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_prompt_text(path, 2), ["first", "second"])
            with self.assertRaisesRegex(ValueError, "need 3"):
                load_prompt_text(path, 3)


if __name__ == "__main__":
    unittest.main()

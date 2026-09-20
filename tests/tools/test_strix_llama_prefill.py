"""Exercise the production #3075 CLI with a stateful public llama API double."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ["The capital city of France is", "The three primary colors are",
           "Water boils at a temperature of", "The Pythagorean theorem states that",
           "In 1969, humans first walked on", "A prime number is a natural number"]
IDS = [[100*s+i+1 for i in range(n)] for s, n in enumerate([6, 5, 6, 7, 8, 9])]


class PrefillProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="llama-prefill-test-")
        cls.directory = Path(cls.temp.name)
        cls.binary = cls.directory / "probe"
        built = subprocess.run(["c++", "-std=c++20", "-O0", "-I", str(ROOT / "tests/tools/llama_prefill_fixture"),
                                "-I", str(ROOT / "third_party"), str(ROOT / "tools/bench/strix_four_engine/llama_prefill_probe.cpp"),
                                "-o", str(cls.binary)], capture_output=True, text=True)
        if built.returncode:
            raise RuntimeError(built.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_probe(self, config=None, fault="", raw=None, full=False):
        config = config if config is not None else dict(schema=1, gguf="fixture.gguf", prompts=PROMPTS, prompt_ids=IDS)
        path = self.directory / "config.json"
        path.write_text(json.dumps(config) if raw is None else raw)
        events = self.directory / "events.jsonl"
        events.write_text("")
        env = dict(os.environ, PROBE_EVENTS=str(events), PROBE_FAULT=fault)
        if full:
            with open("/dev/full", "w") as output:
                run = subprocess.run([str(self.binary), "--config", str(path)], env=env, stdout=output, stderr=subprocess.PIPE, text=True, timeout=30)
            records = []
        else:
            run = subprocess.run([str(self.binary), "--config", str(path)], env=env, capture_output=True, text=True, timeout=30)
            records = [json.loads(line) for line in run.stdout.splitlines()]
        return run, records, [json.loads(line) for line in events.read_text().splitlines()]

    def test_captures_all_modes_and_full_rows(self):
        run, records, events = self.run_probe()
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(records), 38)
        for key, value in dict(schema=1, vocab_size=151936, repetitions=3).items():
            self.assertIn(key, records[0])
            self.assertIs(type(records[0][key]), int)
            self.assertEqual(records[0][key], value)
        self.assertEqual(records[0]["type"], "header")
        self.assertEqual(records[0]["score_semantics"], "raw_logits")
        self.assertEqual(records[0]["requested"], dict(n_gpu_layers=-1, n_ctx=8192, n_seq_max=4,
                                                     kv_unified=False, type_k="bfloat16", type_v="bfloat16",
                                                     flash_attn="auto", n_batch=2048, n_ubatch=512,
                                                     n_threads=4, n_threads_batch=4))
        self.assertEqual(records[0]["resolved"], dict(n_ctx=8192, n_ctx_seq=2048, n_seq_max=4,
                                                    n_batch=2048, n_ubatch=512, n_threads=4, n_threads_batch=4))
        self.assertEqual(records[-1], dict(schema=1, type="diagnostic_complete", score_records=36))
        self.assertEqual(len([e for e in events if e["op"] == "context_new"]), 18)
        self.assertEqual(len([e for e in events if e["op"] == "decode"]), 27)
        self.assertEqual(len([e for e in events if e["op"] == "get"]), 36)
        self.assertEqual([e["sequence"] for e in events if e["op"] == "tokenize"], list(range(6)))
        self.assertEqual([e["layers"] for e in events if e["op"] == "model_load"], [-1])
        for e in events:
            if e["op"] == "context_new":
                self.assertEqual({k:e[k] for k in ["ctx", "slots", "unified", "k", "v"]},
                                 dict(ctx=8192, slots=4, unified=False, k=1, v=1))
        original = []
        for s in range(4):
            for p, token in enumerate(IDS[s]):
                original.append(dict(original_index=len(original), token_id=token, position=p,
                                     sequence_id=s, logits_requested=p == len(IDS[s])-1))
        partitions = [[0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21], [5], [16,22], [23]]
        decode_events = iter(e for e in events if e["op"] == "decode")
        for mode in ["independent", "combined", "partitioned"]:
            for rep in range(3):
                groups = ([list(range(0,6)), list(range(6,11)), list(range(11,17)), list(range(17,24))]
                          if mode == "independent" else [list(range(24))] if mode == "combined" else partitions)
                for group in groups:
                    actual = next(decode_events)["rows"]
                    self.assertEqual(actual, [{k:v for k,v in original[i].items() if k != "original_index"} for i in group])
        # Explicit synchronization must follow every decode before another API operation.
        for i, e in enumerate(events):
            if e["op"] == "decode":
                self.assertEqual(events[i+1], dict(op="sync", context=e["context"]))
        counts = {}
        for r in records[1:-1]:
            self.assertIs(type(r["schema"]), int)
            self.assertEqual(r["schema"], 1)
            self.assertEqual(r["type"], "scores")
            for key in ["sequence_id", "repetition", "original_terminal_index", "batch_terminal_index", "argmax"]:
                self.assertIs(type(r[key]), int)
            s = r["sequence_id"]
            key = r["mode"], r["repetition"], s
            self.assertNotIn(key, counts)
            counts[key] = True
            self.assertEqual(r["prompt_ids"], IDS[s])
            self.assertEqual(r["original_terminal_index"], [5, 10, 16, 23][s])
            if r["mode"] == "independent":
                batches = [[row for row in original if row["sequence_id"] == s]]
                getter = len(IDS[s])-1
            elif r["mode"] == "combined":
                batches = [original]
                getter = [5,10,16,23][s]
            else:
                batches = [[original[i] for i in group] for group in partitions[:[2,1,3,4][s]]]
                getter = [0,9,0,0][s]
            self.assertEqual(r["submitted_batches"], batches)
            self.assertEqual(r["batch_terminal_index"], getter)
            self.assertEqual(r["argmax"], 42)
            self.assertEqual([x["token_id"] for x in r["top10"]], [42, 43, 0, 1, 2, 3, 4, 5, 6, 7])
            expected = [(-t/8+s) for t in range(151936)]
            expected[42] = expected[43] = 100+s
            self.assertEqual(r["logits"], expected)
            for entry in r["top10"]:
                self.assertIs(type(entry["token_id"]), int)
                self.assertIn(type(entry["logit"]), (int, float))
                self.assertEqual(entry["logit"], r["logits"][entry["token_id"]])
        self.assertEqual(set(k[0] for k in counts), {"independent", "combined", "partitioned"})
        for rep in range(3):
            self.assertEqual([r["sequence_id"] for r in records[1:-1] if r["mode"] == "partitioned" and r["repetition"] == rep], [1,0,2,3])
        ops = [e["op"] for e in events]
        self.assertEqual(ops.count("model_load"), 1)
        self.assertEqual(ops.count("model_free"), 1)
        self.assertEqual(ops.count("context_free"), 18)
        self.assertEqual(ops.count("batch_new"), ops.count("batch_free"))

    def test_refuses_bad_config(self):
        base = dict(schema=1, gguf="fixture.gguf", prompts=PROMPTS, prompt_ids=IDS)
        bad = [(dict(base, schema=True), "invalid schema"), (dict(base, schema=1.0), "invalid schema"),
               (dict(base, schema=2), "invalid schema"), (dict(base, unknown=1), "config keys mismatch"),
               ({k:v for k,v in base.items() if k != "gguf"}, "config keys mismatch"),
               (dict(base, gguf=""), "invalid GGUF path"), (dict(base, gguf=2), "invalid GGUF path"),
               (dict(base, prompts=PROMPTS[::-1]), "six canonical raw prompts required"),
               (dict(base, prompt_ids=IDS[:5]), "six token arrays required"),
               (dict(base, prompt_ids=[[]]+IDS[1:]), "invalid prompt IDs"),
               (dict(base, prompt_ids=[[1]*2049]+IDS[1:]), "invalid prompt IDs"),
               (dict(base, prompt_ids=[[True]]+IDS[1:]), "token ID out of range"),
               (dict(base, prompt_ids=[[1.5]]+IDS[1:]), "token ID out of range"),
               (dict(base, prompt_ids=[[-1]]+IDS[1:]), "token ID out of range"),
               (dict(base, prompt_ids=[[151936]]+IDS[1:]), "token ID out of range"),
               (dict(base, prompt_ids=[[1, 2]]+IDS[1:]), "experiment lengths changed")]
        for config, message in bad:
            with self.subTest(config=config):
                run, records, _ = self.run_probe(config)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(message, run.stderr)
                self.assertFalse(any(r["type"] == "diagnostic_complete" for r in records))
        run, _, _ = self.run_probe(raw=" " * (1048576+1))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("config exceeds 1 MiB", run.stderr)
        run, _, _ = self.run_probe(raw='{"schema":1,"schema":1}')
        self.assertIn("duplicate config key", run.stderr)

    def test_refuses_missing_gpu_before_model_load(self):
        run, records, events = self.run_probe(fault="gpu_unavailable")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("GPU offload unavailable", run.stderr)
        self.assertEqual(records, [])
        self.assertEqual(events, [dict(op="backend_init"), dict(op="backend_free")])

    def test_runtime_errors_cleanup(self):
        failures = dict(model="model allocation failed", context="context allocation failed", batch="batch allocation failed",
                        vocab="vocabulary mismatch", defaults="public context defaults changed", resolved="resolved context mismatch",
                        tokenize="tokenization mismatch", decode="decode failed", null="missing logits", nonfinite="nonfinite logits")
        failures.update({k:"public context defaults changed" for k in ["ubatch", "threads", "threads_batch", "offload", "op_offload", "flash"]})
        # Change only a public getter result, not requested params or their defaults.
        failures.update({k:"resolved context mismatch" for k in ["resolved_ctx", "resolved_slots", "resolved_batch",
                        "resolved_ubatch", "resolved_threads", "resolved_threads_batch"]})
        for fault, message in failures.items():
            with self.subTest(fault=fault):
                run, records, events = self.run_probe(fault=fault)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(message, run.stderr)
                self.assertFalse(any(r["type"] == "diagnostic_complete" for r in records))
                ops = [e["op"] for e in events]
                self.assertEqual(ops.count("backend_init"), ops.count("backend_free"))
                self.assertEqual(ops.count("model_load")-(fault == "model"), ops.count("model_free"))
                self.assertEqual(ops.count("context_new")-(fault == "context"), ops.count("context_free"))
                self.assertEqual(ops.count("batch_new"), ops.count("batch_free"))

    def test_output_failure_cleanup(self):
        run, _, events = self.run_probe(full=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("output failure", run.stderr)
        ops = [e["op"] for e in events]
        self.assertEqual(ops.count("model_load"), ops.count("model_free"))
        self.assertEqual(ops.count("context_new"), ops.count("context_free"))

    def test_cli_refuses_missing_input(self):
        for args in [[], ["--wrong", "config"], ["--config", str(self.directory / "absent")]]:
            with self.subTest(args=args):
                run = subprocess.run([str(self.binary), *args], capture_output=True, text=True, timeout=10)
                self.assertNotEqual(run.returncode, 0)
                self.assertEqual(run.stdout, "")
                self.assertIn("llama-prefill-probe:", run.stderr)


if __name__ == "__main__":
    unittest.main()

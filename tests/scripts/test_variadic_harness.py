#!/usr/bin/env python3
"""Controls for the variadic-load harness, against a mock OpenAI stream.

`.agents/specs/bench-qwen38-exl3-variadic.md`, #2970. The harness measures two
engines on a leased GPU. Every property it claims is a property of the CLIENT
and the REPORT, not of the GPU, so all of them are checkable here with no
lease, no toolchain, no weights and no network beyond a loopback socket.

The mock server streams `tokens_per_chunk` tokens in each SSE chunk and reports
its own `usage`, which is what lets these tests separate a token count from a
chunk count. That separation is the defect the harness exists to avoid: one
engine here streams 4.46 tokens per chunk and the other 1.08, so a client that
counted chunks would compare two different quantities and call it a result.

What each case pins:

  PercentileContract      linear interpolation, matching numpy's default and
                          `vllm/benchmarks/serve.py:739`
  ClosedLoopConcurrency   c workers really overlap, and c=1 really does not
  WarmupIsExcluded        the measured window contains no warmup request, and
                          the with-warmup view still sees them
  ChunkingIsRecovered     tokens per chunk is read back from usage, and the
                          TTFT correction moves in the right direction
  UsageIsMandatory        a server that reports no usage is marked `chunks`,
                          never silently counted
  AcceptanceFromMetrics   the /metrics delta becomes an acceptance rate, and an
                          engine with no /metrics reports absent rather than 0
  CorpusIsDeterministic   the corpus is a function of (sources, seed, weights)
  LongBandIsRequestable   `--weights` reaches the XXL band, the band's target
                          fits the context the row serves, and a malformed or
                          non-finite spelling is refused rather than silently
                          ignored
  BinaryCacheRestores     job.sh restores a cached vllm-server from a share that
                          carries no execute bit, instead of rebuilding it
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "benchmarks" / "variadic"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


client = _load("variadic_client", HARNESS / "client.py")
report = _load("variadic_report", HARNESS / "report.py")


class MockHandler(BaseHTTPRequestHandler):
    """One OpenAI streaming endpoint, with the two knobs that matter."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):            # silence the default stderr spam
        return

    def do_GET(self):                                       # noqa: N802
        cfg = self.server.cfg
        if self.path != "/metrics" or not cfg["metrics"]:
            self.send_error(404)
            return
        with self.server.lock:
            drafted = self.server.drafted
            accepted = self.server.accepted
        body = (
            "# TYPE vllm:spec_decode_num_draft_tokens_total counter\n"
            f'vllm:spec_decode_num_draft_tokens_total{{model="m"}} {drafted}\n'
            "# TYPE vllm:spec_decode_num_accepted_tokens_total counter\n"
            f'vllm:spec_decode_num_accepted_tokens_total{{model="m"}} {accepted}\n'
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):                                      # noqa: N802
        cfg = self.server.cfg
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        # `inflight` counts GENERATION, not the socket. A client that has read
        # `[DONE]` and closed its connection can start its next request while
        # this handler is still returning, and counting the socket would then
        # report a phantom overlap at c=1.
        with self.server.lock:
            self.server.inflight += 1
            self.server.peak = max(self.server.peak, self.server.inflight)
        released = False
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            n_tok = cfg["tokens"]
            per = cfg["tokens_per_chunk"]
            time.sleep(cfg["prefill_s"])
            emitted = 0
            while emitted < n_tok:
                k = min(per, n_tok - emitted)
                time.sleep(cfg["per_token_s"] * k)
                emitted += k
                self._sse({"choices": [{"delta": {"content": "ab " * k}}]})
            usage = {"prompt_tokens": cfg["prompt_tokens"],
                     "completion_tokens": n_tok}
            if cfg["accepted"]:
                usage["accepted_draft_tokens"] = cfg["accepted"]
            self._sse({"choices": [], "usage": usage} if cfg["usage"]
                      else {"choices": []})
            with self.server.lock:
                self.server.drafted += cfg["drafted"]
                self.server.accepted += cfg["accepted"]
                self.server.inflight -= 1
                released = True
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        finally:
            if not released:
                with self.server.lock:
                    self.server.inflight -= 1

    def _sse(self, obj):
        payload = ("data: " + json.dumps(obj) + "\n\n").encode()
        self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
        self.wfile.flush()


def _stop(srv):
    srv.shutdown()
    srv.server_close()


def start_mock(**over):
    cfg = {"tokens": 8, "tokens_per_chunk": 1, "per_token_s": 0.01,
           "prefill_s": 0.02, "prompt_tokens": 100, "usage": True,
           "metrics": True, "drafted": 10, "accepted": 6}
    cfg.update(over)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    srv.cfg = cfg
    srv.lock = threading.Lock()
    srv.inflight = 0
    srv.peak = 0
    srv.drafted = 0
    srv.accepted = 0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def run_leg(url, tmp, label, concurrency, num_prompts, warmup, arm="OURS",
            corpus=None):
    corpus_path = corpus or make_corpus(tmp, num_prompts + warmup + 2)
    out = Path(tmp) / f"{label}.json"
    rc = subprocess.run(
        [sys.executable, str(HARNESS / "client.py"), "--url", url,
         "--dataset", str(corpus_path), "--num-prompts", str(num_prompts),
         "--warmup", str(warmup), "--concurrency", str(concurrency),
         "--max-tokens", "8", "--label", label, "--arm", arm,
         "--round", "1", "--out", str(out)],
        capture_output=True, text=True, timeout=300)
    return rc, json.loads(out.read_text()) if out.exists() else None


def make_corpus(tmp, n):
    path = Path(tmp) / "corpus.json"
    path.write_text(json.dumps([
        {"id": f"X-{i:04d}", "band": "S",
         "conversations": [{"from": "human", "value": f"question {i}"},
                           {"from": "gpt", "value": ""}]}
        for i in range(n)]))
    return path


class PercentileContract(unittest.TestCase):
    """Linear interpolation between order statistics, numpy's default."""

    def test_matches_hand_computed_linear_interpolation(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        # idx = (n-1)*p/100 = 3*0.5 = 1.5 -> 2.0 + (3.0-2.0)*0.5
        self.assertAlmostEqual(report.percentile(xs, 50), 2.5)
        self.assertAlmostEqual(report.percentile(xs, 0), 1.0)
        self.assertAlmostEqual(report.percentile(xs, 100), 4.0)
        # idx = 3*0.9 = 2.7 -> 3.0 + (4.0-3.0)*0.7
        self.assertAlmostEqual(report.percentile(xs, 90), 3.7)

    def test_a_nearest_rank_implementation_would_fail_this(self):
        """The mutation this case exists to catch."""
        xs = [1.0, 2.0, 3.0, 4.0]
        nearest_rank = sorted(xs)[min(len(xs) - 1, int(len(xs) * 0.9))]
        self.assertNotAlmostEqual(report.percentile(xs, 90), nearest_rank)

    def test_single_sample_and_empty(self):
        self.assertEqual(report.percentile([7.0], 99), 7.0)
        self.assertTrue(report.percentile([], 50) != report.percentile([], 50))


class ClosedLoopConcurrency(unittest.TestCase):
    """c workers overlap; c=1 never does."""

    def test_concurrency_one_never_overlaps(self):
        srv, url = start_mock()
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "c1", 1, 6, 2)
        _stop(srv)
        self.assertEqual(rc.returncode, 0, rc.stderr)
        self.assertEqual(srv.peak, 1, "c=1 put more than one request in flight")

    def test_concurrency_four_really_overlaps(self):
        srv, url = start_mock()
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "c4", 4, 12, 4)
        _stop(srv)
        self.assertEqual(rc.returncode, 0, rc.stderr)
        self.assertEqual(srv.peak, 4,
                         f"c=4 peaked at {srv.peak} concurrent requests")


class WarmupIsExcluded(unittest.TestCase):
    def test_measured_window_holds_no_warmup_request(self):
        srv, url = start_mock()
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "w", 2, 6, 3)
        _stop(srv)
        self.assertEqual(rc.returncode, 0, rc.stderr)
        phases = [r["phase"] for r in leg["records"]]
        self.assertEqual(phases.count("warmup"), 3)
        self.assertEqual(phases.count("measured"), 6)
        measured = report.ok_records(leg, "measured")
        self.assertTrue(all(r["phase"] == "measured" for r in measured))
        # And the with-warmup view still sees every request.
        self.assertEqual(len(report.ok_records(leg, None)), 9)

    def test_measured_wall_clock_excludes_the_warmup_phase(self):
        srv, url = start_mock(per_token_s=0.02)
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "w2", 1, 4, 4)
        _stop(srv)
        s = leg["summary"]
        self.assertGreater(s["warm_wall_s"], 0.0)
        # 4 warm + 4 measured at c=1; the measured wall must be about half the
        # total, not all of it.
        self.assertLess(s["measured_wall_s"],
                        0.75 * (s["warm_wall_s"] + s["measured_wall_s"]))


class ChunkingIsRecovered(unittest.TestCase):
    """Two engines, one token per chunk and four, must not be compared raw."""

    def _leg(self, tmp, per_chunk, label, arm):
        srv, url = start_mock(tokens=16, tokens_per_chunk=per_chunk,
                              per_token_s=0.01, prefill_s=0.05)
        rc, leg = run_leg(url, tmp, label, 1, 6, 2, arm=arm)
        _stop(srv)
        self.assertEqual(rc.returncode, 0, rc.stderr)
        return leg

    def test_tokens_per_chunk_comes_from_usage_not_from_the_chunk_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            fine = self._leg(tmp, 1, "fine", "THEIRS")
            coarse = self._leg(tmp, 4, "coarse", "OURS")
        a_fine = report.axes(report.ok_records(fine, "measured"))
        a_coarse = report.axes(report.ok_records(coarse, "measured"))
        self.assertAlmostEqual(sum(a_fine["toks_per_chunk"])
                               / len(a_fine["toks_per_chunk"]), 1.0, places=6)
        self.assertAlmostEqual(sum(a_coarse["toks_per_chunk"])
                               / len(a_coarse["toks_per_chunk"]), 4.0, places=6)
        # Both engines produced the same 16 tokens, so a chunk count would have
        # said one produced a quarter of the other.
        self.assertEqual(sum(a_fine["out_tokens"]), sum(a_coarse["out_tokens"]))

    def test_the_coarse_engine_pays_for_its_chunking_in_raw_ttft(self):
        with tempfile.TemporaryDirectory() as tmp:
            fine = self._leg(tmp, 1, "fine", "THEIRS")
            coarse = self._leg(tmp, 4, "coarse", "OURS")
        a_fine = report.axes(report.ok_records(fine, "measured"))
        a_coarse = report.axes(report.ok_records(coarse, "measured"))
        raw_f = report.percentile(a_fine["ttft"], 50)
        raw_c = report.percentile(a_coarse["ttft"], 50)
        cor_c = report.percentile(a_coarse["ttft_corrected"], 50)
        self.assertGreater(raw_c, raw_f,
                           "the coarse engine should look slower to first CHUNK")
        self.assertLess(cor_c, raw_c,
                        "the correction must remove chunking, not add to it")
        # It must not over-correct past the fine engine's own prefill.
        self.assertGreater(cor_c, 0.0)

    def test_first_chunk_tokens_tracks_the_servers_granularity(self):
        with tempfile.TemporaryDirectory() as tmp:
            coarse = self._leg(tmp, 4, "coarse", "OURS")
        a = report.axes(report.ok_records(coarse, "measured"))
        est = sum(a["first_chunk_tokens"]) / len(a["first_chunk_tokens"])
        self.assertGreater(est, 2.0)
        self.assertLess(est, 6.0)


class UsageIsMandatory(unittest.TestCase):
    def test_a_server_without_usage_is_marked_chunks_not_counted_silently(self):
        srv, url = start_mock(usage=False)
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "nousage", 1, 4, 2)
        _stop(srv)
        a = report.axes(report.ok_records(leg, "measured"))
        self.assertEqual(a["counted_by"], "chunks")
        self.assertEqual(a["n"], 0, "a leg with no usage must yield no axis")


class AcceptanceFromMetrics(unittest.TestCase):
    def test_the_metrics_delta_becomes_a_rate(self):
        srv, url = start_mock(drafted=10, accepted=6)
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "acc", 1, 5, 2)
        _stop(srv)
        acc = report.acceptance(leg)
        self.assertEqual(acc["source"], "/metrics")
        # Five measured requests between the two scrapes.
        self.assertAlmostEqual(acc["draft"], 50.0)
        self.assertAlmostEqual(acc["accepted"], 30.0)
        self.assertAlmostEqual(acc["rate"], 0.6)

    def test_an_engine_without_metrics_falls_back_to_usage(self):
        srv, url = start_mock(metrics=False, accepted=7)
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "theirs", 1, 5, 2, arm="THEIRS")
        _stop(srv)
        acc = report.acceptance(leg)
        self.assertEqual(acc["source"], "usage")
        self.assertEqual(acc["accepted"], 35)

    def test_an_engine_exposing_nothing_reports_absent_not_zero(self):
        srv, url = start_mock(metrics=False, accepted=0)
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "silent", 1, 4, 2, arm="THEIRS")
        _stop(srv)
        acc = report.acceptance(leg)
        self.assertEqual(acc["source"], "absent")
        self.assertNotIn("rate", acc)


class ThroughputDenominatorExcludesWarmup(unittest.TestCase):
    """The number the phase labels exist to protect.

    A fresh review broke `measured_wall_s` to `warm_wall + measured_wall` and the
    whole suite stayed green: the records, their phase labels and every
    percentile were still right, and only the headline throughput halved. The
    old bound (`measured < 0.75 * total`) did not bite because warmup and
    measured cost the same. Here the warmup phase is deliberately EXPENSIVE, so
    a denominator that absorbs it cannot pass.
    """

    def test_the_measured_wall_clock_is_the_measured_phase_and_nothing_else(self):
        srv, url = start_mock(tokens=4, per_token_s=0.05, prefill_s=0.02)
        with tempfile.TemporaryDirectory() as tmp:
            # 8 warmup requests against 2 measured, at c=1: the warmup phase is
            # four times the measured one.
            rc, leg = run_leg(url, tmp, "denom", 1, 2, 8)
        _stop(srv)
        self.assertEqual(rc.returncode, 0, rc.stderr)
        s = leg["summary"]
        measured = report.ok_records(leg, "measured")
        # Every measured request is sequential at c=1, so the phase's wall clock
        # is the sum of their latencies plus the client's own per-request gap.
        span = sum(r["latency"] for r in measured)
        self.assertGreaterEqual(s["measured_wall_s"], span)
        self.assertLess(s["measured_wall_s"], span * 1.5,
                        "the measured wall clock absorbed time no measured "
                        "request spent")
        self.assertGreater(s["warm_wall_s"], 3 * s["measured_wall_s"],
                           "the warmup phase should dominate this fixture")

    def test_throughput_uses_the_measured_denominator(self):
        srv, url = start_mock(tokens=4, per_token_s=0.05, prefill_s=0.02)
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "denom2", 1, 2, 8)
        _stop(srv)
        a = report.axes(report.ok_records(leg, "measured"))
        rate = sum(a["out_tokens"]) / leg["summary"]["measured_wall_s"]
        both = sum(a["out_tokens"]) / (leg["summary"]["measured_wall_s"]
                                       + leg["summary"]["warm_wall_s"])
        self.assertGreater(rate, 3 * both,
                           "a denominator that included warmup would be "
                           "indistinguishable here")


class TpotMatchesUpstream(unittest.TestCase):
    """`tpot` is the declared primary inter-token axis and had no value control.

    vllm/benchmarks/serve.py:610 divides by `output_tokens - 1`, not by
    `output_tokens`. The mock's per-token cost is known, so the arithmetic is
    checkable exactly rather than by inspection.
    """

    def test_tpot_divides_by_output_tokens_minus_one(self):
        per_tok = 0.02
        srv, url = start_mock(tokens=11, tokens_per_chunk=1,
                              per_token_s=per_tok, prefill_s=0.10)
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "tpot", 1, 6, 2)
        _stop(srv)
        a = report.axes(report.ok_records(leg, "measured"))
        mean_tpot = sum(a["tpot"]) / len(a["tpot"]) / 1000.0
        # 11 tokens: ttft covers the prefill and token 1, then 10 more gaps of
        # `per_tok`. (latency - ttft) / (11 - 1) == per_tok.
        self.assertAlmostEqual(mean_tpot, per_tok, delta=0.5 * per_tok)
        # Dividing by n instead of n-1 gives 10/11 of this, which is inside that
        # delta, so pin the ratio against the wrong divisor directly.
        wrong = mean_tpot * 11 / 10
        self.assertLess(abs(mean_tpot - per_tok), abs(wrong - per_tok),
                        "n-1 must fit the mock better than n")

    def test_e2el_and_ttft_bracket_tpot(self):
        srv, url = start_mock(tokens=11, tokens_per_chunk=1, per_token_s=0.02,
                              prefill_s=0.10)
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "brk", 1, 6, 2)
        _stop(srv)
        a = report.axes(report.ok_records(leg, "measured"))
        for ttft, e2el, tpot in zip(a["ttft"], a["e2el"], a["tpot"]):
            self.assertAlmostEqual(e2el - ttft, tpot * 10, delta=0.2 * tpot * 10)


class ColdStartNamesTheFirstRequest(unittest.TestCase):
    """The cold-start column printed `min(ttft)` under a "first request" heading.

    At c > 1 the warmup phase is concurrent, and the cheapest warmup request is
    exactly the one that did NOT pay the cold start. The published claim this
    column carries is "request i=0 cost us 27.5 s", so printing the minimum
    understates it in the direction that flatters.
    """

    def _leg_with_a_slow_first_request(self, tmp):
        # One record with a large ttft and three cheap ones, in dispatch order.
        recs = []
        for i, ttft in enumerate([2.75, 0.09, 0.08, 0.085]):
            recs.append({
                "i": i, "id": f"X-{i:04d}", "band": "S", "phase": "warmup",
                "ok": True, "ttft": ttft, "latency": ttft + 0.2,
                "n_chunks": 4, "itls": [0.05, 0.05, 0.05],
                "t_dispatch": i * 0.001,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "accepted_draft_tokens": None,
                "first_chunk_chars": 4, "total_chars": 20,
            })
        for i in range(4, 8):
            recs.append({
                "i": i, "id": f"X-{i:04d}", "band": "S", "phase": "measured",
                "ok": True, "ttft": 0.09, "latency": 0.29, "n_chunks": 4,
                "itls": [0.05, 0.05, 0.05], "t_dispatch": i * 0.001,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "accepted_draft_tokens": None,
                "first_chunk_chars": 4, "total_chars": 20,
            })
        leg = {"summary": {"leg": "OURS-r1-c1", "arm": "OURS", "round": 1,
                           "concurrency": 1, "warmup_requests": 4,
                           "measured_requests": 4, "warm_wall_s": 3.5,
                           "measured_wall_s": 1.2},
               "config": {}, "metrics_before": {"available": False},
               "metrics_after": {"available": False}, "records": recs}
        path = Path(tmp) / "OURS-r1-c1.json"
        path.write_text(json.dumps(leg))
        return path

    def test_the_column_prints_the_first_request_not_the_cheapest(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._leg_with_a_slow_first_request(tmp)
            rc = subprocess.run(
                [sys.executable, str(HARNESS / "report.py"), "--dir", tmp,
                 "--glob", "*-r*-c*.json"],
                capture_output=True, text=True, timeout=120)
        self.assertEqual(rc.returncode, 0, rc.stderr)
        row = next(ln for ln in rc.stdout.splitlines()
                   if ln.startswith("| OURS | 1 | 1 | 2750.0"))
        self.assertIn("2750.0", row)
        self.assertNotIn("| 80.0 |", row.split("2750.0")[0] + "|",
                         "the cheapest warmup request must not be published "
                         "as the first one")


class PercentileHeaderMatchesWhatIsComputed(unittest.TestCase):
    """The header was hardcoded while the values came from PCTS.

    Setting `PCTS = (50, 50, 50, 50)` left the whole suite green, including the
    header check, so a reader could take p99 off a column holding p50.
    """

    def test_the_header_is_derived_from_pcts(self):
        srv, url = start_mock()
        with tempfile.TemporaryDirectory() as tmp:
            corpus = make_corpus(tmp, 20)
            run_leg(url, tmp, "OURS-r1-c1", 1, 6, 2, arm="OURS", corpus=corpus)
            _stop(srv)
            rc = subprocess.run(
                [sys.executable, str(HARNESS / "report.py"), "--dir", tmp,
                 "--glob", "*-r*-c*.json"],
                capture_output=True, text=True, timeout=120)
        self.assertEqual(rc.returncode, 0, rc.stderr)
        header = next(ln for ln in rc.stdout.splitlines()
                      if ln.startswith("| arm | c | n | axis |"))
        cells = [c.strip() for c in header.split("|")[5:-3]]
        self.assertEqual(cells, [f"p{p:g}" for p in report.PCTS], header)
        self.assertEqual(len(set(report.PCTS)), len(report.PCTS),
                         "PCTS must not repeat a percentile")


class ReportTablesAreWellFormed(unittest.TestCase):
    """Every markdown row carries exactly as many cells as its header.

    The statistics were right and the table was not: the axis name and p50
    rendered into one cell, so every value after it sat under the heading to its
    left and a reader taking p95 took p90. Nothing in this file could see that,
    because nothing read the markdown.
    """

    def _report(self, tmp):
        srv_o, url_o = start_mock(tokens=12, tokens_per_chunk=4, metrics=True)
        srv_t, url_t = start_mock(tokens=12, tokens_per_chunk=1, metrics=False,
                                  accepted=5)
        corpus = make_corpus(tmp, 40)
        for rnd in (1, 2):
            for c in (1, 2):
                run_leg(url_o, tmp, f"OURS-r{rnd}-c{c}", c, 6, 2, arm="OURS",
                        corpus=corpus)
                run_leg(url_t, tmp, f"THEIRS-r{rnd}-c{c}", c, 6, 2,
                        arm="THEIRS", corpus=corpus)
        _stop(srv_o)
        _stop(srv_t)
        for name in ("OURS", "THEIRS"):
            for rnd in (1, 2):
                for c in (1, 2):
                    src = Path(tmp) / f"{name}-r{rnd}-c{c}.json"
                    data = json.loads(src.read_text())
                    data["summary"]["round"] = rnd
                    src.write_text(json.dumps(data))
        rc = subprocess.run(
            [sys.executable, str(HARNESS / "report.py"), "--dir", tmp,
             "--glob", "*-r*-c*.json"],
            capture_output=True, text=True, timeout=300)
        self.assertEqual(rc.returncode, 0, rc.stderr)
        return rc.stdout

    def test_every_row_has_its_header_s_column_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._report(tmp)
        width = None
        checked = 0
        for line in out.splitlines():
            if not line.startswith("|"):
                width = None
                continue
            cells = len(line.split("|"))
            if width is None:
                width = cells
                continue
            if set(line.replace("|", "").replace("-", "").strip()) == set():
                self.assertEqual(cells, width, f"separator row: {line}")
                continue
            self.assertEqual(cells, width,
                             f"row has {cells} cells, header has {width}: {line}")
            checked += 1
        self.assertGreater(checked, 40, "the report rendered almost no rows")

    def test_the_percentile_table_carries_every_reported_percentile(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._report(tmp)
        header = next(ln for ln in out.splitlines()
                      if ln.startswith("| arm | c | n | axis |"))
        for pct in ("p50", "p90", "p95", "p99"):
            self.assertIn(pct, header)
        self.assertIn("max", header)
        row = next(ln for ln in out.splitlines()
                   if ln.startswith("| OURS |") and " ttft ms |" in ln)
        self.assertEqual(len(row.split("|")), len(header.split("|")), row)

    def test_both_warmup_views_are_printed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._report(tmp)
        self.assertIn("warm only, warmup discarded", out)
        self.assertIn("with warmup included", out)


class CorpusIsDeterministic(unittest.TestCase):
    """A published length histogram has to be reproducible from the seed."""

    def _sources(self, tmp):
        gsm = Path(tmp) / "gsm.jsonl"
        gsm.write_text("".join(
            json.dumps({"question": "q " * (5 + i % 20)}) + "\n"
            for i in range(200)))
        he = Path(tmp) / "he.jsonl"
        he.write_text("".join(
            json.dumps({"prompt": "def f%d():\n    pass\n" % i + "x " * (10 + i)})
            + "\n" for i in range(164)))
        son = Path(tmp) / "sonnet.txt"
        son.write_text("".join(f"line {i} of the verse here\n" for i in range(517)))
        return gsm, he, son

    def _run(self, tmp, out, man, seed=0, count=40, weights=None):
        gsm, he, son = self._sources(tmp)
        argv = [sys.executable, str(HARNESS / "build_corpus.py"),
                "--gsm8k", str(gsm), "--humaneval", str(he),
                "--sonnet", str(son),
                "--count", str(count), "--seed", str(seed),
                "--out", str(out), "--manifest", str(man)]
        if weights is not None:
            argv += ["--weights", weights]
        return subprocess.run(argv, capture_output=True, text=True, timeout=120)

    def _build(self, tmp, out, man, seed=0, count=40, weights=None):
        rc = self._run(tmp, out, man, seed=seed, count=count, weights=weights)
        self.assertEqual(rc.returncode, 0, rc.stderr)
        return json.loads(Path(man).read_text())

    def test_same_seed_gives_the_same_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self._build(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man")
            b = self._build(tmp, Path(tmp) / "b.json", Path(tmp) / "b.man")
        self.assertEqual(a["corpus_sha256"], b["corpus_sha256"])

    def test_a_different_seed_gives_different_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self._build(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man")
            c = self._build(tmp, Path(tmp) / "c.json", Path(tmp) / "c.man", seed=1)
        self.assertNotEqual(a["corpus_sha256"], c["corpus_sha256"])

    def test_band_counts_sum_to_the_requested_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            man = self._build(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man",
                              count=137)
        self.assertEqual(sum(man["band_counts"].values()), 137)
        self.assertEqual(sum(v["n"] for v in man["realised_chars"].values()), 137)

    def test_every_source_is_pinned_by_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            man = self._build(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man")
        for name, src in man["sources"].items():
            self.assertEqual(len(src["sha256"]), 64, name)

    # The corpus that `docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/`
    # published is a pure function of (sources, seed, weights, count), so any
    # change to the default weights, to the band order, or to the number of
    # values each band draws from the shared `random.Random` moves these bytes.
    # This golden is over the fixture sources above, not the pinned corpora,
    # because a test may not download 3 corpora; it pins the same property.
    DEFAULT_GOLDEN_192 = \
        "689e73227fdf07018260d1f8603b3d1d0d281f0762eb4b075cfacad8b4dfc953"

    def test_the_default_corpus_bytes_are_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            man = self._build(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man",
                              count=192)
            blob = (Path(tmp) / "a.json").read_bytes()
        self.assertEqual(hashlib.sha256(blob).hexdigest(),
                         self.DEFAULT_GOLDEN_192)
        self.assertEqual(man["corpus_sha256"], self.DEFAULT_GOLDEN_192)

    def test_the_default_weights_spelled_out_give_the_same_bytes(self):
        """`--weights` with today's distribution is not a different corpus."""
        with tempfile.TemporaryDirectory() as tmp:
            man = self._build(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man",
                              count=192,
                              weights="S=0.35,M=0.40,L=0.15,XL=0.10")
        self.assertEqual(man["corpus_sha256"], self.DEFAULT_GOLDEN_192)

    def test_the_bands_are_ordered_by_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            man = self._build(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man",
                              count=200)
        med = {b: v["median"] for b, v in man["realised_chars"].items()}
        self.assertLess(med["S"], med["L"])
        self.assertLess(med["L"], med["XL"])


class LongBandIsRequestable(unittest.TestCase):
    """ISSUE-LOCAL-01M2TNH3A9DKGWCRADM4Q8J30C.

    Every published band stops at about 3.3k prompt tokens, which is 40% of the
    `--max-model-len 8192` both engines serve, so the corpus could not express
    the question `BENCH-QWEN38-EXL3-LONGCTX` asks. `XXL` is that band, and
    `--weights` is how a run asks for it without editing the generator.
    """

    _sources = CorpusIsDeterministic._sources
    _run = CorpusIsDeterministic._run
    _build = CorpusIsDeterministic._build

    def test_xxl_items_land_in_the_intended_character_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            man = self._build(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man",
                              count=200,
                              weights="S=0.2,M=0.2,L=0.2,XL=0.2,XXL=0.2")
        target = man["rules"]["XXL_target_chars"]
        xxl = man["realised_chars"]["XXL"]
        self.assertEqual(xxl["n"], 40)
        # The band draws k around `XXL_k_mid`, so it has a spread by design.
        # What it may not do is overlap XL or miss its target by a factor.
        self.assertGreater(xxl["min"], man["realised_chars"]["XL"]["max"])
        self.assertGreater(xxl["median"], 0.75 * target)
        self.assertLess(xxl["median"], 1.25 * target)

    # The served configuration this band exists for, from
    # `.agents/specs/bench-qwen38-exl3-longctx.md` §2. These numbers are the
    # BUDGET, and none of them is derived from `XXL_TARGET_CHARS`: that is the
    # whole point of the case below, which pins the generator's constant
    # against the context rather than against itself.
    SERVED_CONTEXT_TOKENS = 8192          # `--max-model-len 8192`
    OUTPUT_TOKENS = 192                   # `max_tokens: 192`
    # MEASURED, from `docs/benchmarks/qwen38-27b-exl3-variadic-gb10.md`: that
    # page ran this corpus through both engines, each rendering its own chat
    # template, and read the counts back from every server's own
    # `usage.prompt_tokens`. The corpus measures 26 to 3233 tokens with the
    # target's own tokenizer and the served histogram runs 78 to 3290, so the
    # template adds +52 at the corpus minimum and +57 at the `XL` top. The
    # LARGER measured delta is the one pinned here, so that the refusal below
    # is conservative at its own boundary: at 50 this case admits a target of
    # 21977 characters, whose ceiling is 7950 served tokens and overruns.
    # G-FITS still decides on the realised counts, because no published run
    # has built the `XXL` band.
    TEMPLATE_TOKENS = 57
    # The most ADVERSE characters-per-prompt-token pairing the published `XL`
    # band realised: 10048 characters at 3233 prompt tokens, the top of the
    # band, from `corpus-manifest.json` and `corpus-token-histogram.md` in
    # `docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/` over the same
    # 144 prompts. The band's mean pairing is 3.21 and its p50 pairing 3.27;
    # the smallest ratio is the one that buys the most tokens per character,
    # so it is the one a fit is checked against.
    CHARS_PER_PROMPT_TOKEN = 3.11
    # The band's expected ceiling over its target: 4.26% k jitter at
    # `k_mid = 47` plus 8.28% sampling spread, scaled from XL's measured 22.7%
    # at `k_mid = 20` by 1/sqrt(k), which is 12.54%. The derivation is in
    # `build_corpus.py`. The terms are carried unrounded because the rounded
    # 4.3% and 8.3% an earlier draft displayed add to 12.6%, not to 12.5%.
    CEILING_OVER_TARGET = 1.125

    def _prompt_budget_tokens(self):
        return (self.SERVED_CONTEXT_TOKENS - self.OUTPUT_TOKENS
                - self.TEMPLATE_TOKENS)

    def test_the_xxl_target_fits_the_served_context(self):
        """The constant is pinned against the CONTEXT, not against itself.

        `test_xxl_items_land_in_the_intended_character_range` reads its bound
        out of the same constant the generator sized the band with, so it
        asserts `X == X +/- 25%` and stays green at any target. 26000 and 30000
        characters both passed it, and both overrun the 8192 tokens this row
        serves. A band that overruns is voided by G-FITS after it has spent the
        lease, so the refusal has to happen here, with no GPU.
        """
        with tempfile.TemporaryDirectory() as tmp:
            man = self._build(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man",
                              count=200,
                              weights="S=0.2,M=0.2,L=0.2,XL=0.2,XXL=0.2")
        target = man["rules"]["XXL_target_chars"]
        budget = self._prompt_budget_tokens()
        ceiling = target * self.CEILING_OVER_TARGET / self.CHARS_PER_PROMPT_TOKEN
        self.assertLess(
            ceiling, budget,
            f"XXL_TARGET_CHARS = {target} gives an expected ceiling of "
            f"{ceiling:.0f} prompt tokens, which does not fit the "
            f"{self.SERVED_CONTEXT_TOKENS} - {self.OUTPUT_TOKENS} - "
            f"{self.TEMPLATE_TOKENS} = {budget} tokens the served "
            f"configuration leaves for the prompt body")
        # And the band as BUILT, not only as configured. A composition rule
        # that overshoots its target harder than the derivation allows would
        # pass the line above and still overrun the server.
        realised_max = man["realised_chars"]["XXL"]["max"]
        self.assertLess(
            realised_max / self.CHARS_PER_PROMPT_TOKEN, budget,
            f"the longest XXL prompt built here is {realised_max} characters, "
            f"which is over the {budget}-token prompt budget")

    def test_xxl_is_xl_with_only_the_length_changed(self):
        """The case above cannot see this, which is why this one exists.

        The spec's claim is that length is the ONLY variable between the two
        long bands, which is what makes the pair readable as one axis. Give XXL
        its own instruction and the lengths still sit where they should, so
        `test_xxl_items_land_in_the_intended_character_range` stays green while
        the comparison has quietly become two workloads.
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "a.json"
            self._build(tmp, out, Path(tmp) / "a.man", count=200,
                        weights="S=0.2,M=0.2,L=0.2,XL=0.2,XXL=0.2")
            corpus = json.loads(out.read_text())
        by_band = {}
        for item in corpus:
            body = item["conversations"][0]["value"]
            by_band.setdefault(item["band"], set()).add(body.split("\n\n")[0])
        self.assertEqual(len(by_band["XL"]), 1, "XL lost its single instruction")
        self.assertEqual(by_band["XXL"], by_band["XL"])

    def test_band_counts_match_the_requested_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            man = self._build(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man",
                              count=100,
                              weights="S=0.1,M=0.2,L=0.2,XL=0.2,XXL=0.3")
        self.assertEqual(man["band_counts"],
                         {"S": 10, "M": 20, "L": 20, "XL": 20, "XXL": 30})
        self.assertEqual(sum(man["band_counts"].values()), 100)

    def test_largest_remainder_still_holds_with_five_bands(self):
        with tempfile.TemporaryDirectory() as tmp:
            man = self._build(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man",
                              count=137,
                              weights="S=0.2,M=0.2,L=0.2,XL=0.2,XXL=0.2")
        self.assertEqual(sum(man["band_counts"].values()), 137)
        self.assertEqual(sorted(man["band_counts"].values()), [27, 27, 27, 28, 28])

    def test_the_manifest_records_the_weights_it_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            man = self._build(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man",
                              count=100,
                              weights="S=0.1,M=0.2,L=0.2,XL=0.2,XXL=0.3")
        self.assertEqual(man["weights"],
                         {"S": 0.1, "M": 0.2, "L": 0.2, "XL": 0.2, "XXL": 0.3})
        self.assertIn("XXL_target_chars", man["rules"])
        self.assertIn("XXL_k_mid", man["rules"])

    def test_weights_are_deterministic(self):
        w = "S=0.2,M=0.2,L=0.2,XL=0.2,XXL=0.2"
        with tempfile.TemporaryDirectory() as tmp:
            a = self._build(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man",
                            count=60, weights=w)
            b = self._build(tmp, Path(tmp) / "b.json", Path(tmp) / "b.man",
                            count=60, weights=w)
        self.assertEqual(a["corpus_sha256"], b["corpus_sha256"])

    def _refuses(self, weights, needle):
        with tempfile.TemporaryDirectory() as tmp:
            rc = self._run(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man",
                           weights=weights)
        self.assertNotEqual(rc.returncode, 0, rc.stdout)
        # Not `unrecognized arguments`: that is argparse refusing the FLAG, and
        # it would pass this case on a tree that never grew `--weights`.
        self.assertNotIn("unrecognized arguments", rc.stderr)
        self.assertIn(needle, rc.stderr,
                      f"refusal for {weights!r} did not name the defect: "
                      f"{rc.stderr!r}")
        return rc

    def test_an_unknown_band_is_refused_by_name(self):
        self._refuses("S=0.5,XXXL=0.5", "XXXL")

    def test_a_negative_weight_is_refused(self):
        self._refuses("S=1.2,M=-0.2", "negative")

    def test_weights_that_do_not_sum_to_one_are_refused(self):
        self._refuses("S=0.5,M=0.4", "sum")

    def test_a_malformed_pair_is_refused(self):
        self._refuses("S:0.5,M=0.5", "S:0.5")

    def test_a_non_numeric_weight_is_refused(self):
        self._refuses("S=half,M=0.5", "half")

    def test_a_nan_weight_is_refused_as_not_finite(self):
        """NaN passes the sum check, because `abs(nan - 1.0) > 1e-9` is False.

        Without the finiteness guard the run does not stop here. It reaches
        `band_counts`, where `int(nan)` raises a bare
        `ValueError: cannot convert float NaN to integer` from inside the
        split, which names neither the band nor the share. The docstring on
        `parse_weights` promises a refusal that names the offending text, so
        the message is the guarantee and `finite` is what pins it.
        """
        rc = self._refuses("S=0.5,M=nan", "finite")
        self.assertIn("M", rc.stderr)
        self.assertNotIn("cannot convert float NaN", rc.stderr)

    def test_an_infinite_weight_is_refused_as_not_finite(self):
        """An infinite share is caught by the same guard, not by the sum.

        The sum check would refuse `inf` too, with a message about the total,
        which sends the reader to the wrong band. This case fails when the
        guard goes, although the invocation is still refused.
        """
        self._refuses("S=0.5,M=inf", "finite")


class BinaryCacheRestores(unittest.TestCase):
    """ISSUE-LOCAL-01M2CZX87ZB0WHRW2YZYPW7VRZ.

    The rc share is CIFS mounted `file_mode=0664,nounix`, so no file on it is
    executable. A cache guard that tests `-x` never holds there, and every
    resume rebuilds the pinned engine from source. With an existence guard, a
    cache that a crash, a full disk, or a failed `.so` copy left incomplete is
    restored on every resume, dies at startup, and reads as a KV pool that does
    not fit. So the restore verifies the bytes against an md5 manifest, and the
    write commits the binary last, by rename.

    These cases execute the real fragments of job.sh under bash: the setup
    lines after `OURS TREE PIN`, the restore branch up to the build `else`
    (whose body is replaced by a marker), and the cache write that follows the
    build's `OURS binary ... md5=` line.
    """

    JOB = HARNESS / "job.sh"
    PIN = "39d3af455866bc47d76b0e0bd27fc3693e9e6ff5"
    ENGINE = b"cached-engine-bytes" * 64
    LIB = b"shared-object-bytes" * 16

    def _lines(self):
        return self.JOB.read_text().splitlines()

    def _setup_fragment(self):
        lines = self._lines()
        pin = [i for i, l in enumerate(lines) if 'res "OURS TREE PIN $PIN"' in l]
        self.assertEqual(len(pin), 1)
        end = lines.index('mkdir -p "$SCRATCH/bin"', pin[0])
        return lines[pin[0] + 1:end + 1]

    def _restore_fragment(self):
        lines = self._lines()
        start = lines.index('mkdir -p "$SCRATCH/bin"') + 1
        end = lines.index("else", start)
        return lines[start:end] + ["else", "    echo REBUILD", "fi"]

    def _write_fragment(self):
        lines = self._lines()
        built = [i for i, l in enumerate(lines)
                 if l.lstrip().startswith('res "OURS binary $BIN md5=')]
        self.assertEqual(len(built), 1)
        end = lines.index("fi", built[0])
        return lines[built[0] + 1:end]

    def _run(self, tmp, body):
        script = "\n".join(
            ["res() { echo \"RES $*\"; }",
             f"W={tmp / 'share'}", f"PIN={self.PIN}",
             f"SCRATCH={tmp / 'scratch'}"]
            + self._setup_fragment() + body)
        return subprocess.run(["bash", "-c", script], capture_output=True,
                              text=True, check=True).stdout

    def _cache(self, tmp, engine, manifest, lib=True):
        d = tmp / "share" / "bin" / self.PIN
        d.mkdir(parents=True)
        (d / "vllm-server").write_bytes(engine)
        (d / "vllm-server").chmod(0o664)       # what the CIFS mount reports
        if lib:
            (d / "libvt.so.1").write_bytes(self.LIB)
        if manifest is not None:
            (d / "MANIFEST.md5").write_text(manifest)
        return d

    def _manifest(self):
        return (f"{hashlib.md5(self.ENGINE).hexdigest()}  vllm-server\n"
                f"{hashlib.md5(self.LIB).hexdigest()}  libvt.so.1\n")

    def test_c_a_verified_cache_without_an_execute_bit_is_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._cache(tmp, self.ENGINE, self._manifest())
            out = self._run(tmp, self._restore_fragment())
            self.assertNotIn("REBUILD", out)
            self.assertIn("OURS binary restored from the share", out)
            restored = tmp / "scratch" / "bin" / "vllm-server"
            self.assertEqual(restored.read_bytes(), self.ENGINE)
            self.assertEqual((tmp / "scratch" / "bin" / "libvt.so.1")
                             .read_bytes(), self.LIB)
            self.assertTrue(restored.stat().st_mode & 0o100,
                            "the restored local copy must be executable")

    def test_a_a_truncated_cached_binary_is_deleted_and_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            d = self._cache(tmp, self.ENGINE[:100], self._manifest())
            out = self._run(tmp, self._restore_fragment())
            self.assertIn("REBUILD", out)
            self.assertNotIn("OURS binary restored from the share", out)
            self.assertIn("REJECTED", out)
            self.assertFalse(d.exists(), "the bad cache must be removed")

    def test_b_a_cached_binary_without_a_manifest_is_deleted_and_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            d = self._cache(tmp, self.ENGINE, None)   # what the old writer left
            out = self._run(tmp, self._restore_fragment())
            self.assertIn("REBUILD", out)
            self.assertIn("REJECTED", out)
            self.assertFalse(d.exists(), "the unverifiable cache must be removed")

    def test_a_cache_missing_a_listed_shared_object_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            d = self._cache(tmp, self.ENGINE, self._manifest(), lib=False)
            out = self._run(tmp, self._restore_fragment())
            self.assertIn("REBUILD", out)
            self.assertIn("REJECTED", out)
            self.assertFalse(d.exists())

    def test_an_absent_cached_binary_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = self._run(tmp, self._restore_fragment())
            self.assertIn("REBUILD", out)
            self.assertNotIn("REJECTED", out)

    def test_d_the_cache_write_leaves_a_manifest_that_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            local = tmp / "scratch" / "bin"
            local.mkdir(parents=True)
            (local / "vllm-server").write_bytes(self.ENGINE)
            (local / "libvt.so.1").write_bytes(self.LIB)
            self._run(tmp, self._write_fragment())
            d = tmp / "share" / "bin" / self.PIN
            self.assertEqual(sorted(p.name for p in d.iterdir()),
                             ["MANIFEST.md5", "libvt.so.1", "vllm-server"],
                             "no .partial file may remain")
            subprocess.run(["md5sum", "-c", "--strict", "MANIFEST.md5"],
                           cwd=d, check=True, capture_output=True)
            self.assertEqual(
                sorted((d / "MANIFEST.md5").read_text().splitlines()),
                sorted(self._manifest().splitlines()))
            # The round trip: what the writer leaves, the restore accepts.
            for p in local.iterdir():
                p.unlink()
            out = self._run(tmp, self._restore_fragment())
            self.assertIn("OURS binary restored from the share", out)
            self.assertEqual((local / "vllm-server").read_bytes(), self.ENGINE)


    def test_a_failed_cache_write_leaves_no_binary_to_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            local = tmp / "scratch" / "bin"
            local.mkdir(parents=True)
            (local / "vllm-server").write_bytes(self.ENGINE)
            (local / "libvt.so.1").write_bytes(self.LIB)
            d = tmp / "share" / "bin" / self.PIN
            d.mkdir(parents=True)
            d.chmod(0o555)                         # every copy into it fails
            try:
                out = self._run(tmp, self._write_fragment())
            finally:
                if d.exists():
                    d.chmod(0o755)
            self.assertIn("OURS binary cache write FAILED", out)
            self.assertFalse((d / "vllm-server").exists())
            out = self._run(tmp, self._restore_fragment())
            self.assertIn("REBUILD", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)

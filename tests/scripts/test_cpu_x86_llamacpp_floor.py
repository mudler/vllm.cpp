#!/usr/bin/env python3
"""Smoke contract for the BACKEND-GATE-CPU-LLAMACPP x86_64 harness (#433).

The committed harness shipped unable to run at all: it set `OUT=evi` and never
created the directory, so every redirection failed, every leg was DISCARDED for
a non-zero exit, and the run ended with `GIVING_UP too many discards` -- blaming
contention for a series it had never started. These tests run the real script
end to end against stub engines and assert it reaches `SERIES_DONE`, that the
figures come out of the script instead of a human's eye, and that its quiet gate
cannot be tripped by the harness's own process tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import shlex
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "cpu-x86-llamacpp-floor.sh"
EVIDENCE = ROOT / "docs/bench-evidence/cpu-x86-llamacpp-20260811.md"

OURS_REPORT = """\
Request throughput (req/s):                0.31
Output token throughput (tok/s):           5.99
Total token throughput (tok/s):            19.53
-------- Prefill vs Decode split (gate #1) --------
Prefill token throughput (tok/s, in/TTFT): 42.39
Output (decode) token throughput (tok/s):  5.99
"""

LLAMA_REPORT = """\
[
  {"n_prompt": 128, "n_gen": 0, "avg_ts": 44.00},
  {"n_prompt": 0, "n_gen": 32, "avg_ts": 6.00},
  {"n_prompt": 128, "n_gen": 32, "avg_ts": 20.00}
]
"""

OURS_RSS_KB = 2971664
LLAMA_RSS_KB = 2965468


def write_exec(path: pathlib.Path, body: str) -> pathlib.Path:
    path.write_text(body)
    path.chmod(0o755)
    return path


def proc_ppid(pid: int) -> int | None:
    try:
        line = pathlib.Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    tail = line.rsplit(") ", 1)[-1].split()
    return int(tail[1]) if len(tail) > 1 else None


class CpuX86FloorHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="cpu-x86-floor-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ours = write_exec(
            self.tmp / "vllm-bench-stub",
            "#!/bin/sh\ncat <<'EOF'\n" + OURS_REPORT + "EOF\n",
        )
        self.llama = write_exec(
            self.tmp / "llama-bench-stub",
            "#!/bin/sh\ncat <<'EOF'\n" + LLAMA_REPORT + "EOF\n",
        )
        # Stands in for `/usr/bin/time -v`, which CI images do not all carry.
        self.timev = write_exec(
            self.tmp / "timev-stub",
            "#!/bin/sh\n"
            'case "$1" in *llama*) rss=%d;; *) rss=%d;; esac\n'
            '"$@"\nrc=$?\n'
            '{ echo "\tUser time (seconds): ${TEST_USER_SECONDS:-100.10}"\n'
            '  echo "\tSystem time (seconds): ${TEST_SYSTEM_SECONDS:-2.80}"\n'
            '  echo "\tMaximum resident set size (kbytes): $rss"; } >&2\n'
            "exit $rc\n" % (LLAMA_RSS_KB, OURS_RSS_KB),
        )

    def run_harness(
        self,
        out: pathlib.Path,
        argv: list[str] | None = None,
        cpu_rows: list[str] | None = None,
        real_cpu: bool = False,
        observe_ancestry: bool = False,
        **env: str,
    ) -> subprocess.CompletedProcess[str]:
        base = {
            **os.environ,
            "M": "/dev/null",
            "OB": str(self.ours),
            "LB": str(self.llama),
            "OUT": str(out),
            "TIMEV": str(self.timev),
            "TASKSET": "",
            "REPS": "1",
            "T": "2",
            "BUSY_WINDOW": "0",
            "QUIET_BUSY": "100",
            "FOREIGN_MAX": "100",
            "WAIT_TIMEOUT": "30",
            "BUILDERS": "no-such-process-name",
        }
        base.update(env)
        if not real_cpu:
            # Replace only the kernel read boundary. Both production sampler
            # call sites, the contention decisions, and the engines stay real.
            rows = cpu_rows or self.cpu_series([1000] * 10, [20, 0, 0, 80, 0, 0, 0, 0, 0, 0])
            reader = self.tmp / "cpu-reader.py"
            ancestry = ""
            if observe_ancestry:
                # Observe live ancestors while the production sampler reads,
                # not before Bash can replace the copied-name fixture.
                ancestry = (
                    "import json, os\n"
                    "chain = []\n"
                    "pid = os.getppid()\n"
                    "while pid > 1:\n"
                    "    stat = pathlib.Path(f'/proc/{pid}/stat').read_text()\n"
                    "    comm = pathlib.Path(f'/proc/{pid}/comm').read_text().strip()\n"
                    "    chain.append([pid, comm])\n"
                    "    pid = int(stat.rsplit(') ', 1)[1].split()[1])\n"
                    f"with pathlib.Path({str(self.tmp / 'cpu-ancestors.jsonl')!r}).open('a') as log:\n"
                    "    log.write(json.dumps(chain) + '\\n')\n"
                )
            reader.write_text(
                "import pathlib\n" + ancestry +
                f"rows = {rows!r}\n"
                f"state = pathlib.Path({str(self.tmp / 'cpu-index')!r})\n"
                "i = int(state.read_text()) if state.exists() else 0\n"
                "state.write_text(str(i + 1))\n"
                "print(rows[min(i, len(rows) - 1)])\n"
            )
            script = self.tmp / "harness.sh"
            script.write_text(SCRIPT.read_text().replace(
                "/proc/stat", f"<(python3 {shlex.quote(str(reader))})"
            ))
            argv = [str(script) if arg == str(SCRIPT) else arg
                    for arg in (argv or ["bash", str(SCRIPT)])]
        return subprocess.run(
            argv or ["bash", str(SCRIPT)],
            cwd=self.tmp,
            env=base,
            text=True,
            capture_output=True,
            check=False,
            timeout=300,
        )

    @staticmethod
    def cpu_series(start: list[int], step: list[int]) -> list[str]:
        return ["cpu  " + " ".join(str(a + i * b) for a, b in zip(start, step))
                for i in range(100)]

    def test_each_endpoint_uses_one_cpu_line(self) -> None:
        # The first interval is 20%. Split endpoints instead report 110%.
        rows = ["cpu  0 0 0 0 0 0 0 0 0 0",
                "cpu  20 0 0 80 0 0 0 0 0 0",
                "cpu  200 0 0 80 0 0 0 0 0 0",
                "cpu  200 0 0 81 0 0 0 0 0 0"]
        rows += self.cpu_series([300, 0, 0, 180, 0, 0, 0, 0, 0, 0],
                                [20, 0, 0, 80, 0, 0, 0, 0, 0, 0])
        got = self.run_harness(self.tmp / "evi", cpu_rows=rows,
                               QUIET_BUSY="20", WAIT_TIMEOUT="0")
        self.assertEqual(got.returncode, 0, got.stdout + got.stderr)
        self.assertIn("SERIES_DONE", got.stdout)

    def test_exact_cpu_shares_reach_both_production_gates(self) -> None:
        cases = [
            # Integer increments beyond IEEE double's exact range must survive.
            ("large", [9007199254740992] + [0] * 9,
             [1, 0, 0, 3, 0, 0, 0, 0, 0, 0], 25),
            # Guest and guest_nice overlap user and nice, not extra total time.
            ("guest", [1000] * 10, [20, 10, 0, 70, 0, 0, 0, 0, 20, 10], 30),
            ("iowait", [1000] * 10, [0, 0, 0, 0, 100, 0, 0, 0, 0, 0], 0),
            ("steal", [1000] * 10, [0, 0, 0, 60, 0, 0, 0, 40, 0, 0], 40),
            ("all_busy_fields", [1000] * 10, [5, 5, 5, 70, 0, 5, 5, 5, 0, 0], 30),
        ]
        for name, start, step, expected in cases:
            with self.subTest(name=name):
                (self.tmp / "cpu-index").unlink(missing_ok=True)
                out = self.tmp / name
                got = self.run_harness(
                    out, cpu_rows=self.cpu_series(start, step),
                    QUIET_BUSY=str(expected), FOREIGN_MAX=str(expected),
                    WAIT_TIMEOUT="0", TEST_USER_SECONDS="0", TEST_SYSTEM_SECONDS="0",
                )
                self.assertEqual(got.returncode, 0, got.stdout + got.stderr)
                for leg in ("ours-1.load", "llama-1.load"):
                    self.assertIn(f"foreign_cpu_pct: {expected}\n", (out / leg).read_text())

    def test_invalid_cpu_samples_never_establish_a_quiet_window(self) -> None:
        valid = "cpu  100 100 100 100 100 100 100 100 0 0"
        cases = [
            (valid, valid),  # zero delta
            (valid, "cpu  101 100 100 100 99 100 100 100 0 0"),  # iowait decreases
            (valid, "cpu  99 100 100 200 100 100 100 100 0 0"),  # busy resets
            (valid, "cpu  100 100 100 99 100 100 100 100 0 0"),
            (valid, "cpu  99999999999999999 100 100 100 100 100 100 100 0 0"),
            (valid, "cpu  x 100 100 100 100 100 100 100 0 0"),
            (valid, "cpu  1 2"),
            (valid, ""),
        ]
        for before, after in cases:
            with self.subTest(after=after):
                (self.tmp / "cpu-index").unlink(missing_ok=True)
                got = self.run_harness(self.tmp / "invalid", cpu_rows=[before, after],
                                       QUIET_BUSY="100", WAIT_TIMEOUT="0")
                self.assertEqual(got.returncode, 4, got.stdout + got.stderr)
                self.assertIn("NO_QUIET_WINDOW", got.stdout)
                self.assertIn("busy=100%", got.stdout)
                self.assertNotIn(" START ", got.stdout)
                self.assertIn("INVALID_CPU_SAMPLE", got.stderr)

    def test_nonaggregate_cpu_labels_are_refused_at_either_endpoint(self) -> None:
        for label in ("cpu0", "intr"):
            for endpoint in (0, 1):
                with self.subTest(label=label, endpoint=endpoint):
                    (self.tmp / "cpu-index").unlink(missing_ok=True)
                    rows = self.cpu_series([100] * 10,
                                           [20, 0, 0, 80, 0, 0, 0, 0, 0, 0])
                    rows[endpoint] = rows[endpoint].replace("cpu ", label + " ", 1)
                    got = self.run_harness(self.tmp / "nonaggregate", cpu_rows=rows,
                                           QUIET_BUSY="100", WAIT_TIMEOUT="0")
                    self.assertEqual(got.returncode, 4, got.stdout + got.stderr)
                    self.assertIn("NO_QUIET_WINDOW", got.stdout)
                    self.assertIn("INVALID_CPU_SAMPLE", got.stderr)
                    self.assertNotIn(" START ", got.stdout)

    def test_decimal_counters_and_own_time_subtraction(self) -> None:
        rows = self.cpu_series([8] * 10, [50, 0, 0, 50, 0, 0, 0, 0, 0, 0])
        rows = ["cpu  " + " ".join(v.zfill(8) for v in row.split()[1:]) for row in rows]
        got = self.run_harness(self.tmp / "evi", cpu_rows=rows, QUIET_BUSY="50",
                               FOREIGN_MAX="25", TEST_USER_SECONDS="0.25",
                               TEST_SYSTEM_SECONDS="0", WAIT_TIMEOUT="0")
        self.assertEqual(got.returncode, 0, got.stdout + got.stderr)
        self.assertIn("foreign_cpu_pct: 25\n", (self.tmp / "evi/ours-1.load").read_text())

    def test_invalid_leg_sample_is_discarded_even_at_full_ceiling(self) -> None:
        rows = self.cpu_series([1000] * 10, [20, 0, 0, 80, 0, 0, 0, 0, 0, 0])[:3]
        rows += [rows[-1]]  # no elapsed CPU time across the engine leg
        got = self.run_harness(self.tmp / "evi", cpu_rows=rows, WAIT_TIMEOUT="0")
        self.assertNotEqual(got.returncode, 0, got.stdout + got.stderr)
        self.assertIn("ours rep=1 DISCARDED", got.stdout)
        self.assertIn("INVALID_CPU_SAMPLE", got.stderr)
        self.assertFalse((self.tmp / "evi/summary.md").exists())

    def test_true_cpu_contention_refuses_or_discards(self) -> None:
        rows = self.cpu_series([1000] * 10, [80, 0, 0, 20, 0, 0, 0, 0, 0, 0])
        got = self.run_harness(self.tmp / "busy", cpu_rows=rows,
                               QUIET_BUSY="79", WAIT_TIMEOUT="0")
        self.assertEqual(got.returncode, 4, got.stdout + got.stderr)
        self.assertIn("busy=80%", got.stdout)
        (self.tmp / "cpu-index").unlink(missing_ok=True)
        got = self.run_harness(self.tmp / "foreign", cpu_rows=rows, FOREIGN_MAX="79",
                               TEST_USER_SECONDS="0", TEST_SYSTEM_SECONDS="0")
        self.assertEqual(got.returncode, 2, got.stdout + got.stderr)
        self.assertIn("foreign=80%", got.stdout)
        self.assertFalse((self.tmp / "foreign/summary.md").exists())

    def test_real_cpu_sampling_smoke(self) -> None:
        got = self.run_harness(self.tmp / "real", real_cpu=True, WAIT_TIMEOUT="0")
        self.assertIn(got.returncode, (0, 4), got.stdout + got.stderr)
        for share in re.findall(r"(?:busy|foreign)=(\d+)%", got.stdout):
            self.assertLessEqual(int(share), 100, got.stdout)
        if got.returncode == 4:
            # A zero-tick or decreasing real sample is not proof of quiet.
            self.assertIn("INVALID_CPU_SAMPLE", got.stderr)
            self.assertNotIn("SERIES_DONE", got.stdout)
        else:
            self.assertIn("SERIES_DONE", got.stdout)

    def test_runs_to_completion_and_creates_its_own_output_dir(self) -> None:
        out = self.tmp / "nested" / "evi"  # does not exist: the shipped bug
        got = self.run_harness(out)
        self.assertEqual(got.returncode, 0, got.stdout + got.stderr)
        self.assertIn("SERIES_DONE", got.stdout)
        self.assertNotIn("GIVING_UP", got.stdout)
        for name in (
            "ours-bench-1.txt",
            "ours-bench-1.time",
            "llama-bench-1.json",
            "llama-bench-1.time",
            "ours-1.load",
            "llama-1.load",
            "summary.md",
        ):
            self.assertTrue((out / name).exists(), f"{name} was not written")

    def test_the_published_figures_are_computed_not_transcribed(self) -> None:
        out = self.tmp / "evi"
        got = self.run_harness(out)
        self.assertEqual(got.returncode, 0, got.stdout + got.stderr)
        summary = (out / "summary.md").read_text()
        ratio = f"{OURS_RSS_KB / LLAMA_RSS_KB:.4f}x"
        self.assertIn(ratio, summary, summary)
        self.assertIn("42.39", summary)  # our prefill, straight out of the report
        self.assertIn("Peak RSS (KB)", summary)

    def test_g5_load_is_recorded_before_and_after_every_leg(self) -> None:
        out = self.tmp / "evi"
        got = self.run_harness(out)
        self.assertEqual(got.returncode, 0, got.stdout + got.stderr)
        for leg in ("ours-1.load", "llama-1.load"):
            text = (out / leg).read_text()
            self.assertRegex(text, r"(?m)^before loadavg: [\d.]+ [\d.]+ [\d.]+$")
            self.assertRegex(text, r"(?m)^after loadavg: [\d.]+ [\d.]+ [\d.]+$")
            self.assertRegex(text, r"(?m)^foreign_cpu_pct: \d+$")
        summary = (out / "summary.md").read_text()
        self.assertIn("G5: load recorded before and after every leg", summary)

    def test_a_contended_leg_is_discarded_and_never_summarised(self) -> None:
        out = self.tmp / "evi"
        got = self.run_harness(out, FOREIGN_MAX="-1")
        self.assertEqual(got.returncode, 2, got.stdout + got.stderr)
        self.assertIn("DISCARDED", got.stdout)
        self.assertIn("GIVING_UP", got.stdout)
        self.assertFalse((out / "summary.md").exists())

    def test_no_quiet_window_stops_instead_of_averaging_through_it(self) -> None:
        out = self.tmp / "evi"
        got = self.run_harness(out, QUIET_BUSY="-1", WAIT_TIMEOUT="0")
        self.assertEqual(got.returncode, 4, got.stdout + got.stderr)
        self.assertIn("NO_QUIET_WINDOW", got.stdout)

    def unique_copy(self, tool: str, name: str) -> pathlib.Path:
        source = shutil.which(tool)
        self.assertIsNotNone(source, f"{tool} is required for this test")
        # tempfile's directory token isolates concurrent fixtures. Keep comm
        # below Linux's 15-character truncation boundary for pgrep -x.
        dest = self.tmp / (name[:6] + self.tmp.name.rsplit("-", 1)[-1])
        shutil.copy(source, dest)
        dest.chmod(0o755)
        return dest

    def test_the_quiet_gate_does_not_see_the_harnesss_own_process_tree(self) -> None:
        """The bug class that has now cost this harness two series.

        `pgrep -f` once matched the waiter's own command line; the one-minute
        load average then counted the harness's own 20-thread leg. Here the
        gate is pointed at a uniquely-named process that is an ANCESTOR of the
        script -- the shape of both shipped bugs -- and the run must still
        complete. The name is unique to this test, so nothing else on the
        machine can make the result depend on what else is running.
        """
        # <= 15 chars: /proc/<pid>/comm truncates, and `pgrep -x` matches
        # comm, so a longer unique name silently matches NOTHING and the test
        # passes for the wrong reason. It did, once, while being written.
        probe = self.unique_copy("bash", "vfloorgate")
        got = self.run_harness(
            self.tmp / "evi",
            # A command after bash prevents tail-exec from erasing this
            # named ancestor. Propagate the actual harness exit status.
            argv=[str(probe), "-c", 'bash "$0"; result=$?; exit "$result"', str(SCRIPT)],
            observe_ancestry=True,
            BUILDERS=probe.name,
            WAIT_TIMEOUT="10",
        )
        self.assertEqual(got.returncode, 0, got.stdout + got.stderr)
        self.assertIn("SERIES_DONE", got.stdout)
        snapshots = [json.loads(line) for line in
                     (self.tmp / "cpu-ancestors.jsonl").read_text().splitlines()]
        self.assertGreaterEqual(len(snapshots), 2)
        for chain in snapshots:
            self.assertIn(probe.name, [comm for _, comm in chain],
                          f"named fixture absent from live sampler ancestry: {chain}")

    def test_the_quiet_gate_still_sees_a_foreign_process_of_the_same_shape(self) -> None:
        """The exclusion must be our own tree, not "never count anything"."""
        probe = self.unique_copy("sleep", "vfloorforeign")
        running = subprocess.Popen([str(probe), "60"])
        self.addCleanup(running.wait)
        self.addCleanup(running.kill)
        got = self.run_harness(
            self.tmp / "evi", BUILDERS=probe.name, WAIT_TIMEOUT="0"
        )
        self.assertEqual(got.returncode, 4, got.stdout + got.stderr)
        self.assertIn("NO_QUIET_WINDOW", got.stdout)
        self.assertIn("builders=1", got.stdout)

    def test_concurrent_fixtures_do_not_share_probe_names(self) -> None:
        other = CpuX86FloorHarnessTests()
        other.setUp()
        self.addCleanup(other.doCleanups)
        probe = self.unique_copy("sleep", "vfloorforeign")
        neighbor = other.unique_copy("sleep", "vfloorforeign")
        for path in (probe, neighbor):
            running = subprocess.Popen([str(path), "60"])
            self.addCleanup(running.wait)
            self.addCleanup(running.kill)
        got = self.run_harness(self.tmp / "evi", BUILDERS=probe.name, WAIT_TIMEOUT="0")
        self.assertEqual(got.returncode, 4, got.stdout + got.stderr)
        self.assertIn("builders=1", got.stdout)

    def test_the_recorded_correctness_hash_matches_the_recorded_output(self) -> None:
        """Review mutated the recorded sha256 and nothing caught it.

        It was unverifiable as well as unguarded: the evidence recorded a hash
        of a continuation produced from a prompt it never wrote down. The
        prompt, the continuation and the hash recipe are committed now, so the
        claim is arithmetic and this test does the arithmetic.
        """
        text = EVIDENCE.read_text()
        recorded = re.search(r"SHA-256 `([0-9a-f]{64})`\. The\n", text)
        self.assertIsNotNone(recorded, "the 32-token correctness hash is not recorded")
        literal = re.search(r"printf '%s' '([^']*)' \| sha256sum", text)
        self.assertIsNotNone(literal, "the hash recipe is not recorded")
        payload = literal.group(1)
        self.assertEqual(
            hashlib.sha256(payload.encode()).hexdigest(),
            recorded.group(1),
            "the recorded sha256 is not the sha256 of the recorded continuation",
        )
        # ... and the recipe must hash the continuation the file displays.
        shown = re.search(r"\n```text\n( Rome\.[^\n]*)\n```\n", text)
        self.assertIsNotNone(shown, "the 32-token continuation is not quoted")
        self.assertEqual(shown.group(1).strip(), payload)

    def test_the_headline_ratio_is_the_ratio_of_the_recorded_values(self) -> None:
        """Review mutated peak RSS to read the wrong process: 1,688x, uncaught.

        Nothing in the tree ties a published number to the artefact it came
        from -- that gap is recorded in the spec's Risks and is bigger than
        this row. What is cheap is refusing to let the three numbers in the
        headline row disagree with each other.
        """
        text = EVIDENCE.read_text()
        row = re.search(
            r"\| \*\*Peak RSS\*\* \| [\d.]+ GiB \(([\d,]+) KB\) \| [\d.]+ GiB "
            r"\(([\d,]+) KB\) \| \*\*([\d.]+)x\*\*",
            text,
        )
        self.assertIsNotNone(row, "the peak RSS row is not in its recorded shape")
        ours = int(row.group(1).replace(",", ""))
        theirs = int(row.group(2).replace(",", ""))
        self.assertEqual(f"{ours / theirs:.4f}", row.group(3))

    def test_the_recorded_recipe_and_the_harness_cannot_drift(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn("TIMEV=${TIMEV:-/usr/bin/time -v}", text)
        self.assertIn("TASKSET=${TASKSET-taskset -c 0-19}", text)
        # The gate decides on measured foreign CPU share, never on a load
        # average the harness itself inflates.
        self.assertNotIn("QUIET_LOAD", text)
        self.assertRegex(text, r"(?m)^\s*if \[ \"\$p\" -le \"\$QUIET_BUSY\" \]")
        evidence = EVIDENCE.read_text()
        self.assertIn("scripts/cpu-x86-llamacpp-floor.sh", evidence)
        self.assertIn(re.search(r"(taskset -c 0-19)", text).group(1), evidence)


if __name__ == "__main__":
    unittest.main()

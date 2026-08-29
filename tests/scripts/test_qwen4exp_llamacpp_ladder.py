#!/usr/bin/env python3
"""Failure-mode proofs for ``scripts/qwen4exp-llamacpp-ladder.sh``.

This harness has not held a lease yet, and that is the point.  The bench
instruments in this tree once carried four box-killing defects at once -- an
unguarded artifact path, ``local a=$1 b=$((a))`` breaking under ``set -u``, a KV
formula that asked for 128 GiB on a 119 GB box, and a wrapper whose absence made
the wrapped command silently not run.  Each of those is cheap to catch here and
expensive to catch on a leased GB10 with a 67 GiB model already resident.

Every test below asserts a NAMED exit code, not merely a non-zero one.  "It
failed" is not a proof that the guard you meant fired; three of the five guards
here would otherwise be indistinguishable from a typo in the fixture.

Two tests are MUTATIONS: they break the script in a scratch copy and assert that
the corresponding check goes red.  A guard that stays green when the thing it
guards is removed is measuring nothing, and the static scan in particular is
worthless without its mutation.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import stat
import subprocess
import tempfile
import textwrap
import time
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "qwen4exp-llamacpp-ladder.sh"

E_USAGE = 2
E_MODEL_MISSING = 10
E_SERVER_MISSING = 11
E_CLIENT_MISSING = 12
E_WRAPPER_MISSING = 13
E_CORPUS_MISSING = 14
E_MEM_BUDGET = 15
E_LOCK_TIMEOUT = 16
E_SERVER_NOT_READY = 17
E_LEG_FAILED = 18
E_KV_OVER_BUDGET = 19

SHARD_TOTAL = 3
# 3 GiB of "weights" and a 128 GiB "box": comfortably inside 85%.
SHARD_BYTES = 1 << 30
BOX_BYTES = 128 * (1 << 30)


def _code_only(script: str) -> str:
    """Drop comment lines.  A scan that reads the prose explaining why a lever is
    forbidden reports the explanation as the lever."""
    return "\n".join(
        line for line in script.splitlines() if not line.lstrip().startswith("#")
    )


def _bare_arithmetic_operands(script: str) -> set[str]:
    """Identifiers used BARE inside ``$(( ))``.

    ``${X}`` inside arithmetic is an ordinary parameter expansion and ``set -u``
    already polices it at the point of use.  The class this scan exists for is
    the BARE name -- ``$((waited + BUSY_WINDOW))`` -- where the shell resolves
    the identifier itself and an unset one aborts the run.
    """
    operands: set[str] = set()
    for expr in re.findall(r"\$\(\(([^()]*)\)\)", script):
        expr = re.sub(r"\$\{[^{}]*\}", " ", expr)
        expr = re.sub(r"\$[A-Za-z_][A-Za-z_0-9]*", " ", expr)
        operands.update(re.findall(r"(?<![\w$])([A-Za-z_][A-Za-z_0-9]*)", expr))
    return operands


def _write_exec(path: pathlib.Path, body: str) -> pathlib.Path:
    path.write_text(textwrap.dedent(body).lstrip())
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class Fixture:
    """A complete, PASSING fixture.  Each test breaks exactly one thing."""

    def __init__(self, root: pathlib.Path, script: pathlib.Path | None = None) -> None:
        self.root = root
        self.script = script or SCRIPT
        self.out = root / "evi"
        self.bin = root / "bin"
        self.bin.mkdir(parents=True, exist_ok=True)
        model_dir = root / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        self.shards = []
        for idx in range(1, SHARD_TOTAL + 1):
            shard = model_dir / f"Q-{idx:05d}-of-{SHARD_TOTAL:05d}.gguf"
            with shard.open("wb") as handle:
                handle.truncate(SHARD_BYTES)
            self.shards.append(shard)
        self.model = self.shards[0]
        self.tokenizer = root / "tokenizer"
        self.tokenizer.mkdir(exist_ok=True)
        (self.tokenizer / "tokenizer.json").write_text("{}")
        self.corpus = root / "corpus.jsonl"
        self.corpus.write_text('{"prompt": "hello"}\n')
        self.lock = root / "gpu.lock"

        self.server = _write_exec(
            self.bin / "llama-server",
            """
            #!/bin/bash
            echo "llama_kv_cache: KV self size = 4096 MiB"
            echo "server listening"
            sleep 600
            """,
        )
        self.client = _write_exec(
            self.bin / "vllm",
            """
            #!/bin/bash
            # Mirror the client's own contract: it writes --result-filename into
            # --result-dir.  A stub that writes nothing would make every leg
            # look like a client that ran and produced nothing.
            dir=""; name=""; prompts=0; conc=0
            while (($#)); do
              case "$1" in
                --result-dir) dir=$2; shift 2 ;;
                --result-filename) name=$2; shift 2 ;;
                --num-prompts) prompts=$2; shift 2 ;;
                --max-concurrency) conc=$2; shift 2 ;;
                *) shift ;;
              esac
            done
            [ -n "$dir" ] && [ -n "$name" ] && cat > "$dir/$name" <<JSON
            {"completed": ${STUB_COMPLETED:-$prompts}, "num_prompts": $prompts,
             "max_concurrency": $conc,
             "output_throughput": $((conc * 10)).5, "total_token_throughput": $((conc * 90)).5,
             "median_ttft_ms": $((100 + conc)).0, "median_tpot_ms": $((20 + conc)).0,
             "median_itl_ms": $((19 + conc)).0, "median_e2el_ms": $((3000 + conc)).0}
            JSON
            echo "stub client ok"
            """,
        )
        self.timev = _write_exec(
            self.bin / "time",
            """
            #!/bin/bash
            # /usr/bin/time -v -o FILE CMD...
            out=""
            while (($#)); do
              case "$1" in
                -v) shift ;;
                -o) out=$2; shift 2 ;;
                *) break ;;
              esac
            done
            "$@"; rc=$?
            [ -n "$out" ] && echo "	Maximum resident set size (kbytes): 123456" > "$out"
            exit $rc
            """,
        )
        self.curl = _write_exec(
            self.bin / "curl",
            """
            #!/bin/bash
            printf '200'
            """,
        )

    def env(self, **overrides: str) -> dict[str, str]:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.root),
            "REPO_ROOT": str(REPO_ROOT),
            "GPU_LOCK": str(self.lock),
            "MEM_TOTAL_BYTES": str(BOX_BYTES),
            "TIMEV_BIN": str(self.timev),
            "CURL_BIN": str(self.curl),
            "READY_TIMEOUT_SECONDS": "10",
            "READY_POLL_INTERVAL": "1",
            "LEG_TIMEOUT_SECONDS": "60",
            "LOCK_WAIT_SECONDS": "5",
            "SHUTDOWN_GRACE_SECONDS": "5",
            "CLOCK_MAX_SECONDS": "20",
        }
        env.update(overrides)
        return env

    def args(self, mode: str = "--self-check", **overrides: str) -> list[str]:
        values = {
            "--model": str(self.model),
            "--server": str(self.server),
            "--client": str(self.client),
            "--tokenizer": str(self.tokenizer),
            "--corpus": str(self.corpus),
            "--out": str(self.out),
        }
        values.update(overrides)
        argv = ["bash", str(self.script), mode]
        for key, value in values.items():
            if value is not None:
                argv += [key, value]
        return argv

    def run(self, mode: str = "--self-check", env=None, timeout=120, **overrides):
        return subprocess.run(
            self.args(mode, **overrides),
            env=env if env is not None else self.env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )


class LadderHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.fx = Fixture(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # ------------------------------------------------------------- control
    def test_control_self_check_passes(self) -> None:
        """The CONTROL.  Without it every refusal below could be a broken fixture."""
        result = self.fx.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SELF_CHECK_OK", result.stdout)

    def test_control_execute_runs_the_whole_ladder(self) -> None:
        """The execute-path CONTROL: 6 rungs x 3 reps = 18 result files."""
        result = self.fx.run("--execute", timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("SERIES_DONE", result.stdout)
        produced = sorted(p.name for p in self.fx.out.glob("llamacpp-c*-r*.json"))
        self.assertEqual(len(produced), 18, produced)
        for conc in (1, 2, 4, 8, 16, 32):
            for rep in (1, 2, 3):
                self.assertIn(f"llamacpp-c{conc}-r{rep}.json", produced)

    def test_ladder_matches_the_published_online_serving_grid(self) -> None:
        """The two arms must be commensurable, so the ladder is COPIED, not chosen.

        If ``online_gate.POINTS`` moves, this test goes red and the two tables
        stop silently drifting apart.
        """
        gate = (REPO_ROOT / "tools" / "bench" / "online_gate.py").read_text()
        points = re.search(r"^POINTS = (\(\(.*\)\))$", gate, re.M)
        self.assertIsNotNone(points, "online_gate.POINTS not found")
        expected = " ".join(
            f"{c}:{n}" for c, n in re.findall(r"\((\d+), (\d+)\)", points.group(1))
        )
        script = SCRIPT.read_text()
        ladder = re.search(r'^readonly LADDER="([^"]+)"$', script, re.M)
        self.assertIsNotNone(ladder)
        self.assertEqual(ladder.group(1), expected)
        for name, value in (("INPUT_LEN", 1024), ("OUTPUT_LEN", 128)):
            self.assertIsNotNone(
                re.search(rf"^readonly {name}={value}$", script, re.M), name
            )
            self.assertIsNotNone(re.search(rf"^{name} = {value}$", gate, re.M), name)
        # The spec's binding minimum, restated as an assertion.
        for required in ("1:", "4:", "8:", "16:", "32:"):
            self.assertIn(required, ladder.group(1))

    # ------------------------------------- 1. `set -u` and every $(( )) operand
    def test_set_u_no_unbound_variable_in_an_empty_environment(self) -> None:
        """DYNAMIC half: with nothing inherited, no operand may be unbound.

        The refusal has to be a NAMED code.  ``set -u`` reports an unbound
        variable and exits 1, so an exit of 1 here is the defect, not the guard.
        """
        bare = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        result = self.fx.run(env=bare)
        self.assertNotIn("unbound variable", result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_set_u_static_every_arithmetic_operand_is_defaulted(self) -> None:
        """STATIC half: name every identifier inside ``$(( ))`` and prove it is bound.

        ``local a=$1 b=$((a))`` is the shape that broke a bench instrument here:
        ``local`` binds the names left to right, so the arithmetic reads a name
        the shell has not assigned yet.  Scanning for the operands and demanding
        each be assigned catches the class rather than that one instance.
        """
        script = SCRIPT.read_text()
        operands = _bare_arithmetic_operands(script)
        self.assertGreater(len(operands), 8, "the operand scan found almost nothing")
        unbound = []
        for name in sorted(operands):
            bound = (
                re.search(rf"^{name}=\$\{{{name}:-", script, re.M)
                or re.search(rf"^readonly {name}=", script, re.M)
                or re.search(rf"^\s*local .*\b{name}\b", script, re.M)
                or re.search(rf"^\s*{name}=", script, re.M)
                or re.search(rf"^\s*for \(\(\s*{name}\s*=", script, re.M)
            )
            if not bound:
                unbound.append(name)
        self.assertEqual(unbound, [], f"arithmetic operands with no binding: {unbound}")

    def test_set_u_mutation_removing_one_default_goes_red(self) -> None:
        """MUTATION: the two checks above are worthless if they cannot fail.

        Strip the default from ``MEM_FRACTION`` -- one of the operands the static
        scan reports -- and both halves must notice.
        """
        mutant_dir = self.root / "mutant"
        mutant_dir.mkdir()
        mutant = mutant_dir / "ladder.sh"
        original = SCRIPT.read_text()
        mutated = original.replace(
            "MEM_FRACTION=${MEM_FRACTION:-85}", "# MEM_FRACTION default deleted"
        )
        self.assertNotEqual(mutated, original, "the mutation did not apply")
        mutant.write_text(mutated)

        fx = Fixture(self.root / "mutant-fx", script=mutant)
        # REPO_ROOT is supplied because it is NOT what is under mutation: without
        # it the wrapper guard refuses first and the mutation is never reached,
        # which would read as a passing test that proved nothing.
        bare = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "REPO_ROOT": str(REPO_ROOT),
        }
        result = fx.run(env=bare)
        self.assertIn("unbound variable", result.stderr)
        self.assertNotEqual(result.returncode, 0)

        self.assertIn("MEM_FRACTION", _bare_arithmetic_operands(mutated))
        self.assertIsNone(re.search(r"^MEM_FRACTION=\$\{MEM_FRACTION:-", mutated, re.M))

    # ------------------------------- 2. KV / context sizing vs REAL box memory
    def test_kv_refuses_when_weights_plus_kv_exceed_the_box(self) -> None:
        """The 128-GiB-on-a-119-GB-box defect, as an assertion."""
        result = self.fx.run(
            env=self.fx.env(MEM_TOTAL_BYTES=str(4 * (1 << 30)), KV_BYTES_PER_TOKEN="65536")
        )
        self.assertEqual(result.returncode, E_MEM_BUDGET, result.stdout + result.stderr)
        self.assertIn("exceeds", result.stderr)

    def test_kv_accepts_a_sizing_that_fits(self) -> None:
        """Control for the two refusals: the same guard must also pass."""
        result = self.fx.run(env=self.fx.env(KV_BYTES_PER_TOKEN="65536"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_kv_refuses_a_context_that_cannot_hold_the_slots(self) -> None:
        """``-c`` is divided by ``-np``: 32 slots of 1152 tokens need 36,864."""
        result = self.fx.run(**{"--ctx-total": "1024"})
        self.assertEqual(result.returncode, E_MEM_BUDGET, result.stdout + result.stderr)
        self.assertIn("divides -c by -np", result.stderr)

    def test_kv_sizes_against_the_boxs_actual_memory_not_a_constant(self) -> None:
        """No hard-coded ceiling: the source of the number is named and read."""
        result = self.fx.run(env={k: v for k, v in self.fx.env().items()
                                  if k != "MEM_TOTAL_BYTES"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertRegex(result.stdout, r"memory total\s+\d+ bytes \(source: (proc-meminfo|nvidia-smi)\)")
        with open("/proc/meminfo") as handle:
            kb = int(re.search(r"MemTotal:\s+(\d+)", handle.read()).group(1))
        self.assertIn(str(kb * 1024), result.stdout)

    def test_kv_the_binding_check_is_the_servers_own_reported_size(self) -> None:
        """The guard that actually binds fires BEFORE any leg runs.

        A static formula can be wrong.  The engine's own allocation cannot be.
        """
        _write_exec(
            self.fx.server,
            """
            #!/bin/bash
            echo "llama_kv_cache: KV self size = 900000 MiB"
            sleep 600
            """,
        )
        result = self.fx.run("--execute", timeout=180)
        self.assertEqual(result.returncode, E_KV_OVER_BUDGET, result.stdout + result.stderr)
        self.assertEqual(list(self.fx.out.glob("llamacpp-c*.json")), [],
                         "a leg ran after the memory refusal")

    # ------------------------------- 3. a missing artifact fails LOUDLY
    def test_artifact_a_missing_shard_refuses_by_name(self) -> None:
        self.fx.shards[1].unlink()
        result = self.fx.run("--execute", timeout=120)
        self.assertEqual(result.returncode, E_MODEL_MISSING, result.stdout + result.stderr)
        self.assertIn(self.fx.shards[1].name, result.stderr)
        self.assertIn("truncated shard measures a truncated file", result.stderr)
        self.assertFalse(self.fx.out.exists(), "an evidence directory was created anyway")

    def test_artifact_a_zero_byte_shard_refuses(self) -> None:
        """A 10 MiB fragment of a 67 GiB shard once read as a present file."""
        self.fx.shards[2].write_bytes(b"")
        result = self.fx.run()
        self.assertEqual(result.returncode, E_MODEL_MISSING, result.stdout + result.stderr)
        self.assertIn(self.fx.shards[2].name, result.stderr)

    def test_artifact_shard_set_comes_from_the_name_not_from_a_glob(self) -> None:
        """``-00001-of-00003`` names three files; a glob would find only what is there."""
        for shard in self.fx.shards[1:]:
            shard.unlink()
        result = self.fx.run()
        self.assertEqual(result.returncode, E_MODEL_MISSING)
        self.assertIn("2 shard(s)", result.stderr)

    def test_artifact_missing_corpus_and_tokenizer_refuse_distinctly(self) -> None:
        self.fx.corpus.write_text("")
        self.assertEqual(self.fx.run().returncode, E_CORPUS_MISSING)
        self.fx.corpus.write_text('{"prompt": "hello"}\n')
        shutil.rmtree(self.fx.tokenizer)
        self.assertEqual(self.fx.run().returncode, E_CORPUS_MISSING)

    def test_artifact_missing_server_and_client_refuse_distinctly(self) -> None:
        """Two different absences must not share one exit code."""
        result = self.fx.run(**{"--server": str(self.root / "nope")})
        self.assertEqual(result.returncode, E_SERVER_MISSING, result.stderr)
        result = self.fx.run(**{"--client": str(self.root / "nope")})
        self.assertEqual(result.returncode, E_CLIENT_MISSING, result.stderr)
        self.assertIn("without it there is no measurement", result.stderr)

    # ------------------------------- 4. a missing WRAPPER cannot mute the leg
    def test_wrapper_absent_time_refuses_before_the_lock(self) -> None:
        result = self.fx.run(env=self.fx.env(TIMEV_BIN=str(self.root / "no-such-time")))
        self.assertEqual(result.returncode, E_WRAPPER_MISSING, result.stdout + result.stderr)
        self.assertIn("not run at all", result.stderr)

    def test_wrapper_absent_from_path_refuses(self) -> None:
        """A bare name is checked with ``command -v``, not assumed."""
        result = self.fx.run(env=self.fx.env(TIMEOUT_BIN="definitely-not-a-binary"))
        self.assertEqual(result.returncode, E_WRAPPER_MISSING, result.stderr)
        result = self.fx.run(env=self.fx.env(FLOCK_BIN="definitely-not-a-binary"))
        self.assertEqual(result.returncode, E_WRAPPER_MISSING, result.stderr)

    def test_wrapper_present_but_silent_fails_the_leg(self) -> None:
        """The second half: a wrapper that RUNS and writes nothing is also a defect.

        Preflight cannot see this one, so the leg asserts the wrapper's own
        output after the fact.  Without this the memory axis silently vanishes
        and the ladder still reports throughput.
        """
        _write_exec(
            self.fx.timev,
            """
            #!/bin/bash
            out=""
            while (($#)); do
              case "$1" in
                -v) shift ;;
                -o) out=$2; shift 2 ;;
                *) break ;;
              esac
            done
            "$@"; rc=$?
            : > "$out"          # ran, wrote an EMPTY report
            exit $rc
            """,
        )
        result = self.fx.run("--execute", timeout=180)
        self.assertEqual(result.returncode, E_LEG_FAILED, result.stdout + result.stderr)
        self.assertIn("carries no rusage", result.stdout)

    def test_wrapper_a_client_that_writes_no_result_fails_the_leg(self) -> None:
        """The third post-leg assertion, and it needed its own test.

        A mutation sweep found this gap: deleting the result-file check left the
        control test GREEN, because the control asserts the files EXIST and the
        stub writes them either way. Only a client that exits 0 and writes
        nothing discriminates -- which is precisely the shape a wrapper failure,
        a full disk or a mis-parsed --result-dir produces in the field.
        """
        _write_exec(self.fx.client, """
            #!/bin/bash
            echo "ran, wrote nothing"
            exit 0
            """)
        result = self.fx.run("--execute", timeout=180)
        self.assertEqual(result.returncode, E_LEG_FAILED, result.stdout + result.stderr)
        self.assertIn("the client wrote no result file", result.stdout)

    def test_wrapper_the_repo_helpers_are_required_not_reimplemented(self) -> None:
        """``.agents/benchmarking.md``: a new harness IMPORTS the clock helper."""
        result = self.fx.run(env=self.fx.env(REPO_ROOT=str(self.root)))
        self.assertEqual(result.returncode, E_WRAPPER_MISSING, result.stderr)
        self.assertIn("gpu_clock_state.py", result.stderr)

    # ------------------------------- the summary is the only publication path
    def test_summary_renders_one_row_per_rung(self) -> None:
        """A human reading JSON files is not a publication path."""
        run = self.fx.run("--execute", timeout=300)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        result = self.fx.run("--summarize", timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        table = (self.fx.out / "ladder-summary.md").read_text()
        for conc in (1, 2, 4, 8, 16, 32):
            self.assertIsNotNone(
                re.search(rf"^\| {conc} \| \d+ \|", table, re.M), f"no row for c={conc}"
            )
        # Prefill and decode are SEPARATE columns; the gate asks for that.
        self.assertIn("median TTFT ms (prefill)", table)
        self.assertIn("median TPOT ms (decode)", table)
        self.assertIn("peak RSS MiB", table)
        self.assertIn("SM clock med MHz", table)
        # A denominator table must not read as a result.
        self.assertIn("DENOMINATOR, not a result", table)
        self.assertIn("mask-only", table)

    def test_summary_refuses_when_a_leg_is_missing(self) -> None:
        """A missing leg must not become a quietly narrower table."""
        run = self.fx.run("--execute", timeout=300)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        (self.fx.out / "llamacpp-c32-r2.json").unlink()
        result = self.fx.run("--summarize", timeout=120)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SUMMARY_INCOMPLETE", result.stderr)
        self.assertIn("c32 rep2", result.stderr)

    def test_summary_refuses_when_requests_did_not_complete(self) -> None:
        """`failed=0` is an axis: a rung that dropped requests is not a rung."""
        run = self.fx.run("--execute", timeout=300, env=self.fx.env(STUB_COMPLETED="1"))
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        result = self.fx.run("--summarize", timeout=120)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("completed=1 of", result.stderr)

    def test_summary_refuses_an_out_directory_that_does_not_exist(self) -> None:
        result = self.fx.run("--summarize", timeout=60,
                             **{"--out": str(self.root / "nope")})
        self.assertEqual(result.returncode, E_USAGE, result.stderr)


    # ------------------------------- 5. it never hangs holding the GPU
    def test_hang_server_that_never_answers_health_times_out(self) -> None:
        _write_exec(self.fx.curl, """
            #!/bin/bash
            printf '000'
            """)
        started = time.monotonic()
        result = self.fx.run("--execute", timeout=120,
                             env=self.fx.env(READY_TIMEOUT_SECONDS="6", READY_POLL_INTERVAL="1"))
        elapsed = time.monotonic() - started
        self.assertEqual(result.returncode, E_SERVER_NOT_READY, result.stdout + result.stderr)
        self.assertLess(elapsed, 60, "the bounded readiness poll did not bound anything")
        self.assertIn("refusing to wait on a held lock", result.stderr)

    def test_hang_server_that_dies_is_noticed_immediately(self) -> None:
        """Not a timeout: a dead server must be reported as dead, not waited on."""
        _write_exec(self.fx.server, """
            #!/bin/bash
            echo "boom"
            exit 3
            """)
        _write_exec(self.fx.curl, """
            #!/bin/bash
            printf '000'
            """)
        started = time.monotonic()
        result = self.fx.run("--execute", timeout=120,
                             env=self.fx.env(READY_TIMEOUT_SECONDS="600"))
        elapsed = time.monotonic() - started
        self.assertEqual(result.returncode, E_SERVER_NOT_READY, result.stdout + result.stderr)
        self.assertIn("exited before it answered", result.stderr)
        self.assertLess(elapsed, 60)

    def test_hang_lock_contention_is_bounded(self) -> None:
        """Another holder must produce a refusal, never an unbounded wait."""
        holder = subprocess.Popen(
            ["flock", "-x", str(self.fx.lock), "-c", "sleep 45"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(1)
            started = time.monotonic()
            result = self.fx.run("--execute", timeout=120,
                                 env=self.fx.env(LOCK_WAIT_SECONDS="3"))
            elapsed = time.monotonic() - started
            self.assertEqual(result.returncode, E_LOCK_TIMEOUT, result.stdout + result.stderr)
            self.assertLess(elapsed, 40, "flock -w did not bound the wait")
        finally:
            holder.kill()
            holder.wait()

    def test_hang_a_stuck_leg_is_killed_by_the_leg_timeout(self) -> None:
        _write_exec(self.fx.client, """
            #!/bin/bash
            sleep 600
            """)
        started = time.monotonic()
        result = self.fx.run("--execute", timeout=180,
                             env=self.fx.env(LEG_TIMEOUT_SECONDS="3"))
        elapsed = time.monotonic() - started
        self.assertEqual(result.returncode, E_LEG_FAILED, result.stdout + result.stderr)
        self.assertIn("124", result.stdout)
        self.assertLess(elapsed, 90)

    def test_hang_the_server_is_reaped_on_every_exit_path(self) -> None:
        """A refusal that leaves llama-server running holds the whole box."""
        marker = self.root / "server-alive"
        _write_exec(self.fx.server, f"""
            #!/bin/bash
            echo "llama_kv_cache: KV self size = 900000 MiB"
            touch {marker}
            trap 'rm -f {marker}; exit 0' TERM INT
            sleep 600 &
            wait
            """)
        result = self.fx.run("--execute", timeout=180)
        self.assertEqual(result.returncode, E_KV_OVER_BUDGET, result.stdout + result.stderr)
        deadline = time.monotonic() + 15
        while marker.exists() and time.monotonic() < deadline:
            time.sleep(0.5)
        self.assertFalse(marker.exists(), "llama-server outlived the refusal")

    def test_hang_no_clock_is_ever_pinned(self) -> None:
        """A clock pin outlives the lease and reprices the next holder's run.

        Inside a lease ``-lgc`` returns LGC_RC=4 anyway, so the only safe shape
        is a harness with no pinning code path at all.
        """
        code = _code_only(SCRIPT.read_text())
        self.assertIn("nvidia-smi", SCRIPT.read_text(), "the scan target moved")
        for forbidden in ("-lgc", "-rgc", "--lock-gpu-clocks", "--reset-gpu-clocks"):
            self.assertNotIn(forbidden, code, f"the harness can pin a clock: {forbidden}")

    def test_hang_every_wait_in_the_script_is_bounded(self) -> None:
        """Static: no ``flock`` without ``-w``, no ``while :`` without a bound."""
        code = _code_only(SCRIPT.read_text())
        # Only INVOCATIONS: a line whose first word is the flock binary.  The
        # `for bin in ... "${FLOCK_BIN}" ...` preflight list is not a wait.
        invocations = re.findall(r'^\s*"\$\{FLOCK_BIN\}"([^\n]*)', code, re.M)
        self.assertEqual(len(invocations), 1, invocations)
        for tail in invocations:
            self.assertIn("-w ", tail, "flock without -w is an unbounded wait")
        self.assertIn('"${TIMEOUT_BIN}" "${LEG_TIMEOUT_SECONDS}"', code)
        self.assertIn("READY_TIMEOUT_SECONDS", code)


if __name__ == "__main__":
    unittest.main()

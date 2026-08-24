#!/usr/bin/env python3
"""ONE LEG of the tower-skip RSS harness, against a fake server.

`tests/scripts/test_tower_skip_rss_report.py` gates the REPORTER: it reads four
finished `/usr/bin/time -v` files and four finished logs and decides a verdict.
It cannot see how those files came to exist. This suite gates the half that
MAKES them -- `run_arm`, its readiness poll and its teardown -- which until now
ran only on a leased box with a checkpoint and was covered by nothing at all.

That gap is not hypothetical. The first real run of the harness
([#1844](https://github.com/mudler/vllm.cpp/issues/1844), `thor:gpu0`,
`d60692c8`) staged 8887294190 B of checkpoint, built two sha256-identical
binaries, and then produced **five 0-byte `.time` files** and a VOID verdict on
both pairs, while this file's sibling suite stayed 60/60 green. Two defects, one
shape -- *the harness could not tell its own process from somebody else's*:

* **Readiness was not attributable.** `run_arm` polled `/health` on a FIXED port
  immediately after launching, so the poll was answered by the PREVIOUS leg's
  server. Every measured leg was declared ready before it had read a tensor and
  was killed mid-load. `warmup` is the control: nothing was listening before it,
  and it alone reached `listening on http://0.0.0.0:18607`.
* **The teardown killed the INSTRUMENT.** `kill "$pid"` signals
  `/usr/bin/time`, not the server. GNU time installs no handler, so the timer
  died before writing its summary and the server was reparented to init and
  KEPT THE PORT -- which is what the previous leg's server was still doing when
  it answered the next leg's poll. Measured here on this box:
  `/usr/bin/time -v -o f sleep 100 & kill $!` leaves `f` at 0 bytes and `sleep`
  alive with ppid 1; signalling the CHILD instead leaves `f` at 752 bytes with
  a `Maximum resident set size` line.

Every case below is behavioural and runs on a plain host: a fake server that
prints a configurable banner after a configurable delay and then answers
`/health`, driven through the harness's own `run_arm` by sourcing the script
with `TOWER_SKIP_RSS_SOURCE_ONLY=1`. No checkpoint, no build, no GPU, no lease.

Each case fails against the pre-#1844 `run_arm`, and the two mutations at the
bottom put that on the record from the other direction: restoring either half
of the defect in a scratch copy turns this suite red.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/mm/tower_skip_rss.sh"

# `run_arm`'s refusal, which the driver below propagates.
EXIT_ARM = 5

# A leg that cannot become ready must fail LOUDLY and quickly. The driver runs
# with second-scale bounds, so anything that reaches this ceiling has hung --
# and the pre-#1844 loop, which is 900 iterations of `sleep 2` with no way to
# shorten it, is exactly what reaching it looks like.
DRIVER_TIMEOUT_S = 90

# A stand-in for `vllm-server`: it takes the same four flags `arm_command`
# passes, prints the same two lines the real server prints around its load
# (`server_main.cpp:1704` for the banner), and answers `/health` with 200.
# Everything the cases vary is an environment variable, so the argv stays the
# argv the harness builds.
FAKE_SERVER = r'''#!/usr/bin/env python3
"""A fake `vllm-server` for the leg harness. Loads nothing, serves /health."""

import argparse
import http.server
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="")
parser.add_argument("--port", type=int, default=0)
parser.add_argument("--device", default="cpu")
parser.add_argument("--language-model-only", action="store_true")
args, _ = parser.parse_known_args()

print("server: loading model from %s" % args.model, file=sys.stderr, flush=True)
time.sleep(float(os.environ.get("FAKE_LOAD_SECONDS", "1")))

if os.environ.get("FAKE_DIE_DURING_LOAD") == "1":
    print("server: the load failed (fake)", file=sys.stderr, flush=True)
    sys.exit(3)

# Enough resident bytes that the instrument's figure is this process and not
# the interpreter's floor. Touched, because untouched pages are not resident.
MB = int(os.environ.get("FAKE_RSS_MB", "48"))
BLOB = bytearray(MB * 1024 * 1024)
for off in range(0, len(BLOB), 4096):
    BLOB[off] = 1


class Handler(http.server.BaseHTTPRequestHandler):
    def reply(self, ok, body):
        self.send_response(200 if ok else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self.reply(self.path == "/health", b'{"status":"ok"}')

    def do_POST(self):  # noqa: N802
        # The muse-glimmer vehicle's workload. Read the body first, or the
        # client sees a reset instead of the answer.
        self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        self.reply(
            self.path == "/v1/completions",
            b'{"choices":[{"text":" Paris."}]}',
        )

    def log_message(self, *unused):
        pass


httpd = http.server.HTTPServer(("127.0.0.1", args.port), Handler)

# The real server prints this AFTER the load and just before it serves. A leg
# whose banner never appears is the case that separates "this leg is ready"
# from "something on this port is ready".
if os.environ.get("FAKE_BANNER", "1") == "1":
    print(
        "server: listening on http://0.0.0.0:%d (model 'fake', HTTP worker pool 1 fixed)"
        % args.port,
        file=sys.stderr,
        flush=True,
    )
httpd.serve_forever()
'''

# The harness's own `run_arm`, driven directly. Everything a leased run resolves
# from a checkpoint and two build trees is supplied by the caller instead;
# nothing here reimplements a step, because a transcription cannot gate the
# function it transcribes.
DRIVER = r'''#!/usr/bin/env bash
# $1 script to source, $2 run directory, $3 port, then (binary, tag) pairs.
set -u
script=$1; out=$2; port=$3
shift 3
legs=("$@")
# Sourced DELIBERATELY without clearing `$@`: a sourced file inherits its
# caller's positional parameters, and the seam has to survive that rather than
# require every caller to remember it.
TOWER_SKIP_RSS_SOURCE_ONLY=1 . "$script"
OUT=$out
PORT=$port
CHECKPOINT=${LEG_CHECKPOINT:-/nonexistent/fake-checkpoint}
WORKLOAD=${LEG_WORKLOAD:-load-only}
SERVED_MODEL_NAME=fake
i=0
while [ "$i" -lt "${#legs[@]}" ]; do
  bin=${legs[$i]}; tag=${legs[$((i + 1))]}; i=$((i + 2))
  if run_arm "$bin" "$tag"; then
    echo "LEG $tag OK"
  else
    rc=$?
    echo "LEG $tag FAILED rc=$rc"
    exit "$rc"
  fi
done
echo "ALL LEGS OK"
exit 0
'''


def a_free_port() -> int:
    """A port nothing is listening on right now."""

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def accepting(port: int) -> bool:
    """Is anything accepting connections on this port? The harness's question."""

    with socket.socket() as probe:
        probe.settimeout(2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


class LegHarness(unittest.TestCase):
    """A scratch run directory, a fake server binary, and a private port."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="tower-skip-arm-"))
        # Reaping comes FIRST so it runs LAST: a case that leaves an orphan --
        # which is precisely what the defect does -- must not leak it into the
        # next case, and the tree must not be removed while a process still
        # holds its binary.
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(self.reap)
        self.out = self.tmp / "out"
        self.out.mkdir()
        self.server = self.tmp / "fake-vllm-server"
        self.server.write_text(FAKE_SERVER, encoding="utf-8")
        self.server.chmod(0o755)
        self.driver = self.tmp / "drive-leg.sh"
        self.driver.write_text(DRIVER, encoding="utf-8")
        self.port = a_free_port()

    def reap(self) -> None:
        """Kill anything still running this case's OWN fake server.

        Matched on the temporary directory's path, which no other process on
        the box can contain -- never on a pattern this test's own command line
        could match.
        """

        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes()
            except OSError:
                continue
            if bytes(str(self.server), "utf-8") in cmdline:
                try:
                    os.kill(int(entry.name), signal.SIGKILL)
                except OSError:
                    pass

    def start_stale_listener(self, load_seconds: str = "0") -> subprocess.Popen:
        """A server on the leg's port that this leg did NOT start."""

        env = dict(os.environ, FAKE_LOAD_SECONDS=load_seconds)
        proc = subprocess.Popen(
            [sys.executable, str(self.server), "--port", str(self.port)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(proc.kill)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if accepting(self.port):
                return proc
            time.sleep(0.2)
        self.fail("the stale listener never came up; the case would prove nothing")
        raise AssertionError  # unreachable, for the type checker

    def run_legs(
        self,
        *tags: str,
        script: Path | None = None,
        env_extra: dict | None = None,
        timeout: int = DRIVER_TIMEOUT_S,
    ) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.update(
            {
                "ARM_READY_TIMEOUT_S": "30",
                "ARM_STOP_TIMEOUT_S": "15",
                "ARM_PORT_FREE_TIMEOUT_S": "15",
                "FAKE_LOAD_SECONDS": "2",
                "LEG_WORKLOAD": "load-only",
            }
        )
        env.update(env_extra or {})
        argv = [
            "bash",
            str(self.driver),
            str(script or SCRIPT),
            str(self.out),
            str(self.port),
        ]
        for tag in tags:
            argv += [str(self.server), tag]
        try:
            return subprocess.run(
                argv, env=env, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired as expired:
            self.fail(
                "the leg did not return within %ds. A leg that cannot become "
                "ready must REFUSE, loudly and quickly; the pre-#1844 poll is "
                "900 iterations of `sleep 2` and reaching this ceiling is what "
                "that looks like.\nstdout: %r\nstderr: %r"
                % (timeout, expired.stdout, expired.stderr)
            )
            raise

    def log(self, tag: str) -> str:
        path = self.out / ("%s.log" % tag)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def time_file(self, tag: str) -> Path:
        return self.out / ("%s.time" % tag)

    def assert_instrument_wrote_a_figure(self, tag: str) -> int:
        """The `.time` file exists, is not empty, and carries a peak RSS."""

        path = self.time_file(tag)
        self.assertTrue(path.exists(), "no %s" % path)
        size = path.stat().st_size
        self.assertGreater(
            size,
            0,
            "%s is 0 bytes. /usr/bin/time was torn down before it wrote its "
            "summary, which is the half of #1844 that made all five .time "
            "files empty on the real run." % path,
        )
        text = path.read_text(encoding="utf-8")
        match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
        self.assertIsNotNone(
            match,
            "no 'Maximum resident set size' line in %s; the reporter would "
            "call the pair VOID.\n%s" % (path, text),
        )
        kbytes = int(match.group(1))
        self.assertGreater(kbytes, 0)
        return kbytes


class OccupiedPortTests(LegHarness):
    """A leg starts into a free port, or it does not start.

    The pre-#1844 `run_arm` launched into whatever was there and then asked the
    port whether it was ready. The port answered yes -- about the other server.
    """

    def test_a_leg_refuses_to_start_into_an_occupied_port(self) -> None:
        self.start_stale_listener()
        out = self.run_legs("default")
        self.assertEqual(
            out.returncode,
            EXIT_ARM,
            "a leg launched into an occupied port and reported success.\n"
            "stdout: %s\nstderr: %s" % (out.stdout, out.stderr),
        )
        self.assertIn(str(self.port), out.stderr)
        self.assertIn("accepting", out.stderr)

    def test_the_refusal_happens_BEFORE_the_server_is_launched(self) -> None:
        """No log, no `.time`, no half-measured leg to reason about later."""

        self.start_stale_listener()
        out = self.run_legs("default")
        self.assertNotEqual(out.returncode, 0)
        self.assertFalse(
            self.time_file("default").exists(),
            "the leg ran the instrument against a port it could not own",
        )


class ReadinessAttributionTests(LegHarness):
    """Ready means THIS leg's process said so, in THIS leg's own log."""

    def test_a_ready_leg_reports_its_own_banner(self) -> None:
        out = self.run_legs("default")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("listening on http://", self.log("default"))

    def test_a_server_that_never_prints_a_banner_is_never_ready(self) -> None:
        """It listens and answers `/health`, and that is not enough.

        This is the property, isolated: a leg is ready when the process the leg
        started says it is serving. An answer from the port alone cannot say
        WHICH process answered, which is the whole of #1844.
        """

        out = self.run_legs("default", env_extra={"FAKE_BANNER": "0"})
        self.assertEqual(
            out.returncode,
            EXIT_ARM,
            "a leg with no banner in its own log was called ready.\n"
            "stdout: %s\nstderr: %s" % (out.stdout, out.stderr),
        )
        self.assertNotIn("listening on http://", self.log("default"))
        self.assertIn("listening on http://", out.stderr)

    def test_a_server_that_dies_during_load_is_reported_not_awaited(self) -> None:
        started = time.monotonic()
        out = self.run_legs("default", env_extra={"FAKE_DIE_DURING_LOAD": "1"})
        elapsed = time.monotonic() - started
        self.assertEqual(out.returncode, EXIT_ARM, out.stdout + out.stderr)
        self.assertLess(
            elapsed,
            25,
            "the leg waited out its readiness bound for a server that had "
            "already exited. This one property the pre-#1844 loop also had, and "
            "it is kept because the readiness rewrite could easily have lost "
            "it: the check is now on the TIMER's /proc state rather than on "
            "`kill -0`, and a leg that hangs for 1800s on a server that is gone "
            "burns a lease to report nothing.",
        )
        self.assertIn("never became ready", out.stderr)


class InstrumentTests(LegHarness):
    """A leg that became ready leaves a figure behind. That is the point."""

    def test_a_ready_leg_leaves_a_time_file_with_a_peak_rss(self) -> None:
        out = self.run_legs("default")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        kbytes = self.assert_instrument_wrote_a_figure("default")
        # The fake touches 48 MiB, so the figure is this process rather than a
        # value the timer inherited from somewhere.
        self.assertGreater(kbytes, 40 * 1024)

    def test_the_leg_leaves_nothing_listening_and_no_orphan(self) -> None:
        out = self.run_legs("default")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertFalse(
            accepting(self.port),
            "the leg returned while something was still accepting on port %d. "
            "Signalling `/usr/bin/time` reparents the server to init and it "
            "keeps the socket; the next leg's poll is then answered by this "
            "one." % self.port,
        )

    def test_the_completion_workload_still_runs_and_is_measured(self) -> None:
        """The muse-glimmer vehicle's arm, which is not readiness-only."""

        out = self.run_legs("default", env_extra={"LEG_WORKLOAD": "completion"})
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assert_instrument_wrote_a_figure("default")


class SequentialLegTests(LegHarness):
    """#1844 as it was observed: leg 2 onwards, on one fixed port."""

    def test_the_second_leg_is_not_answered_by_the_first(self) -> None:
        out = self.run_legs("default", "lmo")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        for tag in ("default", "lmo"):
            self.assertIn(
                "listening on http://",
                self.log(tag),
                "leg %s never reached its own banner -- the shape of the real "
                "run, where every measured leg stopped at `loading model "
                "from`" % tag,
            )
            self.assert_instrument_wrote_a_figure(tag)

    def test_five_legs_in_the_declared_order_all_produce_figures(self) -> None:
        """`ARM_PLAN`'s shape: a warmup and two pairs, one port, in sequence."""

        tags = ("warmup", "default", "lmo", "default2", "lmo2")
        out = self.run_legs(*tags, timeout=180)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        for tag in tags:
            self.assert_instrument_wrote_a_figure(tag)


class MutationTests(LegHarness):
    """Restore either half of the defect in a scratch copy; this suite reds.

    A test that has never been seen failing for the reason it names has proven
    nothing, and both halves of #1844 are one line each.
    """

    def script_with(self, old: str, new: str) -> Path:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertEqual(
            text.count(old), 1, "mutation anchor is not unique: %r" % old
        )
        copy = self.tmp / "mutated.sh"
        copy.write_text(text.replace(old, new, 1), encoding="utf-8")
        return copy

    def test_signalling_the_timer_instead_of_the_server_is_caught(self) -> None:
        """The teardown half: `kill "$pid"` kills `/usr/bin/time`."""

        mutated = self.script_with(
            '    kill -TERM "$srv" 2>/dev/null',
            '    kill -TERM "$pid" 2>/dev/null  # MUTATION: signal the timer',
        )
        out = self.run_legs("default", script=mutated)
        empty = self.time_file("default").stat().st_size == 0
        self.assertTrue(
            empty or out.returncode != 0,
            "the mutation signalled the instrument and the leg still reported "
            "a figure. This suite would not have caught #1844.\nstdout: %s\n"
            "stderr: %s" % (out.stdout, out.stderr),
        )

    def test_dropping_the_occupied_port_refusal_is_caught(self) -> None:
        """The readiness half: start into whatever is already there."""

        mutated = self.script_with(
            "  if port_is_accepting; then",
            "  if false; then  # MUTATION: start into an occupied port",
        )
        self.start_stale_listener()
        out = self.run_legs("default", script=mutated)
        self.assertNotEqual(
            out.returncode,
            0,
            "the mutation launched a leg into an occupied port and the leg "
            "reported success.\nstdout: %s\nstderr: %s" % (out.stdout, out.stderr),
        )


class SeamTests(unittest.TestCase):
    """The source-only seam is what makes every case above possible."""

    def test_sourcing_the_script_runs_nothing(self) -> None:
        """And it survives the caller's own arguments, which it inherits."""

        out = subprocess.run(
            [
                "bash",
                "-c",
                'TOWER_SKIP_RSS_SOURCE_ONLY=1 . "$0" && '
                "declare -F run_arm stop_arm wait_for_arm_ready port_is_accepting",
                str(SCRIPT),
                "--an-argument-the-harness-would-refuse",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        for name in ("run_arm", "stop_arm", "wait_for_arm_ready", "port_is_accepting"):
            self.assertIn(name, out.stdout)

    def test_the_seam_is_the_last_thing_before_the_first_statement(self) -> None:
        """A definition added below the guard would be silently unreachable."""

        text = SCRIPT.read_text(encoding="utf-8")
        guard = text.index('if [ "${TOWER_SKIP_RSS_SOURCE_ONLY:-0}" = 1 ]; then')
        after = text[guard:]
        self.assertNotIn("\nrun_arm() {", after)
        self.assertNotIn("\nstop_arm() {", after)


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False, verbosity=2).result.wasSuccessful() else 1)

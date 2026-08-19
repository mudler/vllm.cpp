#!/usr/bin/env python3
"""The tooling wave resolves its machine values or refuses, and keeps its records.

Issue #1190, row `ENV-AGNOSTIC-W1-TOOLING`, spec
`.agents/specs/env-agnostic-w1-tooling.md`.

`scripts/` and `tools/` carried one operator's host, oracle venv, checkout and
CUTLASS path as HARD-CODED DEFAULTS. A second developer running any of them did
not get an error; they got a silently wrong answer:

  - `tools/parity/dump_{gdn,moe,qwen3_5_mtp}.py` defaulted `--pin` to
    `/home/mudler/_git/vllm`, which is the tree the ORACLE MATH executes from.
    A wrong tree there dumps goldens from the wrong source and writes the wrong
    provenance into the manifest.
  - `scripts/regen-triton-aot.sh` defaulted CUTLASS to `~/cutlass_probe` — the
    very default the campaign's worked example removed from
    `scripts/dgx-bringup.sh` — while REGENERATING artifacts that get committed.
    `.agents/environment.md:388-400` measures a configure that misses CUTLASS as
    moving the SACRED `test_qwen27_paged_engine` from 235/235 to 234/235 with
    the source untouched, so a stale default is a false green.
  - `scripts/laguna_longctx_bench.sh` defaulted its checkpoint and its binary
    into one operator's `$HOME` layout.

The other half of this suite guards the SECOND class. The campaign's rule keeps
a dated measurement's host literal, because rewriting it falsifies a record.
`ProvenanceTests` pins those sentences, so a later blind sweep of the tree goes
red instead of quietly corrupting the evidence. Every other assertion here would
pass a `sed` that converted everything.

No case in this file needs torch, numpy, cmake or a GPU: the dumpers run under
stub modules, and each refusal is asserted BEFORE the step that would need real
hardware.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DUMPERS = {
    ROOT / "tools/parity/dump_gdn.py": "--pin",
    ROOT / "tools/parity/dump_moe.py": "--pin",
    ROOT / "tools/parity/dump_qwen3_5_mtp.py": "--pinned-vllm",
}
REGEN = ROOT / "scripts/regen-triton-aot.sh"
LAGUNA = ROOT / "scripts/laguna_longctx_bench.sh"

# Every file this row converts. Read as a set so a new resolved literal in any of
# them is caught, not only in the ones a case names.
CONVERTED = (
    "scripts/check-triton-aot-drift.sh",
    "scripts/regen-triton-aot.sh",
    "scripts/laguna_longctx_bench.sh",
    "scripts/glm4-neartie-gap.py",
    "scripts/internlm2-neartie-gap.py",
    "scripts/qwen3-neartie-gap.py",
    "tools/parity/README.md",
    "tools/parity/dump_gdn.py",
    "tools/parity/dump_moe.py",
    "tools/parity/dump_qwen36.py",
    "tools/parity/dump_qwen3_5_mtp.py",
    "tools/parity/dump_tokenizer.py",
    "tools/parity/dump_tokenizer_mistral.py",
    "tools/parity/verify_tokenizer_gguf.py",
)

# One operator's resolved values. A reader copies any of these, so they are wrong
# in a comment as well as in code.
RESOLVED_LITERALS = (
    "dgx.casa",
    "/home/mudler",
    "venvs/vllm-oracle",
    "work/vllm.cpp",
    "work/vllm-pin",
    "work/apex",
    "cutlass_probe",
    "/usr/local/cuda",
    "laguna-xs-nvfp4",
    "laguna-n4-build",
)

# The classification rule's second class, verbatim. Key = path, value = the
# sentence fragments that must SURVIVE. Rewriting any of these falsifies a
# recorded measurement or a named-instrument argument.
PROVENANCE = {
    "scripts/check-gate-commands.py": (
        "because dgx.casa was unreachable at the SSH layer",
    ),
    "scripts/gen-vulkan-spirv.py": (
        "Neither of our boxes has one",
        "`dgx.casa`",
        "measured 2026-07-22",
    ),
    "scripts/mtp-k-gt-1-neartie-gap.py": (
        "Defaults are the dgx.casa harness laid down by this row",
        '"/home/mudler/mtpgate"',
    ),
    "third_party/README.md": (
        "The pin matches the loader version\non `dgx.casa` (1.4.328)",
    ),
    "tools/bench/gpu_clock_state.py": (
        "On `dgx.casa` (GB10, driver 580.159.03)",
        "boot f6bbbfc6",
    ),
    "tools/parity/dump_tokenizer_gpt4o.py": (
        "The literal this replaced named `/mnt/nas_share`",
    ),
    # Both classes in one file: the measured sentence stays while the copyable
    # command above it converts.
    "scripts/glm4-neartie-gap.py": ("# GB10 (dgx.casa) has UNIFIED memory",),
    "scripts/internlm2-neartie-gap.py": ("# GB10 (dgx.casa) has UNIFIED memory",),
    "scripts/qwen3-neartie-gap.py": ("# GB10 (dgx.casa) has UNIFIED memory",),
}

REFUSAL_MARKERS = (".env", "Never substitute another")

_STUB = """\
class _Symbol:
    def __init__(self, name): self._name = name
    def __getattr__(self, item): return _Symbol(self._name + "." + item)
    def __call__(self, *a, **k): return _Symbol(self._name + "()")
    def __hash__(self): return hash(self._name)
    def __eq__(self, other): return getattr(other, "_name", None) == self._name
    def __repr__(self): return self._name


def __getattr__(name):
    return _Symbol(%r + "." + name)
"""


def stub_path(directory: Path) -> str:
    """A PYTHONPATH whose torch/numpy are inert, so no case needs either.

    The dumpers touch `torch.float32` at import time, so an empty module is not
    enough; these answer any attribute with a hashable sentinel. Nothing in a
    refusal path computes with them.
    """
    stubs = directory / "stubs"
    stubs.mkdir(parents=True, exist_ok=True)
    for name in ("torch", "numpy"):
        (stubs / f"{name}.py").write_text(_STUB % name, encoding="utf-8")
    existing = os.environ.get("PYTHONPATH", "")
    return f"{stubs}:{existing}" if existing else str(stubs)


def clean_env(directory: Path, **overrides: str) -> dict[str, str]:
    """The caller's environment minus every key this row resolves."""
    managed = (
        "VLLM_SOURCE",
        "VLLM_ORACLE",
        "CUTLASS_DIR",
        "VLLM_CPP_CUTLASS_DIR",
        "DEVICE_TOOLKIT_ROOT",
        "TRITON_PYTHON",
    )
    env = {k: v for k, v in os.environ.items() if k not in managed}
    env["HOME"] = str(directory / "home")
    (directory / "home").mkdir(parents=True, exist_ok=True)
    env["PYTHONPATH"] = stub_path(directory)
    env.update(overrides)
    return env


def run_dumper(script: Path, directory: Path, env: dict[str, str], *args: str):
    return subprocess.run(
        [sys.executable, str(script), "--out", str(directory / "out"), *args],
        cwd=script.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def staged_shell(directory: Path, script: Path, env_text: str | None) -> Path:
    """Copy a bash script into a scratch root so a scratch `.env` is the one read.

    Both scripts resolve the repository root as their own parent's parent, so the
    copy keeps the `scripts/` level. Writing into the real checkout would clobber
    the developer's own untracked file.
    """
    (directory / "scripts").mkdir(parents=True, exist_ok=True)
    target = directory / "scripts" / script.name
    shutil.copy2(script, target)
    if env_text is not None:
        (directory / ".env").write_text(env_text, encoding="utf-8")
    return target


def run_shell(target: Path, env: dict[str, str], *args: str):
    return subprocess.run(
        ["bash", str(target), *args],
        cwd=target.parent.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


class DumperRefusalTests(unittest.TestCase):
    """The pinned-source tree is named or the dump does not start."""

    def test_each_dumper_refuses_vllm_source_by_name(self):
        for script, flag in DUMPERS.items():
            with self.subTest(script=script.name), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                extra = (
                    ["--model", "unused", "--tag", "27b"]
                    if script.name == "dump_qwen3_5_mtp.py"
                    else []
                )
                result = run_dumper(script, directory, clean_env(directory), *extra)
                self.assertEqual(result.returncode, 3, result.stderr)
                self.assertIn("REFUSED: VLLM_SOURCE is unset", result.stderr)
                for marker in REFUSAL_MARKERS:
                    self.assertIn(marker, result.stderr)
                self.assertIn(flag, result.stderr)
                # Nothing was produced, so the refusal preceded every dump.
                self.assertFalse((directory / "out").exists())
                self.assertNotIn("Traceback", result.stderr)

    def test_each_dumper_takes_the_pin_from_the_process_environment(self):
        for script, flag in DUMPERS.items():
            with self.subTest(script=script.name), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                pin = directory / "pinned-vllm"
                pin.mkdir()
                extra = (
                    ["--model", "unused", "--tag", "27b"]
                    if script.name == "dump_qwen3_5_mtp.py"
                    else []
                )
                env = clean_env(directory, VLLM_SOURCE=str(pin))
                result = run_dumper(script, directory, env, *extra)
                self.assertNotIn("REFUSED", result.stderr)
                # It resolved, and it SAID which tree the oracle math will run
                # from, which is the only thing this case claims.
                self.assertIn(f"pinned vLLM source: {pin}", result.stderr)

    def test_the_flag_still_wins_over_the_environment(self):
        for script, flag in DUMPERS.items():
            with self.subTest(script=script.name), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                chosen = directory / "explicit"
                chosen.mkdir()
                extra = (
                    ["--model", "unused", "--tag", "27b"]
                    if script.name == "dump_qwen3_5_mtp.py"
                    else []
                )
                env = clean_env(directory, VLLM_SOURCE=str(directory / "ignored"))
                result = run_dumper(
                    script, directory, env, flag, str(chosen), *extra
                )
                self.assertNotIn("REFUSED", result.stderr)
                self.assertIn(f"pinned vLLM source: {chosen}", result.stderr)
                self.assertNotIn("ignored", result.stdout + result.stderr)


class RegenRefusalTests(unittest.TestCase):
    """The Triton AOT regen refuses before it configures anything."""

    def test_each_required_value_refuses_by_name_before_cmake(self):
        for missing in ("VLLM_ORACLE", "CUTLASS_DIR"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                target = staged_shell(directory, REGEN, None)
                supplied = {
                    key: str(directory / "placeholder")
                    for key in ("VLLM_ORACLE", "CUTLASS_DIR")
                    if key != missing
                }
                result = run_shell(target, clean_env(directory, **supplied))
                self.assertEqual(result.returncode, 3, result.stderr)
                self.assertIn(f"REFUSED: {missing} is unset", result.stderr)
                for marker in REFUSAL_MARKERS:
                    self.assertIn(marker, result.stderr)
                self.assertNotIn("=== configure", result.stdout)
                self.assertFalse((directory / "build-triton-regen").exists())

    def test_values_resolve_from_the_untracked_env_file(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            venv = directory / "oracle"
            venv.mkdir()
            target = staged_shell(
                directory,
                REGEN,
                f"VLLM_ORACLE={venv}\nCUTLASS_DIR={directory}/cutlass\n",
            )
            result = run_shell(target, clean_env(directory))
            # Past both refusals and stopped at the interpreter probe, because
            # the scratch venv has no bin/python. Nothing configures.
            self.assertNotIn("REFUSED", result.stderr)
            self.assertIn(f"{venv}/bin/python", result.stderr)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertNotIn("=== configure", result.stdout)

    def test_a_blank_line_in_the_file_does_not_erase_the_shell_answer(self):
        """A `.env` copied from `.env.example` declares EVERY key blank.

        `set -a; . ./.env; set +a` then assigns the empty string over a value
        the caller exported, and the script refuses a session that is correctly
        configured. This is the defect the campaign's worked example found.
        """
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            venv = directory / "oracle"
            venv.mkdir()
            target = staged_shell(directory, REGEN, "VLLM_ORACLE=\nCUTLASS_DIR=\n")
            env = clean_env(
                directory, VLLM_ORACLE=str(venv), CUTLASS_DIR=str(directory)
            )
            result = run_shell(target, env)
            self.assertNotIn("REFUSED", result.stderr)
            self.assertIn(f"{venv}/bin/python", result.stderr)

    def test_an_oracle_that_is_an_interpreter_is_used_as_one(self):
        """`.env.example` declares VLLM_ORACLE as a venv root OR an executable."""
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            interpreter = directory / "oracle-python3"
            interpreter.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")
            interpreter.chmod(0o755)
            target = staged_shell(directory, REGEN, None)
            env = clean_env(
                directory,
                VLLM_ORACLE=str(interpreter),
                CUTLASS_DIR=str(directory),
            )
            result = run_shell(target, env)
            self.assertNotIn("REFUSED", result.stderr)
            combined = result.stdout + result.stderr
            self.assertIn(str(interpreter), combined)
            self.assertNotIn(f"{interpreter}/bin/python", combined)


class LagunaRefusalTests(unittest.TestCase):
    """The long-context harness names what it needs instead of guessing a $HOME."""

    def test_it_refuses_each_missing_operand_before_taking_the_gpu_lock(self):
        for supplied in ([], ["/scratch/model"]):
            with self.subTest(argc=len(supplied)), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                target = staged_shell(directory, LAGUNA, None)
                env = clean_env(directory)
                result = run_shell(target, env, *supplied)
                self.assertEqual(result.returncode, 3, result.stderr)
                self.assertIn("REFUSED:", result.stderr)
                self.assertIn("CHECKPOINT_ROOT" if not supplied else "binary",
                              result.stderr)
                self.assertIn("laguna_longctx_bench.sh", result.stderr)
                # No output tree, and therefore no lock and no `sudo`.
                home = Path(env["HOME"])
                self.assertEqual(list(home.glob("laguna-longctx-*")), [])


class NoResolvedLiteralTests(unittest.TestCase):
    """A converted file carries no operator-resolved value in either half."""

    def test_no_converted_file_names_a_resolved_value(self):
        for relative in CONVERTED:
            text = (ROOT / relative).read_text(encoding="utf-8")
            keep = PROVENANCE.get(relative, ())
            for fragment in keep:
                text = text.replace(fragment, "")
            for literal in RESOLVED_LITERALS:
                with self.subTest(path=relative, literal=literal):
                    # Short message on purpose: the default dumps the whole file.
                    self.assertNotIn(
                        literal, text, f"{relative} still names {literal!r}"
                    )

    def test_no_converted_script_carries_a_home_anchored_fallback(self):
        """`${KEY:-}` saves an answer. `${KEY:-$HOME/...}` is the defect.

        A relative default such as `${BUILD_DIR:-build-triton-regen}` is nobody's
        path and stays. What may not stay is a fallback anchored in a home
        directory or an absolute one, because that is always SOME developer's
        layout and it resolves for the reader instead of refusing.

        `GPU_LOCK` is the one exemption, and `.env.example` states why: a mutex
        only excludes people who take the SAME file, so a divergent value
        serialises the holder with nobody (#777). Every script falls back to the
        same path deliberately.
        """
        pattern = re.compile(r"\$\{[A-Za-z_0-9]+:-[^}]*(?:\$HOME|\$\{HOME\}|/home/|~/)")
        for relative in CONVERTED:
            with self.subTest(path=relative):
                text = (ROOT / relative).read_text(encoding="utf-8")
                text = text.replace("${GPU_LOCK:-$HOME/gpu.lock}", "")
                found = pattern.search(text)
                self.assertIsNone(found, found.group(0) if found else "")


class ProvenanceTests(unittest.TestCase):
    """The second class stays literal. A blind sweep of the tree goes red here."""

    def test_every_recorded_measurement_keeps_its_host(self):
        for relative, fragments in PROVENANCE.items():
            text = (ROOT / relative).read_text(encoding="utf-8")
            for fragment in fragments:
                with self.subTest(path=relative, fragment=fragment[:40]):
                    self.assertIn(fragment, text)


if __name__ == "__main__":
    unittest.main()

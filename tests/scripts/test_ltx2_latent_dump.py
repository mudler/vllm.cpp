#!/usr/bin/env python3
"""`tools/oracle/ltx2_latent_dump.py`'s controls are exercised, not just written.

Two defects reached a queued `dgx` lease in that file and neither was detectable
by anything in this tree, because nothing executed it: `git grep -l
ltx2_latent_dump` matched the spec, the lease harness, and the file itself.

**Defect 1, an incomplete rename.** `write_latent` records the faithful copy
under `raw_native`; two reads in the reporting block spelled it `raw_bf16`. The
loop always runs, so the first record raised `KeyError` before the byte-length
assertion, before the frame control, before the manifest was written, and the
process exited rc 1 -- which is not one of the documented exit codes. The lease
would have paid ~42 GB of CIFS staging, a torch install and a ~94 s render to
reach it.

**Defect 2, a control that passes on zero evidence.** The frame check iterated
whatever `frames_dir.glob("frame_*.ppm")` returned and counted disagreements. An
empty glob yields no disagreements, `matched = n_frames - 0 = 0 - 0 = 0`, and
the driver printed "CONTROL PASSED: this run reproduces the committed reference
byte for byte" and returned 0. A decode that produced NO frames -- an `av`
failure, a truncated mp4, a wrong output path -- read as a passing control. That
is the failure the file's own docstring names: "a counter that could not be
written reads exactly like a counter that was never incremented".

The repair makes the control compare the observation set against the EXPECTATION
set: every frame the committed `SHA256SUMS` names must be present and must
agree. The expected count is derived from that file, never written here as a
literal, because a transcription cannot gate the thing it transcribes.

**Defect 3, an assertion that the controls are CALLED rather than OBEYED.** The
first repair proved reachability by walking `main()`'s AST for a `Call` node.
That sees the call and cannot see the verdict: delete `if rc: return rc` after
`report_latents`, or the `if not frame_check["ok"]: return 64` after the frame
control, and both `Call` nodes survive while the driver computes a correct
`ok: False`, prints "CONTROL PASSED" and returns 0. That is defect 2 again, one
layer up. So the AST walk is kept as a precondition and the evidence is an
end-to-end `main()` drive with the GPU, torch, upstream and checkpoint seams
stubbed: it asserts rc 64 on a decode missing a committed frame and rc 62 on a
short-written latent. Both mutations red only against that drive.

This suite needs no build, no GPU, no network, no torch and no checkpoint --
which is the point. All three defects were catchable on a laptop.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DUMP_PY = ROOT / "tools/oracle/ltx2_latent_dump.py"
SUMS = ROOT / "tests/parity/goldens/ltx2_oracle/SHA256SUMS"

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        failures.append(msg)


def load_module():
    spec = importlib.util.spec_from_file_location("ltx2_latent_dump_under_test", DUMP_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def latent_record(written: int, expected: int) -> dict:
    """A `write_latent`-shaped record, spelled the way `write_latent` spells it."""
    return {
        "shape": [1, 4, 8],
        "torch_dtype": "torch.bfloat16",
        "itemsize": 2,
        "expected_bytes": expected,
        "raw_native": {"path": "/tmp/video_latent_bfloat16.raw", "bytes": written,
                       "sha256": "0" * 64},
        "npy_f32": {"path": "/tmp/video_latent_f32.npy", "bytes": 4 * 32,
                    "sha256": "1" * 64},
        "stats_f32": {"min": -1.0, "max": 1.0, "mean": 0.0, "std": 1.0, "abs_sum": 1.0},
    }


mod = load_module()
source = DUMP_PY.read_text(encoding="utf-8")
tree = ast.parse(source)
functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

print("=== the reporting path is reachable and callable at all ===")
check("report_latents" in functions,
      "the latent reporting block is a callable function a test can drive")
check("check_frames" in functions,
      "the frame control is a callable function a test can drive")
check("committed_frame_digests" in functions,
      "the expected frame set is derived by a named function")

if not {"report_latents", "check_frames", "committed_frame_digests"} <= set(functions):
    print(f"FAILED ({len(failures)})")
    sys.exit(1)

print("=== D1: every key the reporting path READS, write_latent WRITES ===")
# Generic, not a spelling list: this is the check that would have caught the
# `raw_bf16` rename at rest, and catches the next one too.
written_keys: set[str] = set()
# Every dict literal in write_latent, nested ones included: the record it
# returns holds sub-dicts, and `rec["raw_native"]["bytes"]` reads a key from
# one of those. Collecting only the outer assignment would report `bytes` as
# stray. This stays strict about what matters -- `raw_bf16` appears in no dict
# literal in that function, at any depth, which is the whole defect.
for node in ast.walk(functions["write_latent"]):
    if isinstance(node, ast.Dict):
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                written_keys.add(key.value)
read_keys: set[str] = set()
for node in ast.walk(functions["report_latents"]):
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        if isinstance(node.slice.value, str):
            read_keys.add(node.slice.value)
check(bool(written_keys), "write_latent's record literal was parsed")
check(bool(read_keys), "report_latents' record reads were parsed")
unknown = sorted(read_keys - written_keys)
check(not unknown,
      f"report_latents reads no key write_latent does not write (stray: {unknown})")

print("=== D1: the reporting path runs instead of raising ===")
try:
    rc = mod.report_latents({"video": latent_record(64, 64)})
    check(rc == 0, "a full-length latent record reports rc 0")
except Exception as exc:  # noqa: BLE001 - the exception IS the defect
    check(False, f"a full-length latent record reports rc 0 (raised {type(exc).__name__}: {exc})")
try:
    rc = mod.report_latents({"video": latent_record(32, 64)})
    check(rc == 62, "a SHORT-WRITE latent record reports rc 62")
except Exception as exc:  # noqa: BLE001
    check(False, f"a SHORT-WRITE latent record reports rc 62 (raised {type(exc).__name__}: {exc})")

print("=== the committed expectation is real and is derived, not transcribed ===")
committed = mod.read_committed_digests(SUMS)
frames = mod.committed_frame_digests(committed)
check(len(committed) > len(frames),
      "SHA256SUMS carries non-frame rows too (mp4, manifest, audio)")
check(len(frames) > 0, "SHA256SUMS names at least one frame_*.ppm")
independent = len([
    line for line in SUMS.read_text().splitlines()
    if re.match(r"^[0-9a-f]{64}\s+frame_\d+\.ppm$", line.strip())
])
check(len(frames) == independent,
      f"the derived frame count ({len(frames)}) equals the file's own frame rows ({independent})")
check(all(re.fullmatch(r"frame_\d+\.ppm", n) for n in frames),
      "every derived expectation key is a frame file name")


def write_frames(root: Path, payloads: dict[str, bytes]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, blob in payloads.items():
        (root / name).write_bytes(blob)
    return root


with tempfile.TemporaryDirectory(prefix="ltx2-latent-dump-test-") as tmp:
    tmp = Path(tmp)
    # A synthetic three-frame expectation. If the expected count were hardcoded
    # to the committed 25, this leg could not pass -- which is what proves the
    # count is derived rather than transcribed.
    blobs = {f"frame_{i:06d}.ppm": f"frame {i}".encode() for i in range(3)}
    synthetic = {name: sha256_bytes(blob) for name, blob in blobs.items()}
    synthetic["upstream-render.mp4"] = "9" * 64

    print("=== D2: ZERO decoded frames is a FAILING control, not a passing one ===")
    empty = write_frames(tmp / "empty", {})
    res = mod.check_frames(empty, synthetic, 0)
    check(res["ok"] is False, "a decode that produced no frames does NOT pass the control")
    check(res["matched"] == 0, "zero frames matched")
    check(len(res["mismatches"]) == len(blobs),
          "every committed frame is reported MISSING rather than silently skipped")

    print("=== D2: a complete, correct decode PASSES ===")
    good = write_frames(tmp / "good", blobs)
    res = mod.check_frames(good, synthetic, len(blobs))
    check(res["ok"] is True, "all committed frames present and agreeing passes the control")
    check(res["matched"] == len(blobs), f"matched == {len(blobs)}, the derived expectation")
    check(res["mismatches"] == [], "no mismatches on a clean run")

    print("=== D2: a PARTIAL decode fails, even though what appeared agreed ===")
    partial = write_frames(tmp / "partial", dict(list(blobs.items())[:2]))
    res = mod.check_frames(partial, synthetic, 2)
    check(res["ok"] is False, "a decode missing one committed frame does NOT pass")
    check(res["matched"] == 2, "the two frames that did appear are still counted as matched")
    check(any(m["frame"] == "frame_000002.ppm" for m in res["mismatches"]),
          "the missing frame is named in the mismatches")

    print("=== D2: a WRONG digest fails ===")
    wrong = write_frames(tmp / "wrong", {**blobs, "frame_000001.ppm": b"perturbed"})
    res = mod.check_frames(wrong, synthetic, len(blobs))
    check(res["ok"] is False, "a frame whose digest disagrees does NOT pass")
    check(any(m["frame"] == "frame_000001.ppm" for m in res["mismatches"]),
          "the disagreeing frame is named in the mismatches")

    print("=== D2: an UNLISTED extra frame fails ===")
    extra = write_frames(tmp / "extra", {**blobs, "frame_000009.ppm": b"unexpected"})
    res = mod.check_frames(extra, synthetic, len(blobs) + 1)
    check(res["ok"] is False, "a decoded frame the committed file does not name does NOT pass")

    print("=== D2: an expectation with NO frame rows is not a pass either ===")
    # Two cases, and only the SECOND one is vacuous. Here three frames DID
    # appear under an expectation naming none of them, so each is an
    # unlisted-frame mismatch and `ok` is false for that reason -- which is not
    # the reason the leg is named for. Keeping it is still worth it, because it
    # is the shape a wrong `--output-path` produces.
    res = mod.check_frames(good, {"upstream-render.mp4": "9" * 64}, len(blobs))
    check(res["ok"] is False,
          "frames that appeared under a frame-less expectation are all unlisted mismatches")
    check(len(res["mismatches"]) == len(blobs),
          "each decoded frame is reported as one the committed file does not name")
    check(res["committed_frames"] == 0, "the empty frame expectation is reported as zero")
    # THE VACUOUS CASE: nothing expected AND nothing observed. There is no
    # mismatch to find, and `matched == 0 == len(expected)`. Every other term in
    # `ok` is therefore satisfied, so `bool(expected)` is the only thing between
    # this and a control that passes on zero evidence. Drop it and this leg is
    # the only one in the suite that reds.
    res = mod.check_frames(empty, {"upstream-render.mp4": "9" * 64}, 0)
    check(res["ok"] is False,
          "an empty expectation compared against an empty observation cannot pass vacuously")
    check(res["mismatches"] == [],
          "the vacuous case genuinely produces NO mismatch, which is why ok must reject it on its own")
    check(res["matched"] == 0 and res["committed_frames"] == 0,
          "matched == committed_frames == 0 in the vacuous case")

print("=== both controls are REACHED from main(), not merely defined ===")
# A `Call` node named in `main()` is a PRECONDITION, not the evidence. Walking
# the AST proves the call is written; it cannot see whether the verdict that
# call returns reaches the exit code. Deleting `if rc: return rc` after
# `report_latents`, or `if not frame_check["ok"]: return 64` after the frame
# control, leaves both Call nodes in place -- and leaves the driver computing a
# correct `ok: False`, printing CONTROL PASSED and returning 0. That is defect 2
# restored one layer up, so the real check is the end-to-end drive below.
called = {
    node.func.id
    for node in ast.walk(functions["main"])
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
}
check("report_latents" in called, "main() calls report_latents")
check("check_frames" in called, "main() calls check_frames")

print("=== main() END TO END: the controls' verdicts reach the EXIT CODE ===")


class _FakeStage:
    """Stands in for `RecordingDiffusionStage` after a render.

    `video_states` is a constructor argument and not a fixed one because the
    HOOK-NEVER-FIRED path is reachable only with an empty one. A fake that
    always supplies a state cannot enter it, so `return 61` sat documented and
    unexecuted while the suite read fully green.
    """

    def __init__(self, video_states: list | None = None) -> None:
        self.video_states = ([types.SimpleNamespace(latent=object())]
                             if video_states is None else video_states)
        self.audio_states: list = []


def drive_main(tmp: Path, *, expectation: dict[str, str], decoded: dict[str, bytes],
               latent_written: int, latent_expected: int,
               video_states: list | None = None) -> tuple[int, str]:
    """Run the driver's own `main()` end to end and return `(rc, stdout)`.

    Every seam that needs a GPU, torch, the upstream checkout or a 42 GB
    checkpoint is replaced, and NOTHING between the two controls and `return`
    is: `report_latents` and `check_frames` are the real functions, and the
    control flow that turns their verdicts into an exit code is the real
    control flow. That is the part an AST walk cannot reach.
    """
    tmp.mkdir(parents=True, exist_ok=True)
    out = tmp / "out"
    sums = tmp / "SHA256SUMS"
    sums.write_text(
        "# a synthetic expectation, so this needs no committed frame\n"
        + "".join(f"{digest}  {name}\n" for name, digest in sorted(expectation.items()))
    )

    def fake_decode(video_path, frames_dir):
        frames_dir.mkdir(parents=True, exist_ok=True)
        for name, blob in decoded.items():
            (frames_dir / name).write_bytes(blob)
        return len(decoded)

    saved_modules = {k: sys.modules.get(k) for k in
                     ("torch", "ltx_pipelines", "ltx_pipelines.ti2vid_one_stage")}
    saved_argv = sys.argv
    saved = {name: getattr(mod, name) for name in
             ("assert_identity", "install_recorder", "write_latent", "decode", "COMMITTED_SUMS")}
    pkg = types.ModuleType("ltx_pipelines")
    pkg.__path__ = []
    one_stage = types.ModuleType("ltx_pipelines.ti2vid_one_stage")
    one_stage.main = lambda: None
    pkg.ti2vid_one_stage = one_stage
    sys.modules["torch"] = types.ModuleType("torch")
    sys.modules["ltx_pipelines"] = pkg
    sys.modules["ltx_pipelines.ti2vid_one_stage"] = one_stage
    mod.assert_identity = lambda source: {"revision": "0" * 40, "worktree_dirty": False}
    mod.install_recorder = lambda module: {
        "pipeline": types.SimpleNamespace(stage=_FakeStage(video_states))}
    mod.write_latent = lambda tensor, out_dir, stem: latent_record(latent_written, latent_expected)
    mod.decode = fake_decode
    mod.COMMITTED_SUMS = sums
    sys.argv = ["ltx2_latent_dump", "--ltx2-source", str(tmp), "--checkpoints", str(tmp),
                "--out", str(out)]
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            rc = mod.main()
    finally:
        sys.argv = saved_argv
        for name, value in saved.items():
            setattr(mod, name, value)
        for key, value in saved_modules.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return rc, buffer.getvalue()


with tempfile.TemporaryDirectory(prefix="ltx2-latent-dump-main-") as tmp:
    tmp = Path(tmp)
    blobs = {f"frame_{i:06d}.ppm": f"frame {i}".encode() for i in range(3)}
    expectation = {name: sha256_bytes(blob) for name, blob in blobs.items()}
    expectation["upstream-render.mp4"] = "9" * 64

    rc, log = drive_main(tmp / "clean", expectation=expectation, decoded=blobs,
                         latent_written=64, latent_expected=64)
    check(rc == 0, f"a complete latent and a complete decode exit 0 (got {rc})")
    check("CONTROL PASSED" in log, "and the driver says the control passed")
    manifest = json.loads(
        (tmp / "clean/out/ltx2_latent_dump_manifest.json").read_text())
    check("environment" in manifest,
          "main() writes an environment block into the manifest")
    check(manifest["environment"]["python"] == sys.version.split()[0],
          "and the environment block was really filled in, not left empty")

    # M4: with `if not frame_check["ok"]: return 64` deleted, the driver still
    # computes ok: False and still prints CONTROL PASSED. Only the exit code
    # moves, so only an exit-code assertion sees it.
    rc, log = drive_main(tmp / "missing", expectation=expectation,
                         decoded=dict(list(blobs.items())[:2]),
                         latent_written=64, latent_expected=64)
    check(rc == 64, f"a decode MISSING a committed frame exits 64, not 0 (got {rc})")
    check("CONTROL PASSED" not in log,
          "and the driver does NOT report the control as passed")

    rc, _ = drive_main(tmp / "wrongdigest", expectation=expectation,
                       decoded={**blobs, "frame_000001.ppm": b"perturbed"},
                       latent_written=64, latent_expected=64)
    check(rc == 64, f"a decode whose digest DISAGREES exits 64 (got {rc})")

    # M5: with `if rc: return rc` deleted after report_latents, a short-written
    # latent runs on into a decode that agrees, and exits 0.
    rc, log = drive_main(tmp / "shortwrite", expectation=expectation, decoded=blobs,
                         latent_written=32, latent_expected=64)
    check(rc == 62, f"a SHORT-WRITTEN latent exits 62 before any decode (got {rc})")
    check("DECODE (the control" not in log,
          "and the run stops at the latent report rather than reaching the decode")

    # M6: with `if not states: return 61` deleted, `states[-1]` raises
    # IndexError and the process exits 1 -- a code this module documents
    # nowhere, on a path that is a finding about the INSTRUMENT rather than a
    # result about the render. Every other leg here hands the fake a video
    # state, so nothing entered this branch and rc 61 was documented and never
    # executed.
    # The guard's absence raises rather than returning, so the call is caught:
    # a traceback out of the suite IS a red, but it names a Python line instead
    # of the guarantee that moved, and the next reader has to work out which.
    try:
        rc, log = drive_main(tmp / "nostates", expectation=expectation,
                             decoded=blobs, latent_written=64, latent_expected=64,
                             video_states=[])
    except Exception as exc:  # noqa: BLE001 - the missing guard IS the defect
        rc, log = -1, ""
        check(False, f"main() must REPORT an empty recording, not raise "
                     f"({type(exc).__name__}: {exc})")
    check(rc == 61, f"a render whose recording hook never fired exits 61 (got {rc})")
    check("RECORDED STATES: 0" in log,
          "and the driver reports the zero it saw rather than inferring it")
    check("LATENT DUMP" not in log,
          "and it stops before dumping, because there is nothing to dump")

print("=== the environment recorder describes, and never raises ===")
fake_torch = types.SimpleNamespace(
    __version__="2.13.0+cu130",
    version=types.SimpleNamespace(cuda="13.0"),
    cuda=types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda i: "NVIDIA GB10",
        get_device_capability=lambda i: (12, 1),
    ),
    _C=types.SimpleNamespace(_cuda_getDriverVersion=lambda: 13000),
)
rec = mod.environment_record(fake_torch)
check(rec["torch"] == "2.13.0+cu130", "the torch version is recorded")
check(rec["gpu"] == "NVIDIA GB10", "the GPU name is recorded")
check(rec.get("capability") == [12, 1],
      "the GPU capability is recorded under the reference manifest's own key "
      "and in its own list-of-ints shape")
check(rec["cuda_driver"] == 13000, "the CUDA driver version is recorded")

# The docstring claims the key names mirror the reference manifest's own
# environment block field by field. That claim was FALSE on one field for a
# whole review round -- `gpu_capability`/"12.1" against `capability`/[12, 1] --
# because nothing read the reference. This reads it. A rename or a reshape on
# either side reds here, which is what makes the claim a gate and not prose.
reference_environment = json.loads(
    (ROOT / "tests/parity/goldens/ltx2_oracle/ltx2_oracle_manifest.json").read_text()
)["environment"]
check(len(reference_environment) >= 6,
      "the reference manifest's environment block is the one with fields in it "
      f"(read {len(reference_environment)} keys)")
absent = sorted(k for k in reference_environment if k not in rec)
check(not absent,
      "every reference-manifest environment key is mirrored here"
      + (f" (absent: {absent})" if absent else ""))
reshaped = sorted(k for k in reference_environment
                  if k in rec and type(rec[k]) is not type(reference_environment[k]))
check(not reshaped,
      "and each mirrored key carries the reference's own type"
      + (f" (reshaped: {reshaped})" if reshaped else ""))


class _Hostile:
    """Every attribute access raises. A driver that dies while describing its own
    environment would destroy the render it just paid for."""

    def __getattr__(self, name):
        raise RuntimeError(f"no {name}")


try:
    rec = mod.environment_record(_Hostile())
    check(rec["torch"] is None and rec["gpu"] is None,
          "an unreadable field is recorded as null rather than raising")
    check(rec["python"] == sys.version.split()[0],
          "and the fields that do not need torch are still filled in")
except Exception as exc:  # noqa: BLE001 - raising here IS the defect
    check(False, f"environment_record must not raise (raised {type(exc).__name__}: {exc})")

if failures:
    print(f"FAILED ({len(failures)})")
    sys.exit(1)
print("PASSED")

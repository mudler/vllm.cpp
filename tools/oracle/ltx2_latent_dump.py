#!/usr/bin/env python3
"""Dump the LTX-2 oracle's FINAL VIDEO LATENT at the pinned #1864 request, and
prove the dump did not change the render.

WHY THIS EXISTS, AND WHY IT IS NOT A FLAG ON `ltx2_oracle.py`.
`ltx2_oracle.py` runs upstream as a SUBPROCESS
(`ltx2_oracle.py:243`, `subprocess.call(argv)`), which is the right shape for a
reference render and the wrong one for a diagnostic: nothing in the parent can
reach `video_state.latent`. Reaching it needs upstream's `main()` to run
IN-PROCESS, which is a different program, so it is a different file.

`LTX25-ADHERENCE-BISECT` (#2514) asks whether our adherence gap is in the latent
or in the VAE decode. Differencing two seed-42 latents cannot answer that:
`.agents/specs/ltx25-oracle-absolute.md` records that the two engines draw
DIFFERENT noise from the same seed integer, so two CORRECT denoisers would also
land on two different latents. The noise-independent experiment is a
cross-decode -- one latent, both VAEs -- and this script produces the one latent.

WHAT IS AND IS NOT MODIFIED. Nothing under the pinned checkout is edited. The
only delta from upstream's own command line is the substitution its own
`RecordingDiffusionStage` docstring prescribes
(`packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:585-621`):

    pipeline.stage = RecordingDiffusionStage(pipeline.stage)

`ti2vid_one_stage.main()` builds its pipeline in a local
(`ti2vid_one_stage.py:254`), so the substitution is installed by wrapping the
CLASS's `__init__` rather than by rewriting `main()`. Upstream's `main()` then
runs verbatim, with upstream's own argv, which is what makes the dumped latent
the oracle's rather than a re-implementation's.

THE CONTROL IS THE POINT, AND IT IS ONE CHECK THAT PROVES THREE THINGS.
EVERY frame named in `tests/parity/goldens/ltx2_oracle/SHA256SUMS` must be
decoded by this run and must match its committed digest. The expected count is
derived from that file, so it cannot drift from it. Comparing only the frames
that happened to appear would let a decode producing NONE of them pass on an
empty intersection, and that is what this check used to do. If they all match,
then at this request
the oracle is reproducible, the recording substitution perturbed nothing, and
this driver is equivalent to upstream's `main()`. If they do not, the latent is
NOT the reference's latent and this script says so rather than scoring it -- a
probe that changed the run is a finding about the probe.

RECORD THE ENVIRONMENT. `environment_record` writes torch, CUDA, driver and GPU
into the manifest. Phase A recorded none of it, which is why its audio
divergence from the committed reference is an unattributed observation rather
than a finding: the one variable that would separate a different torch from a
perturbing probe was not captured.

ASSERT THE DUMP RAN. A counter that could not be written reads exactly like a
counter that was never incremented, which has cost this campaign real time. So
the recorded state list must be non-empty and the written file's length must
equal the product of its own recorded shape and itemsize, or this exits non-zero.

Exit codes:
  0  latent written AND the frames match the committed digests
  60 upstream identity assertion failed
  61 the recording stage captured no state (the hook never fired)
  62 the written latent's byte length disagrees with its recorded shape
  63 the render itself failed
  64 the decoded frames do NOT reproduce the committed digests: a committed
     frame is missing, a digest disagrees, an unnamed frame appeared, or the
     expectation itself names no frame (the probe perturbed the run, or the run
     is not reproducible; either way the latent is not usable)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ltx2_oracle import (  # noqa: E402
    COMPONENTS,
    HEIGHT,
    NUM_FRAMES,
    NUM_INFERENCE_STEPS,
    PROMPT,
    SEED,
    WIDTH,
    assert_identity,
    decode,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_SUMS = REPO_ROOT / "tests/parity/goldens/ltx2_oracle/SHA256SUMS"
_FRAME_NAME = re.compile(r"frame_\d+\.ppm")


def read_committed_digests(path: Path) -> dict[str, str]:
    """Parse the committed SHA256SUMS into {basename: digest}.

    The file carries a prose preamble, so only lines whose first field is 64 hex
    characters are taken. A parse that silently produced an EMPTY map would make
    the frame check vacuous and it would still exit 0, so the caller refuses an
    empty result rather than trusting it.
    """
    digests: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            try:
                int(parts[0], 16)
            except ValueError:
                continue
            digests[Path(parts[-1]).name] = parts[0].lower()
    return digests


def install_recorder(module) -> dict:
    """Wrap TI2VidOneStagePipeline.__init__ so every instance records its stage.

    Returns a dict that will hold the constructed pipeline. `main()` keeps the
    pipeline in a local and never returns it, so this is the only way to reach
    the recorded states after the run.
    """
    from ltx_pipelines.utils.blocks import RecordingDiffusionStage

    holder: dict = {}
    original = module.TI2VidOneStagePipeline.__init__

    def recording_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.stage = RecordingDiffusionStage(self.stage)
        holder["pipeline"] = self

    module.TI2VidOneStagePipeline.__init__ = recording_init
    return holder


def write_latent(tensor, out_dir: Path, stem: str) -> dict:
    """Write the latent twice and record both.

    The bf16 `.npy` is the FAITHFUL record: it is the tensor upstream's VAE
    actually consumed, in the dtype it consumed it in. The f32 `.npy` is the
    CONSUMABLE one, because our side carries this latent as `std::vector<float>`
    (`src/vllm/multimodal/ltx2_video.cpp:2970`) and a cross-decode has to feed it
    something it can read. Recording both is the memory-format comparison
    AGENTS.md requires, rather than a claim that the two are interchangeable.
    """
    import numpy as np

    import torch

    detached = tensor.detach().to("cpu")
    native = detached.contiguous()
    f32 = detached.to(torch.float32).contiguous().numpy()

    # numpy has no bfloat16, so the faithful copy is written as RAW BYTES beside
    # a manifest naming its dtype and shape. Writing it through numpy would have
    # to upcast, and a bf16 tensor stored as f32 and called "the bf16 record" is
    # exactly the on-disk-dtype-is-not-runtime-dtype defect. `view(torch.uint8)`
    # on a 1-D contiguous tensor reinterprets the same storage, so these are the
    # bytes the VAE read, in their own order.
    raw_path = out_dir / f"{stem}_{str(native.dtype).replace('torch.', '')}.raw"
    raw_path.write_bytes(native.reshape(-1).view(torch.uint8).numpy().tobytes())

    f32_path = out_dir / f"{stem}_f32.npy"
    np.save(f32_path, f32)

    shape = list(native.shape)
    itemsize = native.element_size()
    expected = itemsize
    for d in shape:
        expected *= d
    record = {
        "shape": shape,
        "torch_dtype": str(native.dtype),
        "itemsize": itemsize,
        "expected_bytes": expected,
        "raw_native": {"path": str(raw_path), "bytes": raw_path.stat().st_size,
                     "sha256": sha256_file(raw_path)},
        "npy_f32": {"path": str(f32_path), "bytes": f32_path.stat().st_size,
                    "sha256": sha256_file(f32_path)},
        "stats_f32": {
            "min": float(f32.min()), "max": float(f32.max()),
            "mean": float(f32.mean()), "std": float(f32.std()),
            "abs_sum": float(abs(f32).sum()),
        },
    }
    return record


def environment_record(torch_module) -> dict:
    """What varied between two runs of this driver, captured rather than assumed.

    The Phase A run recorded none of this, and that cost an attribution: its
    `audio.wav` differed from the committed reference on 94.2% of samples and
    nothing in the record could separate "a different torch moved the audio VAE"
    from "the recording probe moved it". The reference render's own manifest
    (`tests/parity/goldens/ltx2_oracle/ltx2_oracle_manifest.json`) records
    `torch 2.13.0+cu130` on an `NVIDIA GB10`, so the comparison was always
    available and simply was not written down.

    Every field is best-effort and no probe here may raise: a driver that dies
    while describing its own environment would destroy the render it just paid
    for. A field that could not be read is recorded as `null`, which is
    distinguishable from a field that read as absent.

    The key names mirror the block `ltx2_oracle.py:276-290` writes inline, so the
    two manifests compare field by field, and this is a superset: every one of
    that block's six keys appears here spelled and shaped the same way, including
    `capability` as a list of ints rather than a dotted string, and it adds
    `cuda_available`, the CUDA runtime version and the driver version, which are
    what a "did torch move?" question actually needs.

    It is not shared code with that block on purpose. Extracting
    it would change the shape of the REFERENCE render's manifest, whose digest is
    committed in `tests/parity/goldens/ltx2_oracle/SHA256SUMS` and which every
    other LTX-2 row compares against; that is a larger blast radius than a
    ten-line probe is worth. If `ltx2_oracle.py` is ever re-run for another
    reason, fold this function in then.
    """
    def probe(fn):
        try:
            return fn()
        except Exception:  # noqa: BLE001 - an unreadable field is a null, not a crash
            return None

    # `getattr(..., None)` only swallows AttributeError, and a torch whose CUDA
    # attribute raises anything else would take the manifest down with it, after
    # the render is already paid for. Every read goes through `probe`.
    cuda = probe(lambda: torch_module.cuda)
    available = probe(lambda: bool(cuda.is_available()))
    return {
        "python": sys.version.split()[0],
        "platform": probe(platform.platform),
        "machine": probe(platform.machine),
        "torch": probe(lambda: torch_module.__version__),
        "torch_cuda": probe(lambda: torch_module.version.cuda),
        "cuda_available": available,
        "gpu": probe(lambda: cuda.get_device_name(0)) if available else None,
        # `capability`, a list of ints, because that is the key and the shape
        # `ltx2_oracle.py:287-288` writes. A dotted "12.1" read better and made
        # the two manifests disagree on the one field a field-by-field diff
        # would then report as absent from one side and extra on the other.
        "capability": probe(
            lambda: list(cuda.get_device_capability(0))
        ) if available else None,
        "cuda_driver": probe(lambda: torch_module._C._cuda_getDriverVersion()),
    }


def committed_frame_digests(committed: dict[str, str]) -> dict[str, str]:
    """The frame rows of the committed sums, which ARE the expected frame set.

    The count is DERIVED from `SHA256SUMS` and is never written here as a
    literal. `NUM_FRAMES` would be a second transcription of the same number,
    and a transcription cannot gate the thing it transcribes: were the reference
    render ever recut at a different length, a hardcoded 25 would keep reading
    as a pass while comparing against nothing.
    """
    return {name: digest for name, digest in committed.items()
            if _FRAME_NAME.fullmatch(name)}


def report_latents(latents: dict) -> int:
    """Print each latent record and return the code its byte length earns.

    0 when every record's written length equals the product of its own recorded
    shape and itemsize, 62 otherwise.

    The key is `raw_native` everywhere, because `write_latent` writes that name.
    Two reads here spelled it `raw_bf16`, left over from an incomplete rename,
    and `latents` always holds at least one entry -- so the loop always ran and
    always raised `KeyError` on the first record. That killed the byte
    assertion, the frame control below it and the manifest, every control in
    this file, before any of them executed, and exited rc 1, which is not one of
    the codes this module documents. `tests/scripts/test_ltx2_latent_dump.py`
    now holds every key this function reads against the keys `write_latent`
    writes, so the next rename cannot half-land either.
    """
    for name, rec in latents.items():
        print(f"  {name:6s} shape={rec['shape']} dtype={rec['torch_dtype']} "
              f"expected={rec['expected_bytes']} raw={rec['raw_native']['bytes']}")
        if rec["raw_native"]["bytes"] != rec["expected_bytes"]:
            print(f"FATAL: {name} latent wrote {rec['raw_native']['bytes']} bytes for a "
                  f"shape that needs {rec['expected_bytes']}. A short write reads "
                  f"exactly like a successful one to every later consumer.")
            return 62
    return 0


def check_frames(frames_dir: Path, committed: dict[str, str], n_frames: int) -> dict:
    """Hold the decoded frames against EVERY frame the committed sums name.

    The comparison runs over the EXPECTATION set, not over the observation set.
    An earlier revision iterated `frames_dir.glob("frame_*.ppm")` and counted
    disagreements among whatever appeared, so a decode that produced no frames
    yielded an empty glob, no mismatches, `matched = n_frames - 0 = 0 - 0 = 0`,
    and the caller printed "CONTROL PASSED: this run reproduces the committed
    reference byte for byte" and returned 0. An `av` failure, a truncated mp4 or
    a wrong output path each read as a PASSING control. That is precisely the
    defect this module's docstring names -- a counter that could not be written
    reads exactly like a counter that was never incremented -- and upstream's
    own `ltx2_oracle.py` `main()` aborts on zero frames where this driver had
    dropped the check.

    So a committed frame that did not appear is a mismatch, a decoded frame the
    committed file does not name is a mismatch, and `ok` additionally requires
    that `matched` reach the derived expectation. An expectation carrying no
    frame rows at all cannot pass either, because a vacuous comparison is not
    evidence.
    """
    expected = committed_frame_digests(committed)
    observed = {path.name: path for path in sorted(frames_dir.glob("frame_*.ppm"))}
    mismatches: list[tuple[str, str, str]] = []
    matched = 0
    for name in sorted(expected):
        path = observed.get(name)
        if path is None:
            mismatches.append((name, expected[name], "MISSING: this run decoded no such frame"))
            continue
        got = sha256_file(path)
        if got == expected[name]:
            matched += 1
        else:
            mismatches.append((name, expected[name], got))
    for name in sorted(observed):
        if name not in expected:
            mismatches.append((name, "not in SHA256SUMS", sha256_file(observed[name])))
    return {
        "committed_entries": len(committed),
        "committed_frames": len(expected),
        "frames_decoded": n_frames,
        "frames_observed": len(observed),
        "matched": matched,
        "ok": bool(expected) and not mismatches and matched == len(expected),
        "mismatches": [
            {"frame": m[0], "expected": m[1], "got": m[2]} for m in mismatches
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ltx2-source", required=True, type=Path)
    ap.add_argument("--checkpoints", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--offload", default="cpu", choices=("none", "cpu", "disk"))
    ap.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("=== IDENTITY (asserted before any weight is opened) ===")
    try:
        identity = assert_identity(args.ltx2_source)
    except SystemExit as exc:
        print(f"FATAL: identity assertion failed: {exc}")
        return 60

    import torch  # noqa: F401  (imported after identity, like ltx2_oracle does)
    import ltx_pipelines.ti2vid_one_stage as one_stage

    holder = install_recorder(one_stage)

    video = args.out / "upstream-render.mp4"
    argv = [
        "ti2vid_one_stage",
        "--transformer-path", str(args.checkpoints / COMPONENTS["transformer"]),
        "--text-encoder-path", str(args.checkpoints / COMPONENTS["text_encoder"]),
        "--video-vae-path", str(args.checkpoints / COMPONENTS["video_vae"]),
        "--audio-vae-path", str(args.checkpoints / COMPONENTS["audio_vae"]),
        "--height", str(HEIGHT), "--width", str(WIDTH),
        "--num-frames", str(NUM_FRAMES),
        "--num-inference-steps", str(args.num_inference_steps),
        "--offload", args.offload,
        "--seed", str(SEED),
        "--prompt", PROMPT,
        "--output-path", str(video),
    ]
    print("=== RENDER (upstream main(), in-process, recording stage installed) ===")
    print("  " + " ".join(argv))
    saved_argv = sys.argv
    sys.argv = argv
    t0 = time.time()
    try:
        one_stage.main()
    except BaseException as exc:  # noqa: BLE001 - the exit code is the report
        print(f"FATAL: upstream main() raised {type(exc).__name__}: {exc}")
        return 63
    finally:
        sys.argv = saved_argv
    render_s = time.time() - t0
    print(f"RENDER secs={render_s:.0f}")

    pipeline = holder.get("pipeline")
    stage = getattr(pipeline, "stage", None)
    states = list(getattr(stage, "video_states", []) or [])
    print(f"=== RECORDED STATES: {len(states)} ===")
    if not states:
        print("FATAL: the recording stage captured no video state. The hook never "
              "fired, so there is nothing to dump. This is a finding about the "
              "instrument and NOT a result about the render.")
        return 61

    print("=== LATENT DUMP ===")
    latents = {"video": write_latent(states[-1].latent, args.out, "video_latent")}
    audio_state = (getattr(stage, "audio_states", []) or [None])[-1]
    if audio_state is not None and getattr(audio_state, "latent", None) is not None:
        latents["audio"] = write_latent(audio_state.latent, args.out, "audio_latent")
    rc = report_latents(latents)
    if rc:
        return rc

    print("=== DECODE (the control: these frames must match the committed digests) ===")
    frames_dir = args.out / "recorded_frames"
    n_frames = decode(video, frames_dir)
    committed = read_committed_digests(COMMITTED_SUMS)
    if not committed:
        print(f"FATAL: parsed ZERO digests out of {COMMITTED_SUMS}. An empty "
              f"expectation would make this check vacuous and still exit 0.")
        return 64
    frame_check = check_frames(frames_dir, committed, n_frames)
    if not frame_check["committed_frames"]:
        print(f"FATAL: parsed ZERO frame_*.ppm rows out of {COMMITTED_SUMS}. The "
              f"file has non-frame rows, so it parsed, but an expectation naming "
              f"no frame cannot be compared against and must not read as a pass.")
        return 64
    print(f"  committed frames = {frame_check['committed_frames']}, decoded = "
          f"{n_frames}, matched committed digests = {frame_check['matched']} of "
          f"{frame_check['committed_frames']}")

    manifest = {
        "issue": 2514,
        "row": "LTX25-ADHERENCE-BISECT",
        "oracle": "ltx-2",
        "identity": identity,
        "request": {"prompt": PROMPT, "height": HEIGHT, "width": WIDTH,
                    "num_frames": NUM_FRAMES,
                    "num_inference_steps": args.num_inference_steps,
                    "seed": SEED, "offload": args.offload, "device": args.device},
        "environment": environment_record(torch),
        "latents": latents,
        "frame_control": frame_check,
        "render_seconds": round(render_s, 1),
        "total_seconds": round(time.time() - started, 1),
        "generated_by": "tools/oracle/ltx2_latent_dump.py",
    }
    (args.out / "ltx2_latent_dump_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["frame_control"], indent=2))

    if not frame_check["ok"]:
        print(f"FATAL: the recorded run's frames do NOT reproduce the committed "
              f"reference: {frame_check['matched']} of "
              f"{frame_check['committed_frames']} committed frames matched. Either "
              f"a committed frame was never decoded, or the recording substitution "
              f"perturbed the render, or this request is not reproducible on this "
              f"worker. The dumped latent is therefore NOT the reference render's "
              f"latent and must not be cross-decoded against it.")
        for m in frame_check["mismatches"][:5]:
            print(f"  {m['frame']}: expected {m['expected']} got {m['got']}")
        return 64

    print("CONTROL PASSED: this run reproduces the committed reference byte for "
          "byte, so the recording hook changed nothing and the dumped latent IS "
          "the reference render's latent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

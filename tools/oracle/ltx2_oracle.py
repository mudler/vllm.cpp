#!/usr/bin/env python3
"""Lightricks `LTX-2` oracle — stand it up, prove it RUNS, record what it ran on.

This is the reproducible recipe behind `.agents/oracles/ltx-2.md`. It resolves the
LTX-2.5 checkpoints from a local directory, asserts the upstream revision *before*
it opens a weight file, runs one FIXED deterministic text-to-video request through
upstream's own one-stage pipeline, and writes the render plus a manifest naming
every byte it consumed.

Why this script exists at all. Until it ran, `.agents/oracles/ltx-2.md` recorded
`gateable = no`, and [#1864](https://github.com/mudler/vllm.cpp/issues/1864) is
what that field owed. Thirteen tracked scripts already imported `ltx_core` /
`ltx_pipelines`, and every one of them ran individual *modules* at reduced
dimensions on synthetic PRNG weights — the committed goldens say so in their own
headers ("no weight byte is checked in"). AGENTS.md sets the bar at "demonstrably
builds and runs the model. Constructing a config proves nothing."

Three things this script refuses to assume.

1. **The oracle's identity is asserted, not trusted, and asserted where it
   matters.** #1864's second finding was that this lane asserted the revision only
   where no weight is loaded, and asserted only the interpreter *path* where
   weights are loaded. `.agents/specs/ltx-2-5.md` §7.0(b) records a decoy
   `ltx_core` that produced byte-identical goldens and exited 0. So both halves are
   checked here, in one place, before the first weight file is opened: the source
   tree's `git rev-parse HEAD` must equal `EXPECTED_LTX2_COMMIT`, *and* every
   imported `ltx_*` module must resolve inside that same tree.

2. **A repo id is not a pin.** [#1723](https://github.com/mudler/vllm.cpp/issues/1723)
   found an LTX-2.5 checkpoint re-quantized in place under an unchanged name
   (+116,384 bytes, one extra tensor), and every measurement on that lane had
   loaded the earlier file. Every checkpoint this script opens is therefore
   recorded by size *and* sha256 in the manifest. The sha256 is the identity; the
   revision is a label.

3. **A render that produced no frames is a failure, not a pass.** The output is
   decoded and counted, and a run that wrote zero frames aborts rather than
   leaving a manifest that implies success.

Usage (see tools/oracle/README.md for the venv recipe):

    python3 tools/oracle/ltx2_oracle.py \
        --ltx2-source /tmp/ltx2/LTX-2 \
        --checkpoints "$CHECKPOINT_ROOT/ltx-2.5" \
        --out /tmp/ltx2/oracle-out \
        --device cuda

Issue: #1864.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Required alongside any LTX quantization arm, and harmless otherwise
# (upstream docs/optimization.md:7).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# The pin recorded in `.agents/oracles/ltx-2.md` — the merge of Lightricks/LTX-2#273,
# authored 2026-08-11. The repository carries no tags, so a commit is the only
# identity it has. NEVER a branch name: `main` moves.
EXPECTED_LTX2_COMMIT = "fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca"

# --- The fixed request -------------------------------------------------------
# Deliberately the cheapest generation that still drives the whole chain: the
# Gemma-4 text tower, the joint video+audio DiT denoise loop, the conv video VAE
# and the audio VAE. Held constant so runs captured on different days compare.
#
# Geometry is not free-form. One-stage asserts height and width divisible by 32
# (upstream helpers.py:540-551) and the causal VAE temporal grid forces
# num_frames = 8k+1 (helpers.py:554-562, time scale factor 8). 320x192x25 is the
# geometry `docs/models/ltx-2-5.md` already uses on our side, so the two engines
# are asked the same question.
PROMPT = "A red fox walks slowly through a snowy pine forest at sunrise, cinematic."
HEIGHT = 192
WIDTH = 320
NUM_FRAMES = 25
NUM_INFERENCE_STEPS = 8
SEED = 42

# The four slots a one-stage render actually consumes. `duration_head` is the
# fifth ModelPaths slot and is optional: it is legal as None, and an explicit
# --num-frames is then required (upstream blocks.py:894-905).
COMPONENTS = {
    "transformer": "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
}

# Sizes as served by Lightricks/LTX-2.5 @ 6c7e5e573ac1667efc83407806fe9b0b93730e60.
# A mismatch here is the #1723 shape and is reported, never silently accepted.
EXPECTED_SIZES = {
    "transformer": 42_018_190_584,
    "text_encoder": 26_263_858_182,
    "video_vae": 1_452_269_922,
    "audio_vae": 364_866_540,
}


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def assert_identity(source: Path) -> dict:
    """Assert the upstream revision AND the resolved import paths, before any
    weight file is opened. Either half alone has already been fooled here."""
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != EXPECTED_LTX2_COMMIT:
        raise SystemExit(
            f"FATAL: upstream revision mismatch.\n"
            f"  {source} is at {head}\n"
            f"  .agents/oracles/ltx-2.md pins {EXPECTED_LTX2_COMMIT}\n"
            f"Advance the pin deliberately, or check out the pinned revision. "
            f"Do not measure against an unpinned tree."
        )
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"],
        capture_output=True, text=True,
    ).stdout.strip()

    origins = {}
    src_resolved = source.resolve()
    for mod in ("ltx_core", "ltx_pipelines"):
        spec = importlib.util.find_spec(mod)
        if spec is None or not spec.origin:
            raise SystemExit(f"FATAL: {mod} is not importable; install the pinned tree.")
        origin = Path(spec.origin).resolve()
        if src_resolved not in origin.parents:
            raise SystemExit(
                f"FATAL: {mod} resolves to {origin}, which is OUTSIDE the pinned "
                f"tree {src_resolved}. That is the decoy-`ltx_core` failure "
                f"`.agents/specs/ltx-2-5.md` §7.0(b) records."
            )
        origins[mod] = str(origin)

    print(f"IDENTITY_OK  revision={head}")
    print(f"             worktree={'DIRTY -- ' + dirty[:80] if dirty else 'clean'}")
    for m, o in origins.items():
        print(f"             {m} -> {o}")
    return {"revision": head, "worktree_dirty": bool(dirty), "module_origins": origins}


def resolve_components(root: Path) -> dict:
    """Resolve, size-check and sha256 every checkpoint BEFORE the pipeline runs."""
    out = {}
    for name, rel in COMPONENTS.items():
        p = root / rel
        if not p.is_file():
            raise SystemExit(f"FATAL: {name} missing at {p}")
        size = p.stat().st_size
        expect = EXPECTED_SIZES[name]
        note = "SIZE_OK" if size == expect else f"SIZE_DIFFERS (expected {expect:,})"
        t0 = time.time()
        digest = sha256_file(p)
        print(f"  {name:13s} {size:>14,} bytes  {note}")
        print(f"                sha256={digest}  ({time.time() - t0:.0f}s)")
        out[name] = {
            "path": str(p),
            "bytes": size,
            "expected_bytes": expect,
            "size_matches_expected": size == expect,
            "sha256": digest,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ltx2-source", required=True, type=Path,
                    help="clone of Lightricks/LTX-2 checked out at the pin")
    ap.add_argument("--checkpoints", required=True, type=Path,
                    help="directory holding the LTX-2.5 slots (HF repo layout)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--offload", default="cpu", choices=("none", "cpu", "disk"),
                    help="upstream OffloadMode. 'cpu' streams the tower and the "
                         "DiT layer-by-layer (~5 GB VRAM); 'none' needs the whole "
                         "DiT resident, which OOM-reboots a GB10.")
    ap.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    ap.add_argument("--skip-sha256", action="store_true",
                    help="skip the digests only when re-running against bytes "
                         "already recorded in an earlier manifest")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("=== IDENTITY (asserted before any weight is opened) ===")
    identity = assert_identity(args.ltx2_source)

    print("=== CHECKPOINTS ===")
    if args.skip_sha256:
        comps = {}
        for name, rel in COMPONENTS.items():
            p = args.checkpoints / rel
            comps[name] = {"path": str(p), "bytes": p.stat().st_size,
                           "sha256": None, "expected_bytes": EXPECTED_SIZES[name],
                           "size_matches_expected":
                               p.stat().st_size == EXPECTED_SIZES[name]}
            print(f"  {name:13s} {comps[name]['bytes']:>14,} bytes  (sha256 skipped)")
    else:
        comps = resolve_components(args.checkpoints)

    print("=== RENDER ===")
    video = args.out / "upstream-render.mp4"
    argv = [
        sys.executable, "-m", "ltx_pipelines.ti2vid_one_stage",
        "--transformer-path", comps["transformer"]["path"],
        "--text-encoder-path", comps["text_encoder"]["path"],
        "--video-vae-path", comps["video_vae"]["path"],
        "--audio-vae-path", comps["audio_vae"]["path"],
        "--height", str(HEIGHT), "--width", str(WIDTH),
        "--num-frames", str(NUM_FRAMES),
        "--num-inference-steps", str(args.num_inference_steps),
        "--offload", args.offload,
        "--seed", str(SEED),
        "--prompt", PROMPT,
        "--output-path", str(video),
    ]
    print("  " + " ".join(argv))
    t0 = time.time()
    rc = subprocess.call(argv)
    render_s = time.time() - t0
    print(f"RENDER rc={rc} secs={render_s:.0f}")
    if rc != 0 or not video.is_file():
        raise SystemExit(f"FATAL: upstream render failed (rc={rc}); no manifest written.")

    print("=== DECODE (frame_%06d.ppm + audio.wav, the layout "
          "scripts/ltx25-render-compare.py reads) ===")
    frames_dir = args.out / "upstream_frames"
    n_frames = decode(video, frames_dir)
    if n_frames == 0:
        raise SystemExit("FATAL: the render decoded to zero frames. A recorder that "
                         "captured nothing is a broken instrument, not a pass.")

    manifest = {
        "issue": 1864,
        "oracle": "ltx-2",
        "upstream": "https://github.com/Lightricks/LTX-2",
        "identity": identity,
        "checkpoints": comps,
        "request": {
            "prompt": PROMPT, "height": HEIGHT, "width": WIDTH,
            "num_frames": NUM_FRAMES,
            "num_inference_steps": args.num_inference_steps,
            "seed": SEED, "offload": args.offload, "device": args.device,
        },
        "result": {
            "video": str(video),
            "video_bytes": video.stat().st_size,
            "frames_decoded": n_frames,
            "render_seconds": round(render_s, 1),
            "total_seconds": round(time.time() - started, 1),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    try:
        import torch
        manifest["environment"]["torch"] = torch.__version__
        if torch.cuda.is_available():
            manifest["environment"]["gpu"] = torch.cuda.get_device_name(0)
            manifest["environment"]["capability"] = list(
                torch.cuda.get_device_capability(0))
    except Exception as e:  # pragma: no cover - environment probe only
        manifest["environment"]["torch_probe_failed"] = f"{type(e).__name__}: {e}"

    mpath = args.out / "ltx2_oracle_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {mpath}")
    print(f"OK: upstream Lightricks/LTX-2 @ {EXPECTED_LTX2_COMMIT[:9]} rendered "
          f"{n_frames} frames at {WIDTH}x{HEIGHT} in {render_s:.0f}s")
    return 0


def decode(src: Path, out: Path) -> int:
    """Decode the rendered mp4 into the PPM+WAV layout the compare tool reads."""
    import wave

    import av
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    container = av.open(str(src))
    n = 0
    for frame in container.decode(video=0):
        a = frame.to_ndarray(format="rgb24")
        with (out / f"frame_{n:06d}.ppm").open("wb") as fh:
            fh.write(f"P6\n{a.shape[1]} {a.shape[0]}\n255\n".encode())
            fh.write(a.tobytes())
        n += 1
    container.close()
    print(f"  video frames: {n}")

    # The audio half is caught, and the video half is not. The gate this script
    # answers to is "did upstream produce frames"; an audio container quirk that
    # raised here would throw away a render that had already succeeded, together
    # with the manifest and the lease that paid for it. The failure is printed
    # and carried, never swallowed silently.
    try:
        container = av.open(str(src))
        if container.streams.audio:
            stream = container.streams.audio[0]
            chunks = [f.to_ndarray() for f in container.decode(audio=0)]
            if chunks:
                d = np.concatenate(chunks, axis=-1)
                if d.ndim == 1:
                    d = d[None, :]
                pcm = ((np.clip(d, -1.0, 1.0) * 32767).astype("<i2")
                       if np.issubdtype(d.dtype, np.floating) else d.astype("<i2"))
                with wave.open(str(out / "audio.wav"), "wb") as w:
                    w.setnchannels(pcm.shape[0])
                    w.setsampwidth(2)
                    w.setframerate(stream.rate)
                    w.writeframes(pcm.T.tobytes())
                print(f"  audio.wav: rate={stream.rate} shape={pcm.shape}")
            else:
                print("  audio stream present but zero frames decoded")
        else:
            print("  no audio stream in the container")
        container.close()
    except Exception as e:  # noqa: BLE001 - reported, and never fatal to a render
        print(f"  AUDIO DECODE FAILED (video frames are unaffected): "
              f"{type(e).__name__}: {e}")
    return n


if __name__ == "__main__":
    sys.exit(main())

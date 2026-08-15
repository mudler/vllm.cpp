#!/usr/bin/env python3
"""MiniMax-Music3 `diffusers` oracle — stand it up, prove it runs, capture goldens.

This is the reproducible recipe behind `.agents/oracles/diffusers.md`. It resolves
a local checkpoint in the *diffusers* packaging, builds the
`MiniMaxMusic3ModularPipeline`, runs one FIXED deterministic request, writes a WAV,
and captures the per-stage reference tensors the gates in
`.agents/specs/minimax-music3.md` §5 are written against.

Three things this script refuses to assume.

1. **The oracle's identity is asserted, not trusted.** The primary oracle is an
   *unmerged* diffusers PR (huggingface/diffusers#14456), so the pin is a commit,
   not a branch. A venv can silently hold a different revision, and this project
   has been burned by exactly that. The installed distribution's recorded VCS
   commit (or the source tree's `git rev-parse HEAD`) must equal
   `EXPECTED_DIFFUSERS_COMMIT` or the run aborts before loading a single weight.

2. **The spec is evidence, not scripture.** Every geometry, dtype and parameter
   count `.agents/specs/minimax-music3.md` §1 claims is re-checked against the
   *loaded* modules, and a disagreement is printed loudly (and, unless
   `--allow-spec-disagreement`, fails the run).

3. **A recorder that captured nothing is a broken instrument, not a pass.** Every
   capture hook asserts it fired; a stage that recorded zero tensors aborts the
   run rather than writing a manifest that silently omits it.

Usage (see tools/oracle/README.md for the venv recipe):

    python3 tools/oracle/music3_oracle.py \
        --checkpoint "$CHECKPOINT_ROOT/minimax-music3" \
        --out tests/parity/goldens/minimax_music3_oracle \
        --device cuda

Issue: #672.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Resolve components from the local checkpoint only. `modular_model_index.json`
# embeds the hub repo id in every ComponentSpec, so without this a typo in
# --checkpoint silently becomes a 57 GB hub download of *some* revision.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402

# The pinned head of huggingface/diffusers#14456 (branch
# huggingface:minimax-music3-integration). NOT on diffusers main.
EXPECTED_DIFFUSERS_COMMIT = "c6da9936e4bda83107943a16eb8682e9a37d8527"

# --- The fixed request -------------------------------------------------------
# Deliberately the *shortest* generation that still drives every stage: the AR
# language model, the depth decoder's 8 codebooks, the condition mix, the DiT
# denoise loop, the vocoder, and the stitch. Held constant so goldens captured on
# different days are comparable.
PROMPT = (
    "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate. "
    "Vocals: soft female lead, close and breathy. "
    "Arrangement: fingerpicked guitar and soft piano."
)
LYRICS = "[verse]\nMorning light filtering through the pine\n"

# The on-disk dtype policy of the released diffusers packaging, which is upstream's
# resolved choice, not ours: `scripts/convert_minimax_music3_to_diffusers.py`
# defaults `--dtype float32` for the transformer / condition encoder / vocoder and
# forces the RVQ depth decoder to bf16 (spec §2.1). The language model ships bf16.
#
# MEASURED 2026-08-14 by this script: loading every component AT its on-disk dtype
# does NOT run. The pipeline casts in exactly two places — `denoise.py:83`
# condition -> transformer.dtype and `decoders.py:84` latents -> vocoder.dtype —
# and nowhere else, so `condition_encoder` and `rvq_depth_decoder` both consume the
# language model's hidden states uncast:
#     RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the
#     same   [condition_embedder_minimax_music3.py:64, self.proj]
# Every runnable configuration therefore satisfies
#     dtype(language_model) == dtype(rvq_depth_decoder) == dtype(condition_encoder)
# with the transformer and the vocoder free. Kept selectable so that finding stays
# reproducible instead of becoming folklore.
ON_DISK_DTYPES = {
    "language_model": torch.bfloat16,
    "rvq_depth_decoder": torch.bfloat16,
    "default": torch.float32,
}

# The gateable configuration: the autoregressive half at its on-disk bf16, the
# acoustic half in fp32 — upstream's conversion default for the DiT and the vocoder
# (spec §2.1), and what SGLang-Omni's README says it runs ("the acoustic stage in
# FP32"). The condition mix is the one component narrowed from its on-disk fp32,
# and only because upstream's pipeline requires it to match the language model; its
# output is cast straight back to fp32 at `denoise.py:83`.
REFERENCE_DTYPES = {
    "language_model": torch.bfloat16,
    "rvq_depth_decoder": torch.bfloat16,
    "condition_encoder": torch.bfloat16,
    "default": torch.float32,
}

# Spec §1, the measured table. Checked against the loaded modules below.
SPEC_FACTS = {
    "transformer_params": 2_400_000_000,      # "2.4B", fp32 on disk
    "rvq_depth_decoder_params": 646_000_000,  # "0.646B", bf16 on disk
    "condition_encoder_tensors": 4,
    "vocoder_params": 54_000_000,
    "language_model_params": 8_584_475_648,   # from the shard index
    "sampling_rate": 44100,
    "channels": 2,
}


# --- Environment -------------------------------------------------------------
def _run(cmd: list[str], cwd: str | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def resolve_diffusers_commit(diffusers_module) -> tuple[str | None, str]:
    """Return (commit, how) for the *installed* diffusers, never for a branch name."""
    try:
        import importlib.metadata as md

        raw = md.distribution("diffusers").read_text("direct_url.json")
        if raw:
            info = json.loads(raw).get("vcs_info") or {}
            commit = info.get("commit_id")
            if commit:
                return commit, "dist-info/direct_url.json"
    except Exception:  # noqa: BLE001 - fall through to the git probe
        pass

    src_root = Path(diffusers_module.__file__).resolve().parents[2]
    if (src_root / ".git").exists():
        commit = _run(["git", "rev-parse", "HEAD"], cwd=str(src_root))
        if commit:
            return commit, f"git rev-parse HEAD in {src_root}"
    return None, "unresolved"


def collect_environment(args) -> dict:
    import diffusers
    import transformers

    commit, how = resolve_diffusers_commit(diffusers)
    env = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()}-{platform.release()}-{platform.machine()}",
        "host": platform.node(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "diffusers_version": diffusers.__version__,
        "diffusers_commit": commit,
        "diffusers_commit_source": how,
        "diffusers_path": diffusers.__file__,
        "numpy": np.__version__,
        "device": args.device,
        "device_name": None,
        "dtype_policy": args.dtype_policy,
    }
    if args.device.startswith("cuda") and torch.cuda.is_available():
        env["device_name"] = torch.cuda.get_device_name(0)
        env["cuda_capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
    return env


def assert_oracle_identity(env: dict, expected: str) -> None:
    commit = env["diffusers_commit"]
    if commit is None:
        raise SystemExit(
            "REFUSING TO RUN: could not resolve the installed diffusers commit "
            f"(probed dist-info/direct_url.json and the source tree at {env['diffusers_path']}). "
            "An oracle whose revision cannot be read is not an oracle."
        )
    if commit != expected:
        raise SystemExit(
            f"REFUSING TO RUN: diffusers is at {commit} ({env['diffusers_commit_source']}), "
            f"expected the pinned {expected}. Reinstall with:\n"
            f"  pip install 'git+https://github.com/huggingface/diffusers@{expected}'"
        )
    print(f"[oracle] diffusers commit ASSERTED {commit} via {env['diffusers_commit_source']}")


# --- Checkpoint --------------------------------------------------------------
COMPONENT_DIRS = {
    "condition_encoder": ["config.json", "diffusion_pytorch_model.safetensors"],
    "language_model": ["config.json", "model.safetensors.index.json"],
    "rvq_depth_decoder": ["config.json", "diffusion_pytorch_model.safetensors"],
    "scheduler": ["scheduler_config.json"],
    "tokenizer": ["tokenizer.json", "tokenizer_config.json"],
    "transformer": ["config.json", "diffusion_pytorch_model.safetensors.index.json"],
    "vocoder": ["config.json", "diffusion_pytorch_model.safetensors"],
}


def verify_checkpoint(root: Path) -> dict:
    """Fail loudly on a partial download rather than half way through a 20 minute run."""
    problems: list[str] = []
    if not (root / "modular_model_index.json").is_file():
        problems.append(f"missing {root / 'modular_model_index.json'} — not a diffusers-arm checkpoint")

    for name, required in COMPONENT_DIRS.items():
        directory = root / name
        if not directory.is_dir():
            problems.append(f"missing component directory {directory}")
            continue
        for filename in required:
            if not (directory / filename).is_file():
                problems.append(f"missing {directory / filename}")

    # Shard indices must be complete: a truncated download presents as a missing shard.
    for name, index_name in (
        ("language_model", "model.safetensors.index.json"),
        ("transformer", "diffusion_pytorch_model.safetensors.index.json"),
    ):
        index_path = root / name / index_name
        if not index_path.is_file():
            continue
        index = json.loads(index_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
        present = 0
        for shard in shards:
            shard_path = root / name / shard
            if shard_path.is_file():
                present += 1
            else:
                problems.append(f"missing shard {shard_path}")
        declared = (index.get("metadata") or {}).get("total_size")
        if declared is not None and present == len(shards):
            actual = sum((root / name / shard).stat().st_size for shard in shards)
            # safetensors files carry a header on top of the tensor bytes.
            if actual < declared:
                problems.append(
                    f"{name} shards total {actual} bytes, index declares {declared} — download incomplete"
                )

    if any(p.name.endswith(".incomplete") for p in root.rglob("*.incomplete")):
        problems.append(f"{root} still holds *.incomplete files — a download is in flight")

    if problems:
        raise SystemExit("CHECKPOINT INCOMPLETE:\n  " + "\n  ".join(problems))

    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts)
    print(f"[oracle] checkpoint verified: {root} ({total / 2**30:.2f} GiB across {len(COMPONENT_DIRS)} components)")
    return {"path": str(root), "bytes": total}


# --- Spec §1 fact check ------------------------------------------------------
def _param_stats(module) -> tuple[int, list[str]]:
    total = sum(p.numel() for p in module.parameters())
    dtypes = sorted({str(p.dtype) for p in module.parameters()})
    return total, dtypes


def check_spec_facts(pipe, args) -> tuple[dict, list[str]]:
    """Re-derive spec §1 from the loaded modules. Returns (facts, disagreements)."""
    facts: dict = {}
    bad: list[str] = []

    def near(actual: int, claimed: int, tol: float = 0.02) -> bool:
        return abs(actual - claimed) <= tol * claimed

    # transformer — 2.4B, fp32
    n, dtypes = _param_stats(pipe.transformer)
    cfg = pipe.transformer.config
    facts["transformer"] = {
        "params": n,
        "dtypes": dtypes,
        "num_layers": getattr(cfg, "num_layers", None),
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "attention_head_dim": getattr(cfg, "attention_head_dim", None),
        "in_channels": getattr(cfg, "in_channels", None),
        "condition_dim": getattr(cfg, "condition_dim", None),
        "rotary_dim": getattr(cfg, "rotary_dim", None),
        "ff_inner_dim": getattr(cfg, "ff_inner_dim", None),
    }
    if not near(n, SPEC_FACTS["transformer_params"]):
        bad.append(f"transformer params {n:,} vs spec §1 ~{SPEC_FACTS['transformer_params']:,}")
    if args.dtype_policy in ("on-disk", "reference") and dtypes != ["torch.float32"]:
        bad.append(f"transformer dtypes {dtypes} vs spec §1 F32")

    # rvq_depth_decoder — 0.646B, bf16
    n, dtypes = _param_stats(pipe.rvq_depth_decoder)
    cfg = pipe.rvq_depth_decoder.config
    facts["rvq_depth_decoder"] = {
        "params": n,
        "dtypes": dtypes,
        "num_codebooks": getattr(cfg, "num_codebooks", None),
        "audio_vocab_size": getattr(cfg, "audio_vocab_size", None),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
    }
    if not near(n, SPEC_FACTS["rvq_depth_decoder_params"]):
        bad.append(f"rvq_depth_decoder params {n:,} vs spec §1 ~{SPEC_FACTS['rvq_depth_decoder_params']:,}")
    if args.dtype_policy in ("on-disk", "reference") and dtypes != ["torch.bfloat16"]:
        bad.append(f"rvq_depth_decoder dtypes {dtypes} vs spec §1 BF16")

    # condition_encoder — exactly four tensors, a learned mix and not a tower
    names = sorted(pipe.condition_encoder.state_dict().keys())
    n, dtypes = _param_stats(pipe.condition_encoder)
    facts["condition_encoder"] = {
        "tensor_names": names,
        "params": n,
        "dtypes": dtypes,
        "num_condition_layers": pipe.condition_encoder.config.num_condition_layers,
        "condition_hidden_dim": pipe.condition_encoder.config.condition_hidden_dim,
        "out_dim": pipe.condition_encoder.config.out_dim,
        "input_sampling_rate": pipe.condition_encoder.config.input_sampling_rate,
        "input_hop_length": pipe.condition_encoder.config.input_hop_length,
        "output_sampling_rate": pipe.condition_encoder.config.output_sampling_rate,
        "output_hop_length": pipe.condition_encoder.config.output_hop_length,
    }
    if len(names) != SPEC_FACTS["condition_encoder_tensors"]:
        bad.append(f"condition_encoder has {len(names)} tensors {names} vs spec §1 four")

    # vocoder — weight-norm (`weight_g`/`weight_v`) and the [8,8,4,2] stack
    wn = [k for k in pipe.vocoder.state_dict() if k.endswith("weight_g") or k.endswith("weight_v")]
    n, dtypes = _param_stats(pipe.vocoder)
    facts["vocoder"] = {
        "params": n,
        "dtypes": dtypes,
        "weight_norm_tensors": len(wn),
        "upsampling_ratios": list(pipe.vocoder.config.upsampling_ratios),
        "latent_channels": pipe.vocoder.config.latent_channels,
        "sampling_rate": int(pipe.vocoder.config.sampling_rate),
    }
    if not wn:
        bad.append("vocoder exposes no weight_g/weight_v pair — spec §1 says weight-norm")
    hop = 1
    for r in pipe.vocoder.config.upsampling_ratios:
        hop *= int(r)
    facts["vocoder"]["derived_hop"] = hop
    if hop != pipe.latent_hop_length:
        bad.append(f"vocoder upsampling product {hop} != latent_hop_length {pipe.latent_hop_length}")

    # language_model — the Qwen3 half at vocab 200000
    n, dtypes = _param_stats(pipe.language_model)
    lm_cfg = pipe.language_model.config
    facts["language_model"] = {
        "params": n,
        "dtypes": dtypes,
        "class": type(pipe.language_model).__name__,
        "vocab_size": lm_cfg.vocab_size,
        "num_hidden_layers": lm_cfg.num_hidden_layers,
        "hidden_size": lm_cfg.hidden_size,
        "num_attention_heads": lm_cfg.num_attention_heads,
        "num_key_value_heads": lm_cfg.num_key_value_heads,
        "tie_word_embeddings": lm_cfg.tie_word_embeddings,
    }
    if not near(n, SPEC_FACTS["language_model_params"]):
        bad.append(f"language_model params {n:,} vs spec §1 {SPEC_FACTS['language_model_params']:,}")

    # §1.1 — the rate is the vocoder's, resample-free
    facts["pipeline"] = {
        "sampling_rate": pipe.sampling_rate,
        "frame_rate": pipe.frame_rate,
        "latent_hop_length": pipe.latent_hop_length,
        "num_codebooks": pipe.num_codebooks,
        "audio_vocab_size": pipe.audio_vocab_size,
        "num_channels_latents": pipe.num_channels_latents,
        "scheduler": type(pipe.scheduler).__name__,
        "scheduler_config": {
            k: v for k, v in dict(pipe.scheduler.config).items() if not k.startswith("_")
        },
    }
    if pipe.sampling_rate != SPEC_FACTS["sampling_rate"]:
        bad.append(f"pipeline sampling_rate {pipe.sampling_rate} vs spec §1.1 {SPEC_FACTS['sampling_rate']}")
    derived = pipe.frame_rate * (
        pipe.condition_encoder.config.output_sampling_rate
        / pipe.condition_encoder.config.input_sampling_rate
        * pipe.condition_encoder.config.input_hop_length
        / pipe.condition_encoder.config.output_hop_length
    )
    facts["pipeline"]["derived_latent_frame_rate"] = derived
    facts["pipeline"]["derived_sampling_rate"] = derived * hop
    if abs(derived * hop - SPEC_FACTS["sampling_rate"]) > 1.0:
        bad.append(f"latent frame rate x hop = {derived * hop} != {SPEC_FACTS['sampling_rate']}")

    return facts, bad


# --- Capture -----------------------------------------------------------------
class StageRecorder:
    """Records the per-stage tensors the spec §5 gates need.

    None of these are pipeline outputs, so each is taken at its source: a module
    forward hook where the stage *is* a module, and a scoped wrapper where it is a
    function. Every recorder is asserted non-empty afterwards — a hook that never
    fired reports as a clean run otherwise.
    """

    def __init__(self, pipe):
        self.pipe = pipe
        self.condition_inputs: list[torch.Tensor] = []
        self.condition_outputs: list[torch.Tensor] = []
        self.rvq_codes: list[torch.Tensor] = []
        self.step_records: list[dict] = []
        self.vocoder_inputs: list[torch.Tensor] = []
        self._handles: list = []
        self._patched: list[tuple] = []

    def __enter__(self):
        def condition_hook(_module, inputs, output):
            self.condition_inputs.append(inputs[0].detach().to("cpu", torch.float32))
            self.condition_outputs.append(output.detach().to("cpu", torch.float32))

        def vocoder_pre_hook(_module, inputs):
            self.vocoder_inputs.append(inputs[0].detach().to("cpu", torch.float32))

        self._handles.append(self.pipe.condition_encoder.register_forward_hook(condition_hook))
        self._handles.append(self.pipe.vocoder.register_forward_pre_hook(vocoder_pre_hook))

        # The RVQ codes never leave the autoregressive loop, so wrap the module-level
        # helper the generation step calls by name.
        from diffusers.modular_pipelines.minimax_music3 import encoders as mm3_encoders

        original_depth = mm3_encoders._generate_depth_codes

        def wrapped_depth(components, last_hidden, semantic_code, generator):
            codes, hidden = original_depth(components, last_hidden, semantic_code, generator)
            # Rows are [conditional, unconditional] and identical by construction.
            self.rvq_codes.append(codes[:1].detach().to("cpu", torch.int32).clone())
            return codes, hidden

        mm3_encoders._generate_depth_codes = wrapped_depth
        self._patched.append((mm3_encoders, "_generate_depth_codes", original_depth))

        # Per-step latents: the scheduler's own output is the denoise trajectory.
        scheduler = self.pipe.scheduler
        original_step = scheduler.step

        def wrapped_step(model_output, timestep, sample, *rest, **kwargs):
            result = original_step(model_output, timestep, sample, *rest, **kwargs)
            latents = result[0] if isinstance(result, tuple) else result.prev_sample
            self.step_records.append(
                {
                    "timestep": float(timestep),
                    "sample_in": sample.detach().to("cpu", torch.float32).clone(),
                    "velocity": model_output.detach().to("cpu", torch.float32).clone(),
                    "latents_out": latents.detach().to("cpu", torch.float32).clone(),
                }
            )
            return result

        scheduler.step = wrapped_step
        self._patched.append((scheduler, "step", None))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        for target, name, original in self._patched:
            if original is None:
                # Instance attribute shadowing a bound method: delete to restore.
                try:
                    delattr(target, name)
                except AttributeError:
                    pass
            else:
                setattr(target, name, original)
        return False

    def assert_armed(self) -> None:
        empty = [
            name
            for name, seq in (
                ("condition_encoder forward hook", self.condition_outputs),
                ("rvq depth-code wrapper", self.rvq_codes),
                ("scheduler.step wrapper", self.step_records),
                ("vocoder forward pre-hook", self.vocoder_inputs),
            )
            if not seq
        ]
        if empty:
            raise SystemExit(
                "BROKEN INSTRUMENT: these recorders captured nothing, so the run cannot be "
                "reported as a per-stage capture: " + ", ".join(empty)
            )


# --- Goldens -----------------------------------------------------------------
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_tensor(out_dir: Path, entries: dict, name: str, tensor: torch.Tensor, note: str) -> None:
    """Store as .npy. bf16 has no numpy dtype, so it is widened to f32 — exact and
    reversible — and the on-device dtype is recorded next to it."""
    original_dtype = str(tensor.dtype)
    array = tensor.detach().cpu()
    if array.dtype in (torch.bfloat16, torch.float16):
        array = array.to(torch.float32)
    array = array.numpy()
    path = out_dir / f"{name}.npy"
    np.save(path, array, allow_pickle=False)
    finite = array[np.isfinite(array)] if array.dtype.kind == "f" else array
    entries[name] = {
        "file": path.name,
        "shape": list(array.shape),
        "stored_dtype": str(array.dtype),
        "device_dtype": original_dtype,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.astype(np.float64).mean()) if finite.size else None,
        "note": note,
    }


# --- Main --------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="local diffusers-arm checkpoint directory")
    parser.add_argument("--out", default=None, help="golden output directory (omit to skip writing goldens)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype-policy",
        default="reference",
        choices=["reference", "on-disk", "bf16", "fp32"],
        help=(
            "'reference' (default, gateable): bf16 AR half, fp32 acoustic half. "
            "'on-disk': the literal disk dtypes — MEASURED NOT RUNNABLE, see ON_DISK_DTYPES. "
            "'bf16': the uniform dtype upstream's own docs prescribe. 'fp32': uniform fp32."
        ),
    )
    parser.add_argument("--audio-duration", type=float, default=1.0, help="seconds; 25 AR frames per second")
    parser.add_argument("--steps", type=int, default=4, help="flow-matching Euler steps per chunk")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--lyrics", default=LYRICS)
    parser.add_argument("--expect-commit", default=EXPECTED_DIFFUSERS_COMMIT)
    parser.add_argument("--facts-only", action="store_true", help="load and check spec §1, do not generate")
    parser.add_argument("--allow-spec-disagreement", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    checkpoint = Path(args.checkpoint).expanduser().resolve()

    env = collect_environment(args)
    assert_oracle_identity(env, args.expect_commit)
    print(json.dumps(env, indent=2, sort_keys=True))

    checkpoint_info = verify_checkpoint(checkpoint)

    from diffusers import ModularPipeline

    dtype_arg = {
        "reference": REFERENCE_DTYPES,
        "on-disk": ON_DISK_DTYPES,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype_policy]

    load_start = time.time()
    pipe = ModularPipeline.from_pretrained(str(checkpoint))
    pipe.load_components(pretrained_model_name_or_path=str(checkpoint), dtype=dtype_arg)
    pipe.to(args.device)
    load_seconds = time.time() - load_start
    print(f"[oracle] components loaded in {load_seconds:.1f}s: {type(pipe).__name__}")

    facts, disagreements = check_spec_facts(pipe, args)
    print(json.dumps(facts, indent=2, sort_keys=True, default=str))
    if disagreements:
        print("\n!!! SPEC §1 DISAGREEMENT — the spec is evidence, not scripture:", file=sys.stderr)
        for item in disagreements:
            print(f"  DISAGREE: {item}", file=sys.stderr)
        if not args.allow_spec_disagreement:
            return 3
    else:
        print("[oracle] spec §1 facts CONFIRMED against the loaded modules")

    if args.facts_only:
        return 0

    # A CPU generator keeps both the AR top-k sampling and the latent noise
    # device-independent (the diffusers convention `_sample_top_k` documents), so
    # the golden does not silently become a per-device artefact.
    generator = torch.Generator("cpu").manual_seed(args.seed)

    run_start = time.time()
    with StageRecorder(pipe) as recorder:
        result = pipe(
            prompt=args.prompt,
            lyrics=args.lyrics,
            audio_duration=args.audio_duration,
            num_inference_steps=args.steps,
            generator=generator,
            output_type="pt",
            output=["audios", "frame_hiddens", "chunk_starts", "latent_chunks"],
        )
        recorder.assert_armed()
    run_seconds = time.time() - run_start

    audios = result["audios"]
    frame_hiddens = result["frame_hiddens"]
    chunk_starts = result["chunk_starts"]
    latent_chunks = result["latent_chunks"]

    sample_rate = pipe.sampling_rate
    duration = audios.shape[-1] / sample_rate
    print(
        f"[oracle] GENERATED in {run_seconds:.1f}s: audios {tuple(audios.shape)} {audios.dtype}, "
        f"{sample_rate} Hz, {duration:.3f} s, {len(chunk_starts)} chunk(s), "
        f"{frame_hiddens.shape[1]} AR frames, {len(recorder.step_records)} denoise steps"
    )
    if audios.ndim != 3 or audios.shape[1] != SPEC_FACTS["channels"]:
        disagreements.append(f"waveform {tuple(audios.shape)} is not (batch, 2, samples) — spec §1.1 says stereo")
    if sample_rate != SPEC_FACTS["sampling_rate"]:
        disagreements.append(f"waveform sample rate {sample_rate} — spec §1.1 says 44100 with no resample")

    if args.out is None:
        return 0

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: dict = {}

    save_tensor(
        out_dir, entries, "frame_hiddens", frame_hiddens[0],
        "AR per-frame hidden states [frames, num_codebooks*hidden]; the condition mix's INPUT",
    )
    save_tensor(
        out_dir, entries, "rvq_codes", torch.cat(recorder.rvq_codes, dim=0),
        "RVQ codes [calls, num_codebooks], column 0 semantic + 7 residual. The first row is the "
        "priming decode step that emits no frame, so rows[1:] align with frame_hiddens",
    )
    for index, tensor in enumerate(recorder.condition_outputs):
        save_tensor(
            out_dir, entries, f"condition_chunk{index}", tensor[0],
            f"condition mix output for window {index} [latent_length, out_dim]",
        )
    steps = recorder.step_records
    for index, record in enumerate(steps):
        if index not in (0, len(steps) - 1):
            continue
        tag = "first" if index == 0 else "last"
        save_tensor(
            out_dir, entries, f"denoise_{tag}_sample_in", record["sample_in"][0],
            f"DiT denoise step {index} ({tag}) input latents, sigma={record['timestep']}",
        )
        save_tensor(
            out_dir, entries, f"denoise_{tag}_velocity", record["velocity"][0],
            f"guided velocity at step {index} ({tag})",
        )
        save_tensor(
            out_dir, entries, f"denoise_{tag}_latents_out", record["latents_out"][0],
            f"latents after scheduler.step {index} ({tag})",
        )
    for index, tensor in enumerate(recorder.vocoder_inputs):
        save_tensor(
            out_dir, entries, f"vocoder_input_chunk{index}", tensor[0],
            f"vocoder input latents for window {index} [latent_channels, latent_length]",
        )
    save_tensor(
        out_dir, entries, "waveform", audios[0],
        "final stereo waveform [channels, samples] at 44100 Hz, resample-free (spec §1.1)",
    )

    wav_path = out_dir / "waveform.wav"
    try:
        import soundfile as sf

        sf.write(str(wav_path), audios[0].float().cpu().numpy().T, sample_rate)
        wav_info = {"file": wav_path.name, "sha256": _sha256(wav_path), "bytes": wav_path.stat().st_size}
    except Exception as exc:  # noqa: BLE001 - the .npy waveform is the golden; the WAV is for ears
        wav_info = {"error": f"{type(exc).__name__}: {exc}"}

    manifest = {
        "model": "MiniMaxAI/MiniMax-Music3 (diffusers arm)",
        "issue": "#672",
        "spec": ".agents/specs/minimax-music3.md",
        "generated_by": "tools/oracle/music3_oracle.py",
        "captured_on": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "oracle": {
            "upstream": "https://github.com/huggingface/diffusers",
            "pull_request": "https://github.com/huggingface/diffusers/pull/14456",
            "pin": env["diffusers_commit"],
            "pin_source": env["diffusers_commit_source"],
        },
        "environment": env,
        "checkpoint": checkpoint_info,
        "request": {
            "prompt": args.prompt,
            "lyrics": args.lyrics,
            "audio_duration_s": args.audio_duration,
            "num_inference_steps": args.steps,
            "seed": args.seed,
            "generator_device": "cpu",
            "dtype_policy": args.dtype_policy,
            "dtypes": (
                {k: str(v) for k, v in dtype_arg.items()} if isinstance(dtype_arg, dict) else str(dtype_arg)
            ),
        },
        "result": {
            "sampling_rate": sample_rate,
            "waveform_shape": list(audios.shape),
            "duration_s": duration,
            "ar_frames": int(frame_hiddens.shape[1]),
            "chunk_starts": [int(x) for x in chunk_starts],
            "num_latent_chunks": len(latent_chunks),
            "denoise_steps_recorded": len(steps),
            "denoise_sigmas": [record["timestep"] for record in steps],
            "rvq_code_rows": int(torch.cat(recorder.rvq_codes, dim=0).shape[0]),
            "load_seconds": round(load_seconds, 2),
            "run_seconds": round(run_seconds, 2),
        },
        "spec_facts": facts,
        "spec_disagreements": disagreements,
        "wav": wav_info,
        "tensors": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str) + "\n")
    print(f"[oracle] wrote {len(entries)} tensors + manifest.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

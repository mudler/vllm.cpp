# vllm.cpp parity harness; oracle = upstream vLLM MTP execution.
"""Capture a standalone Qwen3.5/3.6 MTP-head parity case.

This drives the real vLLM speculative-decoding path with k=1, intercepts the
first non-dummy Qwen3_5MTP forward, and stores its exact draft inputs and
outputs.  The corresponding C++ gate is the focused ``qwen3.5 MTP standalone
head parity`` case in ``test_op_parity``.

Run on dgx.casa after acquiring the project GPU lock::

    flock /tmp/gpu bash -lc '
      cd ~/work/vllm.cpp
      VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        ~/venvs/vllm-oracle/bin/python \
        tools/parity/dump_qwen3_5_mtp.py \
        --model <snapshot> --tag 27b \
        --pinned-vllm /home/mudler/work/vllm-pin \
        --out tests/parity/goldens'

The installed wheel is used for the executable engine.  The script refuses to
run unless the executable MTP class AST (init/forward/compute_logits) matches
the pinned tree; only known loader-only members are stripped before comparing.
With the v0.25.0 oracle the two trees are byte-identical for
``qwen3_5_mtp.py``, so the stripping is a no-op and the check is exact.  The
existing whole-model oracle audit covers the shared Qwen3.5 decoder layer
implementation (dump_qwen36.py).

Both gate checkpoints declare a ``*ForConditionalGeneration`` architecture, so
the speculator takes vLLM's multimodal branch and calls the draft model with
``input_ids=None`` plus a prebuilt ``inputs_embeds``
(``v1/spec_decode/llm_base_proposer.py:962-991`` @ v0.25.0).  The exact token
ids are still materialized one line earlier, when the proposer calls
``Qwen3_5MTP.embed_input_ids(self.input_ids[:num_tokens])`` (:971), so the
harness also patches ``embed_input_ids`` and recovers the ids from there.  The
capture requires the recovered ids to cover the whole forward (no cudagraph
padding rows), which ``enforce_eager=True`` guarantees.
"""

from __future__ import annotations

import argparse
import ast
import copy
import inspect
import os
import pathlib
from typing import Any

import torch

from dump_common import save_case


CAPTURE: dict[str, torch.Tensor] = {}
CAPTURE_ARMED = False
# Token ids recovered from the proposer's embed_input_ids call; see the module
# docstring (the multimodal branch passes input_ids=None to forward()).
EMBEDDED_IDS: dict[str, torch.Tensor] = {}
# One-line diagnostics for every armed forward that did NOT capture, so a
# future upstream signature change fails loudly instead of silently.
REJECTED: list[str] = []
IDS_SOURCE: list[str] = []
CLASS_NAMES = (
    "Qwen3_5MultiTokenPredictor",
    "Qwen3_5MTP",
    "Qwen3_5MoeMTP",
)

# pip-vLLM 0.24 carries the pre-AutoWeightsLoader implementation for the fused
# Qwen3.5 MoE tensors. It changes loading mechanics, not model execution. Strip
# exactly this known delta before comparing the executable class structure.
IGNORED_METHODS = {
    "Qwen3_5MultiTokenPredictor": {
        "load_fused_expert_weights",
        "load_weights",
    },
}
IGNORED_ASSIGNMENTS = {
    "Qwen3_5MultiTokenPredictor": {"hf_to_vllm_mapper"},
}


def _ignored_assignment(node: ast.AST, names: set[str]) -> bool:
    if isinstance(node, ast.Assign):
        return any(isinstance(target, ast.Name) and target.id in names
                   for target in node.targets)
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and node.target.id in names
    return False


def _normalized_class(node: ast.ClassDef) -> str:
    normalized = copy.deepcopy(node)
    ignored_methods = IGNORED_METHODS.get(node.name, set())
    ignored_assignments = IGNORED_ASSIGNMENTS.get(node.name, set())
    normalized.body = [
        member for member in normalized.body
        if not (
            isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            and member.name in ignored_methods
        )
        and not _ignored_assignment(member, ignored_assignments)
    ]
    return ast.dump(normalized, include_attributes=False)


def _class_asts(path: pathlib.Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found = {
        node.name: _normalized_class(node)
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in CLASS_NAMES
    }
    missing = sorted(set(CLASS_NAMES) - set(found))
    if missing:
        raise RuntimeError(f"{path}: missing expected classes {missing}")
    return found


def verify_mtp_source(installed: pathlib.Path, pinned_root: pathlib.Path) -> None:
    pinned = pinned_root / "vllm/model_executor/models/qwen3_5_mtp.py"
    if not pinned.is_file():
        raise RuntimeError(f"pinned MTP source not found: {pinned}")
    installed_asts = _class_asts(installed)
    pinned_asts = _class_asts(pinned)
    drift = [name for name in CLASS_NAMES if installed_asts[name] != pinned_asts[name]]
    if drift:
        raise RuntimeError(
            "installed vLLM executable MTP semantics drift from pin for "
            f"{drift}; do not generate a golden from this environment"
        )
    print(
        "verified executable MTP class ASTs against pin "
        f"(known loader-only drift stripped): {pinned}"
    )


def _argument(args: tuple[Any, ...], kwargs: dict[str, Any], name: str, idx: int):
    if name in kwargs:
        return kwargs[name]
    return args[idx] if len(args) > idx else None


def install_capture_patch(mtp_cls) -> None:
    original_forward = mtp_cls.forward
    original_embed = mtp_cls.embed_input_ids

    # llm_base_proposer.py:971 feeds the real ids here on the multimodal branch
    # and then hands forward() only the resulting inputs_embeds.
    def capture_embed_input_ids(self, input_ids, *args, **kwargs):
        if isinstance(input_ids, torch.Tensor):
            EMBEDDED_IDS["input_ids"] = input_ids.detach().clone()
        return original_embed(self, input_ids, *args, **kwargs)

    def capture_forward(self, *args, **kwargs):
        output = original_forward(self, *args, **kwargs)
        if not CAPTURE_ARMED or CAPTURE:
            return output

        input_ids = _argument(args, kwargs, "input_ids", 0)
        positions = _argument(args, kwargs, "positions", 1)
        target_hidden = _argument(args, kwargs, "hidden_states", 2)
        inputs_embeds = _argument(args, kwargs, "inputs_embeds", 4)
        ids_source = "forward(input_ids=...)"
        if input_ids is None and isinstance(inputs_embeds, torch.Tensor):
            input_ids = EMBEDDED_IDS.get("input_ids")
            ids_source = "embed_input_ids(self.input_ids[:num_tokens])"
        if not all(isinstance(t, torch.Tensor)
                   for t in (input_ids, positions, target_hidden, output)):
            REJECTED.append(
                "non-tensor argument: "
                f"input_ids={type(input_ids).__name__} "
                f"positions={type(positions).__name__} "
                f"hidden_states={type(target_hidden).__name__} "
                f"output={type(output).__name__}"
            )
            return output
        if output.ndim != 2 or output.shape[0] < 16:
            REJECTED.append(f"output shape {tuple(output.shape)} (need [T>=16,H])")
            return output
        if input_ids.numel() != output.shape[0]:
            raise RuntimeError(
                f"recovered input ids ({input_ids.numel()}) do not cover the "
                f"MTP forward ({output.shape[0]} rows) via {ids_source}; "
                "padding rows would poison the golden"
            )
        print("capturing MTP forward; token ids recovered from", ids_source)
        IDS_SOURCE.append(ids_source)

        logits = self.compute_logits(output)
        if logits is None:
            raise RuntimeError("Qwen3_5MTP.compute_logits returned None")
        topk_values, topk_indices = torch.topk(logits.float(), 8, dim=-1)
        CAPTURE.update(
            input_ids=input_ids.detach().to(torch.int32).cpu().clone(),
            positions=positions.detach().to(torch.int32).cpu().clone(),
            target_hidden=target_hidden.detach().cpu().clone(),
            mtp_hidden=output.detach().cpu().clone(),
            expected_argmax=logits.detach().argmax(-1).to(torch.int32).cpu().clone(),
            topk_values=topk_values.detach().cpu().clone(),
            topk_indices=topk_indices.detach().to(torch.int32).cpu().clone(),
        )
        return output

    mtp_cls.forward = capture_forward
    mtp_cls.embed_input_ids = capture_embed_input_ids


def normalize_positions(positions: torch.Tensor) -> tuple[torch.Tensor, str]:
    if positions.ndim == 1:
        return positions.contiguous(), "[T]"
    if positions.ndim == 2:
        if positions.shape[0] not in (1, 3):
            raise RuntimeError(f"unsupported MTP positions shape {tuple(positions.shape)}")
        first = positions[0]
        for row in positions[1:]:
            if not torch.equal(row, first):
                raise RuntimeError(
                    "multimodal MRoPE position rows differ; the M-mtp-0 text gate "
                    "only supports the equal-row text case"
                )
        return first.contiguous(), f"{list(positions.shape)} equal rows -> row 0"
    raise RuntimeError(f"unsupported MTP positions rank {positions.ndim}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True, choices=("27b", "35b"))
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--pinned-vllm",
        default="/home/mudler/_git/vllm",
        help="checkout at the parity pin e24d1b24",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Modern computing evolved from mechanical calculators through "
            "vacuum tubes, transistors, integrated circuits, and massively "
            "parallel processors used for scientific simulations."
        ),
    )
    parser.add_argument("--gpu-mem", type=float, default=0.8)
    args = parser.parse_args()

    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise RuntimeError(
            "set VLLM_ENABLE_V1_MULTIPROCESSING=0 so the capture patch and "
            "engine execute in the same process"
        )

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    installed_source = pathlib.Path(inspect.getsourcefile(Qwen3_5MTP) or "")
    verify_mtp_source(installed_source, pathlib.Path(args.pinned_vllm))
    install_capture_patch(Qwen3_5MTP)

    print("vllm from:", vllm.__file__, "version", vllm.__version__)
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=256,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_mem,
        dtype="bfloat16",
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
    )

    global CAPTURE_ARMED
    CAPTURE_ARMED = True
    result = llm.generate(
        args.prompt,
        SamplingParams(temperature=0.0, max_tokens=2),
    )
    CAPTURE_ARMED = False
    if not CAPTURE:
        raise RuntimeError(
            "no >=16-row real Qwen3_5MTP forward was captured; rejected "
            f"forwards: {REJECTED or 'none (forward never ran)'}"
        )

    positions, positions_layout = normalize_positions(CAPTURE["positions"])
    input_ids = CAPTURE["input_ids"].reshape(-1)
    target_hidden = CAPTURE["target_hidden"]
    mtp_hidden = CAPTURE["mtp_hidden"]
    expected_argmax = CAPTURE["expected_argmax"].reshape(-1)
    tokens = input_ids.numel()
    if tokens < 16 or expected_argmax.numel() != tokens:
        raise RuntimeError(
            f"MTP gate needs >=16 positions; captured T={tokens}, "
            f"argmax={expected_argmax.numel()}"
        )
    if target_hidden.dtype != torch.bfloat16 or mtp_hidden.dtype != torch.bfloat16:
        raise RuntimeError(
            "expected BF16 target/MTP hidden states, got "
            f"{target_hidden.dtype}/{mtp_hidden.dtype}"
        )

    hf = llm.llm_engine.vllm_config.model_config.hf_text_config
    generated = list(result[0].outputs[0].token_ids)
    print("captured rows:", tokens, "generated:", generated)
    save_case(
        pathlib.Path(args.out),
        f"qwen3_5_mtp_head_{args.tag}",
        "qwen3_5_mtp_head",
        {
            "input_ids": input_ids,
            "positions": positions,
            "target_hidden": target_hidden,
            "mtp_hidden": mtp_hidden,
            "expected_argmax": expected_argmax,
            "topk_values": CAPTURE["topk_values"],
            "topk_indices": CAPTURE["topk_indices"],
        },
        {
            "model": args.model,
            "tag": args.tag,
            "prompt": args.prompt,
            "tokens": tokens,
            "hidden_size": int(hf.hidden_size),
            "vocab_size": int(hf.vocab_size),
            "mtp_num_hidden_layers": int(getattr(hf, "mtp_num_hidden_layers", 1)),
            "positions_layout": positions_layout,
            "input_ids_source": IDS_SOURCE[0],
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 1},
            "installed_source": str(installed_source),
            "pinned_forward_ast_verified": True,
            "installed_loader_drift_allowed": sorted(
                IGNORED_METHODS["Qwen3_5MultiTokenPredictor"]
            ),
            "generated_ids": generated,
        },
        ["input_ids", "positions", "target_hidden"],
        ["mtp_hidden", "expected_argmax"],
        # Consumed by test_op_parity's focused MTP head case. `logits_*` are
        # the same bounds the whole-model logits gates use
        # (qwen36_logits_{27,35}b). The hidden states are bf16 on BOTH sides,
        # so one ULP at |h| ~ 8 is already 0.0625 and an element-wise absolute
        # bound below that is unsatisfiable even for a perfect port; what a
        # STRUCTURAL error does move is the vector's direction and scale, so
        # that is what the hidden state is gated on. `hidden_report_*` only
        # parameterize the reported out-of-tolerance element count.
        {
            "argmax": ("exact on rows with a nonzero oracle top1-top2 margin; "
                       "exact-tie rows must select a tied maximum"),
            "logits_atol": 0.05,
            "logits_rtol": 0.05,
            "hidden_min_row_cosine": 0.995,
            "hidden_scale_tol": 0.01,
            "hidden_rel_rms_max": 0.02,
            "hidden_report_atol": 0.05,
            "hidden_report_rtol": 0.05,
        },
    )


if __name__ == "__main__":
    main()

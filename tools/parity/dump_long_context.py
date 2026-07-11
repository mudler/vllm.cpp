# vllm.cpp parity harness; oracle = pinned upstream vLLM.
"""Dump long-context RoPE caches and outputs from pinned vLLM classes.

Run inside the oracle environment with this repository's tools/parity on
sys.path, for example:

  ~/venvs/vllm-oracle/bin/python tools/parity/dump_long_context.py \
      --out tests/parity/goldens --only yarn

The W5 matrix deliberately uses small original context lengths while sampling
the same boundary set {0, 1, original-1, original, scaled-max-1}. Full cache
tensors are therefore committed without multi-megabyte fixtures.
"""

import argparse
import importlib.util
import os
import pathlib
import subprocess
import sys
import types

import torch

from dump_common import save_case

F32_TOL = {"atol": 1e-5, "rtol": 1e-5}
BF16_TOL = {"atol": 1e-2, "rtol": 1.6e-2}

UPSTREAM_PIN = "e24d1b24fe96a56ba8b0d653efa076d03eb95d6c"
_ROPE_FACTORY = None
_ORACLE_LOADER = "installed-vllm"


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _package(name: str, path: pathlib.Path | None = None):
    module = types.ModuleType(name)
    if path is not None:
        module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _direct_source_factory():
    """Load exact pinned classes without importing vLLM's unrelated runtime.

    A machine crash can leave torch/numpy intact while the full oracle venv is
    absent. The RoPE files only need CustomOp/platform registration scaffolding
    to execute their PyTorch-native cache and forward methods. This fallback
    stubs that scaffolding, but imports and executes the pinned common.py,
    base.py, yarn_scaling_rope.py and mrope.py files verbatim. It is therefore a
    direct-source oracle, not a reimplementation of their formulas.
    """
    upstream = pathlib.Path(
        os.environ.get("VLLM_UPSTREAM_DIR", "/home/mudler/_git/vllm")
    ).resolve()
    actual_pin = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_pin != UPSTREAM_PIN:
        raise RuntimeError(
            f"pinned source mismatch: expected {UPSTREAM_PIN}, got {actual_pin}"
        )

    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            del sys.modules[name]

    root = _package("vllm", upstream / "vllm")
    root.__version__ = "0.24.0+direct-source-e24d1b24"
    _package("vllm.model_executor", upstream / "vllm/model_executor")
    _package("vllm.model_executor.layers", upstream / "vllm/model_executor/layers")
    rope_dir = upstream / "vllm/model_executor/layers/rotary_embedding"
    _package("vllm.model_executor.layers.rotary_embedding", rope_dir)

    class CustomOp(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            del args, kwargs
            super().__init__()
            self._forward_method = self.forward_native

        @classmethod
        def register(cls, name):
            del name
            return lambda implementation: implementation

        def enabled(self):
            return False

        def forward(self, *args, **kwargs):
            return self.forward_native(*args, **kwargs)

    custom_op = types.ModuleType("vllm.model_executor.custom_op")
    custom_op.CustomOp = CustomOp
    sys.modules[custom_op.__name__] = custom_op

    logger = types.ModuleType("vllm.logger")
    logger.init_logger = lambda name: __import__("logging").getLogger(name)
    sys.modules[logger.__name__] = logger

    class Platform:
        @staticmethod
        def is_cpu():
            return True

    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = Platform()
    sys.modules[platforms.__name__] = platforms

    _package("vllm.utils", upstream / "vllm/utils")
    torch_utils = types.ModuleType("vllm.utils.torch_utils")
    torch_utils.direct_register_custom_op = lambda **kwargs: None
    sys.modules[torch_utils.__name__] = torch_utils

    _package("vllm._aiter_ops", upstream / "vllm/_aiter_ops")

    class RocmAiterOps:
        @staticmethod
        def is_triton_rotary_embed_enabled():
            return False

    aiter = types.ModuleType("vllm._aiter_ops.rocm_aiter_ops")
    aiter.rocm_aiter_ops = RocmAiterOps()
    sys.modules[aiter.__name__] = aiter
    sys.modules["vllm._aiter_ops"].rocm_aiter_ops = aiter.rocm_aiter_ops

    class TritonStub:
        @staticmethod
        def jit(function):
            return function

        @staticmethod
        def next_power_of_2(value):
            return 1 << (value - 1).bit_length()

    triton_utils = types.ModuleType("vllm.triton_utils")
    triton_utils.triton = TritonStub()
    triton_utils.tl = types.SimpleNamespace(constexpr=object())
    sys.modules[triton_utils.__name__] = triton_utils

    _load_module(
        "vllm.model_executor.layers.rotary_embedding.common",
        rope_dir / "common.py",
    )
    _load_module(
        "vllm.model_executor.layers.rotary_embedding.base",
        rope_dir / "base.py",
    )
    yarn_module = _load_module(
        "vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope",
        rope_dir / "yarn_scaling_rope.py",
    )
    mrope_module = _load_module(
        "vllm.model_executor.layers.rotary_embedding.mrope",
        rope_dir / "mrope.py",
    )

    def factory(
        head_size,
        max_position,
        is_neox_style=True,
        rope_parameters=None,
        dtype=None,
    ):
        del max_position
        if dtype is None:
            dtype = torch.get_default_dtype()
        params = rope_parameters or {}
        base = params.get("rope_theta", 10000)
        rotary_dim = params.get("rope_dim")
        if not rotary_dim:
            partial = params.get("partial_rotary_factor", 1.0)
            if partial <= 0.0 or partial > 1.0:
                raise ValueError(
                    f"partial_rotary_factor={partial} must be between 0.0 and 1.0"
                )
            rotary_dim = int(head_size * partial)
        if params.get("rope_type", "default") != "yarn":
            raise ValueError("direct-source W5 loader only accepts yarn")
        factor = params["factor"]
        original = params["original_max_position_embeddings"]
        extra = {
            key: value
            for key, value in params.items()
            if key
            in {
                "extrapolation_factor",
                "attn_factor",
                "beta_fast",
                "beta_slow",
                "apply_yarn_scaling",
                "truncate",
            }
        }
        if "mrope_section" in params:
            extra.pop("apply_yarn_scaling", None)
            return mrope_module.MRotaryEmbedding(
                head_size,
                rotary_dim,
                original,
                base,
                is_neox_style,
                dtype,
                mrope_section=params["mrope_section"],
                mrope_interleaved=params.get("mrope_interleaved", False),
                scaling_factor=factor,
                **extra,
            )
        return yarn_module.YaRNScalingRotaryEmbedding(
            head_size,
            rotary_dim,
            original,
            base,
            is_neox_style,
            factor,
            dtype,
            **extra,
        )

    return factory


def _get_rope_factory():
    global _ROPE_FACTORY, _ORACLE_LOADER
    if _ROPE_FACTORY is not None:
        return _ROPE_FACTORY
    try:
        from vllm.model_executor.layers.rotary_embedding import get_rope

        _ROPE_FACTORY = get_rope
    except ModuleNotFoundError as error:
        print(
            f"full vLLM import unavailable ({error}); using verified pinned "
            "direct-source class loader"
        )
        _ROPE_FACTORY = _direct_source_factory()
        _ORACLE_LOADER = "verified-pinned-direct-source"
    return _ROPE_FACTORY


def _dump_case(
    root: pathlib.Path,
    name: str,
    *,
    head_size: int,
    max_position: int,
    is_neox_style: bool,
    rope_parameters: dict,
    dtype: torch.dtype,
    positions: torch.Tensor,
    seed: int,
    num_q_heads: int = 2,
    num_kv_heads: int = 1,
) -> None:
    torch.manual_seed(seed)
    get_rope = _get_rope_factory()
    rope = get_rope(
        head_size=head_size,
        max_position=max_position,
        is_neox_style=is_neox_style,
        rope_parameters=rope_parameters,
        dtype=dtype,
    )
    tokens = positions.shape[-1]
    query = torch.randn(tokens, num_q_heads * head_size, dtype=dtype)
    key = torch.randn(tokens, num_kv_heads * head_size, dtype=dtype)
    query_out, key_out = rope.forward_native(
        positions, query.clone(), key.clone()
    )
    args = {
        "head_size": head_size,
        "max_position": max_position,
        "is_neox_style": is_neox_style,
        "rope_parameters": rope_parameters,
        "dtype": "bf16" if dtype == torch.bfloat16 else "f32",
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "class_name": type(rope).__name__,
        "seed": seed,
        "oracle_loader": _ORACLE_LOADER,
    }
    save_case(
        root,
        name,
        "long_context_rope",
        {
            "cos_sin_cache": rope.cos_sin_cache,
            "positions": positions,
            "q_in": query,
            "k_in": key,
            "q_out": query_out,
            "k_out": key_out,
        },
        args,
        ["positions", "q_in", "k_in"],
        ["cos_sin_cache", "q_out", "k_out"],
        BF16_TOL if dtype == torch.bfloat16 else F32_TOL,
    )


def dump_yarn(root: pathlib.Path) -> None:
    original = 32
    factor = 4.0
    max_position = int(original * factor)
    boundary_positions = torch.tensor(
        [0, 1, original - 1, original, max_position - 1], dtype=torch.int64
    )

    base = {
        "rope_type": "yarn",
        "rope_theta": 10000.0,
        "rope_dim": 8,
        "factor": factor,
        "original_max_position_embeddings": original,
    }
    _dump_case(
        root,
        "long_rope_yarn_neox_f32_truncate",
        head_size=16,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters=dict(base),
        dtype=torch.float32,
        positions=boundary_positions,
        seed=101,
    )

    gptj = {
        **base,
        "extrapolation_factor": 0.75,
        "attn_factor": 0.8,
        "beta_fast": 16,
        "beta_slow": 2,
        "apply_yarn_scaling": False,
        "truncate": False,
    }
    _dump_case(
        root,
        "long_rope_yarn_gptj_f32_untruncated_no_mscale",
        head_size=16,
        max_position=max_position,
        is_neox_style=False,
        rope_parameters=gptj,
        dtype=torch.float32,
        positions=boundary_positions,
        seed=102,
    )

    _dump_case(
        root,
        "long_rope_yarn_neox_bf16",
        head_size=16,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters={**base, "attn_factor": 1.1},
        dtype=torch.bfloat16,
        positions=boundary_positions,
        seed=103,
    )

    # Sections sum to rope_dim / 2. Values exercise all scaled boundaries while
    # remaining within MRotaryEmbedding's 4x-enlarged YaRN cache.
    mrope_positions = torch.tensor(
        [
            [0, 1, original - 1, original, max_position - 1],
            [1, 2, original, original + 1, max_position - 2],
            [2, 3, original + 1, original + 2, max_position - 3],
        ],
        dtype=torch.int64,
    )
    mrope = {**base, "mrope_section": [1, 1, 2]}
    _dump_case(
        root,
        "long_rope_yarn_mrope_f32_contiguous",
        head_size=16,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters=mrope,
        dtype=torch.float32,
        positions=mrope_positions,
        seed=104,
    )
    _dump_case(
        root,
        "long_rope_yarn_mrope_bf16_interleaved",
        head_size=16,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters={**mrope, "mrope_interleaved": True},
        dtype=torch.bfloat16,
        positions=mrope_positions,
        seed=105,
    )
    _dump_case(
        root,
        "long_rope_yarn_mrope_f32_text_1d",
        head_size=16,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters={**mrope, "mrope_interleaved": True},
        dtype=torch.float32,
        positions=boundary_positions,
        seed=106,
    )


DUMPERS = {"yarn": dump_yarn}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    output = pathlib.Path(args.out)
    for family, dumper in DUMPERS.items():
        if args.only is None or family in args.only:
            dumper(output)

#!/usr/bin/env python3
"""Check both Qwen3 oracle scripts' production LLM construction paths."""

from __future__ import annotations

import builtins
import io
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


class _FakeArray:
    def reshape(self, *shape):
        return self


def fake_numpy() -> ModuleType:
    module = ModuleType("numpy")
    module.fromfile = lambda *args, **kwargs: _FakeArray()
    module.load = lambda *args, **kwargs: _FakeArray()
    return module


def load_script(name: str, path: Path) -> ModuleType:
    original_import = builtins.__import__

    def reject_vllm(module_name, *args, **kwargs):
        if module_name == "vllm" or module_name.startswith("vllm."):
            raise AssertionError(f"{path.name} imported vLLM during module loading")
        return original_import(module_name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"numpy": fake_numpy()}), mock.patch(
        "builtins.__import__", side_effect=reject_vllm
    ):
        spec.loader.exec_module(module)
    return module


class _LlmConstructed(Exception):
    """Stop a script after observing its production LLM constructor call."""


class OracleModeTests(unittest.TestCase):
    def run_to_llm(self, module: ModuleType, argv: list[str]):
        constructor_calls = []
        output = io.StringIO()
        fake_vllm = ModuleType("vllm")

        def observing_llm(**kwargs):
            constructor_calls.append(kwargs)
            raise _LlmConstructed

        fake_vllm.LLM = observing_llm
        fake_vllm.SamplingParams = object
        with mock.patch.dict(sys.modules, {"vllm": fake_vllm}), mock.patch.object(
            sys, "argv", [str(module.__file__), *argv]
        ), mock.patch.object(module.os, "makedirs"), mock.patch(
            "sys.stdout", output
        ):
            with self.assertRaises(_LlmConstructed):
                module.main()

        self.assertEqual(len(constructor_calls), 1)
        return constructor_calls[0], output.getvalue()

    def assert_modes(self, filename: str, required_args: list[str], gpu_arg: str) -> None:
        path = ROOT / "scripts" / filename
        module = load_script(filename.replace("-", "_"), path)

        for eager in (False, True):
            argv = [*required_args, gpu_arg, "0.42"]
            if eager:
                argv.append("--enforce-eager")
            kwargs, narration = self.run_to_llm(module, argv)
            self.assertEqual(
                kwargs,
                {
                    "model": "model",
                    "dtype": "bfloat16",
                    "enforce_eager": eager,
                    "gpu_memory_utilization": 0.42,
                },
            )
            self.assertIs(kwargs["enforce_eager"], eager)
            mode = "eager diagnostic" if eager else "production"
            self.assertIn(mode, narration.lower())
            self.assertIn(f"enforce_eager={eager}", narration)

    def test_oracle_capture_modes(self) -> None:
        self.assert_modes(
            "qwen3-oracle-capture.py",
            ["--model", "model", "--out-dir", "/unused"],
            "--gpu-mem",
        )

    def test_neartie_gap_modes(self) -> None:
        self.assert_modes(
            "qwen3-neartie-gap.py",
            ["--model", "model", "--golden-dir", "/unused"],
            "--gpu-mem-util",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

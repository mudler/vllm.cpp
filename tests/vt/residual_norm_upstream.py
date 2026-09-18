"""Export the pinned BF16 normalization tests without changing their fixtures.

Port of vLLM e126687a9a828d513c01a07cd69f025f27d63280:
tests/kernels/core/test_layernorm.py::test_rms_norm and
tests/kernels/ir/test_layernorm.py::{TestRMSNorm,TestFusedAddRMSNorm}.

The original core test and IR provider/semantics/opcheck tests execute. Their
native references and original torch-generated inputs are exported for C++.
BF16 is the measured compiled-expression policy. F16/F32 activation modes are
explicitly outside that policy and are refused by the shared descriptor tests.
The C++ ABI represents weight=None with an explicit unit BF16 gamma, and plain
RMSNorm with a zero base. Torch's registration/opcheck is run here; C++ checks
its own typed descriptor, dispatch, output ownership, strides, and aliases.
"""
import argparse
import hashlib
import importlib
import itertools
import json
import sys
from pathlib import Path

PIN = "e126687a9a828d513c01a07cd69f025f27d63280"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import vllm
    assert vllm.__version__ == "0.28.1rc1.dev132+ge126687a9"
    sys.path.insert(0, str(args.source))
    core = importlib.import_module("tests.kernels.core.test_layernorm")
    ir_test = importlib.import_module("tests.kernels.ir.test_layernorm")
    from vllm import ir
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.utils.torch_utils import set_random_seed
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    active = {}

    def save(name, value):
        raw = value.detach().cpu().reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
        file = f"{active['id']}-{name}.bin"
        (args.output / file).write_bytes(raw)
        active["files"][name] = {"file": file, "shape": list(value.shape),
            "stride": list(value.stride()), "dtype": str(value.dtype),
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def capture(x, residual, weight, epsilon, output):
        if active.get("captured"):
            return
        active["captured"] = True
        active["rows"], active["width"] = x.shape
        active["epsilon"] = epsilon
        active["device"] = x.device.index
        active["add_residual"] = residual is not None
        save("a", x)
        save("base", torch.zeros_like(x) if residual is None else residual)
        save("gamma", weight)
        norm, res = output if residual is not None else (output, None)
        save("norm", norm)
        if res is not None:
            save("residual", res)
        operation = ir.ops.rms_norm if residual is None else ir.ops.fused_add_rms_norm
        native = operation.impls["native"].impl_fn
        inputs = (x,) if residual is None else (x, residual)
        unit = native(*inputs, None, epsilon)
        unit_with_weight = native(*inputs, torch.ones_like(weight), epsilon)
        torch.testing.assert_close(unit, unit_with_weight)
        save("unit_norm", unit if residual is None else unit[0])

    original_core = core.RMSNorm.forward_native

    def core_native(layer, x, residual=None):
        output = original_core(layer, x, residual)
        capture(x, residual, layer.weight.data, layer.variance_epsilon, output)
        return output

    def begin(name, tolerance, **parameters):
        active.clear()
        active.update({"id": name, "pin": PIN, "seed": 0, "tolerance": tolerance,
                       "parameters": parameters, "files": {}})
        set_random_seed(0)

    def finish():
        assert active.pop("captured")
        cases.append(dict(active))
        (args.output / "cases.json").write_text(json.dumps({"pin": PIN, "complete": False,
                                                           "cases": cases}, indent=2) + "\n")
        print(active["id"], "PASS", flush=True)

    with torch.inference_mode(), set_current_vllm_config(VllmConfig()):
        core.RMSNorm.forward_native = core_native
        try:
            for index, values in enumerate(itertools.product(core.NUM_TOKENS, core.HIDDEN_SIZES,
                    core.ADD_RESIDUAL, core.CUDA_DEVICES, (False, True))):
                rows, width, add, device, strided = values
                begin(f"core-{index}", {"atol": 1e-2, "rtol": 1e-2}, rows=rows,
                      width=width, add_residual=add, device=device, strided=strided)
                core.test_rms_norm(None, rows, width, add, torch.bfloat16, 0, device, strided)
                finish()
        finally:
            core.RMSNorm.forward_native = original_core
        torch.set_default_device("cuda:0")
        for fused, cls, operation, global_name in (
            (False, ir_test.TestRMSNorm, ir.ops.rms_norm, "rms_norm_native"),
            (True, ir_test.TestFusedAddRMSNorm, ir.ops.fused_add_rms_norm, "fused_add_rms_norm_native")):
            original = getattr(ir_test, global_name)

            def observed(*positional, **keywords):
                result = original(*positional, **keywords)
                x = positional[0]
                residual = positional[1] if fused else None
                gamma = positional[2 if fused else 1]
                epsilon = positional[3 if fused else 2]
                capture(x, residual, gamma, epsilon, result)
                return result

            setattr(ir_test, global_name, observed)
            try:
                for index, (rows, width, epsilon) in enumerate(itertools.product(
                        ir_test.NUM_TOKENS, ir_test.COMMON_HIDDEN_SIZES, (1e-6, 1e-5))):
                    begin(f"ir-{'add' if fused else 'rms'}-{index}", operation.get_tolerance(torch.bfloat16),
                          rows=rows, width=width, epsilon=epsilon, provider="vllm_c")
                    cls().test_impls(torch.bfloat16, rows, width, epsilon, "vllm_c")
                    finish()
                # These original semantics tests use 4x8 regardless of the class
                # grid. Run each distinct epsilon with the original seed zero.
                setattr(ir_test, global_name, original)
                for epsilon in (1e-6, 1e-5):
                    cls().test_native_semantics(torch.bfloat16, 1, 2048, epsilon)
                    cls().test_torch_opcheck(torch.bfloat16, 1, 2048, epsilon, "vllm_c")
            finally:
                setattr(ir_test, global_name, original)
    record = {"pin": PIN, "complete": True, "cases": cases,
              "applicability": {"BF16": "executed", "F16": "refused compiled-expression activation dtype",
                                "F32": "refused compiled-expression activation dtype"},
              "adaptations": ["C++ unit gamma represents absent weight", "zero base represents plain RMSNorm",
                              "C++ native provider replaces the upstream provider parameter; original vllm_c test still executes",
                              "Torch registration/opcheck runs in this exporter; C++ validates its own typed ABI"]}
    (args.output / "cases.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

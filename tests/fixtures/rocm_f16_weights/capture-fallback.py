# Capture the pinned vLLM ordinary fallback fixture for the C++ test.
# Upstream: e126687a9a828d513c01a07cd69f025f27d63280,
# tests/model_executor/layers/test_rocm_unquantized_gemm.py:152.
# Run in that pinned ROCm oracle environment under the operator's GPU lease.
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers import utils

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.manual_seed(0)
# Preserve the pinned upstream random-normal fixture and its dimensions. The
# seed and copied bit patterns make the otherwise unseeded test reproducible.
x_cpu = torch.randn(6, 64, dtype=torch.float16)
w_cpu = torch.randn(128, 64, dtype=torch.float16)
x = x_cpu.to('cuda')
w = w_cpu.to('cuda')
wvsplitk = MagicMock(side_effect=lambda weight, inp, _, __: inp @ weight.t())
llmm1 = MagicMock(side_effect=lambda weight, inp, _: inp @ weight.t())
with patch.object(utils, 'use_aiter_triton_gemm', lambda *args: False), \
     patch.object(utils.envs, 'VLLM_ROCM_USE_SKINNY_GEMM', True), \
     patch('vllm.platforms.rocm.on_gfx1x', lambda: True), \
     patch('vllm.platforms.rocm.on_gfx9', lambda: False), \
     patch('vllm.platforms.rocm.on_gfx950', lambda: False), \
     patch('vllm.platforms.rocm.on_gfx1250', lambda: False), \
     patch.object(utils, 'num_compute_units', lambda: 120), \
     patch.object(utils.ops, 'wvSplitK', wvsplitk), \
     patch.object(utils.ops, 'LLMM1', llmm1):
    out = utils.rocm_unquantized_gemm_impl(x, w, None)
    ref = torch.nn.functional.linear(x, w, None)
    wvsplitk.assert_not_called()
    llmm1.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
torch.cuda.synchronize()
def bits(value):
    return value.detach().cpu().contiguous().view(torch.uint16).flatten().tolist()
report = dict(upstream_revision='e126687a9a828d513c01a07cd69f025f27d63280',
              upstream_test='tests/model_executor/layers/test_rocm_unquantized_gemm.py:152:test_rocm_unquantized_gemm_gfx1x_n_gt_5_falls_back',
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              harness_adaptation='Seed 0 records upstream normal-distribution F16 inputs; copy exact CPU bits to ROCm; local C++ F32 result is rounded to F16 before comparison',
              torch_version=torch.__version__, torch_hip=torch.version.hip,
              device=torch.cuda.get_device_name(), dtype='float16',
              x_shape=[6,64], weight_shape=[128,64], atol=1e-3, rtol=1e-3,
              x_bits=bits(x_cpu), weight_bits=bits(w_cpu), result_bits=bits(out),
              raw_input_output_sha256=hashlib.sha256(x_cpu.numpy().tobytes()+w_cpu.numpy().tobytes()+out.cpu().numpy().tobytes()).hexdigest(),
              skinny_calls=wvsplitk.call_count + llmm1.call_count, result='PASS')
args.output.write_text(json.dumps(report,indent=2)+'\n')
print('F16_PRIMITIVE_ORACLE_PASS', json.dumps({k:v for k,v in report.items() if not k.endswith('_bits')}), flush=True)

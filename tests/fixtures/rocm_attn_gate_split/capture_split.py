"""Capture only split preparation from vLLM e126687a9a's fused preamble test.

The fused norm, rotary, and MRoPE operations are outside this fixture.
"""
import hashlib
import json
from pathlib import Path

import torch
import vllm
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.utils.torch_utils import set_random_seed


def main():
    assert 'e126687a9' in vllm.__version__, vllm.__version__
    root = Path(__file__).parent
    data = bytearray()
    report = {
        'upstream_revision': 'e126687a9a828d513c01a07cd69f025f27d63280',
        'upstream_test': 'tests/kernels/test_fused_qk_norm_rope_gate.py:49-67',
        'model_split': 'vllm/model_executor/models/qwen3_next.py:424-433',
        'adaptation': 'Only split preparation; contiguous query/gate snapshots preserve upstream BF16 bytes.',
        'seed': 13, 'dtype': 'bfloat16', 'head_dim': 256,
        'vllm': vllm.__version__, 'torch': torch.__version__, 'hip': torch.version.hip,
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cases': [],
    }
    config = VllmConfig()
    report['compilation_custom_ops'] = config.compilation_config.custom_ops
    with torch.inference_mode(), set_current_vllm_config(config):
        device = torch.device('cuda', torch.accelerator.current_device_index())
        torch.set_default_device(device)
        for tokens in (1, 4, 37):
            for heads, kv_heads in ((24, 4), (16, 2)):
                set_random_seed(13)
                qgate = torch.randn(tokens, heads * 2 * 256, dtype=torch.bfloat16, device=device)
                # Keep the fixture's key generation and RNG consumption even
                # though this split operation does not consume the key tensor.
                torch.randn(tokens, kv_heads * 256, dtype=torch.bfloat16, device=device)
                query, gate = torch.chunk(qgate.view(tokens, heads, -1), 2, dim=-1)
                assert torch.equal(query, qgate.view(tokens, heads, 512)[..., :256])
                assert torch.equal(gate, qgate.view(tokens, heads, 512)[..., 256:])
                torch.cuda.synchronize()
                case = {'tokens': tokens, 'query_heads': heads, 'kv_heads': kv_heads, 'head_dim': 256}
                for name, tensor in (('input', qgate), ('query', query), ('gate', gate)):
                    raw = tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                    case[name] = {'offset': len(data), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                    data.extend(raw)
                report['cases'].append(case)
    report['data_file'] = 'cases.bin'
    report['data_bytes'] = len(data)
    report['data_sha256'] = hashlib.sha256(data).hexdigest()
    (root / 'cases.bin').write_bytes(data)
    (root / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()

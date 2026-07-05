#!/usr/bin/env python
# Repack oracle: dump the RAW modelopt fp4 experts (packed uint8 + fp8 scales +
# scale2) AND vLLM's marlin-repacked outputs (weight + S0E5M3 scales + global),
# so the C++ MarlinRepack* can be diffed bit-for-bit against vLLM's own recipe
# (marlin_utils_fp4.prepare_nvfp4_moe_layer_for_marlin). Run in the oracle venv.
import json, os, torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    nvfp4_marlin_process_scales, nvfp4_marlin_process_global_scale, _nvfp4_compute_scale_factor)

torch.manual_seed(1)
dev = "cuda"; dt = torch.bfloat16
E, K, N = 8, 2048, 512          # experts, hidden(=size_k), intermediate(=size_n): the 35B gate shape
OUT = "/tmp/repack_dump"; os.makedirs(OUT, exist_ok=True)

def save(name, t):
    torch.as_tensor(t).contiguous().cpu().numpy().tofile(f"{OUT}/{name}.bin")

# --- per-expert NVFP4 quant (linear scales, modelopt style) ---
W = torch.randn(E, N, K, device=dev, dtype=dt) * 0.2
packed = []; bscale = []; s2 = []
for e in range(E):
    amax = W[e].abs().max().float()
    g = (448.0 * 6.0) / amax / float(os.environ.get("GDIV", "1"))  # GDIV>1 under-utilizes e4m3 range -> comb_sf>1
    p, bs = ops.scaled_fp4_quant(W[e], g, is_sf_swizzled_layout=False)
    packed.append(p.view(torch.uint8)); bscale.append(bs.view(torch.uint8)); s2.append((1.0 / g).reshape(1).float())
raw_packed = torch.stack(packed, 0)     # [E, N, K/2] uint8
raw_scale = torch.stack(bscale, 0)      # [E, N, K/16] uint8 (fp8-e4m3 bits)
raw_scale2 = torch.cat(s2, 0)           # [E] f32

# --- vLLM golden: weight repack ---
perm = torch.empty(0, dtype=torch.int, device=dev)
qw = []
for e in range(E):
    q = raw_packed[e].view(torch.int32).T.contiguous()
    qw.append(ops.gptq_marlin_repack(b_q_weight=q, perm=perm, size_k=K, size_n=N, num_bits=4, is_a_8bit=False))
gold_weight = torch.stack(qw, 0)        # [E, K/16, N*2] int32

# --- vLLM golden: scale processing (combined_scale_factor across all experts) ---
scales_bf = raw_scale.view(torch.float8_e4m3fn).to(dt)   # [E, N, K/16] bf16
comb_sf = _nvfp4_compute_scale_factor(scales_bf, dt)
sc = []
for e in range(E):
    s = scales_bf[e].T
    ms = marlin_permute_scales(s=s, size_k=K, size_n=N, group_size=16)
    ms, _ = nvfp4_marlin_process_scales(ms, scale_factor=comb_sf, a_dtype=dt)
    sc.append(ms)
gold_scale = torch.stack(sc, 0)         # [E, K/16, N] fp8-e4m3
gold_global = (nvfp4_marlin_process_global_scale(raw_scale2, dt) / comb_sf).float()  # [E] f32

save("raw_packed", raw_packed)
save("raw_scale", raw_scale)
save("raw_scale2", raw_scale2)
save("gold_weight", gold_weight.view(torch.int32))
save("gold_scale", gold_scale.view(torch.uint8))
save("gold_global", gold_global)
json.dump(dict(E=E, K=K, N=N, comb_sf=float(comb_sf),
               weight_shape=list(gold_weight.shape), scale_shape=list(gold_scale.shape)),
          open(f"{OUT}/params.json", "w"))
print("comb_sf", float(comb_sf), "weight", tuple(gold_weight.shape), "scale", tuple(gold_scale.shape))
print("dumped to", OUT)

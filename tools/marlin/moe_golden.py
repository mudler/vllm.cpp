import json, os, torch
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales, marlin_make_workspace_new
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    nvfp4_marlin_process_scales, nvfp4_marlin_process_global_scale, _nvfp4_compute_scale_factor)

torch.manual_seed(0)
dev="cuda"; dt=torch.bfloat16
E,K,N,M,TOPK,BLK = 8,256,128,16,2,16   # experts, hidden, intermediate, tokens, top_k, moe_block
OUT="/tmp/moe_dump"; os.makedirs(OUT, exist_ok=True)
def save(name, t):
    t=t.contiguous().cpu()
    if t.dtype==torch.bfloat16: t=t.view(torch.int16)
    t.numpy().tofile(f"{OUT}/{name}.bin")

# --- per-expert NVFP4 quant (linear scales, modelopt style) ---
W = torch.randn(E,N,K,device=dev,dtype=dt)*0.2
packed=[]; bscale=[]; s2=[]
for e in range(E):
    amax = W[e].abs().max().float()
    g = (448.0*6.0)/amax                       # input_global_scale
    p,bs = ops.scaled_fp4_quant(W[e], g, is_sf_swizzled_layout=False)
    packed.append(p.view(torch.uint8)); bscale.append(bs); s2.append((1.0/g).reshape(1).float())
w13 = torch.stack(packed,0)                      # [E,N,K/2] uint8
w13_scale = torch.stack(bscale,0)                # [E,N,K/16] fp8e4m3
w13_scale_2 = torch.cat(s2,0)                    # [E] f32

# --- repack to marlin (w13, num_shards=1, no pad: K%128==0, N%64==0) ---
perm = torch.empty(0,dtype=torch.int,device=dev)
qw=[]
for e in range(E):
    q = w13[e].view(torch.int32).T.contiguous()
    qw.append(ops.gptq_marlin_repack(b_q_weight=q, perm=perm, size_k=K, size_n=N, num_bits=4, is_a_8bit=False))
w13_marlin = torch.stack(qw,0)                   # [E, K/16, N*8/pack] int32

scales_bf = w13_scale.to(dt)
comb_sf = _nvfp4_compute_scale_factor(scales_bf, dt)
sc=[]
for e in range(E):
    s = scales_bf[e].T
    ms = marlin_permute_scales(s=s, size_k=K, size_n=N, group_size=16)
    ms,_ = nvfp4_marlin_process_scales(ms, scale_factor=comb_sf, a_dtype=dt)
    sc.append(ms)
w13_scale_marlin = torch.stack(sc,0)             # [E, K/16, N] fp8e4m3
g_scales = nvfp4_marlin_process_global_scale(w13_scale_2, dt) / comb_sf   # [E] f32

# --- align ---
topk_ids = torch.randint(0,E,(M,TOPK),device=dev,dtype=torch.int32)
topk_w   = torch.rand(M,TOPK,device=dev,dtype=torch.float32)
sorted_ids, expert_ids, num_pad = moe_align_block_size(topk_ids, BLK, E)
sorted_ids=sorted_ids.to(torch.int32); expert_ids=expert_ids.to(torch.int32); num_pad=num_pad.to(torch.int32)
workspace = marlin_make_workspace_new(dev, 4)

X = torch.randn(M,K,device=dev,dtype=dt)*0.2
C = ops.moe_wna16_marlin_gemm(
    X, None, w13_marlin, None, w13_scale_marlin, None, g_scales, None, None, None,
    workspace, sorted_ids, expert_ids, num_pad, topk_w,
    moe_block_size=BLK, top_k=TOPK, mul_topk_weights=False,
    b_q_type=scalar_types.float4_e2m1f, size_m=M, size_n=N, size_k=K,
    is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
    thread_k=-1, thread_n=-1, blocks_per_sm=0)
print("golden C", tuple(C.shape), C.dtype, "finite", torch.isfinite(C).all().item())

save("a", X); save("b_q_weight", w13_marlin); save("b_scales", w13_scale_marlin.view(torch.uint8))
save("global_scale", g_scales.float()); save("workspace", workspace.to(torch.int32))
save("sorted_ids", sorted_ids); save("expert_ids", expert_ids); save("num_pad", num_pad)
save("topk_w", topk_w); save("C_golden", C)
json.dump(dict(E=E,K=K,N=N,M=M,TOPK=TOPK,BLK=BLK,
               sorted_len=int(sorted_ids.numel()), expert_ids_len=int(expert_ids.numel()),
               workspace_len=int(workspace.numel()),
               bqw=list(w13_marlin.shape), bsc=list(w13_scale_marlin.shape),
               C=list(C.shape)), open(f"{OUT}/params.json","w"))
print("dumped to", OUT)

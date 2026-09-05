#!/usr/bin/env python3
"""Run every arm on the COMMITTED chunked-oracle golden
tests/parity/goldens/gdn_prefill_bf16_realdims/ -- real vLLM Triton
chunk_gated_delta_rule output dumped on a GB10. Metric = max|diff|, the same
metric the manifest's 2.29e-04 / 2.25e-03 use."""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_decomp import *

G = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'tests', 'parity', 'goldens', 'gdn_prefill_bf16_realdims')
man = json.load(open(os.path.join(G, 'manifest.json')))
L = lambda n: np.load(os.path.join(G, n))
def unbf16(u):  # raw bf16 bit patterns stored as uint16
    return (u.astype(np.uint32) << np.uint32(16)).view(np.float32)
q, k, v = unbf16(L('q.npy')), unbf16(L('k.npy')), unbf16(L('v.npy'))
g, beta = L('g.npy'), L('beta.npy')
si, so, oo = L('state_in.npy'), L('state_out.npy'), L('out.npy')
qsl = L('query_start_loc.npy')
print("shapes/dtypes:", {n: (a.shape, a.dtype.str) for n, a in
      [('q',q),('k',k),('v',v),('g',g),('beta',beta),('state_in',si),('out',oo),
       ('state_out',so),('qsl',qsl)]})
scale = man['args']['scale']; Hk, Hv = man['args']['Hk'], man['args']['Hv']
Dk, Dv = man['args']['Dk'], man['args']['Dv']
ratio = Hv // Hk
print("qsl:", qsl, " scale:", scale, " Hk,Hv,Dk,Dv:", Hk, Hv, Dk, Dv)
print("state_in seq0 nonzero:", int(np.count_nonzero(si[0])), " seq1 nonzero:", int(np.count_nonzero(si[1])))
# inputs are stored f32-of-bf16; assert that (a probe on the wrong inputs is not a probe)
for nm, a in [('q', q), ('k', k), ('v', v)]:
    assert np.array_equal(a.view(np.uint32), bf16(a).view(np.uint32)), f"{nm} is not bf16-exact"
print("CHECK q,k,v are bf16-exact values: OK")

def run(arm):
    o = np.zeros(oo.shape, dtype=np.float64)  # container only; each arm computed in its own dtype
    S = np.zeros(si.shape, dtype=np.float64)
    nch = 0
    for s in range(len(qsl) - 1):
        b, e = int(qsl[s]), int(qsl[s + 1])
        for hv in range(Hv):
            hk = hv // ratio
            qh = q[b:e, hk, :].astype(np.float32); kh = k[b:e, hk, :].astype(np.float32)
            vh = v[b:e, hv, :].astype(np.float32)
            gh = g[b:e, hv].astype(np.float32); bh = beta[b:e, hv].astype(np.float32)
            h0 = si[s, hv].astype(np.float32)
            if arm == 'ref':
                oh, Sh = seq_scan(qh, kh, vh, gh, bh, h0, scale, np.float64)
            elif arm == 'seq_f32':
                oh, Sh = seq_scan(qh, kh, vh, gh, bh, h0, scale, np.float32, prescale_q=True)
            elif arm == 'chunk_f64':
                oh, Sh, c = chunked(qh,kh,vh,gh,bh,h0,scale,np.float64,64,R={}); nch += c
            elif arm == 'chunk_f32':
                oh, Sh, c = chunked(qh,kh,vh,gh,bh,h0,scale,np.float32,64,R={}); nch += c
            elif arm == 'chunk_up':
                oh, Sh, c = chunked(qh,kh,vh,gh,bh,h0,scale,np.float32,64,R=UPSTREAM); nch += c
            elif arm == 'chunk_up_tf32':
                oh, Sh, c = chunked(qh,kh,vh,gh,bh,h0,scale,np.float32,64,R=UPSTREAM_TF32); nch += c
            o[b:e, hv, :] = oh; S[s, hv] = Sh
    return o, S, nch

oref, Sref, _ = run('ref')
res = {a: run(a) for a in ['seq_f32','chunk_f64','chunk_f32','chunk_up','chunk_up_tf32']}
mx = lambda a, b: float(np.abs(np.asarray(a,np.float64) - np.asarray(b,np.float64)).max())

print(f"\n=== A. reproduce the manifest's own numbers ===")
print(f"  manifest chunk_vs_sequential_max_abs_out   = {man['args']['chunk_vs_sequential_max_abs_out']:.6e}")
print(f"  manifest chunk_vs_sequential_max_abs_state = {man['args']['chunk_vs_sequential_max_abs_state']:.6e}")
print(f"  GOLDEN out.npy       vs my f64 sequential ref : max|d| = {mx(oo, oref):.6e}")
print(f"  GOLDEN state_out.npy vs my f64 sequential ref : max|d| = {mx(so, Sref):.6e}")
print(f"\n=== B. does the numpy replica reproduce the REAL Triton kernel? ===")
for a in ['chunk_up', 'chunk_up_tf32', 'chunk_f32', 'chunk_f64', 'seq_f32']:
    o, S, _ = res[a]
    print(f"  {a:14s} vs GOLDEN out.npy : max|d| = {mx(o, oo):.6e}   vs state_out.npy : {mx(S, so):.6e}")
print(f"\n=== C. THE DECOMPOSITION, on this golden, max|diff| vs the exact f64 recurrence ===")
print(f"  {'arm':16s} {'max|d| out':>14s} {'max|d| state':>14s}")
for a in ['seq_f32','chunk_f64','chunk_f32','chunk_up','chunk_up_tf32']:
    o, S, _ = res[a]
    print(f"  {a:16s} {mx(o, oref):14.6e} {mx(S, Sref):14.6e}")
print(f"\n  chunks processed by the chunked arms: {res['chunk_up'][2]} "
      f"(expect {sum((int(qsl[s+1])-int(qsl[s])+63)//64 for s in range(len(qsl)-1))*Hv})")
print(f"\n=== D. where would an f32-chunked CPU port SIT? (max|diff| on out) ===")
print(f"  f32-chunked  vs  our CPU sequential f32 : {mx(res['chunk_f32'][0], res['seq_f32'][0]):.6e}")
print(f"  f32-chunked  vs  vLLM golden out.npy    : {mx(res['chunk_f32'][0], oo):.6e}")
print(f"  our CPU seq  vs  vLLM golden out.npy    : {mx(res['seq_f32'][0], oo):.6e}")

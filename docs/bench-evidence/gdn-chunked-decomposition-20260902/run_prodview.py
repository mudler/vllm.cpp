#!/usr/bin/env python3
"""Arm-vs-arm at the PRODUCTION output dtype: how many bf16 output elements
differ from vLLM's real chunked kernel, per arm."""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_decomp import *
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
                 'tests', 'parity', 'goldens', 'gdn_prefill_bf16_realdims')
man = json.load(open(os.path.join(G, 'manifest.json'))); L = lambda n: np.load(os.path.join(G, n))
unb = lambda u: (u.astype(np.uint32) << np.uint32(16)).view(np.float32)
q, k, v = unb(L('q.npy')), unb(L('k.npy')), unb(L('v.npy'))
g, beta = L('g.npy'), L('beta.npy')
si, so, oo = L('state_in.npy'), L('state_out.npy'), L('out.npy'); qsl = L('query_start_loc.npy')
scale = man['args']['scale']; Hv = man['args']['Hv']
def go(kind, R=None):
    o = np.zeros(oo.shape, np.float64); S = np.zeros(si.shape, np.float64)
    for s in range(len(qsl) - 1):
        b, e = int(qsl[s]), int(qsl[s + 1])
        for hv in range(Hv):
            a = (q[b:e,0,:], k[b:e,0,:], v[b:e,hv,:], g[b:e,hv], beta[b:e,hv], si[s,hv], scale)
            if kind == 'seq64':   oh, Sh    = seq_scan(*a, np.float64)
            elif kind == 'seq32': oh, Sh    = seq_scan(*a, np.float32, prescale_q=True)
            else:                 oh, Sh, _ = chunked(*a, np.float32, 64, R=R)
            o[b:e, hv, :] = oh; S[s, hv] = Sh
    return o, S
ONLY_OUT = {kk: IDENT for kk in UPSTREAM}; ONLY_OUT['o_st'] = bf16
oref, Sref = go('seq64'); oseq, Sseq = go('seq32')
oup, Sup = go('chunk', UPSTREAM); of32, Sf32 = go('chunk', {}); oio, Sio = go('chunk', ONLY_OUT)
mx = lambda a, b: float(np.abs(a - b).max())
N = oo.size
bfd = lambda a, b: int((bf16(a).view(np.uint32) != bf16(b).view(np.uint32)).sum())
print("=== at the PRODUCTION output dtype (bf16) ===  elements:", N)
print(f"  {'arm':34s} {'max|d| vs vLLM golden':>22s} {'bf16 elems differing':>22s}")
for nm, a in [("vLLM golden out.npy", oo), ("bf16-placement replica", oup),
              ("chunked, f32 intermediates", oio), ("chunked, f32, f32 out", of32),
              ("our CPU sequential f32", oseq), ("exact f64 sequential", oref)]:
    print(f"  {nm:34s} {mx(a, oo):22.4e} {bfd(a, oo):>17d}/{N}")
print(f"\n  f32-intermediate chunked vs our CPU sequential: max|d|={mx(oio,oseq):.4e} bf16 differing={bfd(oio,oseq)}/{N}")
print("\n=== STATE (f32, never bf16) ===")
for nm, a in [("vLLM golden state_out", so), ("bf16-placement replica", Sup),
              ("chunked f32 intermediates", Sio), ("our CPU sequential f32", Sseq)]:
    print(f"  {nm:34s} max|d| vs vLLM golden = {mx(a, so):.4e}")

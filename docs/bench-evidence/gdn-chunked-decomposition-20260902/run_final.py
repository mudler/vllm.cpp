#!/usr/bin/env python3
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_decomp import *
Dk = Dv = 128; scale = float(Dk) ** -0.5; BT = 64
ARMS = ['seq_f32', 'chunk_f64', 'chunk_f32', 'chunk_up']

def sweep(T, seeds, nheads=8):
    E = {a: {'fro': [], 'sum': [], 'S': []} for a in ARMS}
    P = {p: [] for p in ['f32chunk_vs_ourCPU', 'f32chunk_vs_vLLM', 'ourCPU_vs_vLLM']}
    n = 0
    for sd in seeds:
        rng = np.random.default_rng(sd)
        for h in range(nheads):
            q, k, v, g, beta, h0 = make_inputs(rng, T, Dk, Dv)
            oref, Sref = seq_scan(q, k, v, g, beta, h0, scale, np.float64)
            got = {}
            got['seq_f32']   = seq_scan(q, k, v, g, beta, h0, scale, np.float32, prescale_q=True)
            got['chunk_f64'] = chunked(q,k,v,g,beta,h0,scale,np.float64,BT,R={})[:2]
            got['chunk_f32'] = chunked(q,k,v,g,beta,h0,scale,np.float32,BT,R={})[:2]
            got['chunk_up']  = chunked(q,k,v,g,beta,h0,scale,np.float32,BT,R=UPSTREAM)[:2]
            for a in ARMS:
                o, S = got[a]
                E[a]['fro'].append(rel(o, oref)); E[a]['sum'].append(relsum(o, oref))
                E[a]['S'].append(rel(S, Sref))
            P['f32chunk_vs_ourCPU'].append(rel(got['chunk_f32'][0], got['seq_f32'][0]))
            P['f32chunk_vs_vLLM'].append(rel(got['chunk_f32'][0], got['chunk_up'][0]))
            P['ourCPU_vs_vLLM'].append(rel(got['seq_f32'][0], got['chunk_up'][0]))
            n += 1
    return E, P, n

def q(x): 
    a = np.array(x); return f"{np.median(a):.3e} [{a.min():.2e},{a.max():.2e}]"

seeds = [301, 302, 303, 304]
for T in [5, 64, 256, 1024]:
    E, P, n = sweep(T, seeds)
    assert np.median(E['chunk_f64']['fro']) < 1e-12
    print(f"\n### T={T}  n={n} (head,seed) samples   metric = median [min,max]")
    print(f"{'arm':12s} {'||o-oref||/||oref||':>30s} {'|S|o|-S|oref||/max':>30s} {'state rel err':>30s}")
    for a in ARMS:
        print(f"{a:12s} {q(E[a]['fro']):>30s} {q(E[a]['sum']):>30s} {q(E[a]['S']):>30s}")
    print(f"  bf16 term / reassociation term (fro) = "
          f"{np.median(E['chunk_up']['fro'])/np.median(E['chunk_f32']['fro']):.0f}x")
    for p in P: print(f"  d({p:22s}) = {q(P[p])}")

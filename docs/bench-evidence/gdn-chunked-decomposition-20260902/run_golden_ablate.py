import sys, os, numpy as np, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_decomp import *
G=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'tests', 'parity', 'goldens', 'gdn_prefill_bf16_realdims')
man=json.load(open(G+'/manifest.json')); L=lambda n: np.load(G+'/'+n)
unb=lambda u:(u.astype(np.uint32)<<np.uint32(16)).view(np.float32)
q,k,v=unb(L('q.npy')),unb(L('k.npy')),unb(L('v.npy')); g,beta=L('g.npy'),L('beta.npy')
si,so,oo=L('state_in.npy'),L('state_out.npy'),L('out.npy'); qsl=L('query_start_loc.npy')
scale=man['args']['scale']; Hv=man['args']['Hv']
SITES=['Ai','wu_op','wu_st','h_snap','h_dot','vnew_st','vdec','Ao','o_st']
LAB={'Ai':"solve_tril out -> bf16       chunk.py:50",
 'wu_op':"(beta*v),(beta*eG*k) -> bf16 wy_fast.py:92,114",
 'wu_st':"u,w stores -> bf16           wy_fast.py:94,116",
 'h_snap':"h chunk-start snap -> bf16   chunk_delta_h.py:352",
 'h_dot':"h operand of w@h^T -> bf16   chunk_delta_h.py:178",
 'vnew_st':"v_new store -> bf16          chunk_delta_h.py:206",
 'vdec':"decayed v_new -> bf16        chunk_delta_h.py:274",
 'Ao':"intra-chunk A -> bf16        chunk_o.py:137",
 'o_st':"o store -> bf16              chunk_o.py:138"}
def run(R, D=np.float32):
    o=np.zeros(oo.shape,np.float64); S=np.zeros(si.shape,np.float64)
    for s in range(len(qsl)-1):
        b,e=int(qsl[s]),int(qsl[s+1])
        for hv in range(Hv):
            oh,Sh,_=chunked(q[b:e,0,:],k[b:e,0,:],v[b:e,hv,:],g[b:e,hv],beta[b:e,hv],
                            si[s,hv],scale,D,64,R=R)
            o[b:e,hv,:]=oh; S[s,hv]=Sh
    return o,S
def ref():
    o=np.zeros(oo.shape,np.float64); S=np.zeros(si.shape,np.float64)
    for s in range(len(qsl)-1):
        b,e=int(qsl[s]),int(qsl[s+1])
        for hv in range(Hv):
            oh,Sh=seq_scan(q[b:e,0,:],k[b:e,0,:],v[b:e,hv,:],g[b:e,hv],beta[b:e,hv],
                           si[s,hv],scale,np.float64)
            o[b:e,hv,:]=oh; S[s,hv]=Sh
    return o,S
oref,Sref=ref()
mx=lambda a,b: float(np.abs(a-b).max())
oup,Sup=run(UPSTREAM); of32,Sf32=run({})
print(f"all bf16 ON  (vLLM)  : out {mx(oup,oref):.4e}  state {mx(Sup,Sref):.4e}")
print(f"all bf16 OFF (f32)   : out {mx(of32,oref):.4e}  state {mx(Sf32,Sref):.4e}")
print(f"\n{'site':30s} {'OFF: out':>11s} {'OFF: state':>11s} {'ONLY: out':>11s} {'ONLY: state':>12s} {'moved o':>8s}")
for s in SITES:
    Roff=dict(UPSTREAM); Roff[s]=IDENT
    Ron={kk:IDENT for kk in UPSTREAM}; Ron[s]=bf16
    ooff,Soff=run(Roff); oon,Son=run(Ron)
    moved = "yes" if not np.array_equal(ooff,oup) else "NO"
    print(f"{LAB[s]:30s} {mx(ooff,oref):11.4e} {mx(Soff,Sref):11.4e} "
          f"{mx(oon,oref):11.4e} {mx(Son,Sref):12.4e} {moved:>8s}")

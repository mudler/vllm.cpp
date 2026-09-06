#!/usr/bin/env python3
"""Compare two arms' VT_MOE_SEL_FP output, GROUPED BY MODEL FORWARD.

WHY THIS AND NOT `selcmp.py` (wave MOEDIV, #2552). That script splits the calls
into PREFILL and DECODE and reports one flip count for each. The open clause
does not ask about "decode": it asks about model forwards 4, 6 and 7, which are
the three whose sampled ids disagree, against forwards 1, 2, 3 and 5, whose ids
AGREE. One aggregated decode number cannot separate those two populations, and
a divergence that is present at every decode forward means something different
from one that starts at forward 4.

THE FORWARD AXIS IS DERIVED, NOT ASSUMED. `VT_MOE_SEL_FP` counts MoE BLOCK
invocations, so the mapping from a call index to a forward is the number of MoE
blocks per forward, L. L is read off the data as the number of prefill (T>1)
digests, and then ASSERTED: the decode digests must be an exact multiple of L,
and every forward must contain exactly L calls. A run whose structure does not
satisfy that is reported as UNSTRUCTURED and no per-forward verdict is quoted,
because an off-by-one in this mapping would attribute a flip to the wrong
forward and that is the whole measurement.

WHAT IS AND IS NOT REPORTED. The verdict is SET EQUALITY of the routed expert
ids, per token slot, per call. That is a discrete property with bimodal error
and it is reported as a count, never as a tolerance. `ulps` is the bf16 boundary
margin the tap already computed. NO RATIO OF NORMS IS QUOTED AS A MAGNITUDE:
this row withdrew `rel(sumabs)` for exactly that, so the digest's x/logit/exp/shr
axes appear only as "first call at which this axis stops being bit-identical",
which is an ordering fact and not a magnitude.
"""
import sys, os, json


def fields(line):
    f = {}
    for tok in line.split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            f[k] = v
    return f


def load(p):
    """-> (digests {call: fields}, toks {(call,tok): fields}, nlines)"""
    dig, toks = {}, {}
    n = 0
    if not os.path.exists(p):
        return dig, toks, n
    for ln in open(p):
        if not ln.startswith('moesel call='):
            continue
        n += 1
        f = fields(ln)
        c = int(f['call'])
        if ln.rstrip().endswith('END'):
            dig[c] = f
        else:
            toks[(c, int(f['tok']))] = f
    return dig, toks, n


def bitdiff(a, b):
    """True when two decimal strings are not the same value. An ORDERING fact."""
    try:
        return float(a) != float(b)
    except (TypeError, ValueError):
        return a != b


A_p, B_p, label = sys.argv[1], sys.argv[2], sys.argv[3]
out_json = sys.argv[4] if len(sys.argv) > 4 else None
dA, tA, nA = load(A_p)
dB, tB, nB = load(B_p)

R = []          # RESULT lines, echoed by the job
summary = {}


def res(s):
    R.append(s)
    print('RESULT ' + s)


print('=== %s ===' % label)
res('SELFWD %s COUNT base_digests=%d other_digests=%d base_tok=%d other_tok=%d'
    % (label, len(dA), len(dB), len(tA), len(tB)))

if not dA or not dB:
    res('SELFWD %s VERDICT VOID: one side printed NO digest line; the instrument did not run'
        % label)
    if out_json:
        json.dump({'void': True}, open(out_json, 'w'))
    sys.exit(40)

# ---- the counted property: `lines=` on the last digest must equal the value
# lines actually seen. An instrument that never ran and two arms whose taps
# agreed look identical in a diff; this is what separates them.
ok_counted = True
for nm, d, t in (('base', dA, tA), ('other', dB, tB)):
    last = max(d)
    got = int(d[last]['lines'])
    want = len(t)
    agree = (got == want)
    ok_counted = ok_counted and agree
    res('SELFWD %s COUNTED-PROPERTY %s last_digest_lines=%d value_lines_seen=%d agree=%s'
        % (label, nm, got, want, agree))
if not ok_counted:
    res('SELFWD %s VERDICT VOID: the counted property disagrees; lines were lost' % label)
    if out_json:
        json.dump({'void': True}, open(out_json, 'w'))
    sys.exit(40)

# ---- derive the forward axis
common = sorted(set(dA) & set(dB))
pre = [c for c in common if int(dA[c]['T']) > 1]
dec = [c for c in common if int(dA[c]['T']) == 1]
L = len(pre)
res('SELFWD %s STRUCTURE calls_common=%d prefill_calls(L)=%d decode_calls=%d T_prefill=%s'
    % (label, len(common), L, len(dec), dA[pre[0]]['T'] if pre else 'NONE'))

structured = (
    L > 0
    and len(dec) % L == 0
    and pre == list(range(0, L))
    and dec == list(range(L, L + len(dec)))
    and all(int(dB[c]['T']) == int(dA[c]['T']) for c in common)
)
res('SELFWD %s STRUCTURE contiguous_and_uniform=%s' % (label, structured))
if not structured:
    res('SELFWD %s VERDICT UNSTRUCTURED: the call->forward mapping is not derivable; no '
        'per-forward verdict is quoted' % label)
    if out_json:
        json.dump({'unstructured': True}, open(out_json, 'w'))
    sys.exit(41)

n_fwd = (L + len(dec)) // L
res('SELFWD %s STRUCTURE forwards=%d moe_blocks_per_forward=%d' % (label, n_fwd, L))

# ---- per-forward verdict
per = {}
print('%-4s %-7s %-6s %-9s %-9s %-10s %-9s' %
      ('fwd', 'phase', 'calls', 'slots', 'FLIPPED', 'hash_mism', 'first_L'))
for f in range(n_fwd):
    calls = list(range(f * L, (f + 1) * L))
    slots = 0
    flipped = []
    for c in calls:
        T = int(dA[c]['T'])
        for t in range(T):
            a = tA.get((c, t))
            b = tB.get((c, t))
            if a is None or b is None:
                continue
            slots += 1
            if a['ids'] != b['ids']:
                flipped.append((c, t, a, b))
    hash_mism = [c for c in calls if dA[c]['sel'] != dB[c]['sel']]
    first_layer = (flipped[0][0] - f * L) if flipped else None
    phase = 'prefill' if int(dA[calls[0]]['T']) > 1 else 'decode'
    per[f] = {
        'phase': phase, 'calls': len(calls), 'slots': slots,
        'flipped': len(flipped), 'hash_mismatch': len(hash_mism),
        'first_flip_layer': first_layer,
        'first_flip': ([flipped[0][0], flipped[0][1],
                        flipped[0][2]['ids'], flipped[0][3]['ids'],
                        flipped[0][2]['ulps'], flipped[0][3]['ulps']]
                       if flipped else None),
    }
    print('%-4d %-7s %-6d %-9d %-9d %-10d %-9s' %
          (f, phase, len(calls), slots, len(flipped), len(hash_mism),
           first_layer if first_layer is not None else '-'))
    res('SELFWD %s FWD %d %s calls=%d slots=%d FLIPPED=%d hash_mismatch=%d first_flip_layer=%s'
        % (label, f, phase, len(calls), slots, len(flipped), len(hash_mism),
           first_layer if first_layer is not None else '-'))
    if flipped:
        c, t, a, b = flipped[0]
        print('     first flip: call=%d layer=%d tok=%d' % (c, c - f * L, t))
        print('        base  ids=%s ulps=%s lo=%s(%s) hi=%s(%s)'
              % (a['ids'], a['ulps'], a['lo'], a['lo_raw'], a['hi'], a['hi_raw']))
        print('        other ids=%s ulps=%s lo=%s(%s) hi=%s(%s)'
              % (b['ids'], b['ulps'], b['lo'], b['lo_raw'], b['hi'], b['hi_raw']))
        res('SELFWD %s FWD %d FIRSTFLIP call=%d layer=%d tok=%d base_ids=%s other_ids=%s '
            'base_ulps=%s other_ulps=%s'
            % (label, f, c, c - f * L, t, a['ids'], b['ids'], a['ulps'], b['ulps']))

# ---- the headline the open clause asks for
DIVERGE = [4, 6, 7]     # the forwards whose SAMPLED ids disagree
AGREE = [1, 2, 3, 5]    # decode forwards whose sampled ids agree
for name, group in (('DIVERGING', DIVERGE), ('AGREEING', AGREE)):
    g = [f for f in group if f in per]
    if not g:
        res('SELFWD %s GROUP %s NOT REACHED (forwards present: %s)'
            % (label, name, sorted(per)))
        continue
    tot = sum(per[f]['flipped'] for f in g)
    slots = sum(per[f]['slots'] for f in g)
    res('SELFWD %s GROUP %s forwards=%s slots=%d FLIPPED=%d'
        % (label, name, g, slots, tot))

reached = max(per) if per else -1
res('SELFWD %s HIGHEST FORWARD REACHED = %d (the open clause needs >= 7)' % (label, reached))

# ---- CALL 0 IS THE BRACKETED QUANTITY. #2552 brackets the layer-0 expert-flip
# threshold between 2.139e-05 (CPU-CTRL vs CUDA-GDNSEQ; layer 0 agrees on all
# five tokens) and 4.999e-04 (CPU-CTRL vs CUDA-PROD; layer 0 flips at token 2),
# both read off the `L00 mhc.mix` tap. The MoE tap's own `x` axis IS that tensor
# -- MOEDIV's first digest reads x=3613.82031 and PREFILLDIV's CPU-CTRL
# `L00 mhc.mix` reads 3613.82031 -- so call 0 here is the same layer-0 boundary
# the bracket is about, measured as a SELECTION rather than as a norm.
c0 = 0
if c0 in dA and c0 in dB:
    T0 = int(dA[c0]['T'])
    flips0 = []
    for t in range(T0):
        a = tA.get((c0, t))
        b = tB.get((c0, t))
        if a is None or b is None:
            continue
        if a['ids'] != b['ids']:
            flips0.append(t)
    res('SELFWD %s CALL0 x_base=%s x_other=%s (compare with the recorded L00 mhc.mix pair)'
        % (label, dA[c0]['x'], dB[c0]['x']))
    res('SELFWD %s CALL0 sel_base=%s sel_other=%s hash_equal=%s'
        % (label, dA[c0]['sel'], dB[c0]['sel'], dA[c0]['sel'] == dB[c0]['sel']))
    res('SELFWD %s CALL0 LAYER-0 VERDICT: tokens=%d FLIPPED=%d -> %s'
        % (label, T0, len(flips0),
           'NO FLIP (layer 0 agrees on every token)' if not flips0
           else 'FLIP at token(s) %s' % flips0))
    for t in range(T0):
        a = tA.get((c0, t))
        b = tB.get((c0, t))
        if a is None or b is None:
            continue
        res('SELFWD %s CALL0 tok=%d equal=%s base_ulps=%s other_ulps=%s base_ids=%s other_ids=%s'
            % (label, t, a['ids'] == b['ids'], a['ulps'], b['ulps'], a['ids'], b['ids']))
else:
    res('SELFWD %s CALL0 ABSENT: no layer-0 digest on one side' % label)

# ---- the exact-bf16-tie rate, which is #2552's bimodal term measured directly
for nm, dd, tt in (('base', dA, tA), ('other', dB, tB)):
    n0 = sum(1 for k in tt if tt[k]['ulps'] == '0')
    nn = sum(1 for k in tt if int(tt[k]['ulps']) >= 0)
    res('SELFWD %s TIE-RATE %s exact_bf16_ties=%d of %d boundaries = %.1f%%'
        % (label, nm, n0, nn, (100.0 * n0 / nn) if nn else 0.0))

# ---- ORDERING ONLY on the value axes. No magnitude is quoted from these.
for axis in ('x', 'logit', 'exp', 'shr'):
    first = None
    for c in common:
        if bitdiff(dA[c][axis], dB[c][axis]):
            first = c
            break
    res('SELFWD %s AXIS %-5s first_call_not_bit_identical=%s (forward=%s) [ORDERING ONLY]'
        % (label, axis, first, (first // L) if first is not None else '-'))

tot_flips = sum(per[f]['flipped'] for f in per)
res('SELFWD %s VERDICT total_flipped_slots=%d over_forwards=%d' % (label, tot_flips, n_fwd))

if out_json:
    json.dump({'per_forward': per, 'L': L, 'n_fwd': n_fwd,
               'total_flipped': tot_flips}, open(out_json, 'w'), indent=1)
sys.exit(0)

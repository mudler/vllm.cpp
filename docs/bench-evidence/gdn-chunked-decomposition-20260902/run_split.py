#!/usr/bin/env python3
"""GDNCPUPORT: is a SCHEDULER-SPLIT prefill reproducible by the chunked arm?

Every number quoted as a derivation in these three test comments comes from
here, so a reader does not have to write this driver to check them:

  tests/vllm/models/test_qwen27_paged_forward.cpp  (the T=6 continuity case)
  tests/vllm/v1/test_llm_engine.cpp                (chunked-prefill logprobs)
  tests/vt/test_ops_gdn.cpp                        (the ported two-call split)

THE QUESTION. A prefill cut across scheduler steps is bit-identical under the
exact sequential recurrence: the recurrence is token-by-token, so where the cut
falls cannot change the arithmetic. It is NOT bit-identical under vLLM's chunked
WY decomposition, because each extra chunk boundary sends the interactions
across it through a BF16-ROUNDED state snapshot (chunk_delta_h.py:178,352)
instead of the intra-chunk f32 path.

Experiment B separates those two candidate causes: `chunk_f32` runs the SAME
chunked algorithm with f32 intermediates. If the discontinuity were the chunked
reassociation, chunk_f32 would show it too. It does not, so the cause is the
bf16 placement -- which is exactly what cannot be given up without giving up the
mirror. Upstream says the same in its own test's words:
`test_chunk_gated_delta_rule_cpu_two_call_split` gates state at atol=rtol=1e-3
and output at 2e-2 with the comment "State must be near-exact; output allows a
looser bound for the bf16 round-trip".

Run: python3 run_split.py     (no arguments, no GPU, ~10 s)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_decomp import UPSTREAM, bf16, chunked, seq_scan  # noqa: E402

D = 128
CHUNK = 64


def inputs(rng, T, g_lo, g_hi):
    """q/k L2-normalised bf16 (the op's precondition), v bf16, g <= 0, beta in (0,1).

    `g_lo/g_hi` is the whole point of the second experiment: upstream's own
    gating (A_log/softplus) makes g roughly -0.7 to -11 PER TOKEN, which decays
    the state to nothing inside one chunk, so where a prefill is split cannot
    matter there. A gentler decay is what actually exercises the boundary.
    """
    q = rng.standard_normal((T, D)).astype(np.float32)
    k = rng.standard_normal((T, D)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    v = bf16(rng.standard_normal((T, D)).astype(np.float32))
    g = rng.uniform(g_lo, g_hi, T).astype(np.float32)
    beta = rng.uniform(0.1, 0.9, T).astype(np.float32)
    return bf16(q), bf16(k), v, g, beta


ARMS = {
    "sequential": lambda a, b, c, d, e, h, s: seq_scan(
        a, b, c, d, e, h, s, np.float32, prescale_q=True),
    "chunk_up": lambda a, b, c, d, e, h, s: chunked(
        a, b, c, d, e, h, s, np.float32, CHUNK, R=UPSTREAM)[:2],
    "chunk_f32": lambda a, b, c, d, e, h, s: chunked(
        a, b, c, d, e, h, s, np.float32, CHUNK, R={})[:2],
}


def run_legs(fn, q, k, v, g, beta, h0, lens, scale):
    """Prefill `lens` as consecutive calls, each seeded with the previous final state."""
    H, outs, t = h0.copy(), [], 0
    for L in lens:
        o, H = fn(q[t:t + L], k[t:t + L], v[t:t + L], g[t:t + L], beta[t:t + L], H, scale)
        outs.append(o)
        t += L
    return np.concatenate(outs, 0), H


def assert_close_ratio(got, ref, atol, rtol):
    """torch.testing.assert_close's own criterion, as a fraction of its bound.

    <= 1.0 means upstream's assertion would pass it. Porting one of upstream's
    numbers as a bare max-abs bound is STRICTER than what upstream asserts.
    """
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    return float((np.abs(got - ref) / (atol + rtol * np.abs(ref))).max())


def experiment_a():
    """T=6 split {3,3} and {2,2,2} -- test_qwen27_paged_forward's own shape."""
    print("=" * 78)
    print("A. T=6, the 27B continuity case's shape. max|d| of a SPLIT prefill")
    print("   against the SAME tokens prefilled in one call.")
    print("=" * 78)
    print(f"{'split':>12} {'arm':12} {'out max|d|':>14} {'state max|d|':>14}")
    rng = np.random.default_rng(7)
    q, k, v, g, beta = inputs(rng, 6, -0.5, 0.0)
    h0 = bf16(rng.standard_normal((D, D)).astype(np.float32) * 0.1)
    scale = 1.0 / np.sqrt(D)
    for lens in ([3, 3], [2, 2, 2]):
        for name, fn in ARMS.items():
            o1, H1 = run_legs(fn, q, k, v, g, beta, h0, [6], scale)
            o2, H2 = run_legs(fn, q, k, v, g, beta, h0, lens, scale)
            do = np.abs(o1.astype(np.float64) - o2).max()
            ds = np.abs(H1.astype(np.float64) - H2).max()
            print(f"{str(lens):>12} {name:12} {do:14.6e} {ds:14.6e}")
    print("\n  READ: `sequential` is EXACTLY 0 -- that is the property the 1e-4")
    print("  bar encoded. `chunk_f32` is ~1e-8, so the chunked REASSOCIATION is")
    print("  continuous. `chunk_up` is ~2e-3, and it differs from chunk_f32 only")
    print("  in where the bf16 rounding falls. The bf16 PLACEMENT is the cause,")
    print("  and it is the half that cannot be dropped without dropping the mirror.")


def experiment_b():
    """upstream's TWO_CALL_SPLITS, on the workload tests/vt/test_ops_gdn.cpp uses."""
    print()
    print("=" * 78)
    print("B. upstream's own TWO_CALL_SPLITS (test_cpu_gdn_ops.py:332-336), scored")
    print("   as a FRACTION of upstream's own assert_close bound (1.0 == its bar).")
    print("=" * 78)
    splits = [(2 * CHUNK, CHUNK), (2 * CHUNK + 17, CHUNK), (2 * CHUNK + 17, CHUNK + 9),
              (4 * CHUNK + 17, 2 * CHUNK), (3 * CHUNK, CHUNK + 1)]
    for label, (g_lo, g_hi) in (("upstream's own gating (g ~ -0.7..-11/token)", (-11.0, -0.7)),
                                ("this suite's g ~ U(-0.5, 0)", (-0.5, 0.0))):
        print(f"\n  {label}")
        print(f"  {'total/split':>14} {'arm':12} {'out ratio':>11} {'state ratio':>12}")
        worst = {}
        for total, split in splits:
            for name, fn in ARMS.items():
                rng = np.random.default_rng(total * 1000 + split)
                q, k, v, g, beta = inputs(rng, total, g_lo, g_hi)
                h0 = np.zeros((D, D), np.float32)  # upstream's zero_state
                scale = 1.0 / np.sqrt(D)
                of, Hf = run_legs(fn, q, k, v, g, beta, h0, [total], scale)
                os_, Hs = run_legs(fn, q, k, v, g, beta, h0, [split, total - split], scale)
                ro = assert_close_ratio(bf16(os_), bf16(of), 2e-2, 2e-2)
                rs = assert_close_ratio(Hs, Hf, 1e-3, 1e-3)
                worst[name] = max(worst.get(name, 0.0), rs)
                if name == "chunk_up":
                    print(f"  {str(total) + '/' + str(split):>14} {name:12} "
                          f"{ro:11.5f} {rs:12.5f}")
        for name in ARMS:
            verdict = "EXCEEDS" if worst[name] > 1.0 else "meets"
            print(f"    worst state ratio, {name:12} = {worst[name]:8.5f}  ({verdict} upstream's own bar)")
    print("\n  READ: on upstream's own gating the state is decayed to nothing")
    print("  inside a chunk, so every arm reads ~0 and the split cannot matter --")
    print("  which is the workload upstream's 1e-3 was set on. On the gentler")
    print("  decay this suite uses, UPSTREAM'S OWN PLACEMENT exceeds upstream's")
    print("  own bound while `sequential` and `chunk_f32` stay at 0. So the bar")
    print("  does not transfer across the workload, and the port is not what")
    print("  fails to meet it. That is why tests/vt/test_ops_gdn.cpp derives its")
    print("  own state bar instead of carrying 1e-3 over unchanged.")


if __name__ == "__main__":
    np.seterr(over="ignore")  # g can underflow exp() on upstream's own gating
    experiment_a()
    experiment_b()

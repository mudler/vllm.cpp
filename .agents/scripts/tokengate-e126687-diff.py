#!/usr/bin/env python3
"""TOKENGATE: diff a CANDIDATE-revision OPT-125m greedy oracle capture against
the committed golden, and re-derive the gate SELECTOR at the candidate.

This is the comparison `scripts/opt-oracle-capture.py` does not do. The capture
script writes a golden and prints a determinism report; nothing in the tree ever
laid two captures beside each other, so the pin-advance W3 step ("regenerate the
golden at the target, diff against the committed one") was done by hand each
time. This makes that step executable.

Three things are checked, and they are NOT the same check:

  1. `greedy_ids.npy` -- the BAR. Byte-equal means the committed golden is still
     the oracle's answer at the candidate revision, so every gate already
     measured against those bytes carries over unchanged.
  2. `p{i}_prompt.i32` -- vLLM's own tokenization. A change here moves the
     INPUT, and a matching `greedy_ids` on a moved input would be a coincidence
     rather than agreement.
  3. `greedy_dist.npy` -- the gate SELECTOR, recomputed from the CANDIDATE's own
     K runs. `tests/vllm/models/test_opt_paged_engine.cpp` uses a STRICT
     token-exact bar with no near-tie band, and what licenses that bar is zero
     multi-valued (prompt,pos) cells. If the candidate oracle is no longer
     self-deterministic, the strict bar must be re-derived, not silently kept --
     and the committed dist file cannot answer that, because it is the OLD
     oracle's measurement.

Exit 0 only when all three hold. Exit 1 is a real difference (a finding, not a
crash); exit 2 is a missing or malformed input, which is this script failing
rather than the comparison failing.

WHICH SIDE'S ABSENCE IS FATAL, because the two are not symmetric. The
CANDIDATE's `greedy_ids.npy` and `greedy_dist.npy` are both required and their
absence exits 2. On the COMMITTED side only `greedy_ids.npy` is required:
`greedy_dist.npy` there is the OLD oracle's measurement of itself, it is not the
licence for the bar at the candidate, and nothing here reads it except to print
its hash beside the candidate's. So deleting the committed dist file exits 0,
deliberately, and the run says `DIST committed ABSENT` rather than passing in
silence. The staged committed set is sha256-asserted by the lease job before
this script runs, so a missing committed file is caught there, not here.
"""
import argparse
import hashlib
import os
import sys

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def need(path):
    if not os.path.exists(path):
        print(f"FATAL missing input: {path}")
        sys.exit(2)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--committed", required=True)
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()

    bad = 0

    # ---- 1. the BAR ---------------------------------------------------------
    c_ids = need(os.path.join(args.committed, "greedy_ids.npy"))
    n_ids = need(os.path.join(args.candidate, "greedy_ids.npy"))
    hc, hn = sha256(c_ids), sha256(n_ids)
    print(f"IDS committed sha256 {hc}")
    print(f"IDS candidate sha256 {hn}")
    a, b = np.load(c_ids), np.load(n_ids)
    print(f"IDS shape committed {a.shape} candidate {b.shape} dtype {a.dtype}/{b.dtype}")
    if a.shape != b.shape:
        print(f"IDS_DRIFT shape {a.shape} != {b.shape}")
        bad = 1
    else:
        diff = np.argwhere(a != b)
        print(f"IDS mismatched_positions {len(diff)} of {a.size}")
        for (i, j) in diff[:32]:
            print(f"IDS_DIFF prompt[{i}] pos {j}: committed {a[i, j]} candidate {b[i, j]}")
        if len(diff):
            bad = 1
        else:
            print("IDS_EXACT True  (the committed bar is the candidate oracle's answer)")
    print(f"IDS_BYTE_EQUAL {hc == hn}")

    # ---- 2. the INPUT -------------------------------------------------------
    n_prompts = a.shape[0]
    for i in range(n_prompts):
        cp = os.path.join(args.committed, f"p{i}_prompt.i32")
        np_ = os.path.join(args.candidate, f"p{i}_prompt.i32")
        if not (os.path.exists(cp) and os.path.exists(np_)):
            print(f"PROMPT[{i}] MISSING committed={os.path.exists(cp)} candidate={os.path.exists(np_)}")
            bad = 1
            continue
        ca = np.fromfile(cp, dtype="<i4")
        na = np.fromfile(np_, dtype="<i4")
        eq = ca.shape == na.shape and bool((ca == na).all())
        print(f"PROMPT[{i}] len {len(ca)}/{len(na)} EQUAL {eq}")
        if not eq:
            print(f"PROMPT_DIFF[{i}] committed {ca.tolist()}")
            print(f"PROMPT_DIFF[{i}] candidate {na.tolist()}")
            bad = 1

    # ---- 3. the gate SELECTOR, recomputed at the CANDIDATE -------------------
    n_dist = need(os.path.join(args.candidate, "greedy_dist.npy"))
    d = np.load(n_dist)
    if d.ndim != 3:
        print(f"FATAL candidate greedy_dist has ndim {d.ndim}, expected 3")
        sys.exit(2)
    N, T, K = d.shape
    multi = 0
    for i in range(N):
        for j in range(T):
            if len(set(int(x) for x in d[i, j, :])) > 1:
                multi += 1
                print(f"SELECTOR_MULTI prompt[{i}] pos {j}: {sorted(set(int(x) for x in d[i, j, :]))}")
    print(f"SELECTOR K={K} multi_valued_cells {multi}")
    if multi == 0:
        print("SELECTOR STRICT token-exact gate is licensed at the candidate")
    else:
        print("SELECTOR NON-DETERMINISTIC at the candidate: the STRICT bar in "
              "test_opt_paged_engine.cpp must be RE-DERIVED, not kept")
        bad = 1
    cd = os.path.join(args.committed, "greedy_dist.npy")
    print(f"DIST candidate sha256 {sha256(n_dist)}")
    if os.path.exists(cd):
        print(f"DIST committed sha256 {sha256(cd)}")
    else:
        # NOT an error, and said out loud so it is not mistaken for one. The
        # committed dist is the previous oracle's self-measurement; the licence
        # for the strict bar at the candidate is the candidate's own K runs,
        # recomputed above.
        print("DIST committed ABSENT (not required: the selector is the "
              "CANDIDATE's own K runs, recomputed above)")

    print(f"TOKENGATE_VERDICT {'PASS' if bad == 0 else 'DRIFT'}")
    sys.exit(bad)


if __name__ == "__main__":
    main()

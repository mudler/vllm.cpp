#!/usr/bin/env python3
# Teacher-forced near-tie gap for the int8-dot LANE pair, with the PINNED
# llama.cpp b10451 as the oracle (KEEPQUANT W4c, issue #3079).
#
# Gap semantics are identical to scripts/qwen3-neartie-gap-transformers.py
# (the ROCm-domain reference): teacher-force on OUR captured prefix, gap at
# cell (i, j) = max(0, argmax_logp - our_logp) in nats over the oracle's own
# logits at position P+j-1, reported in mnats; gap == 0 with different ids is
# an exact tie (legit); a cell whose gap exceeds the ratified band is a
# forward divergence. The only difference is the oracle: raw f32 logits
# dumped from the pinned libllama (llama_get_logits_ith) by the minimal
# reader, instead of a HF model — `llama-perplexity --save-all-logits` at the
# pin cannot serve (65535-level uint16-quantized log-softmax rows, positions
# n_ctx/2..n_ctx-2 only, >= 2*n_ctx tokens required; see the row spec).
#
# Oracle dump layout (written by the reader, little-endian):
#   char[8] "VTLGDUMP"; int32 n_positions; int32 n_vocab;
#   per position: int32 token_id, then n_vocab float32 logits.
# The reader's dump is self-describing: the script verifies the embedded
# token ids equal prompt_ids + our_ids per prompt (bit-level proof the oracle
# teacher-forced OUR prefix), and that n_vocab == 248320.
#
# Emits into --golden-dir:
#   our_ids.npy            [N,T] i32   (from the raw .i32 capture)
#   neartie_gap_mnats.npy  [N,T] i32   (99_999_000 = outside top-K)
#   greedy_ids.npy         [N,T] i32   (optional: the pin's greedy, from
#                                       --greedy-prefix greedy_p{i}.i32)
from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np

PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "In the beginning God created",
    "The quick brown fox jumps over",
    "def fibonacci(n):",
    "Water boils at a temperature of",
    "The theory of relativity was developed by",
    "To be or not to be, that is",
    "The largest planet in our solar system is",
    "Machine learning is a subfield of",
    "The mitochondria is the powerhouse of",
    "Roses are red, violets are",
    "The first president of the United States was",
    "E equals m c",
    "A journey of a thousand miles begins with",
    "The chemical symbol for gold is",
]
OUTSIDE_TOPK_MNATS = 99_999_000
MAGIC = b"VTLGDUMP"
EXPECTED_N_VOCAB = 248_320  # the qwen3 vocab this script is written for


def read_dump(path: str):
    raw = open(path, "rb").read()
    if raw[:8] != MAGIC:
        raise SystemExit(f"{path}: bad magic {raw[:8]!r}")
    n, n_vocab = struct.unpack_from("<ii", raw, 8)
    if n_vocab != EXPECTED_N_VOCAB:
        raise SystemExit(f"{path}: n_vocab {n_vocab}, expected {EXPECTED_N_VOCAB}")
    off = 16
    toks = np.empty(n, dtype="<i4")
    rows = np.empty((n, n_vocab), dtype=np.float32)
    for i in range(n):
        toks[i], = struct.unpack_from("<i", raw, off)
        off += 4
        rows[i] = np.frombuffer(raw, dtype="<f4", count=n_vocab, offset=off)
        off += 4 * n_vocab
    if off != len(raw):
        raise SystemExit(f"{path}: trailing bytes ({len(raw) - off})")
    return toks, rows


def log_softmax(row: np.ndarray) -> np.ndarray:
    # float64 accumulation for a stable difference at band-edge cells; the
    # logits themselves are the oracle's raw f32.
    r = row.astype(np.float64)
    m = r.max()
    return r - (m + np.log(np.exp(r - m).sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-dir", required=True)
    ap.add_argument("--oracle-dir", required=True,
                    help="dir with oracle_logits_p{i}.dump from the reader")
    ap.add_argument("--our-ids", default="our_ids.i32",
                    help="raw int32 dump from VT_DUMP_IDS (N*T LE i32)")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--greedy-prefix", default="greedy_p{i}.i32",
                    help="optional per-prompt pin greedy ids (builds "
                         "greedy_ids.npy); empty string skips")
    args = ap.parse_args()

    gdir, odir = args.golden_dir, args.oracle_dir
    our_path = os.path.join(gdir, args.our_ids)
    if not os.path.isfile(our_path):
        print(f"missing {our_path}", file=sys.stderr)
        return 1
    our = np.fromfile(our_path, dtype="<i4")
    N = len(PROMPTS)
    T = args.max_tokens
    if our.size != N * T:
        print(f"{our_path}: {our.size} ids, expected {N}*{T}", file=sys.stderr)
        return 1
    our = our.reshape(N, T)

    greedy = None
    if args.greedy_prefix:
        g = np.zeros((N, T), dtype="<i4")
        for i in range(N):
            p = os.path.join(odir, args.greedy_prefix.format(i=i))
            if not os.path.isfile(p):
                print(f"missing {p}", file=sys.stderr)
                return 1
            g[i] = np.fromfile(p, dtype="<i4")
        greedy = g

    gap_mnats = np.zeros((N, T), dtype="<i4")
    max_gap = 0.0
    worst = None
    n_div = 0
    n_flip = 0
    n_exact_tie = 0
    dist = np.zeros(N, dtype="<i4")  # per-prompt max gap
    print(f"=== teacher-forced near-tie gap (llama.cpp b10451 oracle, "
          f"raw-f32 dumps): {gdir} ===")
    for i in range(N):
        pfile = os.path.join(gdir, f"p{i}_prompt.i32")
        prompt_ids = np.fromfile(pfile, dtype="<i4").tolist()
        P = len(prompt_ids)
        our_ids = [int(x) for x in our[i]]
        toks, rows = read_dump(os.path.join(odir, f"oracle_logits_p{i}.dump"))
        expect = np.array(prompt_ids + our_ids, dtype="<i4")
        if toks.tolist() != expect.tolist():
            print(f"  p{i:2d}: DUMP token ids do not teacher-force OUR prefix",
                  file=sys.stderr)
            return 1
        for j in range(T):
            row = rows[P + j - 1]
            logp = log_softmax(row)
            our_tid = our_ids[j]
            k = min(args.topk, logp.shape[0])
            topi = np.argpartition(-logp, k - 1)[:k]
            top_set = {int(t) for t in topi}
            arg_tid = int(np.argmax(logp))
            arg_lp = float(logp[arg_tid])
            if our_tid in top_set:
                our_lp = float(logp[our_tid])
                gap = max(0.0, arg_lp - our_lp)
                gap_mnats[i, j] = int(round(gap * 1000.0))
                if gap > 0.0 and our_tid != arg_tid:
                    n_flip += 1
                if gap == 0.0 and our_tid != arg_tid:
                    n_exact_tie += 1
                if gap > max_gap:
                    max_gap, worst = gap, (i, j, gap)
            else:
                gap_mnats[i, j] = OUTSIDE_TOPK_MNATS
                print(
                    f"  p{i:2d} tok{j:2d}: OUR TOKEN {our_tid} OUTSIDE "
                    f"top-{args.topk}")
            if greedy is not None and our_ids[j] != int(greedy[i, j]):
                n_div += 1
                print(
                    f"  p{i:2d} tok{j:2d}: our={our_tid} "
                    f"llamacpp_greedy={int(greedy[i, j])} "
                    f"tf_argmax={arg_tid} gap={gap_mnats[i, j] / 1000.0:.4f} "
                    f"nats")
        dist[i] = gap_mnats[i].max()
        print(f"  prompt {i:2d}: max gap {dist[i] / 1000.0:.4f} nats")
        print(f"  prompt {i}/{N - 1} done")

    order = np.argsort(-dist)
    print("=== per-prompt max gap (mnats), worst first ===")
    for i in order:
        print(f"  p{i:2d}: {dist[i]}")
    if worst is not None:
        print(f"=== worst cell {worst[0]},{worst[1]} gap {worst[2] * 1000.0:.0f}"
              f" mnats; flips(argmax!=ours,gap>0)={n_flip} "
              f"exact-ties(gap==0,ids differ)={n_exact_tie} "
              f"token-divergent vs greedy_ids={n_div} ===")
    ids_out = os.path.join(gdir, "our_ids.npy")
    gap_out = os.path.join(gdir, "neartie_gap_mnats.npy")
    np.save(ids_out, our)
    np.save(gap_out, gap_mnats)
    if greedy is not None:
        gout = os.path.join(gdir, "greedy_ids.npy")
        np.save(gout, greedy)
    print(f"wrote {ids_out} + {gap_out} + greedy {gap_mnats.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

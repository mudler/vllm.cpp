#!/usr/bin/env python3
"""Token-length histogram of a corpus, using the checkpoint's own tokenizer.

The PUBLISHED histogram is read back from each server's `usage.prompt_tokens`,
because each engine renders its own chat template and that adds tokens this
script does not see. This is the corpus's own shape, for sizing: it answers
"does the longest prompt fit `--max-model-len`" before a lease is taken.
"""
import argparse, bisect, json, math, statistics
from tokenizers import Tokenizer

def pc(xs, p):
    xs = sorted(xs)
    i = (len(xs) - 1) * p / 100
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--tokenizer", required=True, help="tokenizer.json")
a = ap.parse_args()
tok = Tokenizer.from_file(a.tokenizer)
per, allv = {}, []
for e in json.load(open(a.corpus)):
    text = next(c["value"] for c in e["conversations"] if c["from"] == "human")
    n = len(tok.encode(text, add_special_tokens=False).ids)
    per.setdefault(e["band"], []).append(n)
    allv.append(n)
print("| band | n | min | p50 | p90 | max | mean |")
print("|---|---|---|---|---|---|---|")
for b in sorted(per, key=lambda k: statistics.median(per[k])):
    v = per[b]
    print(f"| `{b}` | {len(v)} | {min(v)} | {pc(v,50):.0f} | {pc(v,90):.0f} "
          f"| {max(v)} | {statistics.mean(v):.0f} |")
print(f"| **all** | {len(allv)} | {min(allv)} | {pc(allv,50):.0f} "
      f"| {pc(allv,90):.0f} | {max(allv)} | {statistics.mean(allv):.0f} |")
edges = [0, 128, 256, 512, 1024, 2048, 4096]
hist = {}
for n in allv:
    hist[bisect.bisect_right(edges, n) - 1] = \
        hist.get(bisect.bisect_right(edges, n) - 1, 0) + 1
print("\n| prompt tokens | prompts |")
print("|---|---|")
for i in sorted(hist):
    hi = edges[i + 1] - 1 if i + 1 < len(edges) else "and up"
    print(f"| {edges[i]}-{hi} | {hist[i]} |")

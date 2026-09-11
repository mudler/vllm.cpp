#!/usr/bin/env python3
"""Compare our DeepSeek-V4 vision token block against the llama.cpp b10766 dump.

Row `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm` W6, issue #2411.

Every file is `[int32 rows][int32 cols][f32 data]`, which is the format
`tools/mtmd/clip.cpp:5853-5873` writes at the pinned oracle for
`MTMD_DEBUG_EMBEDDINGS=<path>`. Pure standard library, so the worker needs no
numpy.

Usage: dsv4v_w6_compare.py <dir> <tag> <lead_pad> <n_llm_h> <n_llm_w>
  reads <dir>/{ours,oracle}-<tag>-block.f32, and when present the stage files
  -vit.f32, -cells.f32 and -input.f32.
"""
import json
import math
import os
import struct
import sys


def load(path):
    with open(path, "rb") as f:
        rows, cols = struct.unpack("<ii", f.read(8))
        data = struct.unpack("<%df" % (rows * cols), f.read(4 * rows * cols))
    return rows, cols, [data[r * cols:(r + 1) * cols] for r in range(rows)]


def bf16(x):
    # round-to-nearest-even to bf16, returned as the widened f32 value
    (u,) = struct.unpack("<I", struct.pack("<f", x))
    if (u & 0x7F800000) == 0x7F800000:
        return x
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", u))[0]


def layout(lead_pad, n_llm_h, n_llm_w):
    # dsv4_get_block_layout + clip.cpp set_input, PROJECTOR_TYPE_DEEPSEEK4V
    rows = n_llm_h + (n_llm_h % 2)
    row_len = n_llm_w + 1
    pad_last = (rows // 2 * row_len) % 2 * 2
    types, cell = ["PAD"] * lead_pad + ["START"], [None] * (lead_pad + 1)
    for t in range(rows * row_len):
        g, rem = divmod(t, 2 * row_len)
        c, r = rem // 2, 2 * g + rem % 2
        if r >= n_llm_h:
            types.append("PAD"); cell.append(None)
        elif c == n_llm_w:
            types.append("NEWLINE"); cell.append(None)
        else:
            types.append("IMAGE"); cell.append(r * n_llm_w + c)
    types += ["PAD"] * pad_last + ["END"]
    cell += [None] * (pad_last + 1)
    return types, cell


def stats(a, b):
    d = [x - y for x, y in zip(a, b)]
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    nd = math.sqrt(sum(x * x for x in d))
    dot = sum(x * y for x, y in zip(a, b))
    return {
        "max_abs": max(abs(x) for x in d),
        "mean_abs": sum(abs(x) for x in d) / len(d),
        "cos": dot / (na * nb) if na and nb else float("nan"),
        "rel_l2": nd / nb if nb else float("nan"),
        "ref_rms": nb / math.sqrt(len(b)),
    }


def matrix_summary(name, ours, ref):
    per = [stats(a, b) for a, b in zip(ours, ref)]
    out = {
        "rows": len(per),
        "max_abs": max(p["max_abs"] for p in per),
        "mean_abs": sum(p["mean_abs"] for p in per) / len(per),
        "min_cos": min(p["cos"] for p in per),
        "mean_cos": sum(p["cos"] for p in per) / len(per),
        "max_rel_l2": max(p["rel_l2"] for p in per),
        "mean_rel_l2": sum(p["rel_l2"] for p in per) / len(per),
        "ref_rms": math.sqrt(sum(p["ref_rms"] ** 2 for p in per) / len(per)),
    }
    print("[%s] %s" % (name, json.dumps(out)))
    return out, per


def best_match(ours, ref):
    """For each of our rows, the reference row with the highest cosine."""
    norms = [math.sqrt(sum(x * x for x in r)) or 1.0 for r in ref]
    hits = []
    for i, a in enumerate(ours):
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        best, arg = -2.0, -1
        for j, b in enumerate(ref):
            c = sum(x * y for x, y in zip(a, b)) / (na * norms[j])
            if c > best:
                best, arg = c, j
        hits.append((arg, best))
    return hits


def main():
    d, tag = sys.argv[1], sys.argv[2]
    lead_pad, n_llm_h, n_llm_w = map(int, sys.argv[3:6])
    report = {"tag": tag, "lead_pad": lead_pad, "grid": [n_llm_h, n_llm_w]}
    types, cell = layout(lead_pad, n_llm_h, n_llm_w)

    orows, ocols, ours = load(os.path.join(d, "ours-%s-block.f32" % tag))
    rrows, rcols, ref = load(os.path.join(d, "oracle-%s-block.f32" % tag))
    report["tokens"] = {"ours": [orows, ocols], "oracle": [rrows, rcols],
                        "layout": len(types)}
    print("tokens ours=%dx%d oracle=%dx%d layout=%d"
          % (orows, ocols, rrows, rcols, len(types)))
    if (orows, ocols) != (rrows, rcols) or orows != len(types):
        report["verdict"] = "SHAPE_MISMATCH"
        print(json.dumps(report))
        json.dump(report, open(os.path.join(d, "report-%s.json" % tag), "w"), indent=1)
        return 2

    # The four sentinels are COPIED weights. The oracle concatenates them in f32
    # (deepseek4v.cpp); we narrow them to bf16 at the join
    # (deepseek_v4_mm.cpp). So the exact comparison is ours == bf16(oracle), and
    # f32 equality is reported beside it.
    sent = {}
    for kind in ("START", "END", "NEWLINE", "PAD"):
        idx = [i for i, t in enumerate(types) if t == kind]
        f32_eq = all(ours[i] == ref[i] for i in idx)
        bf_eq = all(all(a == bf16(b) for a, b in zip(ours[i], ref[i])) for i in idx)
        st = stats(ours[idx[0]], ref[idx[0]])
        sent[kind] = {"rows": idx, "f32_exact": f32_eq,
                      "bf16_of_oracle_exact": bf_eq, "max_abs": st["max_abs"]}
    report["sentinels"] = sent
    print("sentinels", json.dumps(sent))

    img = [i for i, t in enumerate(types) if t == "IMAGE"]
    summ, per = matrix_summary("block-image-rows", [ours[i] for i in img],
                               [ref[i] for i in img])
    report["image_rows"] = summ
    worst = sorted(range(len(img)), key=lambda k: per[k]["rel_l2"])[-5:]
    report["worst_image_rows"] = [
        {"row": img[k], "cell": cell[img[k]], **per[k]} for k in reversed(worst)]
    for w in report["worst_image_rows"]:
        print("worst", json.dumps(w))
    allrows = [stats(a, b) for a, b in zip(ours, ref)]
    report["all_rows"] = {
        "max_abs": max(p["max_abs"] for p in allrows),
        "mean_abs": sum(p["mean_abs"] for p in allrows) / len(allrows),
        "min_cos": min(p["cos"] for p in allrows)}

    # PERMUTATION: does a re-ordering of our image rows collapse the error? If
    # the identity is already the best match for every row, no permutation can.
    hits = best_match([ours[i] for i in img], [ref[i] for i in img])
    ident = sum(1 for k, (arg, _) in enumerate(hits) if arg == k)
    report["permutation"] = {"identity_is_best": ident, "of": len(img),
                             "min_best_cos": min(c for _, c in hits)}
    print("permutation", json.dumps(report["permutation"]))

    for stage in ("input", "vit", "cells"):
        po = os.path.join(d, "ours-%s-%s.f32" % (tag, stage))
        pr = os.path.join(d, "oracle-%s-%s.f32" % (tag, stage))
        if not (os.path.exists(po) and os.path.exists(pr)):
            report[stage] = "absent"
            continue
        a = load(po); b = load(pr)
        if a[:2] != b[:2]:
            report[stage] = {"shape_mismatch": [a[:2], b[:2]]}
            print(stage, report[stage])
            continue
        s, _ = matrix_summary(stage, a[2], b[2])
        if stage == "input":
            s["bf16_of_oracle_exact"] = all(
                x == bf16(y) for ra, rb in zip(a[2], b[2]) for x, y in zip(ra, rb))
        report[stage] = s

    json.dump(report, open(os.path.join(d, "report-%s.json" % tag), "w"), indent=1)
    print("REPORT", os.path.join(d, "report-%s.json" % tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())

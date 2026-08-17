#!/usr/bin/env python3
"""SPEC-MTP-K-GT-1: adjudicate the our-ON vs our-OFF divergence against the oracle.

Our spec-ON stream is not token-identical to our spec-OFF stream on 3 of the 4
prompts, at the SAME position for every k and for the padded control too. Under
greedy accept-iff-equal the emitted stream must not depend on k, so this is
either a real defect or a genuine tie resolved differently by the two paths.

This instrument decides it the way the ratified near-tie doctrine decides it
(`kNearTieMnats = 500`, see scripts/qwen3-apc-neartie-gap.py, the sibling this
mirrors). It teacher-forces the PINNED vLLM oracle on each arm's exact token
sequence and records, at the divergent position, the gap in nats between the
oracle's own argmax logprob and each arm's token.

ONLY THE FIRST DIVERGENCE PER PROMPT IS ADJUDICABLE. After it the two arms carry
different prefixes, so any later position compares two different conditionings
and measures nothing. Every later position is reported as CONTEXT and is
excluded from the verdict, with its own count printed so the reader can see how
many were examined and how many were judged.

A gate that cannot say HOW MANY things it examined has not reported, so every
block below prints its own denominator.

VERDICT per divergent position:
  TIE     both arms' tokens sit within 500 mnats of the oracle argmax and both
          are inside the oracle's top-K. The oracle's own greedy is flat here.
  DEFECT  an arm's token is outside the oracle top-K, or its gap exceeds
          500 mnats. That is a real forward divergence, not a tie.
"""
import argparse
import json
import os
import sys

NEAR_TIE_MNATS = 500          # the ratified band, in milli-nats
OUTSIDE_TOPK_MNATS = 99_999_000

# Defaults are the dgx.casa harness laid down by this row. They are ARGUMENTS so
# that the committed file and the file that executed are byte-identical: a script
# edited to be commitable after the fact is not the instrument that reported.
ROOT = "/home/mudler/mtpgate"
# THE LOCAL SNAPSHOT, not the repo id: `unsloth/Qwen3.6-27B-NVFP4` HEAD has
# MOVED upstream, so a probe that re-resolves the NAME is not reading the bytes
# the arms read. Mirrors oracle_mtp.py.
TARGET = ("/home/mudler/.cache/huggingface/hub/"
          "models--unsloth--Qwen3.6-27B-NVFP4/snapshots/"
          "890bdef7a42feba6d83b6e17a03315c694112f2a")
PIN_COMMIT = "555967922"
PIN_FLASHINFER = "0.6.15.post1"
PIN_TRANSFORMERS = "5.14.1"


def assert_oracle_identity():
    """A venv resolving to a different tree is a deterministic WRONG oracle."""
    import flashinfer
    import transformers
    import vllm
    problems = []
    if PIN_COMMIT not in vllm.__version__:
        problems.append(f"vllm {vllm.__version__} does not carry pin {PIN_COMMIT}")
    if flashinfer.__version__ != PIN_FLASHINFER:
        problems.append(f"flashinfer {flashinfer.__version__} != {PIN_FLASHINFER}")
    if transformers.__version__ != PIN_TRANSFORMERS:
        problems.append(f"transformers {transformers.__version__} != {PIN_TRANSFORMERS}")
    if problems:
        raise SystemExit("ORACLE IDENTITY MISMATCH, aborting:\n  " + "\n  ".join(problems))
    print(f"oracle identity OK: vllm={vllm.__version__} "
          f"flashinfer={flashinfer.__version__} transformers={transformers.__version__} "
          f"file={vllm.__file__}", flush=True)


def load_tokens(runs, arm):
    p = os.path.join(runs, arm, "output_token_ids.json")
    return json.load(open(p)) if os.path.exists(p) else None


def parse_prompt_ids(runs, arm):
    """Our engine's own prompt tokenization, as the arm recorded it."""
    import re
    log = os.path.join(runs, arm, "run.log")
    if not os.path.exists(log):
        return {}
    txt = open(log, errors="replace").read()
    return {int(m.group(1)): [int(x) for x in m.group(2).split(",")]
            for m in re.finditer(r"^PROMPT_IDS (\d+) ([0-9,]+)$", txt, re.M)}


def divergences(off, arm_tokens):
    """[(prompt_index, position, off_token, arm_token)] for every mismatch."""
    out = []
    for i, (x, y) in enumerate(zip(off, arm_tokens)):
        for j, (p, q) in enumerate(zip(x, y)):
            if p != q:
                out.append((i, j, p, q))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=TARGET)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--gpu-mem-util", type=float, default=0.75)
    args = ap.parse_args()
    args.runs = args.runs or f"{args.root}/final"
    args.out = args.out or f"{args.root}/final/adjudication.json"

    assert_oracle_identity()
    from vllm import LLM, SamplingParams

    meta = json.load(open(f"{args.root}/workload_meta.json"))
    prompts = [m["text"] for m in meta]

    off = load_tokens(args.runs, "ours_off")
    if off is None:
        raise SystemExit(f"ABORT: no ours_off tokens under {args.runs}")

    arms = [f"ours_on_k{k}" for k in (2, 3, 4)] + [f"padded_k{k}" for k in (2, 3, 4)]
    arm_tokens = {a: load_tokens(args.runs, a) for a in arms}
    present = {a: t for a, t in arm_tokens.items() if t is not None}
    print(f"arms loaded: {len(present)}/{len(arms)} "
          f"({', '.join(sorted(present))})", flush=True)
    if not present:
        raise SystemExit("ABORT: no ON arms to adjudicate")

    # ── which positions diverge, and which of them are adjudicable ───────────
    all_div = {a: divergences(off, t) for a, t in present.items()}
    n_div = sum(len(v) for v in all_div.values())
    # First divergence per (arm, prompt). Only the earliest per prompt across all
    # arms shares a prefix with OFF, and each arm's own first per prompt does too.
    first = {}       # (arm, prompt) -> (pos, off_tok, arm_tok)
    for a, v in all_div.items():
        for (i, j, p, q) in v:
            if (a, i) not in first:
                first[(a, i)] = (j, p, q)
    n_adjudicable = len(first)
    print(f"divergent positions found: {n_div}, adjudicable (first per arm+prompt): "
          f"{n_adjudicable}, excluded as post-divergence context: "
          f"{n_div - n_adjudicable}", flush=True)

    # Collapse to the distinct (prompt, position, candidate-set) probes. Every
    # arm that first splits at the same position of the same prompt shares one
    # prefix, so one forward answers all of them.
    probes = {}      # (prompt, pos) -> {"off": tok, "cands": {tok: [arms]}}
    for (a, i), (j, p, q) in sorted(first.items()):
        key = (i, j)
        e = probes.setdefault(key, {"off": p, "cands": {}})
        if e["off"] != p:
            raise SystemExit(f"ABORT: inconsistent OFF token at prompt {i} pos {j}")
        e["cands"].setdefault(q, []).append(a)
    print(f"distinct probe points: {len(probes)}", flush=True)

    # ── the oracle, in the SAME configuration the gate's OFF arm uses ────────
    # max_logprobs defaults to 20 and vLLM REFUSES a request asking for more, so
    # the engine is built with headroom rather than sitting on the boundary. A
    # probe that aborts on its own limit would present as a verdict about the
    # code (see [[broken-instruments-fail-toward-a-code-verdict]]).
    llm = LLM(model=args.model, max_model_len=2048, max_num_seqs=4,
              gpu_memory_utilization=args.gpu_mem_util,
              max_logprobs=max(args.topk, 20) + 1,
              enable_prefix_caching=False, disable_log_stats=False)

    records, checked, defects, ties = [], 0, 0, 0
    for (i, j), e in sorted(probes.items()):
        # vLLM's own tokenization of this prompt string.
        pid = list(llm.generate([prompts[i]],
                                SamplingParams(temperature=0.0, max_tokens=1)
                                )[0].prompt_token_ids)
        our_pid = parse_prompt_ids(args.runs, "ours_off").get(i)
        pid_match = (our_pid == pid) if our_pid is not None else None

        # The SHARED PREFIX: prompt + the tokens both arms agree on. The
        # distribution at the next position is what both arms sampled from.
        prefix = pid + [int(x) for x in off[i][:j]]
        cand_ids = [e["off"]] + sorted(e["cands"])

        # One forced forward per candidate. prompt_logprobs carries the FORCED
        # token's own logprob even when it falls outside top-K, so a candidate
        # that is outside the band is still measured rather than guessed.
        dist, forced = None, {}
        for c in cand_ids:
            o = llm.generate({"prompt_token_ids": prefix + [int(c)]},
                             SamplingParams(temperature=0.0, max_tokens=1,
                                            prompt_logprobs=args.topk))[0]
            d = o.prompt_logprobs[len(prefix)] or {}
            forced[int(c)] = d.get(int(c))
            if dist is None or (not dist and d):
                dist = d

        # An empty distribution is an INSTRUMENT failure, not a tie and not a
        # defect. Say so instead of letting max() raise into an ambiguous
        # traceback that a reader could score as either.
        if not dist:
            raise SystemExit(f"ABORT: oracle returned no prompt_logprobs at "
                             f"prompt {i} position {j}. The probe is broken, "
                             f"nothing is adjudicated")

        arg = max(dist, key=lambda t: dist[t].logprob)
        arg_lp = dist[arg].logprob
        ranked = sorted(dist.items(), key=lambda kv: -kv[1].logprob)
        top5 = [{"token": int(t), "logprob": float(l.logprob),
                 "text": getattr(l, "decoded_token", None)} for t, l in ranked[:5]]
        rank_of = {int(t): r for r, (t, _) in enumerate(ranked)}

        cands = []
        pos_defect = False
        for c in cand_ids:
            lp = forced.get(int(c))
            inside = int(c) in rank_of and rank_of[int(c)] < args.topk
            if lp is None:
                gap_mnats, verdict = OUTSIDE_TOPK_MNATS, "NO_LOGPROB"
                pos_defect = True
            else:
                gap = max(0.0, arg_lp - lp.logprob)
                gap_mnats = int(round(gap * 1000.0))
                if not inside:
                    verdict = f"OUTSIDE_TOP{args.topk}"
                    pos_defect = True
                elif gap_mnats > NEAR_TIE_MNATS:
                    verdict = "OUTSIDE_BAND"
                    pos_defect = True
                else:
                    verdict = "IN_BAND"
            cands.append({
                "token": int(c),
                "text": getattr(lp, "decoded_token", None) if lp else None,
                "role": "OFF" if c == e["off"] else "ON",
                "arms": ["ours_off"] if c == e["off"] else sorted(e["cands"][c]),
                "logprob": float(lp.logprob) if lp else None,
                "rank": rank_of.get(int(c)),
                "gap_mnats": gap_mnats,
                "verdict": verdict,
            })
        checked += 1
        if pos_defect:
            defects += 1
        else:
            ties += 1
        records.append({
            "prompt": i, "kind": meta[i]["kind"], "position": j,
            "prompt_token_ids_match_ours": pid_match,
            "our_prompt_token_ids": our_pid, "vllm_prompt_token_ids": pid,
            "argmax": {"token": int(arg), "logprob": float(arg_lp),
                       "text": getattr(dist[arg], "decoded_token", None)},
            "top5": top5, "candidates": cands,
            "position_verdict": "DEFECT" if pos_defect else "TIE",
        })

    # ── report ──────────────────────────────────────────────────────────────
    print("=" * 78)
    for r in records:
        print(f"prompt {r['prompt']} ({r['kind']}) position {r['position']}: "
              f"{r['position_verdict']}")
        print(f"        prompt tokenization ours==vLLM: {r['prompt_token_ids_match_ours']}")
        print(f"        oracle argmax {r['argmax']['token']} "
              f"({r['argmax']['text']!r}) logprob {r['argmax']['logprob']:.6f}")
        for c in r["candidates"]:
            print(f"        {c['role']:3s} token {c['token']:>6} ({c['text']!r}) "
                  f"rank {c['rank']} gap {c['gap_mnats']} mnats -> {c['verdict']}"
                  f"   arms={','.join(c['arms'])}")
        print("        top5 " + ", ".join(
            f"{t['token']}({t['text']!r}) {t['logprob']:.4f}" for t in r["top5"]))
    print("=" * 78)
    all_pid_ok = all(r["prompt_token_ids_match_ours"] is True for r in records)
    print(f"PROMPT_TOKENIZATION_IDENTICAL={all_pid_ok} "
          f"({len(records)} probe points checked)")
    print(f"POSITIONS_EXAMINED={n_div} POSITIONS_ADJUDICABLE={n_adjudicable} "
          f"PROBE_POINTS={checked} TIES={ties} DEFECTS={defects}")
    print(f"ADJUDICATION={'NEAR_TIE' if defects == 0 and checked > 0 else 'DEFECT'}"
          f"  band={NEAR_TIE_MNATS} mnats topk={args.topk}")

    with open(args.out, "w") as fh:
        json.dump({"band_mnats": NEAR_TIE_MNATS, "topk": args.topk,
                   "target": args.model, "runs": args.runs,
                   "positions_examined": n_div,
                   "positions_adjudicable": n_adjudicable,
                   "probe_points": checked, "ties": ties, "defects": defects,
                   "prompt_tokenization_identical": all_pid_ok,
                   "records": records}, fh, indent=2)
    print(f"WROTE {args.out}", flush=True)
    return 1 if defects else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""#2809 -- score the ratified near-tie band of `multimodal-speed.md` §12.2 on the
Qwen3.8-27B Q4_K_M ROCm arm with the PRIMARY oracle, vLLM 5559679229 on gfx1151.

The four conjuncts and the definition of every one of them are taken from
`scripts/mm/a3_voxtral_neartie_gate.py`, which is the harness §12.2 names. In
particular a step is `n_divergent` when the oracle's teacher-forced top logprob
exceeds ours by more than 1e-9, so an EXACT tie is not divergent -- that is
exactly why the Voxtral precedent passed at `worst_gap 0.0000`.

CORRECTNESS ONLY. No timing, throughput, latency or memory figure is taken from
anything here, and no cross-engine ratio is computed. This arm's declared token
gate reads FAIL and AGENTS.md §Gates admits no performance result from it.
#2497 already carries one retraction for exactly that.

NO HSA_OVERRIDE_GFX_VERSION anywhere: that knob makes the runtime report a
different device and would void an oracle measurement.
"""
import hashlib
import json
import os

OUT = os.environ["OUT_JSON"]
MODEL = os.environ["MODEL"]
TOK = os.environ["TOK"]
MMPROJ = os.environ.get("MMPROJ") or None
QUANT = os.environ.get("QUANT") or None
GMU = float(os.environ.get("GMU", "0.60"))
MAXLEN = int(os.environ.get("MAXLEN", "2048"))
EAGER = os.environ.get("EAGER", "1") == "1"
K = int(os.environ.get("TOPK", "20"))
OURS_JSON = os.environ["OURS_JSON"]
SELF_JSON = os.environ["SELF_JSON"]          # this config's OWN recorded greedy tokens
LLAMACPP_TXT = os.environ["LLAMACPP_TXT"]    # oracle_hip.txt, the b10451 stream

BAND_NATS = 0.5
ARGMAX_EPS = 1e-9

# The SIX prompts of the declared Q4_K_M token gate, byte for byte. Their file
# form (one per line, newline-terminated) hashes to
# c8a5080046c3206a1186b42a320a21e887691eff43d2116364192f58e5776c7e.
PROMPTS = [
    "The capital city of France is",
    "The three primary colors are",
    "Water boils at a temperature of",
    "The Pythagorean theorem states that",
    "In 1969, humans first walked on",
    "A prime number is a natural number",
]
PROMPTS_SHA256 = "c8a5080046c3206a1186b42a320a21e887691eff43d2116364192f58e5776c7e"

# The llama.cpp b10451 oracle's own PROMPT_IDS, which #2788 fed to vLLM rather
# than re-tokenizing, so a tokenizer difference cannot contaminate the score.
ORACLE_PROMPT_IDS = [
    [760, 6511, 3177, 314, 9338, 369],
    [760, 2250, 5839, 7736, 513],
    [27336, 85895, 506, 264, 9039, 314],
    [760, 5187, 92068, 43687, 55877, 5134, 421],
    [623, 220, 16, 24, 21, 24, 11, 12313, 1118, 14428, 383],
    [32, 9944, 1324, 369, 264, 5629, 1324],
]


def read_llamacpp(path: str) -> list[list[int]]:
    gen: dict[int, list[int]] = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("GEN_IDS "):
                f = line.split()
                gen[int(f[1])] = [int(x) for x in f[2:]]
    return [gen[i] for i in sorted(gen)]


def read_records(path: str) -> list[list[int]]:
    d = json.loads(open(path).read())
    return [r["gen_ids"] for r in d["records"]]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") is None, "override set"
    leaked = sorted(k for k in os.environ
                    if k.startswith(("HSA_", "ROCR_", "PYTORCH_", "HIP_")))
    print("INHERITED_DEVICE_ENV =", leaked or "NONE", flush=True)

    body = "".join(p + "\n" for p in PROMPTS).encode()
    got = hashlib.sha256(body).hexdigest()
    print("PROMPTS_SHA256 =", got, flush=True)
    assert got == PROMPTS_SHA256, got

    import torch
    import vllm

    print("VLLM_VERSION =", vllm.__version__, vllm.__file__, flush=True)
    print("TORCH        =", torch.__version__, "hip", torch.version.hip, flush=True)
    print("DEVICE       =", torch.cuda.get_device_properties(0).gcnArchName, flush=True)
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1151")
    from vllm.platforms.rocm import on_gfx1151, on_gfx1x

    print("ON_GFX1151   =", on_gfx1151(), " ON_GFX1X =", on_gfx1x(), flush=True)
    if QUANT == "gguf":
        import vllm_gguf_plugin
        import vllm_gguf_plugin._C_gguf as _cg

        print("PLUGIN_C_EXT =", _cg.__file__, flush=True)

    ours = json.loads(open(OURS_JSON).read())
    print("OURS_JSON sha256 =", sha256_file(OURS_JSON), flush=True)
    selfs = read_records(SELF_JSON)
    print("SELF_JSON sha256 =", sha256_file(SELF_JSON), flush=True)
    lcpp = read_llamacpp(LLAMACPP_TXT)
    print("LLAMACPP_TXT sha256 =", sha256_file(LLAMACPP_TXT), flush=True)
    for nm, X in (("ours", ours), ("self", selfs), ("llamacpp", lcpp)):
        assert len(X) == 6 and all(len(x) == 48 for x in X), (nm, [len(x) for x in X])
        print(f"STREAM {nm:9s} lengths {[len(x) for x in X]}", flush=True)

    # NEGATIVE CONTROL: prompt 0's own recorded stream with step 10 replaced by a
    # token the oracle cannot prefer. If this does not register a divergence with
    # a large gap, the instrument is not discriminating and no PASS below means
    # anything.
    ctrl = [list(selfs[0])]
    ctrl[0][10] = 9999

    from vllm import LLM, SamplingParams

    kw = dict(model=MODEL, tokenizer=TOK, gpu_memory_utilization=GMU,
              max_model_len=MAXLEN, max_num_seqs=1, max_num_batched_tokens=MAXLEN,
              limit_mm_per_prompt={"image": 0, "video": 0}, enforce_eager=EAGER,
              trust_remote_code=False, enable_prefix_caching=False)
    if QUANT:
        kw["quantization"] = QUANT
    if MMPROJ:
        kw["model_loader_extra_config"] = {"mm_proj": MMPROJ}
    print("LLM_KWARGS   =", kw, flush=True)
    llm = LLM(**kw)
    print("ENGINE_UP", flush=True)

    # Every teacher-forced request in ONE batch. max_tokens=1 because the
    # measurement is prompt_logprobs over prompt+tokens, not a generation.
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1,
                        prompt_logprobs=K, ignore_eos=True)
    jobs = []           # (stream, prompt_index, base, tokens)
    for nm, X in (("ours", ours), ("self", selfs), ("llamacpp", lcpp),
                  ("control_corrupt", ctrl)):
        for i, toks in enumerate(X):
            jobs.append((nm, i, len(ORACLE_PROMPT_IDS[i]),
                         list(ORACLE_PROMPT_IDS[i]) + list(toks), list(toks)))

    from vllm.inputs import TokensPrompt

    outs = llm.generate([TokensPrompt(prompt_token_ids=j[3]) for j in jobs], sp)
    print("TEACHER_FORCE_DONE", len(outs), flush=True)

    res: dict = {}
    for (nm, i, base, full, toks), o in zip(jobs, outs):
        plp = o.prompt_logprobs
        assert plp is not None and len(plp) == len(full), (nm, i, len(full))
        steps = []
        for s, tok in enumerate(toks):
            d = plp[base + s]
            assert d is not None, (nm, i, s)
            assert tok in d, f"{nm} p{i} step {s}: our token {tok} absent from the returned dict"
            our_lp = d[tok].logprob
            top_tok, top_e = max(d.items(), key=lambda kv: kv[1].logprob)
            top_lp = top_e.logprob
            gap = top_lp - our_lp
            steps.append({"step": s, "our_tok": tok, "our_lp": our_lp,
                          "top_tok": int(top_tok), "top_lp": top_lp, "gap": gap,
                          "our_rank": d[tok].rank,
                          "argmax": gap <= ARGMAX_EPS,
                          "tok_match": int(top_tok) == int(tok)})
        res.setdefault(nm, []).append({"prompt": i, "steps": steps})

    summary = {}
    for nm, prompts in res.items():
        n_div = 0
        n_tok_mismatch = 0
        n_exact_tie = 0
        worst = 0.0
        worst_at = None
        fails = []
        div_rows = []
        for pr in prompts:
            for st in pr["steps"]:
                if not st["tok_match"]:
                    n_tok_mismatch += 1
                    if st["argmax"]:
                        n_exact_tie += 1
                if not st["argmax"]:
                    n_div += 1
                    div_rows.append({"prompt": pr["prompt"], **st})
                    if st["gap"] > worst:
                        worst, worst_at = st["gap"], (pr["prompt"], st["step"])
                    if st["gap"] > BAND_NATS:
                        fails.append([pr["prompt"], st["step"], st["our_tok"], st["gap"]])
        result = "PASS" if not fails else "FAIL"
        conj = (result == "PASS" and n_div == 0 and len(fails) == 0
                and worst <= BAND_NATS)
        summary[nm] = {"n_divergent": n_div, "worst_gap_nats": worst,
                       "worst_at": worst_at, "over_band_failures": fails,
                       "result": result, "four_conjuncts": "PASS" if conj else "FAIL",
                       "n_token_mismatch": n_tok_mismatch,
                       "n_exact_tie_mismatch": n_exact_tie,
                       "n_strictly_positive_gap": n_div,
                       "divergent_steps": div_rows,
                       "total_steps": sum(len(p["steps"]) for p in prompts)}

    cfg = "eager" if EAGER else "compiled"
    for nm, s in summary.items():
        print(f"\n=== §12.2 NEAR-TIE BAND | oracle vLLM {cfg} | stream {nm} ===", flush=True)
        for r in s["divergent_steps"]:
            print(f"  p{r['prompt']}/{r['step']:2d} our {r['our_tok']:6d} lp {r['our_lp']:9.6f}"
                  f"  top {r['top_tok']:6d} lp {r['top_lp']:9.6f}  gap {r['gap']:9.6f}"
                  f"  rank {r['our_rank']}", flush=True)
        print(f"  result               = {s['result']}", flush=True)
        print(f"  n_divergent          = {s['n_divergent']} of {s['total_steps']}", flush=True)
        print(f"  over_band_failures   = {len(s['over_band_failures'])}", flush=True)
        print(f"  worst_gap_nats       = {s['worst_gap_nats']:.6f} @ {s['worst_at']}", flush=True)
        print(f"  token mismatches     = {s['n_token_mismatch']}"
              f" (exact ties {s['n_exact_tie_mismatch']},"
              f" strictly positive {s['n_strictly_positive_gap']})", flush=True)
        print(f"  FOUR CONJUNCTS       = {s['four_conjuncts']}", flush=True)

    c = summary["control_corrupt"]
    ok = c["n_divergent"] >= 1 and c["worst_gap_nats"] > 1.0
    print(f"\nINSTRUMENT_CONTROL corrupt-step discriminates = {'PASS' if ok else 'FAIL'}"
          f" (n_divergent {c['n_divergent']}, worst {c['worst_gap_nats']:.6f})", flush=True)

    with open(OUT, "w") as fh:
        json.dump({"issue": 2809, "oracle": "vllm", "config": cfg,
                   "vllm_version": vllm.__version__, "model": MODEL,
                   "quantization": QUANT, "enforce_eager": EAGER, "topk": K,
                   "prompts_sha256": PROMPTS_SHA256,
                   "band_nats": BAND_NATS, "argmax_eps": ARGMAX_EPS,
                   "summary": summary, "per_step": res}, fh, indent=1)
    print("WROTE", OUT, flush=True)
    assert ok, "the negative control did not discriminate; no verdict above is usable"
    print("DONE_MARKER_Q4KM_NEARTIE_VLLM", flush=True)


if __name__ == "__main__":
    main()

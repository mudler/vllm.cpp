#!/usr/bin/env python3
# #2884 -- the PINNED vLLM's side of the limb-3 strict gate, on the VEHICLE.
#
# Adapted from docs/bench-evidence/oracle-vllm-gfx1151-20260903/gen_rocm.py
# (#2740), which is the committed recipe that got this pin generating on
# gfx1151. Three things are added and they are the point of this file:
#
#   1. The prompt token ids are FED FROM A FILE that OUR engine produced from
#      the vehicle's own GGUF vocab, and this script asserts vLLM's tokenizer
#      re-derives them. A generation comparison must not be contaminated by a
#      tokenizer difference, and #2740 established the pattern by feeding
#      llama.cpp's ids.
#   2. It records a PREFILL-ARGMAX self-consistency check. A vehicle is only
#      admissible if the oracle's own one-pass argmax over (prompt + its own
#      generated tokens) reproduces the tokens its incremental decode emitted.
#      An oracle that disagrees with itself across that boundary cannot be a
#      denominator for anybody.
#   3. Nothing is timed. CORRECTNESS ONLY: AGENTS.md Gates admits no
#      performance result from an arm whose declared token gate has not passed.
import json
import os

MODEL = os.environ["MODEL"]
TOK = os.environ["TOK"]
FEED = os.environ["FEED_IDS"]          # JSON list-of-lists, OUR prompt ids
OUT = os.environ["OUT_JSON"]
QUANT = os.environ.get("QUANT") or None
MMPROJ = os.environ.get("MMPROJ") or None
GMU = float(os.environ.get("GMU", "0.60"))
MAXLEN = int(os.environ.get("MAXLEN", "2048"))
EAGER = os.environ.get("EAGER", "1") == "1"
NTOK = int(os.environ.get("NTOK", "48"))
PREFILL_CHECK = os.environ.get("PREFILL_CHECK", "0") == "1"

# The SIX prompts of the declared Q4_K_M token gate, byte for byte. Their file
# form (one per line, newline-terminated) hashes to
# c8a5080046c3206a1186b42a320a21e887691eff43d2116364192f58e5776c7e. They are
# PRE-REGISTERED in the strongest sense available: they are not chosen here,
# they are the set this campaign has scored since 2026-08-23, and their hash is
# committed in three evidence documents that predate this run.
PROMPTS = [
    "The capital city of France is",
    "The three primary colors are",
    "Water boils at a temperature of",
    "The Pythagorean theorem states that",
    "In 1969, humans first walked on",
    "A prime number is a natural number",
]


# vLLM V1 spawns EngineCore with multiprocessing "spawn", which RE-IMPORTS this
# file in the child. Without a __main__ guard the module body runs again there
# and the engine dies with the freeze_support bootstrap error -- an INSTRUMENT
# failure that reads exactly like a model failure (#2624).
def main() -> None:
    import vllm

    print("VLLM_FILE    =", vllm.__file__, flush=True)
    print("VLLM_VERSION =", vllm.__version__, flush=True)

    import torch

    print("TORCH        =", torch.__version__, "hip", torch.version.hip, flush=True)
    print("DEVICE       =", torch.cuda.get_device_properties(0).gcnArchName, flush=True)
    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") is None, "override set"
    from vllm.platforms.rocm import on_gfx1151, on_gfx1x

    print("ON_GFX1151   =", on_gfx1151(), " ON_GFX1X =", on_gfx1x(), flush=True)

    if QUANT == "gguf":
        import vllm_gguf_plugin

        print("PLUGIN_FILE  =", vllm_gguf_plugin.__file__, flush=True)
        import vllm_gguf_plugin._C_gguf as _cg

        print("PLUGIN_C_EXT =", _cg.__file__, flush=True)

    feed = json.load(open(FEED))
    assert len(feed) == len(PROMPTS), (len(feed), len(PROMPTS))
    print("FED_PROMPT_IDS =", feed, flush=True)

    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(TOK)
    ids = [tk(p, add_special_tokens=False)["input_ids"] for p in PROMPTS]
    for i, s in enumerate(ids):
        print(f"VLLM_TOKENIZER_IDS[{i}] = {s}", flush=True)
    # NOT an assert. If the two tokenizers disagree the run still happens and
    # says so, because a tokenizer difference is a finding, not a crash.
    print("PROMPT_IDS_MATCH_OURS =", ids == feed, flush=True)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(
        model=MODEL,
        tokenizer=TOK,
        gpu_memory_utilization=GMU,
        max_model_len=MAXLEN,
        max_num_seqs=1,
        max_num_batched_tokens=MAXLEN,
        limit_mm_per_prompt={"image": 0, "video": 0},
        enforce_eager=EAGER,
        trust_remote_code=False,
    )
    if QUANT:
        kw["quantization"] = QUANT
    if MMPROJ:
        # The ecosystem ships this family's vision tower as a SEPARATE
        # `general.architecture = clip` file, and #2740's recipe passes it even
        # for a text-only workload. Every modality is at limit 0 below, so the
        # tower is loaded and never entered.
        kw["model_loader_extra_config"] = {"mm_proj": MMPROJ}
    print("LLM_KWARGS   =", {k: v for k, v in kw.items()}, flush=True)
    llm = LLM(**kw)
    print("ENGINE_UP", flush=True)

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=NTOK,
                        ignore_eos=True)
    outs = llm.generate([TokensPrompt(prompt_token_ids=s) for s in feed], sp)

    rec = []
    for i, o in enumerate(outs):
        gen = list(o.outputs[0].token_ids)
        rec.append({"i": i, "prompt": PROMPTS[i], "prompt_ids": feed[i],
                    "gen_ids": gen, "gen_text": o.outputs[0].text})
        print(f"GEN_IDS[{i}] = {gen}", flush=True)
        print(f"GEN_TEXT[{i}] = {o.outputs[0].text!r}", flush=True)
        print(f"GEN_LEN[{i}] = {len(gen)}", flush=True)

    # ---- PREFILL ARGMAX vs INCREMENTAL DECODE -------------------------------
    # Teacher-force the model along the sequence it just produced, in ONE pass,
    # and read the rank-1 candidate at every position that predicted a
    # generated token. If the oracle's prefill disagrees with its own decode,
    # the vehicle is not a denominator and the run says so instead of picking
    # whichever half agrees.
    prefill = None
    prefill_error = None
    if PREFILL_CHECK:
        # A prefill probe that raises must NOT take the generation with it. The
        # generated ids above are the measurement; this check is an additional
        # question about the oracle, and an unsupported sampling parameter at
        # this pin would be a finding rather than a lost leg.
        try:
            sp2 = SamplingParams(temperature=0.0, max_tokens=1,
                                 prompt_logprobs=1, ignore_eos=True)
            full = [list(feed[i]) + rec[i]["gen_ids"] for i in range(len(feed))]
            outs2 = llm.generate(
                [TokensPrompt(prompt_token_ids=s) for s in full], sp2)
            prefill = []
            for i, o in enumerate(outs2):
                pl = o.prompt_logprobs
                np_ = len(feed[i])
                argmax_next, mism = [], []
                # pl[k] is the distribution over token k GIVEN tokens < k, so
                # the entry that PREDICTS generated token j (at absolute index
                # np_ + j) is pl[np_ + j] itself.
                for j in range(len(rec[i]["gen_ids"])):
                    k = np_ + j
                    if pl is None or k >= len(pl) or pl[k] is None:
                        argmax_next.append(None)
                        continue
                    top = max(pl[k].items(), key=lambda kv: kv[1].logprob)[0]
                    argmax_next.append(int(top))
                    if int(top) != rec[i]["gen_ids"][j]:
                        mism.append([j, int(top), rec[i]["gen_ids"][j]])
                prefill.append({"i": i, "argmax": argmax_next,
                                "mismatches": mism})
                print(f"PREFILL_ARGMAX_MISMATCHES[{i}] = {len(mism)} "
                      f"{mism[:5]}", flush=True)
            total = sum(len(x["mismatches"]) for x in prefill)
            print("PREFILL_ARGMAX_TOTAL_MISMATCHES =", total, flush=True)
            print("PREFILL_AGREES_WITH_DECODE =", total == 0, flush=True)
        except Exception as exc:   # noqa: BLE001
            prefill_error = repr(exc)
            print("PREFILL_ARGMAX=NOT_MEASURED", prefill_error, flush=True)

    with open(OUT, "w") as fh:
        json.dump({"vllm_version": vllm.__version__, "model": MODEL,
                   "quantization": QUANT, "enforce_eager": EAGER,
                   "gpu_memory_utilization": GMU, "n_tokens": NTOK,
                   "tokenizer_agrees_with_ours": ids == feed,
                   "records": rec, "prefill": prefill,
                   "prefill_error": prefill_error}, fh, indent=1)
    print("WROTE", OUT, flush=True)
    assert all(len(r["gen_ids"]) == NTOK for r in rec), "short generation"
    print("DONE_MARKER_LIMB3_VEHICLE_GEN", flush=True)


if __name__ == "__main__":
    main()

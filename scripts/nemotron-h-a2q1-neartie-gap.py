#!/usr/bin/env python3
# A2-Q1 / #1388 -- the oracle's OWN top-2 margin at the position the device
# mamba arm loses, and the per-position margin profile of the whole golden.
#
# ★ THIS HAS NEVER PRODUCED A MEASUREMENT, and the reason is not in this file.
# Five runs on `dgx:gpu0` (2026-08-19/20, logs under /workspace/a2q1-neartie/)
# were each killed by the host-memory watchdog during ENGINE START-UP, before
# any token was generated. That is #1431, and it also makes the 2026-08-18
# result behind "the oracle DOES run a model in a lease" non-reproducing. Read
# #1431 before spending a lease on this script: the block is the host, and no
# engine knob tried there avoided it.
#
# A sibling of scripts/*-neartie-gap.py, not an edit of one: those teacher-force
# vLLM on OUR generated ids for a 16-prompt corpus and emit .npy gap files a
# gate reads. This one answers a single triage question about ONE recorded
# divergence and prints it, so it is one file per row, read with a glob
# (AGENTS.md "Records"), rather than another writer of a shared surface.
#
# RUN IT INSIDE A LEASE, NEVER OVER ssh. The WHEEL is the one measured on that
# box. The LOAD CONFIGURATION is not: run 4 of the five ran the configuration
# that survived on 2026-08-18 and died anyway, so it is a starting point under
# investigation, not a known-good recipe. For the wheel see
# /workspace/oracle-vllm/README-WHEELS.md
# (vLLM's default FLASH_ATTN carries no sm_12x SASS and cannot run on a GB10;
# and the wheel needs a PEP 427 name before pip will take it, #1416):
#
#   rc run -d dgx:gpu0 --max-runtime 3h -- bash /workspace/a2q1-neartie/job.sh
#
# Environment: MODEL (checkpoint dir), GOLDEN (oracle.json), GMU, KVBYTES.
# A2-Q1 / #1388 STEP ONE -- the cheapest question, asked before any numerics work.
#
# The device mamba arm loses exactly ONE token: prompt 2, generation position 32
# (the LAST position on the LONGEST prompt), got 11286 where the golden says
# 3468. Positions 1..31 are byte-identical, so the two arms reach position 32
# with the SAME prefix. The question is therefore entirely about the ORACLE:
# how confidently does vLLM itself separate its argmax from the runner-up there?
#
# This TEACHER-FORCES the pinned oracle on the golden prefix and reads its own
# top-K. A margin at the bf16 noise floor makes this a tie-break, not a defect
# ([[near-tie-distributional-gate]]); a large margin makes it a real divergence.
import json, os, sys, time

OURS = {2: {32: 11286}}      # arm token, 1-based generation position, per prompt

def topk(d, k=8):
    return sorted(d.items(), key=lambda kv: -kv[1].logprob)[:k]

def main():
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    import vllm
    print("VLLM_FILE", vllm.__file__, flush=True)
    print("VLLM_VERSION", vllm.__version__, flush=True)
    assert "555967922" in vllm.__version__, f"WRONG COMMIT {vllm.__version__}"
    assert "site-packages" in vllm.__file__, f"NOT AN INSTALLED WHEEL {vllm.__file__}"

    golden = json.load(open(os.environ["GOLDEN"]))
    print("GOLDEN_MODEL", golden["model"], "rev", golden["revision"], flush=True)
    print("GOLDEN_VLLM", golden["vllm"], flush=True)
    assert golden["vllm"].endswith("g555967922"), golden["vllm"]
    entries = golden["golden"]
    T = int(golden["sampling"]["max_tokens"])

    # ★ THE CONFIGURATION BELOW IS NOT KNOWN TO SURVIVE. It is the one that used
    # to survive, plus one variable. Both halves of that sentence were measured.
    #
    # FIVE runs on `dgx:gpu0`, 2026-08-19/20, logs under /workspace/a2q1-neartie/.
    # Every one was killed by the host-memory watchdog inside engine START-UP,
    # before a token existed (#1431). The watchdog FLOOR was not constant across
    # them, so the kill values are not comparable as a series:
    #
    #   run  stamp             deviation from the reference config    floor   killed at
    #   1    20260820T002359Z  + max_logprobs=64                      15000   12597 MB
    #   2    20260820T024243Z  + enforce_eager=True, len 256, seqs 1  20000   19433 MB
    #   3    20260820T025540Z  + num_gpu_blocks_override=8            20000   19797 MB
    #   4    20260820T030818Z  NONE -- the reference config itself    15000   13941 MB
    #   5    20260820T031644Z  + kv_cache_memory_bytes=4 GiB          15000   14846 MB
    #
    # RUN 4 IS THE ONE THAT MATTERS, and it falsifies the premise this block used
    # to state. /workspace/nhspeed/oracle_only.sh attempt `a` loaded this same
    # 20.1 GiB checkpoint on this same box on 2026-08-18, bottomed out at
    # minMemAvailable_MB=51528, and generated all three prompts
    # (`ORACLE TOKEN MATCH: 180/192`). Run 4 is that configuration, unchanged, and
    # it died. So the reference is NOT a safe base to copy; it is a REGRESSION to
    # be explained (#1431). The environment is not byte-identical either -- run 4
    # built its venv at /tmp/a2q1-oracle with a fresh `pip install torch==2.13.0`
    # and a reinstalled wheel, where 08-18 used /tmp/nhspeed-oracle -- and an
    # identity assert on the vLLM COMMIT cannot see a torch or flashinfer delta.
    # That delta is an UNEXCLUDED candidate for the regression.
    #
    # Runs 2 and 3 were killed ~5 GB earlier in the drawdown than the others. That
    # they would ALSO have crossed 15000 MB is an INFERENCE, not a measurement:
    # the host trace was still falling at ~1.18 GB/s with no arrest when the kill
    # landed. Do not quote runs 2 and 3 as 15000 MB results.
    #
    # WHAT THE FIVE RUNS EXCLUDE. Host memory sits flat through the whole weight
    # load, then falls ~76 GB in ~60 s at ~1.2 GB/s just after the mamba page-size
    # print. Not `torch.compile`: run 5 hit the AOT cache (`torch.compile took
    # 0.30 s`). Not CUDA graph capture: run 2 ran `enforce_eager=True`. Not KV
    # SIZING: run 3 overrode the block count and run 5 set an absolute budget.
    # What is left is THE FIRST FORWARD. Under `kv_cache_memory_bytes` vLLM runs
    # `profile_run()` FIRST (v1/worker/gpu_worker.py:465-468, "still need a profile
    # run which compiles the model for max_num_batched_tokens") and only THEN logs
    # "Initial free memory ... skipped memory profiling" (:470-482). Run 5's log
    # reaches neither string, so the process died INSIDE that forward. This
    # supersedes #1185's "after compilation -- profiling forward or graph capture",
    # and it is recorded there rather than chased here.
    #
    # WHY THE ENGINE IS OTHERWISE UNTOUCHED. The question is asked with
    # REQUEST-level sampling params, which cost nothing at start-up.
    # `max_logprobs` stays at its engine default of 20 (config/model.py:228) and
    # the requests below ask for exactly 20 -- ample for a two-way tie, and one
    # more engine knob not turned.
    #
    # THE ONE VARIABLE. `kv_cache_memory_bytes` (config/cache.py:182, honoured at
    # v1/worker/gpu_worker.py:465-476) takes an ABSOLUTE KV budget, so it does not
    # depend on a utilization fraction -- which matters because `nvidia-smi`
    # reports `memory.total = [N/A]` on this GB10. 4 GiB against three sequences of
    # at most 45 tokens is an over-provision, not a tuned value: a block here is
    # roughly 120 MB (23 mamba layers of conv + f32 SSM state, plus an attention
    # page vLLM raises to 4176 tokens so it is >= the mamba page), so ~30 blocks
    # for a job that needs one per prompt. `gpu_memory_utilization` is left at the
    # reference 0.30 and is NOT the knob being turned.
    #
    # It did not help. Run 5 IS this configuration, and it died too. The knob is
    # kept because it removes KV sizing from the candidate list, not because it
    # works. READ #1431 BEFORE SPENDING A LEASE HERE.
    kw = dict(model=os.environ["MODEL"], max_model_len=512, max_num_seqs=8,
              gpu_memory_utilization=float(os.environ.get("GMU", "0.30")),
              max_num_batched_tokens=512, enforce_eager=False,
              kv_cache_memory_bytes=int(os.environ.get("KVBYTES", str(4 * 1024**3))))
    mode = "reference-config+kvbytes"
    print("MODE", mode, flush=True)
    print("ORACLE_KW", {k: v for k, v in kw.items() if k != "model"}, flush=True)
    llm = LLM(**kw)
    try:
        cc = llm.llm_engine.vllm_config.cache_config
        print("ORACLE_CACHE block_size=%s num_gpu_blocks=%s override=%s" %
              (cc.block_size, getattr(cc, "num_gpu_blocks", None),
               getattr(cc, "num_gpu_blocks_override", None)), flush=True)
    except Exception as e:
        print("ORACLE_CACHE_UNREADABLE:", e, flush=True)
    tok = llm.get_tokenizer()
    def show(t):
        try:
            return repr(tok.decode([int(t)]))
        except Exception:
            return "<?>"

    # ── LEG A: free-running greedy, the gate's own workload. Context only: it
    # says whether the oracle at THIS config reproduces its own committed
    # golden. It does NOT answer the margin question, because a diverging
    # oracle reaches position 32 on a different prefix.
    print(f"\n########## [{mode}] LEG A: free-running greedy vs the committed golden ##########", flush=True)
    spA = SamplingParams(temperature=0.0, max_tokens=T, ignore_eos=True)
    for i, e in enumerate(entries):
        out = llm.generate([TokensPrompt(prompt_token_ids=e["prompt_token_ids"])], spA)[0]
        got = list(out.outputs[0].token_ids)
        exp = e["token_ids"]
        n = min(len(got), len(exp))
        mism = [j for j in range(n) if got[j] != exp[j]]
        print(f"LEGA[{mode}] prompt {i}: compared={n} matched={n-len(mism)} "
              f"first_mismatch={(mism[0]+1) if mism else None} "
              f"mismatch_positions_1based={[j+1 for j in mism]}", flush=True)
        print(f"LEGA[{mode}] prompt {i} got: {','.join(str(x) for x in got)}", flush=True)

    # ── LEG B: TEACHER-FORCED on the golden prefix. This is the measurement.
    # One prefill over prompt + golden[0:T-1] gives, via prompt_logprobs, the
    # oracle's own top-K at generation positions 1..T-1, and the single sampled
    # step gives position T -- the position that diverges.
    print(f"\n########## [{mode}] LEG B: teacher-forced top-K on the GOLDEN prefix ##########", flush=True)
    K = 20   # == the engine default cap; raising it is an engine knob, see above
    spB = SamplingParams(temperature=0.0, max_tokens=1, logprobs=K, prompt_logprobs=K)
    for i, e in enumerate(entries):
        P = e["prompt_token_ids"]
        g = e["token_ids"]
        full = list(P) + list(g[:T-1])          # predict generation position T
        out = llm.generate([TokensPrompt(prompt_token_ids=full)], spB)[0]

        # positions 1..T-1 come from prompt_logprobs; index p in `full` holds
        # the distribution that PRODUCED full[p], so generation position j
        # (1-based) is index len(P)+j-1.
        plp = out.prompt_logprobs
        if plp is None:
            print(f"LEGB[{mode}] p{i}: prompt_logprobs came back None -- the INSTRUMENT "
                  f"did not measure positions 1..{T-1}; position {T} below still stands",
                  flush=True)
            plp = [None] * len(full)
        margins = []
        for j in range(1, T):
            d = plp[len(P) + j - 1] or {}
            if len(d) < 2:
                margins.append(None); continue
            tk = topk(d, 2)
            margins.append(tk[0][1].logprob - tk[1][1].logprob)
        fin = margins + [None]

        # position T: the sampled step's own logprob dict
        dT = out.outputs[0].logprobs[0]
        tkT = topk(dT, 8)
        argT = tkT[0][0]
        marginT = tkT[0][1].logprob - tkT[1][1].logprob if len(tkT) > 1 else float("inf")
        fin[T-1] = marginT

        print(f"\n--- prompt {i} ({len(P)} prompt tokens) ---", flush=True)
        print(f"LEGB[{mode}] p{i} teacher-forced argmax at generation position {T}: "
              f"{argT} {show(argT)}  golden={g[T-1]} {show(g[T-1])}  "
              f"match={argT == g[T-1]}", flush=True)
        print(f"LEGB[{mode}] p{i} TOP-2 MARGIN at position {T}: {marginT:.6f} nats", flush=True)
        print(f"LEGB[{mode}] p{i} top-8 at position {T}:", flush=True)
        for r, (t, lp) in enumerate(tkT):
            mark = ""
            if t == g[T-1]: mark += " <== GOLDEN/vLLM-greedy"
            if t == OURS.get(i, {}).get(T): mark += " <== OURS (device mamba arm)"
            print(f"    rank {r} (vllm_rank={lp.rank}): id={t:<7d} logprob={lp.logprob:+.6f} "
                  f"p={pow(2.718281828459045, lp.logprob):.6f} {show(t)}{mark}", flush=True)
        ours = OURS.get(i, {}).get(T)
        if ours is not None:
            if ours in dT:
                gap = tkT[0][1].logprob - dT[ours].logprob
                print(f"LEGB[{mode}] p{i} OUR TOKEN {ours} {show(ours)}: "
                      f"vllm_rank={dT[ours].rank} logprob={dT[ours].logprob:+.6f} "
                      f"gap_to_vllm_argmax={gap:.6f} nats", flush=True)
            else:
                print(f"LEGB[{mode}] p{i} OUR TOKEN {ours} OUTSIDE vLLM top-{K} -- REAL DIVERGENCE", flush=True)

        fmt = ", ".join("None" if m is None else f"{j+1}:{m:.4f}" for j, m in enumerate(fin))
        print(f"LEGB[{mode}] p{i} per-position top-2 margins (nats): {fmt}", flush=True)
        fin2 = [m for m in fin if m is not None]
        if fin2:
            order = sorted(range(len(fin)), key=lambda j: (fin[j] if fin[j] is not None else 1e9))
            print(f"LEGB[{mode}] p{i} TIGHTEST 5 positions: "
                  f"{[(j+1, round(fin[j], 5)) for j in order[:5]]}", flush=True)
            print(f"LEGB[{mode}] p{i} median margin={sorted(fin2)[len(fin2)//2]:.4f} "
                  f"min={min(fin2):.6f} max={max(fin2):.4f}", flush=True)
    print(f"\nDONE_MARKER_NEARTIE mode={mode}", flush=True)

if __name__ == "__main__":
    main()

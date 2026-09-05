#!/usr/bin/env python3
"""ONE client, two engines. The whole point of this file is that the timing
code is byte-identical on both sides, so the counting convention cannot be a
difference between the engines.

It speaks OpenAI /v1/chat/completions with stream=true against whatever host
and port it is given, so it does not know or care which engine answers. Each
engine applies its OWN chat template server-side, which is what both published
protocols do.

Reported per leg, both conventions, from the same timings:

  decode-only rate  = 1000 / mean(tpot),  tpot = (latency - ttft)/(n_out - 1)
  whole-run rate    = sum(completion_tokens) / wall_clock_of_the_whole_leg

The second is the one that includes prefill. Neither is privileged here; both
are printed for both engines.

ignore_eos is deliberately NOT sent. Both published protocols let the model
stop, and forcing a fixed length is one of the differences that made the
earlier numbers non-comparable.
"""
import argparse, json, statistics, sys, time, urllib.request, urllib.error


def stream_one(url, model, messages, max_tokens, temperature, top_p, top_k,
               seed, timeout):
    """One streaming chat completion. Returns a dict of raw timings.

    ttft is the wall time to the first chunk that carries actual CONTENT, not
    to the first chunk of any kind: a role-only opening delta would otherwise
    read as a token and shorten ttft on whichever engine happens to send one.
    """
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    ttft = None
    chunk_times = []        # arrival time of every content-carrying chunk
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            # BOTH fields count. Their server splits the model's <think> block
            # into `reasoning_content` and only the tail into `content`; ours
            # leaves the tags inline in `content`. Counting only `content`
            # would have measured their TTFT to the END of the reasoning block
            # and dropped every reasoning token from the rate -- an instrument
            # failure that reads as a result, and one that runs against them.
            piece = delta.get("content") or delta.get("reasoning_content")
            if not piece:
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            chunk_times.append(now)
            text.append(piece)
    latency = time.perf_counter() - t0

    return {
        "accepted_draft_tokens": (usage or {}).get("accepted_draft_tokens"),
        "ttft": ttft,
        "latency": latency,
        "n_chunks": len(chunk_times),
        "itls": [chunk_times[i] - chunk_times[i - 1]
                 for i in range(1, len(chunk_times))],
        "usage": usage,
        "text": "".join(text),
    }


def summarize(name, recs, wall):
    """Both conventions, computed once, from the same records."""
    ok = [r for r in recs if r.get("ok")]
    out = {"leg": name, "requests": len(recs), "ok": len(ok),
           "wall_s": wall}
    if not ok:
        out["error"] = "no successful request"
        return out

    # completion_tokens: prefer the server's own usage count. Fall back to the
    # streamed chunk count and SAY SO, because a chunk is not always a token.
    n_out, counted_by = [], set()
    for r in ok:
        u = r.get("usage") or {}
        if u.get("completion_tokens"):
            n_out.append(int(u["completion_tokens"])); counted_by.add("usage")
        else:
            n_out.append(r["n_chunks"]); counted_by.add("chunks")
    out["completion_tokens_counted_by"] = "+".join(sorted(counted_by))

    n_in = [int((r.get("usage") or {}).get("prompt_tokens") or 0) for r in ok]
    tpots = [(r["latency"] - r["ttft"]) / (n - 1)
             for r, n in zip(ok, n_out) if n > 1 and r["ttft"] is not None]

    total_out = sum(n_out)
    total_decode_s = sum(r["latency"] - r["ttft"]
                         for r in ok if r["ttft"] is not None)

    out["prompt_tokens_total"] = sum(n_in)
    out["completion_tokens_total"] = total_out
    out["mean_completion_tokens"] = total_out / len(ok)
    out["mean_ttft_ms"] = 1000 * statistics.mean(
        r["ttft"] for r in ok if r["ttft"] is not None)
    out["median_ttft_ms"] = 1000 * statistics.median(
        r["ttft"] for r in ok if r["ttft"] is not None)
    if tpots:
        out["mean_tpot_ms"] = 1000 * statistics.mean(tpots)
        out["median_tpot_ms"] = 1000 * statistics.median(tpots)
        # THE DECODE-ONLY CONVENTION, per stream, as the published page defines it.
        out["decode_only_tok_s"] = 1.0 / statistics.mean(tpots)
    # The same thing pooled rather than averaged over streams, so a long
    # request cannot be given the same weight as a short one.
    if total_decode_s > 0:
        out["decode_only_tok_s_pooled"] = total_out / total_decode_s
    # THE WHOLE-RUN CONVENTION: prefill is inside the denominator.
    out["whole_run_tok_s"] = total_out / wall if wall > 0 else 0.0
    # Acceptance, where the engine exposes it. Ours does not expose it on the
    # server path at all (it lives on the internal runner, which only the bench
    # client reaches), so this stays absent for OURS and is reported as absent
    # rather than silently zero.
    acc = [r["accepted_draft_tokens"] for r in ok
           if r.get("accepted_draft_tokens") is not None]
    if acc:
        out["accepted_draft_tokens_total"] = sum(acc)
        out["accepted_draft_tokens_per_output_token"] = sum(acc) / total_out
    else:
        out["accepted_draft_tokens_total"] = None
    itls = [x for r in ok for x in r["itls"]]
    if itls:
        out["median_itl_ms"] = 1000 * statistics.median(itls)
    out["failures"] = [r.get("err") for r in recs if not r.get("ok")][:5]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="http://host:port")
    ap.add_argument("--model", default="model")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num-prompts", type=int, default=164)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--label", default="leg")
    ap.add_argument("--out", default=None, help="write the JSON record here")
    args = ap.parse_args()

    with open(args.dataset) as f:
        data = json.load(f)
    prompts = []
    for item in data[:args.num_prompts]:
        for turn in item["conversations"]:
            if turn["from"] == "human":
                prompts.append(turn["value"])
                break
    if len(prompts) != args.num_prompts:
        print(f"WARN: wanted {args.num_prompts} prompts, got {len(prompts)}",
              file=sys.stderr)

    url = args.url.rstrip("/") + "/v1/chat/completions"
    recs = []
    wall0 = time.perf_counter()
    for i, p in enumerate(prompts):
        try:
            r = stream_one(url, args.model, [{"role": "user", "content": p}],
                           args.max_tokens, args.temperature, args.top_p,
                           args.top_k, args.seed, args.timeout)
            r["ok"] = r["ttft"] is not None and r["n_chunks"] > 0
            if not r["ok"]:
                r["err"] = "empty completion"
        except Exception as e:                        # noqa: BLE001
            r = {"ok": False, "err": f"{type(e).__name__}: {e}"}
        r["i"] = i
        recs.append(r)
        if (i + 1) % 20 == 0:
            print(f"  [{args.label}] {i + 1}/{len(prompts)} "
                  f"t={time.perf_counter() - wall0:.0f}s", flush=True)
    wall = time.perf_counter() - wall0

    summary = summarize(args.label, recs, wall)
    print("CLIENT_RESULT " + json.dumps(summary), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary,
                       "config": vars(args),
                       "records": [{k: v for k, v in r.items() if k != "text"}
                                   for r in recs],
                       "first_text": next((r.get("text") for r in recs
                                           if r.get("ok")), "")[:600]},
                      f, indent=1)
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

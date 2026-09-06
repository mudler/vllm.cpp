#!/usr/bin/env python3
"""ONE client, two engines, a variadic load at a fixed concurrency.

The timing code is byte-identical on both sides, so the counting convention
cannot be a difference between the engines. It speaks OpenAI
/v1/chat/completions with stream=true against whatever host and port it is
given, and it does not know which engine answers.

WHAT THIS ADDS OVER THE CONCURRENCY-1 HEAD-TO-HEAD CLIENT:

  * A closed loop at concurrency C. C worker threads each take the next prompt
    and send it as soon as their previous request completes. This is vLLM's
    `--request-rate inf --max-concurrency C` (vllm/benchmarks/serve.py:787).
  * An explicit warmup phase, run to completion before the measured window
    opens, and RECORDED rather than thrown away, so both the warm-only and the
    with-warmup views come out of one set of records.
  * Per-request chunk statistics: the character length of the first content
    chunk, the total character length, and the chunk count. Two engines that
    stream at different granularities do not have the same TTFT, and this is
    what lets the report say by how much.
  * A /metrics scrape either side of the measured window, so acceptance is
    readable on our side (`vllm:spec_decode_*`) as well as theirs.

It computes no percentiles. `report.py` does, from the records this writes, so
the statistics can be recomputed without rerunning the GPU.

ignore_eos is deliberately NOT sent. Both published protocols let the model
stop, and forcing a fixed length is one of the differences that made earlier
numbers non-comparable.

TWO DELIBERATE DIVERGENCES FROM vLLM's OWN CLIENT, declared here because
`report.py` says its metric definitions are mirrored from `serve.py`:

  * `latency` stops after the response stream is fully read, where vLLM stops at
    the last data frame (`vllm/benchmarks/lib/endpoint_request_func.py:541`).
    `[DONE]` and the connection close are therefore inside `e2el`, and so inside
    `tpot`. Over loopback this is small, and it is the same code on both
    engines, but it is not upstream's definition.
  * `ttft` requires a chunk that carries CONTENT, where vLLM takes the first
    chunk with `choices` at all
    (`endpoint_request_func.py:520-524`). One of the two engines here opens with
    a role-only delta and the other does not, so upstream's rule would have
    measured a protocol difference.
"""
import argparse
import json
import queue
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request


def stream_one(url, model, messages, args):
    """One streaming chat completion. Returns raw timings; no statistics.

    `ttft` is the wall time to the first chunk that carries CONTENT, not to the
    first chunk of any kind: a role-only opening delta would otherwise read as a
    token and shorten ttft on whichever engine happens to send one.
    """
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.seed is not None:
        body["seed"] = args.seed
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    ttft = None
    chunk_times = []
    chunk_chars = []
    text_len = 0
    first_chunk_chars = 0
    usage = None
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
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
            # leaves the tags inline in `content`. Counting only `content` would
            # measure their TTFT to the END of the reasoning block and drop
            # every reasoning token from the rate.
            piece = delta.get("content") or delta.get("reasoning_content")
            if not piece:
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
                first_chunk_chars = len(piece)
            chunk_times.append(now - t0)
            chunk_chars.append(len(piece))
            text_len += len(piece)
    latency = time.perf_counter() - t0

    return {
        "ttft": ttft,
        "latency": latency,
        "n_chunks": len(chunk_times),
        "first_chunk_chars": first_chunk_chars,
        "total_chars": text_len,
        # Chunk arrival gaps. This is vLLM's `itl` (serve.py:614): a gap between
        # STREAMED CHUNKS, which is a per-token gap only on an engine that
        # streams one token per chunk.
        "itls": [chunk_times[i] - chunk_times[i - 1]
                 for i in range(1, len(chunk_times))],
        "usage": usage,
        "accepted_draft_tokens": (usage or {}).get("accepted_draft_tokens"),
    }


def scrape_metrics(base_url, timeout=10.0):
    """The Prometheus text exposition, filtered to the speculative families.

    Ours serves these since #2770 landed. Theirs has no /metrics at all, so an
    absent scrape is recorded as absent and never as zero.
    """
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/metrics",
                                    timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
    except Exception as exc:                                    # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    out = {"available": True, "counters": {}}
    for line in body.splitlines():
        if line.startswith("#") or "spec_decode" not in line:
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        try:
            out["counters"][parts[0].strip()] = float(parts[1])
        except ValueError:
            continue
    return out


def run_phase(prompts, args, url, phase, concurrency, progress_every=25):
    """Run `prompts` closed-loop at `concurrency`. Returns (records, wall_s).

    The wall clock starts when the first worker is released and stops when the
    last request completes, so it is the duration the throughput denominators
    use (vllm/benchmarks/serve.py `dur_s`).
    """
    work = queue.Queue()
    for i, item in enumerate(prompts):
        work.put((i, item))
    records = [None] * len(prompts)
    done = [0]
    lock = threading.Lock()
    start_gate = threading.Barrier(concurrency + 1)

    def worker():
        start_gate.wait()
        while True:
            try:
                i, item = work.get_nowait()
            except queue.Empty:
                return
            t_dispatch = time.perf_counter()
            try:
                rec = stream_one(url, args.model,
                                 [{"role": "user", "content": item["text"]}],
                                 args)
                rec["ok"] = rec["ttft"] is not None and rec["n_chunks"] > 0
                if not rec["ok"]:
                    rec["err"] = "empty completion"
            except Exception as exc:                            # noqa: BLE001
                rec = {"ok": False, "err": f"{type(exc).__name__}: {exc}"}
            rec.update({"i": i, "id": item["id"], "band": item["band"],
                        "phase": phase, "t_dispatch": t_dispatch})
            records[i] = rec
            with lock:
                done[0] += 1
                n = done[0]
            if n % progress_every == 0:
                print(f"  [{args.label}/{phase}] {n}/{len(prompts)} "
                      f"t={time.perf_counter() - t_dispatch:.0f}s", flush=True)

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(concurrency)]
    for t in threads:
        t.start()
    start_gate.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    for rec in records:
        if rec is not None:
            rec["t_dispatch"] -= t0
    return [r for r in records if r is not None], wall


def headline(records, wall):
    """Enough to read the log while the job runs. report.py does the real work."""
    ok = [r for r in records if r.get("ok")]
    if not ok:
        return {"ok": 0, "error": "no successful request"}
    out_toks = [int((r.get("usage") or {}).get("completion_tokens") or 0)
                for r in ok]
    if not all(out_toks):
        return {"ok": len(ok), "error": "a request reported no usage token count"}
    tpots = [(r["latency"] - r["ttft"]) / (n - 1)
             for r, n in zip(ok, out_toks) if n > 1]
    return {
        "ok": len(ok),
        "requests": len(records),
        "wall_s": wall,
        "output_tokens": sum(out_toks),
        "output_tok_s": sum(out_toks) / wall if wall > 0 else 0.0,
        "mean_ttft_ms": 1000 * statistics.mean(r["ttft"] for r in ok),
        "median_ttft_ms": 1000 * statistics.median(r["ttft"] for r in ok),
        "mean_tpot_ms": 1000 * statistics.mean(tpots) if tpots else None,
        "decode_only_tok_s": 1.0 / statistics.mean(tpots) if tpots else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="http://host:port")
    ap.add_argument("--model", default="model")
    ap.add_argument("--dataset", required=True, help="corpus from build_corpus.py")
    ap.add_argument("--num-prompts", type=int, default=128,
                    help="MEASURED requests; warmup is on top of this")
    ap.add_argument("--warmup", type=int, default=None,
                    help="warmup requests; default max(concurrency, 4)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--label", default="leg")
    ap.add_argument("--arm", default="", help="OURS or THEIRS, recorded only")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    warmup = args.warmup if args.warmup is not None else max(args.concurrency, 4)
    with open(args.dataset, encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for entry in data:
        text = next(t["value"] for t in entry["conversations"]
                    if t["from"] == "human")
        items.append({"id": entry["id"], "band": entry.get("band", "?"),
                      "text": text})
    need = warmup + args.num_prompts
    if len(items) < need:
        print(f"ERROR: corpus has {len(items)} prompts, leg needs {need}",
              file=sys.stderr)
        return 2
    warm_items, measured_items = items[:warmup], items[warmup:need]

    url = args.url.rstrip("/") + "/v1/chat/completions"

    # WARMUP RUNS TO COMPLETION FIRST. Overlapping it with the measured window
    # would put a cold request inside the measured wall clock, which is the
    # thing the discard exists to remove.
    print(f"[{args.label}] warmup {warmup} at c={args.concurrency}", flush=True)
    warm_recs, warm_wall = run_phase(warm_items, args, url, "warmup",
                                     args.concurrency)

    m_before = scrape_metrics(args.url)
    print(f"[{args.label}] measured {args.num_prompts} at c={args.concurrency}",
          flush=True)
    meas_recs, meas_wall = run_phase(measured_items, args, url, "measured",
                                     args.concurrency)
    m_after = scrape_metrics(args.url)

    summary = {
        "leg": args.label,
        "arm": args.arm,
        "round": args.round,
        "concurrency": args.concurrency,
        "warmup_requests": warmup,
        "measured_requests": args.num_prompts,
        "warm_wall_s": warm_wall,
        "measured_wall_s": meas_wall,
        "headline_warm": headline(meas_recs, meas_wall),
        "headline_all": headline(warm_recs + meas_recs, warm_wall + meas_wall),
    }
    print("CLIENT_RESULT " + json.dumps(summary), flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary,
                       "config": vars(args),
                       "metrics_before": m_before,
                       "metrics_after": m_after,
                       "records": warm_recs + meas_recs}, f)
    return 0 if summary["headline_warm"].get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

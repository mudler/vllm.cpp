#!/usr/bin/env python3
"""Independent re-derivation from the RAW per-leg .err/.rc/.jsonl records.
Deliberately does NOT read RESULT.json's summary or legs blocks for any figure;
it reads RESULT.json only at the end, to compare."""
import json, re, statistics as st, pathlib

# The raw run directory on the share, and this evidence directory as the
# fallback so the script runs against its own committed artefacts. The
# committed copies differ only in name: `*.err` -> `*.err.txt`, `*.out` ->
# `*.out.txt`, and the clock `*.jsonl` gzipped, because `.gitignore` and
# check-pr-size's BENCH_EVIDENCE_RUN decide which extensions may land.
SHARE = pathlib.Path("/mnt/nas_share/rc/strix-arm-2933/out")
HERE = pathlib.Path(__file__).resolve().parent
OUT = SHARE if (SHARE / "RESULT.json").exists() else HERE
print(f"reading from {OUT}")


def raw(name: str) -> str:
    """Read a per-leg capture under either the share's name or the committed one."""
    for candidate in (OUT / name, OUT / f"{name}.txt"):
        if candidate.exists():
            return candidate.read_text()
    raise FileNotFoundError(name)


def raw_lines(name: str) -> list[str]:
    """Read a clock capture, plain on the share and gzipped once committed."""
    plain = OUT / name
    if plain.exists():
        return plain.read_text().splitlines()
    gz = OUT / f"{name}.gz"
    if gz.exists():
        import gzip
        return gzip.decompress(gz.read_bytes()).decode().splitlines()
    raise FileNotFoundError(name)
RE_RUN = re.compile(
    r"vllm-cli: run=(\d+)/(\d+) finish_reason=(\S+) prompt_tokens=(\d+) "
    r"completion_tokens=(\d+) secs=([\d.]+) tok_s=([\d.]+)")

def med(xs):
    return st.median(xs)

legs = {}
for n in (64, 128):
    for r in (1, 2, 3, 4):
        tag = f"n{n}-r{r}"
        err = raw(f"{tag}.err")
        rc = int((OUT / f"{tag}.rc").read_text().strip())
        reps = []
        for m in RE_RUN.finditer(err):
            reps.append(dict(run=int(m.group(1)), of=int(m.group(2)),
                             finish=m.group(3), ptok=int(m.group(4)),
                             ctok=int(m.group(5)), secs=float(m.group(6)),
                             tok_s=float(m.group(7))))
        # fault detection, independent of the fold
        faults = [w for w in ("recoverable page fault", "Memory access fault",
                              "Aborted", "core dumped", "HSA_STATUS_ERROR",
                              "terminate called")
                  if w in err]
        tier = len([l for l in err.splitlines()
                    if "reference" in l.lower() and "tier" in l.lower()])
        kept = reps[1:]           # cold run discarded, per the design
        legs[tag] = dict(n=n, rc=rc, reps=reps, kept=kept, faults=faults,
                         tier=tier,
                         secs=med([x["secs"] for x in kept]),
                         tok_s=med([x["tok_s"] for x in kept]))

print("=== per-leg, re-derived from .err ===")
for tag, L in legs.items():
    print(f"{tag}: rc={L['rc']} reps={len(L['reps'])} kept={len(L['kept'])} "
          f"ctok={{{','.join(str(x['ctok']) for x in L['reps'])}}} "
          f"finish={{{','.join(sorted(set(x['finish'] for x in L['reps'])))}}} "
          f"median_secs={L['secs']:.3f} median_tok_s={L['tok_s']:.4f} "
          f"faults={L['faults']} tier_lines={L['tier']}")

arms = {}
for n in (64, 128):
    ls = [legs[f"n{n}-r{r}"] for r in (1, 2, 3, 4)]
    ts = [L["tok_s"] for L in ls]
    ss = [L["secs"] for L in ls]
    arms[n] = dict(tok_s=med(ts), secs=med(ss),
                   spread=(max(ts) - min(ts)) / med(ts) * 100,
                   legs=len(ls), ok=sum(1 for L in ls if L["rc"] == 0 and not L["faults"]))
    print(f"\nn{n}: median whole-completion {arms[n]['tok_s']:.4f} tok/s "
          f"({arms[n]['secs']:.4f} s), spread {arms[n]['spread']:.4f}% of median, "
          f"{arms[n]['ok']} of {arms[n]['legs']} clean")

# slope-derived decode, paired by round
slopes = []
for r in (1, 2, 3, 4):
    ta = legs[f"n64-r{r}"]["secs"]; tb = legs[f"n128-r{r}"]["secs"]
    s = (128 - 64) / (tb - ta)
    slopes.append(s)
    print(f"pair r{r}: t_a={ta:.3f} t_b={tb:.3f} slope={s:.6f} tok/s "
          f"fixed_cost={ta - 64 / s:.6f}s")
dec = med(slopes)
dec_spread = (max(slopes) - min(slopes)) / dec * 100
print(f"\ndecode DERIVED: {dec:.6f} tok/s, spread {dec_spread:.6f}%, "
      f"{len(slopes)} of 4 pairs usable")

# clocks, from the raw jsonl
allmhz, allbusy = [], []
print("\n=== clocks, re-derived from .jsonl ===")
for tag in legs:
    rows = [json.loads(l) for l in raw_lines(f"clock-{tag}.jsonl") if l.strip()]
    mhz = [x["sclk_mhz"] for x in rows]; busy = [x["busy_percent"] for x in rows]
    allmhz += mhz; allbusy += busy
    boots = set(x["boot_id"] for x in rows)
    print(f"clock-{tag}: n={len(rows)} sclk mean={st.mean(mhz):.1f} "
          f"min={min(mhz)} max={max(mhz)} busy mean={st.mean(busy):.1f} boots={len(boots)}")
print(f"ALL LEGS POOLED: n={len(allmhz)} sclk mean={st.mean(allmhz):.1f} "
      f"busy mean={st.mean(allbusy):.1f}")

# ratios against the carried constants
VLLM_WHOLE = 6.734; LCPP_DECODE = 12.219
print(f"\nwhole completion  vLLM {VLLM_WHOLE} / ours {arms[64]['tok_s']:.4f} = "
      f"{VLLM_WHOLE / arms[64]['tok_s']:.6f}x")
print(f"decode  llama.cpp {LCPP_DECODE} / ours(derived) {dec:.6f} = "
      f"{LCPP_DECODE / dec:.6f}x")

# --- now compare against RESULT.json ---
R = json.loads(raw("RESULT.json"))
checks = []
def ck(name, mine, theirs, tol=5e-4):
    ok = abs(mine - theirs) <= tol * max(1.0, abs(theirs))
    checks.append((name, mine, theirs, ok))
ck("n64 median tok_s", arms[64]["tok_s"], R["arms"]["n64"]["median_whole_completion_tok_s"])
ck("n128 median tok_s", arms[128]["tok_s"], R["arms"]["n128"]["median_whole_completion_tok_s"])
ck("n64 median secs", arms[64]["secs"], R["arms"]["n64"]["median_whole_completion_secs"])
ck("n128 median secs", arms[128]["secs"], R["arms"]["n128"]["median_whole_completion_secs"])
ck("n64 spread%", arms[64]["spread"], R["arms"]["n64"]["leg_spread_pct_of_median"])
ck("n128 spread%", arms[128]["spread"], R["arms"]["n128"]["leg_spread_pct_of_median"])
ck("decode derived", dec, R["decode_derived"]["median_decode_tok_s_derived"])
ck("decode spread%", dec_spread, R["decode_derived"]["spread_pct_of_median"])
for i, r in enumerate(R["decode_derived"]["rows"]):
    ck(f"pair r{i+1} slope", slopes[i], r["slope_tok_s"])
ck("ratio vllm/ours whole", VLLM_WHOLE / arms[64]["tok_s"],
   R["ratios"]["like_for_like_whole_completion__vllm_over_vllmcpp"]["value"])
ck("ratio lcpp/ours decode", LCPP_DECODE / dec,
   R["ratios"]["like_for_like_decode__llamacpp_over_vllmcpp_derived"]["value"])
faultstr = f"{sum(1 for L in legs.values() if L['rc'] == 0 and not L['faults'])} of 8"
checks.append(("fault_rate_all_legs", f"0 faults, {faultstr} clean",
               R["fault_rate_all_legs"], R["fault_rate_all_legs"] == "0 of 8" and faultstr == "8 of 8"))
tt = sum(L["tier"] for L in legs.values())
checks.append(("reference_tier_lines total", tt, 0, tt == 0))

print("\n=== CLAIMS CHECKED ===")
bad = 0
for name, mine, theirs, ok in checks:
    if not ok: bad += 1
    print(f"[{'OK ' if ok else 'MISMATCH'}] {name}: mine={mine} result.json={theirs}")
print(f"\nCLAIMS_CHECKED={len(checks)}  MISMATCHES={bad}")

# --- the generated text itself: is the 128-token leg the 64-token leg continued? ---
print("\n=== generated text, checked as bytes ===")
outs = {}
for n in (64, 128):
    hs = set()
    for r in (1, 2, 3, 4):
        b = raw(f"n{n}-r{r}.out").encode()
        outs.setdefault(n, b); hs.add(b)
    print(f"n{n}: {len(hs)} distinct completion(s) across 4 legs "
          f"({'byte-identical' if len(hs) == 1 else 'NOT identical'})")
a, b = outs[64], outs[128]
print(f"n128 continues n64 exactly: {b.startswith(a.rstrip(b"\n"))} "
      f"(the 64-token capture adds a trailing newline the 128-token one does not)")

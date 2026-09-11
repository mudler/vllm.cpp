#!/usr/bin/env bash
# Binding x86_64 A/B series for BACKEND-GATE-CPU-LLAMACPP (#433).
# Evidence: docs/bench-evidence/cpu-x86-llamacpp-20260811.md
# Run from the repo root of a CPU-only Release build (build-cpu/).
#
# Strictly interleaved -- ours rep N, then llama rep N -- so neither engine can
# own a quiet window the other did not also get.
#
# This box is shared and another session's parallel build drove the load average
# from 3.8 to 82 mid-series once already, so every leg is gated on both sides:
# it will not START until the box is measurably idle, and it is DISCARDED if a
# foreign process burned CPU while it RAN.
#
# THE GATE MUST NOT SEE ITSELF. Two instances of that bug have already cost this
# harness a series each:
#   1. `pgrep -f` on a compiler-name pattern matched the waiter's own command
#      line, so the gate blocked on its own reflection while the box was idle.
#   2. The one-minute load average includes THIS harness. A 20-thread leg drives
#      it to 20-33 on its own, so a load-based gate can never re-arm after the
#      first leg: it is waiting for its own decay. The quiet gate therefore
#      measures FOREIGN CPU SHARE from /proc/stat over a fresh window, which is
#      zero while the waiter sleeps and is not decayed by anything we did.
# Load averages are still RECORDED before and after every leg -- gate G5 asks
# for them -- they are simply not what the gate decides on.
set -u

M=${M:-/home/mudler/.config/dante-desktop/models/Qwen3.5-2B-UD-Q8_K_XL.gguf}
LB=${LB:-/home/mudler/llamacpp-build-cpu-bench/bin/llama-bench}
OB=${OB:-./build-cpu/examples/vllm-bench}
OUT=${OUT:-evi}
T=${T:-20}
REPS=${REPS:-3}
# The recorded recipe. Overridable only so the smoke test can run without a
# 20-core box or GNU time; every recorded leg used these defaults verbatim.
TIMEV=${TIMEV:-/usr/bin/time -v}
TASKSET=${TASKSET-taskset -c 0-19}   # `-`, not `:-`: an explicit "" means none
# Percent of ALL cpus that may be busy with work that is not ours. 10% of 20
# vCPUs is two cores of background noise.
QUIET_BUSY=${QUIET_BUSY:-10}
FOREIGN_MAX=${FOREIGN_MAX:-10}
BUSY_WINDOW=${BUSY_WINDOW:-5}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-3600}
# Exact process-NAME match (-x, no -f). Matching full command lines instead
# matches this script's own gate and the pgrep call itself, which deadlocks the
# wait on its own reflection -- that cost a whole series. Compilers are named
# because they are the observed offender on this box; they are NOT the whole
# net, which is why the CPU-share check below is unconditional. Our own two
# benchmark binaries are deliberately ABSENT from this list: naming them would
# be the self-reflection bug a third time.
BUILDERS=${BUILDERS:-'cc1plus|cc1|cc1obj|lto1|nvcc|hipcc|ninja|collect2|as|ld|ld.lld|gcc|g\+\+|clang|clang\+\+|cmake|make|ctest|python3|python|cargo|rustc|node|rsync|ffmpeg'}

mkdir -p "$OUT" || { echo "cannot create OUT=$OUT"; exit 2; }

CLK=$(getconf CLK_TCK)
NCPU=$(getconf _NPROCESSORS_ONLN)

load1() { cut -d' ' -f1 /proc/loadavg; }
loadall() { cut -d' ' -f1-3 /proc/loadavg; }

ppid_of() {  # comm can contain spaces and parens, so split after the LAST ')'
  local line
  line=$(cat "/proc/$1/stat" 2>/dev/null) || return 1
  [ -n "$line" ] || return 1
  printf '%s\n' "${line##*) }" | cut -d' ' -f2
}

# A candidate that is us, our ancestor, or our descendant is not a foreign
# process. Naming a process class and then matching yourself is the bug that
# has now cost this harness two series; excluding our own tree kills the class
# rather than the two instances of it.
is_ours() {
  local p=$1 q
  q=$p
  while [ -n "$q" ] && [ "$q" -gt 1 ] 2>/dev/null; do
    [ "$q" = "$$" ] && return 0
    q=$(ppid_of "$q") || break
  done
  q=$$
  while [ -n "$q" ] && [ "$q" -gt 1 ] 2>/dev/null; do
    [ "$q" = "$p" ] && return 0
    q=$(ppid_of "$q") || break
  done
  return 1
}

# pgrep -c already prints 0 on no match and exits 1, so a `|| echo 0` fallback
# emits "0\n0" and every numeric test that consumes it fails. Count PIDs, not
# `-c`, so our own tree can be filtered out.
builders() {
  local n=0 p
  for p in $(pgrep -x "$BUILDERS" 2>/dev/null); do
    is_ours "$p" || n=$((n + 1))
  done
  echo "$n"
}

# "busy" and "total" jiffies across all cpus. iowait is NOT counted as busy: a
# neighbour waiting on disk does not steal our cores. steal IS counted -- on a
# KVM guest that is exactly the co-tenant this box keeps losing series to.
stat_snapshot() {
  local line
  IFS= read -r line < /proc/stat || return 1
  printf '%s\n' "$line"
}

invalid_cpu_sample() {
  echo "INVALID_CPU_SAMPLE: unavailable or inconsistent counters" >&2
  echo 100  # conservative sentinel, never a valid measurement
  return 1
}

cpu_share() {  # before, after, own jiffies
  local -a before after
  local i a b delta total=0 busy=0 own=$3
  read -r -a before <<< "$1"
  read -r -a after <<< "$2"
  if [ "${before[0]-}" != cpu ] || [ "${after[0]-}" != cpu ]; then
    invalid_cpu_sample; return 1
  fi
  # Eight 16-digit decimal counters, multiplied by 100, fit signed 64-bit.
  # Validate before arithmetic: Bash otherwise accepts expressions and octal.
  [[ "$own" =~ ^[0-9]{1,16}$ ]] || { invalid_cpu_sample; return 1; }
  own=$((10#$own))
  for i in {1..8}; do
    a=${before[$i]-}; b=${after[$i]-}
    if ! [[ "$a" =~ ^[0-9]{1,16}$ && "$b" =~ ^[0-9]{1,16}$ ]]; then
      invalid_cpu_sample; return 1
    fi
    a=$((10#$a)); b=$((10#$b))
    # Linux documents that iowait can decrease. Do not call that sample quiet.
    if (( b < a )); then invalid_cpu_sample; return 1; fi
    delta=$((b - a)); total=$((total + delta))
    if (( i != 4 && i != 5 )); then busy=$((busy + delta)); fi
  done
  # Guest and guest_nice overlap user and nice: do not add fields 9 and 10.
  if (( total <= 0 )); then invalid_cpu_sample; return 1; fi
  busy=$((busy - own))
  (( busy < 0 )) && busy=0
  echo $((100 * busy / total))
}

# Percent of the whole machine busy over a fresh BUSY_WINDOW. Nothing here is
# decayed and nothing here counts a process that has already exited, so the
# harness cannot gate on its own previous leg.
busy_pct() {
  local s0 s1
  s0=$(stat_snapshot) || { invalid_cpu_sample; return 1; }
  sleep "$BUSY_WINDOW"
  s1=$(stat_snapshot) || { invalid_cpu_sample; return 1; }
  cpu_share "$s0" "$s1" 0
}

wait_quiet() {
  local waited=0 p b valid
  while :; do
    b=$(builders)
    p=$(busy_pct)          # consumes BUSY_WINDOW seconds
    valid=$?
    waited=$((waited + BUSY_WINDOW))
    if [ "$p" -le "$QUIET_BUSY" ] && [ "$b" -eq 0 ] && [ "$valid" -eq 0 ]; then return 0; fi
    if [ "$waited" -ge "$WAIT_TIMEOUT" ]; then
      # The spec's stop condition: the box cannot be brought under the ceiling,
      # so the axes stay PENDING a quiet host. Stop; do not average through it.
      echo "NO_QUIET_WINDOW after ${waited}s (busy=${p}% builders=$b load=$(loadall))"
      exit 4
    fi
    sleep 15; waited=$((waited + 15))
    if [ $((waited % 300)) -lt 20 ]; then
      echo "waiting for quiet: ${waited}s busy=${p}% builders=$b load=$(load1)"
    fi
  done
}

# Own CPU seconds of the leg, from the GNU time report it just wrote.
own_cpu_jiffies() {
  awk -v clk="$CLK" '
    /User time \(seconds\)/   {u=$NF}
    /System time \(seconds\)/ {s=$NF}
    END {printf "%d", (u+s)*clk}
  ' "$1" 2>/dev/null
}

run_leg() {  # engine rep -> 0 accepted, 1 discard
  local eng=$1 rep=$2 rc stem s0 s1 own fpct bafter valid
  stem="$OUT/$eng-$rep"
  wait_quiet || return 1
  # G5: the load average before and after EVERY leg, in a file, per leg.
  {
    echo "leg=$eng rep=$rep"
    echo "before uptime: $(uptime)"
    echo "before loadavg: $(loadall)"
    echo "before builders: $(builders)"
  } > "$stem.load"
  echo "$eng rep=$rep START load=$(loadall) builders=$(builders)"
  s0=$(stat_snapshot)
  if [ "$eng" = ours ]; then
    # shellcheck disable=SC2086
    $TIMEV $TASKSET env VLLM_CPP_CPU_THREADS=$T \
      "$OB" --model "$M" --num-prompts 1 --input-len 128 --output-len 32 \
      --concurrency 1 --seed 0 --temperature 0 \
      > "$OUT/ours-bench-$rep.txt" 2> "$OUT/ours-bench-$rep.time"
    rc=$?
    own=$(own_cpu_jiffies "$OUT/ours-bench-$rep.time")
  else
    # shellcheck disable=SC2086
    $TIMEV $TASKSET "$LB" -m "$M" -p 128 -n 32 -pg 128,32 \
      -t $T -ngl 0 -r 3 -o json \
      > "$OUT/llama-bench-$rep.json" 2> "$OUT/llama-bench-$rep.time"
    rc=$?
    own=$(own_cpu_jiffies "$OUT/llama-bench-$rep.time")
  fi
  s1=$(stat_snapshot)
  bafter=$(builders)
  # Everything the machine burned while our leg ran, minus what our leg burned,
  # as a share of the machine. This is the check the old post-leg test did not
  # make: it re-tested `builders` only, so a leg could run straight through
  # load 80 from any non-compiler source and still be ACCEPTED.
  fpct=$(cpu_share "$s0" "$s1" "$own")
  valid=$?
  {
    echo "after uptime: $(uptime)"
    echo "after loadavg: $(loadall)"
    echo "after builders: $bafter"
    echo "exit: $rc"
    echo "own_cpu_jiffies: $own"
    echo "foreign_cpu_pct: $fpct"
    echo "cpu_sample_status: $valid"
    echo "ncpu: $NCPU"
  } >> "$stem.load"
  echo "$eng rep=$rep END exit=$rc load=$(loadall) builders=$bafter foreign=${fpct}%"
  if [ "$valid" -ne 0 ] || [ "$rc" -ne 0 ] || [ "$bafter" -ne 0 ] || [ "$fpct" -gt "$FOREIGN_MAX" ]; then
    echo "$eng rep=$rep DISCARDED (exit=$rc builders_after=$bafter foreign=${fpct}%)"
    return 1
  fi
  return 0
}

summarize() {  # every published figure comes from here, never from a human eye
  OUT="$OUT" REPS="$REPS" python3 - <<'PY'
import json, os, pathlib, re, statistics, sys

out = pathlib.Path(os.environ["OUT"])
reps = int(os.environ["REPS"])
RSS = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def rss(path):
    if not path.exists():
        return None
    m = RSS.search(path.read_text(errors="replace"))
    return int(m.group(1)) if m else None


def ours_rates(path):
    if not path.exists():
        return {}
    got = {}
    for line in path.read_text(errors="replace").splitlines():
        for key, tag in (
            ("prefill", "Prefill token throughput"),
            ("decode", "Output (decode) token throughput"),
            ("e2e", "Total token throughput"),
        ):
            if line.startswith(tag):
                got[key] = float(line.rsplit(":", 1)[1])
    return got


def llama_rates(path):
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(errors="replace"))
    except json.JSONDecodeError:
        return {}
    got = {}
    for row in rows:
        npr, ngen = int(row.get("n_prompt", 0)), int(row.get("n_gen", 0))
        key = "prefill" if ngen == 0 else "decode" if npr == 0 else "e2e"
        got[key] = float(row["avg_ts"])
    return got


legs = {"ours": [], "llama": []}
for rep in range(1, reps + 1):
    legs["ours"].append(
        (rss(out / f"ours-bench-{rep}.time"), ours_rates(out / f"ours-bench-{rep}.txt"))
    )
    legs["llama"].append(
        (rss(out / f"llama-bench-{rep}.time"), llama_rates(out / f"llama-bench-{rep}.json"))
    )


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def spread(values):
    values = [v for v in values if v is not None]
    if len(values) < 2 or not min(values):
        return None
    return 100.0 * (max(values) - min(values)) / min(values)


lines = ["# Series summary (generated by scripts/cpu-x86-llamacpp-floor.sh)", ""]
lines.append("| Axis | vllm.cpp | llama.cpp | Ratio | Better |")
lines.append("|---|---:|---:|---:|---|")
axes = [
    ("Peak RSS (KB)", "rss", "lower"),
    ("Prefill tok/s", "prefill", "higher"),
    ("Decode tok/s", "decode", "higher"),
    ("E2E tok/s", "e2e", "higher"),
]
failures = []
for label, key, better in axes:
    if key == "rss":
        ours = med([r for r, _ in legs["ours"]])
        them = med([r for r, _ in legs["llama"]])
        sp = (spread([r for r, _ in legs["ours"]]), spread([r for r, _ in legs["llama"]]))
    else:
        ours = med([d.get(key) for _, d in legs["ours"]])
        them = med([d.get(key) for _, d in legs["llama"]])
        sp = (
            spread([d.get(key) for _, d in legs["ours"]]),
            spread([d.get(key) for _, d in legs["llama"]]),
        )
    if ours is None or them is None:
        lines.append(f"| {label} | {ours} | {them} | n/a, a leg is missing | - |")
        failures.append(f"{label}: no ratio, a leg is missing")
        continue
    ratio = ours / them if better == "lower" else them / ours
    verdict = "ours" if ratio < 1.0 else ("tie" if ratio == 1.0 else "llama.cpp")
    lines.append(f"| {label} | {ours:.4f} | {them:.4f} | {ratio:.4f}x | {verdict} |")
    lines.append(
        f"| &nbsp;&nbsp;leg spread | {sp[0] if sp[0] is None else f'{sp[0]:.3f}%'} "
        f"| {sp[1] if sp[1] is None else f'{sp[1]:.3f}%'} | | |"
    )

lines += ["", "## G5: load recorded before and after every leg", ""]
lines.append("| Leg | before 1m | after 1m | foreign CPU during leg | verdict |")
lines.append("|---|---:|---:|---:|---|")
for rep in range(1, reps + 1):
    for eng in ("ours", "llama"):
        path = out / f"{eng}-{rep}.load"
        if not path.exists():
            lines.append(f"| {eng} {rep} | - | - | - | MISSING |")
            failures.append(f"{eng} rep {rep}: no load record, G5 cannot be reported")
            continue
        text = path.read_text()

        def field(tag, text=text):
            m = re.search(rf"^{tag}: (.*)$", text, re.M)
            return m.group(1).strip() if m else "-"

        before = field("before loadavg").split()[0]
        after = field("after loadavg").split()[0] if field("after loadavg") != "-" else "-"
        lines.append(
            f"| {eng} {rep} | {before} | {after} | {field('foreign_cpu_pct')}% | accepted |"
        )

(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
if failures:
    print("SUMMARY_INCOMPLETE: " + "; ".join(failures), file=sys.stderr)
    sys.exit(1)
PY
}

R=1
attempt=0
while [ "$R" -le "$REPS" ]; do
  attempt=$((attempt + 1))
  if [ "$attempt" -gt 24 ]; then echo "GIVING_UP too many discards"; exit 2; fi
  if run_leg ours "$R" && run_leg llama "$R"; then
    echo "pair rep=$R ACCEPTED"
    R=$((R + 1))
  else
    echo "pair rep=$R RETRY"
    rm -f "$OUT/ours-bench-$R.txt" "$OUT/ours-bench-$R.time" \
          "$OUT/llama-bench-$R.json" "$OUT/llama-bench-$R.time" \
          "$OUT/ours-$R.load" "$OUT/llama-$R.load"
  fi
done
summarize || { echo "SERIES_INCOMPLETE"; exit 3; }
echo "SERIES_DONE"

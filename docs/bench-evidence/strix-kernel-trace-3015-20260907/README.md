# Strix kernel investigation, 7 September 2026

Row: `BACKEND-GATE-ROCM-LLAMACPP`. Issues: [#3015](https://github.com/mudler/vllm.cpp/issues/3015), [#3039](https://github.com/mudler/vllm.cpp/issues/3039), and [#3040](https://github.com/mudler/vllm.cpp/issues/3040).

This folder preserves diagnostic captures. `TOKEN_GATE=FAIL` is carried from
the earlier survey, not remeasured here. Invalid profiler timestamps prevent
kernel timing attribution. The matched llama.cpp trace remains owed under
#3040. No performance result or default change is accepted.

## Provenance

The source directory is
`/mnt/nas_share/rc/strix-3015-20260907.Gq6MW5`. Capture names replace `/` with
`--`. CSV and JSONL files use lossless gzip with no original-name or timestamp
header. Other captures append `.txt`, including executable worker snapshots
that are retained as evidence rather than executable analysis tools.
`SOURCE_SHA256SUMS.txt` records the original bytes and relative source paths.
`SHA256SUMS.txt` records the packaged files.

`build-deps--build-state.json.txt` contains the source archives, binary and
library hashes, image ID, model path, and build lease identity. The archives,
model, and binaries are not copied into this folder.

| Input | Pin |
|---|---|
| vllm.cpp source | `f98b638673b4d2edc0250eec56d229357ea38ab1` |
| vllm.cpp archive SHA256 | `a57ae15a89351f216fa7002c4300fa72d67e48b234ac6dd898f115751c812647` |
| llama.cpp source | `10bf611e533d81f739128304991c5e133c6aebd8`, stock b10451 |
| llama.cpp archive SHA256 | `0f43551d3da94ba0d2af80e202177008a91966de66dbabf87a2042377d16d133` |
| Rebuilt image | `d554fe849c5e3c4d6b81227e0b87f165fa3b7b015899872bd93bc732c12ffeb6` |
| Model | `Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008 bytes |
| Model SHA256 checked by the worker | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` |
| Compiler | AMD clang 22.0.0git, roc-7.2.4, `f58b06dce1f9c15707c5f808fd002e18c2accf7e` |
| ccache | 4.9.1 |

The first profiled attempt failed to load `libdw.so.1`. The dependency log
records installation of `libdw1t64` and `hsa-amd-aqlprofile`, followed by a
new image and fresh builds. Both initial and rebuilt manifests, build states,
compiler logs, build commands, and CMake caches are retained. The capture
logs report HIP runtime 7.2.0 and HSA 8.20.4. A separate post-measurement
version probe in lease `0650e658-2657-4f37-8af9-d8c5ca4c2cd4` used the same
immutable image. `profiler-version-final.log.txt` reports rocprofv3 1.1.0,
revision `97f5574fe2fdc7bef44fb01545347912ee9f1779`, ROCm 7.2.4, and
installed package `rocprofiler-sdk` version `1.1.0-93~24.04`.

## Trace disposition

The full trace in lease `43c3e415-e8e5-4918-ab42-b4d9c3dcab29` reported a GPU
hang and then stalled until the 1,200-second timeout. It contains 16 timestamp
swap warnings. The failed capture and partial CSV files remain available.
Issue #3039 repairs the delayed response to a live fault. It does not explain
or repair the GPU fault itself.

The unprofiled control completed 64 tokens in one cold completion of 12.667
seconds. This is a control result, not a benchmark. Profiled `--help` also
completed. The kernel-only control in lease
`25065496-5849-4a03-820b-df55045aa9eb` completed 64 tokens, but its log contains
62 timestamp swap warnings. This narrows the failure conditions without
establishing the cause.

The inspected SDK source at
[`profiling_time.hpp:82`](https://github.com/ROCm/rocm-systems/blob/97f5574fe2fdc7bef44fb01545347912ee9f1779/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/tracing/profiling_time.hpp#L82)
supports strict failure. Its default path swaps inverted timestamps and
adjusts timestamps outside the CPU bounds. This source inspection explains
the warning text and matches the separately captured installed SDK revision.
Do not derive kernel durations, time shares, or host idle bounds from these
timestamps.

The complete kernel-only CSV contains 85,737 dispatch rows, including 64
`ArgmaxK` and 4,096 `MoeSiluMulK` calls. These counts identify executed
kernels, not their timing contribution.

| Kernel metadata | Dispatches | Scratch_Size | VGPR_Count |
|---|---:|---:|---:|
| `KQuantGemmK<unsigned short, 2>`, Q6_K fallback | 2,056 | 272 | 120 |
| `KQuantGemmKCoopQ6K`, both output types | 568 | 0 | 72 |

Scratch and VGPR fields are kernel resource metadata. They do not measure
spill traffic or prove that the fallback is a bottleneck. Q6_K dispatch is
identified by the pinned local source in `src/vt/rocm/rocm_grouped_gemm.hip`.

## Switch measurement

The switch-only lease is `b7809cf0-eaf2-4581-acc8-6f64388f9a8e`. Its runner
comes from `dcb5351cd`, SHA256
`a3cb5f1d694bc35ce15995ca994fb4e40c9b7f06cf4445d70d729e140dfc1b47`.
All 12 legs completed. The controller reported success after 15 minutes
47 seconds; the client log contains an unexpected stream EOF. Independent
folding of raw stderr, stdout, clock samples, and exit codes reproduces all
12 recorded legs and all six paired summaries.

Each duration below is the median of three warm whole completions, including
prefill, after one discarded cold completion. The ratio is default seconds
divided by candidate seconds. Each candidate enables only the named switch.

| Switch | Pair | Default seconds | Candidate seconds | Ratio |
|---|---:|---:|---:|---:|
| `VT_ROCM_Q8K_BLOCK` | 0 | 12.001 | 11.303 | 1.061754 |
| `VT_ROCM_Q8K_BLOCK` | 1 | 12.101 | 11.329 | 1.068144 |
| `VT_ROCM_Q8K_BLOCK` | 2 | 12.116 | 11.351 | 1.067395 |
| `VT_ROCM_Q6K_SMALL_PRIVATE` | 0 | 12.129 | 13.585 | 0.892823 |
| `VT_ROCM_Q6K_SMALL_PRIVATE` | 1 | 12.128 | 13.556 | 0.894659 |
| `VT_ROCM_Q6K_SMALL_PRIVATE` | 2 | 12.136 | 13.603 | 0.892156 |

Median ratios are 1.067395 for Q8 and 0.892823 for Q6. These are diagnostic
observations, not accepted performance gains or regressions: `TOKEN_GATE=FAIL`
is carried, and warm output equality remains PENDING. The CLI emits cold text
only, whose bytes match across these legs; it emits no warm text or token IDs.
No default changes follow from these measurements.

Reproduce the pair results
from the repository root with the existing worker functions. This command
checks the worker pin before import and reads raw stderr, stdout, clocks, and
return codes before comparing with the recorded summaries.

```sh
python3 - <<'PY'
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
base = Path("docs/bench-evidence/strix-kernel-trace-3015-20260907")
path = Path("tools/bench/strix_kernel_trace/worker.py")
assert hashlib.sha256(path.read_bytes()).hexdigest() == "a3cb5f1d694bc35ce15995ca994fb4e40c9b7f06cf4445d70d729e140dfc1b47"
spec = importlib.util.spec_from_file_location("worker", path)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
legs = []
for switch in worker.SWITCHES:
    for pair in range(3):
        for arm in ("default", "candidate"):
            prefix = base / f"switches--{switch}-{pair}-{arm}"
            def capture(suffix):
                return Path(str(prefix) + "--" + suffix)
            runs = worker.parse_ours(capture("stderr.txt").read_text())
            with gzip.open(capture("clocks.jsonl.gz"), "rt") as stream:
                samples = [json.loads(line) for line in stream]
            command = json.loads(capture("command.json.txt").read_text())
            assert command["profiled"] is False
            active = [name for name in worker.SWITCHES if name + "=1" in command["argv"]]
            assert active == ([switch] if arm == "candidate" else [])
            legs.append(dict(switch=switch, pair=pair, arm=arm, runs=runs,
                             clocks=worker.fold_clocks(runs, samples),
                             output_sha256=worker.digest(capture("stdout.txt")),
                             returncode=json.loads(capture("exit.json.txt").read_text())["returncode"]))
recorded = json.loads((base / "switches--legs.json.txt").read_text())
key = lambda leg: (leg["switch"], leg["pair"], leg["arm"])
assert sorted(legs, key=key) == sorted(recorded, key=key)
result = worker.fold_pairs(legs)
assert result == json.loads((base / "switches--paired-result.json.txt").read_text())
print(json.dumps(result, indent=2))
print("All 12 raw legs and six paired summaries match")
PY
```

The five pre-switch samples report zero percent busy, 600 MHz, and
163,110,912 bytes of used VRAM. The accompanying attempts to inspect old
container processes failed. These samples do not establish successful
process inspection or occupancy. GPU busy percentage is sampled activity,
not occupancy or a quantitative bound on host idle time.

## Verify capture integrity

The combined runtime at `dcb5351cd` passed 37 focused tests and an independent
review that detected all 59 scratch mutations. Reviewer preflight exited 0
with no failed checks and five missing-argument skips: ARM ISA, CPU ISA,
CUDA fat-gencode, PR size, and Triton multiarch. This is not a wholly green
preflight. Raw reviewer logs are the three
`verification--strix-combined-review-*.log.txt` captures. The independent
operator focused run is `verification--strix-3015-operator-final-focused.log.txt`.
These verification captures came from the identically named files under
`/tmp`, without the packaging prefix or `.txt` suffix. They are covered by
the packaged checksum manifest, rather than the NAS source manifest.
The operator's final preflight also exited 0 with no failed checks and the same
five skips, recorded in
`verification--strix-3015-operator-final-preflight.log.txt`.

From this folder, verify the packaged bytes:

```sh
sha256sum -c SHA256SUMS.txt
```

Check that every copied or decompressed capture matches its source checksum:

```sh
python3 - <<'PY'
import gzip
import hashlib
from pathlib import Path
for line in Path("SOURCE_SHA256SUMS.txt").read_text().splitlines():
    expected, relative = line.split("  ", 1)
    name = relative.replace("/", "--")
    compressed = relative.endswith((".csv", ".jsonl"))
    path = Path(name + (".gz" if compressed else ".txt"))
    data = gzip.decompress(path.read_bytes()) if compressed else path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected, relative
print("All source capture hashes match")
PY
```

Read dispatch counts and resource metadata without using timestamp columns:

```sh
python3 - <<'PY'
import collections
import csv
import gzip
from pathlib import Path
path = Path("diag-kernel-control-asaj36ta--trace--e5367aefafa8--1_kernel_trace.csv.gz")
with gzip.open(path, "rt") as stream:
    rows = list(csv.DictReader(stream))
print("Dispatch rows:", len(rows))
for name in ("ArgmaxK", "MoeSiluMulK", "KQuantGemmK<unsigned short, 2>", "KQuantGemmKCoopQ6K"):
    selected = [row for row in rows if name in row["Kernel_Name"]]
    resources = collections.Counter((row["Scratch_Size"], row["VGPR_Count"]) for row in selected)
    print(name, len(selected), "(Scratch_Size, VGPR_Count):", dict(resources))
PY
```

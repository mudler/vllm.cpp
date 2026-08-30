# The llama.cpp ARM of the `MODEL-MM-QWEN4-EXP` speed gate: artifact, build and harness, 29 August 2026

Preparation evidence for G4 in
[`../../.agents/specs/qwen4-exp-flash-next.md`](../../.agents/specs/qwen4-exp-flash-next.md),
and the RUN half of gateability that
[#2060](https://github.com/mudler/vllm.cpp/issues/2060) owes for the
[`llama-cpp-qwen4exp`](../../.agents/oracles/llama-cpp-qwen4exp.md) oracle.

**No vllm.cpp arm ran and no ratio appears anywhere in this file.**
`ModelRegistry::Forward` still refuses `Qwen4ExpForConditionalGeneration` by name
(the `VT_CHECK(false, ...)` in `ForwardQwen4ExpForConditionalGeneration`,
`src/vllm/model_executor/models/qwen4_exp_registry.cpp:266`), so there is no
second arm to compare against and nothing here is a result. This file records
the denominator side being made ready, correct and provable, so that the A/B is
one lease away rather than one campaign away.

**Nothing here is a timing.** One prompt, greedy, no repetitions, no contention
control, no clock window. `AGENTS.md` admits `gateable = yes` on a build and a
run; it does not admit a number from the run that proves it.

## The developer's target, quoted

> we should be faster than llama.cpp

> especially at high concurrency

That is why the deliverable is a LADDER and not a point, and why the harness
below refuses to run a single rung.

## 1. The staged artifact inventory

Seven `unsloth/Qwen3.8-Flash-Next-GGUF` quants, staged on the house NAS
(`//192.168.68.102/Data`, mounted at `/mnt/nas_share` on the development box).

| Arm | Path under `/mnt/nas_share/checkpoints/` | Shards | Total bytes | GiB |
|---|---|---:|---:|---:|
| UD-IQ1_S | `qwen3.8-flash-next-ud-iq1_s/UD-IQ1_S` | 3 | 72,546,461,344 | 67.56 |
| UD-IQ1_M | `qwen3.8-flash-next-ud-iq1_m/UD-IQ1_M` | 3 | 74,538,755,776 | 69.42 |
| UD-Q2_K_XL | `qwen3.8-flash-next-ud-q2_k_xl/UD-Q2_K_XL` | 3 | 78,869,128,864 | 73.45 |
| UD-IQ3_XXS | `qwen3.8-flash-next-ud-iq3_xxs/UD-IQ3_XXS` | 3 | 81,961,823,936 | 76.33 |
| UD-Q3_K_XL | `qwen3.8-flash-next-ud-q3_k_xl/UD-Q3_K_XL` | 3 | 89,986,353,824 | 83.81 |
| UD-IQ4_XS | `qwen3.8-flash-next-ud-iq4_xs/UD-IQ4_XS` | 3 | 93,682,584,224 | 87.25 |
| UD-Q4_K_XL | `qwen3.8-flash-next-ud-q4_k_xl/UD-Q4_K_XL` | 4 | 111,334,654,784 | 103.69 |

Every arm carries a `SHA256SUMS.txt` beside it. The download is COMPLETE, which
is the fact that changed since 27 August: at that point `UD-IQ1_S` held 3.3 GiB
of 67.56 GiB and its first shard was a 10 MiB fragment, and the gateability
attempt refused on the missing shard. `UD-IQ1_S` now measures
72,546,461,344 bytes across three shards, byte-for-byte the size the spec
records from the Hub.

**The 10,946,624-byte first shard is genuine, not a fragment.** Every one of the
seven arms has a first shard of exactly that size and a different sha256, and
each arm's three or four shards sum to the published total. A size that repeats
across seven quantizations is the split boundary, not a truncated download.

### Which box can see them, measured rather than assumed

| Path | Development box | `dgx` / `thor` lease | `orin` |
|---|---|---|---|
| `/mnt/nas_share/checkpoints/...` | yes, the CIFS mount | **NO** | no |
| `/mnt/nas_share/rc/...` = `/workspace` | yes | yes | no |

`rc describe dgx:gpu0` states the mapping in the host's own usage sheet:
`/workspace` inside a job is `/usr/local/nas_share/rc` on the host, which is the
`rc/` SUBFOLDER of the `//192.168.68.102/Data` share. `checkpoints/` is a sibling
of `rc/`, not a child, so **a lease cannot reach the staged quants where they are
staged**. The same sheet states that the folder is shared between `dgx` and
`thor` and is NOT shared with `orin`, which has local storage only.

`UD-IQ1_S` was therefore COPIED (not moved -- other rows cite the original paths)
into `/mnt/nas_share/rc/q4exp-bench/UD-IQ1_S/`, where a lease reaches it as
`/workspace/q4exp-bench/UD-IQ1_S/`. Copy rate measured at about 72 MB/s over the
share; the three shards match the source sizes exactly.

### The first LOCAL sha256 of this artifact

The spec records the three `UD-IQ1_S` hashes as Hub `lfs.oid` values and notes
that a local sha256 was owed. Computed here on the staged copy, which verifies
the download and the copy in one pass:

```sh
cd /mnt/nas_share/rc/q4exp-bench/UD-IQ1_S && sha256sum Qwen3.8-Flash-Next-UD-IQ1_S-0000*.gguf
```

| Shard | Local sha256, computed 29 August 2026 | Agrees with the Hub `lfs.oid` the spec records |
|---|---|---|
| `-00001-of-00003` | `88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd` | yes |
| `-00002-of-00003` | `3a62e35bbf9add4733bd1438ebd3a67649d5edd6cb0e72bb78e33c913992b2b6` | yes |
| `-00003-of-00003` | `0e25ceaeb89b8a80aa973c6c0c7448943682f7408c2855b2ebd016b7643a861a` | yes |

Three for three. That is worth stating rather than assuming, because
`unsloth/Qwen3.8-Flash-Next-GGUF` is a PUBLIC repository and the Hub fabricates
`lfs.oid` for GATED ones; a recorded oid is only a hash where the repository is
open, and this one is. The `SHA256SUMS.txt` staged beside each arm carries the
same three values, so it is now a verified transcription rather than an
unchecked one.

## 2. The build

CUDA, for GB10, inside an `rc` lease on `dgx:gpu0`. The 2026-08-27 build was
CPU-only on the x86_64 development host; this one is the arm the A/B will
actually run.

| Fact | Value |
|---|---|
| where | `rc run -d dgx:gpu0 --max-runtime 90m`, worker `rc-worker-4b8lj`, root in an Ubuntu 24.04 container |
| never | no `ssh` to `dgx.casa`; the lease is the only path to a fleet device |
| GPU | `NVIDIA GB10`, compute capability `12.1`, driver `580.173.02`, `memory.total` reads `[N/A]` |
| source | fresh `git init` into an EMPTY directory, `git fetch --depth 1 origin <pin>`, `git checkout FETCH_HEAD` |
| `git rev-parse HEAD` | `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`, asserted EQUAL to the pin by the script |
| `git status --porcelain` | **0 bytes** |
| the pin knows the architecture | `git ls-files \| grep -i qwen4exp` names `conversion/qwen4exp.py` and `src/models/qwen4exp.cpp` |
| toolkit | `cuda-toolkit-13-0`, installed in the job; `nvcc` release 13.0, V13.0.88 |
| host compiler | GNU 13.3.0 |
| version string | `0.3.0-dev (build 1, commit 035e227)`, ggml `0.22.0` |

```sh
cmake -S "$SRC" -B "$SRC/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$SRC/build" -j 4 --target llama-server llama-cli llama-bench
```

**`121` becomes `121a` and llama.cpp does that itself.** Configure prints
`Replacing 121 in CMAKE_CUDA_ARCHITECTURES with 121a`, so the recipe asks for
`121` and the artifact is the architecture-specific `sm_121a` the GB10 wants.
That is worth recording rather than inferring, because a reader who wants to
reproduce this will otherwise wonder which of the two was built.

### It is a CUDA build, proven three ways

- **`ldd` resolves the CUDA runtime**: `libcudart.so.13`, `libcublas.so.13`,
  `libcublasLt.so.13`, `libcuda.so.1`, and `libggml-cuda.so.0` at 60,178,552
  bytes.
- **The cubin histogram is uniform**: 142 `.cu.o` objects, `cuobjdump --list-elf`
  over all 142, **142 `sm_121a`, objects scanned 142**. No object escaped the
  scan, which is the check an empty histogram would otherwise pass silently.
  The suffix is load-bearing and an earlier draft of this file dropped it,
  transcribing the histogram as `sm_121`. Read off `cubin.log` rather than
  recalled — `grep -o 'sm_[0-9]*a\?' cubin.log | sort | uniq -c` returns
  `142 sm_121a` and nothing else — the histogram is a SECOND, independent
  witness that the `121` in the recipe became `121a` in the artifact. The
  configure log says llama.cpp intended it; the cubins say it happened. The
  mistaken transcription was in this project's own disfavour: it made the
  stronger of those two legs say nothing.
- **`libllama.so` carries 68 `qwen4exp` strings.** The architecture links; it is
  not merely present in source.

Build time about 14 minutes at `-j 4`; source plus build tree 439 MiB; `/tmp` on
the worker never dropped below 1,969 GiB.

### Where the artifact is, and one thing that will bite the next reader

`/workspace/q4exp-bench/llamacpp-pr27742-cuda/bin`, with `BUILD-RECORD.txt`
beside it carrying the pin, the porcelain assertion, the flags and a sha256 for
every file.

| Binary | sha256 |
|---|---|
| `llama-server` | `61dafdea9dd3ac9e6bf15311f514b41f995640dcc91b81fb16779c834881ba1a` |
| `llama-cli` | `a926cd915720f094f8614d8157d79802bc2357c3284b16c8801c385a84a5ad3e` |
| `llama-bench` | `ba819d978e29f881c64a628e434d989e002a9f562c0920a4de70a47d72b53874` |
| `libggml-cuda.so.0.22.0` | `402286675f15ac1870687f1748c3d8ed20edaa74e6a8aa95bbd82df653e9fd2b` |
| `libllama.so.0.3.0` | `17fdb43313d06018214b8bb4bf079cdbfe863e6fb713b362a4d3ccd9e6489457` |

`GGML_NATIVE=ON`, so these hashes identify a build on THIS host and never a
tree.

**`/workspace` is mounted `file_mode=0664`, so a binary copied there arrives
`-rw-rw-r--` and cannot be executed, whatever mode `cp` asked for.** The
listing above reads `-rw-rw-r-- llama-server`. `chmod` on that mount does not
stick either. Anything that runs these binaries has to stage them onto the
worker's own `/tmp` first, which is what the decode proof does, and which also
keeps the hot path off a network filesystem.

## 3. The proof of decode

**It loads, and it generates coherent text.** That is the RUN half of
gateability, on the GB10 arm, against the staged artifact, through the same
`llama-server` OpenAI endpoint the ladder drives.

```sh
llama-server -m .../Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf \
  --alias gate --host 127.0.0.1 --port 8077 \
  -ngl 99 -c 4096 -np 1 --cont-batching --flash-attn on

curl -s http://127.0.0.1:8077/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"gate","prompt":"The capital of France is",
       "max_tokens":64,"temperature":0,"seed":0}'
```

Server log:

```text
srv load_model: loading model '/workspace/q4exp-bench/UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf'
cmn         init: llama threadpool init, n_threads = 20
srv load_model: initializing, n_slots = 1, n_ctx_slot = 4096, kv_unified = 'false'
srv llama_server: model loaded
srv llama_server: listening on http://127.0.0.1:8077
```

The completion, first sentence, verbatim:

```text
" Paris. Given a list of countries and their capitals, answer the question
'What is the capital of France?\n\n<think>\nThe user is asking me to answer
the question \"What is the capital of France?\" ..."
```

`"system_fingerprint":"b1-035e227"`, `"finish_reason":"length"`,
`completion_tokens: 64`, `prompt_tokens: 5`. `/props` reports
`"model_ftype":"IQ1_S - 1.5625 bpw"` and `"build_info":"b1-035e227"`.

**`b1-035e227` is the pin, reported by the running server**, so the binary that
answered is provably the one built at the pinned object, and not something else
that happened to be on the box.

| Fact | Value |
|---|---|
| load time to first `/health` 200 | **990 s**, from `/workspace` (CIFS) |
| box memory | 119 GiB total, 114 available before the load |
| prompt eval | 5 tokens in 757.82 ms |
| eval | 64 tokens in 2487.45 ms |
| `graphs reused` | 0 |

**Those four numbers are NOT a measurement and may not be quoted as one.** One
prompt, one repetition, five prompt tokens, no clock window, no contention
control, a 4,096-token context and one slot. They are recorded because a run
that produced no numbers at all is indistinguishable from a run that did not
happen. `.agents/benchmarking.md` says what a quotable number needs, and this
has none of it.

### Two things this run found, both of which would have cost a ladder lease

**`/workspace` strips the execute bit.** The first attempt refused at exec:
`error while loading shared libraries: libllama-common.so.0`, rc=127, on both
binaries. Two causes stacked. The mount is `file_mode=0664`, so a copied binary
is not executable and `chmod` does not stick; and the build's copy-out used
`find -type f`, which skips symlinks, so every SONAME link was left behind while
the loader asks for the SONAME. The fix is to stage onto the worker's own `/tmp`,
rebuild the `lib*.so.MAJOR` links there, and ASSERT with `ldd` that nothing reads
`not found` BEFORE the model load. That assertion now runs first and reported
`llama-cli: 0 unresolved`, `llama-server: 0 unresolved`. Discovering this after a
16-minute load, inside a ladder lease, is the expensive version of the same
finding.

**`llama-cli` at this revision rejects `-no-cnv`**: `error: invalid argument`.
The flag was renamed upstream. It does not touch the ladder, which drives
`llama-server`, and the server path is the one this proof establishes. The
`llama-cli` leg of the proof script is owed a corrected flag.

### One fidelity fact this run establishes, and a comparison must carry

`/props` reports `"modalities":{"vision":false,"video":false,"audio":false}`.
**This denominator serves TEXT ONLY.** `MODEL-MM-QWEN4-EXP` is a multimodal port
and its scope includes the image and video path. An arm that runs a vision tower
does strictly more work per request than this one, so a ratio taken across that
difference measures a configuration difference and reads as a performance one.
State which side ran what, exactly as [`llama-cpp.md`](../../.agents/oracles/llama-cpp.md)
already requires for the `blk.64` tensors the stock pin silently ignores.

## 4. The harness

`scripts/qwen4exp-llamacpp-ladder.sh`, with
`tests/scripts/test_qwen4exp_llamacpp_ladder.py` as its proof.

**The ladder is copied, not chosen.** `POINTS`, `INPUT_LEN`, `OUTPUT_LEN` and the
three repetitions are `tools/bench/online_gate.py`'s published grid, the one
rendered in [`../benchmarks/vllm-online-serving.md`](../benchmarks/vllm-online-serving.md),
so a cell of the llama.cpp table can be read beside a cell of ours:

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| prompts | 6 | 6 | 12 | 24 | 96 | 192 |

1,024 in / 128 out, three repetitions, medians. The spec asks for
`c = 1, 4, 8, 16, 32` "at minimum"; this is that set plus the `c=2` the published
grid carries. `test_ladder_matches_the_published_online_serving_grid` reads
`POINTS` out of `online_gate.py` and asserts the two agree, so the tables cannot
drift apart silently.

**One instrument, not two.** The harness times nothing itself. Every timed
request is issued by the pinned `vllm bench serve`, pointed at `llama-server`'s
OpenAI-compatible endpoint. `.agents/benchmarking.md` warns that two ratio sets
that disagree are frequently two harnesses; the cheapest way not to have that
argument is to have one client.

**"With the same flags" is now ASSERTED, and the first version of this file was
wrong to state it.** The flag SEQUENCE the harness issues is compared against the
one `online_gate.build_client_command` builds, by
`test_client_flag_list_matches_build_client_command`, which reads both. It found
two real divergences, and both are repaired:

| Flag | Our arm (`build_client_command`) | What the harness did | Now |
|---|---|---|---|
| `--num-warmups` | `OnlineRun.num_warmups`, which defaults to the **concurrency** | hardcoded `0` | the concurrency |
| `--dataset-path` | one **disjoint** partition per (concurrency, repetition), which `prepare_corpus_views` refuses to let overlap | ONE file for all 18 legs | `c<C>-r<R>.jsonl`, the same partition set by the same names |

Neither is cosmetic. At c=32 our arm would have taken 32 warmups and llama.cpp
none, and this repository's own
[`../benchmarks/vllm-online-serving.md`](../benchmarks/vllm-online-serving.md)
records warmup as exactly where two engines differ most. And with
`--disable-shuffle` a shared corpus replays the same prompts every repetition,
warming `llama-server`'s slot prefix cache in a way our arm never sees, while
G4's wording is "identical artifact, **prompts**, token counts, sampling and
concurrency". `test_client_num_warmups_is_the_concurrency_exactly_as_our_arm_takes_it`
and `test_client_each_leg_consumes_its_own_corpus_partition` read what the client
was actually handed on all 18 legs rather than trusting the invocation text, and
`test_ladder_repetitions_match_online_gate_repetitions` pins the third grid axis,
which previously had no assertion at all.

**It cannot pin a clock.** Inside an `rc` lease `nvidia-smi -lgc` returns
`LGC_RC=4` even as root (#1354), so the SM clock here is SAMPLED through
`tools/bench/gpu_clock_state.py` and never pinned. A pin outlives the lease and
reprices the next holder's measurement, so the harness has no code path that
could leave one, and a test asserts the absence.

### The five failure modes, each proven rather than asserted

Each row names the guard, the test that fires it, and the MUTATION that proves
the test can go red. `AGENTS.md` asks a fresh reviewer to mutate each claimed
guarantee rather than read it; these are run in a scratch copy and the tree is
restored byte-for-byte, verified by sha256, after every one.

| # | Failure mode | Guard | Test | Mutation | Verdict |
|---|---|---|---|---|---|
| 1 | an arithmetic expansion breaks under `set -u` | every identifier used bare inside `$(( ))` is assigned in one defaults block | `test_set_u_no_unbound_variable_in_an_empty_environment` (dynamic, `env -i`), `test_set_u_static_every_arithmetic_operand_is_defaulted` (static scan) | `test_set_u_mutation_removing_one_default_goes_red` deletes the `MEM_FRACTION` default | both halves go red |
| 2 | the KV/context sizing exceeds the box | a STATIC weights+KV check against the box's own memory, then a post-launch check that takes the engine's own reported size if it prints one, the supplied `KV_BYTES_PER_TOKEN` if it does not, and REFUSES if neither exists | `test_kv_refuses_when_weights_plus_kv_exceed_the_box`, `test_kv_refuses_a_context_that_cannot_hold_the_slots`, `test_kv_sizes_against_the_boxs_actual_memory_not_a_constant`, `test_kv_the_binding_check_is_the_servers_own_reported_size`, `test_kv_refuses_when_the_server_reports_no_kv_size`, `test_kv_a_supplied_per_token_cost_binds_when_the_server_is_silent`, `test_kv_a_supplied_per_token_cost_still_refuses_over_budget` | M2, M2b, M2c delete each refusal in turn; `test_kv_mutation_restoring_the_fail_open_default_goes_red` restores the zero-term fall-through | RED, RED, RED, RED |
| 3 | a missing artifact silently measures nothing | the shard SET is derived from the `-00001-of-00003` name, and each shard must exist and be non-empty | `test_artifact_a_missing_shard_refuses_by_name`, `test_artifact_a_zero_byte_shard_refuses`, `test_artifact_shard_set_comes_from_the_name_not_from_a_glob` | M1 deletes the shard guard | RED |
| 4 | a missing wrapper makes the wrapped command not run | `command -v` on every wrapper BEFORE the lock, then the wrapper's OWN output asserted after each leg | `test_wrapper_absent_time_refuses_before_the_lock`, `test_wrapper_absent_from_path_refuses`, `test_wrapper_present_but_silent_fails_the_leg`, `test_wrapper_a_client_that_writes_no_result_fails_the_leg` | M3, M3b, M3c | RED, RED, RED |
| 5 | the script hangs while holding the GPU | `flock -w`, a bounded readiness poll, dead-server detection, a leg `timeout`, and a teardown trap on every exit path | `test_hang_lock_contention_is_bounded`, `test_hang_server_that_never_answers_health_times_out`, `test_hang_server_that_dies_is_noticed_immediately`, `test_hang_a_stuck_leg_is_killed_by_the_leg_timeout`, `test_hang_the_server_is_reaped_on_every_exit_path`, `test_hang_no_clock_is_ever_pinned` | M4, M5, M6, M7 | RED, RED, RED, RED |

**11 mutations went red with none unarmed when the sweep ran on 29 August.** The
sweep restored the script after each one and asserted the restored sha256 equalled
the one taken before it: `a361ffb5a31b004949344f6877cb56106af0b0f954848b74a5cc9fa7d072b215`
both times.

**Only TWO of those eleven are committed as tests, and that is a recorded gap
rather than a silence.** The sweep DRIVER is not in this tree, so nine of the
eleven mutations are a claim about a run that happened once on one machine and
cannot be re-executed by a reader. The two that are executable are
`test_set_u_mutation_removing_one_default_goes_red` and, added by this repair,
`test_kv_mutation_restoring_the_fail_open_default_goes_red`. Bringing the driver
in is owed; see [What is still owed](#5-what-is-still-owed). A mutation nobody
can re-run is evidence with a shelf life.

### Two of these were found rather than confirmed

**The `set -u` guard was RED on the first run, on the real script.**
`GPU_LOCK=${GPU_LOCK:-$HOME/gpu.lock}` aborts with `HOME: unbound variable` in
any environment that does not export `HOME` -- an `rc` job, a cron entry, a
reproduction under `env -i` -- and it aborts on line one of the defaults block,
before a single guard can name what is wrong. It is now
`${GPU_LOCK:-${HOME:-/tmp}/gpu.lock}` and the same command exits 10 with a named
refusal.

**The mutation sweep found a guard that was not armed.** M3c deleted the
result-file assertion and `test_control_execute_runs_the_whole_ladder` stayed
GREEN, because the control asserts the files EXIST and the stub client writes
them either way. Only a client that exits 0 and writes nothing discriminates,
which is exactly what a full disk or a mis-parsed `--result-dir` looks like in
the field. `test_wrapper_a_client_that_writes_no_result_fails_the_leg` was added
for it and M3c is red against that one. A mutation that leaves a test green is
the sweep working, not the sweep failing.

### One incident, recorded because it is the trap this repository already knows

A mutation sweep was killed mid-run in this session and left the tree MUTATED:
`if [[ ${waited} -ge ${READY_TIMEOUT_SECONDS} ]]` had become `if false`, which
turns the bounded readiness poll into an unbounded wait on a held lock. The
mutant then survived a later edit and shipped into a saved copy, and the only
reason it was caught is that a test ran the real script against a server that
never answers and timed out. The driver now restores on every exit path and
writes the pristine text to a sidecar, and the restored sha256 is compared
against the one taken before the sweep.

### The KV guard was failing open on the real denominator

The most expensive defect this harness carried, found by a fresh review before
it ever held a lease, and measured against this change's own production capture
rather than reasoned about.

`assert_reported_kv_fits` extracted `KV self size = N` from the server log and,
finding nothing, did `[[ -n ${kv_mib} ]] || kv_mib=0`. **`llama-server` at this
pin prints no such line.** `decode-proof/llama-server.log` is the COMPLETE server
output — the decode proof redirects both streams into it with no filter — and it
is 1,862 bytes carrying no `KV self size`, no `llama_kv_cache:` sizing line and
no allocation summary of any kind. Sixteen minutes pass between `load_model:` and
`threadpool init` with nothing printed in between. `/props` carries no KV bytes
either; its only sizing fields are `n_ctx = 4096` and `total_slots = 1`.

Meanwhile `KV_BYTES_PER_TOKEN` defaulted to `0`, which made the static budget
check weights-only by design and said so. **So on the real box neither check had
a KV term**, while the ladder configures `CTX_TOTAL=49152` across 32 slots
against a 67.5 GiB model on a 119 GiB unified-memory device that reboots rather
than swaps. That is the "128 GiB on a 119 GB box" family the guard was written
to prevent, wearing the guard's own name.

The fixture disagreed with the denominator, which is why no test caught it: every
server stub in `test_qwen4exp_llamacpp_ladder.py` emitted the line, so the suite
never fed the guard a log without one.

**What was changed.** The post-launch check now prefers the engine's own reported
size if a pin ever prints one, falls back to a supplied `KV_BYTES_PER_TOKEN` and
NAMES the fallback in its output, and **refuses (`E_KV_UNREPORTED`, 21) when
neither exists.** The fixture server is now silent about KV exactly as the real
one is, so every execute-path test feeds the guard a log with no KV line and
passes only because the environment supplies a per-token cost;
`test_kv_refuses_when_the_server_reports_no_kv_size` removes it and asserts the
refusal with no leg written, and
`test_kv_mutation_restoring_the_fail_open_default_goes_red` puts the zero-term
fall-through back and asserts the same scenario stops refusing.

**Nothing else binds, and this is stated rather than papered over.** The real
server was read for an alternative and has none: the log carries no sizing line,
`/props` carries no bytes, and `/metrics` publishes a KV *usage ratio* rather
than a size. A post-launch RSS check was considered and REJECTED, because whether
GB10's unified `cudaMalloc` allocations appear in `/proc/<pid>/smaps_rollup` is
exactly the kind of thing that cannot be settled without a lease, and a guard
resting on an unverified assumption is the defect again. **The ladder therefore
must not run until a leased load supplies `KV_BYTES_PER_TOKEN`.** No lease was
taken for this repair.

## 4b. Reproducibility: what came into the tree, and what is owed

Three gaps were named by the fresh review. Each was decided rather than deferred
silently.

| Gap | Decision | Why |
|---|---|---|
| the build and decode-proof scripts lived on the NAS, not in the tree | **brought in**, as `scripts/qwen4exp-llamacpp-build-cuda.sh` and `scripts/qwen4exp-llamacpp-decode-proof.sh` | they are ~7 KB each and they are the recipes for the two jobs that flipped an oracle to `gateable = yes`. A recipe that lives only beside its own output is a recipe nobody can re-run, and `ArmScriptsInTheTreeTest` now pins both to this record |
| the toolchain was recorded but NOT pinned: `apt-get install -y cuda-toolkit-13-0` is unversioned, so a rerun gets whatever apt serves | **made fail-closed in the same change** | the channel cannot be pinned from here, but the OUTCOME can be asserted. `EXPECT_NVCC=13.0.88` — the version the evidence above records — is compared against what `nvcc --version` reports and the build exits 89 on a mismatch, naming both. Drift is now visible instead of silent, which is the property that was missing |
| only 1 of 11 mutations was committed as a test, because the sweep driver is not in the tree | **2 of 11 now; the driver is OWED and has an owner** | the second is this repair's own KV mutation. Committing a general sweep driver is its own unit of work with its own review, and doing it inside a repair branch would bundle unrelated work. It is listed under "What is still owed" and owned by `MODEL-MM-QWEN4-EXP` |

Two smaller repairs rode along, both in the direction of asserting rather than
transcribing:

- **The cubin histogram lost the architecture suffix.** The build script ran
  `grep -o 'sm_[0-9]*'`, a pattern that CANNOT print the `a`, so 142 `sm_121a`
  cubins rendered as `142 sm_121` and that reading went into the oracle record.
  The pattern is now `sm_[0-9]*a\?` and
  `test_build_script_cubin_scan_can_see_the_architecture_suffix` asserts it
  against a real cubin line, with the truncated pattern kept inline as the
  mutation that demonstrates the defect.
- **The decode proof did not assert which build answered.** It printed the staged
  binaries' sha256 under a comment saying they "must equal BUILD-RECORD.txt" and
  compared nothing. It now reads the pin out of `BUILD-RECORD.txt`, asserts it
  equals the object this proof is about, and hard-exits on any binary whose
  staged hash differs from the recorded one.

## 5. What is still owed

- The ladder itself. It needs `dgx:gpu0` for long enough to walk six rungs three
  times, and that lease is not taken here.
- The vllm.cpp arm. There is nothing to compare until the forward exists.
- The `llama-cli` leg of the proof script, which needs the renamed conversation
  flag. `llama-server` is the path the ladder uses and it is proven.
- A tokenizer snapshot and a frozen 1,024-token corpus partition for the client,
  both of which the harness refuses without.
- **A `KV_BYTES_PER_TOKEN` measured from a real load. The ladder now REFUSES
  without one, and this is a hard blocker rather than a nicety.** See
  [the KV guard was failing open](#the-kv-guard-was-failing-open-on-the-real-denominator)
  below: `llama-server` at this pin reports no KV size at all, so the sentence
  that used to stand here -- "the measured post-launch check binds either way" --
  was false on the only server this harness will ever face. Only a leased load
  can supply the number, and no lease was taken for this repair.
- The fidelity statement every comparison against this oracle must carry: #27742
  is mask-only, so a LONG-CONTEXT decode win over it is partly their mask
  approach and not purely our gather. Short-context and high-concurrency cells,
  which is where the developer's target lives, are unaffected.
- **A committed mutation-sweep driver.** Nine of the eleven mutations recorded
  above are not re-executable from this tree. Owned by `MODEL-MM-QWEN4-EXP`.
- **A toolchain that is pinned rather than asserted.** `EXPECT_NVCC` makes drift
  refuse; it does not make apt serve one version. Owned by
  `MODEL-MM-QWEN4-EXP`.

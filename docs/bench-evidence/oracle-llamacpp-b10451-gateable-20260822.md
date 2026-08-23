# llama.cpp `b10451` builds and runs, 2026-08-22

Gateability evidence for the `llama-cpp` secondary oracle, recorded by row
`ORACLE-LLAMACPP-GATEABLE` for
[#857](https://github.com/mudler/vllm.cpp/issues/857). `AGENTS.md`
§"Measure gateability" admits `gateable = yes` only after an oracle
"demonstrably builds and runs the model". This file is that demonstration.

**This is not a floor and not a comparison.** No vllm.cpp arm ran. The timings
in §"Timings, and why they bind nothing" are one repetition on an unpinned
clock and must not be quoted as a measurement.
[#1003](https://github.com/mudler/vllm.cpp/issues/1003) still owes every
llama.cpp floor in this tree.

## What ran

One `rc run` job on `thor:gpu0`, job `d96c2867-4344-4064-84e3-d3a04a1b1925`,
worker `rc-worker-kk96r`, 2026-08-22T21:18:32Z to 21:34:10Z. The job script is
archived at `/mnt/nas_share/rc/oracle857/job/job.sh` and its output at
`/mnt/nas_share/rc/oracle857/out/thor-857-20260822T210223Z/`.

`AGENTS.md` §"Work on a GPU happens inside a lease" makes the lease the only
path to a fleet device. Nothing reached the box by `ssh`, and `rc devices`
reported `thor:gpu0` `ready` again after the job, so the resource was verified
returned rather than assumed.

### Why `thor:gpu0` and not `dgx:gpu0`

#857 asks for dgx.casa and `.env` names it `GATE_HOST`. At claim time
`dgx:gpu0` was held by another session's clock experiment with four more jobs
queued behind it, and queueing behind that is correct rather than contending
with it. The build and the run use no GPU and no CUDA toolkit, so the device is
a CPU and a filesystem. `thor:gpu0` is a fleet device reached by the same
lease, it shares the same `/workspace` folder as dgx, and it has 14 cores and
122 GB against a 17 GB artifact. `orin:gpu0` is refused for two reasons: its
`/workspace` is local disk that does not hold the artifact, and 32 GB of RAM
leaves no headroom over a 17 GB model.

Gateability is a property of the pin, not of a host. A floor is host-specific,
and #1003 owes one per host.

## Measured identity

Every value below was asserted inside the job and the job fails on a mismatch.

```text
pin                        = 10bf611e533d81f739128304991c5e133c6aebd8
pin_label                  = b10451
src_head                   = 10bf611e533d81f739128304991c5e133c6aebd8
tag_commit (b10451^{commit}) = 10bf611e533d81f739128304991c5e133c6aebd8
git_status_porcelain       = EMPTY (0 bytes)
llama_completion_sha256    = f8cb9a223d130dfc1cc347ce5cfd8ff1512b4da80d37f593a81074ef6a34f8a3
llama_bench_sha256         = 9a0db394d35170be3284fc7482f6c6c1e337ff53067e986e42e2524109728d52
gguf_sha256                = 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
host                       = rc-worker-kk96r (thor:gpu0)
nproc                      = 14
```

The binary reports its own provenance, independent of the assertion:

```text
version: 0.1.0-dev (build 1, commit 10bf611)
built with GNU 13.3.0 for Linux aarch64
```

### The label and the object agree, on the remote

This is the check the retired `pin_label = b9892` would have failed, and #857
exists because a label and an object disagreed. The job fetched both from
`https://github.com/ggml-org/llama.cpp` rather than reading a developer's
clone, which is the only form of the check a grafted local clone cannot
corrupt:

```text
 * branch    10bf611e533d81f739128304991c5e133c6aebd8 -> FETCH_HEAD    fetch_commit_rc=0
 * [new tag] b10451 -> b10451                                          fetch_tag_rc=0
```

Confirmed a second way from the devbox, with the retired object as the control:

| Call | Result |
|---|---|
| `gh api repos/ggml-org/llama.cpp/git/ref/tags/b10451` | `-> commit 10bf611e533d81f739128304991c5e133c6aebd8` |
| `gh api repos/ggml-org/llama.cpp/commits/10bf611e5...` | 200, `llama : check LoRA tensor data is within file bounds (#27056)` |
| `gh api repos/ggml-org/llama.cpp/commits/237ad9b96...` | HTTP 422, `No commit found for SHA` |

`git describe --tags 10bf611e5` returns `b10451` with no `-N-g` suffix, and
`git branch -r --contains 10bf611e5` lists `origin/master`. The pin is stock
upstream at an exact release tag, not a fork object.

### The artifact is the recorded one

A remote hash is not admissible, because HuggingFace's `lfs.oid` is fabricated
for gated repositories. The artifact was verified by parsing it and by hashing
the bytes the run actually read.

| Check | Result |
|---|---|
| GGUF magic and version | `GGUF`, version 3 |
| Tensor count | 866 |
| `general.architecture` | `qwen35`, registered at `src/llama-arch.cpp:41` at the pin |
| Alignment | 32 |
| Computed data end vs file size | `17106775008 == 17106775008` |
| sha256 of the NAS source | `7e78da5d...c6fe169` |
| sha256 of the worker-local copy the run read | `7e78da5d...c6fe169` |

The GGUF is `Qwen3.8-27B-Q4_K_M.gguf` from `unsloth/Qwen3.8-27B-GGUF` @
`fe1e2a23d973adb629709749dc4f6756df66ef10`, as `docs/USAGE.md` records it.

llama.cpp mmaps the GGUF, and a CIFS mmap is not a run surface, so the job
copied the artifact to the worker's local disk and re-hashed it there before
running. The hash chain therefore covers the exact bytes that fed the model.

## Build recipe

The worker is Ubuntu 24.04 running as root: gcc 13.3.0, cmake 3.28.3, ninja
1.11.1, git 2.43.0. Nothing was installed, and no CUDA toolkit is needed.

```sh
git init -q . && git remote add origin https://github.com/ggml-org/llama.cpp
git fetch --depth 1 origin 10bf611e533d81f739128304991c5e133c6aebd8
git checkout -q FETCH_HEAD
git fetch --depth 1 origin refs/tags/b10451:refs/tags/b10451

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF

cmake --build "$BUILD" -j 4 --target llama-completion llama-bench
```

`-j 4` is deliberate. Unconstrained parallelism has OOM-rebooted a fleet box.

### The binary is `llama-completion`, and this is not the recipe a reader remembers

At `b10451` `llama-cli` lives in `tools/cli`, its `add_subdirectory` sits inside
`if (LLAMA_BUILD_SERVER)` in `tools/CMakeLists.txt`, it links
`llama-server-impl`, it sets `params.verbosity = LOG_LEVEL_ERROR`, and it no
longer calls `common_perf_print`. It is a chat client. The one-shot completion
binary is `llama-completion` in `tools/completion`, which links `llama-common`
and `llama` only and sits outside the server guard.

A recipe carried over from an older release either produces no binary with
`LLAMA_BUILD_SERVER=OFF`, or produces a chat client whose output a postcondition
cannot read. Read `tools/CMakeLists.txt` at the pin before reusing this.

### Native flags

`GGML_NATIVE=ON`, so the binary is tuned to the box and its sha256 identifies a
build on a named host, never a tree. A user gets a native build, so a floor must
be one. cmake detected:

```text
-- ARM detected flags: -march=armv8-a+crc+lse+rcpc+rdma+dotprod+fp16fml+sb+ssbs
   +sve2-sm4+sve2-aes+sve2-sha3+sve2-bitperm+i8mm+bf16+flagm+pauth
```

and the run reported `NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 |
SVE = 1 | DOTPROD = 1 | SVE_CNT = 16 | OPENMP = 1 | REPACK = 1`. The k-quant
fast path is live, `i8mm` included.

## Run recipe and output

```sh
llama-completion -m Qwen3.8-27B-Q4_K_M.gguf \
    -p 'The capital city of France is' -n 48 -c 512 -t 14 \
    --temp 0 --top-k 1 --seed 1234 -no-cnv --no-warmup
```

`run_rc=0`, and stdout, verbatim and complete:

```text
The capital city of France is Paris.
The capital city of Germany is Berlin.
The capital city of Italy is Rome.
The capital city of Spain is Madrid.
The capital city of Portugal is Lisbon.
The capital city of Greece is Athens.
```

The text is coherent and factually correct on all six capitals, so the model was
loaded and decoded rather than merely opened. Three postconditions gated this
rather than the exit code alone: `stdout_bytes` 228 exceeds the 29-byte prompt,
the combined output contains `common_perf_print`, and the binary is an
executable file on disk. An exit code cannot tell a load that never decoded from
a run, and a missing wrapper makes a wrapped command silently not run.

## `b10451` ignores the embedded MTP block, and a quant-matched comparison must account for it

The loader emitted 15 warnings, all for `blk.64`:

```text
W model has unused tensor blk.64.attn_norm.weight (size = 20480 bytes) -- ignoring
...
W model has unused tensor blk.64.nextn.eh_proj.weight (size = 55705600 bytes) -- ignoring
W model has unused tensor blk.64.nextn.shared_head_norm.weight (size = 20480 bytes) -- ignoring
```

llama.cpp loads 851 of the artifact's 866 tensors. The 15 it drops total
**289,527,808 bytes (0.270 GiB)**, and four of them are the `nextn.*` multi-token
prediction head. So at this pin llama.cpp runs 64 layers and no MTP head on this
checkpoint.

This is a fidelity note for [#821](https://github.com/mudler/vllm.cpp/issues/821)
and #1003, not a defect here. It means a "quant-matched against the same weights"
comparison is not automatically a matched-work comparison: if a vllm.cpp arm runs
block 64 or the MTP head, it is doing strictly more work per token than the
denominator. Whoever takes the floor must state which side ran what, or the ratio
measures a configuration difference and reads as a performance one.

## Timings, and why they bind nothing

Recorded only so the log is complete. **Do not quote these.**

```text
common_perf_print:        load time =     705.83 ms
common_perf_print: prompt eval time =     705.38 ms /  6 tokens (117.56 ms/tok,  8.51 tok/s)
common_perf_print:        eval time =    9575.56 ms / 47 runs  (203.74 ms/tok,  4.91 tok/s)
common_perf_print:       total time =   10295.82 ms / 53 tokens
common_perf_print:    graphs reused =        46
```

One repetition. No clock pinning, which no job in a lease can do: `nvidia-smi
-lgc` is refused because `CAP_SYS_ADMIN` is absent from the container's bounding
set. No contention control. A six-token prompt is far too short to characterise
prefill. The artifact was staged over CIFS. Every one of those makes this a
by-product of the proof that the model ran, not a measurement.

## What this establishes, and what it does not

Established:

- Stock llama.cpp at `10bf611e5` fetches from the upstream remote, and tag
  `b10451` names that same object on that remote.
- It configures and builds CPU-only from a clean tree at that exact object.
- The resulting binary loads the recorded Qwen3.8-27B Q4_K_M artifact and
  generates coherent text.

Not established, and left open on purpose:

- Any speed or memory floor, on any host. #1003 owes all thirteen.
- Any vllm.cpp comparison. That is #821 W3.
- The same build on `dgx:gpu0`. Gateability does not need it; a GB10 floor does,
  and #1003 owns that.
- Anything about `llama-cli`, the server, or the web UI, none of which were
  built.

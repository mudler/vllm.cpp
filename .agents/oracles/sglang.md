# SGLang — cross-check, performance floor, and vLLM-absent model paths

Two distinct roles, and one forbidden one — the methodology is
[`../specs/sglang-parity-oracle.md`](../specs/sglang-parity-oracle.md) and the
enumerated surface is [`../sglang-matrix.md`](../sglang-matrix.md):

- **Correctness cross-check.** Where SGLang and vLLM agree on greedy tokens for
  the same model, ours matches both. Where they diverge, **vLLM wins** — SGLang
  never re-specifies a vLLM-derived behavior.
- **Performance floor.** Wherever SGLang beats vLLM on an equivalent workload,
  SGLang is the binding floor.
- **Model paths vLLM does not implement at all.** This is the case the fallback
  rule was written for. Where that path is served by the SGLang-Omni pipeline
  runtime rather than SGLang proper — `MiniMax-Music3` is the first — the binding
  record is [`sglang-omni.md`](sglang-omni.md), not this one.
- **Forbidden:** porting SGLang's data structures as a second, incompatible
  abstraction. A SGLang-distinct behavior is an opt-in over our vLLM-derived
  design, never a fork of the engine.

**Two SGLang revisions already appear in specs, for two different purposes.**
The oracle pin below is the *source* pin — the tree every `file:line` in
`sglang-matrix.md` was read from. The benchmark harness in
[`../specs/cuda-sglang-low-concurrency.md`](../specs/cuda-sglang-low-concurrency.md)
separately pins tag `v0.5.13` / `28b095c01005d4a3a2a5b637b7d028b07fba31b2` and
its runtime image, and that pin is never silently substituted into the other's
evidence.

**SGLang-Omni is a third repository and has its own record**, since #672:
[`sglang-omni.md`](sglang-omni.md). This record covers SGLang proper. A model
present in one and absent from the other is the normal case, which is why the two
are not folded together.

**It ran once, on 2026-07-28.** AGENTS.md sets the bar at "demonstrably builds
and runs the model", and this oracle cleared it under `CLAIM-SGLANG-PERF-BENCH`
(commit `0a07ac769`): the `lmsysorg/sglang:v0.5.15-cu130@sha256:d0a667e` arm64
image was pulled and ran `unsloth/Qwen3.6-27B-NVFP4` @ `890bdef7` on GB10
sm_121a with CUDA graphs captured, needing no from-source build. Three
repetitions at c8 and c16, both arms driving the identical corpus and emitting
exactly 80 by 128 output tokens with zero errors. Numbers, corpus and teardown
discipline are in [`../sglang-matrix.md`](../sglang-matrix.md) under "Perf
oracle results", and `docs/STATUS.md` carries the same measurement. **Those
numbers stand.** Nothing below retracts them.

It also carries a note about `docs/STATUS.md` that no longer resolves: that file
was retired by [#1714](https://github.com/mudler/vllm.cpp/issues/1714). The
measurement is unaffected and lives in the two records named above.

**`gateable` was `no` from 2026-08-18 to 2026-08-23, because that method became
forbidden and nothing replaced it. A permitted route now exists and the flag is
`yes` again on THAT route, not on this one.** The paragraphs below are why the
image path is closed, and they stay closed.
`../sglang-matrix.md` records the box state of the 2026-07-28 run as
"idle GB10 sm_121a, one `flock $GPU_LOCK`", its evidence sits under a host home
directory, and the container recipe at
[`../specs/cuda-sglang-low-concurrency.md`](../specs/cuda-sglang-low-concurrency.md)
lines 226, 265, 328 and 330 is one `docker pull` and two `docker run`
invocations. That is `ssh` to the host
plus a file mutex, and AGENTS.md §"Work on a GPU happens inside a lease" now
makes the `rc` lease the required path to `dgx:gpu0` and forbids reaching a
fleet device by `ssh` to run work directly. The same section records that bypass
costing a measurement on 2026-08-17, which
[`../specs/minimax-music3.md`](../specs/minimax-music3.md) §13.10 still carries
as a VOID speed axis.

**No permitted substitute was demonstrated by any of the checks below, and that
is what changed on 2026-08-23.** The three read-only checks of 2026-08-18 remain
correct about the container path: there is still no `--image`, still no container
runtime on the worker, and `kubectl` is still not the answer. What they could not
see is that SGLang needs no container at all. Read them as the reason the IMAGE
route is dead, never as a claim that the oracle is unreachable.

`rc run --help` carries `--as`, `--cwd`, `-d`, `--explain`,
`--idle-timeout`, `--max-runtime`, `--no-wait`, `--priority`, `--select`,
`--timeout` and `--tty`, and no `--image`. The `rc describe dgx:gpu0` usage
sheet lists the worker's contents as `bash git curl wget ssh | gcc g++ make
cmake ninja pkg-config | python3 pip venv | jq rsync tar unzip | kubectl |
nvidia-smi`, which names no container runtime. `kubectl` is there and is not the
answer: a sibling pod puts work on the box outside the lease the controller
accounts for, so the fleet reports `dgx:gpu0` free while the pod holds the GPU,
which is the `ssh` failure again. **`scripts/dgx-sglang-low-concurrency.sh` is
therefore unrunnable as written**: it hard-codes
`SG_IMAGE_DEFAULT='docker.io/lmsysorg/sglang:v0.5.13-cu130-runtime@sha256:9631280f…'`
at `:10-11` and passes it on as `--image`. It implements `--dry-run` only and
refuses every other mode, so the execution half it plans for was never written
and would need the forbidden path. Its default also pins v0.5.13 rather than
this record's v0.5.15.

**Why it went to `no` rather than to `yes` with a caveat, and why that reasoning
survives the flip back.** AGENTS.md defines `gateable` as PRESENT reachability —
the oracle "must demonstrably build and run the model" — and states that an
ungateable lane "records `gateable = no` and names the issue that owes the
measurement", because that record "makes the ungateable lane visible debt".
`scripts/check-oracle-pins.py` admits no third key, so a caveat carried only in
prose would have left the machine-readable value reading `yes` for a reader of
[#979](https://github.com/mudler/vllm.cpp/issues/979) who greps the block and
budgets an arm nobody could produce. That was right on 2026-08-18 and it is the
same rule that admits `yes` now: the flag reports what is reachable TODAY, and
today a lease-compliant job serves the model. `gateable` is not a history field
in either direction. The 2026-07-28 numbers never needed it, and the 2026-08-23
run does not retroactively make the image path permitted.

**One pin, two delivery artifacts, and the second one is the reachable one.**
The route out is a PyPI wheel installed inside an `rc` lease, and it is specified by row
`SGLANG-ORACLE-LEASE-WHEEL` in
[`../specs/sglang-wheel-in-lease.md`](../specs/sglang-wheel-in-lease.md). The
wheels are `sglang-0.5.15-cp312-cp312-manylinux_2_34_aarch64.whl`, 12,716,006
bytes, sha256
`1c2d2602b4ba04c6a71d2f3bf2e3654da53987536f0d65dbe4f57cdc65c9812e`, and
`sglang_kernel-0.4.4-cp310-abi3-manylinux2014_aarch64.whl`, 34,243,333 bytes,
sha256 `727e4bc53abeade20260186f99199200320b9fa51f8de7af90c01524cff73e5d`. They
sit under THIS pin rather than under a second oracle id, because an oracle id
names an upstream and not a delivery, and because the wheel's `sglang/` tree is
byte-identical to `f63458b5` for every file that exists in both. `pin` does not
move.

**The aarch64 kernel-wheel coverage is no longer untested. It was tested and it
passed**, by static analysis on 2026-08-19.
`sgl_kernel/sm100/common_ops.abi3.so` holds 56 fatbin containers of 6 cubins
each, every container declaring architectures 90, 100, 103, 110, 120 and 121,
and **zero containers lack 121**. Each payload was decompressed and its ELF
`e_flags` matched against the declared architecture: 336 of 336 agree. NVIDIA's
`cuobjdump -lelf`, release 13.0, V13.0.85, names the six targets, and the 121
cubins are **`sm_121a`**, the architecture-accelerated variant, 56 of them, one
per container; the plain ones are the 90 cubins.
`sgl-kernel/CMakeLists.txt:221` builds `-gencode=arch=compute_121a,code=sm_121a`
on aarch64 at CUDA 13.0 and `:127` builds plain `sm_90`, six gencode targets for
six cubins per container. GB10's coverage is therefore accelerated and not
merely present. This record said the inverse until 2026-08-19, read off the low
byte of `e_flags`, which carries no base-versus-`a` distinction at
`EI_ABIVERSION = 8`; the spec's `sm_121a` section records why that read was
wrong in both directions. `load_utils.py` selects `sm90/` only
at compute capability 90, so GB10 takes `sm100/`. The 5.6% size gap against the
x86_64 wheel is one absent library, `flash_ops.abi3.so`, and capability refuses
FlashAttention-3 on this device in any case. The wheel carries no runtime commit
assertion, because `sglang/_version.py` sets `__commit_id__` to `None`, so
identity is asserted against the committed per-file manifest
[`../specs/sglang-wheel-in-lease.json`](../specs/sglang-wheel-in-lease.json).

**It ran again on 2026-08-23, by a permitted route, and `gateable` is `yes` on
THAT run.** Two `rc run` jobs on `dgx:gpu0` and nothing else: no `ssh`, no
container image, no file mutex outside the lease. The route is a PyPI wheel in a
virtual environment under `/tmp`, driven by
[`../../scripts/rc-sglang-oracle-lease.sh`](../../scripts/rc-sglang-oracle-lease.sh).

Job `86282a1a-6e07-4099-b2e8-f4768aa714e8`, 20:35:04Z to 21:03:34Z, exit 0,
installed both wheels and hashed the bytes that landed, not a remote digest:
`12716006`/`1c2d2602…` and `34243333`/`727e4bc5…`, both `WHEEL_SHA_OK=1`. It
asserted identity from `cd /`: **3338 of 3338** files against the committed
manifest, zero missing, zero extra, zero differing, at pin `f63458b5…`. It also
installed `flashinfer-jit-cache==0.6.12+cu130`, which CLOSES the warm-cache
difference the spec had only been able to state. No compiler ran for any of the
199 resolved packages: `torch 2.11.0+cu130`, `transformers 5.12.1`,
`flashinfer 0.6.12`, `nvidia-cutlass-dsl 4.5.2`, `sgl-deep-gemm 0.1.4`,
`triton 3.6.0`. On `NVIDIA GB10` capability `(12, 1)`:
`is_sm100_supported=False`, `is_sm120_supported=True`,
`is_flashinfer_available=True`. `LGC_RC=4` reproduced
[#1354](https://github.com/mudler/vllm.cpp/issues/1354) on a fourth job.

Job `b9e7709d-cc96-4247-9d01-c611bce707ac`, from 21:51:56Z, re-asserted that
identity in 11 s on the surviving environment, then served
`/workspace/ckpt/qwen3.8-27b-hf` — `Qwen3_5ForConditionalGeneration`, bf16,
55,586,040,114 bytes — to readiness in **454 s** on `/health_generate`, with
decode CUDA graphs captured at `bs=[1,2,4,8,12,16,24,32]`. **The backends are
read from the server's own log, never assumed:** `attention_backend='flashinfer'`
("Attention backend not specified. Use flashinfer backend by default"),
`moe_runner_backend='auto'`, `mamba_backend='triton'`,
`linear_attn_backend='triton'`, and "Using hybrid linear attention backend for
hybrid GDN models". `flashinfer` is what `srt/server_args.py:4337-4361` predicts
for a non-MLA model on this device, and the server saying it is what makes it a
measurement. `moe_runner_backend='auto'` is NAMED and UNEXERCISED: this model has
no expert layer, so a MoE arm is still owed.

One timed c1 leg completed: **6 of 6 requests, 0 failed, 6144 input and 768
output tokens — exactly 6 x 128 — in 173.23 s**, through `/v1/completions`, the
same client path the 2026-07-28 image run drove. A discarded warmup leg before it
reached 4.431 output tok/s against the timed leg's 4.433, a 0.05% difference,
which is the measurement showing the warm JIT cache worked.

**Three things this run does NOT say.** The c8 leg is **VOID**: the BOX REBOOTED
DURING it at 22:27:00Z, so the job never printed its own teardown assertion. It
is not a concurrency-8 datapoint of any kind, because the leg had issued only its
single warmup request and `sglang-c8.log` is 0 bytes — one request was in flight
at the moment of death, not eight. The next job on `dgx:gpu0` read
`boot_id=26394f62…` against this row's `02d5a76f…` with the kernel PID counter
down from 3510 to 594
(`/mnt/nas_share/rc/gdn-moe-packed-ba/logs/gate-ab.log`, lines 1-3 and 18-20),
and the 5,000 MB `MemAvailable` watchdog never fired because the machine died
with 15,449 MB available. A third job, `0f84b66d` at 23:10:48Z, asserted the
resource came back (`COMPUTE_APPS=0`, `SGENV_PROCS=0`) and read the post-reboot
`boot_id` itself; that job archived nothing of its own, so its output is read
back with `rc logs 0f84b66d-1c30-4de5-bdb8-ee7b058f284a` and was copied to
`/mnt/nas_share/rc/sglang-w2/out/reap-20260823T231048Z/job.log` after the fact. The clock cannot be pinned inside a lease and drifted
**7.59%** against the 5% ceiling on a GB10 at 84 C with software thermal
slowdown active, so **no ratio may be divided out of any of these numbers and
none is offered**. And no vllm.cpp arm ran beside any of it. The full record,
including the host-memory floor of 14,935 MB this configuration leaves, is in
[`../specs/sglang-wheel-in-lease.md`](../specs/sglang-wheel-in-lease.md), which
is where `evidence` now points.

`scripts/dgx-sglang-low-concurrency.sh` stays unrunnable and is NOT patched: it
has no execution half at all, `:55-57` refuses every mode except `--dry-run`,
and `:5-7` records that the image pull and the `docker run` are what its P2
would have added, which is the forbidden path. Its replacement is
[`../../scripts/rc-sglang-oracle-lease.sh`](../../scripts/rc-sglang-oracle-lease.sh),
with the identity gate at
[`../../scripts/sglang_lease_identity.py`](../../scripts/sglang_lease_identity.py)
and its mutation suite at
[`../../tests/scripts/test_sglang_lease_identity.py`](../../tests/scripts/test_sglang_lease_identity.py).

**This is not a reversal of the 2026-08-16 correction.** This record said
`gateable = no` until 2026-08-16, for two and a half weeks after the run, on a
different ground: the two scoping specs it cited really were read-only, and
neither was revisited when the separate perf claim executed. That correction to
`yes` under [#979](https://github.com/mudler/vllm.cpp/issues/979) was right about
what had been measured, and it discharged the SGLang third of the three
gateability debts [#647](https://github.com/mudler/vllm.cpp/issues/647) holds
open. That third returns to open under #1265, on the new ground. The pin is not
the problem. The reach is.

**Still open at this pin, and not covered by gateability:** the greedy token-ID
correctness cross-check (`SGLANG-ORACLE-CORRECT`, `INVENTORIED`), so SGLang binds
as a performance floor only for a model whose own correctness gate passed. The
c1, c2 and c4 points, the 35B arm and the shared-prefix cache-on arm are unrun.
No DSpark speculator ships in the pinned tree: `python/sglang/srt/speculative/` at
`f63458b5` carries DFlash, EAGLE, ngram and frozen-KV MTP and nothing named
`dspark`, so any drafted SGLang arm needs a deliberate pin advance first. That
claim is scoped to the directory on purpose. DSpark does NOT postdate this pin.
`docs_new/index.mdx:86,107,108,127` is tracked at `f63458b5` and links the
2026-07-06 lmsys blog announcing it, three days before the pinned commit's own
2026-07-09 date, and `speculative/spec_info.py:60-70` registers out-of-tree
algorithms at runtime, so absence from the directory listing is not absence at
runtime. This file said the pin predated DSpark until 2026-08-16. That wider
wording is withdrawn under
[#979](https://github.com/mudler/vllm.cpp/issues/979).

```oracle-pin
id = sglang
role = secondary
upstream = https://github.com/sgl-project/sglang
scope = a model or serving path SGLang implements and vLLM does not, plus the SGLang correctness cross-check and performance floor
pin = f63458b5beaceabbd9d749b9fc956370e1b649e6
pin_label = v0.5.15
pinned_on = 2026-07-27
gateable = yes
evidence = .agents/specs/sglang-wheel-in-lease.md
```

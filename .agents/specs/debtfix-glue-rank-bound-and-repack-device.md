# DEBTFIX — the glue tensor writes past `kMaxRank`, and `quant_repack` cannot see the device

**Row:** `-` (two standing defects, each already filed; no roadmap row owns either)
**Issues:** [#2435](https://github.com/mudler/vllm.cpp/issues/2435),
[#2406](https://github.com/mudler/vllm.cpp/issues/2406)
**State:** `ACTIVE`
**Base:** `origin/main` at `63889449c`

## Scope

Two unrelated correctness defects that share one property: both are invisible to
every gate that currently runs, and both were found by reading rather than by a
red.

**This wave does not turn `sanitize-cpu` green on its own.** It removes the
finding #2435 names; the sweep in `## Evidence` found a second, unrelated one
(#2540) that the lane's abort-on-first-finding had been hiding. Both facts are
recorded rather than one.

1. **#2435** — `vllm::dense_attn::MakeTensor`
   (`include/vllm/model_executor/models/dense_device_glue.h`, `MakeTensor`) writes
   `t.shape[i]` and `t.stride[i]` for `i` up to `shape.size() - 1` with no bound,
   while `vt::Tensor` fixes `kMaxRank = 4`. Any rank-5 shape writes past both
   fixed arrays. Give the write site the bound its sibling
   `vt::Tensor::Contiguous` (`src/vt/tensor.cpp:19-20`) has carried all along,
   repair every caller the bound then refuses, and free the four allocations
   `test_glm_moe_dsa_schedule` leaks — the second half of the issue, which an
   earlier pass of this spec wrongly concluded did not exist (`## Evidence`).
2. **#2406** — `GgufLoadPolicy::FromEnv`
   (`src/vllm/model_executor/model_loader/gguf_keep_quant.cpp`) resolves
   `quant_repack` from a pure host-ISA probe with no device term, so an aarch64
   i8mm host loading `--device cuda` repacks Q8_0 weights into the ARM
   `block_q8_0x4` interleave and stages them to a card whose kernels read plain
   `block_q8_0`. Give the line the resolved device its sibling `elem_kn_repack`
   already reads two lines below.

### Out of scope, recorded under `## Owed`

* Raising `vt::kMaxRank`. Four is a vt design constant; every op, every kernel
  and `sizeof(vt::Tensor)` depend on it.
* Dequantizing the `qwen4_exp` hyper-connection weights at load. See
  `## Why the device gate and not the dequant`.
* The aarch64 CUDA measurement #2406 asks for. This wave has no lease on a
  fleet aarch64 CUDA box and does not claim one.

## Defect 1 — the unbounded write

### What it does

`MakeTensor` sets `t.rank = shape.size()` and then walks `i` from `rank - 1`
down to `0`, writing `t.shape[i]`, writing `t.stride[i]`, and reading
`t.shape[i]` back into the running product. At rank 5 the `i == 4` iteration
writes eight bytes past the end of `shape` (into `stride[0]`) and eight bytes
past the end of `stride` (into the three storage-marker bools). The `i == 0`
iteration then overwrites `stride[0]` again, so the shape damage is
self-healing and only the marker write survives — `stride[4] = 1` sets
`Tensor::repacked = true` on a tensor that was never repacked.

That is why no value gate sees it. The tensor comes out arithmetically usable,
and the corrupted member is a storage marker that only `kMatmulBTQuant` reads.

### The population

Thirteen call sites, all in tests, all the same paged-KV allocation
`{2, num_blocks, block_size, num_kv_heads, head_size}`:

| file | lines |
|---|---|
| `tests/vllm/models/test_qwen4_exp_layer_loop.cpp` | 495, 685, 828, 1116, 1484, 1721, 2038, 2131, 2295, 2547 |
| `tests/vllm/models/test_qwen4_exp_inject_residency.cpp` | 216, 329 |
| `tests/vllm/models/test_qwen4_exp_matmul_bt_dtype.cpp` | 327 |

Every one of them exists to size and fill one buffer. The tensor view it
produces is never consumed as a tensor: the fixture reads `kv_b.t().data` and
carries the five dimensions to `PagedKvCache` as separate scalars
(`include/vllm/model_executor/models/qwen3_5.h:78-98`). So `{2 * num_blocks,
block_size, num_kv_heads, head_size}` is the same bytes, the same element count
and the same truth, at a rank the type can hold.

**Production already spells it at rank 4**, which is the strongest argument that
this is a fixture spelling and not a lost dimension. `dense_attn::KvSlice`
(`include/vllm/model_executor/models/dense_attn_block.h:364-382`) is what every
attention block reads the page through, and it builds a **rank-4 strided view**
by hand — `shape = {num_blocks, block_size, Hkv, Dh}`, `stride[0] = 2 * bs * h *
dd`, with the K/V axis carried as a byte offset rather than a dimension. Nothing
in `src/` ever asks `vt::Tensor` to hold five dimensions of a KV page.

### Design

`MakeTensor` gains

```cpp
VT_CHECK(shape.size() <= static_cast<size_t>(vt::kMaxRank), "...");
```

before it writes anything, mirroring `Tensor::Contiguous`. The thirteen call
sites fold the leading `2` into the block dimension.

**The bound belongs here and not in `vt/tensor.h`.** `vt::Tensor` is a plain
aggregate with public arrays; a bound is enforceable only where something
writes. `Tensor::Contiguous` is the one other writer and it already has one.
This closes the parallel path rather than adding a third.

## Defect 2 — the device-blind repack

### Why the device gate and not the dequant

[#2406's own comment](https://github.com/mudler/vllm.cpp/issues/2406) records a
second option: dequantize `hc_*_down` / `hc_*_up` to bf16 at load, which mirrors
vLLM and removes all 194 repack-hazard tensors of the released artifact at once.
This spec rejects it, for reasons and not for size.

1. **The defect is in the policy line, not in one architecture's tensors.** The
   dequant removes `qwen4_exp`'s population and leaves the same device-blind
   line deciding for every Q8_0 weight of `qwen3_5`, `glm5_next`,
   `glm_moe_dsa`, `deepseek_v4` and `laguna` on the same host. The device gate
   fixes the class.
2. **The device gate provably cannot move the CPU control; the dequant can.**
   On `dev == kCPU` the gated expression is character-for-character the old one,
   so the released artifact's `--device cpu` decode is unchanged by
   construction. Dequantizing 194 weights replaces `kMatmulBTQuant` with a bf16
   GEMM on every hyper-connection projection — different arithmetic, and the
   eight-token control `11751 13 15767 411 2029 11 1092 369` would have to be
   re-earned on a box this wave has no lease for.
3. **The upstream evidence is about intent, and it is off-pin.** vLLM's
   hyper-connection linears pass `quant_config=None` explicitly, and the decode
   GEMM demands packed row-major bf16 on both operands. **Read at
   `vllm-project/vllm` `e126687a9a`, which is 1,465 commits past this tree's
   parity pin `555967922` — a forward reference to an unpinned upstream, quoted
   as intent and never as an anchor.** Verified in a local clone of that
   revision, not transcribed from the issue:

   | claim | `file:line` at `e126687a9a` | read |
   |---|---|---|
   | the three HC linears are unquantized | `vllm/models/qwen4_exp/nvidia/hyperconnection.py:102`, `:113`, `:122` | `quant_config=None,` |
   | their dtype is bf16 by literal | `vllm/models/qwen4_exp/nvidia/model.py:260`, `:436` | `params_dtype=torch.bfloat16,` |
   | the decode GEMM's operand predicate | `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py:92-99` | `_is_packed_row_major(weight) and ... weight.dtype == torch.bfloat16` |

   The paths are `vllm/models/qwen4_exp/`, not `vllm/model_executor/models/`.
   vLLM loads safetensors and never meets a Q8_0 hyper-connection weight, so it
   states no position on the GGUF arm. The oracle that does is
   `llama-cpp-qwen4exp`, which keeps them typed and declares all six
   `GGML_OP_MUL_MAT`.
4. **Correctness first.** Turning repack off on a non-CPU device costs the i8mm
   interleave on any Q8_0 weight that reaches the documented CUDA-to-CPU drain
   (`src/vt/cuda/cuda_quant_dot.cu`). That is a speed question on an already-
   degraded path, and AGENTS.md `## Gates` settles the order.

The dequant stays a live option for the `qwen4_exp` row as a residency and
parity question. It is recorded under `## Owed`, not refused.

### Design

```cpp
p.quant_repack = QuantRepackForDevice(p.keep_quant, p.cpu_ref,
                                      vt::cpu::QuantRepackActive(), dev);
```

with

```cpp
bool QuantRepackForDevice(bool keep_quant, bool cpu_ref,
                          bool host_repack_active, vt::DeviceType dev);
```

declared beside the policy and defined in its translation unit.

**Why a named predicate and not `&& dev == kCPU` inline.** `QuantRepackActive()`
is a hard compile-time `false` on every non-aarch64 target
(`src/vt/cpu/cpu_quant_repack_arm.cpp:275`), so an inline gate asserted through
`FromEnv` is **vacuous on x86 CI**: `quant_repack` is already false there for the
wrong reason, and the assertion would be a mute switch that passes whether the
device term exists or not. Taking `host_repack_active` as a parameter is the same
move `RouteGgufTensor` already made for the device — the ROCm routing case
(`tests/vllm/test_gguf_keep_quant.cpp`, "keep-quant routing respects the RUNNING DEVICE's format set") says so in its own comment: "the
whole point of removing the probe is that a device's routing is checkable from a
host that is not that device."

## Tests

**#2435.** The red is a sanitizer run, not an assertion. `test_qwen4_exp_layer_loop`
exits 1; under this lane's `-fno-sanitize-recover=all` it aborts at the first
finding, so it reports no assertions at all (see `## Evidence` — the issue's own
`309/309, SUCCESS!` was measured on a recoverable build). The gate is therefore the process exit code of
the `address,undefined` build, plus a count: the number of
`dense_device_glue.h:5[678]: runtime error` lines, which must go 3 → 0. A doctest
case is added beside it (`tests/vllm/models/test_dense_device_glue_rank.cpp`)
asserting that a rank-5 shape now `CHECK_THROWS` and that a rank-4 one is
unchanged, so the bound has an assertion of its own on every lane including the
ones with no sanitizer.

**#2406.** A doctest case over `QuantRepackForDevice`'s full truth table, run on
every host. The discriminating row is
`QuantRepackForDevice(true, false, /*host_repack_active=*/true, kCUDA) == false`
against `... kCPU) == true`. The `expect.quant_repack` mirror in the same file's
"production default is keep-quant" case is updated to the same predicate so the env-load
comparison stays apples-to-apples on an i8mm host.

## Gates

```sh
cmake -S . -B build-san -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_CUDA=OFF \
      -DVLLM_CPP_SANITIZE='address,undefined'
cmake --build build-san -j 2  # -j 6 when the box is idle
UBSAN_OPTIONS=print_stacktrace=1 ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1 \
  VT_POOL_BYPASS=1 ctest --test-dir build-san --output-on-failure
```

plus the ordinary CPU suite, and `scripts/agent-preflight.sh --staged`.

## Evidence

Host: this development box, x86-64, `g++ 13.3.0`, Debug + `-fsanitize=address,undefined`
with the lane's own `-fno-sanitize-recover=all`. Base `origin/main` `63889449c`;
every table below was RE-MEASURED on the merged head after `origin/main` moved
twenty and then a further twenty-nine commits (QUANT-IQ3S edits two of the binaries these tables name, so the
numbers were re-taken rather than carried across: `test_gguf_keep_quant`
9987 -> 10311 and `test_glm5_next_bridge` 32562 -> 32563 assertions, both still
`rc=0` with zero findings).

### #2435 — red, then green

**RED**, on the base tree with nothing applied:

```
$ UBSAN_OPTIONS=print_stacktrace=1 ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1 \
    VT_POOL_BYPASS=1 ./tests/test_qwen4_exp_layer_loop ; echo $?
dense_device_glue.h:56:14: runtime error: index 4 out of bounds for type 'long int [4]'
    #0 vllm::dense_attn::MakeTensor(...)
    #1 vllm::dense_attn::DBuf::DBuf(...)
    #2 DOCTEST_ANON_FUNC_2 tests/vllm/models/test_qwen4_exp_layer_loop.cpp:497
1
```

Ten lines of output and **not one doctest assertion**, reproduced three times.

**THE ISSUE'S DESCRIPTION OF THE RED IS WRONG ON THIS TREE, and the correction
matters.** #2435 records three UBSan lines (`:56`, `:57`, `:58`), a
`LeakSanitizer` report, and `309/309, SUCCESS!` beside them. This lane compiles
`-fno-sanitize-recover=all` (`CMakeLists.txt:260`), so the FIRST finding aborts:
there is one line, no leak check runs at all, and no assertion executes. The
figure quoted for the assertions was measured on a recoverable build. Nothing
about the defect changes; what changes is that the sanitizer lane never reported
whether this test passes — it reported that it started.

**GREEN**, with the bound and the thirteen fixture repairs:

```
test_qwen4_exp_layer_loop            rc=0 ubsan=0 leaks=0 assertions: 341 | 341 passed
test_device_pool                     rc=0 ubsan=0 leaks=0 assertions:  42 |  42 passed
test_gguf_keep_quant                 rc=0 ubsan=0 leaks=0 assertions: 9987 | 9987 passed
test_qwen4_exp_inject_residency      rc=0 ubsan=0 leaks=0 assertions:   8 |   8 passed
test_qwen4_exp_matmul_bt_dtype       rc=0 ubsan=0 leaks=0 assertions:   2 |   2 passed
test_glm5_next_bridge                rc=0 ubsan=0 leaks=0 assertions: 32562 | 32562 passed
test_resident_weight_host_addressable rc=0 ubsan=0 leaks=0 assertions:  85 |  85 passed
```

The counted property is `grep -c "runtime error"`: **1 → 0**, and the process
exit code `1 → 0`.

### The "384 byte leak" IS a leak, and this spec said it was not

**This section previously concluded that #2435's second half did not exist. That
conclusion was wrong, and CI refuted it.** It is kept as a correction rather than
edited away, because the reasoning that produced it is the reusable part.

What the earlier pass measured is true and was not enough. Under the lane's own
`VT_POOL_BYPASS=1`, LeakSanitizer reports nothing on any of the fifteen binaries
that construct a `DBuf`; drop that variable and the same binaries report
`53696 byte(s) leaked in 114 allocation(s)`, every stack `DevicePool::Get`. That
is genuinely the pool's deliberate retention. **The error was the inference**:
the figure matched nothing, so the pass concluded the reported 384 bytes were the
same artefact seen on a smaller tree — a number explained by a mechanism that
produces a different number. The sweep it rested on was scoped to the fifteen
`dense_device_glue.h` includers, and the leaking binary is not one of them.

`sanitize-cpu (address,undefined)` on this branch's own head ran all 701 tests
and named it:

```
687/701 Test #687: test_glm_moe_dsa_schedule ...***Failed
[doctest] test cases:  12 |  12 passed | 0 failed | 0 skipped
[doctest] assertions: 533 | 533 passed | 0 failed |
[doctest] Status: SUCCESS!
==39368==ERROR: LeakSanitizer: detected memory leaks
Direct leak of 128 byte(s) ... #3 operator() tests/vllm/models/test_glm_moe_dsa_schedule.cpp:680
Direct leak of 128 byte(s) ... #3 operator() tests/vllm/models/test_glm_moe_dsa_schedule.cpp:680
Direct leak of  64 byte(s) ... #3 operator() tests/vllm/models/test_glm_moe_dsa_schedule.cpp:680
Direct leak of  64 byte(s) ... #3 operator() tests/vllm/models/test_glm_moe_dsa_schedule.cpp:680
SUMMARY: AddressSanitizer: 384 byte(s) leaked in 4 allocation(s).
```

128 + 128 + 64 + 64 = **384 bytes in 4 allocations**, #2435's figure exactly,
under `VT_POOL_BYPASS=1`, with the pool not involved at all: the allocations go
straight to `Backend::Alloc`.

**The cause.** `TEST_CASE("The router gate GEMM at f32 reproduces the exact
products a bf16 store loses")` declares `std::vector<void*> owned;` and pushes
every allocation into it — it was written to free them — and never does. The same
file's `Harness` collects into `owned_` and frees from `~Harness`; this case
duplicated the vector and dropped the destructor.

**The fix is RAII, not a trailing loop.** `RequireFinite` and the `REQUIRE`s in
that body throw, and doctest unwinds, so a free loop at the end of the case is
skipped on exactly the runs where a leak matters least to nobody. A scope-exit
guard frees on both paths, matching `~Harness` twenty lines above it.

**Red, then green, measured locally on this head:**

```
with the guard      rc=0  533 | 533 passed  Status: SUCCESS!   no LeakSanitizer line
M6 (guard removed)  rc=1  533 | 533 passed  Status: SUCCESS!   384 byte(s) in 4 allocations
restored            rc=0  533 | 533 passed  Status: SUCCESS!   no LeakSanitizer line
```

The counted property is the SUMMARY line: 384 -> 0 bytes, 4 -> 0 allocations, and
the process exit code 1 -> 0.

**Why the shape of this defect is worth keeping.** It is the same shape as #2435
itself, one level quieter: `Status: SUCCESS!` over 533 passing assertions and a
nonzero exit. `#2435` described precisely that combination and this spec read it
as a description of `test_qwen4_exp_layer_loop`, which under
`-fno-sanitize-recover=all` cannot produce it. **The issue was reporting two
binaries, not one**, and the assertion count it quoted belonged to the other one.

### #2406 — red, then green

The predicate's discriminating row, run on this x86 host where the whole
`FromEnv` path is blind:

| assertion | before the device term | after |
|---|---|---|
| `QuantRepackForDevice(true, false, true, kCPU)` | `true` | `true` |
| `QuantRepackForDevice(true, false, true, kCUDA)` | `true` (RED) | `false` |
| `QuantRepackForDevice(true, false, true, kROCM)` | `true` (RED) | `false` |

`test_gguf_keep_quant -tc="quant_repack is decided WITH the resolved device (#2406)"`
runs 10 assertions, all passing, `rc=0`. The whole binary is 9987/9987, `rc=0`.

**WHAT THIS EVIDENCE CANNOT DO, stated plainly.** No x86 host can gate the WIRE.
`vt::cpu::QuantRepackActive()` is a literal `false` off aarch64
(`src/vt/cpu/cpu_quant_repack_arm.cpp:275`), so `FromEnv(kCUDA).quant_repack` is
`false` on this box whether the device term exists or not. The truth table gates
the rule; the wire assertion carries `VT_GGUF_KEEP_QUANT=1` so that it becomes
discriminating on an aarch64 i8mm box, and it is inert here. The CPU control
`11751 13 15767 411 2029 11 1092 369` was NOT re-run: `thor:gpu0` is held by
another wave. It is unchanged by construction rather than by measurement — on
`dev == kCPU` the gated expression is character-for-character the old one.

### Every test that constructs the glue types, under the same sanitizers

The static scan for rank-5 shapes reads brace literals; a shape built into a
`std::vector` at run time would escape it. So every test file that includes
`dense_device_glue.h` or `dense_attn_block.h` — the fifteen that can construct a
`DBuf` or call `MakeTensor` at all — was built and run:

| binary | rc | UBSan | leaks | assertions |
|---|---|---|---|---|
| `test_device_pool` | 0 | 0 | 0 | 42 / 42 |
| `test_glm_moe_dsa_schedule` | 0 | 0 | 0 | 533 / 533, **leak 384 -> 0** |
| `test_glm5_next_moe` | 0 | 0 | 0 | 12731 / 12731 |
| `test_gguf_keep_quant` | 0 | 0 | 0 | 10311 / 10311 |
| `test_glm5_next_bridge` | 0 | 0 | 0 | 32563 / 32563 |
| `test_resident_weight_host_addressable` | 0 | 0 | 0 | 85 / 85 |
| `test_qwen4_exp_inject_residency` | 0 | 0 | 0 | 8 / 8 |
| `test_qwen4_exp_matmul_bt_dtype` | 0 | 0 | 0 | 2 / 2 |
| `test_kv_cache_fp8_wiring` | 0 | 0 | 0 | 487 / 487 |
| `test_load_direct_upload` | 0 | 0 | 0 | 203 / 203 |
| `test_qwen3_5_gdn_spec_routing` | 0 | 0 | 0 | 82 / 82 |
| `test_qwen3_dflash2_draft` | 0 | 0 | 0 | 449 / 449 |
| `test_qwen4_exp_forward` | 0 | 0 | 0 | 429 / 429 |
| `test_qwen4_exp_layer_loop` | 0 | 0 | 0 | 341 / 341 |
| `test_qwen4_exp_qsa_block` | 0 | 0 | 0 | 5937 / 5937 |
| `test_mistral_paged_engine` | 0 | 0 | 0 | 0 / 0 — a SELF-DECLARED skip, not a silent one: "mistral-7B-v0.3 checkpoint absent; skipping (dgx-only)" |
| `test_dots3_note_attn` | **1** | **1** | 0 | aborted before any |
| `test_muse_glimmer_text` | **1** | **1** | 0 | aborted before any |

No rank-5 caller survived the scan, and none of the twelve reports a leak.

### THE LANE HAS A SECOND FINDING, AND IT IS NOT THIS ONE

`test_dots3_note_attn` and `test_muse_glimmer_text` both abort on the SAME
UBSan report, and it is not the rank overrun:

```
src/vt/cpu/cpu_matmul_elem.cpp:577:61: runtime error: load of misaligned address
0x782f0e901907 for type 'const uint16_t', which requires 2 byte alignment
    #0 vt::cpu::WidenRowToF32   cpu_matmul_elem.cpp:577
    #1 RmsNormKernel            cpu_ops.cpp:557
    #2 vt::RmsNorm              ops.cpp:1043
    ...
    #7 Dots3NoteModel::ForwardDevice / muse_glimmer.cpp:298
    #9 vllm::ModelRegistry::Forward   model_registry.cpp:646
```

The operand is the RMSNorm **gamma** (`cpu_ops.cpp:557` widens `w.data` once per
call), reached through `ResidentWeight(d, layer.input_layernorm, {H})` on both
models, at an odd address. **It is a different subsystem and a different defect**,
and this branch touches none of the nine files in either stack — `git diff
--name-only 63889449c HEAD` and the stacks' file list have an empty intersection.

It was hidden behind #2435 because the lane aborts at the first finding and
`test_qwen4_exp_layer_loop` sorts earlier. Filed as
[#2540](https://github.com/mudler/vllm.cpp/issues/2540) and listed under
`## Owed` below. **The claim this wave can make is therefore narrower than the
one it was asked for**: the finding that #2435 names is gone, and
`sanitize-cpu (address,undefined)` has at least one more before it is green.

### The mutations

Each applied in this worktree, rebuilt, run, then restored and verified
byte-for-byte (`git status --porcelain` empty, `md5sum` recorded). No mutation
ran against a red baseline, and every build returned rc 0 before its run.

| # | mutation | rebuilt | result |
|---|---|---|---|
| M1 | delete `CheckRank` from `MakeTensor` only | `test_device_pool`, rc 0 | rc **1**: UBSan `index 4 out of bounds` at the `CHECK_THROWS` line, process aborted |
| M2 | delete `CheckRank` from `DBuf`'s constructor only | `test_device_pool`, rc 0 | rc **1**: `CHECK(a.allocs() == 0)` reads `1 == 0` — the pool block really is stranded, 11/12 assertions pass |
| M3 | make `CheckRank` throw unconditionally | `test_qwen4_exp_layer_loop`, rc 0 | rc **1**: `ModelRegistry::Forward reaches it on a loaded qwen4exp GGUF` THREW the bound's own message |
| M4 | revert the `FromEnv` wire to the old inline expression | `test_gguf_keep_quant`, rc 0 | rc **0**, 9987/9987 — **SURVIVED**, see below |
| M5 | drop `&& dev == kCPU` from `QuantRepackForDevice` | `test_gguf_keep_quant`, rc 0 | rc **1**: the `kCUDA`, `kROCM` and `kXPU` rows read `true == false`, 7/10 assertions pass |
| M6 | never instantiate the scope guard in `test_glm_moe_dsa_schedule` | that target, rc 0 | rc **1**: `SUMMARY: AddressSanitizer: 384 byte(s) leaked in 4 allocation(s)` over `533 passed` and `Status: SUCCESS!` — CI's exact red, reproduced locally |

**M3 is the reachability evidence.** The guard is not reached only by the case
written for it: a production entry point, `ModelRegistry::Forward`, allocates
through `dense_attn::DBuf` and therefore through `CheckRank` on every step.

**M4 SURVIVED, and that is a finding rather than a footnote.** On an x86 host no
test in this tree can tell the wired predicate from the expression it replaced,
because `vt::cpu::QuantRepackActive()` is a compile-time `false` here and makes
`quant_repack` false under either wiring. M5 shows the RULE is gated; the WIRE
is gated only on an aarch64 i8mm host, where the case's `VT_GGUF_KEEP_QUANT=1`
arm becomes discriminating. This is stated in the case's own comment.

**AND NO PRE-MERGE CI LANE GATES THE WIRE EITHER.** `build-test-cpu-arm64` is the
only aarch64 job that runs on a pull request, and it builds exactly four targets
— `test_cpu_isa_arm`, `test_ops_matmul_elem`, `test_ops_quant_dot`,
`test_ops_quant_repack` — and runs no `ctest` at all, so it never reaches
`GgufLoadPolicy::FromEnv`. `build-test-cpu-arm64-full` does build the suite, and
its `if:` is `github.event_name == 'schedule' || github.event_name ==
'workflow_dispatch'`, so it reads `skipping` on every pull request and first
executes one scheduled cycle AFTER a merge. A green `build-test-cpu-arm64` on
this branch therefore says nothing about #2406, and an earlier draft of this
record inferred from the runner's architecture that it did.

What stands behind the ungated wire is the pair of runtime tripwires that
already existed — `qwen3_5.cpp`'s `VT_CHECK(!w.repacked)` at device staging and
`ResidentWeight`'s — and those ARE gated everywhere, by
`test_resident_weight_host_addressable` (85/85, rc 0). A reverted wire on an
aarch64 CUDA box therefore refuses by name instead of emitting wrong tokens.

## Risks

* The rank bound turns a silent corruption into a throw. If any caller outside
  the thirteen builds a rank-5 shape from a runtime vector, the static scan
  cannot see it and the suite run is what finds it. That is the point of running
  the whole suite rather than the one test.
* Gating `quant_repack` on the device removes the i8mm interleave from any
  aarch64 `--device cuda` load. No arm of this fleet gates that path today, so
  the loss is unmeasured; it is named here rather than assumed to be zero.

## Stop conditions

Stop and report if the rank bound reds a test whose rank-5 shape is genuinely
consumed as a rank-5 tensor. That would make this a `kMaxRank` question and not
a bounds question, and it is a different row.

## Owed

* Raising `vt::kMaxRank`, if a production path ever needs rank 5.
* Dequantizing the `qwen4_exp` hyper-connection weights at load
  (`MODEL-MM-QWEN4-EXP`), with the residency and token evidence that needs.
* The aarch64 CUDA measurement of the repack trade that #2406 asks for.
* ~~**The misaligned bf16 gamma load**
  ([#2540](https://github.com/mudler/vllm.cpp/issues/2540)).~~ NO LONGER OWED
  HERE. `FIX-UNALIGNED-CONSUMERS-2540`
  ([`unaligned-safetensors-consumers.md`](unaligned-safetensors-consumers.md))
  claimed and fixed it on 2026-09-02, and the issue's own `Row:` line now names
  that row. The guess recorded here — that the root cause was "wherever
  `ResidentWeight` produced an odd `bytes.data()`" — was measured and is FALSE:
  the producer is the safetensors payload base, which is `8 + <JSON header
  length>` and therefore odd for roughly half of all files. Nothing produced a
  defect; `WidenRowToF32` read bytes it was not entitled to assume were aligned.

## Now

`ACTIVE`.

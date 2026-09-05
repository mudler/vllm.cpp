# Unaligned safetensors bytes — repair the CONSUMERS, keep the zero-copy borrow

Identity: `FIX-UNALIGNED-CONSUMERS-2540`

Issues: [#2540](https://github.com/mudler/vllm.cpp/issues/2540) (bf16 RMSNorm
gamma), [#2558](https://github.com/mudler/vllm.cpp/issues/2558) (EXL3 `suh`,
`svh` and the trellis), [#2578](https://github.com/mudler/vllm.cpp/issues/2578)
(`WeightF32`, found by this row's own green run).

Class predecessors: [#301](https://github.com/mudler/vllm.cpp/issues/301), which
left the `vt::LoadUnaligned` seam;
[`.agents/specs/unaligned-safetensors-loaders.md`](unaligned-safetensors-loaders.md)
(`FIX-UNALIGNED-LOADERS-772`), which repaired the four loader casts and states
the design this row reuses; and
[#1359](https://github.com/mudler/vllm.cpp/issues/1359).

Status: `ACTIVE`. Base `aedad724c974f3da73f70981bd3653bdca0821b6`.

## Scope

Make the three CPU consumers that read a borrowed safetensors payload as
`uint16_t` tolerate an odd base address, using the existing `vt::LoadUnaligned`
seam and the byte-cursor shape `vt::cpu::LoadF32` and `cpu_layernorm.cpp`
already use:

| Consumer | Site at the base SHA | Reached by |
|---|---|---|
| `vt::cpu::WidenRowToF32` | `src/vt/cpu/cpu_matmul_elem.cpp:577` | bf16 RMSNorm gamma (#2540) |
| `HadRowBlock` / `HadRows` | `src/vt/cpu/cpu_exl3_kernels.cpp:130` | EXL3 `suh` / `svh` (#2558) |
| `TileWord32` | `src/vt/cpu/cpu_exl3_dequant.cpp:64` | EXL3 trellis, kI8 read as `uint16_t` (#2558) |
| `WeightF32` | `src/vllm/model_executor/models/qwen3_5.cpp:1028` | a bf16 attention weight widened for the f32 kernels (#2578) |

**The fourth row was not in the dispatch, and it is not scope creep.**
`-fno-sanitize-recover=all` stops the binary at the FIRST report, so each repair
in this class reveals the next site rather than a clean run. Repairing the EXL3
operands made `test_qwen35_exl3` abort one frame later, in `WeightF32`, reached
from `ModelRegistry::Forward` through `FullAttnBlockPaged`. It is the same
defect, the same idiom and the same test, so it is fixed in this flow with its
own issue, as AGENTS.md's in-flow rule requires. The target state is not reached
without it.

The pointer chains that feed those three from a borrowed mapping move to byte
arithmetic in the same change, because advancing a misaligned `uint16_t*` is
undefined in its own right and a fix that only repairs the final load leaves the
`Exl3DecodeTile` / `Exl3ReconstructInner` / `MoeGemm` walk forming one.

Change no loaded value, no shape rule, no dtype rule, no error message, no
default and no public behavior. `include/vllm.h` is untouched. Four declarations
in `include/vt/ops.h` widen a parameter from `const uint16_t*` to `const void*`;
every existing caller passes a `uint16_t*`, which converts implicitly, so no
call site outside the two `.cpp` files under repair changes.

Explicitly excluded:

* `BorrowStTensorBytes` and every other producer. See `## Why not the producer`.
* The device arms. CUDA, ROCm and Vulkan read device allocations, which are
  256-byte aligned by the allocator; no borrowed host pointer reaches them.
* The `NarrowRowFromF32` / `StoreF32At` store side, which writes engine-allocated
  output buffers, never a file mapping. That is the polarity
  `cpu_layernorm.cpp:44-46` states and this row keeps it.
* `test_qwen4_exp_layer_loop`, repaired by `33b08e463` before this base.

## Root cause, re-verified at the base SHA

A safetensors payload starts at `8 + <JSON header length>`
(`safetensors_reader.cpp:78`, `data_base = bytes + 8 + header_len`). A header
length is arbitrary, so a tensor routinely begins on an odd byte. That is an
ordinary file, not a corrupt one — the same argument the class spec makes.

`BorrowStTensorBytes` (`qwen3_5_weights.cpp:490`) borrows the mapped bytes
verbatim, `OwnedTensor::View` aliases them into a `vt::Tensor`, and the three
consumers above then form a `const uint16_t*` over an odd address and load
through it. Under `sanitize-cpu (address,undefined)` the lane's own
`-fno-sanitize-recover=all` turns that into an abort before any doctest
assertion is reported.

The trellis case is the sharpest statement of the polarity: `vt::Exl3Gemm`
requires `trellis.dtype == DType::kI8` ("the trellis travels as opaque i8
BYTES", `ops.cpp:5490`), so its natural alignment requirement is 1 and an odd
base is not merely legal but expected. `TileWord32`'s own comment already says
the tile "may sit at any alignment inside a safetensors mmap" — it assembles the
uint32 by hand for exactly that reason — and then reads it through
`tile[2 * index]`. The comment states the contract the code does not keep.

## Why not the producer

External PR #2561 fixes the same seven tests the other way: a fail-closed
2-byte-alignment gate inside `BorrowStTensorBytes`, so a misaligned tensor falls
back to `MakeOwned` + `memcpy`. It works — its sanitize run reported no
`misaligned address` — and it is the shape #2558's own body proposes. This row
rejects it, and the measurement is the reason.

`tests/vllm/test_load_direct_upload.cpp`'s fixture header is 171 bytes, so the
payload base is 179, and **every** tensor in the file is refused:

| Tensor | Base | `% 2` | `% 4` |
|---|---:|---:|---:|
| `w` | 179 | 1 | 3 |
| `v` | 195 | 1 | 3 |
| `f` | 211 | 1 | 3 |

Recomputed from `Header()` and `U64Le(h.size())` at this base SHA, independently
of #2561. Since a header length is arbitrary, the parity of the base is a coin
flip per file, so a producer gate switches off direct upload for every tensor in
roughly half of all bf16 checkpoints — for a hazard that costs nothing to
tolerate. It also turns two passing tests red:
`tests/vllm/test_load_direct_upload.cpp:142` (`CHECK(w.bytes.borrowed())`) and
`tests/vllm/models/test_qwen3_5_dense_load_residency.cpp:296` ("the MODELOPT
NVFP4 arm BORROWS the mapping, it does not copy"). Both must stay green, and
they are this row's separating criterion.

`vt::LoadUnaligned` is a `std::memcpy` of `sizeof(T)` bytes. The class spec
measured it at `-O2` as instruction-for-instruction identical to the raw cast
(`movzwl (%rdi,%rax,2)`), with no `memcpy` call emitted. The consumer repair
therefore costs nothing on the borrow path, and the borrow survives.

#2540's body asks for the fix at the producer too ("find where the odd `w.data`
comes from and fix it THERE"). That instruction was written before #2558
identified the producer. The producer is the file, and a file is not a defect.

## Design

**`WidenRowToF32`.** The `kF16` and `kBF16` arms take a `const unsigned char*`
byte cursor and read each element with `vt::LoadUnaligned<uint16_t>`. The `kF32`
arm already went through `std::memcpy` and is untouched. This is the single
choke point: `RmsNormKernel` (`cpu_ops.cpp:557`), the elementwise matmul
(`cpu_ops.cpp:270`) and paged attention (`cpu_paged_attn.cpp:233`) all widen
through it, and it already takes `const void* src`, so no caller changes.

**`HadRowBlock` / `HadRows`.** `pre` and `post` become `const void*`. They are
the weight vectors `suh` and `svh`, which the borrow hands over as mapped bytes;
`in` and `out` stay typed because they are the activation and the engine's own
scratch, which `cpu_backend.cpp:20,42` allocates through `std::aligned_alloc(64,
...)`. Both call sites pass `suh.data` / `svh.data` rather than
`Ptr<uint16_t>()`, so no misaligned typed pointer is formed at all. The MoE
arm's `table` lambda returns `const void*` for the same reason.

**The trellis chain.** `TileWord32` takes a `const void*` tile base and reads its
two halves with `vt::LoadUnaligned<uint16_t>`. `Exl3TileCodeword`,
`Exl3DecodeTile`, `Exl3ReconstructInner`, `Exl3DequantLinear` and `MoeGemm` take
`const void*` for the trellis, and every walk over it advances a `const unsigned
char*` by `tile_words * sizeof(uint16_t)` bytes.

**`WeightF32`.** The same byte cursor plus `vt::LoadUnaligned<uint16_t>` over
`w.bytes.data()`, which is already a `const uint8_t*` — the `reinterpret_cast`
to `const uint16_t*` was the only thing making the load undefined. It has no
unit case and gets none: it lives in an anonymous namespace in `qwen3_5.cpp` and
nothing outside that file can name it. Its gate is `test_qwen35_exl3` under the
sanitizer, which reaches it through `ModelRegistry::Forward`, and that is the
stronger gate — a production entry point rather than a hand-constructed operand.

**That `sizeof(uint16_t)` is the whole risk of this change**, exactly as it was
for `CopyRawNK` in the class spec: the tile stride counts 16-bit WORDS and the
new cursor counts BYTES. It is what the value gate below targets, and it is why
the gate asserts byte equality against the aligned run rather than a tolerance —
a factor of two on the tile stride decodes a different weight, which a tolerance
on a random-bit trellis could plausibly absorb and an equality cannot.

## Tests and RED evidence

Two cases, both added to the suite that already owns the subject, so no new
`tests/CMakeLists.txt` row is created and no existing registration moves.

1. `tests/vt/test_ops_rmsnorm_weight_dtype.cpp` — a bf16 gamma placed at an ODD
   byte address inside an over-allocated buffer, run through `vt::RmsNorm` on
   the CPU, asserted byte-identical to the same gamma at an even address. The
   case `REQUIRE`s the address parity first, so a case that silently landed on
   an even address cannot read as a pass.
2. `tests/vt/test_exl3_gemm.cpp` — `suh`, `svh` and the trellis each placed at an
   ODD byte address, run through `vt::Exl3Gemm` on the CPU, asserted
   byte-identical to the aligned run. Same `REQUIRE` on each parity.

RED is taken two ways, because the two failures are different:

* Under `-fsanitize=address,undefined` on the pre-fix tree, each case aborts
  with `runtime error: load of misaligned address ... for type 'const uint16_t'`
  at the site named in `## Scope`. That is the sanitize-cpu lane's failure.
* Without a sanitizer the pre-fix binary passes on x86, which is precisely why
  this class survived three sweeps. The equality assertion is therefore aimed at
  the byte-stride mutation the refactor could introduce, not at the original
  defect, and it is red on any lane if the stride is wrong.

## Gates

```sh
cmake -S . -B build-sanitize -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_CUDA=OFF \
      -DVLLM_CPP_SANITIZE='address,undefined'
cmake --build build-sanitize -j 4 --target \
      test_llama_embedding_fold test_muse_glimmer_text test_muse_glimmer_text_fallback \
      test_dots3_note_attn test_openai_api_server test_capi test_qwen35_exl3 \
      test_ops_rmsnorm_weight_dtype test_exl3_gemm
VT_POOL_BYPASS=1 UBSAN_OPTIONS=print_stacktrace=1 ./build-sanitize/tests/<each>
```

Zero `misaligned address` reports, and none at `cpu_matmul_elem.cpp:577`,
`cpu_exl3_kernels.cpp:130` or `cpu_exl3_dequant.cpp:64`.

The separating gate, on the same build:

```sh
ctest --test-dir build -R 'test_load_direct_upload|test_qwen3_5_dense_load_residency'
```

Both must PASS. A change that reds either of them has become #2561 and is not
this row.

## Risks and stop conditions

* **Byte-vs-word stride.** Bounded by the equality gate above and by the
  existing bit-exact EXL3 decode suites (`test_exl3_dequant`, `test_exl3_gemv`,
  `test_exl3_moe`, `test_exl3_gemm`), which compare against references built
  from definitions rather than from the implementation.
* **A `const void*` parameter accepts anything.** The four widened declarations
  lose the compiler's `uint16_t*` type check. They are host decoders of a
  checkpoint FORMAT whose operand is stated in the declaration's own comment as
  "the tile's 16\*bits int16 words as stored", and every caller in the tree
  passes exactly that. Recorded as the cost of removing the misaligned typed
  pointer, not as a free change.
* **No belt is added to `vt::Exl3Gemm`.** #2561 added `VT_CHECK`s asserting
  2-byte alignment of `suh`, `svh` and `trellis`. After this repair those
  operands are legitimately misaligned on any ordinary checkpoint, so such a
  check would refuse the borrow path this row exists to preserve — it would be
  #2561's producer gate relocated one frame up. A refusal must name something
  the kernel cannot represent, and misalignment is no longer one.
* Stop and report on ENOSPC or an OOM symptom rather than pressing on. The box
  has both histories.

## Owed

* [#2579](https://github.com/mudler/vllm.cpp/issues/2579). A grep at the base SHA
  finds 20 further `reinterpret_cast<const uint16_t*>(<OwnedTensor>.bytes.data())`
  across ten model files (`gemma4_moe.cpp` alone has 10). Whether any of them is
  actually reached with an odd base is UNMEASURED, so the list is a population of
  the SHAPE and not a list of findings; turning it into findings needs a run per
  model rather than a grep. Fixing 20 sites across ten models is also not the unit
  of work #2540 and #2558 describe, and each wants its own model suite green
  beside it. Filed rather than swept, and named here so the debt is visible.

* [#2597](https://github.com/mudler/vllm.cpp/issues/2597). Making the trellis a
  byte cursor widened five weight-side operands from `const uint16_t*` to
  `const void*` (`Exl3TileCodeword`, `Exl3DecodeTile`, `Exl3ReconstructInner`,
  `Exl3DequantLinear`, and the file-local `MoeGemm`; `HadRowBlock`'s `pre` and
  `post` went with them). `const void*` accepts any pointer without a
  diagnostic, so a future caller that passes the wrong array, or forgets the
  factor of two the pointer type used to supply, compiles and decodes garbage.
  No caller is wrong today and the suites are green, so this is a latent hazard
  and not a defect. The repair is a one-member wrapper whose own alignment is 1,
  never a re-typed `const uint16_t*`, and it crosses the public ABI in
  `include/vt/ops.h`, so it wants its own row and spec rather than riding here.
  Raised by the fresh review of #2581 as its non-blocking finding 2.

* [#2605](https://github.com/mudler/vllm.cpp/issues/2605). doctest splits `-tc=`
  on commas, so a `TEST_CASE` name carrying one cannot be selected by it and a
  focused run by name prints `assertions: 0` and `SUCCESS!` at rc 0. #2601 was
  one instance and this row renames it; 1202 names tree-wide have the same
  shape, four of them in the two files this row touched. The durable answer is a
  checker that refuses the shape, diff-scoped or baselined so those 1202 do not
  red ordinary work. That changes gate semantics and wants its own spec and a
  red-before test, so it is filed rather than swept, and renaming the 1202 is
  not asked for.

* [#2606](https://github.com/mudler/vllm.cpp/issues/2606). `check-agent-record`'s
  issue-ownership check reads an untracked snapshot, and its own docstring says
  an absent snapshot is a SKIP that the caller must report and `--fail-on-skip`
  must redden. Neither exists: `skips` is populated and never read, the flag is
  not defined, and no workflow step runs `agent-issue-index.py --refresh`, so
  `.github/workflows/ci.yml:187` always runs it with the input absent. Measured
  on this row's own head: rc 1 with the snapshot present, rc 0 and no skip line
  with the same tree and that one file moved away. A local operator gets the red
  and CI gets the green, which is the wrong polarity for the surface that decides
  whether work lands. Filed rather than fixed here, because a record gate's
  semantics and a CI workflow are not a test rename.

## Now

`ACTIVE`.

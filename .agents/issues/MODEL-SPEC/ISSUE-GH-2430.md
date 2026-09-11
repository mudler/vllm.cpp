ID: ISSUE-GH-2430
Title: MakeTensor indexes vt::Tensor's rank-4 shape/stride arrays by an unchecked incoming rank, writing out of bounds for 5-D shapes
Row: MODEL-SPEC
State: OPEN
Kind: UNKNOWN
GitHub: 2430
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `-`
>
> `sanitize-cpu (address,undefined)` is red on `main`, and has been on every pull request
> that reaches the run stage. It is not caused by any of them. Seen on #2414 (whose entire
> diff is one workflow line and a comment), #2397 and #2396.
>
> ## The defect
>
> `include/vllm/model_executor/models/dense_device_glue.h:56`, in `dense_attn::MakeTensor`:
>
> ```cpp
> t.rank = static_cast<int>(shape.size());
> int64_t acc = 1;
> for (int i = t.rank - 1; i >= 0; --i) {
>   t.shape[i] = shape[static_cast<size_t>(i)];   // <-- line 56
>   t.stride[i] = acc;
>   acc *= t.shape[i];
> }
> ```
>
> `vt::Tensor` (include/vt/tensor.h:12,20-21) fixes `constexpr int kMaxRank = 4` with
> `int64_t shape[kMaxRank]` and `int64_t stride[kMaxRank]`. `MakeTensor` takes the incoming
> `shape.size()` as the rank and indexes both arrays by it, with **no bounds check**. A
> rank-5 shape therefore writes `shape[4]` and `stride[4]`, one past each array.
>
> UBSan on CI:
>
> ```
> include/vllm/model_executor/models/dense_device_glue.h:56:14: runtime error:
>   index 4 out of bounds for type 'long int [4]'
>     #0 vllm::dense_attn::MakeTensor(void*, vt::DType, vt::Device, std::vector<long> const&)
>     #1 vllm::dense_attn::DBuf::DBuf(vllm::dense_attn::Dev, vt::DType, std::vector<long> const&, void const*)
>     #2 DOCTEST_ANON_FUNC_2 tests/vllm/models/test_qwen4_exp_layer_loop.cpp:490
> ```
>
> Failing test: `test_qwen4_exp_layer_loop` (#200 of 686).
>
> This is not only a sanitizer complaint. `shape[4]` lands on the adjacent `stride[0]`
> under the usual layout, so a rank-5 `MakeTensor` silently corrupts the stride it is
> about to compute. It is quiet in a normal build.
>
> ## Why it stayed hidden
>
> `MakeTensor` has 645 call sites across `src include tests`. The rank contract is
> enforced nowhere: nothing refuses a rank above `kMaxRank`, so an over-rank shape is
> accepted, written out of bounds, and only UBSan objects.
>
> ## What to decide
>
> Two shapes, and this is a design decision rather than a one-line repair, which is why
> this is filed rather than fixed in flow:
>
> 1. Refuse it: `VT_CHECK(shape.size() <= kMaxRank, ...)` naming the rank and the caller.
>    Matches the project's refuse-loudly idiom, but turns whatever currently passes rank 5
>    into a hard failure, so the caller at `test_qwen4_exp_layer_loop.cpp:490` has to be
>    reconciled first.
> 2. Raise `kMaxRank`. Widens every `vt::Tensor` in the tree; needs a memory-footprint
>    argument.
>
> Either way the caller producing a rank-5 shape needs to be understood first: a 5-D shape
> reaching a rank-4 tensor type may itself be the bug.
>

## Resolution

-

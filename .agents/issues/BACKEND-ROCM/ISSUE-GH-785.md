ID: ISSUE-GH-785
Title: ROCm: PagedAttnPrefillSharedKWmma has never launched — the host-side #if defined(VT_ROCWMMA_OK) guard is always false, so d=256/d=512 prefill silently runs the scalar kernel
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: 785
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-14
Updated: 2026-08-18
Closed: 2026-08-18

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Summary
>
> `src/vt/rocm/rocm_paged_attn.hip:2071` guards both `PagedAttnPrefillSharedKWmma`
> launches with:
>
> ```cpp
> // Host must not launch WMMA stubs on non-gfx1200/1201 fatbin slices
> // (kernel body is #else empty there → silent garbage out).
> #if defined(VT_ROCWMMA_OK)
> ```
>
> `VT_ROCWMMA_OK` is defined at the top of the file only under
> `#if defined(__gfx1200__) || defined(__gfx1201__)`. **Those macros are defined
> on HIP's device compilation pass only, never on the host pass.** The guard is
> therefore always false on the host, and the enclosed launches are deleted from
> the host translation unit entirely.
>
> The intent of the guard is correct and the comment describing it is accurate.
> The mechanism does not work.
>
> **Consequence: the rocWMMA prefill path at `head_dim=256` and `head_dim=512`
> has never executed.** Every such prefill has silently fallen through to the
> scalar `PagedAttnPrefillSharedK` immediately below. Any performance attributed
> to the WMMA prefill kernel when it landed was the scalar kernel's.
>
> ## Evidence
>
> Preprocess the **host** compilation pass with the build's own command — flags
> taken verbatim from `build-hip/compile_commands.json`, only `-c` replaced with
> `-E --cuda-host-only`:
>
> ```sh
> clang++ … -E --cuda-host-only -o host_pass.i -x hip src/vt/rocm/rocm_paged_attn.hip
> ```
>
> | grepped in `host_pass.i` | count |
> |---|---|
> | `prefill_sharedk_wmma` — the string literal inside **both** launch sites | **0** |
> | `PagedAttnPrefillSharedKWmma` | 1 — the template definition only, no launch |
> | `rocwmma` — the header itself | **0** |
> | `PagedAttnPrefillSharedK` (scalar, the fallthrough) | 3 — definition **and** launches |
> | `PagedAttnDecodeGqaWmmaPartial` — a *runtime*-gated WMMA arm, for contrast | 2 — definition **and** launch |
>
> Both WMMA launch sites are gone; the scalar fallthrough is present; a
> runtime-gated WMMA launch in the same file survives. That last row is the
> control: it shows the disappearance is caused by the compile-time guard, not by
> anything about WMMA kernels generally.
>
> Independently confirmed with a standalone `.hip` probe on the same toolchain,
> which prints `HOST: __gfx1200__ is NOT defined` and
> `DEVICE: __gfx1200__ check result = 1`.
>
> ## Why no test caught it
>
> The bug's entire nature is that it is invisible to a green suite. The fallthrough
> is a *correct* scalar kernel, so every numerical test passes, every golden holds,
> and nothing declines. Only a performance measurement or a kernel trace would
> show it — and only if someone looked for the kernel name specifically.
>
> ## Suggested fix
>
> The same one already applied to the decode arm: replace the compile-time host
> guard with a cached runtime `hipGetDeviceProperties` arch check
> (`IsGfx1200Or1201()` already exists in this file), mirroring vLLM's own
> `is_navi_gpu()` in `csrc/rocm/attention.cu`. The kernel body's *own*
> `#if !defined(VT_ROCWMMA_OK)` guard is correctly evaluated per device target and
> must stay — it is what makes the non-gfx12 fatbin slice safe.
>
> ## What a red-first test looks like
>
> This needs a test that can see a dead launch, which no numerical test can:
>
> - a host-pass assertion (the preprocessor check above, as a script gate), or
> - a kernel-trace gate asserting `PagedAttnPrefillSharedKWmma` actually appears
>   when a d=256 prefill runs on gfx1200.
>
> Either fails today and passes after the fix.
>
> ## Also owed
>
> Re-measure whatever was attributed to the WMMA prefill kernel. Its landing
> evidence describes the scalar kernel.
>
> ## Scope note
>
> Found while working on an unrelated decode-attention change, which hit the same
> mechanism and fixed it locally for the decode dispatch only. This issue covers
> the **already-merged prefill** path, which was deliberately left untouched
> there: it is default-on production behavior in a different code path and
> deserves its own change rather than being folded into an unrelated one.
>

## Resolution

GitHub records closing pull request #1186 (https://github.com/mudler/vllm.cpp/pull/1186) merged on 2026-08-18 as commit `ae581da3e437ef1aa1749d8b20ac91346709cefe`. GitHub closed issue #785 on 2026-08-18.

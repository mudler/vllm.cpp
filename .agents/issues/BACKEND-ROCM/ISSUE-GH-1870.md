ID: ISSUE-GH-1870
Title: VT_GGUF_KEEP_QUANT=0 is unreachable on a 16 GiB ROCm card: the bf16 fallback OOMs with an allocator throw, not a refusal that names the cause
Row: BACKEND-ROCM
State: OPEN
Kind: bug
GitHub: 1870
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-24
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM`
>
> `VT_GGUF_KEEP_QUANT=0` is documented as a same-binary opt-out. On a 16 GiB
> discrete ROCm card it is **not reachable on any model that needs it**: the
> engine dies with a raw allocator failure rather than refusing with a message
> that names the cause.
>
> ```
> engine-fatal: EngineCore busy loop threw: vt rocm: hipMalloc: out of memory
> async-llm: output handler saw engine death: EngineCore encountered an issue.
> vllm-cli: completion failed (status 3)
> ```
>
> ## Reproduced
>
> `4b1154bc5`, RX 9060 XT (gfx1200, 16304 MiB), ROCm 7.2.3, `--device auto`,
> `--max-tokens 32`. Free VRAM asserted above 13 GiB with no resident model
> process before each arm.
>
> | model | file | keep-quant default | `VT_GGUF_KEEP_QUANT=0` |
> |---|---|---|---|
> | `Qwen3.6-14B-A3B-VibeForged-v2-Q4_K_M.gguf` | 7.87 GiB | runs, 13.000 / 13.145 t/s | **`hipMalloc: out of memory`** |
> | `Ornith-1.5-9B-Q4_K_M.gguf` | 5.23 GiB | runs, 18.393 / 18.574 t/s | **`hipMalloc: out of memory`** |
>
> Both architectures, MoE and dense, and both well inside the card when the
> weights stay compressed.
>
> ## The OOM itself is not the surprising part
>
> A Q4_K_M expanding to bf16 is roughly a 4x increase in resident weights. 5.23
> GiB becomes about 20 GiB and does not fit 15.92 GiB. That arithmetic is
> expected. Three things around it are the issue:
>
> 1. **The documented contract does not mention a precondition.**
>    `docs/ENVIRONMENT.md:94` reads: "`0` disables it and expands to BF16". It
>    presents a toggle. It does not say the toggle needs roughly 4x the file size
>    in device memory, so on this class of board the documented behaviour is
>    unreachable and the document is wrong by omission rather than by statement.
>
> 2. **The failure is an allocator throw, not a refusal.** `AGENTS.md` requires
>    an unimplemented or unreachable arm to "refuse ... with a message that names
>    the missing part". `hipMalloc: out of memory` names nothing: not the knob
>    that caused it, not the expansion, not the budget required. A user who sets
>    the variable to compare two arms gets a crash whose cause is not in the
>    message.
>
> 3. **It removes the same-binary A/B lever on this hardware.** The ON-versus-OFF
>    pair is how keep-quant's cost is attributed, and `AGENTS.md` requires a
>    same-binary A/B before a performance result is accepted. On a 16 GiB board
>    that pair no longer exists, so the cost of keep-quant cannot be measured
>    here at all. This is not hypothetical: it blocked the attribution
>    [#1863](https://github.com/mudler/vllm.cpp/issues/1863) wanted, and that
>    issue records the gap rather than the number because of it.
>
> ## Related gap, same area
>
> `kMoeGroupedGemmBf16` is unregistered on ROCm (`src/vt/rocm/rocm_ops.hip` has
> zero occurrences; CUDA has it). So even where device memory allowed the
> expansion, the bf16-expanded MoE arm has no provider on this backend. Whatever
> shape the fix takes, a refusal that names the missing piece is more useful than
> either failure mode as it stands.
>
> ## What a fix probably looks like
>
> Not proposed as a design, only to bound the scope: resolve the expanded
> residency requirement at load, compare it against the device budget, and refuse
> with a message naming `VT_GGUF_KEEP_QUANT`, the required bytes and the
> available bytes — before any allocation is attempted. Whether the knob should
> additionally be ignored with a warning rather than refused is a product
> decision, not something this issue settles.
>
> ## History
>
> Split out of [#1506](https://github.com/mudler/vllm.cpp/issues/1506), whose
> title claim ("ROCm has no `kMatmulBTQuant` provider") stopped being true when
> [#523](https://github.com/mudler/vllm.cpp/pull/523) registered the op on
> 2026-08-21. Its `1.73x peak RSS` finding is what survived, and this is that
> finding re-measured: on this card the penalty is no longer a ratio, it is a
> refusal to run. Filed separately rather than by re-scoping that issue, because
> the index is append-only and an edited row is duplicated rather than merged.
>
> Owned by `BACKEND-ROCM`.
>

## Resolution

-

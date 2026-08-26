# turboderp-org `exllamav3` — the EXL3 trellis format, and the DeepSeek-V4 support that reads it

EXL3 is a variant of **QTIP**: a procedural codebook encoding high-dimensional
vectors into tail-biting trellis structures, deviating from QTIP in how tensors
are regularized and packed (`doc/exl3.md:3`, `README.md:6,171`). vLLM defines no
such format — `layers/quantization/` at the parity pin
`5559679229bc961848b121ccdeaa8fa5d79bec98` registers no `exl3`, `exllamav3` or
trellis method — so `MODEL-DSV4-EXL3` is the fallback case the rule admits, and
this file is the pin it requires.

**Provenance of that vLLM negative: MEASURED, 2026-08-25.** It was first
established by the row's W1 spike and recorded in
[`../specs/model-dsv4-exl3.md`](../specs/model-dsv4-exl3.md), and it has since
been re-derived first-hand in the pinned checkout resolved from `.env`
(`VLLM_SOURCE`), at `HEAD = 5559679229bc`, which is the parity pin itself:

```sh
grep -rl -i 'exl3\|exllamav3\|trellis' --include='*.py' --include='*.cu' \
     --include='*.cuh' --include='*.h' .      # -> 0 files
ls vllm/model_executor/layers/quantization/*.py
# auto_awq auto_gptq awq_triton base_config bitsandbytes experts_int8
# fbgemm_fp8 fp8 fp_quant humming input_quant_fp8 kv_cache modelopt
# moe_wna16 mxfp4 qutlass_utils torchao  -- no EXL3, no trellis method
```

Zero matches across the whole tree, and no EXL3 entry among the registered
quantization methods. The earlier draft of this file said the negative could
not be re-derived here because no vLLM checkout was on the host; that was wrong
— the pinned checkout is the one `.env` names, outside the writing agent's
worktree. The claim now rests on a measurement rather than on a citation, which
matters because it is the single premise the whole registration stands on: if
vLLM implemented EXL3, this oracle would be inadmissible under the primary
rule.

**Why the fallback is DeepSeek-V4-shaped rather than only format-shaped.** The
pinned HEAD registers `DeepseekV4Model` as a first-class architecture
(`exllamav3/architecture/architectures.py:8,70`) alongside a DSpark drafter
(`deepseek_v4.py:10,30` -> `deepseek_v4_mtp.py`, 372 lines), a 1744-line
`exllamav3/modules/dsv4.py`, and three CUDA kernels with tests behind them:
`exllamav3/exllamav3_ext/{dsv4_compress.cu,dsa_topk.cu,hc_mix.cu}` and
`tests/test_dsv4_{cached,compress_kernel,state}.py`. Note the kernel path is
nested — `exllamav3/exllamav3_ext/`, not a top-level `exllamav3_ext/`.

**It keys on the tensor names our checkpoint actually ships.** `deepseek_v4.py`
builds `layers.{idx}` (`:134`), `{key}.ffn` (`:167`) and
`experts.{expert_idx}.{w1,w2,w3}` (`:172-174`), which composes to
`layers.0.ffn.experts.0.w1` — the name
`.agents/specs/model-dsv4-exl3.md` records reading out of
`exl3-layer-000-tp4-rank1.safetensors` as `layers.0.ffn.experts.0.w1.rank1`. The
staged `config.json` declares `architectures = ["DeepseekV4ForCausalLM"]`,
`model_type = "deepseek_v4"`, and
`quantization_config = {quant_method: "exl3", bits: 3.0, codebook: "mcg",
version: "rank-sliced-deepseek-v4-v1"}`. So the artifact and this upstream are
the same key scheme, not two conventions that happen to overlap.

**The checkpoint was quantized by an OLDER revision of this same repository.**
Its `config.json` carries
`hybrid_tr3_tail.exllamav3_revision = 787d1582267117d6ee83c90014f03b525b14754f`
(and `hybrid_tr3_tail.source_revision = 9e165c30e2704aec5d9d593cce3eebd58bbef1cb`),
read 2026-08-25 from the NAS-staged
`/mnt/nas_share/rc/ckpt/dsv4-flash-0731-spark-exl3/config.json`. Two limits on
that fact: the key is scoped to `hybrid_tr3_tail` rather than declared for the
whole artifact, and **`787d1582` is not reachable from the pin in the clone this
file was written against**, because that clone is `depth = 1` (`.git/shallow`
present, `git rev-list --count HEAD` = 1). Absence there is a property of the
clone, not evidence about upstream ancestry. Nobody should cite it as one.

## Scope, and what this oracle may not do

Use it for the EXL3 format and its kernels — trellis packing and the MCG
codebook, the `suh`/`svh` Hadamard-and-sign vectors, `had_r_128`, `exl3_gemm`,
the `m <= 8` GEMV arm, `exl3_moe` — and for DeepSeek-V4 behavior that follows
from reading a checkpoint stored in that format. `LinearEXL3.quant_type` is
`"exl3"` (`exllamav3/modules/quant/exl3.py:18`).

It is a mirror source for ONE quantization family and its execution, not a
design reference. Where vLLM defines behavior — attention, MLA, MoE routing,
sampling, scheduling, the serving path — vLLM wins, exactly as `AGENTS.md`
requires. "exllamav3 does it differently" is never on its own a reason to
diverge, and when vLLM registers an EXL3 method the row reconciles onto vLLM and
records the change in its spec.

## A prerequisite this oracle imposes on any run

**`tp_import_split` is a runtime IPC split, not a reader of pre-sliced files.**
The producer side is a `multiprocessing.shared_memory` arena
(`exllamav3/model/model_tp_shared.py:23,40`), and every importer takes its slice
by `consumer.recv(..., slice_dim, first, last)` out of that arena:
`TPTensorWrapper.tp_import_split` at `model_tp_shared.py:285-292`, and
`LinearEXL3.tp_import_split` at `exl3.py:285-329`, which slices `svh` on dim 0
and `trellis` on dim 1 in `first // 16 .. last // 16` units. Nothing in the tree
reads a `.rank{r}` filename: a `grep -rn 'rank{|\.rank\b|rank_%d|f"rank'` over
its Python returns only NCCL rank bookkeeping in
`exllamav3/model/model_tp_backend.py`.

Our artifact ships 43 layers x 4 `.rank{r}` shards on disk. Therefore a
**TP1-coalesced checkpoint is a prerequisite** before exllamav3 can load it on
one GB10 — which is what W1b's `LoadDeepseekV4Exl3` already produces — or the
run needs four GPUs that a single Spark does not have. This is a scheduling
fact, not a defect in either side.

## Gateability

`gateable = no`, and [#1901](https://github.com/mudler/vllm.cpp/issues/1901) owes
the measurement.

`AGENTS.md` admits an oracle as gateable only once it demonstrably BUILDS and
RUNS the model. Neither has happened here. Every `MODEL-DSV4-EXL3` gate that
exists today runs against the in-tree CPU reference
(`src/vt/cpu/cpu_exl3_dequant.cpp`) or against an independently derived
double-precision Sylvester H128 — never against an exllamav3 execution. The
row's own spec says so of its single real-checkpoint anchor: those spot values
"were NOT produced by running upstream's kernel", because `ext.reconstruct` is a
CUDA extension and the implementing host has no GPU. That removes transcription
error from our side and cannot detect a defect in exllamav3 itself, since both
sides read one source.

The two reasons it is unmeasured are current rather than permanent. The host
this file was written on has no CUDA toolkit (`which nvcc` and `which
nvidia-smi` both return nothing), and the fleet device that would do it read
`dgx:gpu0  unhealthy (no contact 7m56s)` from `rc devices` on 2026-08-25.

**No gate is invalidated by registering it today.** The registration closes a
policy gap — a row mirroring an unregistered upstream — rather than repairing a
measurement. The debt falls due at the row's **W3a**, before any end-to-end
token gate binds to this oracle.

```oracle-pin
id = exllamav3
role = secondary
upstream = https://github.com/turboderp-org/exllamav3
scope = the EXL3 trellis quantization format, its kernels, and the DeepSeek-V4 support the pinned HEAD carries, none of which vLLM or vLLM-Omni implements
pin = 2398c05635fbbad01a0a51dce63c85c6c8a8450e
pin_label = v1.4.3
pinned_on = 2026-08-25
gateable = no
evidence = #1901
```

**Identity, asserted rather than assumed.** The revision above was read from a
clone whose `origin` is `https://github.com/turboderp-org/exllamav3`, whose
worktree is clean (`git status --porcelain` empty) and whose `HEAD` is
`2398c05635fbbad01a0a51dce63c85c6c8a8450e`, subject "Merge branch 'pr-298' into
dev", authored 2026-08-23. `git rev-parse v1.4.3^{commit}` resolves to the same
SHA, so the pin and the tag are one commit, and `exllamav3/version.py:1` declares
`__version__ = "1.4.3"`. The repository is MIT (`LICENSE:1-3`, "Copyright (c)
2025 Turboderp"). Nothing here is a claim about upstream `master` today: this
record, like the checker that reads it, is network-free.

## Anchors used by the port

Read at the pin above. Re-verify before relying on them.

| Piece | Anchor |
|---|---|
| `LinearEXL3`, the four owned tensors | `exllamav3/modules/quant/exl3.py:16-40` |
| TP split semantics, inverted by W1b | `exllamav3/modules/quant/exl3.py:285-329` |
| MCG codeword decode | `exllamav3/exllamav3_ext/quant/codebook.cuh:67-75` |
| codeword window unpack | `exllamav3/exllamav3_ext/quant/exl3_dq.cuh:15-31` |
| H128 + sign composition | `exllamav3/modules/quant/exl3.py:227-237` |
| `had_r_128` activation transform | `exllamav3/exllamav3_ext/quant/hadamard.cu:88` |
| `exl3_gemm` shape table | `exllamav3/exllamav3_ext/quant/exl3_kernel_map.cuh:53-62` |
| DeepSeek-V4 tensor keys | `exllamav3/architecture/deepseek_v4.py:134,167,172-174` |
| DSpark drafter | `exllamav3/architecture/deepseek_v4_mtp.py` |

The format description these anchors ground lives in
[`../specs/model-dsv4-exl3.md`](../specs/model-dsv4-exl3.md) §"The format, as
measured". This file carries identity, not methodology.

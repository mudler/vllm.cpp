# ROCm gfx1200 (RX 9060 XT) — M2/M4: Gemma-3-1B-it clean; Qwen3-0.6B is a genuine cross-version near-tie

**Row:** `BACKEND-ROCM` (backend-matrix, `ACTIVE`).
**Claim:** unclaimed — this spec documents a closed investigation, not a landed
fix. Gemma-3-1B-it is real-oracle-verified clean (M4). Qwen3-0.6B's one
near-tie prompt is not a defect on either of our backends — see Outcome.
**Issue:** [#269](https://github.com/mudler/vllm.cpp/issues/269).
**Origin:** [issue #41](https://github.com/mudler/vllm.cpp/issues/41), the
gfx1200 M0/M1 report, first M2 attempt on this board. Distinct defect from
#201 (`hipblasGemmEx` overload) and #132 (`-O0` teardown deadlock), neither of
which reproduced here (and both since fixed on main, `ce134e1d`, picked up
when this investigation rebased onto `origin/main` 2026-08-10).
**Board:** AMD Radeon RX 9060 XT (`gfx1200`, Navi 44, RDNA4, discrete — no
reference tier). ROCm 7.2.3 (nixpkgs `rocmPackages.*`), hipClang/Clang 22.0.0.
M0/M1 independently MET on this board (separate report to #41).

---

## Outcome (2026-08-10)

### Gemma-3-1B-it: clean, unambiguous, real-oracle-verified — M4 met

Greedy (`--temperature 0`, `--max-tokens 8`), the same 6-prompt battery the
SACRED Gemma-3 gate uses (`scripts/gemma3-oracle-capture.py`), full text and
token ids, not just the last printed line:

| Prompt | Our CPU | Our ROCm | vLLM 0.19.1 (prebuilt gfx120X image) | vLLM `555967922` (exact pin, built from source) |
|---|---|---|---|---|
| "The capital of France is" | ` Paris.\n\nThe largest city in France` | identical | identical | identical |
| "The largest planet in our solar system is" | ` Jupiter, and it's a truly` | identical | identical | identical |
| "Water boils at a temperature of" | ` 100 degrees Celsius.\n\n` | identical | identical | identical |
| "The chemical symbol for gold is" | ` Au.\n\nThe chemical formula for gold` | identical | identical | identical |
| "The first president of the United States was" | ` George Washington.\n\nThe second president of` | identical | identical | identical |
| "Roses are red, violets are" | ` blue,\nI like to eat a` | identical | identical | identical |

**48/48 tokens identical across all four sources**, including two independent
real vLLM-ROCm installs on this exact board (a prebuilt AMD image and a
from-source build at this project's own pinned commit). This is the first
real exercise of the gemma `(1+w)` RmsNorm code path
(`vt::RmsNorm{gemma=true}` — sandwich norms + QK-norm,
`rocm_rmsnorm.hip:69-110`) plus GeGLU (`kGeluAndMul`/`rocm_ops.hip`) and the
dual per-layer RoPE-theta routing on this board, and it never comes near a
tie anywhere in the battery. No kernel changes needed or made. **M4, not just
M2, genuinely met for this model on this board — no caveat.**

### Qwen3-0.6B: not a kernel bug on either backend — a genuine, version-sensitive near-tie

`Qwen3ForCausalLM` (Qwen3-0.6B bf16), prompt `'The capital of france is'`,
greedy, `--temperature 0`, K=5 (all four sources below are internally
deterministic — 5/5 identical each):

| Source | Result |
|---|---|
| Our CPU backend | `' Paris. The capital of the United States'` |
| Our ROCm backend | `' 1000000'` |
| vLLM 0.19.1 (prebuilt gfx120X image) | `' 1000000'` — agrees with our ROCm |
| vLLM `555967922` (exact pin, built from source, ROCm 7.2.3) | `' Paris. The capital of the United States'` — agrees with our CPU |

**The real reference implementation does not agree with itself across its own
versions on this input.** That is the decisive fact: this cannot be framed as
"one of our backends is wrong," because the two things being compared *against*
also disagree with each other. Final-position top-5 logits (our own two
backends) explain the mechanism:

| | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| CPU | `ĠParis` 15.1216 | `Ġlocated` 15.0541 | `Ġ` 15.0483 | `Ġthe` 14.9142 | `Ġin` 14.5216 |
| ROCm | `Ġ` 15.1009 | `ĠParis` 15.0907 | `Ġlocated` 15.0620 | `Ġthe` 14.9496 | `Ġin` 14.5309 |

Same five candidates, same magnitudes, clustered within ~0.07 logit units of
each other on both backends (max logit ≈15). CPU resolves the cluster toward
`Paris` by 0.0675 over its runner-up; ROCm resolves it toward `Ġ` (a bare
leading-space token — content-free) by just 0.0102 over `Paris`. Picking a
content-free token derails the rest of greedy decoding into out-of-distribution
territory, which is why the completion reads as total garbage (`' 1000000'`)
rather than a merely-different-but-plausible word.

**Per-layer activation drift** (L2 norm of the last token's hidden row after
every decoder layer, CPU vs ROCm, same prompt):

| Layer | CPU L2 | ROCm L2 | rel. diff |
|---|---|---|---|
| L0 | 3.876 | 3.886 | 0.271% |
| L1 | 4.867 | 4.869 | 0.035% |
| L2 | 4.597 | 4.599 | 0.039% |
| L3 | 6.253 | 6.277 | 0.381% |
| L4 | 5.542 | 5.536 | 0.109% |
| L5 | 5.952 | 5.942 | 0.155% |
| L6 | 8.451 | 8.422 | 0.334% |
| L7 | 11.707 | 11.715 | 0.070% |
| L8 | 9.796 | 9.787 | 0.101% |
| L9 | 10.276 | 10.247 | 0.279% |
| L10 | 14.790 | 14.715 | 0.503% |
| L11 | 11.852 | 11.822 | 0.248% |
| L12 | 12.502 | 12.430 | 0.582% |
| L13 | 12.488 | 12.415 | 0.582% |
| L14 | 12.563 | 12.564 | 0.009% |
| L15 | 16.946 | 16.993 | 0.275% |
| L16 | 20.044 | 20.022 | 0.109% |
| L17 | 31.010 | 31.030 | 0.064% |
| L18 | 30.977 | 31.031 | 0.173% |
| L19 | 43.472 | 43.698 | 0.521% |
| L20 | 53.452 | 53.369 | 0.155% |
| L21 | 66.609 | 66.510 | 0.148% |
| L22 | 79.621 | 80.029 | 0.513% |
| L23 | 85.998 | 85.555 | 0.515% |
| L24 | 127.735 | 128.331 | 0.466% |
| L25 | 104.664 | 104.763 | 0.095% |
| L26 | 112.977 | 112.578 | 0.353% |
| L27 (final) | 216.317 | 217.610 | 0.598% |

Divergence is already present at **layer 0** (0.271%) — rules out "something
breaks at layer N" entirely. It oscillates in a tight 0.01%–0.6% band for all
28 layers with no jump, no blow-up, sometimes CPU higher and sometimes ROCm
higher. This is the textbook signature of ordinary bf16 rounding +
reduction-order noise (hipBLASLt's GEMM accumulation order vs. CPU's fixed
sequential order accumulating independently at every op), not a localized
defect anywhere in the stack. It happens to be enough, by the final layer, to
tip an unusually thin 0.07%-of-max-logit tie — and which way it tips depends
on the exact kernel/version path, ours or vLLM's own.

**What was ruled out along the way**, in order:
1. The `is_cuda()`-vs-`is_cpu()` host-pointer-aliasing defect at
   `dense_attn_block.h:181` (`ResidentWeight`) — already fixed generically,
   confirmed correct for `kROCM` in current source.
2. The embedding gather itself — pulled `model.embed_tokens.weight` directly
   from the safetensors file, ran `vt::Embedding` on CPU and ROCm at the real
   row indices 47587 ("france") and 9625 ("France"), both **bit-exact**
   against the CPU oracle (see `scratch_diag_embed.cpp`, not for landing).
3. A discrete localized op bug — the per-layer drift table above shows none;
   drift is present from layer 0 and grows uniformly.
4. **"One implementation is simply correct"** — ruled out directly: the exact
   same reference implementation, at two different points in its own history,
   lands on both sides.

**Debug instrumentation used, then reverted** (not landed, not left in the
tree): two `VT_DEBUG_LOGITS=1` hooks in `sampler.cpp` (`greedy_argmax_host`
and the async `Sampler::forward` path — CPU and ROCm take different code
paths by default, both needed instrumenting for a fair comparison) and one
`VT_DEBUG_LAYERS=1` hook in `qwen3.cpp`'s `ForwardLayers` loop.

## Real-oracle setup (2026-08-10, reproducible)

Docker was the practical path — pip/PyTorch-vs-NixOS's non-FHS layout is what
made earlier oracle attempts finicky here, and Docker sidesteps it entirely.
Two tiers, cheapest first:

**Tier 1 — prebuilt, fast, close-but-not-pinned.** AMD ships an image
explicitly targeting this arch family:

```sh
docker pull rocm/vllm:rocm7.13.0_gfx120X-all_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1   # 10.8 GB
docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  --shm-size=8g -v <model-dir>:/models/<name>:ro \
  rocm/vllm:rocm7.13.0_gfx120X-all_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1 \
  python3 -c '<vllm.LLM(...) offline-inference script>'
```

**Tier 2 — exact pinned commit, built from source.** No gfx120X-targeted
prebuilt exists at this project's pin (`555967922`, RDNA4 is too new for AMD
to have shipped one), but `rocm/vllm-dev:base` — the same base image vLLM's
own official `docker/Dockerfile.rocm` builds from — already carries ROCm
7.2.3 (**matches this board's native build exactly**), PyTorch
2.12.0+git6bbd260, and Triton 3.7.1 with `gfx1200` in `PYTORCH_ROCM_ARCH`.
`requirements/rocm.txt` at the pin carries no torch/torchvision/torchaudio
pin, so installing it does not clobber the base image's ROCm-built torch.
Full recipe:

```sh
docker pull rocm/vllm-dev:base   # 10.4 GB

cat > Dockerfile.rocm-oracle <<'EOF'
FROM rocm/vllm-dev:base
ENV PYTORCH_ROCM_ARCH=gfx1200
ENV VLLM_TARGET_DEVICE=rocm
ENV MAX_JOBS=6
# setup.py imports setuptools_rust unconditionally; give it a real toolchain.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
RUN git clone https://github.com/vllm-project/vllm.git /vllm-src
WORKDIR /vllm-src
RUN git checkout 555967922
RUN python3 -m pip install --upgrade pip setuptools wheel setuptools-rust
RUN python3 -m pip install -r requirements/rocm.txt
RUN python3 -m pip install --no-build-isolation -v -e .
EOF

docker build -f Dockerfile.rocm-oracle -t vllm-rocm-oracle:555967922-gfx1200 .
```

Compiled clean in ~6.5 minutes (hipify reported **zero unsupported CUDA
function calls**), producing `vllm-0.23.1rc1.dev1511+g555967922.rocm723` —
the project's exact pinned vLLM commit, natively built for `gfx1200` on this
board's own ROCm version. `torch.cuda.get_device_properties(0).gcnArchName`
reports `gfx1200`, `current_platform` resolves to `RocmPlatform`, in both
tiers.

## What this means for the row

**Gemma-3-1B-it: M4 genuinely met, no caveat.** Real-oracle-verified,
token-exact, two independent oracles including this project's own exact pin.

**Qwen3-0.6B: not a row defect, and not closeable by a strict token-exact
gate on this specific prompt.** Four independent measurements — our CPU, our
ROCm, and two different real vLLM versions — split 2-and-2, and the two real
vLLM results are individually deterministic. That is direct, measured proof
this input is a genuine near-arbitrary tie, not a bug in either of our
backends or in either oracle. This is exactly the case
[`near-tie-distributional-gate`](../verification.md) exists for: the right
bar here is "our token is a member of the reference's own K-run/cross-version
output set," not strict equality on one arbitrarily-chosen reference build.
No kernel fix is owed on this row. This board joins the already-documented
France/Italy Qwen3 near-tie (elsewhere in the project) as the same phenomenon
class, now with the strongest possible confirmation: the reference itself
doesn't hold still on it.

## Environment note (reproducibility)

This board is accessed through `nix develop .#rocm-shell` (local, uncommitted
`flake.nix` addition — see the M0/M1 report) for the ROCm/HIP `build-hip`
build itself; the oracle containers are unrelated to that and need no ROCm
toolchain on the host beyond the kernel driver + `/dev/kfd`/`/dev/dri`.
`ROCM_PATH` for `build-hip` is nixpkgs' `clr` output (has
`lib/cmake/hip-lang`); hipBLAS/hipBLASLt/hipblas-common are merged into a
small writable overlay at `~/.cache/vllm-cpp-rocm-overlay` because nixpkgs
ships them as separate store paths. None of this affects either finding —
both reproduce identically build-to-build, deterministic at
`--temperature 0`.

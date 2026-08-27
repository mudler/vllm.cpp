# Oracle recipes

One script per upstream oracle that has to be *stood up* before it can answer
anything — installed at a pinned revision, pointed at a local checkpoint, and
made to actually generate. The pin each script asserts belongs to the matching
record in [`.agents/oracles/`](../../.agents/oracles/); the script is the thing
that proves the record.

## `music3_oracle.py` — MiniMax-Music3 via `diffusers`

Primary oracle for
[`.agents/specs/minimax-music3.md`](../../.agents/specs/minimax-music3.md)
(#672). The pin is an **unmerged PR head**, so it is a commit and never a branch
name: `huggingface/diffusers#14456` at
`c6da9936e4bda83107943a16eb8682e9a37d8527`. diffusers `main` does not carry this
model at all.

### The venv

`torch` is the expensive part and every box that can run this already has one, so
the venv inherits it rather than installing a second copy. On a box whose torch
lives in the *user* site directory (`~/.local/lib/pythonX.Y/site-packages`),
`--system-site-packages` alone does not reach it — a venv disables user site — so
add it back with a `.pth`:

```sh
python3 -m venv --system-site-packages ~/venvs/music3-oracle
SP=~/venvs/music3-oracle/lib/python3.12/site-packages
echo "$HOME/.local/lib/python3.12/site-packages" > "$SP/_user_site_torch.pth"
~/venvs/music3-oracle/bin/pip install \
    "git+https://github.com/huggingface/diffusers@c6da9936e4bda83107943a16eb8682e9a37d8527" \
    "transformers==5.14.1" soundfile
~/venvs/music3-oracle/bin/python -c "import torch, transformers, diffusers; \
    print(torch.__version__, transformers.__version__, diffusers.__version__)"
```

The venv's own `site-packages` precedes the `.pth` path, so `transformers` and
`diffusers` resolve to the pinned copies and `torch` to the inherited one. The
script prints the resolved path of each, so a mis-shadowed import is visible in
the manifest rather than silent.

### The checkpoint

The **diffusers arm** only — `condition_encoder/`, `language_model/`,
`rvq_depth_decoder/`, `scheduler/`, `tokenizer/`, `transformer/`, `vocoder/`,
plus `modular_model_index.json`; ~26.6 GiB. The native arm (`qwen_7B/`,
`flowmatching_vae.pth`, `dav.pth`) is deliberately not fetched — spec §2.

Naming the components is what was actually used to fetch this copy, and it is the
safer direction: an allow-list cannot pick up a 29 GB native arm that a new
exclude pattern fails to match.

```sh
hf download MiniMaxAI/MiniMax-Music3 --local-dir "$CHECKPOINT_ROOT/minimax-music3" \
    --include 'condition_encoder/*' --include 'language_model/*' \
    --include 'rvq_depth_decoder/*' --include 'scheduler/*' --include 'tokenizer/*' \
    --include 'transformer/*' --include 'vocoder/*' \
    --include 'modular_model_index.json' --include 'config.json'
```

`modular_model_index.json` embeds the *hub repo id* in every component spec, so
the script overrides `pretrained_model_name_or_path` with the local directory and
sets `HF_HUB_OFFLINE=1`. Without both, a typo in `--checkpoint` becomes a silent
57 GB download of whatever revision the hub is serving that day. It verifies
shard counts and sizes against each `*.index.json` first, because a partial
download otherwise presents as a confusing failure ten minutes into a load.

### Running it

```sh
~/venvs/music3-oracle/bin/python tools/oracle/music3_oracle.py \
    --checkpoint "$CHECKPOINT_ROOT/minimax-music3" \
    --out tests/parity/goldens/minimax_music3_oracle \
    --device cuda
```

`--facts-only` loads every component and checks spec §1 without generating.
`--device cpu` works and is what the committed goldens were captured on; it is
slow (see below) but needs no GPU and no `flock` on a shared one.

### The dtype policy, and why `on-disk` is not the default

The released packaging stores the language model and the RVQ depth decoder in
bf16 and the condition encoder, transformer and vocoder in fp32 (spec §2.1).
**Loading every component at its on-disk dtype does not run.** The pipeline casts
in exactly two places — `denoise.py:83` condition → `transformer.dtype` and
`decoders.py:84` latents → `vocoder.dtype` — so the condition encoder and the
depth decoder both consume the language model's hidden states uncast and raise

```
RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same
```

at `condition_embedder_minimax_music3.py:64`. Every runnable configuration
satisfies `dtype(language_model) == dtype(rvq_depth_decoder) ==
dtype(condition_encoder)`, leaving the transformer and vocoder free. Hence
`--dtype-policy reference` (the default): bf16 autoregressive half, fp32 acoustic
half, which is upstream's conversion default for the DiT and the vocoder and what
SGLang-Omni's README says it runs. `--dtype-policy on-disk` is kept so the
failure stays reproducible; `bf16` is the uniform dtype upstream's own docs
prescribe.

### What it costs

Measured 2026-08-14, CPU, 20 cores, checkpoint over SMB: 111 s to load, 926 s to
generate 1.0 s of audio at `--audio-duration 1.0 --steps 4`. The autoregressive
stage dominates — it runs 8.6B params per 40 ms frame — so runtime scales with
`audio_duration` far more steeply than with `--steps`.

## `ltx2_oracle.py` — LTX-2.5 via Lightricks' own `LTX-2` runtime

Secondary oracle for [`.agents/specs/ltx-2-5.md`](../../.agents/specs/ltx-2-5.md)
(#1864). The pin is a commit and never a branch, because the repository carries
**no tags**: `Lightricks/LTX-2` at
`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`, the merge of `Lightricks/LTX-2#273`.

This script is what moved [`.agents/oracles/ltx-2.md`](../../.agents/oracles/ltx-2.md)
off `gateable = no`. Before it, thirteen tracked scripts imported `ltx_core` /
`ltx_pipelines` and every one of them ran individual *modules* at reduced
dimensions on synthetic PRNG weights — the committed goldens say so in their own
headers, "no weight byte is checked in". Nothing had run the model.

### The venv

`--extra natten` pins torch to `2.13.0+cu132` and pulls a CUDA extension. The
**conv** video VAE needs none of it, so the venv installs plain torch and takes
the conv arm. `transformers` must stay **below 5.15**, and that is upstream's pin rather than
this project's finding: `packages/ltx-core/pyproject.toml:18` reads
`transformers>=5.8.0,<5.15`, and the comment above it at `:13-17` gives the
reason — 5.15.0 routes config attribute access through the heterogeneity layer,
which raises `AmbiguousGlobalPerLayerAttributeError` on a global
`config.head_dim` that its own `gemma4_unified` code still reads, so the Gemma-4
text encoder cannot be built at all. **Quoted, not reproduced**: the 2026-08-27
run used 5.14.1, so it never exercised the failure.

```sh
python3 -m venv ~/venvs/ltx2-oracle
~/venvs/ltx2-oracle/bin/pip install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu130
git clone https://github.com/Lightricks/LTX-2 /tmp/ltx2/LTX-2
git -C /tmp/ltx2/LTX-2 checkout fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca
~/venvs/ltx2-oracle/bin/pip install \
    -e /tmp/ltx2/LTX-2/packages/ltx-core \
    -e /tmp/ltx2/LTX-2/packages/ltx-pipelines
```

Note the index root `https://download.pytorch.org` answers **403** to a bare
`GET`; the `/whl/cuNNN` paths answer 200. Re-measured from the coordinator on
2026-08-27 (`curl -s -o /dev/null -w '%{http_code}'` → `403` on `/`, `200` on
`/whl/cu130`). Probing the root and concluding "no egress" is a false negative
this recipe already tripped over once.

### The checkpoints

Four of `ModelPaths`' five slots — a one-stage render does not need the
`duration_head`, and an explicit `--num-frames` then satisfies
`require_num_frames_source`. All four are `Lightricks/LTX-2.5`; the sizes and
digests belong to [`docs/USAGE.md`](../../docs/USAGE.md) and the script re-checks
them at load.

The **bf16** Gemma-4 tower is required and cannot be substituted. Upstream
contains **zero** `torchao` references (`grep -rin torchao` over the clone
returns no line), so the NVFP4-torchao tower this project's own renders use is
not loadable here; and `_ENCODE_MODEL_TYPES` plus `_check_gemma_version` reject
a stock Google Gemma 4. There is no precomputed-embedding bypass for
text-to-video either — only `HDRICLoraPipeline` accepts embeddings, and it is
video-to-video.

### Running it

```sh
~/venvs/ltx2-oracle/bin/python tools/oracle/ltx2_oracle.py \
    --ltx2-source /tmp/ltx2/LTX-2 \
    --checkpoints "$CHECKPOINT_ROOT/ltx-2.5" \
    --out /tmp/ltx2/oracle-out \
    --device cuda
```

`--offload cpu` is the default and is deliberate. Upstream's `OffloadMode`
docstring puts `NONE` at "~28 GB for LTX-2" resident, and free VRAM on a GB10 is
not a constant to plan against: the five `#1864` job logs report
`vram free=` **33.7, 58.6, 84.3, 89.2 and 99.2** GiB of the same 119.6 GiB
unified pool, depending on what the page cache and the previous tenant left
behind. This repository has OOM-**rebooted** that box before, for causes it
records elsewhere and none of which is an LTX offload mode
(`.agents/environment.md:668`, `:1767`, `.agents/roadmap_v1.md:47`); the reason
to prefer `cpu` here is that the render is the deliverable and its speed is not. `cpu` streams the
tower and the DiT layer-by-layer at roughly 5 GB VRAM. The render is the
deliverable, not its speed.

The script writes `upstream-render.mp4`, decodes it to
`upstream_frames/frame_%06d.ppm` plus `audio.wav` — the layout
[`scripts/ltx25-render-compare.py`](../../scripts/ltx25-render-compare.py) reads,
since that tool takes PPM directories and never an mp4 — and records every input
byte in `ltx2_oracle_manifest.json`.

### The toolchain, which is not optional and is not obvious

A render that never leaves Python still needs a **C compiler and the CPython
headers**, because the Gemma-4 text tower's RoPE step reaches a Triton kernel.
`modeling_gemma4_unified.py:278` evaluates
`inv_freq_expanded.float() @ position_ids_expanded.float()`; `torch._native`'s
`eager_router` dispatches that `[B,D,1] @ [B,1,S]` shape to `bmm_outer_product`'s
Triton implementation; and before Triton can launch anything it JIT-builds a
CPython extension, `cuda_utils`, by shelling out to `gcc`. On a stock Ubuntu
24.04 image that compile fails on the first line of the translation unit:

```text
tu.c:1:10: fatal error: Python.h: No such file or directory
    1 | #include <Python.h>
```

Measured 2026-08-27 on the `dgx:gpu0` worker: `gcc 13.3.0` present,
`/usr/include/python3.12` **absent**, `python3-config` **absent**. `apt-get
install -y python3-dev` is the whole fix, and it belongs in the setup step rather
than the render step, because the failure otherwise arrives twenty seconds into a
render that has already staged 65.3 GiB of checkpoints and loaded a 22 B model.
After the install the same `cuda_utils.c` compiles, `bmm_outer_product`
dispatches, and a plain Triton kernel compiles and launches.

The **link** half of the same command is a separate question and was fine here:
`-l:libcuda.so.1 -L/lib/aarch64-linux-gnu` returned `rc=0` on `dgx` before any
install. It fails on the `thor:gpu0` worker of the same image (`rc` job
`2006eb26-6737-4c2e-b9f5-7e7f84c18251`, 2026-08-26:
`/usr/bin/ld: cannot find -l:libcuda.so.1`), which is Tegra and keeps libcuda
under `/opt/nvidia/l4t-gpu-libs/nvgpu`. Triton takes a
`TRITON_LIBCUDA_PATH` override for that case; it changes where the linker looks
and nothing about what computes, which is why it is admissible and disabling the
JIT is not.

**Do not route around it.** `torch` will fall back to an eager implementation if
asked, and that changes which kernel computed the reference — which is the one
thing a reference render may not do. Provision the toolchain; do not disable the
JIT.

Two properties of the failure are worth knowing before you meet it.
`triton/runtime/build.py` calls `subprocess.check_call(cc_cmd,
stdout=subprocess.DEVNULL)` with **no `stderr=` argument**, and
`CalledProcessError` carries only the argv and the return code, so the compiler's
own diagnostic is not in the traceback and not in the log. And a preflight that
only imports Triton proves nothing: the extension is built lazily, on the first
kernel launch. Compile something.

### What it produced, 2026-08-27

The first time upstream `Lightricks/LTX-2` RENDERED in this tree. Two tracked
scripts had already touched real checkpoint bytes — `measure-ltx2-prompt-adaln.py`
forwards one `AdaLayerNormSingle` out of a 21 B DiT, and
`measure-ltx2-keyframes-meta.py` builds on the meta device — so "real weights" is
not the line that was crossed. Running the model is. `rc` job `44159e4f-f810-4c16-a4ab-67a0b3019f0c` on `dgx:gpu0`, 5m31s of
device time, of which the render was 93.8 s and hashing the four checkpoints
(70,099,185,228 bytes, 65.3 GiB) was 149 s. Output: 25 frames at 320x192 and 1.02 s of stereo 48 kHz audio.

The manifest, the mp4 and the digests of every frame are committed under
[`tests/parity/goldens/ltx2_oracle/`](../../tests/parity/goldens/ltx2_oracle/),
and that manifest path is the `evidence` field of
[`.agents/oracles/ltx-2.md`](../../.agents/oracles/ltx-2.md), which the pin
checker requires to exist. The 25 PPM frames themselves are not committed; they
are on the NAS at `/workspace/ltx2-oracle/out/upstream_frames/` and `SHA256SUMS`
makes a later copy checkable.

### The identity assert does not reach the render subprocess (#2055)

Read this beside the script's own docstring, which claims more than the code
delivers. `ltx2_oracle.py` asserts the revision and the resolved `ltx_*` import
paths **in the parent**, and then renders in a child started with `python -m`,
which puts the **current working directory** on that child's `sys.path`. The
parent's `find_spec` never consults it, so a directory holding a decoy
`ltx_pipelines`, made the CWD, is imported by the process that loads the weights
while the process that checked identity sees nothing. A fresh reviewer of #2053
demonstrated exactly that.

The 2026-08-27 render is unaffected — `render.sh` issues no `cd` and
`/workspace/ltx2-oracle/` holds no `ltx_*` package — and the manifest records
module origins inside the pinned clone. The fix is `-P` (or `PYTHONSAFEPATH=1`)
plus an explicit `cwd=` on the subprocess, with the reviewer's decoy as the
red-first test.

**Why the script is not edited here.** Its sha256 is the provenance chain that
`.agents/oracles/ltx-2.md`'s `gateable = yes` rests on: the job log prints the
digest it executed, and it equals the committed file byte for byte. Changing the
file for a hardening that altered no result would trade a verifiable fact for a
better-worded comment. #2055 owns the fix, in a change that can re-run it.

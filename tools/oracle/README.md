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

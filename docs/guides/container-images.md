# Run container images

Use the published container lane that matches your backend.

Published to one GHCR package with the lane in the tag. Every lane is a
`linux/amd64` + `linux/arm64` manifest, so the same tag works on both.

| tag | what it is |
|---|---|
| `:<version>-cuda` / `-vulkan` / `-cpu` | **immutable.** Never republished |
| `:latest-cuda` / `-vulkan` / `-cpu` | moves to the newest **release** |
| `:latest` | the **cpu** lane, so pulling it on a machine with no accelerator gets a working server rather than a library-load failure |
| `:main-cuda` / `-vulkan` / `-cpu` | moves with **main**: rebuilt when container infrastructure changes and nightly otherwise. Convenience, not a release, no support claim |

The entrypoint is `vllm-server`, so flags go straight after the image name and
the server keeps its own default of `0.0.0.0:8000`:

```sh
docker run --rm -p 8000:8000 \
  -v /path/to/models:/models:ro \
  ghcr.io/mudler/vllm.cpp:latest \
  --model /models/Qwen3.6-35B-A3B
```

For the CUDA lane, the GPU driver comes from the host through the container
runtime; the image carries only the CUDA *runtime* libraries it links:

```sh
docker run --rm --gpus all -p 8000:8000 \
  -v /path/to/models:/models:ro \
  ghcr.io/mudler/vllm.cpp:latest-cuda \
  --model /models/Qwen3.6-35B-A3B
```

`/models` is the weights mount and `/cache` is the tokenizer/HF cache. The
container runs as **uid 1000**, so `/cache` must be writable by it and the
weights under `/models` must be READABLE by it. A model file with mode `0600`
owned by another uid fails as `safetensors: cannot open file`, which reads like
a corrupt checkpoint rather than a permissions problem.

## Picking the right flags for your GPU

The two NVIDIA families need **different** invocations, and this is verified on
both rather than inferred:

| host | verified on | flags |
|---|---|---|
| SBSA / datacenter arm64, x86_64 | GB10 `sm_121a` | `--gpus all` |
| Jetson / Tegra (L4T) | AGX Orin `sm_87`, L4T R36.4.3 | `--runtime nvidia --gpus all` |

On Jetson, `--gpus all` **alone is refused** ("invoking the NVIDIA Container
Runtime Hook directly ... is not supported"), and `--runtime nvidia` **alone**
starts a container with no driver that dies on `libcuda.so.1: cannot open
shared object file`, which looks like a broken image rather than a missing
flag. Use both:

```sh
docker run --rm --runtime nvidia --gpus all -p 8000:8000 \
  -v /path/to/models:/models:ro \
  ghcr.io/mudler/vllm.cpp:latest-cuda \
  --model /models/Qwen3-0.6B
```

That exact recipe was run on an AGX Orin with `Qwen/Qwen3-0.6B`: the server
serves `/v1/completions` and `tegrastats` shows `GR3D_FREQ` at 95-97% during
generation, so decode is on the GPU.

## If the server exits at startup

| symptom | cause |
|---|---|
| `safetensors: cannot open file` | the weights are not readable by **uid 1000**. The container runs as uid 1000; a `0600` model owned by another user fails here and looks like a corrupt checkpoint |
| `libcuda.so.1: cannot open shared object file` | no driver in the container, on Jetson, add `--gpus all` alongside `--runtime nvidia` |
| `--model <dir> is required` | the server takes flags directly; everything after the image name goes to `vllm-server` |

## Building and validating an image locally

One Dockerfile, one target per lane. The builder stage runs the same
`scripts/build-*-release.sh` the release workflow runs, so there is no second
build definition to drift:

```sh
docker build -f docker/Dockerfile --target cpu \
  --build-arg VERSION=0.0.1 \
  --build-arg SOURCE_SHA=$(git rev-parse HEAD) \
  --build-arg JOBS=$(nproc) \
  -t vllm-cpp:local-cpu .
```

Then gate it. Without `--model` the validator checks configuration and layout
and says plainly that the image has no runtime evidence; with one it also boots
the server, requires `/health` and `/version`, runs the image's own declared
healthcheck, and requires a clean SIGTERM shutdown:

```sh
python3 scripts/validate-container-image.py \
  --image vllm-cpp:local-cpu --lane cpu --version 0.0.1 \
  --model /path/to/opt-125m
```

`scripts/check-container-matrix.py` keeps `release/container-matrix.json` and
the Dockerfile agreeing about lanes, tags and digest-pinned bases;
`scripts/check-container-workflow.py` holds the publish workflow to its
least-privilege stages. Both run in preflight and CI.

To exercise the release pipeline without publishing anything, trigger its
manual entry point:

```sh
gh workflow run release.yml --ref main
```

Manual runs are always dry runs. Publication additionally requires the exact
tag declared in `release/release-version.json` (currently
`v0.0.3-pre.1`), a release matrix whose required lanes are all marked
ready, successful verification and attestation jobs, and approval of the
protected `release` environment. Build and verification jobs have read-only
repository permissions; only attestation receives OIDC authority, and only the
final protected job receives `contents: write`. The current declaration is a
prerelease; the publisher must pass GitHub's prerelease flag and a manual dry
run cannot publish.

Any OpenAI client works by pointing its `base_url` at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(client.completions.create(model="Qwen3.6-35B-A3B",
                                prompt="The capital of France is",
                                max_tokens=64).choices[0].text)
```

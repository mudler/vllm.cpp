# Run container images

Use the published container lane that matches your backend.

The project publishes one GHCR package with the lane in the tag. Every lane is a
`linux/amd64` + `linux/arm64` manifest, so the same tag works on both.

| Tag | Meaning |
|---|---|
| `:<version>-cuda`, `-vulkan`, or `-cpu` | Immutable and never republished |
| `:latest-cuda`, `-vulkan`, or `-cpu` | Moves to the newest release |
| `:latest` | Selects the CPU lane, which works without an accelerator |
| `:main-cuda`, `-vulkan`, or `-cpu` | Moves with `main`. The project rebuilds it after container infrastructure changes and on the nightly schedule. It is not a release. |

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
container runs as UID `1000`, so that user must be able to write to `/cache`
and read the weights under `/models`. A model file with mode `0600` owned by
another user fails as `safetensors: cannot open file`, which can look like
a corrupt checkpoint rather than a permissions problem.

## Pick the flags for your GPU

The two NVIDIA families need different invocations. Both rows were verified on
the listed hardware.

| Host | Verified on | Flags |
|---|---|---|
| SBSA / datacenter arm64, x86_64 | GB10 `sm_121a` | `--gpus all` |
| Jetson / Tegra (L4T) | AGX Orin `sm_87`, L4T R36.4.3 | `--runtime nvidia --gpus all` |

On Jetson, the runtime refuses `--gpus all` by itself ("invoking the NVIDIA
Container Runtime Hook directly ... is not supported"). Using only
`--runtime nvidia`
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

| Symptom | Cause |
|---|---|
| `safetensors: cannot open file` | UID `1000` cannot read the weights. A `0600` model owned by another user fails here and can look corrupt. |
| `libcuda.so.1: cannot open shared object file` | no driver in the container, on Jetson, add `--gpus all` alongside `--runtime nvidia` |
| `--model <dir> is required` | the server takes flags directly; everything after the image name goes to `vllm-server` |


Point any OpenAI client `base_url` at the server:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(client.completions.create(model="Qwen3.6-35B-A3B",
                                prompt="The capital of France is",
                                max_tokens=64).choices[0].text)
```

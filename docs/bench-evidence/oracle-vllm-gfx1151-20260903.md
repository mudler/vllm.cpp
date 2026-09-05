# The PRIMARY oracle runs on `gfx1151`, and it does not pass this gate either

Issue [#2740](https://github.com/mudler/vllm.cpp/issues/2740), row
`BACKEND-GATE-ROCM-LLAMACPP`, spec
[`oracle-vllm-gfx1151.md`](../../.agents/specs/oracle-vllm-gfx1151.md).

**No speed, latency or throughput figure is taken anywhere below, and no memory
figure is compared.** `AGENTS.md` §Gates admits a performance result from an arm
only after that arm's declared token gate passes, and the Qwen3.8-27B Q4_K_M
ROCm arm's gate reads `FAIL` at 3 of 6 ([token-gate
v2](qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md)).
[#2497](https://github.com/mudler/vllm.cpp/issues/2497) has already had one
measurement retracted for exactly this. This document changes no verdict of
#2497 or [#2534](https://github.com/mudler/vllm.cpp/issues/2534).

Memory figures do appear, and they are identity rather than performance: 16.08
GiB of loaded weights, 64 GiB of board memory, the 53.8 GB bf16 checkpoint
sitting beside the GGUF. They are here to prove which artifact was fed, and none
of them measures this engine against another. The verbatim `rc` logs committed
beside this file carry vLLM's own `tqdm` `est. speed ... toks/s` progress text
on four lines of `job-phase3.txt`. Nothing here reads it, and a captured log is
not edited to make a sentence true.

## Why this was asked at all

That arm is gated against llama.cpp `b10451`, a **secondary** oracle.
`AGENTS.md` §"When vLLM has no implementation" admits one only where vLLM
implements nothing. Everyone working this campaign, including the agents that
wrote the evidence above, has assumed vLLM cannot run on `gfx1151`. **Nobody had
measured it.** Reading the pin contradicts the assumption, and the contradiction
is the reason #2740 exists.

## What ran

`rc` jobs on `strix:gpu0`, worker `rc-worker-lcjhd`, boot id
`a5bc8128-f6ad-4767-8614-6923f88032e1`, x86-64, 32 cores, 62 GiB host RAM,
2026-09-03. Nothing reached the box by `ssh`. Raw logs on the share under
`/mnt/nas_share/rc/vllm-gfx1151-2740/out/`, and the four job logs plus the exact
build and run recipe are committed beside this file in
[`oracle-vllm-gfx1151-20260903/`](oracle-vllm-gfx1151-20260903/):
`phase1b.sh` (torch arch), `phase2.sh` (build), `phase3.sh` (plugin and
generation), `phase4.sh` (the repeat legs and the plugin object), `gen_rocm.py`, the four
job logs as `job-phase*.txt` (renamed from `.log`, which `.gitignore` drops),
the probes, the four token files, and `compare.py`.

The `rc` job ids, in order:

```text
5dfdcc50-2220-4649-b5a1-99f20dd547de  phase 1b  torch gfx coverage
dcf0d65e-d61f-4364-b31d-80ce6f14affc  phase 2   the build that succeeded
b2a8d0e1-67ea-4275-bb5a-7006bd1d06aa  phase 3   plugin + eager x2 + compiled
bb2ffe74-83e9-4fed-9a35-f9367fefe76b  phase 4   compiled x2 + the plugin object
```

Five earlier jobs failed on absent packages and are named in the build section
by id.

`HSA_OVERRIDE_GFX_VERSION` is **not set in any job**. That knob makes the
runtime report a different device, which is the one thing an oracle measurement
cannot survive. All four jobs printed `HSA_OVERRIDE_GFX_VERSION=UNSET`
(`phase1b.sh:30`, `phase2.sh:31`, `phase3.sh:40`, `phase4.sh:38`), and both
generating phases assert it again in-process, before they load anything, with a
bare `assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") is None`
(`gen_rocm.py:61`), which is the strongest of the three forms because it stops
the run rather than printing a line somebody has to read.

Phases 1b and 3 additionally sweep the whole variable family and printed
`inherited_env NONE` (`phase1b.sh:31`, `phase3.sh:41`). Phases 2 and 4 print the
single variable only, so that family sweep covers two of the four jobs and not
all four. The knob that decides the answer is covered everywhere; the wider
family is not, and this file says which is which.

## The device, as three independent things report it

```text
kfd node 1        gfx_target_version = 110501
rocminfo agent    Name: gfx1151   Marketing Name: AMD Radeon Graphics   Compute Unit: 40
rocminfo ISA      Name: amdgcn-amd-amdhsa--gfx1151
                  Name: amdgcn-amd-amdhsa--gfx11-generic
torch             name='AMD Radeon 8060S'  gcnArchName='gfx1151'
                  total_memory=68719476736   (64 GiB)
ROCm userspace    /opt/rocm-7.2.4, hip 7.2.53211-97f5574fe27.2.4
```

## THE PREREQUISITE: does PyTorch carry gfx1151 device code? YES

This is measured first and alone, because the previous attempt at this class of
question died here on `thor:gpu0` (`sm_110`) and was misdiagnosed three times
before anyone read the object.

vLLM's own build expects `torch 2.13.0` on ROCm
(`CMakeLists.txt:72`, `TORCH_SUPPORTED_VERSION_ROCM`), and that exact version
exists on the ROCm 7.2 index, matching the board's ROCm 7.2.4:

```text
INDEX rocm7.2 http=200 cp312_manylinux_2_28_x86_64: 2.11.0+rocm7.2 2.12.0+rocm7.2 2.12.1+rocm7.2 2.13.0+rocm7.2 2.14.0+rocm7.2
INDEX rocm7.1 http=200 cp312_manylinux_2_28_x86_64: 2.10.0+rocm7.1 2.11.0+rocm7.1 2.12.0+rocm7.1 2.12.1+rocm7.1 2.13.0+rocm7.1
INDEX rocm7.0 http=200 cp312_manylinux_2_28_x86_64: 2.10.0+rocm7.0
INDEX rocm6.4 http=200 cp312_manylinux_2_28_x86_64: 2.8.0+rocm6.4 2.9.0+rocm6.4 2.9.1+rocm6.4
INSTRUMENT_CHECK positive_control=FOUND torch-2.13.0+rocm7.2
INSTRUMENT_CHECK negative_control=absent (the grep discriminates)
```

The first run of that probe read `no ROCm torch index carries a wheel for this
interpreter` off a **working** index, because it grepped for `linux_x86_64` where
the index writes `manylinux_2_28_x86_64`. The controls above exist because of
that, and they are why the second reading is trustworthy where the first was not.

### 1. The object, read with `llvm-objdump --offloading`

`torch-2.13.0+rocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl` installed into a
venv, `libtorch_hip.so` sha256
`1c4555561a3771a068179f7460a77a29f1f72af00828515f5f6103888ce4da75`,
659,768,953 bytes. The distinct offload targets it carries, **verbatim**:

```text
hipv4-amdgcn-amd-amdhsa--gfx900
hipv4-amdgcn-amd-amdhsa--gfx906
hipv4-amdgcn-amd-amdhsa--gfx908
hipv4-amdgcn-amd-amdhsa--gfx90a
hipv4-amdgcn-amd-amdhsa--gfx942
hipv4-amdgcn-amd-amdhsa--gfx950
hipv4-amdgcn-amd-amdhsa--gfx1030
hipv4-amdgcn-amd-amdhsa--gfx1100
hipv4-amdgcn-amd-amdhsa--gfx1101
hipv4-amdgcn-amd-amdhsa--gfx1102
hipv4-amdgcn-amd-amdhsa--gfx1103
hipv4-amdgcn-amd-amdhsa--gfx1150
hipv4-amdgcn-amd-amdhsa--gfx1151
hipv4-amdgcn-amd-amdhsa--gfx1200
hipv4-amdgcn-amd-amdhsa--gfx1201

OFFLOAD_BUNDLES_TOTAL=2447
OFFLOAD_GFX1151_BUNDLES=153
```

`llvm-objdump` segfaulted (rc 139) after emitting 2464 lines, so this listing is
not certified complete by the tool itself. It is certified by agreeing exactly,
15 for 15, with a second instrument that shares none of its machinery.

### 2. What the build says about itself, by execution

```text
torch.__version__          = 2.13.0+rocm7.2
torch.version.hip          = 7.2.53211
torch.version.cuda         = None
torch.cuda.get_arch_list() = ['gfx900', 'gfx906', 'gfx908', 'gfx90a', 'gfx942',
                              'gfx1030', 'gfx1100', 'gfx1101', 'gfx1102',
                              'gfx1103', 'gfx1200', 'gfx1201', 'gfx950',
                              'gfx1150', 'gfx1151']
torch.cuda.is_available()  = True
device_count               = 1
device[0] name='AMD Radeon 8060S' gcnArchName='gfx1151' total_memory=68719476736
```

### 3. It computes. Real kernels, on the real device

```text
SMOKE_MATMUL    torch.float32 1024^3 max_rel_err = 1.303e-06
SMOKE_MATMUL    torch.float16 1024^3 max_rel_err = 3.727e-04
SMOKE_MATMUL   torch.bfloat16 1024^3 max_rel_err = 2.984e-03
SMOKE_SOFTMAX sum = 1.0
SMOKE_SDPA       = 0.7279645204544067
MEM free=68270686208 total=68719476736
SMOKE=PASS
```

### 4. Triton compiles and runs a kernel on gfx1151

This matters more than a matmul does, because vLLM's RDNA paths **are** Triton
kernels: `triton_w4a16.py:215` is a block-size table introduced by
`# Tuned for RDNA 3.5 (gfx1151, 40 CUs, 32-wide wavefronts)`, and
`chunk_scaled_dot_kkt.py:28` sets `_CAST_DOT_TO_K_DTYPE = on_gfx1x()` on the
linear-attention path this GDN-hybrid model uses.

```text
triton.__version__ = 3.7.1
TRITON_JIT_ON_GFX1151 = PASS
```

The first attempt at this probe printed
`ValueError: @jit functions should be defined in a Python file`. That was the
probe's own body arriving on stdin, not the device. The second printed
`fatal error: Python.h: No such file or directory` from inside Triton's AMD
driver, which JIT-compiles `hip_utils.c` at import. Both are the harness. The
`PASS` above is from a probe written to a real file on a host with
`python3-dev`.

**Verdict on the prerequisite: `TORCH_GFX1151=YES`, measured four ways, with no
`HSA_OVERRIDE_GFX_VERSION`.** The `thor:gpu0` failure mode does not reproduce
here: this is not an architecture PyTorch omits.

## The pin's own gfx1151 surface, re-read on the worker

Every anchor below was read out of the **committed** object
`5559679229bc961848b121ccdeaa8fa5d79bec98` (tree
`d18e26f18474c5407c698eb8a829732c13a37f9d`), staged on the worker from
`vllm-pin.tar.gz` sha256
`7d8bd182057aa17d227caacb7c343258a39af1b8e320fb2bd484c0358b3a98e5`, 6110 files,
`src_manifest_LC_ALL_C=b037645415bb07eccbf9b69f6a8d69b51b14131bac3f1e93a143961af0f46483`.

```text
CMakeLists.txt:52   set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201")
CMakeLists.txt:72   set(TORCH_SUPPORTED_VERSION_ROCM "2.13.0")
rocm.py:77              "0x1586": "AMD_Radeon_8060S",  # gfx1151, Strix Halo
rocm.py:214         _ON_GFX1151 = "gfx1151" in _GCN_ARCH
registry.py:572         "Qwen3_5ForConditionalGeneration": ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
triton_w4a16.py:215             # Tuned for RDNA 3.5 (gfx1151, 40 CUs, 32-wide wavefronts).
chunk_scaled_dot_kkt.py:28      _CAST_DOT_TO_K_DTYPE = on_gfx1x()
docker/Dockerfile.rocm_base:30  ARG PYTORCH_ROCM_ARCH=gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1200;gfx1201;gfx1150;gfx1151
```

The checkpoint's own `config.json` declares
`"architectures": ["Qwen3_5ForConditionalGeneration"]`, `"model_type":
"qwen3_5"`, so `registry.py:572` is the row that matters and not a near neighbour.

## The scoring instrument, controlled before it was used

[`compare.py`](oracle-vllm-gfx1151-20260903/compare.py) scores three token
streams against each other. It was run on the **recorded** streams first, with
vllm.cpp's own ids fed into the vLLM slot, and it reproduces the published
[token-gate v2](qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md) table
exactly — same three prompts, same three indices, same token pairs — while the
identity leg reads 0 of 6:

```text
| 1 `The three primary colors are`        | DIVERGE | 45 | 303 | 1521 |
| 3 `The Pythagorean theorem states that` | DIVERGE | 45 |  25 |  393 |
| 5 `A prime number is a natural number`  | DIVERGE | 32 |  16 |   15 |
VLLMCPP_vs_LLAMACPP_DIVERGENCES=3/6      <- reproduces the published table
VLLMCPP_vs_VLLM_DIVERGENCES=0/6          <- the identity control
```

The prompts are fixed by their hash rather than by retyping: the six lines,
newline-terminated, hash to
`c8a5080046c3206a1186b42a320a21e887691eff43d2116364192f58e5776c7e`, which is the
`prompts_sha256` the v2 evidence records. The engine is fed the **oracle's own
`PROMPT_IDS`** rather than a re-tokenization, so a tokenizer difference cannot
contaminate a generation comparison; the tokenizer is separately re-derived and
the two are printed side by side.

## The build: `BUILD_RC=0`

```text
VLLM_TARGET_DEVICE=rocm  PYTORCH_ROCM_ARCH=gfx1151  MAX_JOBS=4
SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0.dev0+g5559679229
pip install --no-build-isolation -e .
BUILD_RC=0
vllm.__version__ = 0.26.0.dev0+g5559679229
```

`setup.py:1026` joins with `.` when the version already carries a `+`, so a
build against another ROCm would have read `…+g5559679229.rocmXXX`; the
suffix is absent because `get_rocm_version()` equals `VLLM_MAIN_CUDA_VERSION`
here. The version string carries the pin, which is the point.

The gfx targets the build actually asked the compiler for, counted out of the
build log: **119 occurrences of `gfx1151`** and exactly one each of the other
fifteen names in `HIP_SUPPORTED_ARCHS`, which is that CMake string being echoed
once. `csrc/rocm/skinny_gemms_int4.hip.o` and its neighbours are RDNA sources
and they compiled.

The compiled extensions, imported from outside the source tree:

```text
EXT vllm._C                        LOADED
EXT vllm._rocm_C                   LOADED
EXT vllm._C_stable_libtorch        LOADED
EXT vllm._moe_C_stable_libtorch    LOADED
```

`vllm._moe_C` does not exist at this pin and its absence is not a failure.

They compute, on the device:

```text
CUSTOM_OP rms_norm     max_abs_err = 0.014302253723144531
CUSTOM_OP silu_and_mul max_abs_err = 0.016417503356933594
CUSTOM_OP silu_and_mul rel_err     = 0.0029627832118421793
CUSTOMS_OPS=PASS
```

The absolute errors are bf16-sized (bf16 has an 8-bit significand), which is why
the relative error is printed beside one of them.

### vLLM resolves the platform, and it resolves it as this device

```text
AMDSMI_HANDLES   = 1
CURRENT_PLATFORM = RocmPlatform
DEVICE_NAME      = AMD_Radeon_8060S
DEVICE_CAP       = DeviceCapability(major=11, minor=5)
ON_GFX1151 = True   ON_GFX1X = True
```

`AMD_Radeon_8060S` is `rocm.py:77`'s table entry resolving against the live
device id, and `DeviceCapability(11, 5)` is gfx1151. `on_gfx1151()` returns
`True`, so every RDNA-3.5 branch in the pin is live on this board rather than
merely present in its source.

### Five absent packages, none of them about gfx1151

The build failed three times before it succeeded, and every failure was a
missing Ubuntu or Python package in the worker image. Recorded because each one,
taken at face value, would have been written down as "vLLM does not build on
gfx1151".

| attempt | what it printed | what was missing |
|---|---|---|
| 1 `153a0563` | `CMake Error at cmake/utils.cmake:10: Unable to find python matching …` | `python3-dev` |
| 2 `e97bd631` | `LoadHIP.cmake:79 … Could not find a package configuration file provided by "rocrand"` | `rocm-libs` (a runtime ROCm, not a development one) |
| 3 `6a0cb05f` | `Imported target "torch" includes non-existent path /usr/include/libdrm` | `libdrm-dev` |
| run 1 `d77ac9ac` | `RuntimeError: operator torchvision::nms does not exist` | the ROCm `torchvision`; pip had resolved the default PyPI one |
| run 1 `d77ac9ac` | `current_platform = UnspecifiedPlatform`, then `NotImplementedError` | `amdsmi` |

The `amdsmi` one is the sharpest and it is worth stating plainly:
`vllm/platforms/__init__.py:110-128` decides "am I ROCm?" by importing `amdsmi`
and counting processor handles. **With `amdsmi` absent, vLLM answers no and
falls back to `UnspecifiedPlatform` on a box whose ROCm stack is otherwise
entirely working.** A generation attempted in that state fails in a way that
reads exactly like "vLLM cannot run here", and it would be a lie. Phase 3
therefore asserts `RocmPlatform` before it loads anything.

## The run: it generates. Four legs, four clean exits, 288 tokens each

`GEN_RC=0` on all four legs, six prompts of 48 tokens each, `ignore_eos`,
greedy (`temperature=0.0`), batch 1, MTP off, `HSA_OVERRIDE_GFX_VERSION` unset.

```text
GEN_RC[gguf-eager]      = 0
GEN_RC[gguf-eager-2]    = 0
GEN_RC[gguf-compiled]   = 0
GEN_RC[gguf-compiled-2] = 0
```

vLLM's own log, not our inference:

```text
INFO [model.py:627] Resolved architecture: Qwen3_5ForConditionalGeneration
INFO [core.py:121] Initializing a V1 LLM engine (v0.26.0.dev0+g5559679229) …
                   load_format=gguf, quantization=gguf, dtype=torch.bfloat16,
                   served_model_name=/workspace/ckpt/qwen38-27b-q4km/Qwen3.8-27B-Q4_K_M.gguf
INFO [gpu_model_runner.py:5397] Model loading took 16.08 GiB memory
INFO [gpu_worker.py:780] Free memory on device (63.85/64.0 GiB) on startup
INFO [qwen_triton_warmup.py:374] Warming up Qwen Triton kernels for model_type=qwen3_5_text
```

**16.08 GiB of weights is the Q4_K_M arm.** The bf16 checkpoint is 53.8 GB, so
that figure alone rules out a silent fallback to the safetensors sitting beside
it on the same share. `served_model_name` names the GGUF and `load_format` is
`gguf`.

`Resolved architecture: Qwen3_5ForConditionalGeneration` is the registry
resolution, by execution. Two attempts to ask `ModelRegistry` the same question
directly failed on my own wrong call signatures (`is_text_generation_model` and
then `resolve_model_cls` each want a `model_config` at this pin); the engine's
own line is the better evidence and it is what is quoted.

The first prompt's output, verbatim:

```text
GEN_TEXT[0] = ' Paris.\nThe capital city of Germany is Berlin.\nThe capital city
               of Italy is Rome.\nThe capital city of Spain is Madrid.\nThe
               capital city of Portugal is Lisbon.\nThe capital city of Greece
               is Athens.\n'
```

### The RDNA path is the one that executed

vLLM JIT-compiled and ran these Triton kernels during inference, named in its own
`jit_monitor` lines:

```text
_fwd_kernel
kernel_paged_attention_2d
_causal_conv1d_update_kernel
fused_recurrent_gated_delta_rule_packed_decode_kernel
WARNING [chunked_prefill_paged_decode.py:419] Cannot use ROCm custom paged
        attention kernel, falling back to Triton implementation.
```

The last two are the gated-delta-net linear-attention path — the half of this
hybrid architecture that the `_CAST_DOT_TO_K_DTYPE = on_gfx1x()` branch exists
for. This is not a model that ran on a generic fallback; it ran on the RDNA
kernels.

The GGUF plugin's own extension, interrogated as the **installed** object rather
than as the wheel, carries exactly one device target:

```text
INSTALLED_SO = …/site-packages/vllm_gguf_plugin/_C_gguf.abi3.so
sha256       = cc9c2440bf488ca22bd79ae4adf9e5f1931de52e55a221e3c45c214730b14eb3
Extracting offload bundle: …_C_gguf.abi3.so.0.host-x86_64-unknown-linux-gnu-
Extracting offload bundle: …_C_gguf.abi3.so.0.hipv4-amdgcn-amd-amdhsa--gfx1151
PLUGIN_GFX1151_BUNDLES=1
INSTALLED_PLUGIN_ARCH_OK
```

Control: the same instrument on `libc10.so`, which must carry no device code,
lists no bundles. Plugin source `d4c1f0d082fc7cd4350da56689109a01c1f29d6c` via
`ggufplugin-src.tar.gz` sha256
`9e15c20e0b75f75bbf886966df07843c4b70a7952fad4b80e8e8183e2f70743b`; the wheel
built here is
`432c21abe80bec2a8156e1c4883d9ac848dfdd8c1bf00ca6acc75b9ac230c615`.

### Both configurations are self-reproducible, and they disagree with each other

```text
EAGER1_EQ_EAGER2       = True      (tokens-gguf-eager.json      sha256 350b5fae…)
COMPILED1_EQ_COMPILED2 = True      (tokens-gguf-compiled.json   sha256 034a1e30…)
EAGER1_EQ_COMPILED1    = False
   prompt 3: first diff at 45   eager=25     compiled=393
   prompt 4: first diff at 13   eager=19820  compiled=6165
```

The two eager legs are byte-identical **files**, and so are the two compiled
legs, so the eager-versus-compiled difference is reproducible rather than noise.
`enforce_eager=True` is a correctness configuration and is not admissible as a
performance denominator; no performance figure is taken from either.

## THE RESULT: the primary oracle does not pass this gate either

The declared gate is 6-of-6 token exactness against llama.cpp `b10451`, on this
artifact, at these six prompts. Scored with
[`compare.py`](oracle-vllm-gfx1151-20260903/compare.py), which reproduces the
published v2 table exactly on the recorded streams.

### vLLM `5559679229` on gfx1151, `enforce_eager=True`, vs llama.cpp `b10451`

| prompt | verdict | first diff | vLLM | llama.cpp |
|---|---|---:|---:|---:|
| 0 `The capital city of France is` | **TOKEN-EXACT 48/48** | — | — | — |
| 1 `The three primary colors are` | DIVERGE | 35 | 4350 | 5844 |
| 2 `Water boils at a temperature of` | DIVERGE | 4 | 11995 | 29922 |
| 3 `The Pythagorean theorem states that` | DIVERGE | 45 | 25 | 393 |
| 4 `In 1969, humans first walked on` | DIVERGE | 14 | 4593 | 22486 |
| 5 `A prime number is a natural number` | **TOKEN-EXACT 48/48** | — | — | — |

`VLLM_vs_LLAMACPP_DIVERGENCES = 4/6`

### The same vLLM, `torch.compile` on, vs the same llama.cpp

| prompt | verdict | first diff | vLLM | llama.cpp |
|---|---|---:|---:|---:|
| 0 `The capital city of France is` | **TOKEN-EXACT 48/48** | — | — | — |
| 1 `The three primary colors are` | DIVERGE | 35 | 4350 | 5844 |
| 2 `Water boils at a temperature of` | DIVERGE | 4 | 11995 | 29922 |
| 3 `The Pythagorean theorem states that` | **TOKEN-EXACT 48/48** | — | — | — |
| 4 `In 1969, humans first walked on` | DIVERGE | 13 | 6165 | 19820 |
| 5 `A prime number is a natural number` | **TOKEN-EXACT 48/48** | — | — | — |

`VLLM_vs_LLAMACPP_DIVERGENCES = 3/6`

### Side by side with the arm the gate convicts

```text
vLLM 5559679229, compiled, on gfx1151   vs llama.cpp b10451   3 of 6 divergent
vLLM 5559679229, eager,    on gfx1151   vs llama.cpp b10451   4 of 6 divergent
vllm.cpp ROCm arm,         on gfx1151   vs llama.cpp b10451   3 of 6 divergent
vllm.cpp ROCm arm          vs vLLM compiled                   5 of 6 divergent
```

**The reference implementation, on the same board, on the same 17 GB artifact,
at the same six prompts, diverges from the secondary oracle at the same rate the
arm under gate does.** Its two own configurations differ from each other on two
of the six.

The clearest single row is prompt 3. The gate convicts our ROCm arm there for
emitting `25` at step 45 where llama.cpp emits `393`. **vLLM in eager mode emits
`25`.** vLLM with `torch.compile` on emits `393`. So the token our arm is
convicted for is the token the primary oracle produces under one of its own two
supported configurations, and the difference between those two configurations is
a compilation flag.

Prompt 4 is the same shape from the other direction: our CPU tier is convicted
at step 14 for `4593`, and eager vLLM emits `4593` there too.

## What this does and does not establish

**Established, by execution:**

- Torch for ROCm 7.2 carries gfx1151 device code, and it computes.
- Triton compiles and runs kernels on gfx1151.
- The pinned vLLM builds for gfx1151, its extensions load and compute, and it
  resolves `RocmPlatform` / `AMD_Radeon_8060S` / `on_gfx1151() == True`.
- The pinned vLLM plus `vllm-gguf-plugin` loads the gated Q4_K_M artifact and
  generates 48 greedy tokens for all six gate prompts, reproducibly, twice per
  configuration.
- On that workload vLLM is not token-exact against llama.cpp `b10451` either.

**NOT established, and not claimed:**

- **That `AGENTS.md`'s gateability bar is met for a token gate.** The bar is that
  the oracle demonstrably builds and runs the model; it does. Whether a *gate*
  should be re-declared against it is a decision for #2497 and #2546, and this
  document does not make it.
- That vllm.cpp is correct, or more correct than anything. Two engines
  disagreeing with a third tells you the third is not a ground truth; it does not
  tell you either of the two is right.
- Any performance, latency or memory comparison. None was taken. The
  `gpu_memory_utilization`, `max_model_len` and `enforce_eager` values here were
  chosen for a correctness run and are not vLLM's production configuration.
- That the arithmetic is matched. `vllm-gguf-plugin` and llama.cpp dequantize
  Q4_K differently; this comparison is exactly the one the current gate makes,
  and its limits are the current gate's limits.
- Anything about `dgx:gpu0`. #2624 is this question on CUDA and it is a
  different device with a different failure.

## Consequence for #2497 and #2546, stated without acting on it

The `TOKEN_GATE=FAIL` in
[token-gate v2](qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md) stands as
measured; nothing here rescores it. What changes is the standing of its
denominator. That evidence already recorded that three of the six contested steps
were llama.cpp disagreeing with **itself** across its own kernel paths. This
document adds that the primary oracle disagrees with llama.cpp at the same rate
as the arm being convicted, and disagrees with itself across a compilation flag.

A gate whose denominator the reference implementation does not satisfy is
measuring the denominator. Re-declaring it belongs to #2546 with a fresh spec, on
the strength of a denominator that can now be run rather than only cited.

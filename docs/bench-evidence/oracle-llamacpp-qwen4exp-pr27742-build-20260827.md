# llama.cpp PR #27742 at `035e2273` BUILDS, and does not yet run, 27 August 2026

Gateability evidence for the `llama-cpp-qwen4exp` secondary oracle
([`../../.agents/oracles/llama-cpp-qwen4exp.md`](../../.agents/oracles/llama-cpp-qwen4exp.md)),
recorded for [#2060](https://github.com/mudler/vllm.cpp/issues/2060). `AGENTS.md`
section "Measure gateability" admits `gateable = yes` only after an oracle
"demonstrably builds and runs the model". This file records **one half of that
demonstration and says so**, which is why the pin still reads `gateable = no`.

**This is not a floor, not a comparison, and carries no timing.** No vllm.cpp arm
ran, no `qwen4exp` weights were loaded, and nothing here may be quoted as a
performance number. [#1003](https://github.com/mudler/vllm.cpp/issues/1003) owes
every llama.cpp floor in this tree, and this run adds nothing to that debt either
way.

## What ran

CPU only, on the local x86_64 development host, with no GPU lease taken and no
fleet device touched. The build uses no CUDA toolkit and no device, so a lease
would have held a GPU for a compiler.

| Fact | Value |
|---|---|
| host | Ubuntu 24.04.3 LTS, Linux 6.8.0-136-generic x86_64 |
| CPU | AMD Ryzen 9 9950X3D 16-Core Processor, `nproc` reports 20 |
| build parallelism | `-j 4`, deliberately modest, because an unconstrained parallel build of this tree has filled this disk before |
| memory | 84 GiB total |
| compiler | GNU 13.3.0 |
| CMake | 3.28.3 |
| date | 27 August 2026 |

## Identity, asserted on the tree and not only the commit

[`../../.agents/oracles/llama-cpp.md`](../../.agents/oracles/llama-cpp.md) requires
that a number recorded against a llama.cpp oracle comes from a fresh `git archive`
or a fresh clone of the pinned SHA, never from a directory somebody develops in.
The developer's local llama.cpp checkout is dirty and was not read at any point.

The source came from a fresh bare repository in a scratch directory, created empty,
whose only remote is `https://github.com/ggml-org/llama.cpp`, fetched by SHA:

```sh
git init --bare .
git remote add origin https://github.com/ggml-org/llama.cpp
git fetch --depth 1 origin 035e22731a7fd70b9854b3a2d64ec68e9b1a45d3
git archive --format=tar 035e22731a7fd70b9854b3a2d64ec68e9b1a45d3 | tar -x -C src
```

The extracted tree was then checked back against the object store, so the archive
is proven to be the pinned revision rather than assumed to be:

| Path | Blob at the pin | sha256 of the extracted file | sha256 of `git cat-file blob` |
|---|---|---|---|
| `src/models/qwen4exp.cpp` | `65e9b19910739dbe6ce49198bc206b9551cfa306` | `1afe4a3c677452680bf8ccbbb98573ee6a645d0729df74a15b0faad38bdf0bc7` | identical |

`conversion/qwen4exp.py` is present at the pin, 11037 bytes, and
`git grep -il qwen4exp 035e2273...` names 14 files.

## Build recipe

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF
cmake --build build --target llama-completion -j 4
```

`llama-completion` is the named target, and named targets are the whole point. A
bare parallel build of this tree links every test binary and has filled this disk
before.

At this revision `tools/cli` is gated behind `LLAMA_BUILD_SERVER`, so `llama-cli`
is not a target when the server is off. `llama-completion` is the model-loading
generation tool that is, which is why the recipe names it.

## Result: the build SUCCEEDS

```
[100%] Linking CXX executable ../../bin/llama-completion
[100%] Built target llama-completion
BUILD_RC=0
```

247 translation units, zero errors, one warning. Binaries:

| Artifact | sha256 |
|---|---|
| `bin/llama-completion` | `63bd7f9e68cc95d9b01258be5c547bc3c22e785acc55065d42f08dbc954acc73` |
| `bin/libllama.so.0.3.0` | `2065004fc2a280b4d203bac03b99c62f4ef24d41051382e1e79246cd0dee400a` |
| `bin/libggml-cpu.so.0.22.0` | `e3f5e0d7bd5bc2678bbd3ff495c2a28764ea4db46bbe31072a3f0c88015e7cca` |

`GGML_NATIVE=ON`, so a recorded binary sha256 identifies a build on this named host
and never a tree.

The binary executes: `llama-completion --version` returns rc=0 and reports
`0.3.0-dev (build 0, commit unknown)`, built with GNU 13.3.0 for Linux x86_64. The
build number is 0 and the commit `unknown` because a `git archive` carries no
repository, which is the intended trade for the identity assertion above.

**The architecture is compiled in, not merely present in the source tree.**
`strings bin/libllama.so` matches `qwen4exp` 228 times. That is the difference
between a file existing and a linker keeping it.

## The one warning is the gap between the pin and the head, made concrete

```
src/models/qwen4exp.cpp:1027:16: warning: unused variable 'mem_size' [-Wunused-variable]
```

That is the exact defect the later commit `6a69a0c qwen4exp: drop an unused
variable that breaks -Werror builds` removes. It is a warning here and not an
error only because this recipe does not set `-Werror`.

Read that as the practical content of the "Why the pin stays at `035e2273`"
section in
[`../../.agents/oracles/llama-cpp-qwen4exp.md`](../../.agents/oracles/llama-cpp-qwen4exp.md).
The pinned revision builds under the recipe above, and a stricter recipe at the
same pin does not. Anyone advancing the pin to the live head `6c5afc86` gains
this fix along with twelve other commits. It is the ONLY `-Werror` repair among
the thirteen. `24ea62d` says "and the fatal-warning build" in its subject, but
its diff touches one file, `src/llama-memory-hybrid-idx.cpp`, and by then this
`mem_size` line had been gone for three commits. That advance breaks no anchor
in this tree, and the record's table holds every check behind that statement. What it costs is THIS file, because the
build measured here is the build of `035e2273`. Advance the pin in a change that
measures the build again at the new object.

## Result: the RUN half is NOT measured

`gateable` stays `no`. The blocker is artifact availability and not the oracle.

No complete `qwen4exp` GGUF existed on this host. The seven
`unsloth/Qwen3.8-Flash-Next-GGUF` quants were mid-download to
`/mnt/nas_share/checkpoints/qwen3.8-flash-next-*`. Only the first directory
existed, `UD-IQ1_S`, holding 3.3 GiB of 67.56 GiB, with a single 10946624-byte
fragment of shard 1 of 3 visible and no `SHA256SUMS.txt`.

The load was attempted anyway rather than assumed, and the refusal names the
missing shard:

```
E gguf_init_from_file: failed to open GGUF file
  '.../Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf' (No such file or directory)
E llama_model_load: error loading model: llama_model_loader: failed to load GGUF
  split from .../Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf
E llama_model_load_from_file_impl: failed to load model
```

That is a missing file, not a rejected architecture, so it says nothing about
whether #27742 can load `qwen4exp` and it must not be cited as if it did.

## What is still owed

[#2060](https://github.com/mudler/vllm.cpp/issues/2060) owes a load and a
generation on a complete `unsloth/Qwen3.8-Flash-Next-GGUF` artifact, with the
checkpoint file name, size, and sha256 recorded, before `gateable` may read `yes`.
A build proves the architecture is declared and links. It does not prove the graph
it builds produces coherent text on real weights, and only the second thing is what
the gateability rule asks for.

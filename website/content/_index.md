---
title: Documentation
---

vllm.cpp is a from-scratch C++20 inference engine. There is no Python and no
PyTorch at inference time: one binary behind a flat C ABI, reading Safetensors
and GGUF, running on CUDA, CPU, Metal and Vulkan from a single source tree.

## Why it exists

Three inference projects each got one thing right, and no engine has all three.

**vLLM** got the serving architecture. Continuous batching and block-paged KV
are the reason it is fast under concurrency.

**llama.cpp** got portability. One binary, no runtime, and it runs where you
actually are rather than where the deployment guide assumes you are.

**SGLang** got the ideas. Radix attention and prefix-aware cache scheduling are
the kind of thing that should be in every engine by now.

So you pick throughput and inherit a nine-gigabyte virtualenv, or you pick
portability and give up the serving architecture, or you pick the clever
scheduling and inherit the virtualenv again. vllm.cpp is an attempt at having
all three in one binary, with no Python and no PyTorch in the process.

## What that commits us to

**Exact tokens, not close ones.** Every architecture is gated on producing the
same token ids as a pinned vLLM oracle on the same workload, with no tolerance
at the top. A port that is nearly right costs you months of chasing quality
problems that are actually a bug. Where the reference genuinely disagrees with
itself, because bf16 greedy does, those positions are found and marked rather
than papered over. Speed is quoted against a reference measured in its own
production configuration.

**Your hardware, not a supported-vendor list.** We are not in a position to drop
a card because it stopped being strategic, so we would rather add the ones other
engines have left behind. Hardware reports and build failures from boards nobody
here owns are among the most useful contributions we get.

**One embedding surface.** The C ABI in `include/vllm.h` is versioned, grows by
appending, and only bumps on an incompatible change. Internals and CLI flags
move fast. That header is the part we keep still.

It is not finished, and the stable overview is in [Project status](../README.md#project-status) rather than
implied away. It is also not a competitor to vLLM in any sense that matters:
this project is measured against vLLM on every commit, it exists because their
design is good, and it is an independent community port that is not affiliated
with or endorsed by them.

The full argument is in
[MANIFESTO.md](https://github.com/mudler/vllm.cpp/blob/main/MANIFESTO.md).
Hardware we do not have, ports of architectures we have not reached, and honest
bug reports help most.

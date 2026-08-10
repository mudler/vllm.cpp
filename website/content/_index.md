---
title: Documentation
---

**A from-scratch C++20 inference engine.** No Python and no PyTorch at inference
time — one binary behind a flat C ABI, reading Safetensors and GGUF, running on
CUDA, CPU, Metal and Vulkan from a single source tree.

## Why it exists

Three inference projects each got one thing right, and no engine has all three.
**vLLM** got the serving architecture: continuous batching and block-paged KV
are why it is fast under concurrency. **llama.cpp** got portability — one
binary, no runtime, running where you actually are rather than where the
deployment guide assumes you are. **SGLang** got the ideas, like radix attention
and prefix-aware cache scheduling.

So you pick throughput and inherit a nine-gigabyte virtualenv, or you pick
portability and give up the serving architecture, or you pick the clever
scheduling and inherit the virtualenv again. vllm.cpp is an attempt at having
all three in one binary, with no Python and no PyTorch in the process.

## What that commits us to

**Exact tokens, not close ones.** Every architecture is gated on producing the
same token ids as a pinned vLLM oracle on the same workload, with no tolerance
at the top — so "grounded in vLLM" is a test result rather than a design claim.
Where the reference genuinely disagrees with itself, because bf16 greedy does,
those positions are found and marked rather than papered over. Speed is only
ever quoted against a reference measured in its own production configuration.

**Your hardware, not a supported-vendor list.** We would rather add the cards
other engines have left behind. Hardware reports and build failures from boards
nobody here owns are among the most useful contributions we get.

**One embedding surface.** The C ABI in `include/vllm.h` is versioned, grows by
appending, and only bumps on an incompatible change. Internals and CLI flags
move fast; that header is the part we keep still.

It is not finished, and the gaps are listed honestly in [Status](STATUS.md)
rather than implied away — nor is it a competitor to vLLM in any sense that
matters. It is an independent community port, measured against vLLM on every
commit, that exists because their design is good.

The full argument is in
[MANIFESTO.md](https://github.com/mudler/vllm.cpp/blob/main/MANIFESTO.md).

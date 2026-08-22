# Why vllm.cpp exists

Three inference projects each got one thing right, and no engine has all three.

**vLLM** got the serving architecture. Continuous batching and block-paged KV
are the reason it is fast under concurrency, and everything since has been built
on that.

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
than papered over. Op level has normal tolerances. The output does not.

**Your hardware, not a supported-vendor list.** We are not in a position to drop
a card because it stopped being strategic, so we would rather add the ones other
engines have left behind. Hardware reports and build failures from boards nobody
here owns are among the most useful contributions we get.

**One embedding surface.** The C ABI in `include/vllm.h` is versioned, grows by
appending, and only bumps on an incompatible change. Internals and CLI flags move
fast. That header is the part we keep still.

**Ideas from wherever they are.** Being an unaffiliated port means we can take
the serving architecture from one project and the scheduling from another in the
same week, which nobody who has to pick a side can do.

## What this is not

It is not finished, and the gaps are described honestly in the
[project status](README.md#project-status) rather than implied away. It is not a research
framework: if you want to patch an engine at runtime to try a new model tomorrow,
Python is better at that and it is not close. The argument here is about what you
deploy, not what you experiment with.

It is also not a competitor to vLLM in any sense that matters. This project is
measured against vLLM on every commit, it exists because their design is good,
and it is an independent community port that is not affiliated with or endorsed
by them.

## What helps most

Hardware we do not have, ports of architectures we have not reached, and honest
bug reports. Come and say hi in the issues.

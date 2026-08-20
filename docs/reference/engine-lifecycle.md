# Engine lifecycle reference

This page describes multi-engine and multi-backend process behavior.

Constructing a `LoadedEngine`, destroying it, and constructing another in the
same process is supported, including on CUDA. Each engine's device-resident MoE
and Marlin constants are owned by the weights they describe and are released
with them.

Before, that state lived in process-lifetime caches keyed on the *address* of a
weights block, so a second engine could land on a freed block's address and
reuse device pointers that had already been freed. Nothing crashed, the CUDA
context is never torn down, so the pointers stayed mapped, it simply produced
corrupted or zeroed output tokens, intermittently
([#237](https://github.com/mudler/vllm.cpp/issues/237)).

More than one **backend** in one process is likewise supported, a CPU forward
running beside a CUDA one, which is what a diffusion pipeline with a host-side
stage does. Until
[#516](https://github.com/mudler/vllm.cpp/issues/516) it was not: the shared
device-scratch pool was a single process-wide free list keyed by byte size class
with no device in the key, so a block allocated through one backend was handed
to the next caller of that size class on another. It has two symptoms and the
direction picks which: a `cudaMalloc` block reaching a CPU forward segfaults in
the host `memcpy`, and a host block reaching a CUDA forward produces output that
is uniformly NaN rather than wrong. Neither can happen now, a scratch pool is
bound to one backend and refuses any other with a `std::logic_error` naming both
,  and no user-facing flag or env var selects the behaviour: it is unconditional.

One consequence is worth knowing before you add a backend. The scratch pool's
residency cap now comes from *that device's* platform rather than from whichever
device resolved first, so constructing a buffer on a backend whose platform was
never registered raises instead of silently inheriting another platform's cap. A
cap read off the wrong platform is a wrong number, not a default, and every
backend the tree ships registers one.

`VT_POOL_BYPASS=1` and `VT_POOL_EXACT=1` keep exactly the meanings
[the environment reference](../ENVIRONMENT.md) records for them. They are debugging lanes, not
timing configurations, and the pool's own test suite is green under both, so
either one stays usable as a discriminator when something else is under
suspicion.

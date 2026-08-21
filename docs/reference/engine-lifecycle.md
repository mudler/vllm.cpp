# Engine lifecycle reference

This page describes multi-engine and multi-backend process behavior.

Constructing a `LoadedEngine`, destroying it, and constructing another in the
same process is supported, including on CUDA. Each engine's device-resident MoE
and Marlin constants are owned by the weights they describe and are released
with them.

One process can also run more than one backend. For example, a CPU forward can
run beside a CUDA forward in a diffusion pipeline. Each scratch pool belongs to
one backend. The pool throws `std::logic_error` if another backend uses it.

The pool gets its residency cap from its own device platform. Buffer creation
fails if the backend has no registered platform. Every shipped backend registers
a platform. No user-facing flag or environment variable changes these rules.

`VT_POOL_BYPASS=1` and `VT_POOL_EXACT=1` keep exactly the meanings
[the environment reference](../ENVIRONMENT.md) records for them. They are debugging lanes, not
timing configurations, and the pool's own test suite is green under both, so
either one stays usable as a discriminator when something else is under
suspicion.

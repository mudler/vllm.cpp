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

A freed block is offered first to a later request in its own size class, and
then, if that class is empty, to a request at least half its size. A block borrowed
this way returns to the class the backend allocated it at, so the larger class
keeps it and can still serve a larger request. Without that second step the pool
retained one block per distinct shape the traffic had ever shown, and a server
that had already served its largest request went on asking the backend for more
memory ([#1922](https://github.com/mudler/vllm.cpp/issues/1922)).

`VT_POOL_BYPASS=1`, `VT_POOL_EXACT=1` and `VT_POOL_BORROW=0` keep exactly the meanings
[the environment reference](../ENVIRONMENT.md) records for them. They are debugging lanes, not
timing configurations, and the pool's own test suite is green under the first
two, so either one stays usable as a discriminator when something else is under
suspicion. `VT_POOL_BORROW=0` is the exception and says so in the environment
reference: it is the red arm of the steady-state gate, so that gate FAILS under
it by construction.

#!/usr/bin/env python3
"""Invoke Triton's AOT compiler with typed numeric target fields.

Triton 3.6's tools/compile.py splits --target but does not convert the CUDA
architecture or warp size to integers before constructing GPUTarget. Keep the
documented CLI and patch only that constructor boundary until upstream fixes it.
"""

import runpy

from triton.backends import compiler


_gputarget = compiler.GPUTarget


def _typed_gputarget(backend: str, arch: str, warp_size: str):
    if backend == "cuda":
        arch = int(arch)
    return _gputarget(backend, arch, int(warp_size))


compiler.GPUTarget = _typed_gputarget
runpy.run_module("triton.tools.compile", run_name="__main__")

#!/usr/bin/env python3
"""Fail if the cuBLASLt matmul kernels break INVOCATION parity with vLLM.

The bug this exists to prevent already shipped (and hid for weeks): Laguna's bf16
M=1 decode GEMVs ran cuBLAS `gemvx<bf16,FLOAT>` (an f32-output template, 204 us
o_proj) where vLLM runs the IDENTICAL kernel as `gemvx<bf16,bf16>` (a bf16-output
template, 139 us). The OUTPUT dtype of the GEMM selects the gemvx template, so an
f32-resident decode stream buys the slower template even though the kernel NAME
matches vLLM's. It hid because parity was "verified" CROSS-TOOL (our nsys vs vLLM's
torch profiler), which cannot compare in-graph template dtypes. The root cause is
CALLER-driven: vt::MatmulBT is dtype-faithful (cuda_matmul.cu `out_type = out.dtype
== kF32 ? CUDA_R_32F : CUDA_R_16BF`); the SLOW template is bought when the C/D layout
dtype is HARDCODED to f32 instead of mirroring the caller's requested output dtype.

Two op-contract invariants over src/vt/cuda/cuda_matmul.cu, both pure functions so
they are unit- and mutation-testable (tests/scripts/test_check_gemv_invocation_
consistency.py), mirroring check-fusion-consistency.py:

A(1) OUTPUT-DTYPE FAITHFULNESS. At every cuBLASLt GEMM site the C/D (output) matrix
     layout — the `lc` guard, whether created via the MakeRowMajor{,Batched} helpers
     or a raw cublasLtMatrixLayoutCreate(&lc...) — must take its dtype from the
     dtype-faithful `out_type` VARIABLE (the ternary on out.dtype), NEVER a hardcoded
     CUDA_R_32F literal. An f32 OUTPUT stays legal when the caller's out.dtype IS f32
     (the ternary yields CUDA_R_32F); the bug is HARDCODING the literal, which forces
     the slow gemvx<bf16,FLOAT> template even when the caller asked for bf16 output.
     The A/B operand layouts (`la`/`lb`, which legitimately use ab_type/a_type) are out
     of scope — the output dtype is what selects the template.

A(2) NAMED ALGO POLICY. Every cuBLASLt heuristic query's requestedAlgoCount must be one
     of the named constants on ALGO_POLICY_NAMES below, never a bare numeric literal and
     never some other identifier, so any algo-policy change (search more than the single
     best algo) is a conscious, greppable decision instead of a `1` scattered across call
     sites. Each name used must also be DECLARED `constexpr int <name> = ...` in the same
     file, so the knob is a real compile-time constant rather than a runtime value the
     gate cannot read. Each cublasLtMatmulAlgoGetHeuristic call must carry the
     /*requestedAlgoCount=*/ marker so the argument stays greppable and this gate cannot
     be bypassed by dropping it.

     THE ALLOWLIST IS THE GATE, and it was widened once, deliberately
     (PERF-FP8-SMALL-M-DISPATCH, #1866). Until then the rule was "not a bare literal",
     which accepted ANY identifier — including a runtime variable holding a swept algo
     count — while the pass message claimed every site routed through
     kGemvHeuristicAlgos. Adding a second policy constant now costs an edit HERE, which
     is what "conscious" was supposed to mean. kFp8AlgoLogCandidates is diagnostic-only:
     it runs on its own heuristic query under VT_GEMM_ALGO_LOG=1 and never touches the
     algo the production query selects, which is why the production sites all still read
     kGemvHeuristicAlgos and invocation parity is unmoved.

These are FLOORS on the ONE shared cuBLASLt matmul translation unit, not per-model
checks: they keep the dtype-faithful contract that makes every bf16 GEMV resolve to
the SAME template vLLM's does. The CALLER-side axis (an f32-resident decode stream
that buys the slow template model-wide) is policed separately by
check-runner-routing-consistency.py invariant (c); this gate owns the op-contract side.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATMUL_CU = ROOT / "src/vt/cuda/cuda_matmul.cu"

# The dtype-faithful output-dtype variable every C/D layout must reference. cuda_matmul.cu
# derives it per function as
#   const cudaDataType_t out_type = out.dtype == DType::kF32 ? CUDA_R_32F : CUDA_R_16BF;
# so the output layout mirrors whatever dtype the CALLER asked for.
OUT_TYPE_VAR = "out_type"

# The named constant the PRODUCTION requestedAlgoCount policy must route through.
ALGO_CONST = "kGemvHeuristicAlgos"

# Every named constant a requestedAlgoCount argument may be. Widening this set is
# the conscious edit invariant A(2) exists to force; see the docstring.
ALGO_POLICY_NAMES = (
    ALGO_CONST,             # production: 1, the single best heuristic, no algo search
    "kFp8AlgoLogCandidates",  # diagnostic only, VT_GEMM_ALGO_LOG=1 (#1866)
)

# --- Invariant A(1): C/D (output) layout dtype -------------------------------------

# cuBLASLt matrix-layout creation via our row-major helpers or the raw API, capturing
# the target layout variable and its dtype argument:
#   MakeRowMajor(lc, out_type, m, n);
#   MakeRowMajorBatched(lc, out_type, m, k, ...);
#   cublasLtMatrixLayoutCreate(&lc.v, out_type, ...);
#   cublasLtMatrixLayoutCreate(&p.lc, out_type, ...);
# A definition head like `MakeRowMajor(LayoutGuard& l, cudaDataType_t t, ...)` never
# matches: the `& ` after the first token means no comma follows it.
_HELPER_LAYOUT = re.compile(
    r"\bMakeRowMajor(?:Batched)?\s*\(\s*(?P<target>[A-Za-z_][\w.]*)\s*,\s*(?P<dtype>[A-Za-z_][\w.]*)"
)
_RAW_LAYOUT = re.compile(
    r"\bcublasLtMatrixLayoutCreate\s*\(\s*&\s*(?P<target>[A-Za-z_][\w.]*)\s*,\s*(?P<dtype>[A-Za-z_][\w.]*)"
)
# A CUDA data-type literal (CUDA_R_32F, CUDA_R_16BF, CUDA_R_8F_E4M3, ...).
_CUDA_TYPE_LITERAL = re.compile(r"^CUDA_[A-Z0-9_]+$")

# --- Invariant A(2): named algo policy ---------------------------------------------

# The requestedAlgoCount argument, keyed off the inline /*requestedAlgoCount=*/ marker
# the call sites carry; the captured token is the argument value.
_REQUESTED_ALGO = re.compile(
    r"/\*\s*requestedAlgoCount\s*=\s*\*/\s*(?P<arg>[A-Za-z_]\w*|\d+)"
)
# A cuBLASLt heuristic query call — the site that must carry the marker above.
_HEURISTIC_CALL = re.compile(r"\bcublasLtMatmulAlgoGetHeuristic\s*\(")
# The constant declaration that makes a policy name a real, greppable knob.
_ALGO_CONST_DECL = re.compile(
    r"\bconstexpr\s+int\s+" + re.escape(ALGO_CONST) + r"\s*="
)


def _decl_re(name: str) -> re.Pattern[str]:
    return re.compile(r"\bconstexpr\s+int\s+" + re.escape(name) + r"\s*=")


def _layout_var(target: str) -> str:
    """Normalize a layout target (`lc`, `lc.v`, `p.lc`) to its variable name."""
    t = target.strip()
    if t.endswith(".v"):
        t = t[:-2]
    return t.split(".")[-1]


def _line_no(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def output_layout_sites(text: str) -> list[tuple[int, str]]:
    """Every C/D (output) matrix-layout create site as (line_no, dtype_token). The C/D
    layout is the one bound to the `lc` guard; A/B operand layouts (`la`/`lb`) are out of
    scope for this invariant (they legitimately use ab_type/a_type)."""
    sites: list[tuple[int, str]] = []
    for rx in (_HELPER_LAYOUT, _RAW_LAYOUT):
        for m in rx.finditer(text):
            if _layout_var(m.group("target")) == "lc":
                sites.append((_line_no(text, m.start()), m.group("dtype")))
    return sorted(sites)


def hardcoded_output_dtype_sites(text: str) -> list[tuple[int, str]]:
    """C/D layout sites whose dtype is a hardcoded CUDA_R_* literal instead of the
    dtype-faithful out_type variable (invariant A(1)). Non-empty == the check fails."""
    return sorted(
        (line_no, dtype)
        for line_no, dtype in output_layout_sites(text)
        if _CUDA_TYPE_LITERAL.match(dtype)
    )


def requested_algo_args(text: str) -> list[tuple[int, str]]:
    """Every /*requestedAlgoCount=*/ argument as (line_no, arg_token)."""
    return sorted(
        (_line_no(text, m.start()), m.group("arg"))
        for m in _REQUESTED_ALGO.finditer(text)
    )


def bare_algo_count_sites(text: str) -> list[tuple[int, str]]:
    """requestedAlgoCount sites whose argument is a bare numeric literal instead of a
    named constant (invariant A(2)). Non-empty == the check fails."""
    return sorted(
        (line_no, arg) for line_no, arg in requested_algo_args(text) if arg.isdigit()
    )


def unmarked_heuristic_calls(text: str) -> int:
    """How many cublasLtMatmulAlgoGetHeuristic calls lack a /*requestedAlgoCount=*/
    marker (each such call could smuggle a bare literal past invariant A(2))."""
    return max(0, len(_HEURISTIC_CALL.findall(text)) - len(requested_algo_args(text)))


def unknown_algo_count_sites(text: str) -> list[tuple[int, str]]:
    """requestedAlgoCount sites whose argument is an identifier that is NOT on
    ALGO_POLICY_NAMES (invariant A(2)). Bare literals are reported separately by
    bare_algo_count_sites; this is the other half, and it is the half the rule
    lacked until #1866 — "not a literal" accepted any name at all, including a
    runtime variable. Non-empty == the check fails."""
    return sorted(
        (line_no, arg)
        for line_no, arg in requested_algo_args(text)
        if not arg.isdigit() and arg not in ALGO_POLICY_NAMES
    )


def undeclared_algo_constants(text: str) -> list[str]:
    """Policy names USED at a requestedAlgoCount site but not declared
    `constexpr int <name> = ...` in the same file. A name that is not a
    compile-time constant is a knob this gate cannot read."""
    used = {arg for _, arg in requested_algo_args(text) if arg in ALGO_POLICY_NAMES}
    return sorted(name for name in used if _decl_re(name).search(text) is None)


def declares_algo_constant(text: str) -> bool:
    """True if ALGO_CONST is declared `constexpr int kGemvHeuristicAlgos = ...`."""
    return _ALGO_CONST_DECL.search(text) is not None


def main() -> int:
    if not MATMUL_CU.exists():
        print(f"ERROR: {MATMUL_CU} not found", file=sys.stderr)
        return 1
    text = MATMUL_CU.read_text(encoding="utf-8", errors="ignore")
    rc = 0

    # Invariant A(1) — C/D layout dtype is the dtype-faithful out_type variable.
    sites = output_layout_sites(text)
    hard = hardcoded_output_dtype_sites(text)
    if hard:
        rc = 1
        print(
            "ERROR: cuBLASLt C/D (output) matrix-layout site(s) in "
            "src/vt/cuda/cuda_matmul.cu HARDCODE a CUDA_R_* dtype literal instead of "
            f"the dtype-faithful {OUT_TYPE_VAR} variable (out.dtype ternary):",
            file=sys.stderr,
        )
        for line_no, dtype in hard:
            print(f"  - cuda_matmul.cu:{line_no}: lc layout dtype is {dtype}", file=sys.stderr)
        print(
            "A hardcoded f32 output layout on a bf16 GEMM forces the slow "
            "gemvx<bf16,FLOAT> template at M=1 (vLLM runs gemvx<bf16,bf16>). Use the "
            f"{OUT_TYPE_VAR} variable so the C/D layout mirrors the caller's requested "
            "output dtype (f32 output stays legal when out.dtype IS f32).",
            file=sys.stderr,
        )
    else:
        print(
            f"OK (out-dtype): {len(sites)} cuBLASLt C/D layout site(s) in cuda_matmul.cu "
            f"all take their dtype from the dtype-faithful {OUT_TYPE_VAR} variable."
        )

    # Invariant A(2) — requestedAlgoCount is a named constant, marker present, declared.
    algo_sites = requested_algo_args(text)
    bare = bare_algo_count_sites(text)
    unknown = unknown_algo_count_sites(text)
    unmarked = unmarked_heuristic_calls(text)
    undeclared = undeclared_algo_constants(text)
    if bare or unknown or unmarked or undeclared:
        rc = 1
        print(
            "ERROR: cuBLASLt algo-policy invocation in src/vt/cuda/cuda_matmul.cu is not "
            f"routed through a named policy constant ({', '.join(ALGO_POLICY_NAMES)}):",
            file=sys.stderr,
        )
        for line_no, arg in bare:
            print(
                f"  - cuda_matmul.cu:{line_no}: requestedAlgoCount is a bare literal "
                f"{arg} (use {ALGO_CONST})",
                file=sys.stderr,
            )
        for line_no, arg in unknown:
            print(
                f"  - cuda_matmul.cu:{line_no}: requestedAlgoCount is `{arg}`, which is "
                "not on ALGO_POLICY_NAMES. Adding a policy constant is a deliberate edit "
                "to this checker, not a new name at a call site",
                file=sys.stderr,
            )
        if unmarked:
            print(
                f"  - {unmarked} cublasLtMatmulAlgoGetHeuristic call(s) lack the "
                "/*requestedAlgoCount=*/ marker, so the argument is not greppable",
                file=sys.stderr,
            )
        for name in undeclared:
            print(
                f"  - {name} is used but not declared `constexpr int {name} = ...`",
                file=sys.stderr,
            )
        print(
            f"Declare `constexpr int {ALGO_CONST} = 1;` and pass it (behind the "
            "/*requestedAlgoCount=*/ marker) at every cublasLtMatmulAlgoGetHeuristic "
            "call, so an algo-search-policy change is a conscious, greppable edit.",
            file=sys.stderr,
        )
    else:
        used = sorted({arg for _, arg in algo_sites})
        print(
            f"OK (algo-count): {len(algo_sites)} requestedAlgoCount site(s) in "
            f"cuda_matmul.cu all route through a named policy constant "
            f"({', '.join(used) if used else 'none used'})."
        )

    return rc


if __name__ == "__main__":
    raise SystemExit(main())

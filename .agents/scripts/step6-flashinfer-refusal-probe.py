"""Hermetic probe: does the committed harness refuse a FlashInfer other than the pin's?

No GPU, no network, no oracle. Builds the same fixture tree
tests/tools/test_online_gate_client.py builds, then calls record_oracle_manifest
twice: once at the pinned FlashInfer (control) and once at the target's 0.6.18.
Prints an explicit rc per leg so a leg that did not run cannot read as a pass.
"""
import pathlib, sys, tempfile, types
from unittest import mock

# Repo root: argv[1] if given, else derived from this file's own location
# (.agents/scripts/<this>), so the probe runs from any checkout. It must NOT be
# an absolute path to a worktree: AGENTS.md requires a worktree be removed when
# its work merges, and a later wave reusing the directory name would make this
# read a different tree and still print a verdict.
ROOT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2]
if not (ROOT / "tools" / "bench" / "online_gate.py").is_file():
    raise SystemExit(
        f"PROBE ABORT: {ROOT} is not a vllm.cpp checkout "
        "(tools/bench/online_gate.py absent). Pass the repo root as argv[1]."
    )
print(f"REPO_ROOT={ROOT}")
sys.path.insert(0, str(ROOT))

from tools.bench.online_gate import (  # noqa: E402
    FLASHINFER_VERSION, PANDAS_VERSION, record_oracle_manifest,
)
from tools.bench.serve_low_common import (  # noqa: E402
    VLLM_DISTRIBUTION_VERSION, VLLM_ORACLE_VERSION, HarnessError,
)

TARGET_FI = "0.6.18"   # requirements/cuda.txt @ e126687a9a, measured in 2026-09-02-e126687.md 5.2

def run(fi_version, label):
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        bin_dir = root / "venv" / "bin"
        package = root / "venv" / "site-packages" / "vllm"
        dist_info = root / "venv" / "site-packages" / f"vllm-{VLLM_ORACLE_VERSION}.dist-info"
        pandas_package = root / "venv" / "site-packages" / "pandas"
        pandas_dist_info = root / "venv" / "site-packages" / f"pandas-{PANDAS_VERSION}.dist-info"
        fi_package = root / "venv" / "site-packages" / "flashinfer"
        fi_dist_info = root / "venv" / "site-packages" / f"flashinfer_python-{fi_version}.dist-info"
        cutlass = fi_package / "data" / "cutlass"
        for d in (bin_dir, package / "benchmarks" / "datasets",
                  package / "entrypoints" / "cli" / "benchmark", dist_info,
                  pandas_package, pandas_dist_info, fi_package, fi_dist_info,
                  cutlass / "include" / "cutlass", cutlass / "tools" / "util" / "include"):
            d.mkdir(parents=True, exist_ok=True)
        client, python, ninja = bin_dir / "vllm", bin_dir / "python", bin_dir / "ninja"
        package_init = package / "__init__.py"
        for p in (client, python, ninja, package_init,
                  package / "benchmarks" / "serve.py",
                  package / "benchmarks" / "datasets" / "datasets.py",
                  package / "entrypoints" / "cli" / "benchmark" / "serve.py",
                  dist_info / "METADATA", dist_info / "RECORD",
                  pandas_package / "__init__.py", pandas_dist_info / "METADATA",
                  pandas_dist_info / "RECORD", fi_package / "__init__.py",
                  fi_dist_info / "METADATA", fi_dist_info / "RECORD",
                  cutlass / "include" / "cutlass" / "cutlass.h",
                  cutlass / "tools" / "util" / "include" / "helper.h"):
            p.write_text(f"{p.name}\n", encoding="utf-8")
        ninja.chmod(0o755)
        mods = {
            "vllm": types.SimpleNamespace(__version__=VLLM_ORACLE_VERSION, __file__=str(package_init)),
            "pandas": types.SimpleNamespace(__version__=PANDAS_VERSION, __file__=str(pandas_package / "__init__.py")),
            "flashinfer": types.SimpleNamespace(__version__=fi_version, __file__=str(fi_package / "__init__.py")),
        }
        dists = {
            "vllm": types.SimpleNamespace(version=VLLM_DISTRIBUTION_VERSION, _path=dist_info),
            "pandas": types.SimpleNamespace(version=PANDAS_VERSION, _path=pandas_dist_info),
            "flashinfer-python": types.SimpleNamespace(version=fi_version, _path=fi_dist_info),
        }
        with mock.patch.dict("sys.modules", mods), \
             mock.patch("importlib.metadata.distribution", lambda n: dists[n]), \
             mock.patch("shutil.which", lambda n, **k: str(ninja) if n == "ninja" else None):
            try:
                record_oracle_manifest(root / "oracle.json", client=client)
                print(f"LEG {label}: RAN, rc=0, manifest written at flashinfer {fi_version}")
                return "PASSED_ALL"
            except HarnessError as e:
                msg = str(e)
                # online_gate.py:3552-3560 is the FlashInfer gate; :3586 is a LATER
                # check. A leg that dies at :3586 got PAST the FlashInfer gate, which
                # is the only thing this probe is asking about.
                kind = "REFUSED_AT_FLASHINFER" if "FlashInfer" in msg else "PASSED_FLASHINFER_THEN_" + msg[:44]
                print(f"LEG {label}: RAN, rc=1, {kind}: {msg}")
                return kind

print(f"PIN_FLASHINFER={FLASHINFER_VERSION}  TARGET_FLASHINFER={TARGET_FI}")
c = run(FLASHINFER_VERSION, "CONTROL(pin)")
t = run(TARGET_FI, "TARGET(0.6.18)")
print(f"SUM CONTROL={c}")
print(f"SUM TARGET={t}")
control_ok = c == "PASSED_ALL" or c.startswith("PASSED_FLASHINFER_THEN_")
target_ok = t == "REFUSED_AT_FLASHINFER"
if control_ok and target_ok:
    print("VERDICT: DISCRIMINATING. The FlashInfer gate admits the pin and refuses 0.6.18.")
else:
    print("VERDICT: INCONCLUSIVE - the probe did not discriminate; do not read it as a result.")

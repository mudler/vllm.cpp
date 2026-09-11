ID: ISSUE-GH-1213
Title: `AGENTS.md` stated that a leased worker "has no compiler, no downloader and no Python, so it cannot produce a runtime in place", and `.agents/environment.md` carried the matching clause twice for `dgx:gpu0`. All three negatives are false. `rc describe dgx:gpu0` states that a job runs as root in an Ubuntu 24.04 container carrying `git`, `curl`, `wget`, `ssh`, `gcc`, `g++`, `make`, `cmake`, `ninja`, `pkg-config`, `python3`, `pip` and `venv`, and it instructs the reader to install anything missing; the one limit it names is the absent CUDA toolkit. Two jobs then compiled inside a lease on 2026-08-18: `claude/mudler-ubuntu-box/qwen38-gate` apt-installed `cuda-nvcc-13-0` from the `ubuntu2404/sbsa` lane and built this tree 1791/1791 to `BUILD_RC=0` (`/mnt/nas_share/rc/qwen38-gate/out-main/cfg.log` records `nvcc` 13.0.88 and `CUDA feature cutlass-fp8: ENABLED for [121a]`), and `/mnt/nas_share/rc/mtp_test/build.sh` cloned `github.com/mudler/llama.cpp` from inside a job and left a 97 MB `libggml-cuda.so` on the share. **Why it matters:** "the lease cannot produce a runtime in place" is the stated basis for treating the pinned vLLM oracle as unreachable from a lease, and that oracle is the denominator for every speed-parity number the project owes, so the premise needs re-testing rather than inheriting. This claims nothing about a model run; #1185 owns that and stays open. FIXED IN FLOW: the `AGENTS.md` paragraph and both `.agents/environment.md` clauses now say what the measurement supports, and the four real limits (no preinstalled CUDA toolkit, global installs leak until the pod restarts, CIFS `/workspace` holds no symlink so build in `/tmp` and `cp -rL`, and `-j 4` because unconstrained parallelism OOM-reboots the box) plus the host-versus-container egress distinction ride with the correction.
Row: ENV-LEASE-RUNTIME-STAGING
State: UNKNOWN
Kind: record
GitHub: 1213
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:395`

### Frozen archive evidence

> | [#1213](https://github.com/mudler/vllm.cpp/issues/1213) | `ENV-LEASE-RUNTIME-STAGING` | `AGENTS.md` stated that a leased worker "has no compiler, no downloader and no Python, so it cannot produce a runtime in place", and `.agents/environment.md` carried the matching clause twice for `dgx:gpu0`. All three negatives are false. `rc describe dgx:gpu0` states that a job runs as root in an Ubuntu 24.04 container carrying `git`, `curl`, `wget`, `ssh`, `gcc`, `g++`, `make`, `cmake`, `ninja`, `pkg-config`, `python3`, `pip` and `venv`, and it instructs the reader to install anything missing; the one limit it names is the absent CUDA toolkit. Two jobs then compiled inside a lease on 2026-08-18: `claude/mudler-ubuntu-box/qwen38-gate` apt-installed `cuda-nvcc-13-0` from the `ubuntu2404/sbsa` lane and built this tree 1791/1791 to `BUILD_RC=0` (`/mnt/nas_share/rc/qwen38-gate/out-main/cfg.log` records `nvcc` 13.0.88 and `CUDA feature cutlass-fp8: ENABLED for [121a]`), and `/mnt/nas_share/rc/mtp_test/build.sh` cloned `github.com/mudler/llama.cpp` from inside a job and left a 97 MB `libggml-cuda.so` on the share. **Why it matters:** "the lease cannot produce a runtime in place" is the stated basis for treating the pinned vLLM oracle as unreachable from a lease, and that oracle is the denominator for every speed-parity number the project owes, so the premise needs re-testing rather than inheriting. This claims nothing about a model run; #1185 owns that and stays open. FIXED IN FLOW: the `AGENTS.md` paragraph and both `.agents/environment.md` clauses now say what the measurement supports, and the four real limits (no preinstalled CUDA toolkit, global installs leak until the pod restarts, CIFS `/workspace` holds no symlink so build in `/tmp` and `cp -rL`, and `-j 4` because unconstrained parallelism OOM-reboots the box) plus the host-versus-container egress distinction ride with the correction. | record |

## Resolution

-

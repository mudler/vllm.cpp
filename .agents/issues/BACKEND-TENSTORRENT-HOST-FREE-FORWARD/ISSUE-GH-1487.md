ID: ISSUE-GH-1487
Title: test_release_metadata is red on aarch64 hosts: the fixture stages the host /bin/true into an x86_64-named archive
Row: BACKEND-TENSTORRENT-HOST-FREE-FORWARD
State: CLOSED
Kind: bug
GitHub: 1487
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-20
Updated: 2026-08-22
Closed: 2026-08-22

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Measured on thalia (aarch64, Blackhole P150 dev host) at `206afb63` == `origin/main`, zero local commits, uncommitted diff touching only `src/vt`/`src/vllm` C++ (the suite reads only committed scripts and its own scratch fixture):
>
> ```
> FAIL: test_metadata_is_installed_archived_and_validated_with_final_sidecars
> AssertionError: 1 != 0 : release archive validation FAILED:
>   - ELF host architecture does not match manifest: ELF 64-bit LSB pie
>     executable, ARM aarch64, ..., stripped
> ```
>
> Cause, read at the source: `tests/scripts/test_release_metadata.py:55` stages the HOST arch into the package — `shutil.copy2("/bin/true", stage / "bin/vllm-server")` — while the test packages it under the hardcoded name `vllm.cpp-0.0.1-linux-x86_64-glibc-cpu.tar.gz` (`:143`). On an x86_64 host the staged ELF matches the manifest by accident; on every aarch64 host (the entire TT dev fleet) the validator correctly refuses. Not the #1353 disk-full mode: 902G free and no scratch-repo failures — the message is the arch mismatch itself.
>
> Impact: `scripts/agent-preflight.sh` cannot go green on aarch64 hosts at all (1 gate red, 86 otherwise ok), which forces every TT row to hand-attribute an inherited red before landing.
>
> Repair direction: make the fixture self-consistent — stage an ELF whose arch matches the archive/manifest the test declares (derive the expected arch from the staged binary, or commit a tiny fixture ELF per declared arch), so the validator is exercised for exactly what it exists to catch: a manifest that lies about the payload.
>
> Found while running preflight for #1476. Owning row `BACKEND-TENSTORRENT-HOST-FREE-FORWARD`, listed under `## Owed` in its spec (no release-infrastructure row is a better owner in `docs/STATUS.md`).

## Resolution

GitHub records closing pull request #1634 (https://github.com/mudler/vllm.cpp/pull/1634) merged on 2026-08-22 as commit `9e56e5c90dbe6289524cb65e64212b32cf19cfa1`. GitHub closed issue #1487 on 2026-08-22.

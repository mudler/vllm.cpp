ID: ISSUE-GH-1661
Title: test_script_stays_shellcheck_clean ERRORs (not skips) on hosts without the shellcheck binary: the guard probes returncode but a missing binary raises
Row: ENG-PREFLIGHT-COMPILES
State: CLOSED
Kind: bug
GitHub: 1661
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-22
Updated: 2026-08-22
Closed: 2026-08-22

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What
>
> `tests/tools/test_online_gate_startup.py:258-260` (`DriverStartupContractTest.test_script_stays_shellcheck_clean`) intends to skip when shellcheck is unavailable:
>
> ```python
> probe = subprocess.run(["shellcheck", "--version"], capture_output=True, check=False)
> if probe.returncode != 0:
>     self.skipTest("shellcheck is unavailable")
> ```
>
> On a host where the `shellcheck` **binary does not exist**, `subprocess.run` raises `FileNotFoundError` before any returncode exists, so the test **ERRORs** instead of skipping — and the `tools suites` gate in `scripts/agent-preflight.sh` goes red:
>
> ```
> ERROR: test_script_stays_shellcheck_clean (tests.tools.test_online_gate_startup.DriverStartupContractTest.test_script_stays_shellcheck_clean)
> FileNotFoundError: [Errno 2] No such file or directory: 'shellcheck'
> ```
>
> Reproduced on `thalia` (aarch64 TT dev host, no shellcheck installed) at `origin/main` `6ec2cf882` with zero local commits touching the module. The test predates the window (last touched by `b95543c44`); it surfaced now because current main's preflight runs the `tools suites` module.
>
> ## Impact
>
> Same failure class #1487 just closed: `agent-preflight` cannot go green on hosts lacking shellcheck, forcing every row to hand-attribute an inherited red.
>
> ## Repair
>
> Wrap the probe in `try/except FileNotFoundError: self.skipTest("shellcheck is unavailable")` (keep the returncode arm for a present-but-broken binary). Found while rebasing #1630/#1634; fixed in flow as a separate unit.

## Resolution

GitHub records closing pull request #1662 (https://github.com/mudler/vllm.cpp/pull/1662) merged on 2026-08-22 as commit `73ada0df8de72853a20f9f0e5b00e33b1ab02a6c`. GitHub closed issue #1661 on 2026-08-22.

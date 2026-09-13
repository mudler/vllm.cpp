ID: ISSUE-GH-3171
Title: windows-msvc: build-windows-release.ps1 calls dumpbin/cl without MSVC env on PATH
Row: ENG-RELEASE-WINDOWS
State: OPEN
Kind: UNKNOWN
GitHub: 3171
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `WINDOWS-584`
>
> `scripts/build-windows-release.ps1` calls `dumpbin` (lines 845, 935-939),
> `cl` (line 957), and reads `VCToolsVersion`/`UCRTVersion` (lines 958-959)
> without setting up the MSVC developer environment. The CMake configure uses
> the Visual Studio generator (`-G "Visual Studio 17 2022"`, line 763), which
> finds the compiler internally through the registry. But the post-build audit
> steps call MSVC tools directly from PowerShell, where `dumpbin` and `cl` are
> not on PATH and the MSVC environment variables are not set.
>
> This was always broken but was masked by the `0xC0000409` crash in
> `test_openai_api_server.exe` (fixed in PR #3168, `fopen("re")` → `fopen("r")`).
> The crash at line 816 prevented execution from ever reaching line 845. Now
> that the crash is fixed, all 82 test cases pass with `Status: SUCCESS!`, and
> execution reaches the `Invoke-CrtAudit` step, which fails:
>
> ```
> Invoke-CrtAudit: The term 'dumpbin' is not recognized as a name of a
> cmdlet, function, script file, or executable program.
> ```
>
> Both `windows-msvc-cpu` and `windows-msvc-vulkan` fail identically on the
> baseline run (34722231726).
>
> The fix: set up the MSVC developer environment (via `vswhere` + `vcvars64.bat`)
> early in the script, before the post-build steps that need `dumpbin`, `cl`,
> `VCToolsVersion`, and `UCRTVersion`.

## Resolution

-

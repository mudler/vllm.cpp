[CmdletBinding()]
param(
    [string]$SourceDir = (Resolve-Path (Join-Path $PSScriptRoot "..")),
    [string]$BuildDir = (Join-Path $SourceDir "build-release-windows-cpu"),
    [string]$StageDir = (Join-Path $BuildDir "stage"),
    [ValidateSet("cpu", "vulkan")][string]$Backend = "cpu",
    [string]$ArtifactId = "",
    [string]$SmokeModel = (Join-Path $SourceDir "tests/vllm/models/fixtures/llama_embed_e2e"),
    [int]$SmokePort = 18080,
    [switch]$ContractTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $ArtifactId) { $ArtifactId = "windows-x86_64-msvc-$Backend" }
if ($ArtifactId -ne "windows-x86_64-msvc-$Backend") {
    throw "artifact ID must exactly match selected Windows backend"
}
function Invoke-Checked {
    param([Parameter(Mandatory)][string]$Program,
          [Parameter(Mandatory)][AllowEmptyCollection()][string[]]$Arguments,
          [scriptblock]$Runner)
    if ($null -eq $Runner) {
        & $Program @Arguments
        $exitCode = $LASTEXITCODE
    } else {
        $exitCode = [int](& $Runner $Program $Arguments)
    }
    if ($exitCode -ne 0) {
        throw "$Program exited with status $exitCode"
    }
}

# `Arguments` is mandatory *and* `[AllowEmptyCollection()]` rather than defaulted
# to `@()`, so that an explicitly empty list binds while an omitted or null one
# stays a hard binding error. A default would silently turn "forwarded nothing"
# into "forwarded an empty list", which is the confusion #512 came from, so both
# halves of that design are pinned here.
function Invoke-CheckedBindingContractTests {
    $recorder = { param([string]$Program, [string[]]$Arguments) return 0 }

    $nullRejected = $false
    try {
        Invoke-Checked "fake-null.exe" $null -Runner $recorder
    } catch {
        $nullRejected = $true
    }
    if (-not $nullRejected) {
        throw "checked invocation bound a null argument list"
    }

    # An omitted mandatory parameter *prompts* in an interactive console host, so
    # asserting the omission in-process would hang a developer's terminal. An API
    # runspace has a host that cannot prompt and reports the binding failure
    # instead. The function under test is rebuilt from the live definition's own
    # source text, so any edit to the real parameter block is what gets asserted.
    $runspace = [powershell]::Create()
    $omissionRejected = $false
    try {
        $null = $runspace.AddScript(@'
param([string]$Body)
Set-Item -LiteralPath function:Invoke-Checked -Value ([scriptblock]::Create($Body))
Invoke-Checked "fake-omitted.exe" -Runner { param([string]$Program, [string[]]$Arguments) return 0 }
'@).AddArgument(${function:Invoke-Checked}.ToString())
        try {
            $null = $runspace.Invoke()
        } catch {
            $omissionRejected =
                $_.Exception.InnerException -is [System.Management.Automation.ParameterBindingException]
            if (-not $omissionRejected) { throw }
        }
        $omissionRejected = $omissionRejected -or @($runspace.Streams.Error | Where-Object {
            $_.Exception -is [System.Management.Automation.ParameterBindingException]
        }).Count -gt 0
    } finally {
        $runspace.Dispose()
    }
    if (-not $omissionRejected) {
        throw "omitting the argument list was not a mandatory-parameter binding error"
    }
}

# The fake-runner arm below never executes `& $Program @Arguments`, so on its own
# it cannot catch an edit that stops propagating the child's exit status or stops
# forwarding argv. This arm drives the real branch end to end.
#
# The program it drives is the PowerShell host executing this script. That is the
# one executable guaranteed to exist wherever this script can run, so the same
# assertions execute on the Windows runners and on POSIX developer boxes with no
# platform branch that could silently no-op on one of them (#512).
function Invoke-CheckedRealProcessContractTests {
    $pwshPath = (Get-Process -Id $PID).Path
    if (-not $pwshPath) {
        throw "real-process contract test could not resolve the running PowerShell host"
    }
    $scratch = Join-Path ([System.IO.Path]::GetTempPath()) `
        ("vllm-cpp-checked-" + [guid]::NewGuid().ToString("n"))
    New-Item -ItemType Directory -Force -Path $scratch | Out-Null
    try {
        Invoke-Checked $pwshPath @("-NoProfile", "-Command", "exit 0")

        $nonzeroRejected = $false
        try {
            Invoke-Checked $pwshPath @("-NoProfile", "-Command", "exit 3")
        } catch {
            $nonzeroRejected = $true
            if ($_.Exception.Message -notmatch 'exited with status 3$') {
                throw "real-process failure did not report the child's own exit status: $($_.Exception.Message)"
            }
        }
        if (-not $nonzeroRejected) {
            throw "real-process nonzero exit status was accepted"
        }

        # Exits 0 only for three *distinct* argv entries, the first of which holds
        # a space: joining, re-quoting, truncating or reordering the forwarded
        # list all land on a different exit status.
        $argvProbe = Join-Path $scratch "argv-probe.ps1"
        @'
if ($args.Count -ne 3) { exit 21 }
if ($args[0] -ne 'one two' -or $args[1] -ne 'three' -or $args[2] -ne 'four') { exit 22 }
exit 0
'@ | Set-Content -LiteralPath $argvProbe -Encoding utf8NoBOM
        Invoke-Checked $pwshPath @("-NoProfile", "-File", $argvProbe, "one two", "three", "four")

        # The production calls this branch exists for forward an explicitly empty
        # list to a program that takes no arguments, so drive that shape for real
        # rather than only through the fake runner (#512).
        $emptyProbe = Join-Path $scratch "empty-probe.ps1"
        @'
if ($args.Count -ne 0) { exit 23 }
exit 0
'@ | Set-Content -LiteralPath $emptyProbe -Encoding utf8NoBOM
        Invoke-Checked $emptyProbe @()
    } finally {
        Remove-Item -Recurse -Force -LiteralPath $scratch -ErrorAction SilentlyContinue
    }
}

# Most of this script's checked invocations run a test executable that takes no
# arguments, so `Invoke-Checked` must bind an explicitly empty argument list and
# still forward it verbatim (#512).
function Invoke-CheckedContractTests {
    $calls = [System.Collections.Generic.List[object]]::new()
    $recorder = {
        param([string]$Program, [string[]]$Arguments)
        $calls.Add([pscustomobject]@{
            Program = $Program
            Arguments = @($Arguments)
        }) | Out-Null
        return 0
    }.GetNewClosure()

    Invoke-Checked "fake-empty.exe" @() -Runner $recorder
    Invoke-Checked "fake-args.exe" @("--help", "--verbose") -Runner $recorder

    if ($calls.Count -ne 2) {
        throw "checked-invocation fake runner was not invoked exactly twice"
    }
    if ($calls[0].Program -ne "fake-empty.exe" -or $calls[1].Program -ne "fake-args.exe") {
        throw "checked invocation did not forward its exact program"
    }
    if ($calls[0].Arguments.Count -ne 0) {
        throw "checked invocation did not forward an explicitly empty argument list"
    }
    if ($calls[1].Arguments.Count -ne 2 -or
        $calls[1].Arguments[0] -ne "--help" -or
        $calls[1].Arguments[1] -ne "--verbose") {
        throw "checked invocation did not forward its exact argument list"
    }

    $failing = { param([string]$Program, [string[]]$Arguments) return 3 }
    foreach ($rejectedName in @("empty", "non-empty")) {
        $rejected = $false
        try {
            if ($rejectedName -eq "empty") {
                Invoke-Checked "fake-fail.exe" @() -Runner $failing
            } else {
                Invoke-Checked "fake-fail.exe" @("--help") -Runner $failing
            }
        } catch {
            $rejected = $true
        }
        if (-not $rejected) {
            throw "nonzero $rejectedName-argument exit status was accepted"
        }
    }

    Invoke-CheckedBindingContractTests
    Invoke-CheckedRealProcessContractTests
}

function Assert-CrtPolicy {
    param([Parameter(Mandatory)][string[]]$DirectiveOutput,
          [Parameter(Mandatory)][string[]]$ImportOutput)
    $directives = $DirectiveOutput -join "`n"
    $imports = $ImportOutput -join "`n"
    if ($directives -notmatch '(?im)DEFAULTLIB\s*:\s*"?LIBCMT"?') {
        throw "COFF CRT audit: no /DEFAULTLIB:LIBCMT static CRT directive found"
    }
    if ($directives -match '(?im)DEFAULTLIB\s*:\s*"?(?:MSVCRT|MSVCPRT|LIBCMTD)"?') {
        throw "COFF CRT audit: dynamic or debug CRT directive found"
    }
    if ($imports -match '(?im)^\s*(?:VCRUNTIME[^\s]*|MSVCP[^\s]*|CONCRT[^\s]*|UCRTBASED?|api-ms-win-crt-[^\s]*|MSVCR[^\s]*)\.dll\s*$') {
        throw "PE CRT audit: dynamic or debug CRT DLL import found"
    }
}

function Invoke-CrtAudit {
    param([Parameter(Mandatory)][string[]]$Artifacts,
          [Parameter(Mandatory)][string]$Server,
          [scriptblock]$DumpbinRunner = {
              param([string]$Mode, [string]$Path)
              $output = & dumpbin $Mode $Path 2>&1
              if ($LASTEXITCODE -ne 0) {
                  throw "dumpbin $Mode failed for $Path with status $LASTEXITCODE"
              }
              return @($output)
          })
    $directiveOutput = @()
    foreach ($artifact in $Artifacts) {
        $directiveOutput += & $DumpbinRunner "/directives" $artifact
    }
    $importOutput = @(& $DumpbinRunner "/imports" $Server)
    Assert-CrtPolicy -DirectiveOutput $directiveOutput -ImportOutput $importOutput
    Write-Host ($directiveOutput -join "`n")
    Write-Host ($importOutput -join "`n")
}

function Invoke-CrtContractTests {
    $good = {
        param([string]$Mode, [string]$Path)
        if ($Mode -eq "/directives") { return '/DEFAULTLIB:"LIBCMT"' }
        return @("$Path", "KERNEL32.dll", "WS2_32.dll")
    }
    Invoke-CrtAudit -Artifacts @("fake-vllm.lib", "fake-server.obj") `
        -Server "fake-vllm-server.exe" -DumpbinRunner $good
    foreach ($bad in @(
        { param($Mode, $Path) if ($Mode -eq "/directives") { '/DEFAULTLIB:"MSVCRT"' } else { 'KERNEL32.dll' } },
        { param($Mode, $Path) if ($Mode -eq "/directives") { '/DEFAULTLIB:"LIBCMT"' } else { 'UCRTBASE.dll' } }
    )) {
        $rejected = $false
        try {
            Invoke-CrtAudit -Artifacts @("fake.lib") -Server "fake.exe" `
                -DumpbinRunner $bad
        } catch {
            $rejected = $true
        }
        if (-not $rejected) { throw "injected bad dumpbin output was accepted" }
    }
}

function Invoke-UnsupportedTierProbe {
    param([Parameter(Mandatory)][string]$TierTest,
          [scriptblock]$Runner)
    $arguments = @(
        '--test-case=elementwise CPU GEMM: the forced tier is the tier that actually ran'
    )
    if ($null -eq $Runner) {
        $probeOutput = @(& $TierTest @arguments 2>&1)
        $probeExitCode = $LASTEXITCODE
    } else {
        $probeResult = & $Runner $TierTest $arguments
        $probeOutput = @($probeResult.Output)
        $probeExitCode = [int]$probeResult.ExitCode
    }
    if ($probeExitCode -ne 1) {
        throw "unsupported forced CPU tier probe exited with status $probeExitCode instead of 1"
    }
    $diagnostic = $probeOutput -join "`n"
    if ($diagnostic -notmatch [regex]::Escape("unknown x86 ISA tier 'amx'")) {
        throw "unsupported forced CPU tier probe did not report the expected diagnostic"
    }
}

function Invoke-UnsupportedTierContractTests {
    $diagnostic = "unknown x86 ISA tier 'amx'"
    $calls = [System.Collections.Generic.List[object]]::new()
    $good = {
        param([string]$Program, [string[]]$Arguments)
        $calls.Add([pscustomobject]@{
            Program = $Program
            Arguments = @($Arguments)
        }) | Out-Null
        [pscustomobject]@{ ExitCode = 1; Output = @($diagnostic) }
    }.GetNewClosure()
    Invoke-UnsupportedTierProbe -TierTest "fake-tier-test.exe" -Runner $good
    if ($calls.Count -ne 1) {
        throw "unsupported-tier fake runner was not invoked exactly once"
    }
    if ($calls[0].Arguments.Count -ne 1 -or
        $calls[0].Arguments[0] -ne
            '--test-case=elementwise CPU GEMM: the forced tier is the tier that actually ran') {
        throw "unsupported-tier fake runner did not receive one exact filter argument"
    }

    $badResults = @(
        [pscustomobject]@{ ExitCode = 0; Output = @($diagnostic) },
        [pscustomobject]@{ ExitCode = 134; Output = @($diagnostic) },
        [pscustomobject]@{ ExitCode = -1073741819; Output = @($diagnostic) },
        [pscustomobject]@{ ExitCode = 3; Output = @($diagnostic) },
        [pscustomobject]@{ ExitCode = 2; Output = @($diagnostic) },
        [pscustomobject]@{ ExitCode = 1; Output = @("wrong diagnostic") }
    )
    foreach ($badResult in $badResults) {
        $runner = {
            param([string]$Program, [string[]]$Arguments)
            return $badResult
        }.GetNewClosure()
        $rejected = $false
        try {
            Invoke-UnsupportedTierProbe -TierTest "fake-tier-test.exe" `
                -Runner $runner
        } catch {
            $rejected = $true
        }
        if (-not $rejected) {
            throw "injected bad unsupported-tier result was accepted"
        }
    }
}

if ($ContractTest) {
    Invoke-CheckedContractTests
    Invoke-CrtContractTests
    Invoke-UnsupportedTierContractTests
    Write-Host "Windows PowerShell/CRT contract tests OK"
    exit 0
}

foreach ($name in @("SOURCE_SHA", "VERSION", "EVIDENCE_URL", "SOURCE_DATE_EPOCH")) {
    if (-not [Environment]::GetEnvironmentVariable($name)) {
        throw "$name is required"
    }
}

if (-not (Test-Path (Join-Path $SmokeModel "config.json"))) {
    throw "Windows runtime smoke model is incomplete: $SmokeModel"
}

$queryDir = Join-Path $BuildDir ".cmake/api/v1/query"
New-Item -ItemType Directory -Force -Path $queryDir | Out-Null
New-Item -ItemType File -Force -Path (Join-Path $queryDir "codemodel-v2") | Out-Null

Invoke-Checked cmake @(
    "-S", $SourceDir,
    "-B", $BuildDir,
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DVLLM_CPP_BUILD_TESTS=ON",
    "-DVLLM_CPP_BUILD_VERSION=$env:VERSION",
    "-DVLLM_CPP_BUILD_EXAMPLES=ON",
    "-DVLLM_CPP_SERVER=ON",
    "-DVLLM_CPP_CUDA=OFF",
    "-DVLLM_CPP_CUDA_ARCHITECTURES=",
    "-DVLLM_CPP_HIP=OFF",
    "-DVLLM_CPP_HIP_ARCHITECTURES=",
    "-DVLLM_CPP_METAL=OFF",
    "-DVLLM_CPP_MLX=OFF",
    "-DMLX_ROOT=",
    "-DVLLM_CPP_TRITON=OFF",
    "-DVLLM_CPP_VULKAN=$(if ($Backend -eq 'vulkan') { 'ON' } else { 'OFF' })",
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"
)
Invoke-Checked python @(
    (Join-Path $SourceDir "scripts/check-windows-portability.py"),
    "--root", $SourceDir,
    "--build-dir", $BuildDir
)

$targets = @(
    "server",
    "test_openai_api_server",
    "test_lmcache_client",
    "test_kv_offload_fs",
    "test_cpu_isa_x86",
    "test_ops_matmul_elem",
    "test_vulkan_loader"
)
if ($Backend -eq "vulkan") {
    $targets += @("test_vulkan_backend", "test_backend_cross_device")
}
Invoke-Checked cmake (@("--build", $BuildDir, "--config", "Release", "--target") + $targets)

foreach ($test in @(
    "test_openai_api_server.exe",
    "test_lmcache_client.exe",
    "test_kv_offload_fs.exe",
    "test_cpu_isa_x86.exe"
)) {
    Invoke-Checked (Join-Path $BuildDir "tests/Release/$test") @()
}
Invoke-Checked (Join-Path $BuildDir "tests/Release/test_vulkan_loader.exe") @()
if ($Backend -eq "vulkan") {
    Invoke-Checked (Join-Path $BuildDir "tests/Release/test_vulkan_backend.exe") @()
    Invoke-Checked (Join-Path $BuildDir "tests/Release/test_backend_cross_device.exe") @()
}

if (Test-Path $StageDir) {
    Remove-Item -Recurse -Force $StageDir
}
Invoke-Checked cmake @(
    "--install", $BuildDir,
    "--config", "Release",
    "--prefix", $StageDir,
    "--component", "vllm-server"
)

$server = Join-Path $StageDir "bin/vllm-server.exe"
if (-not (Test-Path $server)) {
    throw "native install did not stage bin/vllm-server.exe"
}
$crtArtifacts = @(
    Get-ChildItem -Path $BuildDir -Recurse -File -Include "*.obj", "vllm*.lib" |
        Where-Object { $_.FullName -notmatch '[\\/](?:_deps|third_party)[\\/]' } |
        ForEach-Object { $_.FullName }
)
if ($crtArtifacts.Count -eq 0) {
    throw "COFF CRT audit found no project objects or static libraries"
}
Invoke-CrtAudit -Artifacts $crtArtifacts -Server $server

Invoke-Checked $server @("--help")

$tierTest = Join-Path $BuildDir "tests/Release/test_ops_matmul_elem.exe"
$savedTier = $env:VT_CPU_MATMUL_TIER
try {
    $env:VT_CPU_MATMUL_TIER = "portable"
    Invoke-Checked $tierTest @()
    $env:VT_CPU_MATMUL_TIER = "avx2"
    Invoke-Checked $tierTest @()
    $env:VT_CPU_MATMUL_TIER = "amx"
    Invoke-UnsupportedTierProbe -TierTest $tierTest
} finally {
    $env:VT_CPU_MATMUL_TIER = $savedTier
}

# Python's Windows subprocess path calls CreateProcess with
# CREATE_NEW_PROCESS_GROUP, letting the smoke target one CTRL_BREAK_EVENT at the
# extracted server without broadcasting to the Actions runner's console.
$smokeHarness = Join-Path $BuildDir "windows_server_smoke.py"
@'
import json
import signal
import subprocess
import sys
import time
import urllib.request

server, model, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
proc = subprocess.Popen(
    [server, "--model", model, "--host", "127.0.0.1", "--port", str(port)],
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
)
try:
    deadline = time.monotonic() + 60
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before health check: {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("/health did not return 200 within 60 seconds")
        time.sleep(0.1)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/version", timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"/version returned {response.status}")
        json.loads(response.read())
    proc.send_signal(signal.CTRL_BREAK_EVENT)
    if proc.wait(timeout=20) != 0:
        raise RuntimeError(f"server did not stop cleanly: {proc.returncode}")
finally:
    if proc.poll() is None:
        proc.kill()
        proc.wait()
'@ | Set-Content -LiteralPath $smokeHarness -Encoding utf8NoBOM

Invoke-Checked python @($smokeHarness, $server, $SmokeModel, "$SmokePort")

$releaseDir = Join-Path $BuildDir "release"
$metadataDir = Join-Path $releaseDir "metadata"
$tierReport = Join-Path $releaseDir "cpu-tier-report.json"
$peReport = Join-Path $releaseDir "pe-audit.json"
$archive = Join-Path $releaseDir "vllm.cpp-$($env:VERSION)-$ArtifactId.zip"
New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null

$passed = @{
    command = "forced Windows CPU tier"; reason = ""; result = "exit 0"
    state = "passed"; url = $env:EVIDENCE_URL
}
$absent = @{
    command = ""; reason = "not executed by the Windows preview gate"; result = ""
    state = "absent"; url = ""
}
@{
    schema = "vllm.cpp.cpu-tier-report.v1"
    selected_tier = "avx2-f16c"
    commands = @("VT_CPU_MATMUL_TIER=portable", "VT_CPU_MATMUL_TIER=avx2")
    tiers = [ordered]@{
        "portable-sse2" = $passed
        "sse2-f16c" = $absent
        "avx2-f16c" = $passed
        "avx512f" = $absent
    }
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $tierReport -Encoding utf8NoBOM

$headerOutput = @(& dumpbin /nologo /headers $server 2>&1)
if ($LASTEXITCODE -ne 0) { throw "dumpbin /headers failed" }
$dependentOutput = @(& dumpbin /nologo /dependents $server 2>&1)
if ($LASTEXITCODE -ne 0) { throw "dumpbin /dependents failed" }
$rawOutput = @(& dumpbin /nologo /rawdata $server 2>&1)
if ($LASTEXITCODE -ne 0) { throw "dumpbin /rawdata failed" }
$machine = if (($headerOutput -join "`n") -match '(?im)^\s*(8664)\s+machine') { $Matches[1] } else { "" }
$imports = @(
    $dependentOutput | ForEach-Object {
        if ($_ -match '^\s*([A-Za-z0-9_.+-]+\.dll)\s*$') { $Matches[1] }
    } | Sort-Object -Unique
)
$debugPaths = @(
    (($headerOutput + $rawOutput) -join "`n") |
        Select-String -AllMatches -Pattern '(?i)[A-Za-z]:[\\/][^\r\n\x00]*?\.pdb' |
        ForEach-Object { $_.Matches.Value } | Sort-Object -Unique
)
@{
    schema = "vllm.cpp.pe-audit.v1"; machine = $machine
    imports = $imports; debug_paths = $debugPaths
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $peReport -Encoding utf8NoBOM

$compiler = (& cl 2>&1 | Select-Object -First 1) -join ""
$toolsetVersion = if ($env:VCToolsVersion) { $env:VCToolsVersion.TrimEnd('\') } else { throw "VCToolsVersion is required" }
$ucrtVersion = if ($env:UCRTVersion) { $env:UCRTVersion.TrimEnd('\') } else { throw "UCRTVersion is required" }
$abiVersion = ($toolsetVersion -split '\.')[0..1] -join '.'
$cAbiVersion = (Select-String -Path (Join-Path $SourceDir "include/vllm.h") -Pattern '^#define VLLM_ABI_VERSION ([0-9]+)$').Matches.Groups[1].Value
Invoke-Checked python @(
    (Join-Path $SourceDir "scripts/release_metadata.py"),
    "--repo-root", $SourceDir, "--build-dir", $BuildDir, "--stage-dir", $StageDir,
    "--output-dir", $metadataDir, "--tier-report", $tierReport,
    "--artifact-id", $ArtifactId, "--channel", "preview", "--backend", $Backend,
    "--version", $env:VERSION, "--c-abi-version", $cAbiVersion,
    "--source-commit", $env:SOURCE_SHA, "--source-clean", "--abi-version", $abiVersion,
    "--compiler", $compiler, "--toolchain", "Visual Studio 2022 v143 /MT",
    "--toolset-version", $toolsetVersion, "--ucrt-version", $ucrtVersion,
    "--pe-report", $peReport, "--evidence-url", $env:EVIDENCE_URL
)
Invoke-Checked python @(
    (Join-Path $SourceDir "scripts/package-server.py"), "--build-dir", $BuildDir,
    "--stage-dir", $StageDir, "--metadata-dir", $metadataDir,
    "--archive", $archive, "--archive-format", "zip", "--config", "Release"
)
$archiveExtract = Join-Path $releaseDir "archive-extracted"
if (Test-Path $archiveExtract) { Remove-Item -Recurse -Force $archiveExtract }
Expand-Archive -LiteralPath $archive -DestinationPath $archiveExtract
$archiveServer = Join-Path $archiveExtract "bin/vllm-server.exe"
if (-not (Test-Path $archiveServer)) {
    throw "final ZIP does not contain bin/vllm-server.exe"
}
Invoke-Checked $archiveServer @("--help")
Invoke-Checked python @($smokeHarness, $archiveServer, $SmokeModel, "$SmokePort")
Invoke-Checked python @(
    (Join-Path $SourceDir "scripts/validate-release-archive.py"),
    "--archive", $archive, "--archive-format", "zip",
    "--checksum", "$archive.sha256", "--provenance", "$archive.provenance.json",
    "--repo-root", $SourceDir, "--forbid-path", $BuildDir
)
Write-Host "Windows native $Backend build/stage/ZIP gate OK: $archive"

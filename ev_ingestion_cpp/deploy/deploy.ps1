<#
.SYNOPSIS
  Deploys the Predator detector from this repo to the Jetson Orin (Phase 33.1).

.DESCRIPTION
  The repo's ev_ingestion_cpp/ directory is the single source of truth.
  1. Computes a git revision tag (short hash, "-dirty" if ev_ingestion_cpp has
     uncommitted changes).
  2. Copies every build input (*.cpp, *.hpp, *.cu, *.cuh, CMakeLists.txt) to
     ~/ev_deploy/src on the Orin, plus the Orin-side build script.
  3. Runs orin_build_install.sh remotely: build -> unit tests -> install ->
     restart -> verify the live /pipeline_stats build_id.

  Uses the SSH alias 'orin-nano' (key-based auth from ~/.ssh/config). No
  credentials are stored or passed by this script.

.PARAMETER NoRestart
  Build, test and install, but do not restart the service.

.EXAMPLE
  powershell -File ev_ingestion_cpp\deploy\deploy.ps1
#>
[CmdletBinding()]
param(
    [switch]$NoRestart,
    [string]$SshHost = "orin-nano"
)
$ErrorActionPreference = "Stop"

$cppDir   = Split-Path -Parent $PSScriptRoot          # ev_ingestion_cpp
$repoRoot = Split-Path -Parent $cppDir

# ---- 1. Revision tag ----
$rev = (git -C $repoRoot rev-parse --short HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or -not $rev) { throw "git rev-parse failed" }
$dirty = git -C $repoRoot status --porcelain -- ev_ingestion_cpp
if ($dirty) { $rev = "$rev-dirty" }
Write-Host "[deploy] Revision: $rev"

# ---- 2. Collect build inputs (top level of ev_ingestion_cpp only) ----
$inputs = Get-ChildItem -LiteralPath $cppDir -File |
    Where-Object { $_.Extension -in ".cpp", ".hpp", ".cu", ".cuh" -or $_.Name -eq "CMakeLists.txt" }
if ($inputs.Count -eq 0) { throw "No build inputs found in $cppDir" }

# ---- 3. Copy to the Orin ----
$paths = $inputs | ForEach-Object { $_.FullName }
& scp -q @paths "${SshHost}:ev_deploy/src/"
if ($LASTEXITCODE -ne 0) { throw "scp of sources failed" }
$hotPixelsFile = Join-Path $cppDir "hot_pixels.txt"
if (Test-Path $hotPixelsFile) {
    & scp -q $hotPixelsFile "${SshHost}:ev_deploy/hot_pixels.txt"
}
& scp -q (Join-Path $PSScriptRoot "orin_build_install.sh") "${SshHost}:ev_deploy/orin_build_install.sh"
if ($LASTEXITCODE -ne 0) { throw "scp of build script failed" }
Write-Host "[deploy] Copied $($inputs.Count) build inputs to ${SshHost}:~/ev_deploy/src"

# ---- 4. Remote build/test/install/restart ----
$flag = if ($NoRestart) { " --no-restart" } else { "" }
& ssh -o BatchMode=yes $SshHost "sed -i 's/\r`$//' ev_deploy/orin_build_install.sh && bash ev_deploy/orin_build_install.sh $rev$flag"
$code = $LASTEXITCODE
switch ($code) {
    0 { Write-Host "[deploy] SUCCESS" }
    5 { Write-Warning "[deploy] Installed, but restart/verify did not complete (exit 5). See output above." }
    default { throw "[deploy] Remote build/install failed with exit code $code" }
}
exit $code

[CmdletBinding()]
param(
    [string]$PiHost = "192.168.0.3",
    [string]$WslDistro = "Ubuntu",
    [switch]$DetachEventCamera
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $RepoRoot ".runtime"
$DashboardUrl = "http://${PiHost}:8080"

Write-Host "[Vollebak] Pausing and parking gimbal" -ForegroundColor Cyan
try {
    $offBody = @{ enabled = $false } | ConvertTo-Json -Compress
    Invoke-RestMethod -Method Post -Uri "$DashboardUrl/api/tracking" -ContentType "application/json" -Body $offBody -TimeoutSec 5 | Out-Null
    Invoke-RestMethod -Method Post -Uri "$DashboardUrl/api/home" -ContentType "application/json" -Body "{}" -TimeoutSec 5 | Out-Null
} catch {
    Write-Warning "Could not reach the Pi dashboard; use the physical servo-power switch if needed."
}

foreach ($name in @("rgb-tunnel", "event-tunnel", "rgb-bridge", "event-detector", "wsl-keepalive")) {
    $pidPath = Join-Path $RuntimeDir "$name.pid"
    if (-not (Test-Path $pidPath)) { continue }
    $savedPid = Get-Content $pidPath -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($savedPid -match '^\d+$') {
        Stop-Process -Id ([int]$savedPid) -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
}

# Also clean up helpers launched before PID tracking was introduced, or whose
# PID files were removed. Match only this project's bridge and reverse ports.
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        ($_.Name -match '^python(?:\.exe)?$' -and $_.CommandLine -match 'rgb_mjpeg_bridge\.py') -or
        ($_.Name -eq 'ssh.exe' -and $_.CommandLine -match '-R\s+(127\.0\.0\.1:)?808[12]:')
    } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

if (Get-Command wsl.exe -ErrorAction SilentlyContinue) {
    & wsl -d $WslDistro -- pkill -TERM -f ev_flicker_detector 2>$null
}

if ($DetachEventCamera -and (Get-Command usbipd.exe -ErrorAction SilentlyContinue)) {
    $cameraLine = @(& usbipd list) | Where-Object { $_ -match '1409:8e00' } | Select-Object -First 1
    if ($cameraLine) {
        $busId = (($cameraLine -split '\s+')[0]).Trim()
        & usbipd detach --busid $busId | Out-Host
    }
}

Write-Host "[Vollebak] Camera helpers stopped. Pi dashboard remains available." -ForegroundColor Green

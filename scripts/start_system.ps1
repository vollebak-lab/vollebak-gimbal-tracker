[CmdletBinding()]
param(
    [string]$PiHost = "192.168.0.3",
    [string]$PiUser = "vollebak",
    [string]$PiRepo = "/home/vollebak/vollebak-gimbal-tracker",
    [string]$WslDistro = "Ubuntu",
    [string]$EventCameraVidPid = "1409:8e00",
    [switch]$NoBrowser,
    [switch]$StartTracking
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $RepoRoot ".runtime"
$KeyPath = Join-Path $RepoRoot ".pi-ssh\vollebak_pi_ed25519"
$DashboardUrl = "http://${PiHost}:8080"
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

function Write-Step([string]$Message) {
    Write-Host "[Vollebak] $Message" -ForegroundColor Cyan
}

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found. Run SETUP_HANDOFF.cmd first."
    }
}

function Test-JsonEndpoint([string]$Uri, [int]$TimeoutSeconds = 3) {
    try {
        $null = Invoke-RestMethod -Uri $Uri -TimeoutSec $TimeoutSeconds
        return $true
    } catch {
        return $false
    }
}

function Wait-For([string]$Description, [scriptblock]$Probe, [int]$Seconds = 20) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if (& $Probe) { return }
        Start-Sleep -Milliseconds 500
    }
    throw "Timed out waiting for $Description."
}

function Stop-TrackedProcess([string]$Name) {
    $pidPath = Join-Path $RuntimeDir "$Name.pid"
    if (-not (Test-Path $pidPath)) { return }
    $savedPid = (Get-Content $pidPath -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($savedPid -match '^\d+$') {
        Stop-Process -Id ([int]$savedPid) -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
}

function Start-TrackedProcess(
    [string]$Name,
    [string]$FilePath,
    [string[]]$ArgumentList,
    [string]$WorkingDirectory = $RepoRoot
) {
    Stop-TrackedProcess $Name
    $process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath (Join-Path $RuntimeDir "$Name.pid") -Value $process.Id
    return $process
}

$SshArgs = @(
    "-i", $KeyPath,
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=5"
)
$PiTarget = "${PiUser}@${PiHost}"

function Invoke-Pi([string]$Command) {
    $output = & ssh @SshArgs $PiTarget $Command 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Pi command failed: $output"
    }
    return $output
}

function Test-PiEndpoint([int]$Port, [string]$Path) {
    # A missing endpoint is the normal signal to create its tunnel. Windows
    # PowerShell otherwise promotes ssh/curl stderr to a terminating error while
    # the script-wide ErrorActionPreference is Stop.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & ssh @SshArgs $PiTarget "curl -fsS --max-time 3 http://127.0.0.1:${Port}${Path} >/dev/null 2>&1" 2>$null
        return ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Start-ReverseTunnel([string]$Name, [int]$Port) {
    Write-Step "Starting reverse tunnel for port $Port"
    $arguments = @(
        "-N", "-T", "-i", $KeyPath,
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
        "-o", "ExitOnForwardFailure=yes",
        "-R", "127.0.0.1:${Port}:127.0.0.1:${Port}",
        $PiTarget
    )
    $null = Start-TrackedProcess $Name "ssh.exe" $arguments
}

Require-Command "ssh.exe"
Require-Command "wsl.exe"
Require-Command "usbipd.exe"
if (-not (Test-Path $KeyPath)) {
    throw "Pi SSH key not found at $KeyPath. Run SETUP_HANDOFF.cmd once."
}

Write-Step "Checking Raspberry Pi at $PiHost"
$null = Invoke-Pi "true"

$python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$rgbArgs = @(
    "scripts/rgb_mjpeg_bridge.py", "--source", "-1", "--preferred-name", "Brio",
    "--host", "127.0.0.1", "--port", "8082",
    "--width", "640", "--height", "480", "--fps", "30", "--quality", "60"
)
if (-not (Test-JsonEndpoint "http://127.0.0.1:8082/health")) {
    Write-Step "Starting Logitech MX Brio bridge"
    if (-not (Test-Path $python)) {
        throw "Windows virtual environment is missing. Run SETUP_HANDOFF.cmd first."
    }
    $null = Start-TrackedProcess "rgb-bridge" $python $rgbArgs
    Wait-For "Logitech bridge" { Test-JsonEndpoint "http://127.0.0.1:8082/health" } 15
} else {
    Write-Step "Logitech bridge already online"
}

$eventAvailable = Test-JsonEndpoint "http://127.0.0.1:8081/stats"
Write-Step "Starting continuous IMX636 hot-plug watchdog"
$null = Start-TrackedProcess "wsl-keepalive" "wsl.exe" @(
    "-d", $WslDistro, "--", "sleep", "infinity"
)
$null = Start-TrackedProcess "event-watchdog" "powershell.exe" @(
    "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $PSScriptRoot "event_camera_watchdog.ps1"),
    "-WslDistro", $WslDistro,
    "-EventCameraVidPid", $EventCameraVidPid
)

if (-not $eventAvailable) {
    $cameraConnected = @(& usbipd.exe list) |
        Where-Object { $_ -match [regex]::Escape($EventCameraVidPid) } |
        Select-Object -First 1
    if ($cameraConnected) {
        try {
            Wait-For "automatic IMX636 attachment and detector" {
                Test-JsonEndpoint "http://127.0.0.1:8081/stats"
            } 30
            $eventAvailable = $true
        } catch {
            Write-Warning "IMX636 is not online yet. The watchdog will keep retrying in the background."
        }
    } else {
        Write-Warning "IMX636 is unplugged. Connect it at any time; the watchdog will detect it automatically."
    }
}

# Resetting/attaching the SuperSpeed event camera can disturb another camera
# already open through DirectShow. Recover the Brio if its measured rate fell.
$rgbHealth = Invoke-RestMethod -Uri "http://127.0.0.1:8082/health" -TimeoutSec 5
$hasSourceName = $rgbHealth.PSObject.Properties.Name -contains "source_name"
$isLogitech = $hasSourceName -and ($rgbHealth.source_name -match 'Brio|Logitech')
if ([double]$rgbHealth.fps -lt 15.0 -or -not $isLogitech) {
    Write-Step "Restarting bridge on the named Logitech MX Brio"
    Get-NetTCPConnection -LocalPort 8082 -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique |
        ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
    Stop-TrackedProcess "rgb-bridge"
    Start-Sleep -Milliseconds 500
    $null = Start-TrackedProcess "rgb-bridge" $python $rgbArgs
    Wait-For "full-rate Logitech bridge" {
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:8082/health" -TimeoutSec 3
            $named = $health.PSObject.Properties.Name -contains "source_name"
            return (
                [double]$health.fps -ge 15.0 -and
                $named -and
                $health.source_name -match 'Brio|Logitech'
            )
        } catch {
            return $false
        }
    } 30
}

if (-not (Test-PiEndpoint 8082 "/health")) {
    Start-ReverseTunnel "rgb-tunnel" 8082
    Wait-For "Pi-to-Logitech tunnel" { Test-PiEndpoint 8082 "/health" } 15
} else {
    Write-Step "Logitech tunnel already online"
}

Start-ReverseTunnel "event-tunnel" 8081
if ($eventAvailable) {
    try {
        Wait-For "Pi-to-event-camera tunnel" { Test-PiEndpoint 8081 "/stats" } 15
    } catch {
        Write-Warning "The event-camera tunnel is starting; the Pi will reconnect automatically."
    }
}

if (-not (Test-JsonEndpoint "$DashboardUrl/api/state" 5)) {
    Write-Step "Starting Pi dashboard"
    $startCommand = "if ! ss -ltn | grep -q ':8080 '; then cd '$PiRepo' && setsid -f env PYTHONPATH=src .venv/bin/python -m vollebak_gimbal ui -c config/pi.yaml --host 0.0.0.0 --port 8080 > .dashboard.log 2>&1 < /dev/null; fi"
    $null = Invoke-Pi $startCommand
    Wait-For "Pi dashboard" { Test-JsonEndpoint "$DashboardUrl/api/state" 3 } 20
}

Write-Step "Applying safe startup state"
$offBody = @{ enabled = $false } | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post -Uri "$DashboardUrl/api/tracking" -ContentType "application/json" -Body $offBody -TimeoutSec 5 | Out-Null
Invoke-RestMethod -Method Post -Uri "$DashboardUrl/api/home" -ContentType "application/json" -Body "{}" -TimeoutSec 5 | Out-Null

if ($StartTracking) {
    Write-Warning "Tracking was explicitly requested. Keep the gimbal area clear."
    $onBody = @{ enabled = $true } | ConvertTo-Json -Compress
    Invoke-RestMethod -Method Post -Uri "$DashboardUrl/api/tracking" -ContentType "application/json" -Body $onBody -TimeoutSec 5 | Out-Null
}

Wait-For "settled Pi camera telemetry" {
    try {
        $readyState = Invoke-RestMethod -Uri "$DashboardUrl/api/state" -TimeoutSec 3
        $rgbReady = $readyState.camera_mode -eq "live" -and [double]$readyState.fps -ge 15.0
        $eventReady = (-not $eventAvailable) -or [bool]$readyState.event_camera.connected
        return ($rgbReady -and $eventReady)
    } catch {
        return $false
    }
} 20

$state = Invoke-RestMethod -Uri "$DashboardUrl/api/state" -TimeoutSec 5
Write-Host ""
Write-Host "System ready" -ForegroundColor Green
Write-Host "  Dashboard: $DashboardUrl"
Write-Host "  Logitech:  $($state.camera) / $($state.camera_mode) / $([math]::Round($state.fps, 1)) FPS"
Write-Host "  IMX636:    $($state.event_camera.status) / $($state.event_camera.event_rate_mev_s) MEv/s"
Write-Host "  Gimbal:    $($state.driver) / pan $($state.pan) / tilt $($state.tilt)"
Write-Host "  Tracking:  $($state.tracking_enabled)"

if (-not $NoBrowser) {
    Start-Process $DashboardUrl
}

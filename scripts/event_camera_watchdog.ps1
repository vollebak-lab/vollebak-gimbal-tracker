[CmdletBinding()]
param(
    [string]$WslDistro = "Ubuntu",
    [string]$EventCameraVidPid = "1409:8e00",
    [int]$PollSeconds = 3,
    [int]$DetectorStartupSeconds = 20,
    [int]$RestartBackoffSeconds = 10
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $RepoRoot ".runtime"
$StatusPath = Join-Path $RuntimeDir "event-camera-status.json"
$DetectorPidPath = Join-Path $RuntimeDir "event-detector.pid"
$LogPath = Join-Path $RuntimeDir "event-camera-watchdog.log"
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

function Convert-ToWslPath([string]$WindowsPath) {
    $resolved = (Resolve-Path $WindowsPath).Path
    if ($resolved -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "Cannot convert path to WSL format: $resolved"
    }
    $drive = $Matches[1].ToLowerInvariant()
    $tail = $Matches[2] -replace '\\', '/'
    return "/mnt/$drive/$tail"
}

function Write-WatchdogLog([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Add-Content -LiteralPath $LogPath -Value $line
}

$script:lastStatusSignature = ""
function Set-CameraStatus(
    [string]$Status,
    [bool]$CameraPresent,
    [string]$UsbState,
    [string]$BusId,
    [bool]$ServiceOnline,
    [string]$Detail = ""
) {
    $signature = "$Status|$CameraPresent|$UsbState|$BusId|$ServiceOnline|$Detail"
    if ($signature -eq $script:lastStatusSignature) { return }

    $payload = [ordered]@{
        status = $Status
        camera_present = $CameraPresent
        usb_state = $UsbState
        bus_id = $BusId
        service_online = $ServiceOnline
        detail = $Detail
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
    }
    $payload | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
    Write-WatchdogLog "$Status - $Detail"
    $script:lastStatusSignature = $signature
}

function Test-EventService {
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:8081/stats" -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}

function Get-EventCameraLine {
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $lines = @(& usbipd.exe list 2>$null)
        if ($LASTEXITCODE -ne 0) { return $null }
        return $lines |
            Where-Object { $_ -match [regex]::Escape($EventCameraVidPid) } |
            Select-Object -First 1
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Invoke-Usbipd([string[]]$Arguments, [int]$TimeoutSeconds = 15) {
    $process = Start-Process -FilePath "usbipd.exe" -ArgumentList $Arguments `
        -WindowStyle Hidden -PassThru
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        return $false
    }
    return ($process.ExitCode -eq 0)
}

function Stop-EventDetector {
    if (Test-Path $DetectorPidPath) {
        $savedProcessId = Get-Content $DetectorPidPath -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($savedProcessId -match '^\d+$') {
            Stop-Process -Id ([int]$savedProcessId) -Force -ErrorAction SilentlyContinue
        }
        Remove-Item -LiteralPath $DetectorPidPath -Force -ErrorAction SilentlyContinue
    }

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & wsl.exe -d $WslDistro -- pkill -TERM -f ev_flicker_detector 2>$null
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}

$WslRepo = Convert-ToWslPath $RepoRoot
$DetectorScript = "$WslRepo/scripts/run_event_detector_wsl.sh"
function Start-EventDetector {
    Stop-EventDetector
    $process = Start-Process -FilePath "wsl.exe" -ArgumentList @(
        "-d", $WslDistro, "--", "bash", $DetectorScript
    ) -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath $DetectorPidPath -Value $process.Id
}

if (-not (Get-Command usbipd.exe -ErrorAction SilentlyContinue)) {
    throw "usbipd.exe was not found. Run SETUP_HANDOFF.cmd first."
}
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "wsl.exe was not found. Enable WSL and run SETUP_HANDOFF.cmd first."
}

$lastDevicePresent = $null
$nextRestartAt = Get-Date
$cameraLine = $null
Set-CameraStatus "starting" $false "unknown" "" $false "Scanning for IMX636 ($EventCameraVidPid)."

while ($true) {
    try {
        $cameraLine = Get-EventCameraLine
        if (-not $cameraLine) {
            if ($null -eq $lastDevicePresent -or $lastDevicePresent) {
                Stop-EventDetector
            }
            $lastDevicePresent = $false
            Set-CameraStatus "waiting_for_camera" $false "disconnected" "" $false `
                "Connect the IDS UE-39B0XCP / Sony IMX636 camera; detection is automatic."
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        $lastDevicePresent = $true
        $busId = (($cameraLine -split '\s+')[0]).Trim()
        if ($cameraLine -match 'Not shared') {
            Set-CameraStatus "setup_required" $true "not_shared" $busId $false `
                "USB sharing needs one Administrator setup: run SETUP_HANDOFF.cmd."
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        if ($cameraLine -notmatch '\bAttached\b') {
            Set-CameraStatus "attaching" $true "shared" $busId $false `
                "Attaching the event camera to WSL."
            if (-not (Invoke-Usbipd @("attach", "--wsl", "--busid", $busId))) {
                Set-CameraStatus "attach_failed" $true "shared" $busId $false `
                    "usbipd could not attach the camera; retrying automatically."
                Start-Sleep -Seconds $PollSeconds
                continue
            }
            Start-Sleep -Seconds 1
            $cameraLine = Get-EventCameraLine
            if (-not $cameraLine -or $cameraLine -notmatch '\bAttached\b') {
                Set-CameraStatus "attach_failed" $true "shared" $busId $false `
                    "The camera did not become attached; retrying automatically."
                Start-Sleep -Seconds $PollSeconds
                continue
            }
        }

        if (Test-EventService) {
            Set-CameraStatus "online" $true "attached" $busId $true `
                "Bart's IMX636 detector is serving live data on port 8081."
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        if ((Get-Date) -lt $nextRestartAt) {
            Set-CameraStatus "detector_retry" $true "attached" $busId $false `
                "Waiting briefly before restarting Bart's detector."
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        Set-CameraStatus "starting_detector" $true "attached" $busId $false `
            "Starting Bart's IMX636 detector in WSL."
        Start-EventDetector

        $deadline = (Get-Date).AddSeconds($DetectorStartupSeconds)
        while ((Get-Date) -lt $deadline -and -not (Test-EventService)) {
            Start-Sleep -Milliseconds 500
        }
        if (Test-EventService) {
            Set-CameraStatus "online" $true "attached" $busId $true `
                "Bart's IMX636 detector is serving live data on port 8081."
        } else {
            $nextRestartAt = (Get-Date).AddSeconds($RestartBackoffSeconds)
            Set-CameraStatus "detector_failed" $true "attached" $busId $false `
                "The detector did not become healthy; it will restart automatically."
        }
    } catch {
        $nextRestartAt = (Get-Date).AddSeconds($RestartBackoffSeconds)
        Set-CameraStatus "error" ($null -ne $cameraLine) "unknown" "" $false $_.Exception.Message
    }

    Start-Sleep -Seconds $PollSeconds
}

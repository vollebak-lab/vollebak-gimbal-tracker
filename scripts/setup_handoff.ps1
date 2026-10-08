[CmdletBinding()]
param(
    [string]$PiHost = "192.168.0.3",
    [string]$PiUser = "vollebak",
    [string]$PiRepo = "/home/vollebak/vollebak-gimbal-tracker",
    [string]$WslDistro = "Ubuntu"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$KeyDir = Join-Path $RepoRoot ".pi-ssh"
$KeyPath = Join-Path $KeyDir "vollebak_pi_ed25519"
$PiTarget = "${PiUser}@${PiHost}"

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' is not installed. See docs/HANDOFF.md."
    }
}

foreach ($command in @("ssh.exe", "scp.exe", "ssh-keygen.exe", "wsl.exe", "usbipd.exe")) {
    Require-Command $command
}

$WindowsPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $WindowsPython)) {
    Require-Command "py.exe"
    Write-Host "Creating the Windows Python environment..." -ForegroundColor Cyan
    & py.exe -3.11 -m venv (Join-Path $RepoRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Windows Python environment." }
    & $WindowsPython -m pip install -e "${RepoRoot}[vision]"
    if ($LASTEXITCODE -ne 0) { throw "Could not install the Windows camera dependencies." }
}

New-Item -ItemType Directory -Force -Path $KeyDir | Out-Null
if (-not (Test-Path $KeyPath)) {
    Write-Host "Generating a dedicated Pi SSH key..." -ForegroundColor Cyan
    & ssh-keygen -q -t ed25519 -N "" -f $KeyPath
}

Write-Host "Installing the SSH key on the Pi. Enter the Pi password when prompted." -ForegroundColor Cyan
Get-Content "$KeyPath.pub" | & ssh -o StrictHostKeyChecking=accept-new $PiTarget `
    'umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; key=$(cat); grep -qxF "$key" ~/.ssh/authorized_keys || printf "%s\n" "$key" >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys'
if ($LASTEXITCODE -ne 0) { throw "Could not install the Pi SSH key." }

$cameraLine = @(& usbipd list) | Where-Object { $_ -match '1409:8e00' } | Select-Object -First 1
if ($cameraLine -and $cameraLine -match 'Not shared') {
    $busId = (($cameraLine -split '\s+')[0]).Trim()
    Write-Host "Windows will request administrator approval to share the IMX636 once." -ForegroundColor Cyan
    $bind = Start-Process -FilePath "usbipd.exe" -Verb RunAs -Wait -PassThru `
        -ArgumentList @("bind", "--busid", $busId)
    if ($bind.ExitCode -ne 0) {
        Write-Warning "The IMX636 could not be shared. RGB still works; retry setup as Administrator."
    }
} elseif (-not $cameraLine) {
    Write-Warning "The IMX636 is not connected. Connect it and rerun setup to authorize USB/IP."
}

Write-Host "Deploying the safe handoff configuration..." -ForegroundColor Cyan
& scp -i $KeyPath (Join-Path $RepoRoot "config\pi.handoff.yaml") "${PiTarget}:${PiRepo}/config/pi.yaml"
if ($LASTEXITCODE -ne 0) { throw "Could not copy the Pi configuration." }

& scp -i $KeyPath (Join-Path $RepoRoot "scripts\vollebak-gimbal.service") "${PiTarget}:/tmp/vollebak-gimbal.service"
if ($LASTEXITCODE -ne 0) { throw "Could not copy the Pi service definition." }

Write-Host "Installing the Pi boot service. Enter the Pi password for sudo when prompted." -ForegroundColor Cyan
& ssh -t -i $KeyPath $PiTarget `
    "sudo systemctl stop vollebak-gimbal.service 2>/dev/null || true; pkill -f '[v]ollebak_gimbal ui' 2>/dev/null || true; sudo install -m 0644 /tmp/vollebak-gimbal.service /etc/systemd/system/vollebak-gimbal.service && sudo systemctl daemon-reload && sudo systemctl enable --now vollebak-gimbal.service"
if ($LASTEXITCODE -ne 0) { throw "Could not install the Pi system service." }

$wslRepo = "/mnt/" + $RepoRoot.Substring(0,1).ToLowerInvariant() + "/" + ($RepoRoot.Substring(3) -replace '\\','/')
& wsl -d $WslDistro -- bash -lc 'test -x "$HOME/predator-event-build/ev_flicker_detector" && test -d "$HOME/metavision-5.2-ids"'
if ($LASTEXITCODE -ne 0) {
    Write-Warning "The WSL OpenEB/detector build is not installed for this Windows user. RGB will work; follow the event-camera section in docs/HANDOFF.md."
}

Write-Host "Setup complete. Double-click RUN_SYSTEM.cmd for normal operation." -ForegroundColor Green
Write-Host "Repository path visible to WSL: $wslRepo"

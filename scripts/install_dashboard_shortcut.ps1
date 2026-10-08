[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $repoRoot "RUN_SYSTEM.cmd"
if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Launcher not found: $launcher"
}

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "Vollebak Gimbal Dashboard.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "$env:SystemRoot\System32\cmd.exe"
$shortcut.Arguments = "/c `"`"$launcher`"`""
$shortcut.WorkingDirectory = $repoRoot
$shortcut.Description = "Start the Vollebak camera, event detector, Pi gimbal dashboard, and open the operator UI"
$shortcut.IconLocation = "$env:SystemRoot\System32\imageres.dll,109"
$shortcut.WindowStyle = 1
$shortcut.Save()

Write-Host "Installed: $shortcutPath" -ForegroundColor Green
Write-Host "The launcher starts with tracking paused and then opens the dashboard."

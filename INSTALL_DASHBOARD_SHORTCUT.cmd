@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_dashboard_shortcut.ps1"
if errorlevel 1 (
  echo.
  echo Shortcut installation failed. Press any key.
  pause >nul
  exit /b 1
)
echo.
echo You can now use the Vollebak Gimbal Dashboard shortcut on your desktop.
timeout /t 5 >nul

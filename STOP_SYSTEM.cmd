@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop_system.ps1"
if errorlevel 1 (
  echo.
  echo Shutdown completed with warnings. Press any key.
  pause >nul
  exit /b 1
)
echo.
echo Camera helpers stopped and gimbal parked.
timeout /t 5 >nul

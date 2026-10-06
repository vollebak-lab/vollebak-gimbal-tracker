@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_system.ps1"
if errorlevel 1 (
  echo.
  echo Startup failed. Review the message above, then press any key.
  pause >nul
  exit /b 1
)
echo.
echo System is ready. This window can be closed.
timeout /t 5 >nul

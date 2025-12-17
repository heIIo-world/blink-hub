@echo off
title Blink Downloader v3.1.0-beta.3 - Service Status

echo ==========================================
echo   Blink Hub v3.1.0-beta.3
echo   Service Status
echo ==========================================
echo.

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

if not exist "%SCRIPT_DIR%\nssm.exe" (
    echo Service is NOT installed (NSSM not found)
    echo.
    echo Run install-service.bat to install as a service
    echo Or run start.bat to run manually
    pause
    exit /b 0
)

echo Service Status:
"%SCRIPT_DIR%\nssm.exe" status BlinkHub
echo.

"%SCRIPT_DIR%\nssm.exe" status BlinkHub | findstr /i "running" >nul
if %errorlevel% equ 0 (
    echo The service is RUNNING
    echo Web interface: http://localhost:8080
) else (
    "%SCRIPT_DIR%\nssm.exe" status BlinkHub | findstr /i "stopped" >nul
    if %errorlevel% equ 0 (
        echo The service is STOPPED
        echo.
        echo To start: Run services.msc, find "Blink Hub", right-click Start
        echo Or run: nssm start BlinkHub
    ) else (
        echo The service may not be installed
        echo Run install-service.bat to install
    )
)

echo.
echo ==========================================
echo   Quick Commands
echo ==========================================
echo.
echo   Start service:    nssm start BlinkHub
echo   Stop service:     nssm stop BlinkHub  
echo   Restart service:  nssm restart BlinkHub
echo   View logs:        type "%SCRIPT_DIR%\logs\service.log"
echo.

pause

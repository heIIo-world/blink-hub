@echo off
title Blink Downloader v3.1.0-beta.3 - Service Uninstaller

echo ==========================================
echo   Blink Hub v3.1.0-beta.3
echo   Service Remover
echo ==========================================
echo.

:: Check for admin rights
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: This script requires administrator privileges.
    echo Please right-click and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: Check if NSSM exists
if not exist "%SCRIPT_DIR%\nssm.exe" (
    echo NSSM not found. Service may not be installed.
    pause
    exit /b 1
)

:: Check if service exists
"%SCRIPT_DIR%\nssm.exe" status BlinkHub >nul 2>&1
if %errorlevel% neq 0 (
    echo Service "BlinkHub" is not installed.
    pause
    exit /b 0
)

echo Stopping service...
"%SCRIPT_DIR%\nssm.exe" stop BlinkHub >nul 2>&1
timeout /t 3 >nul

echo Removing service...
"%SCRIPT_DIR%\nssm.exe" remove BlinkHub confirm

echo.
echo ==========================================
echo   Service removed successfully
echo ==========================================
echo.
echo The Blink Downloader service has been removed.
echo Your settings, credentials, and downloaded videos are preserved.
echo.
echo You can still run the app manually with start.bat
echo Or reinstall the service with install-service.bat
echo.

pause

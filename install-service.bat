@echo off
title Blink Downloader - Service Installer v3.1.0-beta.3
setlocal enabledelayedexpansion

echo ==========================================
echo   Blink Hub v3.1.0-beta.3
echo   Service Installer
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

:: Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

:: Get Python path
for /f "tokens=*" %%i in ('where python') do (
    set "PYTHON_PATH=%%i"
    goto :found_python
)
:found_python
echo Found Python: %PYTHON_PATH%

:: Check if virtual environment exists, create if not
if not exist "%SCRIPT_DIR%\venv" (
    echo Creating virtual environment...
    python -m venv "%SCRIPT_DIR%\venv"
)

:: Install dependencies
echo Installing dependencies...
call "%SCRIPT_DIR%\venv\Scripts\pip.exe" install -r "%SCRIPT_DIR%\requirements.txt" -q

:: Check if NSSM exists, download if not
if not exist "%SCRIPT_DIR%\nssm.exe" (
    echo.
    echo Downloading NSSM (service manager)...
    
    :: Try PowerShell download
    powershell -Command "& {Invoke-WebRequest -Uri 'https://nssm.cc/release/nssm-2.24.zip' -OutFile '%SCRIPT_DIR%\nssm.zip'}" 2>nul
    
    if exist "%SCRIPT_DIR%\nssm.zip" (
        echo Extracting NSSM...
        powershell -Command "& {Expand-Archive -Path '%SCRIPT_DIR%\nssm.zip' -DestinationPath '%SCRIPT_DIR%\nssm_temp' -Force}"
        
        :: Copy the correct architecture
        if exist "%ProgramFiles(x86)%" (
            copy "%SCRIPT_DIR%\nssm_temp\nssm-2.24\win64\nssm.exe" "%SCRIPT_DIR%\nssm.exe" >nul
        ) else (
            copy "%SCRIPT_DIR%\nssm_temp\nssm-2.24\win32\nssm.exe" "%SCRIPT_DIR%\nssm.exe" >nul
        )
        
        :: Cleanup
        rmdir /s /q "%SCRIPT_DIR%\nssm_temp" 2>nul
        del "%SCRIPT_DIR%\nssm.zip" 2>nul
        
        echo NSSM downloaded successfully.
    ) else (
        echo.
        echo ERROR: Could not download NSSM automatically.
        echo Please download NSSM manually from https://nssm.cc/download
        echo Extract nssm.exe to: %SCRIPT_DIR%
        echo Then run this script again.
        pause
        exit /b 1
    )
)

:: Check if service already exists
"%SCRIPT_DIR%\nssm.exe" status BlinkHub >nul 2>&1
if %errorlevel% equ 0 (
    echo.
    echo Service already exists. Stopping and removing old service...
    "%SCRIPT_DIR%\nssm.exe" stop BlinkHub >nul 2>&1
    timeout /t 2 >nul
    "%SCRIPT_DIR%\nssm.exe" remove BlinkHub confirm >nul 2>&1
    timeout /t 2 >nul
)

:: Create the service runner script
echo Creating service runner...
(
echo import subprocess
echo import sys
echo import os
echo.
echo os.chdir^(r"%SCRIPT_DIR%"^)
echo subprocess.run^([
echo     r"%SCRIPT_DIR%\venv\Scripts\python.exe",
echo     "-m", "uvicorn",
echo     "app:app",
echo     "--host", "127.0.0.1",
echo     "--port", "8080"
echo ]^)
) > "%SCRIPT_DIR%\service_runner.py"

:: Install the service
echo.
echo Installing Blink Downloader service...
"%SCRIPT_DIR%\nssm.exe" install BlinkHub "%SCRIPT_DIR%\venv\Scripts\python.exe" "%SCRIPT_DIR%\service_runner.py"

:: Configure the service
"%SCRIPT_DIR%\nssm.exe" set BlinkHub AppDirectory "%SCRIPT_DIR%"
"%SCRIPT_DIR%\nssm.exe" set BlinkHub DisplayName "Blink Hub"
"%SCRIPT_DIR%\nssm.exe" set BlinkHub Description "Automatically downloads videos from Blink cameras"
"%SCRIPT_DIR%\nssm.exe" set BlinkHub Start SERVICE_AUTO_START
"%SCRIPT_DIR%\nssm.exe" set BlinkHub AppStdout "%SCRIPT_DIR%\logs\service.log"
"%SCRIPT_DIR%\nssm.exe" set BlinkHub AppStderr "%SCRIPT_DIR%\logs\service.log"
"%SCRIPT_DIR%\nssm.exe" set BlinkHub AppRotateFiles 1
"%SCRIPT_DIR%\nssm.exe" set BlinkHub AppRotateBytes 1048576

:: Create logs directory
if not exist "%SCRIPT_DIR%\logs" mkdir "%SCRIPT_DIR%\logs"

:: Start the service
echo Starting service...
"%SCRIPT_DIR%\nssm.exe" start BlinkHub

timeout /t 3 >nul

:: Check if service is running
"%SCRIPT_DIR%\nssm.exe" status BlinkHub | findstr /i "running" >nul
if %errorlevel% equ 0 (
    echo.
    echo ==========================================
    echo   SUCCESS! Service installed and running
    echo ==========================================
    echo.
    echo The Blink Downloader is now running as a Windows service.
    echo.
    echo   Web Interface: http://localhost:8080
    echo   Service Name:  BlinkHub
    echo   Logs:          %SCRIPT_DIR%\logs\service.log
    echo.
    echo The service will:
    echo   - Start automatically when Windows boots
    echo   - Run scheduled downloads in the background
    echo   - Keep running even when you log out
    echo.
    echo To manage the service:
    echo   - Open Services (services.msc)
    echo   - Find "Blink Hub"
    echo   - Right-click to Start/Stop/Restart
    echo.
    echo Or use: uninstall-service.bat to remove it
    echo.
) else (
    echo.
    echo WARNING: Service installed but may not be running.
    echo Check the logs at: %SCRIPT_DIR%\logs\service.log
    echo.
)

pause

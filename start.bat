@echo off
title Blink Hub v3.1.0
cd /d "%~dp0"

echo ========================================
echo    Blink Hub
echo    Version: 3.1.0
echo    Build: 2025-12-12
echo ========================================
echo.

:: Kill any existing instances first
echo Checking for existing instances...

:: Stop Windows service if running
net stop BlinkHub >nul 2>&1

:: Kill pythonw (hidden instances)
taskkill /F /IM "pythonw.exe" >nul 2>&1

:: Kill python running uvicorn (check by window title won't work, so check port)
:: Check if port 8080 is in use and kill that process
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8080.*LISTENING"') do (
    echo Stopping process on port 8080 (PID: %%a)
    taskkill /F /PID %%a >nul 2>&1
)

:: Also try killing any uvicorn processes
wmic process where "commandline like '%%uvicorn%%app:app%%'" delete >nul 2>&1

:: Wait for port to be released
echo Waiting for port to be released...
timeout /t 2 /nobreak >nul

:: Double-check and retry if still in use
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8080.*LISTENING"') do (
    echo Port still in use, force killing PID %%a...
    taskkill /F /PID %%a >nul 2>&1
    timeout /t 1 /nobreak >nul
)
echo.

:: Try to find a working Python installation
set PYTHON_CMD=

:: First try 'py' launcher (most reliable on Windows)
py --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py
    goto :found_python
)

:: Try python3
python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python3
    goto :found_python
)

:: Try python and verify it actually works
python -c "print('ok')" >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    goto :found_python
)

:: No working Python found
echo ERROR: Python is not installed or not working properly.
echo.
echo Please install Python 3.9+ from https://python.org
echo Make sure to check "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:found_python
echo Found Python: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

:: Check if virtual environment exists and is valid
if exist "venv\Scripts\python.exe" (
    echo Using existing virtual environment...
    goto :activate_venv
)

:: Create virtual environment
echo Creating virtual environment...
%PYTHON_CMD% -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo.

:activate_venv
:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install/update dependencies
echo Checking dependencies...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo.
    echo Installing dependencies with verbose output...
    pip install -r requirements.txt
)
echo.

:: Create folders if they don't exist
if not exist "downloads" mkdir downloads
if not exist "logs" mkdir logs
if not exist "thumbnails" mkdir thumbnails

echo Starting server...
echo.
echo ==========================================
echo  Open your browser to: http://localhost:8080
echo  Press Ctrl+C to stop the server
echo ==========================================
echo.

:: Run the server
python -m uvicorn app:app --host 127.0.0.1 --port 8080

pause

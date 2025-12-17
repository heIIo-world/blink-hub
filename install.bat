@echo off
setlocal enabledelayedexpansion

echo.
echo  ============================================
echo   Blink Video Downloader - Installer
echo   v3.1.0-beta.20
echo  ============================================
echo.

REM Get the directory where this script is located
set "INSTALL_DIR=%~dp0"
cd /d "%INSTALL_DIR%"

echo  Install Location: %INSTALL_DIR%
echo.

REM ============================================
REM Step 1: Check for Python
REM ============================================
echo [1/5] Checking for Python...

python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python is not installed or not in PATH.
    echo.
    echo  Please install Python 3.10 or later:
    echo    1. Go to https://www.python.org/downloads/
    echo    2. Download Python 3.12 or later
    echo    3. IMPORTANT: Check "Add Python to PATH" during install!
    echo    4. Run this installer again
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo  Found Python %PYTHON_VERSION%

REM ============================================
REM Step 2: Create Virtual Environment
REM ============================================
echo.
echo [2/5] Setting up virtual environment...

if not exist "venv" (
    echo  Creating new virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo  ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo  Virtual environment already exists
)

REM ============================================
REM Step 3: Install Python Dependencies
REM ============================================
echo.
echo [3/5] Installing Python dependencies...

call venv\Scripts\activate.bat

echo  Installing packages from requirements.txt...
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

if errorlevel 1 (
    echo  ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo  Dependencies installed successfully

REM ============================================
REM Step 4: Check/Install ffmpeg
REM ============================================
echo.
echo [4/5] Checking for ffmpeg...

where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo  ffmpeg not found in PATH
    
    REM Check if we have a local copy
    if exist "%INSTALL_DIR%ffmpeg\bin\ffmpeg.exe" (
        echo  Found local ffmpeg installation
        set "PATH=%INSTALL_DIR%ffmpeg\bin;%PATH%"
    ) else (
        echo.
        echo  Downloading ffmpeg...
        
        REM Create ffmpeg directory
        if not exist "ffmpeg" mkdir ffmpeg
        
        REM Download ffmpeg using PowerShell
        powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile 'ffmpeg-temp.zip'}"
        
        if exist "ffmpeg-temp.zip" (
            echo  Extracting ffmpeg...
            powershell -Command "Expand-Archive -Path 'ffmpeg-temp.zip' -DestinationPath 'ffmpeg-extract' -Force"
            
            REM Move contents from nested folder
            for /d %%D in (ffmpeg-extract\ffmpeg-*) do (
                xcopy /E /Y "%%D\bin" "ffmpeg\bin\" >nul
            )
            
            REM Cleanup
            del ffmpeg-temp.zip
            rmdir /s /q ffmpeg-extract
            
            echo  ffmpeg installed to %INSTALL_DIR%ffmpeg\bin
        ) else (
            echo.
            echo  WARNING: Could not download ffmpeg automatically.
            echo  Video thumbnails will not work without ffmpeg.
            echo.
            echo  To install manually:
            echo    1. Download from https://www.gyan.dev/ffmpeg/builds/
            echo    2. Extract ffmpeg.exe to %INSTALL_DIR%ffmpeg\bin\
            echo.
        )
    )
) else (
    echo  ffmpeg is already installed
)

REM ============================================
REM Step 5: Create Directories and Shortcuts
REM ============================================
echo.
echo [5/5] Creating directories and shortcuts...

if not exist "downloads" mkdir downloads
if not exist "logs" mkdir logs
if not exist "thumbnails" mkdir thumbnails

REM Create a run script
echo @echo off > run.bat
echo cd /d "%%~dp0" >> run.bat
echo call venv\Scripts\activate.bat >> run.bat
echo set "PATH=%%~dp0ffmpeg\bin;%%PATH%%" >> run.bat
echo python app.py >> run.bat

REM Create a desktop shortcut using PowerShell
echo.
set /p CREATE_SHORTCUT="Create desktop shortcut? (Y/N): "
if /i "%CREATE_SHORTCUT%"=="Y" (
    powershell -Command "& {$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\Blink Video Downloader.lnk'); $Shortcut.TargetPath = '%INSTALL_DIR%run.bat'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.IconLocation = 'shell32.dll,12'; $Shortcut.Save()}"
    echo  Desktop shortcut created!
)

REM ============================================
REM Done!
REM ============================================
echo.
echo  ============================================
echo   Installation Complete!
echo  ============================================
echo.
echo  To start Blink Video Downloader:
echo    - Double-click run.bat
echo    - Or use the desktop shortcut
echo.
echo  The app will open in your web browser at:
echo    http://localhost:5000
echo.
echo  First time setup:
echo    1. Log in with your Blink account
echo    2. Complete 2FA verification
echo    3. Start downloading your videos!
echo.

set /p START_NOW="Start Blink Video Downloader now? (Y/N): "
if /i "%START_NOW%"=="Y" (
    echo.
    echo  Starting application...
    call run.bat
)

pause

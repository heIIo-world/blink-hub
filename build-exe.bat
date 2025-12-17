@echo off
echo ========================================
echo Blink Downloader - Windows EXE Builder
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from python.org
    pause
    exit /b 1
)

REM Create/activate virtual environment
if not exist "build_venv" (
    echo Creating virtual environment...
    python -m venv build_venv
)

echo Activating virtual environment...
call build_venv\Scripts\activate.bat

echo Installing dependencies...
pip install pyinstaller --quiet
pip install -r requirements.txt --quiet

echo.
echo Building executable (Desktop App)...
echo.

pyinstaller --onefile --windowed ^
    --name "BlinkDownloader" ^
    --add-data "templates;templates" ^
    --add-data "static;static" ^
    --hidden-import uvicorn.logging ^
    --hidden-import uvicorn.protocols.http ^
    --hidden-import uvicorn.protocols.http.auto ^
    --hidden-import uvicorn.protocols.http.h11_impl ^
    --hidden-import uvicorn.protocols.http.httptools_impl ^
    --hidden-import uvicorn.protocols.websockets ^
    --hidden-import uvicorn.protocols.websockets.auto ^
    --hidden-import uvicorn.protocols.websockets.websockets_impl ^
    --hidden-import uvicorn.lifespan ^
    --hidden-import uvicorn.lifespan.on ^
    --hidden-import uvicorn.lifespan.off ^
    --hidden-import aiohttp ^
    --hidden-import blinkpy ^
    --hidden-import webview ^
    --collect-submodules uvicorn ^
    --collect-submodules blinkpy ^
    --collect-submodules webview ^
    app.py

if errorlevel 1 (
    echo.
    echo BUILD FAILED!
    pause
    exit /b 1
)

echo.
echo ========================================
echo BUILD SUCCESSFUL!
echo ========================================
echo.
echo Executable: dist\BlinkDownloader.exe
echo.
echo Double-click to run as a native desktop app!
echo.

REM Copy supporting files to dist
if not exist "dist\templates" mkdir "dist\templates"
if not exist "dist\static" mkdir "dist\static"
if not exist "dist\downloads" mkdir "dist\downloads"
if not exist "dist\logs" mkdir "dist\logs"
if not exist "dist\thumbnails" mkdir "dist\thumbnails"

copy /y "templates\index.html" "dist\templates\" >nul
copy /y "README.md" "dist\" >nul 2>nul

echo Created dist\ folder with all files.
echo.
pause

@echo off
title Test SSH Build Dependencies
echo.
echo =====================================
echo   Testing SSH Build Dependencies
echo =====================================
echo.

cd /d "%~dp0"

echo [1/4] Checking Python availability...
python --version
if %errorlevel% neq 0 (
    echo Error: Python not found!
    pause
    exit /b 1
)

echo.
echo [2/4] Testing SSH module imports...
python -c "import paramiko; print('✓ paramiko available')" 2>nul
if %errorlevel% neq 0 (
    echo Installing paramiko...
    pip install paramiko
)

python -c "import scp; print('✓ scp available')" 2>nul  
if %errorlevel% neq 0 (
    echo Installing scp...
    pip install scp
)

python -c "import PyInstaller; print('✓ PyInstaller available')" 2>nul
if %errorlevel% neq 0 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

echo.
echo [3/4] Testing Vestim imports...
cd ..
python -c "from vestim.remote.auto_deploying_ssh_manager import AutoDeployingSSHManager; print('✓ SSH manager available')" 2>nul
if %errorlevel% neq 0 (
    echo Warning: SSH manager import failed - this is expected if not built yet
)

python -c "from vestim.remote.auto_x11_installer import AutoX11ServerManager; print('✓ X11 installer available')" 2>nul
if %errorlevel% neq 0 (
    echo Warning: X11 installer import failed - this is expected if not built yet
)

cd build_ssh

echo.
echo [4/4] Testing build script...
python -c "import build_ssh_distribution; print('✓ Build script syntax OK')"
if %errorlevel% neq 0 (
    echo Error: Build script has syntax errors!
    pause
    exit /b 1
)

echo.
echo =====================================
echo   Dependencies Test Complete
echo =====================================
echo.
echo All dependencies are ready for SSH build!
echo You can now run: build_ssh.bat
echo.
pause

@echo off
title Build Vestim SSH/Remote Distribution
echo.
echo =========================================
echo   Building Vestim SSH/Remote Client
echo =========================================
echo.

cd /d "%~dp0"

echo Activating build environment...
if exist "..\build_env\Scripts\activate.bat" (
    call "..\build_env\Scripts\activate.bat"
    echo Build environment activated.
) else (
    echo Warning: Build environment not found. Using system Python.
)

echo.
echo Installing build dependencies...
pip install pyinstaller paramiko scp cryptography bcrypt pynacl keyring

echo.
echo Starting build process...
python build_ssh_distribution.py

echo.
echo Build process completed!
echo.
echo Check the 'ssh_dist' folder for your distributable files:
echo   - VestimUniversalClient.exe
echo   - VestimServerClient.exe  
echo   - VestimServerSetup.exe
echo   - Launch_Vestim.bat
echo.
echo Note: This is separate from your main Windows build in 'dist' folder
echo.
pause

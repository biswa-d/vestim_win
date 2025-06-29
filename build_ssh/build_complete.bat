@echo off
title Complete Vestim SSH Build Process
echo.
echo =========================================
echo   Complete Vestim SSH Build Process
echo =========================================
echo.

cd /d "%~dp0"

REM Step 1: Build executables
echo [1/3] Building executables...
call build_ssh.bat

if %errorlevel% neq 0 (
    echo Error: Build failed!
    pause
    exit /b 1
)

REM Step 2: Create installer (if Inno Setup is available)
echo.
echo [2/3] Creating installer...
where iscc >nul 2>nul
if %errorlevel% equ 0 (
    echo Inno Setup found. Creating installer...
    iscc vestim_ssh_installer.iss
    if %errorlevel% equ 0 (
        echo Installer created successfully!
    ) else (
        echo Warning: Installer creation failed.
    )
) else (
    echo Inno Setup not found. Skipping installer creation.
    echo You can install Inno Setup from: https://jrsoftware.org/isinfo.php
)

REM Step 3: Create portable ZIP package
echo.
echo [3/3] Creating portable package...
if exist "VestimSSH_Portable.zip" del "VestimSSH_Portable.zip"

cd ssh_dist
if exist "..\..\..\Program Files\7-Zip\7z.exe" (
    "..\..\..\Program Files\7-Zip\7z.exe" a -tzip "..\VestimSSH_Portable.zip" *
) else if exist "%ProgramFiles%\7-Zip\7z.exe" (
    "%ProgramFiles%\7-Zip\7z.exe" a -tzip "..\VestimSSH_Portable.zip" *
) else (
    echo 7-Zip not found. Creating portable folder instead...
    cd ..
    if exist "VestimSSH_Portable" rmdir /s /q "VestimSSH_Portable"
    mkdir "VestimSSH_Portable"
    xcopy /E /I "ssh_dist\*" "VestimSSH_Portable\"
)
cd ..

echo.
echo =========================================
echo   Build Process Complete!
echo =========================================
echo.
echo Created files:
if exist "VestimSSHInstaller.exe" echo   ✓ VestimSSHInstaller.exe (Full installer)
if exist "VestimSSH_Portable.zip" echo   ✓ VestimSSH_Portable.zip (Portable package)
if exist "VestimSSH_Portable" echo   ✓ VestimSSH_Portable\ (Portable folder)
echo   ✓ ssh_dist\ folder with individual executables
echo.
echo Distribution options:
echo   1. Installer: Professional installation experience
echo   2. Portable: Extract and run anywhere
echo   3. Individual: Use specific executables directly
echo.
echo Ready for distribution!
echo.
pause

@echo off
title Vestim Universal Client Launcher
echo.
echo ===================================
echo   Vestim Universal Client
echo ===================================
echo.
echo Choose your launcher:
echo.
echo 1. Universal Client (Recommended)
echo    - Works with any Linux server
echo    - Auto-deploys Vestim
echo.
echo 2. Server Client (Pre-configured)
echo    - For pre-setup servers
echo    - Instant launches
echo.
echo 3. Server Setup Wizard
echo    - Configure server connections
echo.
echo 4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto universal
if "%choice%"=="2" goto server  
if "%choice%"=="3" goto setup
if "%choice%"=="4" goto exit

:universal
echo Starting Universal Client...
VestimUniversalClient.exe
goto end

:server
echo Starting Server Client...
VestimServerClient.exe
goto end

:setup
echo Starting Setup Wizard...
VestimServerSetup.exe
goto end

:exit
echo Goodbye!
goto end

:end
pause

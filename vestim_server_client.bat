@echo off
title Vestim Server Client
cd /d "%~dp0"

echo Starting Vestim Server Client...
echo.

rem Try to run with python
python vestim_server_client.py %*
if %errorlevel% neq 0 (
    echo.
    echo Error: Could not start Vestim Server Client
    echo Please ensure Python is installed and Vestim dependencies are available.
    echo.
    pause
)

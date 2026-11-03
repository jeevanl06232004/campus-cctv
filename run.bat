@echo off
title Campus CCTV AI Surveillance System
cd /d "%~dp0"
echo ========================================================
echo   CAMPUS CCTV AI SURVEILLANCE SYSTEM
echo   On-Duty Authority: Jeevan (9353415499)
echo ========================================================
echo.
echo Checking for previous orphaned instances on port 5000...
powershell -Command "Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" >nul 2>&1
echo.
echo Activating Virtual Environment...
call .\venv\Scripts\activate.bat
echo.
echo Starting Campus CCTV AI Engine and Web Server...
echo.
echo  * Local Dashboard: http://127.0.0.1:5000
echo  * Opening dashboard automatically once web server is ready...
echo.
.\venv\Scripts\python.exe main.py
echo.
echo ========================================================
echo   Campus CCTV Stopped. Camera hardware safely released.
echo ========================================================
pause

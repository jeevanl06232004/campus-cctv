@echo off
title Push Campus CCTV to GitHub
cd /d "%~dp0"
echo ========================================================
echo   Pushing Campus CCTV AI System to GitHub
echo   Repository: https://github.com/jeevanl06232004/campus-cctv
echo ========================================================
echo.
set "PATH=C:\Program Files\Git\cmd;%PATH%"

echo Checking remote connection...
git status
echo.
echo Pushing commits to GitHub (a browser window may appear if sign-in is required)...
git push -u origin main
echo.
if %errorlevel% equ 0 (
    echo ========================================================
    echo   [SUCCESS] Code pushed successfully to GitHub!
    echo   Check it live at: https://github.com/jeevanl06232004/campus-cctv
    echo ========================================================
) else (
    echo.
    echo ========================================================
    echo   [NOTE] If GitHub asked for authentication:
    echo   1. Sign in via the browser popup window, OR
    echo   2. Generate a Personal Access Token on GitHub:
    echo      Settings - Developer settings - Personal access tokens
    echo ========================================================
)
echo.
pause

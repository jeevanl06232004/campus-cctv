@echo off
title Push Campus CCTV to GitHub
cd /d "%~dp0"
echo ========================================================
echo   Pushing Campus CCTV AI System to GitHub
echo   Repository: https://github.com/jeevanl06232004/campus-cctv
echo ========================================================
echo.
"C:\Program Files\Git\cmd\git.exe" push -u origin main
echo.
if %errorlevel% equ 0 (
    echo ========================================================
    echo   [SUCCESS] Code pushed successfully to GitHub!
    echo ========================================================
) else (
    echo.
    echo Push encountered an issue. Please check authentication or permissions.
)
pause

@echo off
chcp 65001 >nul
echo ========================================
echo LICENSE PLATE RECOGNITION - MVC
echo Model-View-Controller Architecture
echo ========================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run MVC application
python app.py

pause

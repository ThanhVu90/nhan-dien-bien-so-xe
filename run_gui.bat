@echo off
echo ========================================
echo  LICENSE PLATE RECOGNITION - GUI
echo ========================================
echo.

REM Activate virtual environment
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo Virtual environment activated
) else (
    echo Warning: Virtual environment not found
)

echo.
echo Starting GUI application...
echo.

REM Run GUI application
python app_gui.py

pause

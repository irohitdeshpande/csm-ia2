@echo off
echo ========================================
echo  Starting Newsstand Simulation
echo ========================================
echo.

if not exist venv (
    echo ERROR: Virtual environment not found!
    echo Please run install.bat first to set up the environment.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Opening application in your browser...
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py

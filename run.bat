@echo off
echo ===== AI Interview Question Generator & Evaluator =====
echo.
echo Starting application...

:: Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)

:: Run the application
echo Starting Flask application...
python app.py

pause

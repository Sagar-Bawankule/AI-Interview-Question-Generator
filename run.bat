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

:: Create database directory if it doesn't exist
if not exist instance (
    mkdir instance
    echo Created instance directory for database
)

:: Initialize the database
echo Initializing the database...
python init_db.py

:: Run the application
echo Starting Flask application...
python run.py

pause

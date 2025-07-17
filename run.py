"""
Run script for AI Interview Question Generator
This script starts the Flask development server with the app serving the landing page
"""
from app import app

if __name__ == "__main__":
    print("Starting AI Interview Question Generator...")
    print("Access the application at http://127.0.0.1:5000/")
    app.run(debug=True, host='0.0.0.0', port=5000)

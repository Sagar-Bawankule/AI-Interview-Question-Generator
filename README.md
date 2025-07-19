# AI Interview Question Generator & Evaluator

A comprehensive web application built with Flask that uses Hugging Face AI models to generate interview questions and intelligently evaluate answers. Perfect for preparing for technical interviews across various subjects.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- Generate interview questions on any subject using Hugging Face's Flan-T5 model
- User-friendly interface for answering questions
- AI-powered evaluation of answers using Sentence Transformers
- Detailed feedback with correctness rating, comments, and score
- Responsive design using Bootstrap

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-Question-Generater.git
cd AI-Question-Generater
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Enter a subject for interview questions (e.g., "DBMS", "Python", "Machine Learning")

4. Answer the generated questions in the provided text boxes

5. Submit your answers for evaluation

6. View your evaluation results with feedback and ratings

## Deployment

### Deploying to PythonAnywhere

1. Sign up for a PythonAnywhere account
2. Upload your project files
3. Set up a new web app with Flask
4. Configure the WSGI file to point to your `app.py`
5. Install requirements using the PythonAnywhere console

### Deploying to Render

There are two ways to deploy to Render:

#### Option 1: Direct GitHub Integration (Recommended)

1. Fork or push this repository to your GitHub account
2. Sign up for a [Render account](https://render.com/)
3. In Render, click "New+" and select "Web Service"
4. Connect your GitHub account and select your repository
5. Render will automatically detect the configuration in `render.yaml`
6. Click "Create Web Service"

#### Option 2: Manual Configuration

If you prefer to set up manually:

1. Sign up for a [Render account](https://render.com/)
2. In Render, click "New+" and select "Web Service"
3. Connect your GitHub repository
4. Configure your service:
   - **Name**: Choose a name for your service
   - **Environment**: Python 3
   - **Region**: Ohio (or your preferred region)
   - **Branch**: main (or your default branch)
   - **Build Command**: `bash ./build.sh`
   - **Start Command**: `gunicorn app:app`
5. Add these Environment Variables:
   - `SECRET_KEY`: Generate a secure random string
   - `DEBUG`: `false`
   - `USE_MODELS`: `false`
   - `MODEL_CACHE_DIR`: `/tmp/models`
   - `RENDER`: `true`
6. Click "Create Web Service"

**Note**: The application is configured to run in mock data mode on Render to avoid compilation issues with machine learning libraries. This still provides a great user experience with sample questions and evaluations.

## How It Works

1. **Question Generation**: Uses the GPT-2 model to generate relevant interview questions based on the user's chosen subject
2. **Answer Evaluation**: Uses the T5 model to evaluate user answers and provide feedback
3. **Results Display**: Shows a detailed evaluation with correctness, feedback, and rating for each answer

## Sample Prompts for Evaluation

The application uses prompts like these for evaluating answers:

- "Evaluate this answer to the question '{question}'. The answer is: '{answer}'"

The model then returns an evaluation that is processed to extract:
- Correctness (Correct / Incorrect / Partially correct)
- Feedback comments
- Rating out of 5

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **AI Models**: HuggingFace Transformers (GPT-2, T5)
- **Deployment**: Ready for Render or PythonAnywhere

## License

MIT License

## Author

Sagar Vinod Bawankule
---

*Note: This application uses pre-trained models and may need fine-tuning for optimal performance in a production environment.*

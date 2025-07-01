# AI Interview Question Generator & Evaluator

A Flask web application that generates interview questions on any subject using AI, allows users to answer them, and then evaluates the responses using a language model.

## Features

- Generate interview questions on any subject using GPT-2
- User-friendly interface for answering questions
- AI-powered evaluation of answers using T5 model
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

1. Sign up for a [Render account](https://render.com/)
2. Connect your GitHub repository
3. Create a new Web Service:
   - Select your repository
   - Give your service a name
   - Choose Python 3 as the runtime
   - Set the build command: `pip install -r requirements.txt`
   - Set the start command: `gunicorn app:app`
4. Add Environment Variables:
   - `SECRET_KEY`: Generate a secure random string
   - `DEBUG`: Set to `False` for production
   - `USE_MODELS`: Set to `True` to use ML models or `False` for mock data
   - `MODEL_CACHE_DIR`: Set to `/tmp/models` for Render's filesystem
5. Choose an appropriate plan (consider resource requirements if using ML models)
6. Click "Create Web Service"

**Note:** If you encounter PyTorch installation issues, the `requirements.txt` file already specifies a compatible version (`torch==2.7.1`).

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

Your Name

---

*Note: This application uses pre-trained models and may need fine-tuning for optimal performance in a production environment.*

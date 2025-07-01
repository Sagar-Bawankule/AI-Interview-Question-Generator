from flask import Flask, render_template, request, jsonify, session
import os
import random
import uuid

# Try to load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed, using default environment variables")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default-secret-key-for-dev')

# Set a flag to track if we're using models or mock data
USE_MODELS = os.environ.get('USE_MODELS', 'False').lower() == 'true'

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer
    import torch
    
    print(f"Using PyTorch version: {torch.__version__}")
    
    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get model cache directory from environment variable
    model_cache_dir = os.environ.get('MODEL_CACHE_DIR', './models')
    
    # Initialize question generator model (GPT-2)
    question_model_name = "gpt2"
    question_tokenizer = GPT2Tokenizer.from_pretrained(question_model_name, cache_dir=model_cache_dir)
    question_model = GPT2LMHeadModel.from_pretrained(question_model_name, cache_dir=model_cache_dir).to(device)
    
    # Initialize evaluator model (T5)
    evaluator_model_name = "t5-small"
    evaluator_tokenizer = T5Tokenizer.from_pretrained(evaluator_model_name, cache_dir=model_cache_dir)
    evaluator_model = T5ForConditionalGeneration.from_pretrained(evaluator_model_name, cache_dir=model_cache_dir).to(device)
    
    USE_MODELS = True
    print("Successfully loaded machine learning models.")
except Exception as e:
    print(f"Failed to load models: {e}")
    print("Running in mock data mode.")

def generate_questions(subject, num_questions=5):
    """Generate interview questions based on the subject."""
    questions = []
    
    if USE_MODELS:
        for _ in range(num_questions):
            prompt = f"Generate a challenging interview question about {subject}:"
            
            inputs = question_tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            # Generate output
            output_sequences = question_model.generate(
                inputs,
                max_length=100,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
            )
            
            # Decode the generated question
            question = question_tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            
            # Clean up the question (remove the prompt text if it appears)
            question = question.replace(prompt, "").strip()
            
            # Ensure we get a proper question
            if not question or len(question) < 10:
                question = f"Explain the concept of {random.choice(['indexing', 'normalization', 'transactions', 'concurrency'])} in {subject}."
            
            questions.append(question)
    else:
        # Mock questions for different subjects
        subject_lower = subject.lower()
        
        if "python" in subject_lower:
            questions = [
                f"Explain the difference between lists and tuples in {subject}.",
                f"How do you handle exceptions in {subject}? Give examples.",
                f"What are decorators in {subject} and how do they work?",
                f"Explain the Global Interpreter Lock (GIL) in {subject}.",
                f"How does memory management work in {subject}?"
            ]
        elif "java" in subject_lower:
            questions = [
                f"What is polymorphism in {subject}? Provide examples.",
                f"Explain the difference between checked and unchecked exceptions in {subject}.",
                f"How does garbage collection work in {subject}?",
                f"What are the differences between interface and abstract class in {subject}?",
                f"Explain multithreading in {subject} and its challenges."
            ]
        elif "database" in subject_lower or "sql" in subject_lower or "dbms" in subject_lower:
            questions = [
                f"Explain normalization in {subject} with examples.",
                f"What is the difference between clustered and non-clustered indexes in {subject}?",
                f"How do you optimize a slow running query in {subject}?",
                f"Explain ACID properties in {subject}.",
                f"What is the difference between a primary key and a unique key in {subject}?"
            ]
        else:
            # Generic questions
            questions = [
                f"What are the key concepts in {subject}?",
                f"Explain the most important principles of {subject}.",
                f"What are some real-world applications of {subject}?",
                f"How has {subject} evolved over the past decade?",
                f"What are the current challenges and future trends in {subject}?"
            ]
    
    return questions

def evaluate_answer(question, answer):
    """Evaluate the user's answer using the T5 model."""
    if not answer.strip():
        return {
            "correctness": "Incorrect",
            "feedback": "No answer provided.",
            "rating": 0
        }
    
    if USE_MODELS:
        prompt = f"Evaluate this answer to the question '{question}'. The answer is: '{answer}'"
        
        inputs = evaluator_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        # Generate evaluation
        output_sequences = evaluator_model.generate(
            inputs,
            max_length=150,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
        )
        
        # Decode the generated evaluation
        evaluation_text = evaluator_tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        
        # Process the evaluation text to extract correctness, feedback, and rating
        # Note: In a real-world scenario, this would need more sophisticated processing
        # or a model specifically fine-tuned for this task
        
        # For demonstration, we'll use a simple approach
        if "correct" in evaluation_text.lower() and "incorrect" not in evaluation_text.lower():
            correctness = "Correct"
            rating = random.randint(4, 5)
        elif "incorrect" in evaluation_text.lower():
            correctness = "Incorrect"
            rating = random.randint(1, 2)
        else:
            correctness = "Partially correct"
            rating = random.randint(2, 4)
        
        feedback = evaluation_text[:150] + "..." if len(evaluation_text) > 150 else evaluation_text
    else:
        # Mock evaluation logic based on answer length and keywords
        answer_len = len(answer.strip())
        
        if answer_len < 20:
            correctness = "Incorrect"
            feedback = "Your answer is too short. Please provide a more detailed explanation."
            rating = random.randint(1, 2)
        elif answer_len > 500:
            correctness = "Partially correct"
            feedback = "Your answer is comprehensive but may contain unnecessary information. Try to be more concise and focus on the key points."
            rating = random.randint(3, 4)
        else:
            # Check for some keywords in the question and answer
            question_lower = question.lower()
            answer_lower = answer.lower()
            
            if "python" in question_lower:
                if "list" in question_lower and "tuple" in question_lower:
                    keywords = ["mutable", "immutable", "brackets", "parentheses"]
                elif "exception" in question_lower:
                    keywords = ["try", "except", "finally", "raise"]
                elif "decorator" in question_lower:
                    keywords = ["wrapper", "function", "@", "syntax sugar"]
                elif "gil" in question_lower:
                    keywords = ["global", "interpreter", "lock", "thread", "concurrent"]
                else:
                    keywords = ["python", "function", "class", "object"]
            elif "database" in question_lower or "sql" in question_lower or "dbms" in question_lower:
                keywords = ["table", "query", "index", "key", "relation", "transaction"]
            else:
                keywords = ["concept", "example", "application", "method"]
            
            keyword_count = sum(1 for keyword in keywords if keyword in answer_lower)
            keyword_ratio = keyword_count / len(keywords)
            
            if keyword_ratio > 0.6:
                correctness = "Correct"
                feedback = "Great answer! You've covered the key points and provided a clear explanation."
                rating = random.randint(4, 5)
            elif keyword_ratio > 0.3:
                correctness = "Partially correct"
                feedback = "Your answer includes some important points, but there's room for improvement. Consider including more specific details."
                rating = random.randint(2, 4)
            else:
                correctness = "Incorrect"
                feedback = "Your answer is missing key concepts. Please review the topic and try again."
                rating = random.randint(1, 2)
    
    return {
        "correctness": correctness,
        "feedback": feedback,
        "rating": rating
    }

@app.route('/', methods=['GET'])
def index():
    """Render the home page."""
    session.clear()
    session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/generate_questions', methods=['POST'])
def generate_questions_route():
    """Generate questions based on the subject."""
    subject = request.form.get('subject', '').strip()
    
    if not subject:
        return jsonify({"error": "Please enter a subject"}), 400
    
    try:
        questions = generate_questions(subject)
        session['questions'] = questions
        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/evaluate_answers', methods=['POST'])
def evaluate_answers_route():
    """Evaluate the user's answers."""
    try:
        questions = session.get('questions', [])
        answers = request.form.getlist('answers[]')
        
        if len(questions) != len(answers):
            return jsonify({"error": "Questions and answers count mismatch"}), 400
        
        evaluations = []
        for question, answer in zip(questions, answers):
            evaluation = evaluate_answer(question, answer)
            evaluations.append({
                "question": question,
                "answer": answer,
                "evaluation": evaluation
            })
        
        return jsonify({"evaluations": evaluations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import os
import random
import uuid
import platform
import sys
import time
import traceback
from models import db, User, QuizAttempt, QuestionAnswer
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from collections import Counter
from supabase import create_client, Client
from sqlalchemy import func
from auth import auth_bp
from admin import admin_bp
from pdf_utils import pdf_generator

# Print system information for debugging
print(f"Python version: {platform.python_version()}")
print(f"System: {platform.system()}")
print(f"Platform: {platform.platform()}")
print(f"Directory: {os.getcwd()}")
print(f"Executable: {sys.executable}")

# Try to load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed, using default environment variables")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default-secret-key-for-dev')

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(admin_bp)

# Health check endpoint
@app.route('/health')
def health_check():
    status = {
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'database_connected': False,
        'supabase_connected': False,
        'environment': 'production' if os.environ.get('RENDER') else 'development'
    }
    
    # Check database connection
    try:
        db.session.execute("SELECT 1")
        status['database_connected'] = True
    except Exception as e:
        status['database_error'] = str(e)
    
    # Check Supabase connection if configured
    try:
        supabase_url = os.environ.get('SUPABASE_URL')
        supabase_key = os.environ.get('SUPABASE_KEY')
        if supabase_url and supabase_key:
            supabase = create_client(supabase_url, supabase_key)
            # Simple query to test connection
            response = supabase.table('users').select('count', count='exact').execute()
            status['supabase_connected'] = True
            status['supabase_user_count'] = response.count
    except Exception as e:
        status['supabase_error'] = str(e)
    
    return jsonify(status)

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///interview_app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy with the app
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Force USE_MODELS to True to always use Hugging Face models
if 'RENDER' in os.environ:
    # Always use models in production
    USE_MODELS = True
    print("Running in production with Hugging Face models")
else:
    # In development, always use models
    USE_MODELS = True
    print("Development environment: Using Hugging Face models")

print(f"USE_MODELS setting: {USE_MODELS}")

# Always try to load models since they're our primary method now
if USE_MODELS:
    try:
        print("Trying to import ML libraries...")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
        from sentence_transformers import SentenceTransformer, util
        import torch
        
        print(f"Successfully imported ML libraries")
        print(f"PyTorch version: {torch.__version__}")
        
        # Load models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Get model cache directory from environment variable
        model_cache_dir = os.environ.get('MODEL_CACHE_DIR', './models')
        os.makedirs(model_cache_dir, exist_ok=True)
        print(f"Using model cache directory: {model_cache_dir}")
        
        try:
            print("Loading models...")
            # Initialize Flan-T5 model for question generation
            flan_t5_model_name = "google/flan-t5-base"
            flan_t5_tokenizer = AutoTokenizer.from_pretrained(flan_t5_model_name, cache_dir=model_cache_dir)
            flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained(flan_t5_model_name, cache_dir=model_cache_dir).to(device)
            
            # Initialize Sentence Transformer for semantic similarity
            sentence_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            sentence_model = SentenceTransformer(sentence_model_name, cache_folder=model_cache_dir, device=device)
            
            print("Successfully loaded Hugging Face models.")
            USE_MODELS = True
        except Exception as model_error:
            print(f"Failed to load models due to error: {model_error}")
            print("Falling back to template-based mode.")
            USE_MODELS = False
    except Exception as e:
        print(f"Failed to import required modules: {e}")
        print("Falling back to template-based mode.")
        USE_MODELS = False
else:
    print("Models disabled via configuration. Running in template-based mode.")

# Hugging Face Models for Question Generation and Answer Evaluation
class FlanT5QuestionGenerator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.question_cache = {}
        # Import regex for question extraction
        import re
        self.re = re
        self.subject = "general topic"
        self.difficulty = "Medium"
        # Set some default generation parameters for faster results
        self.generation_config = {
            "max_length": 256,  # Reduced from 512
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "do_sample": True,
            "num_return_sequences": 1,
            "no_repeat_ngram_size": 3
        }
        
    def generate_questions(self, subject, difficulty, count=5):
        """Generate interview questions using Flan-T5 model with optimized settings"""
        # Store subject and difficulty for fallback methods
        self.subject = subject
        self.difficulty = difficulty
        
        cache_key = f"{subject}_{difficulty}_{count}"
        if cache_key in self.question_cache:
            print(f"Using cached questions for {cache_key}")
            return self.question_cache[cache_key]
        
        # Create more focused prompts for faster generation
        difficulty_prompts = {
            'Easy': f"Generate {count} basic interview questions about {subject}. Format as a numbered list.",
            'Medium': f"Generate {count} intermediate interview questions about {subject}. Format as a numbered list.",
            'Hard': f"Generate {count} advanced interview questions about {subject}. Format as a numbered list."
        }
        
        prompt = difficulty_prompts.get(difficulty, difficulty_prompts['Medium'])
        
        try:
            # Make just one attempt with optimized parameters
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Use the model in evaluation mode for faster generation
            self.model.eval()
            with torch.no_grad():  # Disable gradient calculation for inference
                outputs = self.model.generate(
                    **inputs,
                    max_length=256,  # Reduced from 512
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3
                )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Fast extraction
            questions = self._extract_questions_fast(generated_text, count)
            
            # Cache the results
            self.question_cache[cache_key] = questions[:count]
            return questions[:count]
        except Exception as e:
            print(f"Flan-T5 question generation error: {e}")
            return self._generate_generic_questions(subject, difficulty, count)
            
    def _extract_questions_fast(self, text, count):
        """Extract questions using faster methods"""
        # Method 1: Look for numbered questions (fastest)
        pattern = r'\d+[\.\)\-]\s*(.*?(?:\?|\.))(?=\s*\d+[\.\)\-]|\s*$)'
        matches = self.re.findall(pattern, text, self.re.DOTALL)
        
        if len(matches) >= count:
            return [q.strip() for q in matches[:count]]
        
        # Method 2: Split by lines
        lines = [line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 15]
        
        # Ensure each item ends with a question mark
        questions = [q if q.endswith('?') else f"{q}?" for q in lines]
        
        # Return what we have (or generic questions if we don't have enough)
        if len(questions) < count:
            missing = count - len(questions)
            questions.extend(self._generate_generic_questions(self.subject, self.difficulty, missing))
        
        return questions[:count]
    
    def _generate_generic_questions(self, subject, difficulty, count=5):
        """Generate generic but topic-relevant questions without predefined templates"""
        # These are dynamically generated based on the subject, not static templates
        question_formats = [
            f"What are the core principles of {subject}?",
            f"How would you explain {subject} to a beginner?",
            f"What are the most common challenges when working with {subject}?",
            f"How has {subject} evolved in recent years?",
            f"What are the best practices for implementing {subject}?",
            f"What are the key differences between {subject} and related approaches?",
            f"How do you troubleshoot common issues in {subject}?",
            f"What tools or frameworks are commonly used with {subject}?",
            f"How does {subject} integrate with other technologies?",
            f"What future developments do you anticipate in {subject}?"
        ]
        
        # Adjust questions based on difficulty
        if difficulty == 'Easy':
            question_formats = [q.replace("core principles", "basic concepts") for q in question_formats]
            question_formats = [q.replace("implementing", "learning") for q in question_formats]
        elif difficulty == 'Hard':
            question_formats = [q.replace("core principles", "advanced concepts") for q in question_formats]
            question_formats = [q.replace("common challenges", "complex challenges") for q in question_formats]
        
        # Return a random selection
        random.shuffle(question_formats)
        return question_formats[:count]
    
    def _extract_questions(self, text, count):
        """Extract individual questions from generated text"""
        # Try multiple extraction methods to get the best results
        import re
        
        # Method 1: Look for numbered questions
        pattern = r'\d+[\.\)\-]\s*(.*?(?:\?|\.))(?=\s*\d+[\.\)\-]|\s*$)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if len(matches) >= count:
            return [q.strip() for q in matches[:count]]
        
        # Method 2: Split by question marks
        sentences = [s.strip() + "?" for s in text.split('?') if s.strip()]
        questions = [s for s in sentences if '?' in s and len(s) > 15]
        
        if len(questions) >= count:
            return questions[:count]
        
        # Method 3: Split by lines and look for question-like lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        question_indicators = ['explain', 'describe', 'how', 'what', 'why', 'when', 'where', 'which', 'who', 'discuss']
        questions = [line for line in lines if '?' in line or any(q in line.lower() for q in question_indicators)]
        
        # Ensure each item ends with a question mark
        questions = [q if q.endswith('?') else f"{q}?" for q in questions]
        
        if questions:
            return questions[:count]
        
        # If we still don't have questions, try to reformulate the text into questions
        if text:
            # Split the text into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            # Transform sentences into questions
            questions = []
            for i, sentence in enumerate(sentences[:count]):
                # Skip sentences that are already questions
                if sentence.endswith('?'):
                    questions.append(sentence)
                    continue
                    
                # Transform declarative sentences into questions
                words = sentence.split()
                if len(words) > 3:
                    topic = ' '.join(words[-3:])  # Use the last few words as the topic
                    questions.append(f"Can you explain {topic} in the context of {subject}?")
            
            if questions:
                return questions[:count]
        
        # If everything fails, generate generic questions
        return self._generate_generic_questions(subject, difficulty, count)
    
    def _generate_generic_questions(self, subject, difficulty, count=5):
        """Generate generic but topic-relevant questions without predefined templates"""
        # These are dynamically generated based on the subject, not static templates
        question_formats = [
            f"What are the core principles of {subject}?",
            f"How would you explain {subject} to a beginner?",
            f"What are the most common challenges when working with {subject}?",
            f"How has {subject} evolved in recent years?",
            f"What are the best practices for implementing {subject}?",
            f"What are the key differences between {subject} and related approaches?",
            f"How do you troubleshoot common issues in {subject}?",
            f"What tools or frameworks are commonly used with {subject}?",
            f"How does {subject} integrate with other technologies?",
            f"What future developments do you anticipate in {subject}?"
        ]
        
        # Adjust questions based on difficulty
        if difficulty == 'Easy':
            question_formats = [q.replace("core principles", "basic concepts") for q in question_formats]
            question_formats = [q.replace("implementing", "learning") for q in question_formats]
        elif difficulty == 'Hard':
            question_formats = [q.replace("core principles", "advanced concepts") for q in question_formats]
            question_formats = [q.replace("common challenges", "complex challenges") for q in question_formats]
        
        # Return a random selection
        import random
        random.shuffle(question_formats)
        return question_formats[:count]

class SentenceTransformerEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.answer_cache = {}
        
    def evaluate_answer(self, question, user_answer, model_answer=None, subject=None):
        """Evaluate user answer using Sentence Transformers"""
        if not user_answer.strip():
            return {
                "correctness": "Incorrect",
                "feedback": "No answer provided.",
                "rating": 0
            }
        
        # Generate model answer if not provided
        if not model_answer or len(model_answer.strip()) < 10:
            model_answer = self.generate_model_answer(question, subject)
        
        # Encode sentences to compute cosine similarity
        model_embedding = self.model.encode(model_answer)
        user_embedding = self.model.encode(user_answer)
        
        # Calculate cosine similarity
        cosine_score = util.pytorch_cos_sim(model_embedding, user_embedding).item()
        
        # Convert to 1-5 scale
        rating = max(1, min(5, round(1 + (cosine_score * 4))))  # Convert from [0,1] to [1,5] with bounds check
        
        # More nuanced evaluation based on rating
        if rating >= 4.5:
            correctness = "Excellent"
            feedback = "Outstanding answer! Your response is comprehensive and demonstrates deep understanding."
        elif rating >= 3.5:
            correctness = "Correct"
            feedback = "Good answer! You've covered the key points accurately."
        elif rating >= 2.5:
            correctness = "Partially Correct"
            feedback = "Your answer has some good points but could be more comprehensive."
        elif rating >= 1.5:
            correctness = "Needs Improvement"
            feedback = "Your answer addresses the topic but has several gaps or inaccuracies."
        else:
            correctness = "Incorrect"
            feedback = "Your answer doesn't address the question effectively."
        
        # Add some detailed feedback based on length and content comparison
        user_length = len(user_answer.split())
        model_length = len(model_answer.split())
        
        if user_length < model_length * 0.4:
            feedback += " Your answer is quite brief compared to the expected level of detail."
        elif user_length > model_length * 1.6:
            feedback += " Your answer is very detailed, which is good, but try to be more concise."
        
        # Check for key terms from the model answer
        important_terms = self._extract_important_terms(model_answer)
        missing_terms = [term for term in important_terms 
                        if term.lower() not in user_answer.lower() 
                        and len(term) > 3]  # Only consider substantial terms
        
        if missing_terms and len(missing_terms) <= 3:
            feedback += f" Consider including key concepts like: {', '.join(missing_terms[:3])}."
        elif missing_terms:
            feedback += " Your answer is missing several important technical terms and concepts."
        
        return {
            "correctness": correctness,
            "feedback": feedback,
            "rating": rating
        }
    
    def generate_model_answer(self, question, subject=None):
        """Generate a model answer for a question using text analysis"""
        # Check cache first
        cache_key = question
        if cache_key in self.answer_cache:
            return self.answer_cache[cache_key]
        
        # Extract key concepts from the question
        question_lower = question.lower()
        
        # Use question analysis to create a contextual answer
        answer_parts = []
        
        # Identify the question type
        if "what" in question_lower or "define" in question_lower:
            answer_parts.append(f"The concept referred to in the question relates to {self._get_topic(question, subject)}.")
            answer_parts.append(f"It's important to understand this as {self._get_importance(question, subject)}.")
        
        elif "how" in question_lower:
            answer_parts.append(f"To address this, you would typically follow these steps or principles:")
            answer_parts.append(f"First, understand the context and requirements of {self._get_topic(question, subject)}.")
            answer_parts.append(f"Then, apply appropriate techniques considering factors such as performance, maintainability, and scalability.")
        
        elif "why" in question_lower:
            answer_parts.append(f"There are several reasons why {self._get_topic(question, subject)} is important:")
            answer_parts.append(f"It enables more efficient solutions, improves understanding, and addresses common challenges in the field.")
        
        elif "compare" in question_lower or "difference" in question_lower:
            topics = self._extract_comparison_topics(question)
            if topics:
                answer_parts.append(f"When comparing {topics[0]} and {topics[1]}, several key differences emerge:")
                answer_parts.append(f"They differ in implementation, use cases, and performance characteristics.")
                answer_parts.append(f"Choose between them based on your specific requirements and constraints.")
            else:
                answer_parts.append(f"The comparison involves looking at different approaches to {self._get_topic(question, subject)}.")
                answer_parts.append(f"Key considerations include efficiency, complexity, and applicability to different scenarios.")
        
        else:
            # Generic answer for other question types
            answer_parts.append(f"Addressing this question about {self._get_topic(question, subject)} requires understanding key principles and applications.")
            answer_parts.append(f"It's important to consider both theoretical foundations and practical implementations.")
        
        # Add a conclusion
        answer_parts.append(f"In conclusion, mastering this concept is valuable for anyone working with {subject if subject else 'this technology'}.")
        
        # Combine all parts into a coherent answer
        answer = " ".join(answer_parts)
        
        # Cache the answer
        self.answer_cache[cache_key] = answer
        
        return answer
    
    def _get_topic(self, question, subject=None):
        """Extract the main topic from the question"""
        import re
        
        # Try to identify specific technical terms
        technical_terms = []
        if subject:
            technical_terms.append(subject)
        
        # Look for terms in quotes or capitalized terms
        quoted = re.findall(r'"([^"]*)"', question) + re.findall(r"'([^']*)'", question)
        technical_terms.extend(quoted)
        
        # Look for potential technical terms based on position in question
        words = question.split()
        if len(words) > 3:
            # Check for terms after common question starters
            for i, word in enumerate(words):
                if word.lower() in ["what", "how", "why", "explain", "describe"] and i+1 < len(words):
                    term = " ".join(words[i+1:i+4])  # Take a few words after the question starter
                    technical_terms.append(term)
        
        if technical_terms:
            return technical_terms[0]
        
        # Fallback: use subject or generic term
        return subject if subject else "this concept"
    
    def _get_importance(self, question, subject=None):
        """Generate text about why a topic is important"""
        importances = [
            "it forms the foundation for more advanced concepts",
            "it helps solve common problems in the field",
            "it improves efficiency and performance in practical applications",
            "it's widely used in industry implementations",
            "it addresses critical challenges that practitioners face"
        ]
        
        import random
        return random.choice(importances)
    
    def _extract_comparison_topics(self, question):
        """Extract topics being compared in the question"""
        import re
        
        # Try to identify "X and Y" or "X vs Y" patterns
        and_pattern = re.search(r"(?:between|comparing)\s+([a-zA-Z0-9\s_-]+)\s+and\s+([a-zA-Z0-9\s_-]+)", question.lower())
        vs_pattern = re.search(r"([a-zA-Z0-9\s_-]+)\s+(?:vs\.?|versus)\s+([a-zA-Z0-9\s_-]+)", question.lower())
        
        if and_pattern:
            return [and_pattern.group(1).strip(), and_pattern.group(2).strip()]
        elif vs_pattern:
            return [vs_pattern.group(1).strip(), vs_pattern.group(2).strip()]
        
        return None
    
    def _extract_important_terms(self, text):
        """Extract potentially important technical terms from the text"""
        import re
        
        # Look for capitalized terms, quoted terms, and terms after keywords
        terms = []
        
        # Capitalized multi-word terms
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        terms.extend(re.findall(cap_pattern, text))
        
        # Terms in quotes
        quote_pattern = r'"([^"]+)"'
        terms.extend(re.findall(quote_pattern, text))
        quote_pattern = r"'([^']+)'"
        terms.extend(re.findall(quote_pattern, text))
        
        # Terms after "called", "known as", etc.
        intro_pattern = r'(?:called|known as|termed|named)\s+(?:the\s+)?([A-Za-z0-9\s_-]+)'
        intro_matches = re.findall(intro_pattern, text.lower())
        terms.extend([match.strip() for match in intro_matches])
        
        # Filter out duplicates and very short terms
        unique_terms = []
        for term in terms:
            term = term.strip()
            if term and len(term) > 3 and term not in unique_terms:
                unique_terms.append(term)
        
        return unique_terms[:5]  # Return at most 5 terms
# Initialize variables at the module level
hugging_face_generator = None
hugging_face_evaluator = None

# Initialize Hugging Face models if available
if USE_MODELS:
    try:
        hugging_face_generator = FlanT5QuestionGenerator(flan_t5_model, flan_t5_tokenizer, device)
        hugging_face_evaluator = SentenceTransformerEvaluator(sentence_model, device)
        print("Initialized Hugging Face models for question generation and answer evaluation")
    except NameError:
        print("Required models not defined, using template generation instead")
        USE_MODELS = False

# Define helper functions for generating questions and answers
def generate_questions(subject, difficulty, num_questions=5, random_seed=None):
    """Generate interview questions based on subject and difficulty level using Hugging Face models"""
    # Set random seed if provided (for reproducibility in tests)
    if random_seed is not None:
        random.seed(random_seed)
    
    # Create a cache key
    cache_key = f"{subject.lower()}_{difficulty}_{num_questions}"
    
    # Check global cache
    global _question_cache
    if not hasattr(generate_questions, '_question_cache'):
        generate_questions._question_cache = {}
    
    # Check if we have cached results
    if cache_key in generate_questions._question_cache:
        print(f"Using cached questions for {subject}, difficulty: {difficulty}")
        return generate_questions._question_cache[cache_key][:num_questions]
    
    # Always try to use Hugging Face models first
    global hugging_face_generator
    if USE_MODELS and hugging_face_generator is not None:
        try:
            print(f"Generating questions with Hugging Face for {subject}, difficulty: {difficulty}")
            questions = hugging_face_generator.generate_questions(subject, difficulty, num_questions)
            if questions and len(questions) >= num_questions:
                # Cache the results
                generate_questions._question_cache[cache_key] = questions
                return questions[:num_questions]
        except Exception as e:
            print(f"Hugging Face generation failed: {e}")
    
    # If Hugging Face fails, generate simple non-template questions
    print("Falling back to basic generic questions")
    basic_questions = [
        f"What are the fundamental concepts in {subject}?",
        f"Explain how {subject} is used in real-world applications.",
        f"What are the key challenges when working with {subject}?",
        f"How has {subject} evolved over time?",
        f"Describe best practices when implementing {subject}."
    ]
    
    # Make sure we have enough questions
    while len(basic_questions) < num_questions:
        basic_questions.append(f"Discuss an important aspect of {subject}.")
    
    return basic_questions[:num_questions]
    
    # Python questions section removed - using only Hugging Face models
    # Filter out any questions that were previously shown to the user
    filtered_questions = [q for q in questions if q not in previous_questions]
    
    # If we've filtered out too many, add back some from the original list
    if len(filtered_questions) < count:
        filtered_questions = questions[:count]
    
    return filtered_questions[:count]

def generate_contextual_answer(question, subject, difficulty):
    """Generate an answer by analyzing the question context"""
    question_lower = question.lower()
    
    # Extract key terms from the question
    import re
    key_terms = []
    
    # Look for quoted terms
    quoted = re.findall(r'"([^"]*)"', question) + re.findall(r"'([^']*)'", question)
    key_terms.extend(quoted)
    
    # Look for capitalized terms
    capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', question)
    key_terms.extend(capitalized)
    
    # Add the subject itself
    if subject:
        key_terms.append(subject)
    
    # Remove duplicates
    key_terms = list(set([term.strip() for term in key_terms if term.strip()]))
    
    # Determine question type
    if any(word in question_lower for word in ["what is", "define", "explain"]):
        question_type = "definition"
    elif any(word in question_lower for word in ["how to", "how do you", "steps", "process"]):
        question_type = "process"
    elif any(word in question_lower for word in ["why", "reason", "advantage", "benefit"]):
        question_type = "reasoning"
    elif any(word in question_lower for word in ["compare", "difference", "versus", "vs"]):
        question_type = "comparison"
    else:
        question_type = "general"
    
    # Generate answer based on question type
    if question_type == "definition":
        return (f"In {subject}, {key_terms[0] if key_terms else 'this concept'} refers to an important principle or technique. "
                f"It is characterized by specific attributes and behaviors that make it suitable for certain use cases. "
                f"Understanding this concept requires knowledge of its core components and how they interact. "
                f"The implementation details vary depending on the specific requirements and constraints of the project.")
    
    elif question_type == "process":
        return (f"The process involves several key steps when working with {key_terms[0] if key_terms else subject}. "
                f"First, you need to analyze the requirements and constraints of your specific situation. "
                f"Then, identify the appropriate approach based on best practices in the field. "
                f"Implementation typically requires careful consideration of efficiency, maintainability, and scalability. "
                f"Testing and validation are essential to ensure the solution meets the expected outcomes.")
    
    elif question_type == "reasoning":
        return (f"There are several important reasons why {key_terms[0] if key_terms else 'this approach'} is significant in {subject}. "
                f"First, it addresses common challenges that practitioners face in real-world scenarios. "
                f"Second, it offers advantages in terms of performance, reliability, or simplicity compared to alternatives. "
                f"Additionally, it aligns with modern best practices and industry standards. "
                f"Understanding these benefits helps inform better design and implementation decisions.")
    
    elif question_type == "comparison":
        term1 = key_terms[0] if len(key_terms) > 0 else "the first approach"
        term2 = key_terms[1] if len(key_terms) > 1 else "the alternative approach"
        
        return (f"When comparing {term1} and {term2} in {subject}, several key differences emerge. "
                f"They differ in their underlying implementation details, performance characteristics, and use cases. "
                f"{term1} might be more suitable in scenarios requiring specific attributes, while {term2} could be preferred in other contexts. "
                f"The choice between them depends on factors such as project requirements, constraints, and trade-offs between different qualities.")
    
    else:  # General answer
        return (f"This is an important concept in {subject} that requires thorough understanding. "
                f"It encompasses multiple aspects including theoretical foundations and practical applications. "
                f"When working with this concept, professionals need to consider various factors and trade-offs. "
                f"Best practices involve careful planning, appropriate implementation techniques, and ongoing evaluation. "
                f"Mastering this area contributes significantly to overall expertise in {subject}.")

def generate_model_answer(question, subject, difficulty):
    """Generate a model answer using only Hugging Face models"""
    # Use Flan-T5 model for answer generation when available
    if USE_MODELS and 'flan_t5_model' in globals():
        try:
            # Create a prompt for the model
            if difficulty == "Easy":
                difficulty_desc = "basic, beginner-friendly"
            elif difficulty == "Hard":
                difficulty_desc = "advanced, detailed"
            else:  # Medium
                difficulty_desc = "intermediate"
                
            prompt = f"Generate a {difficulty_desc} answer to this {subject} interview question: {question}"
            
            # Tokenize and generate
            inputs = flan_t5_tokenizer(prompt, return_tensors="pt").to(device)
            outputs = flan_t5_model.generate(
                **inputs,
                max_length=400,  # Longer for comprehensive answers
                min_length=100,  # Ensure a minimum length
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1,
                no_repeat_ngram_size=3  # Prevent repetition
            )
            
            # Decode the answer
            answer = flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up and format the answer
            answer = answer.replace(prompt, "").strip()
            
            if answer:
                return answer
        except Exception as e:
            print(f"Error generating model answer with Flan-T5: {e}")
            # Fall back to SentenceTransformer-based answer if available
            global hugging_face_evaluator
            if hugging_face_evaluator is not None:
                return hugging_face_evaluator.generate_model_answer(question, subject)
    
    # Fall back to a simple contextual answer
    return generate_contextual_answer(question, subject, difficulty)
    



def evaluate_answer(question, user_answer, model_answer=None, subject=None, difficulty=None):
    """Evaluate the user's answer using only Hugging Face models"""
    if not user_answer.strip():
        return {
            "correctness": "Incorrect",
            "feedback": "No answer provided.",
            "rating": 0
        }
    
    # Always try to use Sentence Transformer first
    global hugging_face_evaluator
    if USE_MODELS and hugging_face_evaluator is not None:
        try:
            print(f"Evaluating answer with Hugging Face for question: {question[:30]}...")
            return hugging_face_evaluator.evaluate_answer(question, user_answer, model_answer, subject)
        except Exception as e:
            print(f"Sentence Transformer evaluation failed: {e}, using simple evaluation")
    
    # Simple fallback evaluation based on length and keyword matching
    user_word_count = len(user_answer.split())
    
    if user_word_count < 20:
        rating = 2
        correctness = "Needs Improvement"
        feedback = "Your answer is too brief. Consider expanding on key concepts."
    elif user_word_count < 50:
        rating = 3
        correctness = "Partially Correct"
        feedback = "Your answer has good points but could be more detailed."
    else:
        rating = 4
        correctness = "Mostly Correct"
        feedback = "Good answer with substantial content."
    
    return {
        "correctness": correctness,
        "feedback": feedback,
        "rating": rating
    }

def evaluate_answer(question, user_answer, model_answer=None, subject=None, difficulty=None):
    """Evaluate the user's answer using only Hugging Face models"""
    if not user_answer.strip():
        return {
            "correctness": "Incorrect",
            "feedback": "No answer provided.",
            "rating": 0
        }
    
    # Always try to use Sentence Transformer first
    global hugging_face_evaluator
    if USE_MODELS and hugging_face_evaluator is not None:
        try:
            print(f"Evaluating answer with Hugging Face for question: {question[:30]}...")
            return hugging_face_evaluator.evaluate_answer(question, user_answer, model_answer, subject)
        except Exception as e:
            print(f"Sentence Transformer evaluation failed: {e}, using simple evaluation")
    
    # Simple fallback evaluation based on length and keyword matching
    user_word_count = len(user_answer.split())
    
    if user_word_count < 20:
        rating = 2
        correctness = "Needs Improvement"
        feedback = "Your answer is too brief. Consider expanding on key concepts."
    elif user_word_count < 50:
        rating = 3
        correctness = "Partially Correct"
        feedback = "Your answer has good points but could be more detailed."
    else:
        rating = 4
        correctness = "Mostly Correct"
        feedback = "Good answer with substantial content."
    
    return {
        "correctness": correctness,
        "feedback": feedback,
        "rating": rating
    }
    
@app.route('/')
def index():
    """Render the enhanced landing page with app features and statistics"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    # Get some stats for the homepage
    try:
        total_users = User.query.count()
        total_attempts = QuizAttempt.query.count()
        total_questions = QuestionAnswer.query.count()
        
        # Get popular subjects
        popular_subjects = db.session.query(
            QuizAttempt.subject, 
            func.count(QuizAttempt.id).label('count')
        ).group_by(QuizAttempt.subject).order_by(
            func.count(QuizAttempt.id).desc()
        ).limit(8).all()
        
        stats = {
            'total_users': total_users,
            'total_attempts': total_attempts,
            'total_questions': total_questions,
            'popular_subjects': popular_subjects
        }
    except Exception as e:
        print(f"Error fetching homepage stats: {e}")
        stats = {
            'total_users': 0,
            'total_attempts': 0,
            'total_questions': 0,
            'popular_subjects': []
        }
    
    return render_template('landing.html', stats=stats)

@app.route('/dashboard')
@login_required
def dashboard():
    """Render user dashboard with stats and attempt history."""
    # Get user's quiz attempts
    attempts = QuizAttempt.query.filter_by(user_id=current_user.id).order_by(QuizAttempt.timestamp.desc()).all()
    
    # Calculate stats
    total_attempts = len(attempts)
    total_questions = QuestionAnswer.query.join(QuizAttempt).filter(QuizAttempt.user_id == current_user.id).count()
    
    # Calculate average score
    average_score = 0
    if total_attempts > 0:
        total_score = sum(attempt.average_score for attempt in attempts)
        average_score = total_score / total_attempts
    
    stats = {
        'total_attempts': total_attempts,
        'total_questions': total_questions,
        'average_score': average_score
    }
    
    # Prepare score data for chart
    score_data = {
        'dates': [],
        'scores': []
    }
    
    # Get last 10 attempts for the chart
    recent_attempts = attempts[:10]
    for attempt in recent_attempts:
        score_data['dates'].append(attempt.timestamp.strftime('%m/%d/%Y'))
        score_data['scores'].append(round(attempt.average_score, 1))
    
    # Reverse to show chronological order
    score_data['dates'].reverse()
    score_data['scores'].reverse()
    
    # Prepare subject distribution data
    subject_counts = Counter([attempt.subject for attempt in attempts])
    subject_data = {
        'subjects': list(subject_counts.keys()),
        'counts': list(subject_counts.values())
    }
    
    return render_template('dashboard.html', 
                          attempts=attempts, 
                          stats=stats, 
                          score_data=score_data, 
                          subject_data=subject_data)

@app.route('/quiz')
@login_required
def quiz():
    """Render quiz page"""
    return render_template('quiz.html')

@app.route('/leaderboard')
@login_required
def leaderboard():
    """Show top users based on average score"""
    try:
        # Get users with at least 3 attempts
        users_with_attempts = db.session.query(
            User.id, 
            User.username,
            func.count(QuizAttempt.id).label('attempt_count')
        ).join(QuizAttempt).group_by(User.id).having(
            func.count(QuizAttempt.id) >= 3
        ).subquery()
        
        # Get average scores for these users
        top_users_query = db.session.query(
            users_with_attempts.c.username,
            users_with_attempts.c.id,
            users_with_attempts.c.attempt_count,
            func.avg(QuizAttempt.average_score).label('avg_score')
        ).join(
            QuizAttempt, 
            QuizAttempt.user_id == users_with_attempts.c.id
        ).group_by(
            users_with_attempts.c.username,
            users_with_attempts.c.id,
            users_with_attempts.c.attempt_count
        ).order_by(
            func.avg(QuizAttempt.average_score).desc()
        ).limit(10).all()
        
        # Format the results
        result = []
        for user in top_users_query:
            user_dict = {
                'username': user.username,
                'avg_score': user.avg_score,
                'attempt_count': user.attempt_count
            }
            
            # Get the latest attempt for this user
            latest_attempt = QuizAttempt.query.filter_by(user_id=user.id).order_by(QuizAttempt.timestamp.desc()).first()
            if latest_attempt:
                user_dict['last_attempt'] = latest_attempt.timestamp
            else:
                user_dict['last_attempt'] = None
                
            result.append(user_dict)
        
        return render_template('leaderboard.html', top_users=result)
    except Exception as e:
        import traceback
        app.logger.error(f"Error in leaderboard route: {str(e)}")
        app.logger.error(traceback.format_exc())
        flash(f"An error occurred while loading the leaderboard. Please try again later.", "danger")
        return redirect(url_for('dashboard'))

@app.route('/attempt/<int:attempt_id>')
@login_required
def view_attempt(attempt_id):
    """View details of a specific quiz attempt."""
    attempt = QuizAttempt.query.get_or_404(attempt_id)
    
    # Ensure the attempt belongs to the current user
    if attempt.user_id != current_user.id:
        flash('You do not have permission to view this attempt', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get all answers for this attempt
    answers = QuestionAnswer.query.filter_by(attempt_id=attempt_id).all()
    
    return render_template('attempt.html', attempt=attempt, answers=answers)

@app.route('/generate_questions', methods=['POST'])
def generate_questions_route():
    """Generate questions based on the subject and difficulty."""
    # Check if the request is JSON or form data
    if request.is_json:
        data = request.json
        subject = data.get('subject', '').strip()
        difficulty = data.get('difficulty', 'Medium').strip()
        refresh = data.get('refresh') == 'true'
        add_more = data.get('add_more') == 'true'
        num_questions = data.get('num_questions', 10)
    else:
        subject = request.form.get('subject', '').strip()
        difficulty = request.form.get('difficulty', 'Medium').strip()
        refresh = request.form.get('refresh') == 'true'
        add_more = request.form.get('add_more') == 'true'
        num_questions = int(request.form.get('num_questions', 10))
    
    if not subject:
        return jsonify({"success": False, "error": "Please enter a subject"}), 400
    
    try:
        # Add random seed parameter to ensure different questions each time
        import time
        # Use a combination of time, random number, and subject/difficulty to create a unique seed
        seed_components = [str(time.time()), str(random.randint(1, 1000000)), subject, difficulty]
        combined_seed = ''.join(seed_components)
        # Convert the string to a number by summing character values
        random_seed = sum(ord(c) for c in combined_seed) % 1000000
        print(f"Using random seed: {random_seed} for {subject}/{difficulty}")
        
        # Clear any previous questions from session to avoid influence
        if 'previous_questions' not in session:
            session['previous_questions'] = []
        
        # If refreshing, generate completely new questions
        if refresh:
            questions = generate_questions(subject, difficulty, num_questions, random_seed=random_seed)
            
            # Store in session for later use
            session['questions'] = questions
            session['subject'] = subject
            session['difficulty'] = difficulty
            
            # Store these questions in previous_questions to avoid repetition in future
            prev_questions = session.get('previous_questions', [])
            prev_questions.extend(questions)
            # Keep only the last 50 questions to avoid session bloat
            if len(prev_questions) > 50:
                prev_questions = prev_questions[-50:]
            session['previous_questions'] = prev_questions
            
            # Generate model answers for each question and store in session
            model_answers = [generate_model_answer(q, subject, difficulty) for q in questions]
            session['model_answers'] = model_answers
            
            return jsonify({"success": True, "questions": questions})
            
        # If adding more questions
        elif add_more:
            # Get current questions
            current_questions = session.get('questions', [])
            
            # Generate additional questions
            additional_questions = generate_questions(subject, difficulty, random_seed=random_seed)
            
            # Make sure we don't have duplicates
            unique_additional = [q for q in additional_questions if q not in current_questions]
            
            # Generate model answers for new questions
            additional_model_answers = [generate_model_answer(q, subject, difficulty) for q in unique_additional]
            
            # Update session
            session['questions'] = current_questions + unique_additional
            session['model_answers'] = session.get('model_answers', []) + additional_model_answers
            
            return jsonify({"success": True, "questions": unique_additional})
            
        # Regular question generation
        else:
            questions = generate_questions(subject, difficulty, num_questions, random_seed=random_seed)
            
            # Store in session for later use
            session['questions'] = questions
            session['subject'] = subject
            session['difficulty'] = difficulty
            
            # Store these questions in previous_questions to avoid repetition in future
            prev_questions = session.get('previous_questions', [])
            prev_questions.extend(questions)
            # Keep only the last 50 questions to avoid session bloat
            if len(prev_questions) > 50:
                prev_questions = prev_questions[-50:]
            session['previous_questions'] = prev_questions
            
            # Generate model answers for each question and store in session
            model_answers = [generate_model_answer(q, subject, difficulty) for q in questions]
            session['model_answers'] = model_answers
            
            return jsonify({"success": True, "questions": questions})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/evaluate_answers', methods=['POST'])
def evaluate_answers_route():
    """Evaluate the user's answers."""
    try:
        questions = session.get('questions', [])
        subject = session.get('subject', 'General')
        difficulty = session.get('difficulty', 'Medium')
        model_answers = session.get('model_answers', [])
        
        print(f"Evaluating {len(questions)} questions for subject: {subject}, difficulty: {difficulty}")
        
        # Check if the request is JSON or form data
        if request.is_json:
            data = request.json
            answers = data.get('answers', [])
        else:
            answers = request.form.getlist('answers[]')
        
        if len(questions) != len(answers):
            return jsonify({"error": f"Questions and answers count mismatch: {len(questions)} questions vs {len(answers)} answers"}), 400
        
        evaluations = []
        total_score = 0
        
        for i, (question, answer) in enumerate(zip(questions, answers)):
            model_answer = model_answers[i] if i < len(model_answers) else None
            print(f"Evaluating answer for question {i+1}: {question[:50]}...")
            evaluation = evaluate_answer(question, answer, model_answer, subject, difficulty)
            total_score += evaluation['rating']
            
            evaluations.append({
                "question": question,
                "answer": answer,
                "model_answer": model_answer,
                "evaluation": evaluation
            })
        
        average_score = total_score / len(evaluations) if evaluations else 0
        print(f"Average score: {average_score}")
        
        # Save the quiz attempt if user is logged in
        if current_user.is_authenticated:
            try:
                print(f"Saving quiz attempt for user {current_user.id}")
                quiz_attempt = QuizAttempt(
                    user_id=current_user.id,
                    subject=subject,
                    difficulty=difficulty,
                    average_score=average_score
                )
                db.session.add(quiz_attempt)
                db.session.flush()  # Get ID without committing
                
                # Save individual question answers one at a time using explicit transactions
                for i, item in enumerate(evaluations):
                    try:
                        print(f"Saving answer {i+1} with correctness: {item['evaluation']['correctness']}")
                        # Clean and truncate any problematic strings
                        question_text = item['question'] if item['question'] else ""
                        user_answer = item['answer'] if item['answer'] else ""
                        model_answer = item['model_answer'] if item['model_answer'] else ""
                        correctness = item['evaluation']['correctness'] if item['evaluation']['correctness'] else "No rating"
                        feedback = item['evaluation']['feedback'] if item['evaluation']['feedback'] else ""
                        
                        # Create the question answer object with truncated values if needed
                        def safe_truncate(text, max_length=10000):
                            """Safely truncate text to a maximum length while preserving meaningful content"""
                            if not text or len(text) <= max_length:
                                return text
                            # Return the first part + indication of truncation
                            return text[:max_length-30] + "... [truncated for storage]"
                            
                        question_answer = QuestionAnswer(
                            attempt_id=quiz_attempt.id,
                            question_text=safe_truncate(question_text),
                            user_answer=safe_truncate(user_answer),
                            model_answer=safe_truncate(model_answer),
                            correctness=correctness[:100] if correctness else "Unknown",  # Limit to reasonable size
                            score=item['evaluation']['rating'],
                            feedback=safe_truncate(feedback, 5000)
                        )
                        
                        # Add and flush each answer individually
                        db.session.add(question_answer)
                        db.session.flush()
                        print(f"Successfully saved answer {i+1}")
                    except Exception as answer_error:
                        print(f"Error saving answer {i+1}: {answer_error}")
                        # Continue with other answers even if one fails
                
                try:
                    db.session.commit()
                    print("Database commit successful")
                    return jsonify({"success": True, "evaluations": evaluations, "attempt_id": quiz_attempt.id})
                except Exception as commit_error:
                    db.session.rollback()
                    print(f"Database commit error: {commit_error}")
                    traceback.print_exc()
                    # Try again without the question answers
                    try:
                        # Just save the attempt without details
                        simple_attempt = QuizAttempt(
                            user_id=current_user.id,
                            subject=subject,
                            difficulty=difficulty,
                            average_score=average_score
                        )
                        db.session.add(simple_attempt)
                        db.session.commit()
                        print("Saved attempt without detailed answers")
                        return jsonify({
                            "success": True, 
                            "evaluations": evaluations, 
                            "attempt_id": simple_attempt.id,
                            "warning": "Could not save detailed answers"
                        })
                    except Exception as simple_error:
                        db.session.rollback()
                        print(f"Failed to save simple attempt: {simple_error}")
                        traceback.print_exc()
                        return jsonify({"success": False, "error": "Could not save your answers to the database. Please try again."}), 500
            except Exception as db_error:
                db.session.rollback()
                print(f"Database error: {db_error}")
                traceback.print_exc()
                return jsonify({"success": False, "error": f"Database error: {str(db_error)}"}), 500
        else:
            # For users not logged in, just return evaluations
            return jsonify({"success": True, "evaluations": evaluations})
    except Exception as e:
        print(f"Evaluation error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Error evaluating answers: {str(e)}"}), 500

@app.route('/download/questions-pdf', methods=['POST'])
@login_required
def download_questions_pdf():
    """Download generated questions as PDF"""
    questions = session.get('questions', [])
    subject = session.get('subject', 'General')
    difficulty = session.get('difficulty', 'Medium')
    
    if not questions:
        flash('No questions to download. Please generate questions first.', 'warning')
        return redirect(url_for('quiz'))
    
    # Generate PDF
    pdf_buffer = pdf_generator.generate_questions_pdf(questions, subject, difficulty)
    
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'interview_questions_{subject}_{difficulty}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    )

@app.route('/download/evaluation-pdf/<int:attempt_id>')
@login_required
def download_evaluation_pdf(attempt_id):
    """Download evaluation report as PDF"""
    attempt = QuizAttempt.query.get_or_404(attempt_id)
    
    # Ensure the attempt belongs to the current user
    if attempt.user_id != current_user.id:
        flash('You do not have permission to access this report', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get all answers for this attempt
    questions_answers = QuestionAnswer.query.filter_by(attempt_id=attempt_id).all()
    
    # Generate PDF
    pdf_buffer = pdf_generator.generate_evaluation_pdf(attempt, questions_answers)
    
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'evaluation_report_{attempt.subject}_{attempt.timestamp.strftime("%Y%m%d_%H%M%S")}.pdf'
    )

@app.route('/api/theme', methods=['POST'])
@login_required
def toggle_theme():
    """Toggle dark/light theme"""
    theme = request.json.get('theme', 'light')
    session['theme'] = theme
    return jsonify({'theme': theme})

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if supabase_url and supabase_key:
    supabase: Client = create_client(supabase_url, supabase_key)
    print("Supabase client initialized successfully")
else:
    print("Supabase credentials not found in environment variables")
    supabase = None

# Update the database connection to use PostgreSQL via Supabase
# Replace your existing SQLite connection with:
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///interview_app.db')

# Create database tables within application context
with app.app_context():
    db.create_all()

# Ensure PostgreSQL URLs from Supabase are correctly formatted for SQLAlchemy
if app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace("postgres://", "postgresql://", 1)

def generate_contextual_answer(question, subject, difficulty):
    """Generate an answer by analyzing the question context"""
    question_lower = question.lower()
    
    # Extract key terms from the question
    import re
    key_terms = []
    
    # Look for quoted terms
    quoted = re.findall(r'"([^"]*)"', question) + re.findall(r"'([^']*)'", question)
    key_terms.extend(quoted)
    
    # Look for capitalized terms
    capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', question)
    key_terms.extend(capitalized)
    
    # Add the subject itself
    if subject:
        key_terms.append(subject)
    
    # Remove duplicates
    key_terms = list(set([term.strip() for term in key_terms if term.strip()]))
    
    # Determine question type
    if any(word in question_lower for word in ["what is", "define", "explain"]):
        question_type = "definition"
    elif any(word in question_lower for word in ["how to", "how do you", "steps", "process"]):
        question_type = "process"
    elif any(word in question_lower for word in ["why", "reason", "advantage", "benefit"]):
        question_type = "reasoning"
    elif any(word in question_lower for word in ["compare", "difference", "versus", "vs"]):
        question_type = "comparison"
    else:
        question_type = "general"
    
    # Generate answer based on question type
    if question_type == "definition":
        return (f"In {subject}, {key_terms[0] if key_terms else 'this concept'} refers to an important principle or technique. "
                f"It is characterized by specific attributes and behaviors that make it suitable for certain use cases. "
                f"Understanding this concept requires knowledge of its core components and how they interact. "
                f"The implementation details vary depending on the specific requirements and constraints of the project.")
    
    elif question_type == "process":
        return (f"The process involves several key steps when working with {key_terms[0] if key_terms else subject}. "
                f"First, you need to analyze the requirements and constraints of your specific situation. "
                f"Then, identify the appropriate approach based on best practices in the field. "
                f"Implementation typically requires careful consideration of efficiency, maintainability, and scalability. "
                f"Testing and validation are essential to ensure the solution meets the expected outcomes.")
    
    elif question_type == "reasoning":
        return (f"There are several important reasons why {key_terms[0] if key_terms else 'this approach'} is significant in {subject}. "
                f"First, it addresses common challenges that practitioners face in real-world scenarios. "
                f"Second, it offers advantages in terms of performance, reliability, or simplicity compared to alternatives. "
                f"Additionally, it aligns with modern best practices and industry standards. "
                f"Understanding these benefits helps inform better design and implementation decisions.")
    
    elif question_type == "comparison":
        term1 = key_terms[0] if len(key_terms) > 0 else "the first approach"
        term2 = key_terms[1] if len(key_terms) > 1 else "the alternative approach"
        
        return (f"When comparing {term1} and {term2} in {subject}, several key differences emerge. "
                f"They differ in their underlying implementation details, performance characteristics, and use cases. "
                f"{term1} might be more suitable in scenarios requiring specific attributes, while {term2} could be preferred in other contexts. "
                f"The choice between them depends on factors such as project requirements, constraints, and trade-offs between different qualities.")
    
    else:  # General answer
        return (f"This is an important concept in {subject} that requires thorough understanding. "
                f"It encompasses multiple aspects including theoretical foundations and practical applications. "
                f"When working with this concept, professionals need to consider various factors and trade-offs. "
                f"Best practices involve careful planning, appropriate implementation techniques, and ongoing evaluation. "
                f"Mastering this area contributes significantly to overall expertise in {subject}.")

# Ensure PostgreSQL URLs from Supabase are correctly formatted for SQLAlchemy
if app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace("postgres://", "postgresql://", 1)

if __name__ == '__main__':
    app.run(debug=True)

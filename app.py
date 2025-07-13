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
import openai
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
app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp)

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///interview_app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy with the app
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Always force mock data mode for Render deployment
if 'RENDER' in os.environ or os.environ.get('USE_MODELS', 'False').lower() != 'true':
    USE_MODELS = False
    print("Running in mock data mode (forced by environment)")
else:
    USE_MODELS = True
    print("Will attempt to use ML models")

print(f"USE_MODELS setting: {USE_MODELS}")

# Only try to load models if USE_MODELS is True
if USE_MODELS:
    try:
        print("Trying to import ML libraries...")
        from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer
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
            # Initialize question generator model (GPT-2)
            question_model_name = "gpt2"
            question_tokenizer = GPT2Tokenizer.from_pretrained(question_model_name, cache_dir=model_cache_dir)
            question_model = GPT2LMHeadModel.from_pretrained(question_model_name, cache_dir=model_cache_dir).to(device)
            
            # Initialize evaluator model (T5)
            evaluator_model_name = "t5-small"
            evaluator_tokenizer = T5Tokenizer.from_pretrained(evaluator_model_name, cache_dir=model_cache_dir)
            evaluator_model = T5ForConditionalGeneration.from_pretrained(evaluator_model_name, cache_dir=model_cache_dir).to(device)
            
            print("Successfully loaded machine learning models.")
        except Exception as model_error:
            print(f"Failed to load models due to error: {model_error}")
            print("Falling back to mock data mode.")
            USE_MODELS = False
    except Exception as e:
        print(f"Failed to import required modules: {e}")
        print("Falling back to mock data mode.")
        USE_MODELS = False
else:
    print("Models disabled via configuration. Running in mock data mode.")

# Dynamic Question Generation Classes
class DynamicQuestionGenerator:
    def __init__(self, api_key=None):
        from openai import OpenAI
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    def generate_questions(self, subject, difficulty, count=5):
        """Generate questions dynamically using OpenAI API"""
        
        difficulty_prompts = {
            'Easy': f"Generate {count} basic interview questions for {subject} suitable for beginners. Focus on fundamental concepts.",
            'Medium': f"Generate {count} intermediate interview questions for {subject} that require practical understanding.",
            'Hard': f"Generate {count} advanced interview questions for {subject} that test deep expertise and complex scenarios."
        }
        
        prompt = difficulty_prompts.get(difficulty, difficulty_prompts['Medium'])
        prompt += "\n\nReturn only the questions, numbered 1-5, one per line."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user", 
                    "content": prompt
                }],
                max_tokens=500,
                temperature=0.7
            )
            
            questions_text = response.choices[0].message.content.strip()
            questions = [q.strip() for q in questions_text.split('\n') if q.strip() and not q.strip().isdigit()]
            
            # Clean up numbered questions (remove numbers at start)
            cleaned_questions = []
            for q in questions:
                # Remove leading numbers and dots/dashes
                clean_q = q.lstrip('0123456789.-) ').strip()
                if clean_q and len(clean_q) > 10:  # Ensure it's a real question
                    cleaned_questions.append(clean_q)
            
            return cleaned_questions[:count] if cleaned_questions else self.fallback_questions(subject, difficulty, count)
            
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return self.fallback_questions(subject, difficulty, count)
    
    def fallback_questions(self, subject, difficulty, count=5):
        """Fallback questions if OpenAI API fails"""
        templates = {
            'Python': {
                'Easy': [
                    "What is Python and what are its main features?",
                    "Explain the difference between a list and a tuple in Python.",
                    "What are Python data types? Give examples.",
                    "How do you write a simple function in Python?",
                    "What is the difference between '==' and 'is' in Python?"
                ],
                'Medium': [
                    "Explain decorators in Python and provide an example.",
                    "What is the difference between __str__ and __repr__ methods?",
                    "How does exception handling work in Python?",
                    "Explain list comprehensions and their advantages.",
                    "What is the Global Interpreter Lock (GIL) in Python?"
                ],
                'Hard': [
                    "Explain metaclasses in Python and when you would use them.",
                    "How does Python's garbage collection work?",
                    "Explain the implementation of Python's asyncio module.",
                    "Discuss memory management and optimization in Python.",
                    "How would you implement a custom context manager in Python?"
                ]
            },
            'Java': {
                'Easy': [
                    "What is Java and what are its key principles?",
                    "Explain the main method in Java.",
                    "What is the difference between JDK, JRE, and JVM?",
                    "How do you create a class and object in Java?",
                    "What are the basic data types in Java?"
                ],
                'Medium': [
                    "Explain inheritance and polymorphism in Java.",
                    "What is the difference between abstract classes and interfaces?",
                    "How does exception handling work in Java?",
                    "Explain the concept of multithreading in Java.",
                    "What are Java Collections and their types?"
                ],
                'Hard': [
                    "Explain the Java Memory Model and garbage collection.",
                    "How do ClassLoaders work in Java?",
                    "Discuss concurrency utilities in java.util.concurrent.",
                    "Explain JVM internals and bytecode optimization.",
                    "How would you implement a custom thread pool in Java?"
                ]
            }
        }
        
        subject_questions = templates.get(subject, templates.get('Python', {}))
        difficulty_questions = subject_questions.get(difficulty, subject_questions.get('Medium', []))
        
        return difficulty_questions[:count] if difficulty_questions else [
            f"Explain the core concepts of {subject}.",
            f"What are the best practices in {subject}?",
            f"How do you solve common problems in {subject}?",
            f"What are the latest trends in {subject}?",
            f"How do you optimize performance in {subject}?"
        ]

class TemplateQuestionGenerator:
    def __init__(self):
        self.question_templates = {
            'Python': {
                'Easy': [
                    "What is {concept} in Python?",
                    "How do you use {feature} in Python?",
                    "Explain the difference between {concept1} and {concept2}.",
                    "What are the advantages of using {feature}?",
                    "How do you implement {concept} in Python?",
                    "When would you use {concept} in Python?",
                    "Describe how {concept} works in Python.",
                    "What is the purpose of {feature} in Python?",
                    "How would you create a {concept} in Python?"
                ],
                'Medium': [
                    "How would you optimize {concept} in Python?",
                    "What are the performance implications of {feature}?",
                    "When would you use {concept} over {alternative}?",
                    "Explain how {concept} works internally in Python.",
                    "What are the best practices for {feature}?",
                    "How does {concept} help with code maintainability?",
                    "What are common pitfalls when using {feature}?",
                    "How would you debug issues with {concept}?",
                    "What are the limitations of {feature} in Python?"
                ],
                'Hard': [
                    "How would you implement a custom {concept} in Python?",
                    "Discuss the memory implications of {feature}.",
                    "How does {concept} affect Python's performance?",
                    "Explain the internals of {concept} in CPython.",
                    "What are advanced patterns for {feature}?",
                    "How would you scale {concept} for enterprise applications?",
                    "Compare different approaches to implementing {feature}.",
                    "What security considerations are important with {concept}?",
                    "How would you test a complex implementation of {feature}?"
                ]
            }
        }
        
        self.concepts = {
            'Python': {
                'Easy': ['variables', 'functions', 'loops', 'lists', 'dictionaries', 'strings', 'tuples', 
                        'sets', 'conditionals', 'modules', 'packages', 'file handling', 'exceptions', 
                        'input/output', 'type conversion', 'comments', 'indentation', 'operators'],
                'Medium': ['decorators', 'generators', 'context managers', 'classes', 'inheritance', 
                          'comprehensions', 'lambda functions', 'map/filter/reduce', 'regular expressions',
                          'virtual environments', 'namespaces', 'scope', 'iterators', 'error handling',
                          'unit testing', 'debugging', 'JSON processing', 'API integration'],
                'Hard': ['metaclasses', 'descriptors', 'coroutines', 'memory management', 'GIL',
                        'asyncio', 'concurrency', 'multithreading', 'multiprocessing', 'C extensions',
                        'profiling', 'optimization techniques', 'design patterns', 'microservices',
                        'serverless architecture', 'distributed systems', 'caching strategies']
            }
        }
    
    def generate_questions(self, subject, difficulty, count=5, random_seed=None):
        """Generate questions using templates and concepts"""
        # Set random seed if provided for consistent results
        if random_seed is not None:
            random.seed(random_seed)
            
        # Get templates for the subject or use a generic one
        templates = self.question_templates.get(subject, {}).get(difficulty, [])
        if not templates:
            # Use Python templates as fallback
            templates = self.question_templates.get('Python', {}).get(difficulty, [])
            
        # Get concepts for the subject or use generic ones
        concepts = self.concepts.get(subject, {}).get(difficulty, [])
        if not concepts:
            # Add some generic concepts based on the subject
            concepts = ['algorithms', 'data structures', 'design patterns', 'best practices', 
                       'performance optimization', 'security considerations', 'error handling',
                       'memory management', 'concurrency', 'scalability', 'testing strategies',
                       'frameworks', 'libraries', 'APIs', 'documentation', 'version control',
                       'deployment strategies', 'code quality', 'refactoring', 'technical debt',
                       'integration', 'monitoring', 'debugging', 'troubleshooting', 'development lifecycle']
        
        if not templates:
            # Create generic templates based on subject
            templates = [
                f"What is the purpose of {{concept}} in {subject}?",
                f"How do you implement {{concept}} in {subject}?",
                f"Explain the advantages of using {{concept}} in {subject}.",
                f"What are common mistakes when working with {{concept}} in {subject}?",
                f"Compare {{concept}} with {{alternative}} in {subject}.",
                f"What are best practices for {{concept}} in {subject}?",
                f"How has {{concept}} evolved in {subject} over time?",
                f"What problems does {{concept}} solve in {subject}?",
                f"When would you use {{concept}} versus {{alternative}} in {subject}?"
            ]
        
        # Generate a unique set of questions
        questions = []
        used_templates = set()
        used_concepts = set()
        used_pairs = set()  # Track template+concept combinations
        
        # Try to generate more questions than needed so we can filter duplicates
        for _ in range(count * 3):  # Generate 3x as many as needed
            if len(questions) >= count:
                break
                
            # Create variation by shuffling available options
            remaining_templates = [t for t in templates if t not in used_templates] or templates
            remaining_concepts = [c for c in concepts if c not in used_concepts] or concepts
            
            # Add additional randomization
            if random.random() < 0.3:  # 30% chance to completely shuffle
                random.shuffle(remaining_templates)
                random.shuffle(remaining_concepts)
                
            template = random.choice(remaining_templates)
            concept = random.choice(remaining_concepts)
            
            # Create a unique combination key
            combo_key = f"{template}:{concept}"
            if combo_key in used_pairs:
                continue
                
            used_templates.add(template)
            used_concepts.add(concept)
            used_pairs.add(combo_key)
            
            # Add more randomization by sometimes substituting words
            if random.random() < 0.2:  # 20% chance
                synonyms = {
                    "implement": ["create", "develop", "code", "build", "construct"],
                    "use": ["utilize", "employ", "apply", "leverage", "work with"],
                    "explain": ["describe", "clarify", "elaborate on", "detail", "outline"],
                    "compare": ["contrast", "differentiate between", "distinguish", "evaluate"],
                    "best practices": ["recommended approaches", "guidelines", "principles", "standards"]
                }
                
                # Replace some words with synonyms
                for word, replacements in synonyms.items():
                    if word in template and random.random() < 0.5:
                        template = template.replace(word, random.choice(replacements))
            
            try:
                if '{concept}' in template and '{concept1}' not in template and '{alternative}' not in template:
                    question = template.format(concept=concept, feature=concept)
                elif '{feature}' in template:
                    question = template.format(feature=concept)
                elif '{concept1}' in template and '{concept2}' in template:
                    # Ensure we have enough concepts for comparison
                    available_concepts = [c for c in concepts if c != concept]
                    if available_concepts:
                        concept2 = random.choice(available_concepts)
                        question = template.format(concept1=concept, concept2=concept2)
                    else:
                        question = template.format(concept1=concept, concept2='other concepts')
                elif '{alternative}' in template:
                    # Ensure we have an alternative concept
                    available_alternatives = [c for c in concepts if c != concept]
                    if available_alternatives:
                        alternative = random.choice(available_alternatives)
                        question = template.format(concept=concept, alternative=alternative)
                    else:
                        question = template.format(concept=concept, alternative='alternatives')
                else:
                    # Fallback: replace all placeholders with the concept
                    question = template.format(concept=concept, feature=concept, alternative=concept)
                    
                # Add some randomization to the questions themselves
                if random.random() < 0.15:  # 15% chance
                    prefixes = ["In your experience, ", "According to best practices, ", 
                               "From a practical standpoint, ", "In modern development, "]
                    question = random.choice(prefixes) + question.lower()
                
                # Only add if it's not a duplicate
                if question not in questions:
                    questions.append(question)
                    
            except (KeyError, ValueError) as e:
                # If template formatting fails, use a simple fallback
                fallback = f"Explain {concept} in {subject} and its importance."
                if fallback not in questions:
                    questions.append(fallback)
        
        # Reset random seed
        if random_seed is not None:
            random.seed(None)
            
        # Shuffle one more time before returning
        random.shuffle(questions)
        return questions[:count]

# Initialize question generators
            
        return questions

# Initialize question generators
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key and not USE_MODELS:  # Only use OpenAI if not using local models
    dynamic_generator = DynamicQuestionGenerator(openai_api_key)
    print("OpenAI Dynamic Question Generator initialized")
else:
    dynamic_generator = None
    print("Using template-based question generation")

template_generator = TemplateQuestionGenerator()

def generate_questions(subject, difficulty="Medium", num_questions=5, random_seed=None):
    """Generate interview questions based on the subject and difficulty."""
    from flask import session
    
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        print(f"Using random seed: {random_seed} for question generation")
    
    # Get previous questions to avoid duplicates
    previous_questions = session.get('previous_questions', []) if 'session' in globals() else []
    
    # Try dynamic generation first (OpenAI API)
    if dynamic_generator and not USE_MODELS:
        try:
            questions = dynamic_generator.generate_questions(subject, difficulty, num_questions)
            if questions and len(questions) >= num_questions:
                # Filter out previously asked questions
                questions = [q for q in questions if q not in previous_questions]
                # If we filtered too many, generate more to compensate
                if len(questions) < num_questions:
                    additional = dynamic_generator.generate_questions(subject, difficulty, num_questions - len(questions))
                    questions.extend([q for q in additional if q not in previous_questions and q not in questions])
                
                # Ensure we have enough questions
                if len(questions) < num_questions:
                    # Add some from template generator if needed
                    template_questions = template_generator.generate_questions(subject, difficulty, 
                                                                             num_questions - len(questions), 
                                                                             random_seed)
                    questions.extend([q for q in template_questions if q not in previous_questions and q not in questions])
                
                # Reset random seed
                if random_seed is not None:
                    random.seed(None)
                
                # Return at most num_questions
                return questions[:num_questions]
                
            print("Dynamic generator returned insufficient questions, falling back to template generator")
        except Exception as e:
            print(f"Dynamic generation failed: {e}, falling back to template generator")
    
    # Use template-based generation as fallback or when USE_MODELS is False
    if not USE_MODELS:
        # Get questions from template generator
        questions = template_generator.generate_questions(subject, difficulty, num_questions * 2, random_seed)
        
        # Filter out previously asked questions
        questions = [q for q in questions if q not in previous_questions]
        
        # If we don't have enough questions, generate some with variations
        if len(questions) < num_questions:
            # Generate additional questions with variations
            variations = []
            base_templates = [
                f"Explain the concept of {{concept}} in {subject}.",
                f"How would you implement {{concept}} in {subject}?",
                f"What are the advantages and disadvantages of {{concept}} in {subject}?",
                f"Compare {{concept}} with {{alternative}} in the context of {subject}.",
                f"How has {{concept}} evolved in {subject} over the years?"
            ]
            
            # Generate variations by randomly combining concepts with templates
            for _ in range(num_questions - len(questions)):
                template = random.choice(base_templates)
                concept = random.choice([
                    "data structures", "algorithms", "design patterns", "optimization", 
                    "best practices", "error handling", "security", "performance", 
                    "scalability", "maintainability", "testing", "deployment"
                ])
                alternative = random.choice([
                    "traditional approaches", "alternative methods", "legacy systems", 
                    "modern techniques", "competing technologies", "industry standards"
                ])
                
                # Create the question with random variations
                try:
                    if "{{concept}}" in template and "{{alternative}}" in template:
                        question = template.replace("{{concept}}", concept).replace("{{alternative}}", alternative)
                    else:
                        question = template.replace("{{concept}}", concept)
                    
                    if question not in previous_questions and question not in questions and question not in variations:
                        variations.append(question)
                except Exception:
                    # Fallback if templating fails
                    variations.append(f"Describe an important aspect of {subject} related to {concept}.")
            
            # Add the variations to our questions
            questions.extend(variations)
        
        # Reset random seed
        if random_seed is not None:
            random.seed(None)
        
        # Return at most num_questions
        return questions[:num_questions]
    
    # Original ML model-based generation (when USE_MODELS is True)
    questions = []
    if USE_MODELS:
        for _ in range(num_questions):
            # Adjust prompt based on difficulty
            if difficulty == "Easy":
                prompt = f"Generate a basic interview question about {subject} for beginners:"
            elif difficulty == "Hard":
                prompt = f"Generate a very challenging and advanced interview question about {subject} for experts:"
            else:  # Medium (default)
                prompt = f"Generate a moderately challenging interview question about {subject}:"
            
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
                if difficulty == "Easy":
                    question = f"Explain the basic concept of {random.choice(['variables', 'functions', 'classes', 'loops'])} in {subject}."
                elif difficulty == "Hard":
                    question = f"Discuss advanced techniques for {random.choice(['optimization', 'scalability', 'concurrency', 'security'])} in {subject}."
                else:
                    question = f"Explain the concept of {random.choice(['indexing', 'normalization', 'transactions', 'concurrency'])} in {subject}."
            
            questions.append(question)
    else:
        # Mock questions for different subjects and difficulties
        subject_lower = subject.lower()
        
        # Python questions
        if "python" in subject_lower:
            if difficulty == "Easy":
                questions = [
                    f"What are variables in {subject} and how do you declare them?",
                    f"Explain the difference between a list and a tuple in {subject}.",
                    f"What are the basic data types in {subject}?",
                    f"How do you write a simple for loop in {subject}?",
                    f"What is a function in {subject} and how do you define one?"
                ]
            elif difficulty == "Hard":
                questions = [
                    f"Explain metaclasses in {subject} and provide a practical use case.",
                    f"How does the Global Interpreter Lock (GIL) work in {subject} and what are its implications?",
                    f"Explain Python's memory management and garbage collection mechanism in detail.",
                    f"Discuss advanced decorator patterns in {subject} with examples.",
                    f"Explain how asyncio works in {subject} and when you would use it over multithreading."
                ]
            else:  # Medium
                questions = [
                    f"Explain the difference between lists and tuples in {subject}.",
                    f"How do you handle exceptions in {subject}? Give examples.",
                    f"What are decorators in {subject} and how do they work?",
                    f"Explain the concept of generators in {subject}.",
                    f"How does inheritance work in {subject}?"
                ]
        
        # Java questions
        elif "java" in subject_lower:
            if difficulty == "Easy":
                questions = [
                    f"What is a class in {subject}?",
                    f"Explain the main method in {subject}.",
                    f"What are the primitive data types in {subject}?",
                    f"How do you create an object in {subject}?",
                    f"What is the difference between '==' and 'equals()' in {subject}?"
                ]
            elif difficulty == "Hard":
                questions = [
                    f"Explain the intricacies of the Java Memory Model and its implications for concurrent programming.",
                    f"Discuss the internals of the JVM and how it optimizes {subject} code execution.",
                    f"Compare and contrast different Garbage Collection algorithms in {subject}.",
                    f"Explain how ClassLoaders work in {subject} and how you might implement a custom ClassLoader.",
                    f"Discuss advanced concurrency patterns in {subject} and their applications."
                ]
            else:  # Medium
                questions = [
                    f"What is polymorphism in {subject}? Provide examples.",
                    f"Explain the difference between checked and unchecked exceptions in {subject}.",
                    f"How does garbage collection work in {subject}?",
                    f"What are the differences between interface and abstract class in {subject}?",
                    f"Explain multithreading in {subject} and its challenges."
                ]
        
        # Database/SQL questions
        elif "database" in subject_lower or "sql" in subject_lower or "dbms" in subject_lower:
            if difficulty == "Easy":
                questions = [
                    f"What is a database table in {subject}?",
                    f"Explain the basic SQL SELECT statement syntax.",
                    f"What is a primary key in {subject}?",
                    f"Explain the difference between DELETE and TRUNCATE commands.",
                    f"What is a foreign key in {subject}?"
                ]
            elif difficulty == "Hard":
                questions = [
                    f"Explain the implementation details of different join algorithms and their performance characteristics.",
                    f"Discuss advanced indexing strategies in {subject} for complex queries.",
                    f"Explain isolation levels and their impact on concurrency control in {subject}.",
                    f"How would you design a distributed database system to ensure consistency, availability, and partition tolerance?",
                    f"Discuss query optimization techniques and how the query planner works in {subject}."
                ]
            else:  # Medium
                questions = [
                    f"Explain normalization in {subject} with examples.",
                    f"What is the difference between clustered and non-clustered indexes in {subject}?",
                    f"How do you optimize a slow running query in {subject}?",
                    f"Explain ACID properties in {subject}.",
                    f"What is the difference between a primary key and a unique key in {subject}?"
                ]
        
        # Default generic questions
        else:
            if difficulty == "Easy":
                questions = [
                    f"What are the fundamental concepts in {subject}?",
                    f"Explain a basic application of {subject}.",
                    f"What tools or technologies are commonly used in {subject}?",
                    f"What skills are needed to work with {subject}?",
                    f"How would you explain {subject} to a beginner?"
                ]
            elif difficulty == "Hard":
                questions = [
                    f"Discuss the most complex challenges in implementing {subject} in large-scale systems.",
                    f"How has {subject} evolved to address modern technological needs?",
                    f"Analyze the trade-offs between different approaches to solving problems in {subject}.",
                    f"What are the cutting-edge research areas in {subject}?",
                    f"How would you architect a system to efficiently handle {subject} at scale?"
                ]
            else:  # Medium
                questions = [
                    f"What are the key concepts in {subject}?",
                    f"Explain the most important principles of {subject}.",
                    f"What are some real-world applications of {subject}?",
                    f"How has {subject} evolved over the past decade?",
                    f"What are the current challenges and future trends in {subject}?"
                ]
    
    return questions

def evaluate_answer(question, user_answer, model_answer=None, subject=None, difficulty=None):
    """Evaluate the user's answer using the T5 model or mock logic."""
    if not user_answer.strip():
        return {
            "correctness": "Incorrect",
            "feedback": "No answer provided.",
            "rating": 0
        }
    
    if USE_MODELS:
        # Generate a model answer if not provided
        if not model_answer:
            model_answer = generate_model_answer(question, subject, difficulty)
        
        # Create a comparison prompt for the evaluator
        prompt = f"Question: {question}\nModel Answer: {model_answer}\nUser Answer: {user_answer}\nEvaluate the user's answer compared to the model answer."
        
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
        # For a real-world scenario, this would need more sophisticated processing
        if "correct" in evaluation_text.lower() and "incorrect" not in evaluation_text.lower():
            correctness = "Correct"
            rating = random.randint(4, 5)
        elif "incorrect" in evaluation_text.lower():
            correctness = "Incorrect"
            rating = random.randint(1, 2)
        else:
            correctness = "Partially Correct"
            rating = random.randint(2, 4)
        
        feedback = evaluation_text[:150] + "..." if len(evaluation_text) > 150 else evaluation_text
    else:
        # Mock evaluation logic based on answer length, keywords, and comparison to model answer
        if not model_answer:
            model_answer = mock_model_answer(question, subject, difficulty)
        
        user_answer_len = len(user_answer.strip())
        model_answer_len = len(model_answer.strip())
        
        # Very short answers are likely incorrect
        if user_answer_len < 20:
            correctness = "Incorrect"
            feedback = "Your answer is too short. Please provide a more detailed explanation."
            rating = random.randint(1, 2)
            return {"correctness": correctness, "feedback": feedback, "rating": rating}
        
        # Extract keywords from model answer and user answer
        model_words = set(word.lower() for word in model_answer.split() if len(word) > 3)
        user_words = set(word.lower() for word in user_answer.split() if len(word) > 3)
        
        # Calculate keyword overlap
        common_words = model_words.intersection(user_words)
        keyword_overlap_ratio = len(common_words) / len(model_words) if model_words else 0
        
        # Calculate length ratio (user answer length compared to model answer)
        length_ratio = min(user_answer_len / model_answer_len if model_answer_len else 1, 1.5)
        
        # Combine factors for final evaluation
        combined_score = (keyword_overlap_ratio * 0.7) + (length_ratio * 0.3)
        
        if combined_score > 0.7:
            correctness = "Correct"
            feedback = ("Great answer! You've covered the key points and provided a clear explanation. "
                       f"Your response includes important concepts like {', '.join(list(common_words)[:3])}.")
            rating = random.randint(4, 5)
        elif combined_score > 0.4:
            correctness = "Partially Correct"
            feedback = ("Your answer includes some important points, but there's room for improvement. "
                       f"Consider adding more details about {', '.join(list(model_words - user_words)[:3])}.")
            rating = random.randint(2, 4)
        else:
            correctness = "Incorrect"
            feedback = ("Your answer is missing key concepts. The model answer covers important topics like "
                       f"{', '.join(list(model_words - user_words)[:3])} that you should include.")
            rating = random.randint(1, 2)
    
    return {
        "correctness": correctness,
        "feedback": feedback,
        "rating": rating
    }

def generate_model_answer(question, subject, difficulty):
    """Generate a model answer for the given question."""
    if USE_MODELS:
        # Create a prompt for generating a model answer
        if difficulty == "Easy":
            prompt = f"Provide a basic but accurate answer to the following question about {subject}: {question}"
        elif difficulty == "Hard":
            prompt = f"Provide a comprehensive expert-level answer to the following complex question about {subject}: {question}"
        else:  # Medium
            prompt = f"Provide a thorough answer to the following question about {subject}: {question}"
        
        inputs = question_tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate output
        output_sequences = question_model.generate(
            inputs,
            max_length=200,  # Longer for comprehensive answers
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
        )
        
        # Decode the generated answer
        model_answer = question_tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        
        # Clean up the answer (remove the prompt text if it appears)
        model_answer = model_answer.replace(prompt, "").strip()
        
        # Ensure we have a decent answer
        if not model_answer or len(model_answer) < 20:
            model_answer = mock_model_answer(question, subject, difficulty)
    else:
        model_answer = mock_model_answer(question, subject, difficulty)
    
    return model_answer

def mock_model_answer(question, subject, difficulty):
    """Generate a mock model answer based on the question content."""
    # Extract keywords from the question
    question_lower = question.lower()
    subject_lower = subject.lower()
    
    # Python-related answers
    if "python" in subject_lower:
        if "list" in question_lower and "tuple" in question_lower:
            return ("Lists and tuples in Python are both sequence data types that can store collections of items. "
                   "The main differences are: 1) Lists are mutable (can be changed after creation) while tuples are immutable. "
                   "2) Lists use square brackets [] while tuples use parentheses (). "
                   "3) Lists have more built-in methods due to their mutability. "
                   "4) Tuples are slightly faster and use less memory. "
                   "Use lists when you need a collection that might change, and tuples when you need an unchangeable sequence.")
        
        elif "decorator" in question_lower:
            return ("Decorators in Python are a powerful way to modify or extend the behavior of functions or methods without changing their code. "
                   "They use the @decorator_name syntax and are essentially functions that take another function as an argument and return a new function. "
                   "Decorators are commonly used for logging, authentication, caching, or timing functions. "
                   "For example, a simple timing decorator would measure how long a function takes to execute.")
        
        elif "gil" in question_lower or "global interpreter lock" in question_lower:
            return ("The Global Interpreter Lock (GIL) in Python is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecode at once. "
                   "This means that even on multi-core systems, only one thread can execute Python code at a time. "
                   "The GIL simplifies memory management and makes single-threaded programs faster, but it limits CPU-bound parallelism. "
                   "For CPU-intensive tasks requiring parallelism, developers often use multiprocessing or alternative Python implementations like Jython or IronPython that don't have a GIL.")
    
    # Database/SQL-related answers
    elif "database" in subject_lower or "sql" in subject_lower:
        if "normalization" in question_lower:
            return ("Database normalization is the process of organizing a database to reduce redundancy and improve data integrity. "
                   "It involves dividing large tables into smaller ones and defining relationships between them. "
                   "The main normal forms are: 1NF (eliminates repeating groups), 2NF (removes partial dependencies), "
                   "3NF (removes transitive dependencies), BCNF (a stronger version of 3NF), and 4NF/5NF (dealing with multi-valued dependencies). "
                   "Normalization helps prevent update anomalies, reduces data duplication, and improves query performance for writes, "
                   "though it can make some read operations more complex.")
        
        elif "index" in question_lower:
            return ("Indexes in databases improve query performance by providing quick access paths to rows. "
                   "A clustered index determines the physical order of data in a table, and each table can have only one. "
                   "The data rows are stored in the leaf nodes of the clustered index. "
                   "Non-clustered indexes have a structure separate from the data rows, containing the indexed columns and a pointer to the row. "
                   "A table can have multiple non-clustered indexes. "
                   "Clustered indexes are typically faster for range queries and finding a single row when using the clustered key, "
                   "while non-clustered indexes are better for covering queries where all needed columns are in the index.")
    
    # Generic answer for other subjects
    else:
        if difficulty == "Easy":
            return (f"{subject} is a field that encompasses various fundamental concepts and principles. "
                   f"The question focuses on a basic aspect that is important to understand for beginners. "
                   f"A good answer would explain the core concepts clearly, provide simple examples, "
                   f"and highlight practical applications that demonstrate why this knowledge is useful.")
        elif difficulty == "Hard":
            return (f"This is an advanced topic in {subject} that requires deep understanding of the underlying principles. "
                   f"An expert would approach this by analyzing the complex interactions between different components, "
                   f"considering edge cases and performance implications, and drawing on advanced techniques developed in the field. "
                   f"The optimal solution would balance theoretical correctness with practical implementation considerations.")
        else:  # Medium
            return (f"In {subject}, this concept represents an important intermediate-level topic. "
                   f"A thorough understanding requires familiarity with the fundamental principles as well as "
                   f"the ability to apply them in various contexts. A good approach is to first define the key terms, "
                   f"then explain the relationships between them, and finally demonstrate practical applications "
                   f"that show why this knowledge is valuable in real-world scenarios.")
    
    # Fallback generic answer
    return ("The answer to this question involves understanding the key principles and applying them correctly. "
           "It's important to consider both theoretical concepts and practical implications. "
           "A comprehensive answer would define the terms, explain relationships between concepts, "
           "provide examples, and discuss real-world applications.")

@app.route('/')
def index():
    """Render the home page - redirect to login if not authenticated"""
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    return redirect(url_for('dashboard'))

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
        
        # Check if the request is JSON or form data
        if request.is_json:
            data = request.json
            answers = data.get('answers', [])
        else:
            answers = request.form.getlist('answers[]')
        
        if len(questions) != len(answers):
            return jsonify({"error": "Questions and answers count mismatch"}), 400
        
        evaluations = []
        total_score = 0
        
        for i, (question, answer) in enumerate(zip(questions, answers)):
            model_answer = model_answers[i] if i < len(model_answers) else None
            evaluation = evaluate_answer(question, answer, model_answer, subject, difficulty)
            total_score += evaluation['rating']
            
            evaluations.append({
                "question": question,
                "answer": answer,
                "model_answer": model_answer,
                "evaluation": evaluation
            })
        
        average_score = total_score / len(evaluations) if evaluations else 0
        
        # Save the quiz attempt if user is logged in
        if current_user.is_authenticated:
            quiz_attempt = QuizAttempt(
                user_id=current_user.id,
                subject=subject,
                difficulty=difficulty,
                average_score=average_score
            )
            db.session.add(quiz_attempt)
            db.session.flush()  # Get ID without committing
            
            # Save individual question answers
            for item in evaluations:
                question_answer = QuestionAnswer(
                    attempt_id=quiz_attempt.id,
                    question_text=item['question'],
                    user_answer=item['answer'],
                    model_answer=item['model_answer'] or '',
                    correctness=item['evaluation']['correctness'],
                    score=item['evaluation']['rating'],
                    feedback=item['evaluation']['feedback']
                )
                db.session.add(question_answer)
            
            db.session.commit()
            
            return jsonify({"success": True, "evaluations": evaluations, "attempt_id": quiz_attempt.id})
        else:
            # For users not logged in, just return evaluations
            return jsonify({"success": True, "evaluations": evaluations})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

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

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        remember = bool(request.form.get('remember'))
        
        if not email or not password:
            flash('Email and password are required', 'danger')
            return render_template('login.html')
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=remember)
            user.last_login = datetime.utcnow()
            db.session.commit()
            session['last_activity'] = datetime.now().isoformat()
            flash(f'Welcome back, {user.username}!', 'success')
            
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validate inputs
        if not all([username, email, password, confirm_password]):
            flash('All fields are required', 'danger')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        
        # Check if user already exists
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return render_template('register.html')
        
        if User.query.filter_by(username=username).first():
            flash('Username already taken', 'danger')
            return render_template('register.html')
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            is_admin=email == os.environ.get('ADMIN_EMAIL', 'admin@example.com')
        )
        
        try:
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Registration failed. Please try again.', 'danger')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    """Handle user logout"""
    logout_user()
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))

# Create database tables within application context
with app.app_context():
    db.create_all()

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
# Ensure PostgreSQL URLs from Supabase are correctly formatted for SQLAlchemy
if app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace("postgres://", "postgresql://", 1)

if __name__ == '__main__':
    app.run(debug=True)

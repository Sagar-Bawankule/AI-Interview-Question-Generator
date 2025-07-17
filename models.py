from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    date_registered = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    attempts = db.relationship('QuizAttempt', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        """Check password"""
        return check_password_hash(self.password_hash, password)
    
    def get_average_score(self):
        """Calculate user's average score"""
        if self.attempts.count() == 0:
            return 0
        return sum(attempt.average_score for attempt in self.attempts) / self.attempts.count()
    
    def get_total_questions(self):
        """Get total questions answered"""
        return sum(attempt.answers.count() for attempt in self.attempts)
    
    def __repr__(self):
        return f'<User {self.username}>'

class QuizAttempt(db.Model):
    __tablename__ = 'quiz_attempt'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    difficulty = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    average_score = db.Column(db.Float, nullable=False)
    
    # Relationships
    answers = db.relationship('QuestionAnswer', backref='attempt', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<QuizAttempt {self.subject} {self.difficulty} by User {self.user_id}>'

class QuestionAnswer(db.Model):
    __tablename__ = 'question_answer'
    
    id = db.Column(db.Integer, primary_key=True)
    attempt_id = db.Column(db.Integer, db.ForeignKey('quiz_attempt.id'), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    user_answer = db.Column(db.Text, nullable=False)
    model_answer = db.Column(db.Text, nullable=False)
    correctness = db.Column(db.Text, nullable=False)  # Changed from String(20) to Text
    score = db.Column(db.Integer, nullable=False)  # 0-5
    feedback = db.Column(db.Text, nullable=False)
    
    def __repr__(self):
        return f'<QuestionAnswer {self.id} Score: {self.score}>'

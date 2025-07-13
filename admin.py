"""
Admin panel for the AI Interview Question Generator
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_login import login_required, current_user
from functools import wraps
from models import db, User, QuizAttempt, QuestionAnswer
from datetime import datetime
import csv
import io
import zipfile
from sqlalchemy import func

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

def admin_required(f):
    """Decorator to require admin access"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('Admin access required', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

@admin_bp.route('/')
@login_required
@admin_required
def dashboard():
    """Admin dashboard with statistics"""
    # Get overall statistics
    total_users = User.query.count()
    total_attempts = QuizAttempt.query.count()
    total_questions = QuestionAnswer.query.count()
    
    # Get recent activity
    recent_attempts = QuizAttempt.query.order_by(QuizAttempt.timestamp.desc()).limit(10).all()
    
    # Get top performers
    top_users = db.session.query(
        User.username, 
        func.avg(QuizAttempt.average_score).label('avg_score'),
        func.count(QuizAttempt.id).label('attempt_count')
    ).join(QuizAttempt).group_by(User.id).order_by(func.avg(QuizAttempt.average_score).desc()).limit(10).all()
    
    # Subject popularity
    subject_stats = db.session.query(
        QuizAttempt.subject,
        func.count(QuizAttempt.id).label('count')
    ).group_by(QuizAttempt.subject).order_by(func.count(QuizAttempt.id).desc()).all()
    
    stats = {
        'total_users': total_users,
        'total_attempts': total_attempts,
        'total_questions': total_questions,
        'recent_attempts': recent_attempts,
        'top_users': top_users,
        'subject_stats': subject_stats
    }
    
    return render_template('admin/dashboard.html', stats=stats)

@admin_bp.route('/users')
@login_required
@admin_required
def users():
    """View all users"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    users = User.query.paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('admin/users.html', users=users)

@admin_bp.route('/users/<int:user_id>')
@login_required
@admin_required
def user_detail(user_id):
    """View user details"""
    user = User.query.get_or_404(user_id)
    attempts = user.attempts.order_by(QuizAttempt.timestamp.desc()).all()
    
    return render_template('admin/user_detail.html', user=user, attempts=attempts)

@admin_bp.route('/attempts')
@login_required
@admin_required
def attempts():
    """View all quiz attempts"""
    page = request.args.get('page', 1, type=int)
    subject_filter = request.args.get('subject')
    difficulty_filter = request.args.get('difficulty')
    
    query = QuizAttempt.query
    
    if subject_filter:
        query = query.filter(QuizAttempt.subject.ilike(f'%{subject_filter}%'))
    if difficulty_filter:
        query = query.filter(QuizAttempt.difficulty == difficulty_filter)
    
    attempts = query.order_by(QuizAttempt.timestamp.desc()).paginate(
        page=page, per_page=20, error_out=False
    )
    
    # Get unique subjects and difficulties for filters
    subjects = db.session.query(QuizAttempt.subject).distinct().all()
    difficulties = db.session.query(QuizAttempt.difficulty).distinct().all()
    
    return render_template('admin/attempts.html', 
                         attempts=attempts, 
                         subjects=[s[0] for s in subjects],
                         difficulties=[d[0] for d in difficulties],
                         current_subject=subject_filter,
                         current_difficulty=difficulty_filter)

@admin_bp.route('/download/data')
@login_required
@admin_required
def download_data():
    """Download all data as CSV files in a ZIP"""
    # Create in-memory ZIP file
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Users CSV
        users_data = io.StringIO()
        users_writer = csv.writer(users_data)
        users_writer.writerow(['ID', 'Username', 'Email', 'Is Admin', 'Date Registered', 'Total Attempts', 'Average Score'])
        
        for user in User.query.all():
            users_writer.writerow([
                user.id, user.username, user.email, user.is_admin,
                user.date_registered, user.attempts.count(), user.get_average_score()
            ])
        
        zip_file.writestr('users.csv', users_data.getvalue())
        
        # Quiz Attempts CSV
        attempts_data = io.StringIO()
        attempts_writer = csv.writer(attempts_data)
        attempts_writer.writerow(['ID', 'User ID', 'Username', 'Subject', 'Difficulty', 'Timestamp', 'Average Score', 'Questions Count'])
        
        for attempt in QuizAttempt.query.all():
            attempts_writer.writerow([
                attempt.id, attempt.user_id, attempt.user.username,
                attempt.subject, attempt.difficulty, attempt.timestamp,
                attempt.average_score, attempt.answers.count()
            ])
        
        zip_file.writestr('quiz_attempts.csv', attempts_data.getvalue())
        
        # Question Answers CSV
        answers_data = io.StringIO()
        answers_writer = csv.writer(answers_data)
        answers_writer.writerow(['ID', 'Attempt ID', 'Question', 'User Answer', 'Model Answer', 'Correctness', 'Score', 'Feedback'])
        
        for answer in QuestionAnswer.query.all():
            answers_writer.writerow([
                answer.id, answer.attempt_id, answer.question_text,
                answer.user_answer, answer.model_answer, answer.correctness,
                answer.score, answer.feedback
            ])
        
        zip_file.writestr('question_answers.csv', answers_data.getvalue())
    
    zip_buffer.seek(0)
    
    return send_file(
        io.BytesIO(zip_buffer.read()),
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'interview_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
    )

@admin_bp.route('/delete/test-data', methods=['POST'])
@login_required
@admin_required
def delete_test_data():
    """Delete all test data (keep users)"""
    try:
        # Delete all question answers first (foreign key constraint)
        QuestionAnswer.query.delete()
        # Delete all quiz attempts
        QuizAttempt.query.delete()
        
        db.session.commit()
        flash('Test data deleted successfully', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting test data: {str(e)}', 'danger')
    
    return redirect(url_for('admin.dashboard'))

@admin_bp.route('/api/stats')
@login_required
@admin_required
def api_stats():
    """API endpoint for dashboard statistics"""
    # Performance over time
    performance_data = db.session.query(
        func.date(QuizAttempt.timestamp).label('date'),
        func.avg(QuizAttempt.average_score).label('avg_score'),
        func.count(QuizAttempt.id).label('attempts')
    ).group_by(func.date(QuizAttempt.timestamp)).order_by(func.date(QuizAttempt.timestamp)).limit(30).all()
    
    # Subject difficulty distribution
    difficulty_data = db.session.query(
        QuizAttempt.difficulty,
        func.avg(QuizAttempt.average_score).label('avg_score'),
        func.count(QuizAttempt.id).label('count')
    ).group_by(QuizAttempt.difficulty).all()
    
    return jsonify({
        'performance_over_time': [
            {
                'date': str(p.date),
                'avg_score': float(p.avg_score),
                'attempts': p.attempts
            } for p in performance_data
        ],
        'difficulty_distribution': [
            {
                'difficulty': d.difficulty,
                'avg_score': float(d.avg_score),
                'count': d.count
            } for d in difficulty_data
        ]
    })

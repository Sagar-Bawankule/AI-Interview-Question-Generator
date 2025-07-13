#!/usr/bin/env python3
"""
Database initialization script for AI Interview Question Generator
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db
from models import User, QuizAttempt, QuestionAnswer

def init_database():
    """Initialize the database with tables"""
    try:
        with app.app_context():
            print("Checking database tables...")
            
            # Check if tables already exist
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            existing_tables = inspector.get_table_names()
            print(f"Existing tables: {existing_tables}")
            
            if not existing_tables:
                print("No tables found, creating new ones...")
                # Create all tables
                db.create_all()
                print("Created all tables successfully!")
            else:
                print("Tables already exist, ensuring they're up to date...")
                # Just ensure tables are created (won't drop existing)
                db.create_all()
                print("Database tables verified/updated!")
            
            # Verify tables exist
            tables = inspector.get_table_names()
            print(f"Final tables: {tables}")
            
            # Check if admin user exists, if not create one
            admin_email = os.environ.get('ADMIN_EMAIL', 'admin@example.com')
            admin_user = User.query.filter_by(email=admin_email).first()
            
            if not admin_user:
                from werkzeug.security import generate_password_hash
                admin_user = User(
                    username='admin',
                    email=admin_email,
                    password_hash=generate_password_hash('admin123'),
                    is_admin=True
                )
                db.session.add(admin_user)
                db.session.commit()
                print(f"Created admin user with email: {admin_email}")
                print("Default admin password: admin123 (please change after first login)")
            else:
                print("Admin user already exists")
                
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("AI Interview Question Generator - Database Setup")
    print("=" * 50)
    
    success = init_database()
    
    if success:
        print("\n✅ Database initialization completed successfully!")
    else:
        print("\n❌ Database initialization failed!")
        sys.exit(1)

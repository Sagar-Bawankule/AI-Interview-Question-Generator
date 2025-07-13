#!/usr/bin/env python3
"""
Database initialization script
"""
import os
from app import app
from models import db, User
from werkzeug.security import generate_password_hash

def init_database():
    """Initialize the database with proper schema"""
    with app.app_context():
        # Drop all existing tables and recreate them
        db.drop_all()
        db.create_all()
        print("Database tables created successfully!")
        
        # Create a default admin user
        admin = User(
            email="admin@example.com",
            username="admin",
            password_hash=generate_password_hash("admin123"),
            is_admin=True
        )
        db.session.add(admin)
        db.session.commit()
        print("Created default admin user: admin@example.com / admin123")
        
        # Print table info
        inspector = db.inspect(db.engine)
        tables = inspector.get_table_names()
        print(f"Created tables: {tables}")
        
        for table in tables:
            print(f"\nTable: {table}")
            columns = inspector.get_columns(table)
            for col in columns:
                print(f"  - {col['name']}: {col['type']}")

if __name__ == '__main__':
    init_database()

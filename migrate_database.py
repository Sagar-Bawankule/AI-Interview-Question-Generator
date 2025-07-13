#!/usr/bin/env python3
"""
Database migration script to fix schema issues
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db
from sqlalchemy import text

def migrate_database():
    """Migrate database schema to match current models"""
    try:
        with app.app_context():
            print("Checking database schema...")
            
            # Check current schema
            result = db.session.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'users'"))
            existing_columns = [row[0] for row in result]
            print(f"Existing columns in users table: {existing_columns}")
            
            # Add missing columns if they don't exist
            missing_columns = []
            
            if 'username' not in existing_columns:
                print("Adding username column...")
                db.session.execute(text("ALTER TABLE users ADD COLUMN username VARCHAR(80)"))
                missing_columns.append('username')
            
            if 'is_admin' not in existing_columns:
                print("Adding is_admin column...")
                db.session.execute(text("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT FALSE"))
                missing_columns.append('is_admin')
            
            if 'is_active' not in existing_columns:
                print("Adding is_active column...")
                db.session.execute(text("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT TRUE"))
                missing_columns.append('is_active')
            
            if 'date_registered' not in existing_columns:
                print("Adding date_registered column...")
                db.session.execute(text("ALTER TABLE users ADD COLUMN date_registered TIMESTAMP DEFAULT CURRENT_TIMESTAMP"))
                missing_columns.append('date_registered')
            
            if 'last_login' not in existing_columns:
                print("Adding last_login column...")
                db.session.execute(text("ALTER TABLE users ADD COLUMN last_login TIMESTAMP"))
                missing_columns.append('last_login')
            
            # Commit the changes
            if missing_columns:
                db.session.commit()
                print(f"Added missing columns: {missing_columns}")
                
                # Update existing users with default usernames if username was missing
                if 'username' in missing_columns:
                    print("Updating existing users with default usernames...")
                    db.session.execute(text("UPDATE users SET username = 'user_' || id WHERE username IS NULL"))
                    db.session.commit()
                    print("Updated usernames for existing users")
            else:
                print("No missing columns found - schema is up to date")
            
            # Verify the schema is now correct
            result = db.session.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'users'"))
            final_columns = [row[0] for row in result]
            print(f"Final columns in users table: {final_columns}")
            
            # Now try to create any missing tables
            db.create_all()
            print("Ensured all tables exist")
            
    except Exception as e:
        print(f"Error migrating database: {str(e)}")
        import traceback
        traceback.print_exc()
        db.session.rollback()
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("AI Interview Question Generator - Database Migration")
    print("=" * 50)
    
    success = migrate_database()
    
    if success:
        print("\n✅ Database migration completed successfully!")
    else:
        print("\n❌ Database migration failed!")
        sys.exit(1)

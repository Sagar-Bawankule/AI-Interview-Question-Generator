#!/usr/bin/env python3
"""
Test script to check if the Flask app starts properly
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_app():
    """Test if the Flask app starts without errors"""
    try:
        print("Testing Flask application startup...")
        
        # Import the app
        from app import app, db, User
        
        # Test database connection
        with app.app_context():
            print("Testing database connection...")
            user_count = User.query.count()
            print(f"Found {user_count} users in database")
            
            # Test if all routes are properly registered
            print("Testing routes...")
            routes = []
            for rule in app.url_map.iter_rules():
                routes.append(f"{rule.rule} -> {rule.endpoint}")
            
            print(f"Registered routes ({len(routes)}):")
            for route in sorted(routes):
                print(f"  {route}")
            
            # Check if critical routes exist
            critical_routes = ['/login', '/register', '/dashboard', '/quiz', '/leaderboard']
            missing_routes = []
            
            for route in critical_routes:
                found = any(rule.rule == route for rule in app.url_map.iter_rules())
                if not found:
                    missing_routes.append(route)
            
            if missing_routes:
                print(f"❌ Missing critical routes: {missing_routes}")
                return False
            else:
                print("✅ All critical routes are registered")
            
        print("✅ Flask application test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Flask app: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("AI Interview Question Generator - App Test")
    print("=" * 50)
    
    success = test_app()
    
    if success:
        print("\n✅ Application is ready to run!")
        print("You can now start the app with: python app.py")
        print("Or use the batch file: run.bat")
    else:
        print("\n❌ Application test failed!")
        sys.exit(1)

#!/usr/bin/env bash
# build.sh - Lightweight build script for Render

set -o errexit

echo "Starting lightweight build process..."
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Create a lightweight requirements file for production
echo "Creating production requirements..."
cat > requirements-prod.txt << EOF
Flask==2.3.3
Flask-Login==0.6.3
Flask-SQLAlchemy==3.0.5
SQLAlchemy==2.0.21
Werkzeug==2.3.7
gunicorn==21.2.0
psycopg2-binary==2.9.7
supabase==1.0.4
python-dotenv==1.0.0
reportlab==4.0.4
Pillow==10.0.0
requests==2.31.0
itsdangerous==2.1.2
Jinja2==3.1.2
MarkupSafe==2.1.3
click==8.1.7
blinker==1.6.3
EOF

echo "Installing lightweight dependencies..."
pip install --upgrade pip
pip install -r requirements-prod.txt

echo "Build completed successfully with lightweight dependencies!"
echo "Application will use template-based question generation to avoid memory issues."

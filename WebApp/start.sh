#!/bin/bash
# Startup script for IASC Phonotactics Web App

echo "========================================="
echo "IASC Phonotactics Generator - Web App"
echo "========================================="
echo ""

# Check if we're in the WebApp directory
if [ ! -f "app.py" ]; then
    echo "Error: This script must be run from the WebApp directory"
    echo "Usage: cd WebApp && ./start.sh"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.8 or later."
    exit 1
fi

# Check if Flask is installed
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Flask not found. Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Check if parent dependencies are installed
python3 -c "import boto3, jinja2, openai" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Some parent IASC dependencies may be missing."
    echo "Please run 'pip install -r ../requirements.txt' if you encounter errors."
    echo ""
fi

echo "Starting IASC Phonotactics Web App..."
echo ""
echo "The app will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask app
python3 app.py

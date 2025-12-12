#!/bin/bash

# Setup script for Student Stress Detection Project
# This script will install required system packages and set up the Python environment

echo "=========================================="
echo "Student Stress Detection - Setup Script"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo "Please do not run this script as root/sudo"
    echo "The script will prompt for sudo when needed"
    exit 1
fi

# Step 1: Install system packages
echo "Step 1: Installing system packages (python3-pip, python3-venv)..."
sudo apt update
sudo apt install -y python3-pip python3-venv

if [ $? -ne 0 ]; then
    echo "Error: Failed to install system packages"
    echo "Please run manually: sudo apt install python3-pip python3-venv"
    exit 1
fi

echo "✓ System packages installed"
echo ""

# Step 2: Create virtual environment
echo "Step 2: Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

echo "✓ Virtual environment created"
echo ""

# Step 3: Activate virtual environment and install dependencies
echo "Step 3: Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install Python dependencies"
    exit 1
fi

echo "✓ Python dependencies installed"
echo ""

echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python run_pipeline.py"
echo ""
echo "To start the web app:"
echo "  cd web_app && python app.py"
echo ""


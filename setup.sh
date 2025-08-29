#!/bin/bash

# Setup script for nsmt project using uv
# Creates virtual environment and installs dependencies

set -e  # Exit on any error

echo "Setting up Neural Spectral Modeling Template project environment with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment with Python 3.9+
echo "Creating virtual environment..."
uv venv .venv --python 3.9

# Activate environment and install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt

echo "Setup complete! To activate the environment, run:"
echo "source .venv/bin/activate"
echo ""
echo "Or for csh/tcsh:"
echo "source .venv/bin/activate.csh"

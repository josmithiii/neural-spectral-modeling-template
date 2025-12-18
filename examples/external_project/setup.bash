#!/usr/bin/env bash

# Setup script for external_project examples
# Installs additional dependencies required for the examples.
# Requires the main project's virtual environment to be activated.

set -e  # Exit on any error

echo "Setting up external_project example environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first."
    echo "Refer to the main project's README or setup.bash for uv installation instructions."
    exit 1
fi

# Check if the main project's virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Main project virtual environment is not activated."
    echo "Please activate it first by running 'source ../../.venv/bin/activate' from this directory,"
    echo "or 'source .venv/bin/activate' from the project root."
    exit 1
fi

echo "Installing additional dependencies (soundfile, mido, resampy) into the active virtual environment..."
uv pip install soundfile mido resampy

echo "External project example setup complete!"
echo "You can now run 'make all' in 'examples/external_project/'."

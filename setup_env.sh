#!/bin/bash

# Setup script for FL Mental Health project
# This ensures proper Python version and environment setup for MLX compatibility

set -euo pipefail

echo "🔧 Setting up FL Mental Health environment..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS. MLX requires macOS with Apple Silicon."
    exit 1
fi

# Check for Python 3.11 or 3.12
PYTHON_CMD=""
for py_version in python3.12 python3.11; do
    if command -v "$py_version" >/dev/null 2>&1; then
        PYTHON_CMD="$py_version"
        echo "✅ Found compatible Python: $PYTHON_CMD"
        break
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    echo "❌ Python 3.11 or 3.12 required for MLX compatibility."
    echo "   Python 3.13 has known issues with MLX."
    echo "   Please install Python 3.11 or 3.12 using Homebrew:"
    echo "   brew install python@3.11"
    exit 1
fi

# Create virtual environment with compatible Python version
VENV_DIR=".venv311"
if [[ -d "$VENV_DIR" ]]; then
    echo "🔄 Virtual environment already exists. Recreating..."
    rm -rf "$VENV_DIR"
fi

echo "📦 Creating virtual environment with $PYTHON_CMD..."
$PYTHON_CMD -m venv "$VENV_DIR"

# Activate environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Environment setup complete!"
echo "💡 To activate the environment, run: source $VENV_DIR/bin/activate"
echo "🚀 To start the portal, run: ./run_portal.sh"

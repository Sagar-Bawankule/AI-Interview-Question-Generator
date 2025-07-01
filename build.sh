#!/usr/bin/env bash
# build.sh - Custom build script for Render

set -o errexit

echo "Installing Python dependencies..."
pip install --upgrade pip

# Install pre-built wheels for Rust-based packages to avoid compilation issues
echo "Installing pre-built wheels for Rust-based packages..."
pip install --no-build-isolation --no-deps tokenizers==0.13.3
pip install --no-build-isolation --no-deps safetensors==0.3.3

# Install CPU-only PyTorch first
echo "Installing CPU-only PyTorch separately..."
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.7.1+cpu

# Then install the rest of the requirements
echo "Installing remaining dependencies..."
pip install -r requirements.txt --no-deps
pip install -r requirements.txt

echo "Creating model cache directory..."
mkdir -p /tmp/models
chmod 777 /tmp/models

echo "Build completed successfully!"

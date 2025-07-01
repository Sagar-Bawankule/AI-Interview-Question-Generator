#!/usr/bin/env bash
# build.sh - Custom build script for Render that avoids Rust compilation

set -o errexit

echo "Starting build process..."
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

echo "Installing Python dependencies from requirements.txt (basic packages)..."
pip install --upgrade pip
pip install -r requirements.txt

# Install pre-built wheels directly from URLs to avoid compilation
echo "Installing pre-built wheels for Rust-based packages..."

# PyTorch (CPU-only)
pip install torch==2.7.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install specific versions of packages with pre-built wheels
pip install \
    https://files.pythonhosted.org/packages/ba/57/ece75d7cd6b8d2b8a71cd2aa9ffbbcb95cc374cba2c80e66e9d0dcea2ace/tokenizers-0.13.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    https://files.pythonhosted.org/packages/8c/43/157277b1c7e38ec05ce1d0bb5bfa27595de7f1c2bf4d655876d82cad392b/safetensors-0.3.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    https://files.pythonhosted.org/packages/f0/92/b66b605df7c4c03ac9e35b1ae6a520b52c02dcdf131ecd1fd20e39f04c8d/transformers-4.32.1-py3-none-any.whl \
    https://files.pythonhosted.org/packages/55/da/24ae9fb3f5d1be9e8effc4cc8c365b245248ce66b8cb3c9e2b12ae86fd3a/huggingface_hub-0.16.4-py3-none-any.whl

echo "Creating model cache directory..."
mkdir -p /tmp/models
chmod 777 /tmp/models

echo "Verifying installations..."
pip list

echo "Build completed successfully!"

#!/bin/bash

env_name=igibson_test_github

# Ensure conda is properly initialized
source $(conda info --base)/etc/profile.d/conda.sh

# Update conda and create environment
conda update -y conda
conda create -y -n $env_name python=3.8
conda activate $env_name


# Detect OS and install CMake accordingly
OS=$(uname)

if [[ "$OS" == "Linux" ]] || [[ "$OS" == "CYGWIN"* ]] || [[ "$OS" == "MINGW"* ]]; then
    # Install CMake via pip for Linux and Windows
    pip install cmake
elif [[ "$OS" == "Darwin" ]]; then
    # Check if Homebrew is installed
    if ! command -v brew &>/dev/null; then
        echo "Homebrew not found. Please install it first: https://brew.sh/"
        exit 1
    fi

    # Check if running on Apple Silicon (M1/M2/M3)
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        sudo xcode-select --switch /Library/Developer/CommandLineTools
        export SDKROOT=$(xcrun --show-sdk-path)
        echo "Installing CMake..."
        brew install cmake
    fi
else
    echo "Unsupported OS: $OS"
    exit 1
fi

pip install igibson==2.2.2 torch stable-baselines3
pip install setuptools==65.6.3
pip install gym==0.22.0 'shimmy>=2.0'

python -m igibson.utils.assets_utils --download_assets

# Install local package (ensure you are in the correct directory)
if [ -f "setup.py" ]; then
    pip install -e .
else
    echo "Warning: setup.py not found. Skipping editable installation."
fi
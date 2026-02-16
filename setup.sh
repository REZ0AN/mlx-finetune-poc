#!/bin/bash

set -e  # Exit on error

echo "================================================"
echo "MLX Fine-tuning PoC Setup (MLX Only)"
echo "================================================"

# Check Python 3.11
echo "[CHECK] Checking Python version..."
if ! command -v python3.11 &> /dev/null
then
    echo "[ERROR] Python 3.11 is required."
    echo "[INFO] Install with: brew install python@3.11"
    exit 1
fi
echo "[OK] Python 3.11 found: $(python3.11 --version)"

# Download Alpaca dataset
echo "[DOWNLOAD] Downloading Alpaca dataset..."
curl -o alpaca_data.json https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
echo "[OK] Dataset downloaded"


# Setup virtual environment
echo "[SETUP] Creating virtual environment..."
python3.11 -m venv mlx-poc

echo "[ACTIVATE] Activating virtual environment..."
source mlx-poc/bin/activate

echo "[INSTALL] Installing MLX and dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[OK] Setup complete! Virtual environment 'mlx-poc' is ready with MLX and dependencies installed."
echo "[INFO] To activate the environment, run: source mlx-poc/bin/activate"
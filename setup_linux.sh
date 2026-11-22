#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y python3-venv python3-pip wmctrl xdotool tesseract-ocr x11-utils

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-linux.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo
echo "Linux dependencies installed. Start Vampire Survivors in this X11 session,"
echo "then run: source .venv/bin/activate && python -B -u ./train_agent.py --full-run --deep-learning"

#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
deps="$PWD/.local-deps/usr"
export PATH="$deps/bin:$PATH"
export LD_LIBRARY_PATH="$deps/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TESSDATA_PREFIX="$deps/share/tesseract-ocr/5/tessdata"

exec .venv/bin/python -B -u train_agent.py "$@"

#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
deps="$PWD/.local-deps/usr"
if [[ -d "$deps/bin" ]]; then
    export PATH="$deps/bin:$PATH"
fi
if [[ -d "$deps/lib/x86_64-linux-gnu" ]]; then
    export LD_LIBRARY_PATH="$deps/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -z "${TESSDATA_PREFIX:-}" && -r "$deps/share/tesseract-ocr/5/tessdata/eng.traineddata" ]]; then
    export TESSDATA_PREFIX="$deps/share/tesseract-ocr/5/tessdata"
fi

exec .venv/bin/python -B -u train_agent.py "$@"

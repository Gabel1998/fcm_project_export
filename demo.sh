#!/usr/bin/env bash
# Start FCM Landmark Detection demo (Streamlit-app).
# Virker uanset hvilken mappe man kører den fra.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -x "$DIR/.venv/bin/streamlit" ]]; then
    echo " Streamlit ikke fundet i $DIR/.venv/bin/."
    echo " Kør først: $DIR/.venv/bin/pip install -r $DIR/requirements.txt"
    exit 1
fi

exec "$DIR/.venv/bin/streamlit" run "$DIR/scripts/03_demo/app.py" "$@"
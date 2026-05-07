#!/usr/bin/env bash
# Skift hvilken trænet model demo-appen bruger.
# Eksempel: ./switch_model.sh v2_robust
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$DIR/models"

if [[ $# -ne 1 ]]; then
    echo "Brug: $0 <version-navn>"
    echo ""
    echo "Tilgængelige versioner:"
    find "$MODELS_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort | sed 's/^/  /'
    echo ""
    CURRENT=$(readlink "$MODELS_DIR/active" 2>/dev/null || echo "(ingen)")
    echo "Aktiv lige nu: $CURRENT"
    exit 1
fi

VERSION="$1"
TARGET="$MODELS_DIR/$VERSION"

if [[ ! -d "$TARGET" ]]; then
    echo "Model-version '$VERSION' findes ikke i $MODELS_DIR/."
    echo ""
    echo "Tilgængelige versioner:"
    find "$MODELS_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort | sed 's/^/  /'
    exit 1
fi

if [[ ! -f "$TARGET/best.pt" ]]; then
    echo "Mappen $TARGET findes, men der er ingen best.pt i den."
    exit 1
fi

ln -sfn "$VERSION" "$MODELS_DIR/active"
echo "Aktiv model: $VERSION"
echo "Genstart Streamlit-appen for at den henter den nye model."

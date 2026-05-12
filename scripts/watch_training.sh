#!/bin/bash
# Watch live progress af YOLO-træning.
#
# Brug:
#   ./scripts/watch_training.sh                       # default: v4_composite
#   ./scripts/watch_training.sh v5_dense              # andet run-navn
#
# Hvad du ser:
#   - Header med kolonne-navne fra results.csv
#   - Sidste 8 epochs med loss + metrics
#   - Opdaterer hver 5. sekund (Ctrl-C for at stoppe)

set -euo pipefail

RUN_NAME="${1:-v4_composite}"
RESULTS_CSV="runs/train/${RUN_NAME}/results.csv"

if [ ! -d "runs/train/${RUN_NAME}" ]; then
    echo "❌ Træning ikke startet endnu: runs/train/${RUN_NAME}/"
    echo "   Kør 'make train' eller '.venv/bin/python scripts/02_yolo_pipeline/03b_train_composite.py' først."
    exit 1
fi

echo "Watcher runs/train/${RUN_NAME}/  (Ctrl-C for at stoppe)"
echo ""

while true; do
    clear
    echo "=== Live træning: ${RUN_NAME} ==="
    echo "$(date '+%H:%M:%S')"
    echo ""

    if [ ! -f "$RESULTS_CSV" ]; then
        echo "Venter på første epoch (results.csv eksisterer ikke endnu)..."
    else
        # Vis header + sidste 8 rækker — let læselig
        echo "Sidste 8 epochs (af results.csv):"
        echo ""
        # Pretty-print med kolonne-alignment
        (head -1 "$RESULTS_CSV"; tail -8 "$RESULTS_CSV") | \
            awk -F',' 'NR==1{for(i=1;i<=NF;i++) printf "%-12s", $i; print ""}
                       NR>1 {for(i=1;i<=NF;i++) printf "%-12s", $i; print ""}'
        echo ""

        # Find peak mAP@50 hidtil
        PEAK_MAP=$(tail -n +2 "$RESULTS_CSV" | awk -F',' '{print $7}' | sort -n | tail -1 2>/dev/null || echo "?")
        echo "Peak mAP@50 indtil videre: ${PEAK_MAP}"
    fi

    echo ""
    echo "Hvis du vil se plots:  open runs/train/${RUN_NAME}/results.png"
    echo "Hvis du vil predicte:  make predict"

    sleep 5
done

#!/bin/bash
# Auto-resume YOLO-træning der crasher pga. MPS-bug i Ultralytics tal.py.
#
# Brug:
#   ./scripts/auto_resume.sh                       # default: v5_no_leak, 20 forsøg
#   ./scripts/auto_resume.sh v5_no_leak            # specifikt run-navn
#   ./scripts/auto_resume.sh v5_no_leak 30         # custom max attempts
#
# Loopet stopper når:
#   - Træning afslutter normalt (succes eller patience-stop) → exit 0
#   - MAX_ATTEMPTS overskredet → exit 1
#   - Du dræber scriptet med Ctrl-C → exit 130

set -u

NAME="${1:-v5_no_leak}"
MAX_ATTEMPTS="${2:-20}"
SLEEP_BETWEEN=5

attempt=0
total_start=$(date +%s)

cleanup() {
    echo ""
    echo "🛑 Stoppet af bruger (Ctrl-C). Last.pt har den senest gemte state."
    echo "   Genstart hvor du slap med: make resume NAME=$NAME"
    echo "   eller fortsæt auto-resume: make auto-resume NAME=$NAME"
    exit 130
}
trap cleanup INT TERM

cat <<EOF
=== Auto-resume af $NAME ===
Max attempts:  $MAX_ATTEMPTS
Sleep mellem:  $SLEEP_BETWEEN sek
Ctrl-C for at stoppe når som helst.

Hver gang ultralytics MPS-bug'en rammer (typisk under validation),
gemmes last.pt og dette script restarter automatisk.

EOF

while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    attempt_start=$(date +%s)
    echo "════════════════════════════════════════════════"
    echo " Attempt $attempt/$MAX_ATTEMPTS  —  $(date '+%H:%M:%S')"
    echo "════════════════════════════════════════════════"

    if make resume NAME="$NAME"; then
        total_min=$((($(date +%s) - total_start) / 60))
        echo ""
        echo "✅ Træning afsluttet normalt efter $attempt attempt(s)."
        echo "   Total kørsel: $total_min minutter"
        echo "   Næste skridt: make save-model NAME=$NAME MODEL_DIR=$NAME"
        exit 0
    else
        rc=$?
        attempt_duration=$((($(date +%s) - attempt_start) / 60))
        echo ""
        echo "❌ Attempt $attempt crashede (exit $rc) efter $attempt_duration min."

        if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
            echo "   Venter $SLEEP_BETWEEN sek og prøver igen..."
            sleep "$SLEEP_BETWEEN"
        fi
    fi
done

total_min=$((($(date +%s) - total_start) / 60))
echo ""
echo "⚠️  $MAX_ATTEMPTS attempts brugt op (total $total_min min)."
echo "    Last.pt har den seneste epoch's state."
echo "    Genstart loop med: make auto-resume NAME=$NAME"
exit 1

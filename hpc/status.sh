#!/bin/bash
# One-screen progress check for the GPU runs. Safe to run any time; it only reads.
#     bash ~/urop_code/hpc/status.sh
#     watch -n 60 bash ~/urop_code/hpc/status.sh      (refreshes every minute; Ctrl+C to stop watching)
source "$HOME/urop_code/hpc/paths.sh" >/dev/null 2>&1
LOGS="$UROP_RESULTS/logs"

echo "== jobs   (Q = waiting for a GPU, R = running, H = waiting for another job to finish)"
JOBS=$(qstat -u "$USER" 2>/dev/null | awk '/^[0-9]/')
if [ -z "$JOBS" ]; then
    echo "   none left: everything has finished (or failed; see the logs below)"
else
    echo "$JOBS" | awk '{n[$10]++} END {printf "   %d running, %d waiting, %d held\n", n["R"], n["Q"], n["H"]}'
    echo "$JOBS" | awk '$10=="R" {print $1, $4, $11}' | while read -r id name elap; do
        log=$(ls "$LOGS"/*_"${id%%.*}".log 2>/dev/null | head -1)
        last=$( [ -n "$log" ] && tail -c 400 "$log" | tr '\r' '\n' | grep -v '^\s*$' | tail -n 1 | cut -c1-110)
        echo "   R ${id%%.*} $name ($elap): ${last:-no log yet}"
    done
fi

echo
echo "== finished LoRAs (fold/class: steps)"
found=$(find "$UROP_WORK/lora" -name steps.txt 2>/dev/null | sort)
if [ -z "$found" ]; then echo "   none yet"; else
    for f in $found; do d=$(dirname "$f"); echo "   $(basename "$(dirname "$d")")/$(basename "$d"): $(cat "$f")"; done | paste - - - - ; fi

echo
echo "== fake images made"
if [ -d "$UROP_WORK/gen" ]; then
    for d in "$UROP_WORK"/gen/*/f*; do
        [ -d "$d" ] && echo "   $(basename "$(dirname "$d")")/$(basename "$d"): $(find "$d" -name '*.png' | wc -l)"
    done | paste - - - -
else echo "   none yet"; fi

echo
echo "== detector results so far (macro accuracy; escapes = defects called OK)"
python3 - "$UROP_RESULTS/vit_results.csv" <<'EOF'
import csv, sys
from collections import defaultdict
try:
    rows = list(csv.DictReader(open(sys.argv[1])))
except FileNotFoundError:
    sys.exit("   none yet")
by = defaultdict(list)
for r in rows:
    by[r["tag"]].append(r)
for tag, rs in by.items():
    acc = [float(r["macro_acc"]) for r in rs]
    esc = [float(r["escape_rate"]) for r in rs]
    print(f"   {tag:<28} folds {','.join(r['fold'] for r in rs):<10} macro {sum(acc)/len(acc):.3f}  escapes {sum(esc)/len(esc):.3f}")
EOF

echo
echo "== problems in the logs"
err=$(grep -l -E "Traceback|Error|Killed|exceeded" "$LOGS"/*.log 2>/dev/null)
if [ -z "$err" ]; then echo "   none"; else
    for f in $err; do echo "   $(basename "$f"): $(grep -E "Error|Killed|exceeded" "$f" | tail -1 | cut -c1-100)"; done; fi

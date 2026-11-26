#!/usr/bin/env bash
# POS launch script, STAGE 1 of 6: tabular models only (~1-1.5h, fits one
# 6h testbed reservation easily).
#
# Usage: pos commands launch nodeA --infile scripts/run_tabular.sh
#
# Assumes the whole project (including Data/Processed, already built locally
# by Notebooks 01-03) has been git-cloned onto the node already, e.g.:
#   pos commands launch nodeA -- git clone <your-repo-url> /root/energy-aware-code-recommendation
#
# Adjust PROJECT_DIR below if you cloned it somewhere else.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/energy-aware-code-recommendation}"
cd "$PROJECT_DIR"

echo "=== Node info ==="
hostname
uname -a
python3 --version
echo

echo "=== Checking for RAPL energy counters ==="
if ls /sys/class/powercap/intel-rapl:*/energy_uj >/dev/null 2>&1; then
    echo "RAPL counters found:"
    ls /sys/class/powercap/intel-rapl:*/energy_uj
    for f in /sys/class/powercap/intel-rapl:*/energy_uj; do
        if ! cat "$f" >/dev/null 2>&1; then
            echo "WARNING: $f is not readable by this user."
        fi
    done
else
    echo "No RAPL counters visible under /sys/class/powercap. Energy measurement"
    echo "will fall back to Scaphandre (if running) or CodeCarbon's software estimate."
fi
echo

echo "=== Setting up Python environment ==="
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r scripts/requirements.txt
echo "Environment ready."
echo

echo "=== STAGE 1/6: Tabular model measurements (10 repetitions, both datasets) ==="
python3 scripts/measure_tabular_models.py
echo

echo "=== Stage 1 complete ==="
echo "Results are under Results/resource_measurements/<run_id>/ inside $PROJECT_DIR."
echo "Copy that run folder back to your laptop NOW, before freeing this allocation:"
echo "  scp -P 10022 -r faroukh@coinbase.net.cit.tum.de:$PROJECT_DIR/Results/resource_measurements/<run_id> ."
echo "(replace <run_id> with the actual folder name printed above)"

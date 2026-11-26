#!/usr/bin/env bash
# POS launch script, STAGES 5-6 of 6: 5 CodeBERT training repetitions on the
# "significant" dataset (~3-3.5h including setup, fits one 6h testbed
# reservation).
#
# Run this SAME script 2 separate times (2 separate testbed reservations) to
# get all 10 repetitions supervisor feedback asked for. Each run writes to its
# own fresh, timestamped results folder -- nothing gets overwritten, so just
# copy each run's folder back to your laptop before freeing that allocation.
#
# Usage: pos commands launch nodeA --infile scripts/run_codebert_significant_chunk.sh
#
# Assumes the whole project (including Data/Processed, already built locally
# by Notebooks 01-03) has been git-cloned onto the node already, e.g.:
#   pos commands launch nodeA -- git clone <your-repo-url> /root/energy-aware-code-recommendation

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
else
    echo "No RAPL counters visible. Energy measurement will fall back to"
    echo "Scaphandre (if running) or CodeCarbon's software estimate."
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

echo "=== CodeBERT: significant dataset, 5 training repetitions (half of 10 needed) ==="
python3 scripts/measure_codebert.py --dataset significant --reps 5
echo

echo "=== This chunk complete ==="
echo "Results are under Results/resource_measurements/<run_id>/ inside $PROJECT_DIR."
echo "Copy that run folder back to your laptop NOW, before freeing this allocation:"
echo "  scp -P 10022 -r faroukh@coinbase.net.cit.tum.de:$PROJECT_DIR/Results/resource_measurements/<run_id> ."
echo "Then run this same script again (fresh node + fresh clone) for the remaining 5 repetitions."

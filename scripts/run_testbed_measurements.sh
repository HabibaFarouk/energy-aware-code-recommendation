#!/usr/bin/env bash
# POS launch script for the LRZ I8 testbed (baltikum/kaunas).
#
# This is what you pass to `pos commands launch`, e.g.:
#
#   pos commands launch nodeA --infile scripts/run_testbed_measurements.sh
#
# It assumes the whole project (this repo, including Data/Processed already
# built locally by Notebooks 01-03) has already been copied onto the node --
# e.g. with the postools file-upload mechanism, or `scp` using the testbed's
# altered SSH port from your own machine before launching the command. Check
# the testbed wiki (https://gitlab.lrz.de/I8-testbeds/wiki) for the exact
# upload step if `pos commands launch --infile` alone does not also copy the
# repository; this script only runs *on* the node, it does not copy itself.
#
# Adjust PROJECT_DIR below to wherever you copy the repo to on the node.

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
    # Confirm they are actually readable without elevated permissions; if this
    # fails, measurement_utils.EnergyCounter will fall back to Scaphandre or
    # CodeCarbon automatically, but it is worth knowing which backend you get.
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

echo "=== Running tabular model measurements (10 repetitions) ==="
python3 scripts/measure_tabular_models.py
echo

echo "=== Running CodeBERT measurements (validated=10, significant=3 repetitions) ==="
python3 scripts/measure_codebert.py
echo

echo "=== All measurements complete ==="
echo "Results are under Results/resource_measurements/<run_id>/ inside $PROJECT_DIR."
echo "Copy that folder back to your laptop before freeing the node allocation, e.g.:"
echo "  scp -P <altered_ssh_port> -r <user>@<node>:$PROJECT_DIR/Results/resource_measurements/<run_id> ."

"""Testbed energy-measurement script: tabular models (Logistic Regression, Random
Forest, XGBoost).

This is the headless, testbed-runnable equivalent of the measurement cell in
Notebooks/04_model_training_measurment.ipynb (Cell 10). It calls the exact same
measurement_utils functions the notebook uses, so the measurement logic is
identical -- only *where* it runs differs.

Why run this on the testbed instead of a laptop:
    measurement_utils.EnergyCounter tries, in order:
      1. Real Intel RAPL hardware counters, read directly from
         /sys/class/powercap/intel-rapl:*/energy_uj (Linux only). This is actual
         hardware energy, not a software estimate.
      2. A Scaphandre metrics endpoint, if one is running.
      3. CodeCarbon's software estimate (recorded separately, always, as a
         secondary column) if neither hardware source is available.
    A laptop usually falls back to (3). A bare-metal Linux testbed node exposes (1)
    automatically through the kernel, with no extra setup needed on our side --
    we only need POS to let us run a plain Python script on the node.

Prerequisites on the testbed node:
    - Python 3 with the same packages as the project's local environment:
      numpy, pandas, scikit-learn, xgboost, psutil, joblib (see requirements.txt
      if one exists, otherwise match the versions listed in environment.json from
      a local run).
    - The full project copied onto the node, with Data/Processed already built
      locally by Notebooks 01-03 (this script does not rebuild features; it only
      measures training/inference that Notebook 04's Cells 1-8 already validated
      for correctness).

Usage on the testbed node (after copying the project and activating the right
Python environment):
    python3 scripts/measure_tabular_models.py

Per supervisor feedback (increase repetitions for the tabular models to 10),
TRAINING_REPETITIONS is set to 10 below.
"""
import sys
import warnings
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = next(
    (
        path
        for path in (SCRIPT_DIR, *SCRIPT_DIR.parents)
        if (path / "Data" / "Processed").is_dir()
    ),
    None,
)
if ROOT is None:
    raise FileNotFoundError(
        "Could not find Data/Processed from this script's location. Copy the full "
        "project (including the Data/Processed folder built locally by "
        "Notebooks 01-03) onto this node first, keeping the same folder layout."
    )
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from measurement_utils import (  # noqa: E402
    initialize_resource_experiment,
    run_inference_measurements,
    run_training_measurements,
    save_resource_summaries,
)

warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)

TRAINING_REPETITIONS = 10
INFERENCE_REPETITIONS = 10
INFERENCE_SAMPLE_PAIRS = 100
INFERENCE_PASSES_PER_REPETITION = 1
PROCESS_SAMPLING_INTERVAL_SECONDS = 0.1


def main():
    resource_state = initialize_resource_experiment(
        ROOT,
        training_repetitions=TRAINING_REPETITIONS,
        inference_repetitions=INFERENCE_REPETITIONS,
        inference_passes=INFERENCE_PASSES_PER_REPETITION,
        inference_sample_pairs=INFERENCE_SAMPLE_PAIRS,
        sampling_interval_seconds=PROCESS_SAMPLING_INTERVAL_SECONDS,
    )

    print("=" * 70)
    print("Tabular model resource measurement")
    print("Run directory:", resource_state["run_root"])
    energy_backend = resource_state["energy_counter"].backend
    print("Energy source:", energy_backend or "unavailable (no RAPL/Scaphandre found)")
    if energy_backend is None:
        print(
            "WARNING: no hardware energy source was detected on this machine. "
            "If this is meant to run on the LRZ testbed, check that "
            "/sys/class/powercap/intel-rapl:*/energy_uj is readable, or that the "
            "Scaphandre metrics endpoint is reachable. CodeCarbon's software "
            "estimate will still be recorded as a fallback."
        )
    print(f"Training repetitions: {TRAINING_REPETITIONS}")
    print(f"Inference repetitions: {INFERENCE_REPETITIONS}")
    print("=" * 70)

    training_measurements, training_idle_measurements = run_training_measurements(
        resource_state
    )
    print(f"Training measurements collected: {len(training_measurements)} rows")

    (
        inference_measurements,
        inference_sample_by_fold,
        inference_idle_measurements,
    ) = run_inference_measurements(resource_state)
    print(f"Inference measurements collected: {len(inference_measurements)} rows")

    training_summary, inference_summary = save_resource_summaries(resource_state)
    print("\nTraining summary:")
    print(training_summary.to_string(index=False))
    print("\nInference summary:")
    print(inference_summary.to_string(index=False))

    print("\nDone. Results written under:", resource_state["run_root"])
    print(
        "Copy that run directory back to your laptop (e.g. scp -P <altered_port> "
        "or the pos file-copy mechanism) and merge it into Results/resource_measurements/ "
        "for Notebook 06's final analysis."
    )


if __name__ == "__main__":
    main()

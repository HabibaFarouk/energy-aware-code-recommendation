"""Testbed energy-measurement script: frozen CodeBERT embeddings + Logistic
Regression head.

This is the headless, testbed-runnable equivalent of
Notebooks/05_codebert_training_measurements.ipynb. It calls the exact same
measurement_utils functions the notebook uses, so the measurement logic is
identical -- only *where* it runs differs (see measure_tabular_models.py's
module docstring for why the testbed gives real RAPL hardware energy instead of
a laptop's software estimate).

Per supervisor feedback: training is repeated 3 times on the validated dataset
(the larger, whole dataset) and 10 times on the significant dataset (the
smaller one). Each repetition re-runs the complete
grouped-training workload (tokenizer/encoder loading, frozen-encoder embedding
extraction, five-fold grouped fitting, and the final full-data fit) -- this is
the expensive part, so expect this script to take a while; time one dataset's
single repetition manually first if you need a duration estimate before
committing to a full run.

Prerequisites on the testbed node:
    - Python 3 with torch and transformers installed in addition to the
      packages listed in measure_tabular_models.py's docstring.
    - The full project copied onto the node, with Data/Processed already built
      locally by Notebooks 01-03.
    - force_cpu=True is used below for determinism and because the testbed
      nodes are not guaranteed to have a GPU; set FORCE_CPU = False if you
      specifically want to measure GPU energy on a node that has one.

Usage on the testbed node:
    python3 scripts/measure_codebert.py
        (runs both datasets, full repetition counts -- the original behaviour)
    python3 scripts/measure_codebert.py --dataset validated --reps 1
        (runs only "validated", 1 training repetition -- use this to split a
        long run across multiple shorter testbed reservations; call this
        three times in three separate 6h reservations to get all 3 validated
        repetitions, since each call writes to its own timestamped results
        folder and nothing is overwritten between calls)
    python3 scripts/measure_codebert.py --dataset significant --reps 5
        (runs only "significant", 5 training repetitions; call this twice to
        get all 10 significant repetitions)
"""
import argparse
import sys
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
    initialize_codebert_resource_experiment,
    measure_codebert_inference,
    train_and_measure_codebert_dataset,
)

MAX_INPUT_LENGTH = 512
BATCH_SIZE = 8
INFERENCE_SAMPLE_PAIRS = 100
INFERENCE_REPETITIONS = 10
PROCESS_SAMPLING_INTERVAL_SECONDS = 0.1

# Per supervisor feedback: validated=3 repetitions, significant=10 repetitions.
TRAINING_REPETITIONS_BY_DATASET = {
    "validated": 3,
    "significant": 10,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CodeBERT resource measurements, optionally restricted "
        "to one dataset and a custom repetition count (useful for splitting a "
        "long run across several shorter testbed reservations)."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(TRAINING_REPETITIONS_BY_DATASET.keys()),
        default=None,
        help="Run only this dataset instead of both. Default: run both "
        "datasets (the original, pre-splitting behaviour).",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=None,
        help="Override the number of training repetitions for the selected "
        "dataset (requires --dataset). Default: use the value from "
        "TRAINING_REPETITIONS_BY_DATASET (validated=3, significant=10).",
    )
    args = parser.parse_args()
    if args.reps is not None and args.dataset is None:
        parser.error("--reps requires --dataset (it only applies to one dataset at a time).")
    return args


def main():
    args = parse_args()

    if args.dataset is not None:
        reps = args.reps if args.reps is not None else TRAINING_REPETITIONS_BY_DATASET[args.dataset]
        datasets_to_run = {args.dataset: reps}
    else:
        datasets_to_run = TRAINING_REPETITIONS_BY_DATASET

    resource_state = initialize_codebert_resource_experiment(
        ROOT,
        sampling_interval_seconds=PROCESS_SAMPLING_INTERVAL_SECONDS,
        training_repetitions_by_dataset=datasets_to_run,
    )

    print("=" * 70)
    print("CodeBERT resource measurement")
    print("Run directory:", resource_state["run_root"])
    energy_backend = resource_state["energy_counter"].backend
    print("Energy source:", energy_backend or "unavailable (no RAPL/Scaphandre found)")
    if energy_backend is None:
        print(
            "WARNING: no hardware energy source was detected on this machine. "
            "If this is meant to run on the LRZ testbed, check that "
            "/sys/class/powercap/intel-rapl:*/energy_uj is readable, or that the "
            "Scaphandre metrics endpoint is reachable."
        )
    print("Training repetitions by dataset (this run):", datasets_to_run)
    print(f"Inference repetitions: {INFERENCE_REPETITIONS}")
    print("=" * 70)

    for dataset_name, training_repetitions in datasets_to_run.items():
        print(f"\n--- {dataset_name}: training ({training_repetitions} repetitions) ---")
        performance, training_measurements = train_and_measure_codebert_dataset(
            resource_state,
            dataset_name=dataset_name,
            max_length=MAX_INPUT_LENGTH,
            batch_size=BATCH_SIZE,
            training_repetitions=training_repetitions,
        )
        print(performance.to_string(index=False))
        print(f"Training measurement rows: {len(training_measurements)}")

        print(f"\n--- {dataset_name}: inference ({INFERENCE_REPETITIONS} repetitions) ---")
        inference_measurements = measure_codebert_inference(
            resource_state,
            dataset_name=dataset_name,
            inference_pairs=INFERENCE_SAMPLE_PAIRS,
            inference_repetitions=INFERENCE_REPETITIONS,
            batch_size=BATCH_SIZE,
        )
        print(
            inference_measurements.query("task == 'inference'").to_string(index=False)
        )

    print("\nDone. Results written under:", resource_state["run_root"])
    print(
        "Copy that run directory back to your laptop and merge it into "
        "Results/resource_measurements/ for Notebook 06's final analysis."
    )


if __name__ == "__main__":
    main()

import json
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
import torch
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATASET_FILES = {
    "validated": "prepared_training_dataset.parquet",
    "significant": "prepared_significant_training_dataset.parquet",
}
MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost")
SCAPHANDRE_METRICS_URL = "http://localhost:8080/metrics"


def _write_json_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite experiment output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2)


def _write_csv_new(path, frame):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite experiment output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _windows_gpu_names():
    if sys.platform != "win32":
        return []
    command = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object -ExpandProperty Name | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    output = completed.stdout.strip()
    if not output:
        return []
    names = json.loads(output)
    return names if isinstance(names, list) else [names]


def _library_versions():
    versions = {}
    for distribution in (
        "psutil",
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "torch",
        "transformers",
        "codecarbon",
        "pyarrow",
    ):
        try:
            versions[distribution] = package_metadata.version(distribution)
        except package_metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


class EnergyCounter:
    def __init__(self, scaphandre_url=SCAPHANDRE_METRICS_URL):
        self.scaphandre_url = scaphandre_url
        self.rapl_counters = self._find_rapl_counters()
        self.scaphandre_available = False
        try:
            self._read_scaphandre_joules()
            self.scaphandre_available = True
        except (OSError, ValueError, TimeoutError, URLError):
            pass

        if self.rapl_counters:
            self.backend = "RAPL"
            self.scope = "CPU package counters exposed by Linux powercap."
        elif self.scaphandre_available:
            self.backend = "Scaphandre"
            self.scope = "Host energy counter reported by Scaphandre; not CPU-package-only energy."
        else:
            self.backend = None
            self.scope = None

    @staticmethod
    def _find_rapl_counters():
        if not sys.platform.startswith("linux"):
            return []
        counters = []
        for energy_file in Path("/sys/class/powercap").glob("intel-rapl:*/energy_uj"):
            if not re.fullmatch(r"intel-rapl:\d+", energy_file.parent.name):
                continue
            range_file = energy_file.parent / "max_energy_range_uj"
            if range_file.is_file():
                energy_file.read_text(encoding="utf-8")
                counters.append(
                    (energy_file, int(range_file.read_text(encoding="utf-8")))
                )
        return counters

    def _read_scaphandre_joules(self):
        with urlopen(self.scaphandre_url, timeout=1) as response:
            metrics = response.read().decode("utf-8")
        for line in metrics.splitlines():
            if line.startswith("scaph_host_energy_microjoules_total"):
                return float(line.rsplit(None, 1)[-1]) / 1_000_000
        raise ValueError(
            "Scaphandre endpoint has no scaph_host_energy_microjoules_total counter."
        )

    def snapshot(self):
        if self.backend == "RAPL":
            return [
                int(path.read_text(encoding="utf-8"))
                for path, _ in self.rapl_counters
            ]
        if self.backend == "Scaphandre":
            return self._read_scaphandre_joules()
        return None

    def delta_joules(self, start, end):
        if start is None or end is None:
            return None
        if self.backend == "Scaphandre":
            return end - start
        return sum(
            ((end_value - start_value) % maximum) / 1_000_000
            for (_, maximum), start_value, end_value in zip(
                self.rapl_counters, start, end
            )
        )


class ProcessResourceSampler:
    def __init__(self, logical_cores, interval_seconds=0.1):
        self.process = psutil.Process()
        self.logical_cores = logical_cores
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.rss_samples = []
        self.thread = None
        self.baseline_rss_bytes = None
        self.cpu_time_start_seconds = None
        self.cpu_time_end_seconds = None
        self.started_at = None
        self.completed_at = None

    def _sample(self):
        while not self.stop_event.wait(self.interval_seconds):
            self.rss_samples.append(self.process.memory_info().rss)

    def __enter__(self):
        self.baseline_rss_bytes = self.process.memory_info().rss
        self.rss_samples.append(self.baseline_rss_bytes)
        cpu_times = self.process.cpu_times()
        self.cpu_time_start_seconds = cpu_times.user + cpu_times.system
        self.started_at = time.perf_counter()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.completed_at = time.perf_counter()
        cpu_times = self.process.cpu_times()
        self.cpu_time_end_seconds = cpu_times.user + cpu_times.system
        self.stop_event.set()
        self.thread.join()
        self.rss_samples.append(self.process.memory_info().rss)
        return False

    def results(self):
        peak_rss = max(self.rss_samples)
        elapsed = self.completed_at - self.started_at
        cpu_time = self.cpu_time_end_seconds - self.cpu_time_start_seconds
        cpu_pct = float(cpu_time / elapsed / self.logical_cores * 100) if elapsed else np.nan
        return {
            "average_process_cpu_utilization_pct_of_total": cpu_pct,
            "baseline_process_memory_mb": self.baseline_rss_bytes / 1024**2,
            "peak_process_memory_mb": peak_rss / 1024**2,
            "memory_increase_during_task_mb": max(
                0, peak_rss - self.baseline_rss_bytes
            )
            / 1024**2,
            "cpu_sample_count": 2,
            "memory_sample_count": len(self.rss_samples),
        }


def _sample_workload(workload, energy_counter, logical_cores, interval_seconds):
    energy_start = energy_counter.snapshot()
    started = time.perf_counter()
    with ProcessResourceSampler(logical_cores, interval_seconds) as sampler:
        workload_result = workload()
    elapsed = time.perf_counter() - started
    energy_end = energy_counter.snapshot()
    return {
        "elapsed_seconds": elapsed,
        **sampler.results(),
        "actual_energy_joules": energy_counter.delta_joules(
            energy_start, energy_end
        ),
        "energy_source": energy_counter.backend,
        "energy_scope": energy_counter.scope,
        "actual_energy_measurement": energy_counter.backend is not None,
    }, workload_result


def _paired_idle(duration, dataset, model, config, task, repetition, fold=None):
    energy_start = None
    if dataset["energy_counter"].backend is not None:
        energy_start = dataset["energy_counter"].snapshot()
    started = time.perf_counter()
    with ProcessResourceSampler(
        dataset["logical_cores"], dataset["sampling_interval_seconds"]
    ) as sampler:
        time.sleep(duration)
    elapsed = time.perf_counter() - started
    energy_end = None
    if dataset["energy_counter"].backend is not None:
        energy_end = dataset["energy_counter"].snapshot()
    row = {
        "dataset": dataset["dataset_name"],
        "model": model,
        "feature_config": config,
        "task": "idle_baseline",
        "paired_task": task,
        "repetition": repetition,
        "elapsed_seconds": elapsed,
        **sampler.results(),
        "actual_energy_joules": dataset["energy_counter"].delta_joules(
            energy_start, energy_end
        ),
        "energy_source": dataset["energy_counter"].backend,
        "energy_scope": dataset["energy_counter"].scope,
        "actual_energy_measurement": dataset["energy_counter"].backend is not None,
    }
    if fold is not None:
        row["fold"] = fold
    return row


def make_model_factories(model_hyperparameters):
    scaler_settings = model_hyperparameters["logistic_regression"]
    if scaler_settings.get("scaler") != "StandardScaler":
        raise ValueError(
            "Notebook 05 currently supports the StandardScaler Logistic Regression pipeline only."
        )
    return {
        "logistic_regression": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=scaler_settings["max_iter"],
                random_state=scaler_settings["random_state"],
            ),
        ),
        "random_forest": lambda: RandomForestClassifier(
            **model_hyperparameters["random_forest"]
        ),
        "xgboost": lambda: xgboost.XGBClassifier(
            **model_hyperparameters["xgboost"]
        ),
    }


def export_saved_group_fold_models(project_root):
    root = Path(project_root).resolve()
    processed_dir = root / "Data" / "Processed"
    results_root = root / "Results"
    exported_rows = []
    for dataset_name, file_name in DATASET_FILES.items():
        frame = pd.read_parquet(processed_dir / file_name)
        model_dir = processed_dir / "models" / dataset_name
        metadata_path = model_dir / "training_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        feature_config = metadata["feature_config"]
        feature_columns = metadata["feature_columns"]
        hyperparameters = metadata.get("model_hyperparameters")
        if not hyperparameters:
            raise ValueError(
                f"Model hyperparameters are missing from {metadata_path}; "
                "run the updated Notebook 04 training cell once to persist them."
            )
        group_column = metadata["group_column"]
        oof_path = results_root / dataset_name / "model_oof_predictions.parquet"
        oof = pd.read_parquet(oof_path)
        oof = oof.loc[
            oof["feature_config"].eq(feature_config)
            & oof["model"].eq("logistic_regression"),
            ["problem_id", "submission_id_v0", "submission_id_v1", "fold"],
        ].copy()
        keys = ["problem_id", "submission_id_v0", "submission_id_v1"]
        if oof.empty or oof.duplicated(keys).any():
            raise ValueError(f"{dataset_name} OOF rows do not define unique selected fold assignments.")
        expected_pairs = set(
            map(tuple, frame[keys].itertuples(index=False, name=None))
        )
        observed_pairs = set(
            map(tuple, oof[keys].itertuples(index=False, name=None))
        )
        if expected_pairs != observed_pairs:
            raise ValueError(
                f"{dataset_name} selected OOF rows do not cover the prepared dataset exactly."
            )
        folds_per_group = oof.groupby("problem_id")["fold"].nunique()
        if not folds_per_group.eq(1).all():
            raise ValueError(f"{dataset_name} problem groups occur in multiple CV folds.")
        fold_model_dir = model_dir / feature_config / "fold_models"
        fold_model_dir.mkdir(parents=True, exist_ok=True)
        factories = make_model_factories(hyperparameters)

        for fold in sorted(oof["fold"].unique()):
            validation_groups = set(
                oof.loc[oof["fold"].eq(fold), "problem_id"].astype(str)
            )
            group_values = frame[group_column].astype(str)
            validation_mask = group_values.isin(validation_groups)
            train_frame = frame.loc[~validation_mask]
            validation_frame = frame.loc[validation_mask]
            train_groups = set(train_frame[group_column].astype(str))
            if not validation_groups or train_groups & validation_groups:
                raise ValueError(f"{dataset_name} fold {fold} is not group-disjoint.")
            if set(validation_frame[group_column].astype(str)) != validation_groups:
                raise ValueError(
                    f"{dataset_name} fold {fold} groups do not match its saved OOF assignments."
                )
            X_train = train_frame[feature_columns].astype(float)
            y_train = train_frame["energy_winner"].astype(int)
            if X_train.isna().any().any() or not np.isfinite(X_train.to_numpy()).all():
                raise ValueError(f"{dataset_name} fold {fold} training features are invalid.")
            if y_train.nunique() != 2:
                raise ValueError(f"{dataset_name} fold {fold} training set lacks a target class.")

            for model_name in MODEL_NAMES:
                model_path = fold_model_dir / f"{model_name}_fold_{fold}.joblib"
                provenance_path = fold_model_dir / f"{model_name}_fold_{fold}.json"
                if model_path.exists() or provenance_path.exists():
                    if not model_path.is_file() or not provenance_path.is_file():
                        raise FileExistsError(
                            f"Partial fold-model artifacts already exist for {model_path}; "
                            "refusing to overwrite."
                        )
                    existing = json.loads(provenance_path.read_text(encoding="utf-8"))
                    if (
                        existing.get("feature_config") != feature_config
                        or existing.get("fold") != int(fold)
                        or set(existing.get("train_groups", [])) != train_groups
                        or set(existing.get("validation_groups", [])) != validation_groups
                    ):
                        raise FileExistsError(
                            f"Existing fold-model provenance does not match {model_path}; "
                            "refusing to overwrite."
                        )
                    exported_rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "fold": int(fold),
                            "status": "already_exported",
                            "training_groups": len(train_groups),
                            "validation_groups": len(validation_groups),
                        }
                    )
                    continue

                fitted = factories[model_name]()
                fitted.fit(X_train, y_train)
                joblib.dump(fitted, model_path)
                provenance = {
                    "dataset": dataset_name,
                    "feature_config": feature_config,
                    "model": model_name,
                    "fold": int(fold),
                    "feature_columns": list(feature_columns),
                    "group_column": group_column,
                    "train_groups": sorted(train_groups),
                    "validation_groups": sorted(validation_groups),
                    "exported_from_saved_oof_fold_assignments": True,
                }
                _write_json_new(provenance_path, provenance)
                del fitted
                exported_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "fold": int(fold),
                        "status": "exported",
                        "training_groups": len(train_groups),
                        "validation_groups": len(validation_groups),
                    }
                )
    return pd.DataFrame(exported_rows)


def initialize_resource_experiment(
    project_root,
    training_repetitions=10,
    inference_repetitions=10,
    inference_passes=10,
    sampling_interval_seconds=0.1,
):
    root = Path(project_root).resolve()
    processed_dir = root / "Data" / "Processed"
    results_root = root / "Results"
    run_id = (
        datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    run_root = results_root / "resource_measurements" / run_id
    run_root.mkdir(parents=True, exist_ok=False)

    logical_cores = psutil.cpu_count(logical=True) or 1
    physical_cores = psutil.cpu_count(logical=False)
    virtual_memory = psutil.virtual_memory()
    windows_gpus = _windows_gpu_names()
    torch_gpus = (
        [
            {"index": index, "name": torch.cuda.get_device_name(index)}
            for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else []
    )
    frequency = psutil.cpu_freq()
    environment = {
        "run_id": run_id,
        "measurement_datetime": datetime.now(timezone.utc).astimezone().isoformat(),
        "operating_system": platform.platform(),
        "system": platform.system(),
        "processor": platform.processor() or platform.machine(),
        "physical_cpu_cores": physical_cores,
        "logical_cpu_threads": logical_cores,
        "cpu_frequency_mhz": frequency.current if frequency else None,
        "total_ram_gb": round(virtual_memory.total / 1024**3, 3),
        "available_ram_at_start_gb": round(virtual_memory.available / 1024**3, 3),
        "windows_gpu_names": windows_gpus,
        "pytorch_cuda_available": torch.cuda.is_available(),
        "pytorch_cuda_devices": torch_gpus,
        "python_version": sys.version,
        "library_versions": _library_versions(),
        "process_memory_measurement": (
            "Python process RSS, sampled periodically with psutil; sampling can miss brief peaks."
        ),
        "process_cpu_measurement": (
            "Python process user/system CPU-time delta divided by task wall time and logical CPU threads; "
            "not system CPU utilization."
        ),
        "sampling_interval_seconds": sampling_interval_seconds,
    }
    _write_json_new(run_root / "environment.json", environment)
    energy_counter = EnergyCounter()
    model_factories = None
    shared_model_hyperparameters = None
    run_metadata = {
        "run_id": run_id,
        "environment_file": "environment.json",
        "training_repetitions": training_repetitions,
        "inference_repetitions": inference_repetitions,
        "inference_passes_per_repetition": inference_passes,
        "inference_definition": (
            "Grouped-CV held-out rows are predicted with their already-fitted fold model; "
            "model loading is excluded from timed prediction."
        ),
        "training_definition": (
            "Fresh estimator instance and fit on the full selected prepared dataset; "
            "inference fold models are not retrained."
        ),
        "preprocessing": (
            "Feature columns and estimator pipeline come from Notebook 04 metadata; "
            "Logistic Regression StandardScaler is included."
        ),
        "datasets": {},
    }

    datasets = {}
    for dataset_name, file_name in DATASET_FILES.items():
        dataset_path = processed_dir / file_name
        model_dir = processed_dir / "models" / dataset_name
        metadata_path = model_dir / "training_metadata.json"
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Prepared dataset is missing: {dataset_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Run Notebook 04 first; missing {metadata_path}")
        model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_hyperparameters = model_metadata.get("model_hyperparameters")
        if not model_hyperparameters:
            raise ValueError(
                f"Selected model hyperparameters are missing from {metadata_path}; "
                "rerun the updated Notebook 04 to save reproducible settings."
            )
        if shared_model_hyperparameters is None:
            shared_model_hyperparameters = model_hyperparameters
            model_factories = make_model_factories(model_hyperparameters)
        elif model_hyperparameters != shared_model_hyperparameters:
            raise ValueError(
                "Validated and significant datasets have different model settings; "
                "run them as separate measurement configurations."
            )
        selected_config = model_metadata["feature_config"]
        feature_columns = model_metadata["feature_columns"]
        frame = pd.read_parquet(dataset_path)
        missing = sorted(set(feature_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"{dataset_name} data lacks selected features: {missing}")
        X = frame[feature_columns].astype(float)
        y = frame["energy_winner"].astype(int)
        if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
            raise ValueError(f"{dataset_name} feature matrix has missing/non-finite values.")
        for model_name in MODEL_NAMES:
            model_path = model_dir / f"{model_name}.joblib"
            if not model_path.is_file():
                raise FileNotFoundError(f"Notebook 04 selected model is missing: {model_path}")

        oof_path = results_root / dataset_name / "model_oof_predictions.parquet"
        if not oof_path.is_file():
            raise FileNotFoundError(f"Notebook 04 OOF predictions are missing: {oof_path}")
        oof = pd.read_parquet(oof_path)
        oof = oof.loc[
            oof["feature_config"].eq(selected_config)
            & oof["model"].eq("logistic_regression")
        ].copy()
        if oof.empty:
            raise ValueError(f"No OOF rows found for {dataset_name}/{selected_config}.")
        keys = ["problem_id", "submission_id_v0", "submission_id_v1"]
        if oof.duplicated(keys).any():
            raise ValueError(f"{dataset_name} OOF rows contain duplicate pair identifiers.")
        fold_model_dir = model_dir / selected_config / "fold_models"
        if not fold_model_dir.is_dir():
            raise FileNotFoundError(
                f"Held-out inference requires fold models from Notebook 04: "
                f"{fold_model_dir}. Run its standalone fold-model export cell; this does not "
                "repeat model comparison or hyperparameter tuning."
            )

        fold_data = {}
        for fold in sorted(oof["fold"].unique()):
            fold_rows = oof.loc[oof["fold"].eq(fold), keys]
            inference_frame = fold_rows.merge(
                frame[keys + feature_columns],
                on=keys,
                how="left",
                validate="one_to_one",
                indicator=True,
            )
            if not inference_frame["_merge"].eq("both").all():
                raise ValueError(f"Could not match {dataset_name} OOF fold {fold} to features.")
            inference_frame = inference_frame.drop(columns="_merge")
            for model_name in MODEL_NAMES:
                model_path = fold_model_dir / f"{model_name}_fold_{fold}.joblib"
                provenance_path = fold_model_dir / f"{model_name}_fold_{fold}.json"
                if not model_path.is_file() or not provenance_path.is_file():
                    raise FileNotFoundError(
                        f"Missing Notebook 04 fold model or provenance for "
                        f"{dataset_name}/{model_name}/fold {fold}."
                    )
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                test_groups = set(inference_frame["problem_id"].astype(str))
                train_groups = set(provenance["train_groups"])
                saved_validation_groups = set(provenance["validation_groups"])
                if test_groups & train_groups:
                    raise ValueError(
                        f"{dataset_name} fold {fold} inference groups overlap model training groups."
                    )
                if test_groups != saved_validation_groups:
                    raise ValueError(
                        f"{dataset_name} fold {fold} OOF groups differ from model provenance."
                    )
            fold_data[int(fold)] = inference_frame

        dataset_run = {
            "dataset_name": dataset_name,
            "frame": frame,
            "X": X,
            "y": y,
            "feature_columns": feature_columns,
            "feature_config": selected_config,
            "model_dir": model_dir,
            "fold_model_dir": fold_model_dir,
            "fold_data": fold_data,
            "energy_counter": energy_counter,
            "logical_cores": logical_cores,
            "sampling_interval_seconds": sampling_interval_seconds,
            "inference_passes": inference_passes,
        }
        datasets[dataset_name] = dataset_run
        dataset_root = run_root / dataset_name
        (dataset_root / "training").mkdir(parents=True, exist_ok=True)
        (dataset_root / "inference").mkdir(parents=True, exist_ok=True)
        run_metadata["datasets"][dataset_name] = {
            "prepared_dataset": str(dataset_path.relative_to(root)),
            "pairs": int(len(frame)),
            "problem_groups": int(frame["group_id"].nunique()),
            "feature_config": selected_config,
            "feature_columns": feature_columns,
            "model_hyperparameters": model_hyperparameters,
            "inference_groups_per_fold": {
                str(fold): int(part["problem_id"].nunique())
                for fold, part in fold_data.items()
            },
            "inference_pairs": int(sum(len(part) for part in fold_data.values())),
        }
        _write_json_new(
            dataset_root / "experiment_metadata.json",
            run_metadata["datasets"][dataset_name],
        )

    state = {
        "root": root,
        "run_id": run_id,
        "run_root": run_root,
        "environment": environment,
        "run_metadata": run_metadata,
        "datasets": datasets,
        "energy_counter": energy_counter,
        "model_factories": model_factories,
        "training_repetitions": training_repetitions,
        "inference_repetitions": inference_repetitions,
        "inference_passes": inference_passes,
        "logical_cores": logical_cores,
        "sampling_interval_seconds": sampling_interval_seconds,
    }
    return state


def run_training_measurements(state):
    rows = []
    idle_rows = []
    for dataset_name, dataset in state["datasets"].items():
        for model_name in MODEL_NAMES:
            for repetition in range(1, state["training_repetitions"] + 1):
                def fit_fresh_model(
                    model_name=model_name, X=dataset["X"], y=dataset["y"]
                ):
                    model = state["model_factories"][model_name]()
                    model.fit(X, y)
                    return model

                measurement, fitted_model = _sample_workload(
                    fit_fresh_model,
                    state["energy_counter"],
                    state["logical_cores"],
                    state["sampling_interval_seconds"],
                )
                measurement.update(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "feature_config": dataset["feature_config"],
                        "task": "training",
                        "repetition": repetition,
                        "n_training_samples": len(dataset["X"]),
                        "n_features": dataset["X"].shape[1],
                    }
                )
                idle = _paired_idle(
                    measurement["elapsed_seconds"],
                    dataset,
                    model_name,
                    dataset["feature_config"],
                    "training",
                    repetition,
                )
                measurement["idle_baseline_energy_joules"] = idle[
                    "actual_energy_joules"
                ]
                measurement["net_actual_energy_joules"] = (
                    measurement["actual_energy_joules"]
                    - idle["actual_energy_joules"]
                    if measurement["actual_energy_joules"] is not None
                    and idle["actual_energy_joules"] is not None
                    else None
                )
                rows.append(measurement)
                idle_rows.append(idle)
                del fitted_model
                print(
                    f"{dataset_name}/{model_name} training "
                    f"{repetition}/{state['training_repetitions']}: "
                    f"{measurement['elapsed_seconds']:.3f}s"
                )

    measurements = pd.DataFrame(rows)
    idle_measurements = pd.DataFrame(idle_rows)
    for dataset_name in state["datasets"]:
        output = state["run_root"] / dataset_name / "training"
        _write_csv_new(
            output / "training_repetitions.csv",
            measurements.loc[measurements["dataset"].eq(dataset_name)],
        )
        _write_csv_new(
            output / "idle_baseline_repetitions.csv",
            idle_measurements.loc[idle_measurements["dataset"].eq(dataset_name)],
        )
    state["training_measurements"] = measurements
    state["training_idle_measurements"] = idle_measurements
    return measurements, idle_measurements


def run_inference_measurements(state):
    rows = []
    fold_rows = []
    idle_rows = []
    for dataset_name, dataset in state["datasets"].items():
        for model_name in MODEL_NAMES:
            for repetition in range(1, state["inference_repetitions"] + 1):
                repetition_folds = []
                repetition_idle = []
                for fold, inference_frame in dataset["fold_data"].items():
                    model_path = (
                        dataset["fold_model_dir"]
                        / f"{model_name}_fold_{fold}.joblib"
                    )
                    model = joblib.load(model_path)
                    X_test = inference_frame[dataset["feature_columns"]].astype(float)
                    prediction_count = len(X_test) * state["inference_passes"]

                    def predict_heldout(
                        model=model,
                        X_test=X_test,
                        passes=state["inference_passes"],
                    ):
                        for _ in range(passes):
                            model.predict(X_test)

                    measurement, _ = _sample_workload(
                        predict_heldout,
                        state["energy_counter"],
                        state["logical_cores"],
                        state["sampling_interval_seconds"],
                    )
                    measurement.update(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "feature_config": dataset["feature_config"],
                            "task": "inference",
                            "repetition": repetition,
                            "fold": fold,
                            "n_unique_samples": len(X_test),
                            "n_predictions": prediction_count,
                            "inference_passes": state["inference_passes"],
                            "average_inference_time_per_sample_seconds": (
                                measurement["elapsed_seconds"] / prediction_count
                            ),
                            "predictions_per_second": (
                                prediction_count / measurement["elapsed_seconds"]
                                if measurement["elapsed_seconds"]
                                else np.nan
                            ),
                        }
                    )
                    idle = _paired_idle(
                        measurement["elapsed_seconds"],
                        dataset,
                        model_name,
                        dataset["feature_config"],
                        "inference",
                        repetition,
                        fold=fold,
                    )
                    measurement["idle_baseline_energy_joules"] = idle[
                        "actual_energy_joules"
                    ]
                    measurement["net_actual_energy_joules"] = (
                        measurement["actual_energy_joules"]
                        - idle["actual_energy_joules"]
                        if measurement["actual_energy_joules"] is not None
                        and idle["actual_energy_joules"] is not None
                        else None
                    )
                    fold_rows.append(measurement)
                    idle_rows.append(idle)
                    repetition_folds.append(measurement)
                    repetition_idle.append(idle)
                    del model

                total_elapsed = sum(row["elapsed_seconds"] for row in repetition_folds)
                total_predictions = sum(row["n_predictions"] for row in repetition_folds)
                energies = [row["actual_energy_joules"] for row in repetition_folds]
                idle_energies = [
                    row["actual_energy_joules"] for row in repetition_idle
                ]
                actual_energy = (
                    sum(energies)
                    if all(value is not None for value in energies)
                    else None
                )
                idle_energy = (
                    sum(idle_energies)
                    if all(value is not None for value in idle_energies)
                    else None
                )
                cpu_parts = [
                    row
                    for row in repetition_folds
                    if np.isfinite(
                        row["average_process_cpu_utilization_pct_of_total"]
                    )
                ]
                average_cpu = (
                    float(
                        np.average(
                            [
                                row[
                                    "average_process_cpu_utilization_pct_of_total"
                                ]
                                for row in cpu_parts
                            ],
                            weights=[row["elapsed_seconds"] for row in cpu_parts],
                        )
                    )
                    if cpu_parts
                    else np.nan
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "feature_config": dataset["feature_config"],
                        "task": "inference",
                        "repetition": repetition,
                        "n_unique_samples": sum(
                            row["n_unique_samples"] for row in repetition_folds
                        ),
                        "n_predictions": total_predictions,
                        "inference_passes": state["inference_passes"],
                        "elapsed_seconds": total_elapsed,
                        "average_inference_time_per_sample_seconds": (
                            total_elapsed / total_predictions
                        ),
                        "predictions_per_second": (
                            total_predictions / total_elapsed
                            if total_elapsed
                            else np.nan
                        ),
                        "average_process_cpu_utilization_pct_of_total": average_cpu,
                        "baseline_process_memory_mb": max(
                            row["baseline_process_memory_mb"]
                            for row in repetition_folds
                        ),
                        "peak_process_memory_mb": max(
                            row["peak_process_memory_mb"]
                            for row in repetition_folds
                        ),
                        "memory_increase_during_task_mb": max(
                            row["memory_increase_during_task_mb"]
                            for row in repetition_folds
                        ),
                        "cpu_sample_count": sum(
                            row["cpu_sample_count"] for row in repetition_folds
                        ),
                        "memory_sample_count": sum(
                            row["memory_sample_count"] for row in repetition_folds
                        ),
                        "actual_energy_joules": actual_energy,
                        "idle_baseline_energy_joules": idle_energy,
                        "net_actual_energy_joules": (
                            actual_energy - idle_energy
                            if actual_energy is not None and idle_energy is not None
                            else None
                        ),
                        "energy_source": state["energy_counter"].backend,
                        "energy_scope": state["energy_counter"].scope,
                        "actual_energy_measurement": (
                            state["energy_counter"].backend is not None
                        ),
                    }
                )
                print(
                    f"{dataset_name}/{model_name} inference "
                    f"{repetition}/{state['inference_repetitions']}: "
                    f"{total_elapsed:.3f}s for {total_predictions:,} predictions"
                )

    measurements = pd.DataFrame(rows)
    fold_measurements = pd.DataFrame(fold_rows)
    idle_measurements = pd.DataFrame(idle_rows)
    for dataset_name in state["datasets"]:
        output = state["run_root"] / dataset_name / "inference"
        _write_csv_new(
            output / "inference_repetitions.csv",
            measurements.loc[measurements["dataset"].eq(dataset_name)],
        )
        _write_csv_new(
            output / "inference_by_fold.csv",
            fold_measurements.loc[fold_measurements["dataset"].eq(dataset_name)],
        )
        _write_csv_new(
            output / "idle_baseline_repetitions.csv",
            idle_measurements.loc[idle_measurements["dataset"].eq(dataset_name)],
        )
    state["inference_measurements"] = measurements
    state["inference_fold_measurements"] = fold_measurements
    state["inference_idle_measurements"] = idle_measurements
    return measurements, fold_measurements, idle_measurements


def create_manual_meter_template(run_root):
    path = Path(run_root) / "external_power_meter_manual_entry.csv"
    if not path.exists():
        _write_csv_new(
            path,
            pd.DataFrame(
                columns=[
                    "initial_energy_reading",
                    "final_energy_reading",
                    "meter_unit",
                    "measurement_duration_seconds",
                    "experiment_name",
                    "model_name",
                    "task",
                    "repetition",
                    "notes",
                ]
            ),
        )
    return path


def process_manual_meter_entries(template_path, results_path):
    template_path = Path(template_path)
    results_path = Path(results_path)
    entries = pd.read_csv(template_path)
    if entries.empty:
        return pd.DataFrame()
    if results_path.exists():
        raise FileExistsError(f"Refusing to overwrite manual-meter results: {results_path}")
    required = [
        "initial_energy_reading",
        "final_energy_reading",
        "meter_unit",
        "measurement_duration_seconds",
        "experiment_name",
        "model_name",
        "task",
        "repetition",
    ]
    missing = sorted(set(required) - set(entries.columns))
    if missing:
        raise ValueError(f"Manual meter template is missing columns: {missing}")
    if entries[required].isna().any().any():
        raise ValueError("Complete every required manual meter field before processing.")
    factors = {
        "j": 1.0,
        "joule": 1.0,
        "joules": 1.0,
        "wh": 3600.0,
        "kwh": 3_600_000.0,
        "mwh": 3.6,
    }
    rows = []
    for _, entry in entries.iterrows():
        unit = str(entry["meter_unit"]).strip().lower()
        if unit not in factors:
            raise ValueError(
                f"Unsupported meter unit {entry['meter_unit']!r}; use J, Wh, kWh, or mWh."
            )
        initial = float(entry["initial_energy_reading"])
        final = float(entry["final_energy_reading"])
        duration = float(entry["measurement_duration_seconds"])
        if not np.isfinite([initial, final, duration]).all():
            raise ValueError("Meter readings and duration must be finite numbers.")
        if final < initial or duration <= 0:
            raise ValueError("Final reading must not be below initial; duration must be positive.")
        task = str(entry["task"]).strip().lower()
        if task not in {"training", "inference", "idle"}:
            raise ValueError("Manual meter task must be training, inference, or idle.")
        repetition = int(entry["repetition"])
        if repetition < 1:
            raise ValueError("Repetition number must be at least one.")
        rows.append(
            {
                **entry.to_dict(),
                "task": task,
                "repetition": repetition,
                "energy_difference_in_meter_units": final - initial,
                "actual_energy_joules": (final - initial) * factors[unit],
                "energy_source": "External power meter (manual entry)",
                "energy_scope": (
                    "Whole-system wall energy; not CPU/package energy."
                ),
                "measurement_type": "manual external-meter reading",
            }
        )
    result = pd.DataFrame(rows)
    _write_csv_new(results_path, result)
    return result


def summarize_resources(frame, task):
    if frame.empty:
        return pd.DataFrame()
    selected = frame.loc[frame["task"].eq(task)]
    if selected.empty:
        return pd.DataFrame()
    aggregations = {
        "repetitions": ("repetition", "nunique"),
        "mean_runtime_seconds": ("elapsed_seconds", "mean"),
        "std_runtime_seconds": ("elapsed_seconds", "std"),
        "mean_process_cpu_utilization_pct_of_total": (
            "average_process_cpu_utilization_pct_of_total",
            "mean",
        ),
        "mean_peak_process_memory_mb": ("peak_process_memory_mb", "mean"),
        "std_peak_process_memory_mb": ("peak_process_memory_mb", "std"),
        "mean_memory_increase_mb": ("memory_increase_during_task_mb", "mean"),
        "mean_actual_energy_joules": ("actual_energy_joules", "mean"),
        "std_actual_energy_joules": ("actual_energy_joules", "std"),
        "mean_net_actual_energy_joules": ("net_actual_energy_joules", "mean"),
        "std_net_actual_energy_joules": ("net_actual_energy_joules", "std"),
    }
    if task == "inference":
        aggregations.update(
            {
                "mean_seconds_per_prediction": (
                    "average_inference_time_per_sample_seconds",
                    "mean",
                ),
                "mean_predictions_per_second": ("predictions_per_second", "mean"),
            }
        )
    return selected.groupby(
        ["dataset", "model", "feature_config"], as_index=False
    ).agg(**aggregations)


def save_resource_summaries(state):
    training = summarize_resources(state["training_measurements"], "training")
    inference = summarize_resources(state["inference_measurements"], "inference")
    for dataset_name in state["datasets"]:
        root = state["run_root"] / dataset_name
        _write_csv_new(
            root / "training" / "training_summary.csv",
            training.loc[training["dataset"].eq(dataset_name)],
        )
        _write_csv_new(
            root / "inference" / "inference_summary.csv",
            inference.loc[inference["dataset"].eq(dataset_name)],
        )
        status = {
            "run_id": state["run_id"],
            "dataset": dataset_name,
            "runtime_and_process_resource_measurements_available": True,
            "actual_energy_available": state["energy_counter"].backend is not None,
            "actual_energy_source": state["energy_counter"].backend,
            "actual_energy_scope": state["energy_counter"].scope,
            "training_repetitions_requested": state["training_repetitions"],
            "inference_repetitions_requested": state["inference_repetitions"],
            "inference_passes_per_repetition": state["inference_passes"],
            "measurement_datetime": state["environment"]["measurement_datetime"],
            "results_directory": str(root.relative_to(state["root"])),
        }
        _write_json_new(root / "measurement_status.json", status)
    _write_json_new(state["run_root"] / "run_metadata.json", state["run_metadata"])
    return training, inference


def run_codecarbon_estimates(state, repetitions=1):
    try:
        from codecarbon import EmissionsTracker
    except ImportError as exc:
        raise ImportError(
            "Install CodeCarbon to enable estimates: pip install codecarbon"
        ) from exc

    rows = []
    for dataset_name, dataset in state["datasets"].items():
        for model_name in MODEL_NAMES:
            estimator_by_fold = {
                fold: joblib.load(
                    dataset["fold_model_dir"]
                    / f"{model_name}_fold_{fold}.joblib"
                )
                for fold in dataset["fold_data"]
            }

            def infer_with_saved_folds(estimators=estimator_by_fold, run=dataset):
                for fold, part in run["fold_data"].items():
                    features = part[run["feature_columns"]].astype(float)
                    for _ in range(state["inference_passes"]):
                        estimators[fold].predict(features)

            workloads = {
                "training": lambda model_name=model_name, X=dataset["X"], y=dataset["y"]: (
                    state["model_factories"][model_name]().fit(X, y)
                ),
                "inference": infer_with_saved_folds,
            }
            for task_name, workload in workloads.items():
                for repetition in range(1, repetitions + 1):
                    tracker = EmissionsTracker(
                        project_name="energy_aware_recommendation",
                        save_to_file=False,
                        save_to_logger=False,
                        save_to_api=False,
                        log_level="error",
                    )
                    started = time.perf_counter()
                    tracker.start()
                    try:
                        workload()
                    finally:
                        tracker.stop()
                    elapsed = time.perf_counter() - started
                    energy_kwh = tracker.final_emissions_data.energy_consumed
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "task": task_name,
                            "repetition": repetition,
                            "elapsed_seconds": elapsed,
                            "estimated_energy_joules": float(energy_kwh) * 3_600_000,
                            "measurement_source": "CodeCarbon estimate",
                            "is_estimate": True,
                            "tracker_configuration": json.dumps(
                                {
                                    "project_name": "energy_aware_recommendation",
                                    "save_to_file": False,
                                    "model_loading_included": False,
                                }
                            ),
                        }
                    )
            del estimator_by_fold
    estimates = pd.DataFrame(rows)
    for dataset_name in state["datasets"]:
        for task_name in ("training", "inference"):
            _write_csv_new(
                state["run_root"]
                / dataset_name
                / task_name
                / "codecarbon_estimates.csv",
                estimates.loc[
                    estimates["dataset"].eq(dataset_name)
                    & estimates["task"].eq(task_name)
                ],
            )
    return estimates


def run_codebert_resource_measurements(
    state,
    training_repetitions=1,
    inference_repetitions=10,
):
    from experiment_utils import codebert_measurement_tasks

    rows = []
    for dataset_name, dataset in state["datasets"].items():
        tasks = codebert_measurement_tasks(
            state["root"] / "Data" / "Processed" / DATASET_FILES[dataset_name],
            dataset["model_dir"],
            force_cpu=True,
        )
        task_repetitions = {
            "training": training_repetitions,
            "inference": inference_repetitions,
        }
        for task_name, repetitions in task_repetitions.items():
            for repetition in range(1, repetitions + 1):
                measurement, _ = _sample_workload(
                    tasks[task_name],
                    state["energy_counter"],
                    state["logical_cores"],
                    state["sampling_interval_seconds"],
                )
                measurement.update(
                    {
                        "dataset": dataset_name,
                        "model": "codebert_frozen_embeddings_logistic_regression",
                        "task": task_name,
                        "repetition": repetition,
                        "n_inference_pairs": (
                            tasks["inference_pairs"]
                            if task_name == "inference"
                            else None
                        ),
                        "measurement_device": str(tasks["device"]),
                        "encoder_loading_included": False,
                        "training_includes_embedding_extraction": (
                            task_name == "training"
                        ),
                        "gpu_monitoring_included": False,
                    }
                )
                if state["energy_counter"].backend is not None:
                    idle = _paired_idle(
                        measurement["elapsed_seconds"],
                        dataset,
                        "codebert_frozen_embeddings_logistic_regression",
                        "codebert_embeddings",
                        task_name,
                        repetition,
                    )
                    idle["n_inference_pairs"] = (
                        tasks["inference_pairs"] if task_name == "inference" else None
                    )
                    measurement["idle_baseline_energy_joules"] = idle[
                        "actual_energy_joules"
                    ]
                    measurement["net_actual_energy_joules"] = (
                        measurement["actual_energy_joules"]
                        - idle["actual_energy_joules"]
                        if measurement["actual_energy_joules"] is not None
                        and idle["actual_energy_joules"] is not None
                        else None
                    )
                    rows.append(idle)
                else:
                    measurement["idle_baseline_energy_joules"] = None
                    measurement["net_actual_energy_joules"] = None
                rows.append(measurement)
        del tasks

    measurements = pd.DataFrame(rows)
    for dataset_name in state["datasets"]:
        output = state["run_root"] / "codebert" / dataset_name
        _write_csv_new(
            output / "codebert_resource_measurements.csv",
            measurements.loc[measurements["dataset"].eq(dataset_name)],
        )
        _write_json_new(
            output / "measurement_status.json",
            {
                "dataset": dataset_name,
                "run_id": state["run_id"],
                "encoder_frozen": True,
                "training_includes_embedding_extraction": True,
                "encoder_loading_included": False,
                "gpu_available_to_pytorch": torch.cuda.is_available(),
                "measurement_device": "cpu",
                "gpu_monitoring_included": False,
                "comparison_with_tabular_models": (
                    "Keep separate; encoder and GPU resource-accounting conditions differ."
                ),
                "energy_source": state["energy_counter"].backend,
                "actual_energy_available": state["energy_counter"].backend is not None,
            },
        )
    return measurements

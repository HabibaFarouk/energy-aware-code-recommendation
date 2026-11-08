import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_tabular_configurations(
    dataset_name,
    frame,
    feature_configurations,
    model_factories,
    selected_feature_config,
    results_dir,
    model_dir,
    n_splits=5,
):
    if selected_feature_config not in feature_configurations:
        raise ValueError(f"Unknown selected feature configuration: {selected_feature_config}")
    required = {"problem_id", "group_id", "energy_winner"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{dataset_name} data is missing columns: {missing}")

    target = frame["energy_winner"].astype(int)
    groups = frame["group_id"]
    if groups.nunique() < n_splits or target.nunique() != 2:
        raise ValueError(
            f"{dataset_name} requires at least {n_splits} problem groups and both target classes."
        )

    fold_rows = []
    oof_frames = []
    selected_models = {}
    selected_feature_columns = {}
    for config_name, columns in feature_configurations.items():
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"{dataset_name}/{config_name} data is missing features: {missing}")
        features = frame[columns].astype(float)
        if features.isna().any().any() or not np.isfinite(features.to_numpy()).all():
            raise ValueError(f"{dataset_name}/{config_name} has missing or non-finite features.")

        config_model_dir = model_dir / config_name
        config_model_dir.mkdir(parents=True, exist_ok=True)
        for model_name, factory in model_factories.items():
            print(f"Evaluating {model_name} on {dataset_name}/{config_name}...")
            for fold, (train_idx, valid_idx) in enumerate(
                GroupKFold(n_splits=n_splits).split(features, target, groups), start=1
            ):
                train_features = features.iloc[train_idx]
                active_columns = train_features.columns[
                    train_features.nunique(dropna=False) > 1
                ].tolist()
                if not active_columns:
                    raise ValueError(
                        f"All {dataset_name}/{config_name} features are constant in fold {fold}."
                    )

                model = factory()
                model.fit(train_features[active_columns], target.iloc[train_idx])
                truth = target.iloc[valid_idx]
                valid_features = features.iloc[valid_idx][active_columns]
                prediction = model.predict(valid_features).astype(int)
                try:
                    probability = model.predict_proba(valid_features)[:, 1]
                except (AttributeError, IndexError):
                    probability = np.full(len(valid_idx), np.nan)

                fold_rows.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "feature_config": config_name,
                    "fold": fold,
                    "accuracy": accuracy_score(truth, prediction),
                    "roc_auc": (
                        roc_auc_score(truth, probability)
                        if truth.nunique() == 2 and np.isfinite(probability).all()
                        else np.nan
                    ),
                    "balanced_accuracy": balanced_accuracy_score(truth, prediction),
                    "f1": f1_score(truth, prediction, zero_division=0),
                    "train_groups": int(groups.iloc[train_idx].nunique()),
                    "validation_groups": int(groups.iloc[valid_idx].nunique()),
                    "constant_features_removed": len(columns) - len(active_columns),
                })
                oof_frames.append(pd.DataFrame({
                    "dataset": dataset_name,
                    "model": model_name,
                    "feature_config": config_name,
                    "fold": fold,
                    "problem_id": frame.iloc[valid_idx]["problem_id"].to_numpy(),
                    "submission_id_v0": frame.iloc[valid_idx]["submission_id_v0"].to_numpy(),
                    "submission_id_v1": frame.iloc[valid_idx]["submission_id_v1"].to_numpy(),
                    "y_true": truth.to_numpy(),
                    "y_pred": prediction,
                    "y_probability": probability,
                    "result_energy_v0": frame.iloc[valid_idx]["result_energy_v0"].to_numpy(),
                    "result_energy_v1": frame.iloc[valid_idx]["result_energy_v1"].to_numpy(),
                    "was_swapped": frame.iloc[valid_idx]["was_swapped"].to_numpy(),
                }))

            active_columns = features.columns[features.nunique(dropna=False) > 1].tolist()
            if not active_columns:
                raise ValueError(f"All {dataset_name}/{config_name} features are constant.")
            final_model = factory()
            final_model.fit(features[active_columns], target)
            joblib.dump(final_model, config_model_dir / f"{model_name}.joblib")
            config_metadata = {
                "dataset": dataset_name,
                "feature_config": config_name,
                "feature_columns": active_columns,
                "zero_variance_features_removed": [
                    column for column in columns if column not in active_columns
                ],
                "target": "energy_winner",
                "group_column": "group_id",
                "group_kfold_splits": n_splits,
                "models": list(model_factories),
            }
            (config_model_dir / "training_metadata.json").write_text(
                json.dumps(config_metadata, indent=2), encoding="utf-8"
            )
            if config_name == selected_feature_config:
                selected_models[model_name] = final_model
                selected_feature_columns[model_name] = active_columns
                joblib.dump(final_model, model_dir / f"{model_name}.joblib")

    fold_metrics = pd.DataFrame(fold_rows)
    performance = fold_metrics.groupby(
        ["dataset", "model", "feature_config"], as_index=False
    ).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(results_dir / "model_fold_metrics.csv", index=False)
    performance.to_csv(results_dir / "model_performance.csv", index=False)
    pd.concat(oof_frames, ignore_index=True).to_parquet(
        results_dir / "model_oof_predictions.parquet", index=False
    )
    return performance, selected_models, selected_feature_columns


def train_codebert_dataset(
    dataset_name: str,
    dataset_path: Path,
    results_dir: Path,
    model_dir: Path,
    model_id: str = "microsoft/codebert-base",
    max_length: int = 256,
    batch_size: int = 8,
):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "The CodeBERT experiments require torch and transformers in this Python environment."
        ) from exc

    required = [
        "problem_id",
        "submission_id_v0",
        "submission_id_v1",
        "code_v0_no_empty_lines",
        "code_v1_no_empty_lines",
        "result_energy_v0",
        "result_energy_v1",
    ]
    frame = pd.read_parquet(dataset_path)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{dataset_name} data is missing columns: {missing}")
    frame = frame.dropna(subset=required).copy().reset_index(drop=True)
    code_columns = ["code_v0_no_empty_lines", "code_v1_no_empty_lines"]
    for column in code_columns:
        frame[column] = frame[column].astype(str)
        frame = frame.loc[frame[column].str.strip().ne("")]
    frame = frame.loc[frame["result_energy_v0"].ne(frame["result_energy_v1"])].reset_index(drop=True)
    frame["energy_winner"] = (frame["result_energy_v1"] < frame["result_energy_v0"]).astype("int8")
    if frame["problem_id"].nunique() < 5 or frame["energy_winner"].nunique() != 2:
        raise ValueError(f"{dataset_name} requires at least five problems and both target classes.")

    all_code = pd.concat([frame[code_columns[0]], frame[code_columns[1]]], ignore_index=True)
    unique_code = pd.unique(all_code)
    code_to_index = {code: index for index, code in enumerate(unique_code)}
    v0_index = np.fromiter(
        (code_to_index[code] for code in frame[code_columns[0]]),
        dtype=np.int64,
        count=len(frame),
    )
    v1_index = np.fromiter(
        (code_to_index[code] for code in frame[code_columns[1]]),
        dtype=np.int64,
        count=len(frame),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoder = AutoModel.from_pretrained(model_id).to(device)
    encoder.eval()
    started = time.perf_counter()
    embeddings = np.empty((len(unique_code), encoder.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(unique_code), batch_size):
            batch_text = unique_code[start : start + batch_size].tolist()
            tokens = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {name: tensor.to(device) for name, tensor in tokens.items()}
            hidden = encoder(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embeddings[start : start + len(batch_text)] = pooled.cpu().numpy()
    embedding_seconds = time.perf_counter() - started

    embedding_v0, embedding_v1 = embeddings[v0_index], embeddings[v1_index]
    features = np.concatenate(
        [embedding_v0, embedding_v1, embedding_v1 - embedding_v0], axis=1
    )
    target = frame["energy_winner"].to_numpy(dtype=np.int8)
    groups = frame["problem_id"].to_numpy()

    def make_head():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
        )

    fold_rows = []
    oof_rows = []
    for fold, (train_idx, valid_idx) in enumerate(
        GroupKFold(n_splits=5).split(features, target, groups), start=1
    ):
        head = make_head()
        head.fit(features[train_idx], target[train_idx])
        truth = target[valid_idx]
        prediction = head.predict(features[valid_idx]).astype(int)
        probability = head.predict_proba(features[valid_idx])[:, 1]
        fold_rows.append(
            {
                "model": "codebert_frozen_embeddings_logistic_regression",
                "feature_config": "codebert_embeddings",
                "fold": fold,
                "roc_auc": roc_auc_score(truth, probability) if np.unique(truth).size == 2 else np.nan,
                "balanced_accuracy": balanced_accuracy_score(truth, prediction),
                "f1": f1_score(truth, prediction, zero_division=0),
                "train_groups": int(pd.Series(groups[train_idx]).nunique()),
                "validation_groups": int(pd.Series(groups[valid_idx]).nunique()),
            }
        )
        oof_rows.append(
            pd.DataFrame(
                {
                    "model": "codebert_frozen_embeddings_logistic_regression",
                    "feature_config": "codebert_embeddings",
                    "fold": fold,
                    "problem_id": frame.iloc[valid_idx]["problem_id"].to_numpy(),
                    "submission_id_v0": frame.iloc[valid_idx]["submission_id_v0"].to_numpy(),
                    "submission_id_v1": frame.iloc[valid_idx]["submission_id_v1"].to_numpy(),
                    "y_true": truth,
                    "y_pred": prediction,
                    "y_probability": probability,
                    "result_energy_v0": frame.iloc[valid_idx]["result_energy_v0"].to_numpy(),
                    "result_energy_v1": frame.iloc[valid_idx]["result_energy_v1"].to_numpy(),
                    "was_swapped": (
                        frame.iloc[valid_idx]["was_swapped"].to_numpy()
                        if "was_swapped" in frame
                        else np.zeros(len(valid_idx), dtype=bool)
                    ),
                }
            )
        )

    fold_metrics = pd.DataFrame(fold_rows)
    performance = pd.DataFrame(
        [
            {
                "model": "codebert_frozen_embeddings_logistic_regression",
                "feature_config": "codebert_embeddings",
                "roc_auc_mean": fold_metrics["roc_auc"].mean(),
                "roc_auc_std": fold_metrics["roc_auc"].std(),
                "balanced_accuracy_mean": fold_metrics["balanced_accuracy"].mean(),
                "balanced_accuracy_std": fold_metrics["balanced_accuracy"].std(),
                "f1_mean": fold_metrics["f1"].mean(),
                "f1_std": fold_metrics["f1"].std(),
            }
        ]
    )
    final_head = make_head()
    final_head.fit(features, target)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_head, model_dir / "codebert_head.joblib")
    fold_metrics.to_csv(results_dir / "codebert_fold_metrics.csv", index=False)
    performance.to_csv(results_dir / "codebert_performance.csv", index=False)
    pd.concat(oof_rows, ignore_index=True).to_parquet(
        results_dir / "codebert_oof_predictions.parquet", index=False
    )
    metadata = {
        "dataset": dataset_name,
        "model_id": model_id,
        "encoder_frozen": True,
        "pooling": "attention-mask mean pooling",
        "max_length": max_length,
        "classifier": "StandardScaler + class-balanced LogisticRegression",
        "feature_layout": ["embedding_v0", "embedding_v1", "embedding_v1_minus_v0"],
        "n_pairs": int(len(frame)),
        "n_problem_groups": int(frame["problem_id"].nunique()),
        "class_counts": {
            str(key): int(value) for key, value in frame["energy_winner"].value_counts().items()
        },
        "group_column": "problem_id",
        "group_kfold_splits": 5,
        "embedding_seconds": embedding_seconds,
        "embedding_unique_code_count": int(len(unique_code)),
    }
    (model_dir / "codebert_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"{dataset_name}: {len(frame):,} pairs, {len(unique_code):,} unique code samples; "
        f"embedding time {embedding_seconds:.1f}s"
    )
    del encoder, tokenizer, embeddings, features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return performance


def codebert_measurement_tasks(
    dataset_path: Path,
    model_dir: Path,
    max_length: int = 256,
    batch_size: int = 8,
    inference_pairs: int = 256,
):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "The CodeBERT measurements require torch and transformers in this Python environment."
        ) from exc

    metadata = json.loads((model_dir / "codebert_metadata.json").read_text(encoding="utf-8"))
    head = joblib.load(model_dir / "codebert_head.joblib")
    frame = pd.read_parquet(dataset_path)
    frame = frame.loc[
        frame["result_energy_v0"].ne(frame["result_energy_v1"])
        & frame["code_v0_no_empty_lines"].notna()
        & frame["code_v1_no_empty_lines"].notna()
    ].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No non-tied code pairs available for {dataset_path}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(metadata["model_id"])
    encoder = AutoModel.from_pretrained(metadata["model_id"]).to(device).eval()
    all_text = pd.concat(
        [frame["code_v0_no_empty_lines"], frame["code_v1_no_empty_lines"]],
        ignore_index=True,
    ).astype(str)
    unique_text = pd.unique(all_text)
    text_index = {text: index for index, text in enumerate(unique_text)}
    index_v0 = np.fromiter(
        (text_index[text] for text in frame["code_v0_no_empty_lines"].astype(str)),
        dtype=np.int64,
        count=len(frame),
    )
    index_v1 = np.fromiter(
        (text_index[text] for text in frame["code_v1_no_empty_lines"].astype(str)),
        dtype=np.int64,
        count=len(frame),
    )
    inference_frame = frame.sample(n=min(inference_pairs, len(frame)), random_state=42)

    def encode(texts):
        vectors = np.empty((len(texts), encoder.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                tokens = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                tokens = {name: tensor.to(device) for name, tensor in tokens.items()}
                hidden = encoder(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                vectors[start : start + len(batch)] = pooled.cpu().numpy()
        return vectors

    def training_task():
        vectors = encode(unique_text)
        vector_v0, vector_v1 = vectors[index_v0], vectors[index_v1]
        features = np.concatenate([vector_v0, vector_v1, vector_v1 - vector_v0], axis=1)
        target = (frame["result_energy_v1"] < frame["result_energy_v0"]).astype(int)
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
        )
        estimator.fit(features, target)

    def inference_task():
        count = len(inference_frame)
        texts = pd.concat(
            [
                inference_frame["code_v0_no_empty_lines"],
                inference_frame["code_v1_no_empty_lines"],
            ],
            ignore_index=True,
        ).astype(str).tolist()
        vectors = encode(texts)
        features = np.concatenate(
            [vectors[:count], vectors[count:], vectors[count:] - vectors[:count]],
            axis=1,
        )
        head.predict_proba(features)

    return {
        "training": training_task,
        "inference": inference_task,
        "inference_pairs": len(inference_frame),
        "encoder": encoder,
        "tokenizer": tokenizer,
        "torch": torch,
    }

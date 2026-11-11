import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _validate_codebert_max_length(max_length, tokenizer, encoder):
    if not isinstance(max_length, int) or not 1 <= max_length <= 512:
        raise ValueError("Standard CodeBERT input length must be between 1 and 512 tokens.")

    max_positions = getattr(encoder.config, "max_position_embeddings", None)
    padding_idx = getattr(encoder.config, "pad_token_id", 0) or 0
    if max_positions is not None:
        model_limit = max_positions - padding_idx - 1
        if max_length > model_limit:
            raise ValueError(
                f"CodeBERT supports at most {model_limit} input tokens, not {max_length}."
            )

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and tokenizer_limit < 1_000_000:
        if max_length > tokenizer_limit:
            raise ValueError(
                f"The tokenizer supports at most {tokenizer_limit} input tokens, "
                f"not {max_length}."
            )


def train_tabular_configurations(
    dataset_name,
    frame,
    feature_configurations,
    model_factories,
    selected_feature_config,
    results_dir,
    model_dir,
    n_splits=5,
    save_fold_models=False,
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
                model = factory()
                model.fit(train_features, target.iloc[train_idx])
                if save_fold_models:
                    fold_model_dir = config_model_dir / "fold_models"
                    fold_model_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(
                        model,
                        fold_model_dir / f"{model_name}_fold_{fold}.joblib",
                    )
                    fold_metadata = {
                        "dataset": dataset_name,
                        "feature_config": config_name,
                        "model": model_name,
                        "fold": fold,
                        "feature_columns": list(columns),
                        "group_column": "group_id",
                        "train_groups": sorted(
                            groups.iloc[train_idx].astype(str).unique().tolist()
                        ),
                        "validation_groups": sorted(
                            groups.iloc[valid_idx].astype(str).unique().tolist()
                        ),
                    }
                    (fold_model_dir / f"{model_name}_fold_{fold}.json").write_text(
                        json.dumps(fold_metadata, indent=2), encoding="utf-8"
                    )
                truth = target.iloc[valid_idx]
                valid_features = features.iloc[valid_idx]
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

            final_model = factory()
            final_model.fit(features, target)
            joblib.dump(final_model, config_model_dir / f"{model_name}.joblib")
            config_metadata = {
                "dataset": dataset_name,
                "feature_config": config_name,
                "feature_columns": list(columns),
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
                selected_feature_columns[model_name] = list(columns)
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


def run_grouped_config_b_ablation(
    validated_frame,
    significant_frame,
    feature_columns,
    model_factories,
    results_dir,
    n_repeats=5,
    test_size=0.2,
    bottom_percent=0.2,
    minimum_low_importance_fits=3,
    random_state=42,
):
    dataset_frames = {
        "validated": validated_frame,
        "significant": significant_frame,
    }
    required = {"group_id", "energy_winner"}
    for dataset_name, frame in dataset_frames.items():
        missing = sorted(required - set(frame.columns))
        missing_features = sorted(set(feature_columns) - set(frame.columns))
        if missing or missing_features:
            raise ValueError(
                f"{dataset_name} data is missing columns: {missing + missing_features}"
            )
        if frame["group_id"].isna().any() or frame["energy_winner"].nunique() != 2:
            raise ValueError(f"{dataset_name} requires non-null groups and both target classes.")

    shared_groups = set(validated_frame["group_id"]) & set(significant_frame["group_id"])
    significant_groups = set(significant_frame["group_id"])
    if significant_groups != shared_groups:
        raise ValueError("Significant problem groups must all be present in validated data.")
    if len(shared_groups) < 5:
        raise ValueError("At least five shared problem groups are required for grouped splits.")
    model_names = [
        name for name in ("logistic_regression", "random_forest", "xgboost")
        if name in model_factories
    ]
    if len(model_names) != 3:
        raise ValueError("Importance-based selection requires Logistic Regression, Random Forest, and XGBoost.")
    if not 0 < bottom_percent < 1:
        raise ValueError("bottom_percent must be between zero and one.")
    if not 1 <= minimum_low_importance_fits <= len(dataset_frames) * len(model_names):
        raise ValueError("minimum_low_importance_fits is outside the number of dataset/model fits.")

    unique_groups = np.array(sorted(shared_groups, key=str), dtype=object)
    splitter = GroupShuffleSplit(
        n_splits=n_repeats,
        test_size=test_size,
        random_state=random_state,
    )
    fold_rows = []
    selection_rows = []
    for repeat, (_, test_group_indices) in enumerate(
        splitter.split(unique_groups, groups=unique_groups), start=1
    ):
        held_out_groups = set(unique_groups[test_group_indices])
        trained_models = {}
        importance_by_run = []
        test_frames = {}
        train_frames = {}

        for dataset_name, frame in dataset_frames.items():
            test_mask = frame["group_id"].isin(held_out_groups)
            train_frames[dataset_name] = frame.loc[~test_mask].reset_index(drop=True)
            test_frames[dataset_name] = frame.loc[test_mask].reset_index(drop=True)
            if train_frames[dataset_name].empty or test_frames[dataset_name].empty:
                raise ValueError(f"{dataset_name} has an empty train or held-out split.")

            train_x = train_frames[dataset_name][feature_columns].astype(float)
            train_y = train_frames[dataset_name]["energy_winner"].astype(int)
            if train_x.isna().any().any() or not np.isfinite(train_x.to_numpy()).all():
                raise ValueError(f"{dataset_name} training features contain invalid values.")
            if train_y.nunique() != 2:
                raise ValueError(f"{dataset_name} training split must contain both target classes.")

            for model_name in model_names:
                model = model_factories[model_name]()
                model.fit(train_x, train_y)
                trained_models[(dataset_name, model_name)] = model
                if model_name == "logistic_regression":
                    importance = np.abs(model[-1].coef_).mean(axis=0)
                else:
                    importance = model.feature_importances_
                importance_rank = pd.Series(importance, index=feature_columns).rank(
                    method="average", pct=True
                )
                low_features = set(
                    importance_rank[importance_rank <= bottom_percent].index
                )
                importance_by_run.append((dataset_name, model_name, low_features))
                for feature in feature_columns:
                    selection_rows.append({
                        "repeat": repeat,
                        "dataset": dataset_name,
                        "model": model_name,
                        "feature": feature,
                        "importance": float(importance[feature_columns.index(feature)]),
                        "rank_percentile": float(importance_rank[feature]),
                        "bottom_percent": feature in low_features,
                    })

        low_counts = pd.Series(0, index=feature_columns, dtype=int)
        low_datasets = {feature: set() for feature in feature_columns}
        for dataset_name, model_name, low_features in importance_by_run:
            for feature in low_features:
                low_counts[feature] += 1
                low_datasets[feature].add(dataset_name)
        removed_features = [
            feature for feature in feature_columns
            if low_counts[feature] >= minimum_low_importance_fits
            and len(low_datasets[feature]) == len(dataset_frames)
        ]
        for row in selection_rows:
            if row["repeat"] == repeat:
                row["selected_for_ablation"] = row["feature"] in removed_features

        for dataset_name in dataset_frames:
            test_frame = test_frames[dataset_name]
            test_x = test_frame[feature_columns].astype(float)
            truth = test_frame["energy_winner"].astype(int)
            ablated_columns = [
                feature for feature in feature_columns if feature not in removed_features
            ]
            for model_name in model_names:
                baseline_model = trained_models[(dataset_name, model_name)]
                baseline_prediction = baseline_model.predict(test_x).astype(int)
                baseline_probability = baseline_model.predict_proba(test_x)[:, 1]
                ablated_model = model_factories[model_name]()
                train_x = train_frames[dataset_name][ablated_columns].astype(float)
                train_y = train_frames[dataset_name]["energy_winner"].astype(int)
                ablated_model.fit(train_x, train_y)
                ablated_x = test_frame[ablated_columns].astype(float)
                ablated_prediction = ablated_model.predict(ablated_x).astype(int)
                ablated_probability = ablated_model.predict_proba(ablated_x)[:, 1]

                for condition, prediction, probability in [
                    ("all_features", baseline_prediction, baseline_probability),
                    ("importance_ablation", ablated_prediction, ablated_probability),
                ]:
                    fold_rows.append({
                        "repeat": repeat,
                        "dataset": dataset_name,
                        "model": model_name,
                        "condition": condition,
                        "removed_features": ",".join(removed_features),
                        "n_removed_features": len(removed_features),
                        "n_train_pairs": len(train_frames[dataset_name]),
                        "n_test_pairs": len(test_frame),
                        "n_test_groups": int(test_frame["group_id"].nunique()),
                        "accuracy": accuracy_score(truth, prediction),
                        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
                        "f1": f1_score(truth, prediction, zero_division=0),
                        "roc_auc": (
                            roc_auc_score(truth, probability)
                            if truth.nunique() == 2
                            else np.nan
                        ),
                    })

    fold_metrics = pd.DataFrame(fold_rows)
    summary = fold_metrics.groupby(
        ["dataset", "model", "condition"], as_index=False
    ).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        mean_removed_features=("n_removed_features", "mean"),
    )
    selection = pd.DataFrame(selection_rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(results_dir / "feature_ablation_folds.csv", index=False)
    summary.to_csv(results_dir / "feature_ablation_summary.csv", index=False)
    selection.to_csv(results_dir / "feature_ablation_selection.csv", index=False)
    return summary, fold_metrics, selection


def train_codebert_dataset(
    dataset_name: str,
    dataset_path: Path,
    results_dir: Path,
    model_dir: Path,
    model_id: str = "microsoft/codebert-base",
    max_length: int = 512,
    batch_size: int = 8,
    force_cpu: bool = False,
):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "The CodeBERT experiments require torch and transformers in this Python environment."
        ) from exc

    if not isinstance(max_length, int) or not 1 <= max_length <= 512:
        raise ValueError("Standard CodeBERT input length must be between 1 and 512 tokens.")

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
    if "energy_winner" in frame.columns:
        frame = frame.dropna(subset=[*required, "energy_winner"]).copy()
    else:
        frame = frame.dropna(subset=required).copy()
    code_columns = ["code_v0_no_empty_lines", "code_v1_no_empty_lines"]
    for column in code_columns:
        frame[column] = frame[column].astype(str)
        frame = frame.loc[frame[column].str.strip().ne("")]
    frame = frame.loc[
        frame["result_energy_v0"].ne(frame["result_energy_v1"])
    ].copy().reset_index(drop=True)
    calculated_target = (
        frame["result_energy_v1"] < frame["result_energy_v0"]
    ).astype("int8")
    if "energy_winner" in frame and not frame["energy_winner"].astype(int).equals(
        calculated_target.astype(int)
    ):
        raise ValueError(f"{dataset_name} saved energy_winner labels are inconsistent.")
    frame["energy_winner"] = calculated_target
    pair_keys = ["problem_id", "submission_id_v0", "submission_id_v1"]
    if frame.duplicated(pair_keys).any():
        raise ValueError(f"{dataset_name} contains duplicate prepared code pairs.")
    if "group_id" in frame and not frame["group_id"].astype(str).equals(
        frame["problem_id"].astype(str)
    ):
        raise ValueError(f"{dataset_name} group_id does not match problem_id.")
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

    device = torch.device(
        "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoder = AutoModel.from_pretrained(model_id).to(device)
    _validate_codebert_max_length(max_length, tokenizer, encoder)
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
    model_dir.mkdir(parents=True, exist_ok=True)
    fold_model_dir = model_dir / "fold_models"
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    for fold, (train_idx, valid_idx) in enumerate(
        GroupKFold(n_splits=5).split(features, target, groups), start=1
    ):
        head = make_head()
        head.fit(features[train_idx], target[train_idx])
        train_groups = sorted(pd.Series(groups[train_idx]).astype(str).unique().tolist())
        validation_groups = sorted(pd.Series(groups[valid_idx]).astype(str).unique().tolist())
        joblib.dump(head, fold_model_dir / f"codebert_head_fold_{fold}.joblib")
        (fold_model_dir / f"codebert_head_fold_{fold}.json").write_text(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "fold": fold,
                    "group_column": "problem_id",
                    "train_groups": train_groups,
                    "validation_groups": validation_groups,
                    "max_length": max_length,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        truth = target[valid_idx]
        prediction = head.predict(features[valid_idx]).astype(int)
        probability = head.predict_proba(features[valid_idx])[:, 1]
        fold_rows.append(
            {
                "model": "codebert_frozen_embeddings_logistic_regression",
                "feature_config": "codebert_embeddings",
                "fold": fold,
                "accuracy": accuracy_score(truth, prediction),
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
                "accuracy_mean": fold_metrics["accuracy"].mean(),
                "accuracy_std": fold_metrics["accuracy"].std(),
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
        "special_tokens_added_by_tokenizer": True,
        "positional_embedding_limit": 512,
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
        "device": str(device),
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
    results_dir: Path,
    max_length: int | None = None,
    batch_size: int = 8,
    inference_pairs: int = 100,
    force_cpu: bool = False,
):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "The CodeBERT measurements require torch and transformers in this Python environment."
        ) from exc

    if not isinstance(inference_pairs, int) or inference_pairs < 1:
        raise ValueError("inference_pairs must be a positive integer.")

    metadata = json.loads((model_dir / "codebert_metadata.json").read_text(encoding="utf-8"))
    trained_max_length = int(metadata["max_length"])
    if max_length is not None and max_length != trained_max_length:
        raise ValueError(
            "CodeBERT measurement input length must match the trained model metadata."
        )
    max_length = trained_max_length
    frame = pd.read_parquet(dataset_path)
    required = [
        "problem_id",
        "submission_id_v0",
        "submission_id_v1",
        "code_v0_no_empty_lines",
        "code_v1_no_empty_lines",
        "result_energy_v0",
        "result_energy_v1",
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"CodeBERT measurement data is missing columns: {missing}")
    frame = frame.loc[
        frame["result_energy_v0"].ne(frame["result_energy_v1"])
        & frame["code_v0_no_empty_lines"].notna()
        & frame["code_v1_no_empty_lines"].notna()
    ].reset_index(drop=True)
    for column in ("code_v0_no_empty_lines", "code_v1_no_empty_lines"):
        frame[column] = frame[column].astype(str)
        frame = frame.loc[frame[column].str.strip().ne("")].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No non-tied code pairs available for {dataset_path}.")
    pair_keys = ["problem_id", "submission_id_v0", "submission_id_v1"]
    if frame.duplicated(pair_keys).any():
        raise ValueError("CodeBERT measurement data contains duplicate prepared pairs.")
    if "group_id" in frame and not frame["group_id"].astype(str).equals(
        frame["problem_id"].astype(str)
    ):
        raise ValueError("CodeBERT measurement group_id does not match problem_id.")

    device = torch.device(
        "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(metadata["model_id"])
    encoder = AutoModel.from_pretrained(metadata["model_id"]).to(device).eval()
    _validate_codebert_max_length(max_length, tokenizer, encoder)
    fold_model_dir = model_dir / "fold_models"
    oof_path = Path(results_dir) / "codebert_oof_predictions.parquet"
    if not oof_path.is_file():
        raise FileNotFoundError(f"CodeBERT out-of-fold predictions are missing: {oof_path}")
    oof = pd.read_parquet(oof_path)
    if oof.empty or oof.duplicated(pair_keys).any():
        raise ValueError("CodeBERT OOF predictions must contain unique held-out pairs.")
    expected_pairs = set(
        map(tuple, frame[pair_keys].itertuples(index=False, name=None))
    )
    observed_pairs = set(
        map(tuple, oof[pair_keys].itertuples(index=False, name=None))
    )
    if expected_pairs != observed_pairs:
        raise ValueError("CodeBERT OOF predictions do not cover the current prepared dataset.")
    if not oof.groupby("problem_id")["fold"].nunique().eq(1).all():
        raise ValueError("A CodeBERT problem group appears in multiple OOF folds.")
    inference_frame = oof[pair_keys + ["fold", "y_true"]].merge(
        frame[
            pair_keys
            + [
                "code_v0_no_empty_lines",
                "code_v1_no_empty_lines",
                "result_energy_v0",
                "result_energy_v1",
            ]
        ],
        on=pair_keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not inference_frame["_merge"].eq("both").all():
        raise ValueError("Could not match CodeBERT OOF pairs to their prepared source code.")
    expected_target = (
        inference_frame["result_energy_v1"] < inference_frame["result_energy_v0"]
    ).astype(int)
    if not inference_frame["y_true"].astype(int).equals(expected_target):
        raise ValueError("CodeBERT OOF labels do not match current prepared energy values.")
    inference_frame = inference_frame.sample(
        n=min(inference_pairs, len(inference_frame)), random_state=42
    ).drop(columns="_merge").reset_index(drop=True)
    heads_by_fold = {}
    for fold in sorted(inference_frame["fold"].unique()):
        model_path = fold_model_dir / f"codebert_head_fold_{int(fold)}.joblib"
        provenance_path = fold_model_dir / f"codebert_head_fold_{int(fold)}.json"
        if not model_path.is_file() or not provenance_path.is_file():
            raise FileNotFoundError(
                f"Missing CodeBERT fold model or provenance for fold {int(fold)}."
            )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        fold_groups = set(
            inference_frame.loc[
                inference_frame["fold"].eq(fold), "problem_id"
            ].astype(str)
        )
        if (
            fold_groups & set(provenance["train_groups"])
            or not fold_groups.issubset(set(provenance["validation_groups"]))
        ):
            raise ValueError(
                f"CodeBERT fold {int(fold)} OOF groups are inconsistent with provenance."
            )
        heads_by_fold[int(fold)] = joblib.load(model_path)

    all_text = pd.concat(
        [
            inference_frame["code_v0_no_empty_lines"],
            inference_frame["code_v1_no_empty_lines"],
        ],
        ignore_index=True,
    ).astype(str)
    unique_text = pd.unique(all_text)
    text_index = {text: index for index, text in enumerate(unique_text)}
    index_v0 = np.fromiter(
        (text_index[text] for text in inference_frame["code_v0_no_empty_lines"].astype(str)),
        dtype=np.int64,
        count=len(inference_frame),
    )
    index_v1 = np.fromiter(
        (text_index[text] for text in inference_frame["code_v1_no_empty_lines"].astype(str)),
        dtype=np.int64,
        count=len(inference_frame),
    )

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
        vectors = encode(unique_text)
        for fold, fold_frame in inference_frame.groupby("fold", sort=True):
            row_indices = fold_frame.index.to_numpy()
            features = np.concatenate(
                [
                    vectors[index_v0[row_indices]],
                    vectors[index_v1[row_indices]],
                    vectors[index_v1[row_indices]] - vectors[index_v0[row_indices]],
                ],
                axis=1,
            )
            heads_by_fold[int(fold)].predict_proba(features)

    return {
        "training": training_task,
        "inference": inference_task,
        "inference_pairs": len(inference_frame),
        "dataset_pairs": len(frame),
        "inference_sample_seed": 42,
        "inference_groups": int(inference_frame["problem_id"].nunique()),
        "device": device,
        "encoder": encoder,
        "tokenizer": tokenizer,
        "torch": torch,
    }

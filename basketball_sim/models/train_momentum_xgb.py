"""Train an XGBoost momentum classifier from precomputed world-model latents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb


def binary_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    accuracy = (tp + tn) / max(1, len(labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    eps = 1e-7
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    logloss = float(-(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)).mean())
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "logloss": logloss,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "positive_rate": float(predictions.mean()),
        "mean_probability": float(probabilities.mean()),
    }


def threshold_search(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    best_f1 = None
    prior_match = None
    actual_positive_rate = float(labels.mean())
    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = binary_metrics(probabilities, labels, float(threshold))
        if best_f1 is None or metrics["f1"] > best_f1["metrics"]["f1"]:
            best_f1 = {"threshold": float(threshold), "metrics": metrics}
        diff = abs(metrics["positive_rate"] - actual_positive_rate)
        if prior_match is None or diff < prior_match["diff"]:
            prior_match = {"threshold": float(threshold), "metrics": metrics, "diff": diff}
    return {
        "actual_positive_rate": actual_positive_rate,
        "best_f1": best_f1,
        "prior_matching": prior_match,
    }


def resolve_threshold(value: str, search: dict) -> float:
    if value == "best-f1":
        return float(search["best_f1"]["threshold"])
    if value == "prior-match":
        return float(search["prior_matching"]["threshold"])
    return float(value)


def load_metadata(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents-path", default="models/momentum_xgb/momentum_latents.npz")
    parser.add_argument("--metadata-path", default="models/momentum_xgb/momentum_latents_config.json")
    parser.add_argument("--output-dir", default="models/momentum_xgb")
    parser.add_argument("--xgb-rounds", type=int, default=450)
    parser.add_argument("--early-stopping-rounds", type=int, default=35)
    parser.add_argument("--eta", type=float, default=0.035)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--subsample", type=float, default=0.92)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--threshold", default="best-f1", help="Numeric value, best-f1, or prior-match.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data = np.load(args.latents_path)
    train_latents = data["train_latents"].astype(np.float32)
    val_latents = data["validation_latents"].astype(np.float32)
    train_labels = data["train_labels"].astype(np.float32)
    val_labels = data["validation_labels"].astype(np.float32)

    pos = float(train_labels.sum())
    neg = float(len(train_labels) - pos)
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "eta": args.eta,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": 2.0,
        "lambda": 1.0,
        "alpha": 0.05,
        "tree_method": "hist",
        "seed": args.seed,
        "scale_pos_weight": neg / max(1.0, pos),
    }
    train_matrix = xgb.DMatrix(train_latents, label=train_labels)
    val_matrix = xgb.DMatrix(val_latents, label=val_labels)
    booster = xgb.train(
        params,
        train_matrix,
        num_boost_round=args.xgb_rounds,
        evals=[(train_matrix, "train"), (val_matrix, "validation")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=25,
    )

    train_prob = booster.predict(train_matrix, iteration_range=(0, booster.best_iteration + 1))
    val_prob = booster.predict(val_matrix, iteration_range=(0, booster.best_iteration + 1))
    search = threshold_search(val_prob, val_labels)
    threshold = resolve_threshold(args.threshold, search)
    train_metrics = binary_metrics(train_prob, train_labels, threshold)
    val_metrics = binary_metrics(val_prob, val_labels, threshold)
    print(f"train metrics: {json.dumps(train_metrics, indent=2)}")
    print(f"validation metrics: {json.dumps(val_metrics, indent=2)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "momentum_xgb.json"
    booster.save_model(model_path)

    metadata = load_metadata(Path(args.metadata_path) if args.metadata_path else None)
    metadata.update(
        {
            "threshold": threshold,
            "threshold_policy": args.threshold,
            "threshold_search": search,
            "best_iteration": int(booster.best_iteration),
            "best_score": float(booster.best_score),
            "xgb_params": params,
            "xgb_training": vars(args),
            "metrics": {
                "train": train_metrics,
                "validation": val_metrics,
            },
        }
    )
    with (output_dir / "momentum_xgb_config.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"saved: {model_path.resolve()}")


if __name__ == "__main__":
    main()

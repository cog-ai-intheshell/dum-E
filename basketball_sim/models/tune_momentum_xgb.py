"""Tune the momentum XGBoost classifier with Optuna.

The default objective is deliberately precision/false-positive aware. It
selects a threshold for each trial, requires a minimum recall, then rewards
precision-heavy F-beta while penalizing false positives among true negatives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import xgboost as xgb

from basketball_sim.models.train_momentum_xgb import (
    binary_metrics,
    load_metadata,
    threshold_search,
)


def fbeta_score(precision: float, recall: float, beta: float) -> float:
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if denominator <= 1e-12:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denominator


def fp_aware_score(metrics: dict, negative_count: int, min_recall: float, beta: float, fp_penalty: float) -> float:
    fp_rate = metrics["confusion"]["fp"] / max(1, negative_count)
    score = fbeta_score(metrics["precision"], metrics["recall"], beta) - fp_penalty * fp_rate
    if metrics["recall"] < min_recall:
        score -= 10.0 * (min_recall - metrics["recall"])
    return float(score)


def threshold_search_fp_aware(
    probabilities: np.ndarray,
    labels: np.ndarray,
    min_recall: float,
    beta: float,
    fp_penalty: float,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
) -> dict:
    labels = labels.astype(np.int64)
    negative_count = int((labels == 0).sum())
    best = None
    for threshold in np.linspace(threshold_min, threshold_max, threshold_steps):
        metrics = binary_metrics(probabilities, labels, float(threshold))
        score = fp_aware_score(metrics, negative_count, min_recall, beta, fp_penalty)
        candidate = {"threshold": float(threshold), "score": score, "metrics": metrics}
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


def suggest_xgb_params(trial: optuna.Trial, train_labels: np.ndarray, seed: int, nthread: int) -> dict:
    pos = float(train_labels.sum())
    neg = float(len(train_labels) - pos)
    base_scale_pos_weight = neg / max(1.0, pos)

    return {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "tree_method": "hist",
        "verbosity": 0,
        "seed": seed,
        "nthread": nthread,
        "eta": trial.suggest_float("eta", 0.006, 0.12, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 18.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.58, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-6, 6.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-3, 30.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-5, 8.0, log=True),
        "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 8.0),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight",
            max(0.25, 0.35 * base_scale_pos_weight),
            max(0.5, 1.35 * base_scale_pos_weight),
            log=True,
        ),
    }


def train_booster(params: dict, train_matrix: xgb.DMatrix, val_matrix: xgb.DMatrix, args) -> xgb.Booster:
    return xgb.train(
        params,
        train_matrix,
        num_boost_round=args.xgb_rounds,
        evals=[(train_matrix, "train"), (val_matrix, "validation")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=False,
    )


def predict_with_best_iteration(booster: xgb.Booster, matrix: xgb.DMatrix) -> np.ndarray:
    best_iteration = int(getattr(booster, "best_iteration", 0))
    return booster.predict(matrix, iteration_range=(0, best_iteration + 1))


def plain_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents-path", default="models/momentum_xgb/momentum_latents.npz")
    parser.add_argument("--metadata-path", default="models/momentum_xgb/momentum_latents_config.json")
    parser.add_argument("--output-dir", default="models/momentum_xgb")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=0, help="Optuna timeout in seconds. 0 = no timeout.")
    parser.add_argument("--xgb-rounds", type=int, default=700)
    parser.add_argument("--early-stopping-rounds", type=int, default=45)
    parser.add_argument("--min-recall", type=float, default=0.60)
    parser.add_argument("--beta", type=float, default=0.45, help="F-beta beta. Values < 1 favor precision.")
    parser.add_argument("--fp-penalty", type=float, default=0.35)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=181)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nthread", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.latents_path)
    train_latents = data["train_latents"].astype(np.float32)
    val_latents = data["validation_latents"].astype(np.float32)
    train_labels = data["train_labels"].astype(np.float32)
    val_labels = data["validation_labels"].astype(np.float32)

    train_matrix = xgb.DMatrix(train_latents, label=train_labels)
    val_matrix = xgb.DMatrix(val_latents, label=val_labels)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params(trial, train_labels, args.seed, args.nthread)
        booster = train_booster(params, train_matrix, val_matrix, args)
        val_prob = predict_with_best_iteration(booster, val_matrix)
        selected = threshold_search_fp_aware(
            val_prob,
            val_labels,
            args.min_recall,
            args.beta,
            args.fp_penalty,
            args.threshold_min,
            args.threshold_max,
            args.threshold_steps,
        )
        trial.set_user_attr("threshold", selected["threshold"])
        trial.set_user_attr("metrics", selected["metrics"])
        trial.set_user_attr("best_iteration", int(booster.best_iteration))
        trial.set_user_attr("best_score", float(booster.best_score))
        return float(selected["score"])

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, timeout=None if args.timeout <= 0 else args.timeout)

    best_params = suggest_xgb_params(study.best_trial, train_labels, args.seed, args.nthread)
    booster = train_booster(best_params, train_matrix, val_matrix, args)
    train_prob = predict_with_best_iteration(booster, train_matrix)
    val_prob = predict_with_best_iteration(booster, val_matrix)
    selected = threshold_search_fp_aware(
        val_prob,
        val_labels,
        args.min_recall,
        args.beta,
        args.fp_penalty,
        args.threshold_min,
        args.threshold_max,
        args.threshold_steps,
    )
    threshold = float(selected["threshold"])
    train_metrics = binary_metrics(train_prob, train_labels, threshold)
    val_metrics = binary_metrics(val_prob, val_labels, threshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "momentum_xgb.json"
    booster.save_model(model_path)

    metadata = load_metadata(Path(args.metadata_path) if args.metadata_path else None)
    metadata.update(
        {
            "definition": "momentum = 1 when the shot is made.",
            "threshold": threshold,
            "threshold_policy": "optuna-fp-aware",
            "threshold_search": {
                "fp_aware": selected,
                "best_f1": threshold_search(val_prob, val_labels)["best_f1"],
                "prior_matching": threshold_search(val_prob, val_labels)["prior_matching"],
            },
            "best_iteration": int(booster.best_iteration),
            "best_score": float(booster.best_score),
            "xgb_params": best_params,
            "xgb_training": vars(args),
            "optuna": {
                "best_value": float(study.best_value),
                "best_trial": int(study.best_trial.number),
                "objective": {
                    "min_recall": args.min_recall,
                    "beta": args.beta,
                    "fp_penalty": args.fp_penalty,
                },
                "trials": [
                    {
                        "number": trial.number,
                        "value": None if trial.value is None else float(trial.value),
                        "params": trial.params,
                        "threshold": trial.user_attrs.get("threshold"),
                        "metrics": trial.user_attrs.get("metrics"),
                        "best_iteration": trial.user_attrs.get("best_iteration"),
                        "best_score": trial.user_attrs.get("best_score"),
                    }
                    for trial in study.trials
                    if trial.state == optuna.trial.TrialState.COMPLETE
                ],
            },
            "metrics": {
                "train": train_metrics,
                "validation": val_metrics,
            },
        }
    )
    with (output_dir / "momentum_xgb_config.json").open("w", encoding="utf-8") as file:
        json.dump(plain_json(metadata), file, indent=2)

    print(f"best trial: {study.best_trial.number}")
    print(f"best objective: {study.best_value:.6f}")
    print(f"selected threshold: {threshold:.3f}")
    print(f"train metrics: {json.dumps(plain_json(train_metrics), indent=2)}")
    print(f"validation metrics: {json.dumps(plain_json(val_metrics), indent=2)}")
    print(f"saved: {model_path.resolve()}")


if __name__ == "__main__":
    main()


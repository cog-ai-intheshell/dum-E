"""Train and save a Temporal Transformer world model.

Example quick smoke run:

    .venv/bin/python -m basketball_sim.models.train_world_model \
      --config dataset_config_test --max-shots 2 --epochs 2 \
      --d-model 48 --num-heads 12 --num-layers 1

Recommended larger run:

    .venv/bin/python -m basketball_sim.models.train_world_model \
      --config dataset_config --max-shots 400 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from basketball_sim.dataset.generate import apply_config, simulate_shot, wind_at
from basketball_sim.dataset.grid import ShotParams, iter_parameter_combinations, load_config_module
from basketball_sim.models.features import (
    CONTEXT_FEATURES,
    SEQUENCE_FEATURES,
    TARGET_FEATURES,
    apply_scaler,
    fit_scaler,
    make_windows,
)
from basketball_sim.models.world_model import TemporalTransformerWorldModel, WorldModelConfig


PHYSICAL_KEY_FIELDS = (
    "initial_force",
    "vertical_angle",
    "horizontal_angle",
    "distance_to_hoop",
    "boy_height",
    "ball_mass",
    "hoop_height",
    "wind_regime_id",
    "wind_strength",
    "wind_orientation",
    "wind_vertical_orientation",
    "wind_spatial_coupling",
    "drag_coeff",
)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def physical_key(params: ShotParams) -> tuple:
    return tuple(getattr(params, field) for field in PHYSICAL_KEY_FIELDS)


def unique_physical_params(config):
    seen = set()
    for params in iter_parameter_combinations(config):
        key = physical_key(params)
        if key in seen:
            continue
        seen.add(key)
        yield params


def ideal_launch_force(distance: float, vertical_angle: float, release_height: float, hoop_height: float, gravity: float):
    theta = math.radians(vertical_angle)
    denominator = 2.0 * math.cos(theta) ** 2 * (distance * math.tan(theta) - (hoop_height - release_height))
    if denominator <= 1e-9:
        return None
    return math.sqrt(gravity * distance * distance / denominator)


def choose(rng: random.Random, values):
    values = list(values)
    return values[rng.randrange(len(values))]


def span(values, fallback_min: float, fallback_max: float) -> tuple[float, float]:
    values = list(values)
    if not values:
        return fallback_min, fallback_max
    return float(min(values)), float(max(values))


def random_augmented_params(config, rng: random.Random) -> ShotParams:
    distance_min, distance_max = span(config.DISTANCE_TO_HOOP_VALUES, 3.5, 8.5)
    wind_max = max([0.0, *[float(value) for value in config.WIND_STRENGTH_VALUES]])
    wind_max = max(5.0, wind_max)

    distance = rng.uniform(max(3.2, distance_min - 1.0), min(9.2, distance_max + 1.0))
    boy_height = choose(rng, config.BOY_HEIGHT_VALUES)
    ball_mass = choose(rng, config.BALL_MASS_VALUES)
    hoop_height = choose(rng, config.HOOP_HEIGHT_VALUES)
    vertical_angle = rng.uniform(38.0, 62.0)
    launch_force = ideal_launch_force(distance, vertical_angle, boy_height * 0.85, hoop_height, config.GRAVITY)
    if launch_force is None:
        launch_force = choose(rng, config.INITIAL_FORCE_VALUES)

    nonzero_regimes = [regime_id for regime_id in config.WIND_REGIME_IDS if regime_id != 0]
    if 0 in config.WIND_REGIME_IDS and rng.random() < 0.48:
        wind_regime_id = 0
        wind_strength = 0.0
        wind_orientation = 0.0
        wind_vertical_orientation = 0.0
        wind_spatial_coupling = 0
    else:
        wind_regime_id = choose(rng, nonzero_regimes or [0])
        wind_strength = rng.triangular(0.4, wind_max, min(2.4, wind_max))
        wind_orientation = rng.uniform(0.0, 360.0)
        wind_vertical_orientation = rng.uniform(-16.0, 16.0)
        wind_spatial_coupling = choose(rng, config.WIND_SPATIAL_COUPLING_VALUES)

    force_offset = rng.triangular(-0.15, 2.25, 0.75) + 0.035 * wind_strength
    initial_force = max(1.0, min(30.0, launch_force + force_offset))
    horizontal_angle = rng.triangular(-2.2, 2.2, 0.0)

    return ShotParams(
        initial_force=initial_force,
        vertical_angle=vertical_angle,
        horizontal_angle=horizontal_angle,
        distance_to_hoop=distance,
        boy_height=boy_height,
        ball_mass=ball_mass,
        hoop_height=hoop_height,
        wind_regime_id=wind_regime_id,
        wind_regime=config.WIND_REGIME_NAMES[wind_regime_id],
        wind_strength=wind_strength,
        wind_orientation=wind_orientation,
        wind_vertical_orientation=wind_vertical_orientation,
        wind_spatial_coupling=wind_spatial_coupling,
        wind_field_visible=1,
        field_view=2,
        past_trail_visible=1,
        drag_coeff=choose(rng, config.DRAG_COEFF_VALUES),
    )


def iter_training_candidates(config, args, rng: random.Random):
    grid_params = None
    grid_index = 0

    if args.augmentation in {"none", "mixed"}:
        grid_params = list(unique_physical_params(config) if args.dedupe_physical else iter_parameter_combinations(config))
        if args.shuffle_candidates:
            rng.shuffle(grid_params)

    if args.augmentation == "none":
        yield from grid_params
        return

    while True:
        use_augmented = args.augmentation == "made-biased" or rng.random() < args.augmented_fraction
        if use_augmented or not grid_params:
            yield random_augmented_params(config, rng)
        else:
            yield grid_params[grid_index % len(grid_params)]
            grid_index += 1


def should_accept_label(label: int, label_counts: dict[int, int], min_label1_fraction: float) -> bool:
    if min_label1_fraction <= 0.0 or label == 1:
        return True
    made = label_counts[1]
    if made == 0:
        return False
    failures = label_counts[0]
    max_failures = math.floor(made * (1.0 - min_label1_fraction) / min_label1_fraction)
    return failures < max_failures


def collect_samples(config, args):
    apply_config(config)
    sequences = []
    contexts = []
    targets = []
    window_shot_ids = []
    shot_labels = []
    label_counts = {0: 0, 1: 0}
    rng = random.Random(args.seed)
    params_iter = iter_training_candidates(config, args, rng)
    target_shots = args.max_shots if args.max_shots > 0 else float("inf")
    candidate_limit = args.candidate_limit if args.candidate_limit > 0 else float("inf")

    generated_shots = 0
    candidates_seen = 0
    for params in params_iter:
        if generated_shots >= target_shots or candidates_seen >= candidate_limit:
            break
        _shot_row, rows, label_row, wind_context = simulate_shot(generated_shots, params)
        candidates_seen += 1
        label = int(label_row["label"])
        if not should_accept_label(label, label_counts, args.min_label1_fraction):
            continue

        shot_sequences, shot_contexts, shot_targets = make_windows(
            params=params,
            rows=rows,
            wind_context=wind_context,
            wind_at=wind_at,
            history_steps=args.history_steps,
            horizon_steps=args.horizon_steps,
            gravity=config.GRAVITY,
        )
        if len(shot_sequences) == 0:
            continue
        sequences.append(shot_sequences)
        contexts.append(shot_contexts)
        targets.append(shot_targets)
        window_shot_ids.append(np.full(len(shot_sequences), generated_shots, dtype=np.int64))
        shot_labels.append(label)
        label_counts[label] += 1
        generated_shots += 1
        if generated_shots % max(1, args.log_every) == 0:
            made_fraction = label_counts[1] / max(1, generated_shots)
            print(
                f"shots used: {generated_shots} "
                f"(label1={label_counts[1]}, ratio={made_fraction:.3f}, candidates={candidates_seen})"
            )

    if not sequences:
        raise RuntimeError("No training windows were produced. Reduce history/horizon or generate longer shots.")
    if generated_shots < target_shots:
        print(f"warning: requested {target_shots} shots, collected {generated_shots} before candidate limit.")

    return (
        np.concatenate(sequences, axis=0),
        np.concatenate(contexts, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(window_shot_ids, axis=0),
        np.asarray(shot_labels, dtype=np.int64),
        generated_shots,
        label_counts,
        candidates_seen,
    )


def split_indices(count: int, validation_fraction: float, seed: int):
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    val_count = max(1, int(count * validation_fraction)) if count > 2 else 0
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)


def split_indices_by_shot(window_shot_ids: np.ndarray, validation_fraction: float, seed: int):
    shot_ids = sorted(int(shot_id) for shot_id in np.unique(window_shot_ids))
    random.Random(seed).shuffle(shot_ids)
    val_count = max(1, int(len(shot_ids) * validation_fraction)) if len(shot_ids) > 2 else 0
    val_shots = set(shot_ids[:val_count])
    train_shots = set(shot_ids[val_count:])

    val_mask = np.asarray([int(shot_id) in val_shots for shot_id in window_shot_ids], dtype=bool)
    train_mask = ~val_mask
    return (
        np.nonzero(train_mask)[0].astype(np.int64),
        np.nonzero(val_mask)[0].astype(np.int64),
        sorted(train_shots),
        sorted(val_shots),
    )


def count_labels_for_shots(shot_labels: np.ndarray, shot_ids: list[int]) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for shot_id in shot_ids:
        label = int(shot_labels[shot_id])
        counts[label] += 1
    return counts


def make_loader(sequences, contexts, targets, indices, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(sequences[indices]),
        torch.from_numpy(contexts[indices]),
        torch.from_numpy(targets[indices]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def parse_metric_horizons(value: str, horizon_steps: int) -> list[int]:
    horizons = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        horizon = int(item)
        if 1 <= horizon <= horizon_steps and horizon not in horizons:
            horizons.append(horizon)
    return horizons or [1, horizon_steps]


def run_epoch(model, loader, optimizer, loss_fn, device):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    for sequence, context, target in loader:
        sequence = sequence.to(device)
        context = context.to(device)
        target = target.to(device)
        with torch.set_grad_enabled(training):
            prediction = model(sequence, context)
            loss = loss_fn(prediction, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_count = sequence.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_count
        total_count += batch_count
    return total_loss / max(1, total_count)


def evaluate_model(model, loader, loss_fn, device, target_scaler: dict, horizons: list[int]) -> dict:
    model.eval()
    target_std = torch.tensor(target_scaler["std"], dtype=torch.float32, device=device).view(1, 1, -1)
    total_loss = 0.0
    total_count = 0
    all_axis_sq = torch.zeros(3, dtype=torch.float32, device=device)
    all_position_sq = torch.zeros((), dtype=torch.float32, device=device)
    all_points = 0
    horizon_stats = {
        horizon: {
            "axis_sq": torch.zeros(3, dtype=torch.float32, device=device),
            "position_sq": torch.zeros((), dtype=torch.float32, device=device),
            "count": 0,
        }
        for horizon in horizons
    }

    with torch.no_grad():
        for sequence, context, target in loader:
            sequence = sequence.to(device)
            context = context.to(device)
            target = target.to(device)
            prediction = model(sequence, context)
            loss = loss_fn(prediction, target)

            batch_count = sequence.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_count
            total_count += batch_count

            error_m = (prediction - target) * target_std
            all_axis_sq += (error_m ** 2).sum(dim=(0, 1))
            all_position_sq += (error_m ** 2).sum(dim=2).sum()
            all_points += batch_count * error_m.shape[1]

            for horizon, stats in horizon_stats.items():
                horizon_error = error_m[:, horizon - 1, :]
                stats["axis_sq"] += (horizon_error ** 2).sum(dim=0)
                stats["position_sq"] += (horizon_error ** 2).sum(dim=1).sum()
                stats["count"] += batch_count

    horizon_rmse = {}
    for horizon, stats in horizon_stats.items():
        count = max(1, stats["count"])
        horizon_rmse[f"t+{horizon}"] = {
            "position_rmse_m": float(torch.sqrt(stats["position_sq"] / count).cpu()),
            "axis_rmse_m": [float(value) for value in torch.sqrt(stats["axis_sq"] / count).cpu().tolist()],
        }

    return {
        "normalized_mse": total_loss / max(1, total_count),
        "overall_position_rmse_m": float(torch.sqrt(all_position_sq / max(1, all_points)).cpu()),
        "overall_axis_rmse_m": [float(value) for value in torch.sqrt(all_axis_sq / max(1, all_points)).cpu().tolist()],
        "horizon_rmse": horizon_rmse,
    }


def format_horizon_metrics(metrics: dict) -> str:
    parts = []
    for horizon, values in metrics["horizon_rmse"].items():
        parts.append(f"{horizon}={values['position_rmse_m']:.3f}m")
    return " ".join(parts)


def save_artifacts(model, config, scalers, args, output_dir: Path, metrics: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "sequence_features": SEQUENCE_FEATURES,
            "context_features": CONTEXT_FEATURES,
            "target_features": TARGET_FEATURES,
        },
        output_dir / "world_model.pt",
    )
    with (output_dir / "world_model_scaler.json").open("w", encoding="utf-8") as file:
        json.dump(scalers, file, indent=2)
    with (output_dir / "world_model_config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "model": config.to_dict(),
                "training": vars(args),
                "metrics": metrics,
                "sequence_features": SEQUENCE_FEATURES,
                "context_features": CONTEXT_FEATURES,
                "target_features": TARGET_FEATURES,
            },
            file,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dataset_config_test")
    parser.add_argument("--output-dir", default="models/world_model")
    parser.add_argument("--max-shots", type=int, default=20, help="0 = all combinations.")
    parser.add_argument("--history-steps", type=int, default=12)
    parser.add_argument("--horizon-steps", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--augmentation", choices=("none", "made-biased", "mixed"), default="none")
    parser.add_argument("--augmented-fraction", type=float, default=0.75)
    parser.add_argument("--min-label1-fraction", type=float, default=0.0)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--dedupe-physical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-by", choices=("shot", "window"), default="shot")
    parser.add_argument("--metric-horizons", default="1,5,10,30")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config_module = load_config_module(args.config)
    print("Collecting training windows...")
    (
        sequences,
        contexts,
        targets,
        window_shot_ids,
        shot_labels,
        generated_shots,
        label_counts,
        candidates_seen,
    ) = collect_samples(config_module, args)
    print(f"shots: {generated_shots}")
    print(f"candidate shots evaluated: {candidates_seen}")
    print(f"label 1 shots: {label_counts[1]} ({label_counts[1] / max(1, generated_shots):.3f})")
    print(f"windows: {len(sequences)}")

    if args.split_by == "shot":
        train_indices, val_indices, train_shots, val_shots = split_indices_by_shot(
            window_shot_ids,
            args.validation_fraction,
            args.seed,
        )
    else:
        train_indices, val_indices = split_indices(len(sequences), args.validation_fraction, args.seed)
        train_shots = sorted(int(shot_id) for shot_id in np.unique(window_shot_ids[train_indices]))
        val_shots = sorted(int(shot_id) for shot_id in np.unique(window_shot_ids[val_indices]))

    print(f"split: {args.split_by}")
    print(f"train shots: {len(train_shots)} / val shots: {len(val_shots)}")
    print(f"train windows: {len(train_indices)} / val windows: {len(val_indices)}")
    print(f"train labels: {count_labels_for_shots(shot_labels, train_shots)}")
    print(f"val labels: {count_labels_for_shots(shot_labels, val_shots)}")

    scalers = {
        "sequence": fit_scaler(sequences[train_indices], axes=(0, 1)),
        "context": fit_scaler(contexts[train_indices], axes=0),
        "target": fit_scaler(targets[train_indices], axes=(0, 1)),
    }
    sequences = apply_scaler(sequences, scalers["sequence"]).astype(np.float32)
    contexts = apply_scaler(contexts, scalers["context"]).astype(np.float32)
    targets = apply_scaler(targets, scalers["target"]).astype(np.float32)

    train_loader = make_loader(sequences, contexts, targets, train_indices, args.batch_size, True)
    val_loader = make_loader(sequences, contexts, targets, val_indices, args.batch_size, False) if len(val_indices) else None
    metric_horizons = parse_metric_horizons(args.metric_horizons, args.horizon_steps)

    model_config = WorldModelConfig(
        sequence_dim=len(SEQUENCE_FEATURES),
        context_dim=len(CONTEXT_FEATURES),
        horizon_steps=args.horizon_steps,
        history_steps=args.history_steps,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = TemporalTransformerWorldModel(model_config)
    device = pick_device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, device)
        if val_loader is not None:
            val_metrics = evaluate_model(model, val_loader, loss_fn, device, scalers["target"], metric_horizons)
            val_loss = val_metrics["normalized_mse"]
            if val_loss < best_val:
                best_val = val_loss
                best_metrics = val_metrics
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(
                f"epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f} "
                f"{format_horizon_metrics(val_metrics)}"
            )
        else:
            print(f"epoch {epoch:03d} train={train_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    save_artifacts(
        model.cpu(),
        model_config,
        scalers,
        args,
        Path(args.output_dir),
        {
            "shots": generated_shots,
            "candidate_shots_evaluated": candidates_seen,
            "label_counts": label_counts,
            "label1_fraction": label_counts[1] / max(1, generated_shots),
            "windows": int(len(sequences)),
            "split": {
                "mode": args.split_by,
                "train_shots": len(train_shots),
                "validation_shots": len(val_shots),
                "train_windows": int(len(train_indices)),
                "validation_windows": int(len(val_indices)),
                "train_label_counts": count_labels_for_shots(shot_labels, train_shots),
                "validation_label_counts": count_labels_for_shots(shot_labels, val_shots),
            },
            "best_validation_loss": None if best_val == float("inf") else best_val,
            "best_validation_metrics": best_metrics,
        },
    )
    print(f"saved: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

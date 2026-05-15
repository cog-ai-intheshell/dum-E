"""Extract world-model latent vectors for momentum classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from basketball_sim.dataset.grid import load_config_module
from basketball_sim.models.features import apply_scaler
from basketball_sim.models.train_world_model import (
    collect_samples,
    count_labels_for_shots,
    pick_device,
    split_indices_by_shot,
)
from basketball_sim.models.world_model import TemporalTransformerWorldModel, WorldModelConfig


def load_world_model(model_dir: Path, device: torch.device):
    checkpoint_path = model_dir / "world_model.pt"
    scaler_path = model_dir / "world_model_scaler.json"
    if not checkpoint_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"Missing world model artifacts in {model_dir}.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    with scaler_path.open("r", encoding="utf-8") as file:
        scalers = json.load(file)

    config = WorldModelConfig.from_dict(checkpoint["config"])
    model = TemporalTransformerWorldModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, scalers


def extract_latents(model, sequences, contexts, batch_size: int, device: torch.device) -> np.ndarray:
    latents = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            end = min(len(sequences), start + batch_size)
            sequence = torch.from_numpy(sequences[start:end]).to(device)
            context = torch.from_numpy(contexts[start:end]).to(device)
            latent = model.encode(sequence, context).cpu().numpy().astype(np.float32)
            latents.append(latent)
    return np.concatenate(latents, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dataset_config")
    parser.add_argument("--world-model-dir", default="models/world_model")
    parser.add_argument("--output-dir", default="models/momentum_xgb")
    parser.add_argument("--max-shots", type=int, default=3000)
    parser.add_argument("--history-steps", type=int, default=0)
    parser.add_argument("--horizon-steps", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--augmentation", choices=("none", "made-biased", "mixed"), default="mixed")
    parser.add_argument("--augmented-fraction", type=float, default=0.85)
    parser.add_argument("--min-label1-fraction", type=float, default=0.20)
    parser.add_argument("--candidate-limit", type=int, default=30000)
    parser.add_argument("--dedupe-physical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--latent-batch-size", type=int, default=4096)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    world_model_dir = Path(args.world_model_dir)
    model, model_config, scalers = load_world_model(world_model_dir, device)
    args.history_steps = args.history_steps or model_config.history_steps
    args.horizon_steps = args.horizon_steps or model_config.horizon_steps

    print("Collecting momentum samples...")
    (
        sequences,
        contexts,
        _targets,
        window_shot_ids,
        shot_labels,
        generated_shots,
        label_counts,
        candidates_seen,
    ) = collect_samples(load_config_module(args.config), args)

    train_indices, val_indices, train_shots, val_shots = split_indices_by_shot(
        window_shot_ids,
        args.validation_fraction,
        args.seed,
    )
    window_labels = shot_labels[window_shot_ids].astype(np.float32)

    print(f"shots: {generated_shots}")
    print(f"candidate shots evaluated: {candidates_seen}")
    print(f"momentum shots: {label_counts[1]} ({label_counts[1] / max(1, generated_shots):.3f})")
    print(f"train shots: {len(train_shots)} / val shots: {len(val_shots)}")
    print(f"train windows: {len(train_indices)} / val windows: {len(val_indices)}")
    print(f"train labels: {count_labels_for_shots(shot_labels, train_shots)}")
    print(f"val labels: {count_labels_for_shots(shot_labels, val_shots)}")

    sequences = apply_scaler(sequences, scalers["sequence"]).astype(np.float32)
    contexts = apply_scaler(contexts, scalers["context"]).astype(np.float32)
    latents = extract_latents(model, sequences, contexts, args.latent_batch_size, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latents_path = output_dir / "momentum_latents.npz"
    np.savez(
        latents_path,
        train_latents=latents[train_indices],
        validation_latents=latents[val_indices],
        train_labels=window_labels[train_indices],
        validation_labels=window_labels[val_indices],
    )
    metadata = {
        "definition": "momentum = 1 when the shot is made.",
        "world_model_dir": str(world_model_dir.resolve()),
        "latent_dim": int(latents.shape[1]),
        "training": vars(args),
        "dataset": {
            "shots": generated_shots,
            "candidate_shots_evaluated": candidates_seen,
            "label_counts": label_counts,
            "label1_fraction": label_counts[1] / max(1, generated_shots),
            "train_shots": len(train_shots),
            "validation_shots": len(val_shots),
            "train_windows": int(len(train_indices)),
            "validation_windows": int(len(val_indices)),
            "train_label_counts": count_labels_for_shots(shot_labels, train_shots),
            "validation_label_counts": count_labels_for_shots(shot_labels, val_shots),
        },
    }
    with (output_dir / "momentum_latents_config.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"saved latents: {latents_path.resolve()}")


if __name__ == "__main__":
    main()

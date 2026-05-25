"""Train the DUM-E catch world model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model.common.features import (
    ACTION_FEATURES,
    CONTEXT_FEATURES,
    SEQUENCE_FEATURES,
    TARGET_FEATURES,
    apply_scaler,
    fit_scaler,
)
from dataset_generation.synthetic import build_window_dataset
from model.models_design.world_model import DUMEWorldModel, DUMEWorldModelConfig


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_indices(length: int, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(length)
    rng.shuffle(indices)
    validation_count = max(1, int(length * validation_fraction))
    return indices[validation_count:], indices[:validation_count]


def make_loader(
    sequences,
    contexts,
    targets,
    actions,
    indices,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(sequences[indices]),
        torch.from_numpy(contexts[indices]),
        torch.from_numpy(targets[indices]),
        torch.from_numpy(actions[indices, :-1]),
        torch.from_numpy(actions[indices, -1:]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, optimizer, device, bce_loss, mse_loss, train: bool) -> dict:
    model.train(train)
    total = 0.0
    future_total = 0.0
    action_total = 0.0
    catch_total = 0.0
    count = 0

    for sequence, context, future_target, action_target, catch_target in loader:
        sequence = sequence.to(device)
        context = context.to(device)
        future_target = future_target.to(device)
        action_target = action_target.to(device)
        catch_target = catch_target.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(sequence, context)
            future_loss = mse_loss(outputs["future"], future_target)
            action_continuous = outputs["action"][:, :-1]
            catch_logit = outputs["action"][:, -1:]
            action_loss = mse_loss(action_continuous, action_target)
            catch_loss = bce_loss(catch_logit, catch_target)
            loss = future_loss + 0.35 * action_loss + 0.55 * catch_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        batch = sequence.shape[0]
        total += float(loss.detach().cpu()) * batch
        future_total += float(future_loss.detach().cpu()) * batch
        action_total += float(action_loss.detach().cpu()) * batch
        catch_total += float(catch_loss.detach().cpu()) * batch
        count += batch

    return {
        "loss": total / max(1, count),
        "future_loss": future_total / max(1, count),
        "action_loss": action_total / max(1, count),
        "catch_loss": catch_total / max(1, count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--history-steps", type=int, default=12)
    parser.add_argument("--horizon-steps", type=int, default=30)
    parser.add_argument("--max-windows-per-shot", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="model/artifacts")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sequences, contexts, targets, actions, metadata = build_window_dataset(
        shots=args.shots,
        history_steps=args.history_steps,
        horizon_steps=args.horizon_steps,
        seed=args.seed,
        max_windows_per_shot=args.max_windows_per_shot,
    )

    sequence_scaler = fit_scaler(sequences, axes=(0, 1))
    context_scaler = fit_scaler(contexts, axes=(0,))
    target_scaler = fit_scaler(targets, axes=(0, 1))
    action_scaler = fit_scaler(actions[:, :-1], axes=(0,))
    sequences = apply_scaler(sequences, sequence_scaler).astype(np.float32)
    contexts = apply_scaler(contexts, context_scaler).astype(np.float32)
    targets = apply_scaler(targets, target_scaler).astype(np.float32)
    actions[:, :-1] = apply_scaler(actions[:, :-1], action_scaler).astype(np.float32)

    train_indices, val_indices = split_indices(len(sequences), args.validation_fraction, args.seed)
    train_loader = make_loader(sequences, contexts, targets, actions, train_indices, args.batch_size, True)
    val_loader = make_loader(sequences, contexts, targets, actions, val_indices, args.batch_size, False)

    config = DUMEWorldModelConfig(
        sequence_dim=len(SEQUENCE_FEATURES),
        context_dim=len(CONTEXT_FEATURES),
        action_dim=len(ACTION_FEATURES),
        horizon_steps=args.horizon_steps,
        history_steps=args.history_steps,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    device = pick_device(args.device)
    model = DUMEWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    history = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, bce_loss, mse_loss, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, bce_loss, mse_loss, train=False)
        record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(record)
        print(
            f"epoch {epoch:03d} "
            f"train={train_metrics['loss']:.4f} "
            f"val={val_metrics['loss']:.4f} "
            f"catch={val_metrics['catch_loss']:.4f}"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_state": model.state_dict(),
        "config": config.to_dict(),
        "scalers": {
            "sequence": sequence_scaler,
            "context": context_scaler,
            "target": target_scaler,
            "action": action_scaler,
        },
        "features": {
            "sequence": SEQUENCE_FEATURES,
            "context": CONTEXT_FEATURES,
            "target": TARGET_FEATURES,
            "action": ACTION_FEATURES,
        },
        "metadata": {
            **metadata,
            "history": history,
            "best_validation_loss": best_val,
        },
    }
    model_path = output_dir / "dum_e_world_model.pt"
    torch.save(artifact, model_path)
    (output_dir / "metadata.json").write_text(json.dumps(artifact["metadata"], indent=2), encoding="utf-8")
    print(f"saved {model_path}")


if __name__ == "__main__":
    main()

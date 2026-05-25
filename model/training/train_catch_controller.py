"""Train the DUM-E motor controller on top of a frozen World Model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model.common.features import MOTION_FEATURES, apply_scaler, fit_scaler
from dataset_generation.synthetic import build_controller_dataset
from model.models_design.catch_controller import DUMECatchController, DUMECatchControllerConfig
from model.models_design.world_model import DUMEWorldModel, DUMEWorldModelConfig
from model.training.train_world_model import pick_device, split_indices


def make_loader(
    sequences,
    contexts,
    motions,
    catch_targets,
    indices,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(sequences[indices]),
        torch.from_numpy(contexts[indices]),
        torch.from_numpy(motions[indices]),
        torch.from_numpy(catch_targets[indices]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, optimizer, device, bce_loss, mse_loss, train: bool) -> dict:
    model.train(train)
    total = 0.0
    motion_total = 0.0
    catch_total = 0.0
    count = 0

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for sequence, context, motion_target, catch_target in loader:
        sequence = sequence.to(device)
        context = context.to(device)
        motion_target = motion_target.to(device)
        catch_target = catch_target.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(sequence, context)
            motion_loss = mse_loss(outputs["joints"], motion_target)
            catch_loss = bce_loss(outputs["catch_logit"], catch_target)
            loss = motion_loss + 0.25 * catch_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()

        batch = sequence.shape[0]
        total += float(loss.detach().cpu()) * batch
        motion_total += float(motion_loss.detach().cpu()) * batch
        catch_total += float(catch_loss.detach().cpu()) * batch
        count += batch

    return {
        "loss": total / max(1, count),
        "motion_loss": motion_total / max(1, count),
        "catch_loss": catch_total / max(1, count),
    }


def load_world_model(world_model_path: Path, device: torch.device) -> tuple[DUMEWorldModel, dict]:
    if not world_model_path.exists():
        raise FileNotFoundError(f"World Model artifact not found: {world_model_path}")
    artifact = torch.load(world_model_path, map_location=device)
    config = DUMEWorldModelConfig.from_dict(artifact["config"])
    model = DUMEWorldModel(config).to(device)
    model.load_state_dict(artifact["model_state"])
    model.eval()
    return model, artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-model-path", default="model/artifacts/dum_e_world_model.pt")
    parser.add_argument("--shots", type=int, default=700)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--history-steps", type=int, default=None)
    parser.add_argument("--horizon-steps", type=int, default=None)
    parser.add_argument("--motion-steps", type=int, default=30)
    parser.add_argument("--max-windows-per-shot", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="model/artifacts")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    world_model, world_artifact = load_world_model(Path(args.world_model_path), device)
    world_config = world_model.config
    history_steps = args.history_steps or world_config.history_steps
    horizon_steps = args.horizon_steps or world_config.horizon_steps
    if history_steps != world_config.history_steps or horizon_steps != world_config.horizon_steps:
        raise ValueError(
            "Controller history/horizon must match the loaded World Model "
            f"({world_config.history_steps}/{world_config.horizon_steps})."
        )

    sequences, contexts, _targets, actions, motions, metadata = build_controller_dataset(
        shots=args.shots,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        motion_steps=args.motion_steps,
        seed=args.seed,
        max_windows_per_shot=args.max_windows_per_shot,
    )

    sequence_scaler = world_artifact["scalers"]["sequence"]
    context_scaler = world_artifact["scalers"]["context"]
    joint_scaler = fit_scaler(motions, axes=(0, 1))
    sequences = apply_scaler(sequences, sequence_scaler).astype(np.float32)
    contexts = apply_scaler(contexts, context_scaler).astype(np.float32)
    motions = apply_scaler(motions, joint_scaler).astype(np.float32)
    catch_targets = actions[:, -1:].astype(np.float32)

    train_indices, val_indices = split_indices(len(sequences), args.validation_fraction, args.seed)
    train_loader = make_loader(sequences, contexts, motions, catch_targets, train_indices, args.batch_size, True)
    val_loader = make_loader(sequences, contexts, motions, catch_targets, val_indices, args.batch_size, False)

    config = DUMECatchControllerConfig(
        world_model_config=world_config.to_dict(),
        horizon_steps=horizon_steps,
        motion_steps=args.motion_steps,
        joint_dim=len(MOTION_FEATURES),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_world_model=True,
    )
    controller = DUMECatchController(config, world_model).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in controller.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    mse_loss = nn.MSELoss()
    positive_count = float(catch_targets.sum())
    negative_count = float(len(catch_targets) - positive_count)
    pos_weight = torch.tensor([negative_count / max(1.0, positive_count)], dtype=torch.float32, device=device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(controller, train_loader, optimizer, device, bce_loss, mse_loss, train=True)
        val_metrics = run_epoch(controller, val_loader, optimizer, device, bce_loss, mse_loss, train=False)
        record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(record)
        print(
            f"epoch {epoch:03d} "
            f"train={train_metrics['loss']:.4f} "
            f"val={val_metrics['loss']:.4f} "
            f"motion={val_metrics['motion_loss']:.4f} "
            f"catch={val_metrics['catch_loss']:.4f}"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {key: value.detach().cpu() for key, value in controller.state_dict().items()}

    if best_state is not None:
        controller.load_state_dict(best_state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_state": controller.state_dict(),
        "config": config.to_dict(),
        "scalers": {
            "joint": joint_scaler,
        },
        "features": {
            "motion": MOTION_FEATURES,
            "world_sequence": world_artifact.get("features", {}).get("sequence"),
            "world_context": world_artifact.get("features", {}).get("context"),
        },
        "metadata": {
            **metadata,
            "history": history,
            "best_validation_loss": best_val,
            "world_model_path": str(args.world_model_path),
        },
    }
    controller_path = output_dir / "dum_e_catch_controller.pt"
    torch.save(artifact, controller_path)
    (output_dir / "controller_metadata.json").write_text(
        json.dumps(artifact["metadata"], indent=2),
        encoding="utf-8",
    )
    print(f"saved {controller_path}")


if __name__ == "__main__":
    main()

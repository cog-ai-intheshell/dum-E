"""Motor-controller policy that uses the frozen DUM-E world model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required for the DUM-E catch controller. Install requirements.txt first."
    ) from exc

from .world_model import DUMEWorldModel


@dataclass(frozen=True)
class DUMECatchControllerConfig:
    world_model_config: dict
    horizon_steps: int = 30
    motion_steps: int = 30
    joint_dim: int = 3
    hidden_dim: int = 128
    dropout: float = 0.08
    freeze_world_model: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DUMECatchControllerConfig":
        return cls(**data)


class DUMECatchController(nn.Module):
    """Policy head on top of a World Model latent state.

    The World Model keeps its job: predict future ball states. This controller
    consumes that latent representation plus the predicted future trajectory and
    emits motor commands for DUM-E's three axes: base, elbow, and wrist.
    """

    def __init__(self, config: DUMECatchControllerConfig, world_model: DUMEWorldModel):
        super().__init__()
        self.config = config
        self.world_model = world_model
        latent_dim = world_model.config.d_model
        future_dim = config.horizon_steps * 3

        self.future_projection = nn.Sequential(
            nn.LayerNorm(future_dim),
            nn.Linear(future_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.policy = nn.Sequential(
            nn.LayerNorm(latent_dim + config.hidden_dim),
            nn.Linear(latent_dim + config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.motion_head = nn.Linear(config.hidden_dim, config.motion_steps * config.joint_dim)
        self.catch_head = nn.Linear(config.hidden_dim, 1)

        if config.freeze_world_model:
            for parameter in self.world_model.parameters():
                parameter.requires_grad = False

    def _world_outputs(self, sequence, context) -> dict[str, object]:
        if self.config.freeze_world_model:
            self.world_model.eval()
            with torch.no_grad():
                return self.world_model(sequence, context)
        return self.world_model(sequence, context)

    def forward(self, sequence, context) -> dict[str, object]:
        world = self._world_outputs(sequence, context)
        latent = world["latent"]
        future = world["future"]
        future_features = self.future_projection(future.reshape(future.shape[0], -1))
        features = torch.cat([latent, future_features], dim=1)
        hidden = self.policy(features)
        joints = self.motion_head(hidden).reshape(
            hidden.shape[0],
            self.config.motion_steps,
            self.config.joint_dim,
        )
        catch_logit = self.catch_head(hidden)
        return {
            "joints": joints,
            "catch_logit": catch_logit,
            "world_future": future,
            "world_action": world["action"],
            "latent": latent,
        }

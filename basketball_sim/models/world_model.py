"""Temporal Transformer world model for basketball trajectory dynamics.

The model learns the latent transition dynamics of the simulator:

    history[t-k:t], context -> positions[t+1:t+H]

It deliberately does not predict the basket label. The label can later be
computed from the predicted trajectory, or used by a separate classifier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without torch installed.
    raise ModuleNotFoundError(
        "PyTorch is required for the Temporal Transformer world model. "
        "Install it in your Python environment, then rerun the command."
    ) from exc


@dataclass(frozen=True)
class WorldModelConfig:
    sequence_dim: int
    context_dim: int
    horizon_steps: int = 30
    history_steps: int = 12
    d_model: int = 192
    num_heads: int = 12
    num_layers: int = 6
    ff_multiplier: int = 4
    dropout: float = 0.08

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WorldModelConfig":
        return cls(**data)


class TemporalTransformerWorldModel(nn.Module):
    """Encoder-only transformer with many attention heads.

    The context vector is projected as a learned conditioning token. The final
    normalized state token predicts the future positions in one multi-horizon
    shot, keeping inference cheap enough for live UI updates.
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.config = config
        self.input_projection = nn.Linear(config.sequence_dim, config.d_model)
        self.context_projection = nn.Sequential(
            nn.Linear(config.context_dim, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.history_steps + 1, config.d_model)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * config.ff_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.readout = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.horizon_steps * 3),
        )

        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def validate_inputs(self, sequence, context) -> None:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, history_steps, sequence_dim].")
        if context.ndim != 2:
            raise ValueError("context must have shape [batch, context_dim].")
        if sequence.shape[1] != self.config.history_steps:
            raise ValueError(
                f"expected {self.config.history_steps} history steps, got {sequence.shape[1]}."
            )

    def encode(self, sequence, context):
        """Return the latent state vector used by downstream classifiers."""
        self.validate_inputs(sequence, context)
        tokens = self.input_projection(sequence)
        context_token = self.context_projection(context).unsqueeze(1)
        tokens = torch.cat([context_token, tokens], dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens)
        return encoded[:, -1, :]

    def decode(self, latent):
        prediction = self.readout(latent)
        return prediction.reshape(latent.shape[0], self.config.horizon_steps, 3)

    def forward(self, sequence, context):
        return self.decode(self.encode(sequence, context))

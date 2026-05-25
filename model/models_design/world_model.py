"""PyTorch world model for DUM-E catch dynamics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required for the DUM-E world model. Install requirements.txt first."
    ) from exc


@dataclass(frozen=True)
class DUMEWorldModelConfig:
    sequence_dim: int
    context_dim: int
    action_dim: int
    horizon_steps: int = 30
    history_steps: int = 12
    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 3
    ff_multiplier: int = 4
    dropout: float = 0.08

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DUMEWorldModelConfig":
        return cls(**data)


class DUMEWorldModel(nn.Module):
    """Encoder-only Transformer with two heads.

    Head 1 predicts future ball positions. Head 2 predicts a compact catch plan:
    target point, base/elbow/wrist axes, time-to-catch, and catch logit.
    """

    def __init__(self, config: DUMEWorldModelConfig):
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
        self.future_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.horizon_steps * 3),
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.action_dim),
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
        self.validate_inputs(sequence, context)
        tokens = self.input_projection(sequence)
        context_token = self.context_projection(context).unsqueeze(1)
        tokens = torch.cat([context_token, tokens], dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens)
        return encoded[:, -1, :]

    def forward(self, sequence, context) -> dict[str, object]:
        latent = self.encode(sequence, context)
        future = self.future_head(latent).reshape(latent.shape[0], self.config.horizon_steps, 3)
        action = self.action_head(latent)
        return {"future": future, "action": action, "latent": latent}

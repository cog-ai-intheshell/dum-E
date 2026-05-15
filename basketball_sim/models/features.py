"""Feature contracts shared by world-model training and live inference."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


SEQUENCE_FEATURES = [
    "time",
    "ball_x",
    "ball_y",
    "ball_z",
    "ball_vx",
    "ball_vy",
    "ball_vz",
    "ball_ax",
    "ball_ay",
    "ball_az",
    "wind_x",
    "wind_y",
    "wind_z",
    "wind_norm",
    "distance_ball_to_hoop",
    "horizontal_distance_to_hoop",
    "speed_norm",
]

CONTEXT_FEATURES = [
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
    "gravity",
]

TARGET_FEATURES = ["ball_x", "ball_y", "ball_z"]

CAMEL_BY_SNAKE = {
    "initial_force": "initialForce",
    "vertical_angle": "verticalAngle",
    "horizontal_angle": "horizontalAngle",
    "distance_to_hoop": "distanceToHoop",
    "boy_height": "boyHeight",
    "ball_mass": "ballMass",
    "hoop_height": "hoopHeight",
    "wind_regime_id": "windRegimeId",
    "wind_strength": "windStrength",
    "wind_orientation": "windOrientation",
    "wind_vertical_orientation": "windVerticalOrientation",
    "wind_spatial_coupling": "windSpatialCoupling",
    "drag_coeff": "dragCoeff",
}


def read_value(source: Any, key: str, default: float = 0.0) -> float:
    if isinstance(source, dict):
        if key in source and source[key] is not None:
            return float(source[key])
        camel_key = CAMEL_BY_SNAKE.get(key)
        if camel_key and camel_key in source and source[camel_key] is not None:
            return float(source[camel_key])
        return float(default)
    if hasattr(source, key):
        value = getattr(source, key)
        if value is not None:
            return float(value)
    return float(default)


def context_vector(params: Any, gravity: float = 9.81) -> np.ndarray:
    values = []
    for key in CONTEXT_FEATURES:
        if key == "gravity":
            values.append(float(gravity))
        else:
            values.append(read_value(params, key))
    return np.asarray(values, dtype=np.float32)


def sequence_vector(row: dict, params: Any | None = None, wind_context: Any | None = None, wind_at=None) -> np.ndarray:
    enriched = dict(row)
    if "speed_norm" not in enriched:
        enriched["speed_norm"] = math.sqrt(
            float(enriched.get("ball_vx", 0.0)) ** 2
            + float(enriched.get("ball_vy", 0.0)) ** 2
            + float(enriched.get("ball_vz", 0.0)) ** 2
        )
    if "distance_ball_to_hoop" not in enriched and params is not None:
        hoop_x = read_value(params, "distance_to_hoop")
        hoop_y = 0.0
        hoop_z = read_value(params, "hoop_height", 3.05)
        dx = float(enriched.get("ball_x", 0.0)) - hoop_x
        dy = float(enriched.get("ball_y", 0.0)) - hoop_y
        dz = float(enriched.get("ball_z", 0.0)) - hoop_z
        enriched["distance_ball_to_hoop"] = math.sqrt(dx * dx + dy * dy + dz * dz)
        enriched["horizontal_distance_to_hoop"] = math.sqrt(dx * dx + dy * dy)
    if "wind_x" not in enriched:
        if wind_at is not None and params is not None and wind_context is not None:
            wind = wind_at(
                float(enriched.get("ball_x", 0.0)),
                float(enriched.get("ball_y", 0.0)),
                float(enriched.get("ball_z", 0.0)),
                float(enriched.get("time", 0.0)),
                params,
                wind_context,
            )
        else:
            wind = [0.0, 0.0, 0.0]
        enriched["wind_x"] = wind[0]
        enriched["wind_y"] = wind[1]
        enriched["wind_z"] = wind[2]
        enriched["wind_norm"] = math.sqrt(wind[0] * wind[0] + wind[1] * wind[1] + wind[2] * wind[2])

    return np.asarray([float(enriched.get(key, 0.0) or 0.0) for key in SEQUENCE_FEATURES], dtype=np.float32)


def target_vector(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[float(row[key]) for key in TARGET_FEATURES] for row in rows],
        dtype=np.float32,
    )


def make_windows(
    params: Any,
    rows: list[dict],
    wind_context: Any | None,
    wind_at,
    history_steps: int,
    horizon_steps: int,
    gravity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences = []
    contexts = []
    targets = []
    last_start = len(rows) - horizon_steps
    for index in range(history_steps - 1, last_start):
        history = rows[index - history_steps + 1 : index + 1]
        future = rows[index + 1 : index + 1 + horizon_steps]
        sequences.append([sequence_vector(row, params, wind_context, wind_at) for row in history])
        contexts.append(context_vector(params, gravity))
        targets.append(target_vector(future))

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(contexts, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def fit_scaler(array: np.ndarray, axes) -> dict:
    mean = array.mean(axis=axes, keepdims=False)
    std = array.std(axis=axes, keepdims=False)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}


def apply_scaler(array: np.ndarray, scaler: dict) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    return (array - mean) / std


def invert_scaler(array: np.ndarray, scaler: dict) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    return array * std + mean

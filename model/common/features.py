"""Feature contracts for the DUM-E catch world model."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from .kinematics import (
    CATCH_RADIUS,
    REST_AXIS_BASE,
    REST_AXIS_ELBOW,
    REST_AXIS_WRIST,
    catch_center,
    inverse_kinematics,
    joint_axes_for_target,
)


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
    "distance_ball_to_catcher",
    "horizontal_distance_to_catcher",
    "speed_norm",
]

CONTEXT_FEATURES = [
    "initial_force",
    "vertical_angle",
    "horizontal_angle",
    "distance_to_dum_e",
    "child_height",
    "ball_mass",
    "catch_height",
    "catch_radius",
    "wind_regime_id",
    "wind_strength",
    "wind_orientation",
    "wind_vertical_orientation",
    "wind_spatial_coupling",
    "drag_coeff",
    "gravity",
]

TARGET_FEATURES = ["ball_x", "ball_y", "ball_z"]

ACTION_FEATURES = [
    "catch_x",
    "catch_y",
    "catch_z",
    "axis_base",
    "axis_elbow",
    "axis_wrist",
    "time_to_catch",
    "catch_probability",
]

MOTION_FEATURES = ["axis_base", "axis_elbow", "axis_wrist"]

ALIASES = {
    "distance_to_dum_e": ("distance_to_dum_e", "distance_to_hoop", "distanceToHoop"),
    "child_height": ("child_height", "boy_height", "boyHeight"),
    "catch_height": ("catch_height", "hoop_height", "hoopHeight"),
    "catch_radius": ("catch_radius", "hoop_radius", "hoopRadius"),
    "initial_force": ("initial_force", "initialForce"),
    "vertical_angle": ("vertical_angle", "verticalAngle"),
    "horizontal_angle": ("horizontal_angle", "horizontalAngle"),
    "ball_mass": ("ball_mass", "ballMass"),
    "wind_regime_id": ("wind_regime_id", "windRegimeId"),
    "wind_strength": ("wind_strength", "windStrength"),
    "wind_orientation": ("wind_orientation", "windOrientation"),
    "wind_vertical_orientation": ("wind_vertical_orientation", "windVerticalOrientation"),
    "wind_spatial_coupling": ("wind_spatial_coupling", "windSpatialCoupling"),
    "drag_coeff": ("drag_coeff", "dragCoeff"),
    "gravity": ("gravity",),
}


def read_value(source: Any, key: str, default: float = 0.0) -> float:
    names = ALIASES.get(key, (key,))
    if isinstance(source, dict):
        for name in names:
            value = source.get(name)
            if value is not None:
                return float(value)
        return float(default)
    for name in names:
        if hasattr(source, name):
            value = getattr(source, name)
            if value is not None:
                return float(value)
    return float(default)


def context_vector(params: Any, gravity: float = 9.81) -> np.ndarray:
    values = []
    for key in CONTEXT_FEATURES:
        if key == "gravity":
            values.append(read_value(params, key, gravity))
        elif key == "catch_radius":
            values.append(read_value(params, key, CATCH_RADIUS))
        else:
            values.append(read_value(params, key))
    return np.asarray(values, dtype=np.float32)


def sequence_vector(
    row: dict,
    params: Any | None = None,
    wind_context: Any | None = None,
    wind_at: Callable | None = None,
) -> np.ndarray:
    enriched = dict(row)
    if "speed_norm" not in enriched:
        enriched["speed_norm"] = math.sqrt(
            float(enriched.get("ball_vx", 0.0)) ** 2
            + float(enriched.get("ball_vy", 0.0)) ** 2
            + float(enriched.get("ball_vz", 0.0)) ** 2
        )

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
            wind = (0.0, 0.0, 0.0)
        enriched["wind_x"] = wind[0]
        enriched["wind_y"] = wind[1]
        enriched["wind_z"] = wind[2]
        enriched["wind_norm"] = math.sqrt(wind[0] ** 2 + wind[1] ** 2 + wind[2] ** 2)

    if params is not None:
        center = catch_center(params)
        dx = float(enriched.get("ball_x", 0.0)) - center[0]
        dy = float(enriched.get("ball_y", 0.0)) - center[1]
        dz = float(enriched.get("ball_z", 0.0)) - center[2]
        enriched.setdefault("distance_ball_to_catcher", math.sqrt(dx * dx + dy * dy + dz * dz))
        enriched.setdefault("horizontal_distance_to_catcher", math.sqrt(dx * dx + dy * dy))

    enriched.setdefault("distance_ball_to_catcher", enriched.get("distance_ball_to_hoop", 0.0))
    enriched.setdefault("horizontal_distance_to_catcher", enriched.get("horizontal_distance_to_hoop", 0.0))

    return np.asarray([float(enriched.get(key, 0.0) or 0.0) for key in SEQUENCE_FEATURES], dtype=np.float32)


def target_vector(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[float(row[key]) for key in TARGET_FEATURES] for row in rows],
        dtype=np.float32,
    )


def plan_from_future(rows: list[dict], params: Any, current_time: float) -> np.ndarray:
    center = catch_center(params)
    best_row = min(
        rows,
        key=lambda row: math.dist(
            (float(row["ball_x"]), float(row["ball_y"]), float(row["ball_z"])),
            center,
        ),
    )
    best_point = (
        float(best_row["ball_x"]),
        float(best_row["ball_y"]),
        max(0.0, float(best_row["ball_z"])),
    )
    best_distance = math.dist(best_point, center)
    pose = inverse_kinematics(best_point, params)
    return np.asarray(
        [
            pose.catch_x,
            pose.catch_y,
            pose.catch_z,
            pose.axis_base,
            pose.axis_elbow,
            pose.axis_wrist,
            max(0.0, float(best_row["time"]) - current_time),
            1.0 if best_distance <= CATCH_RADIUS else 0.0,
        ],
        dtype=np.float32,
    )


def _smoothstep(value: np.ndarray) -> np.ndarray:
    return value * value * (3.0 - 2.0 * value)


def motion_from_future(rows: list[dict], params: Any, motion_steps: int) -> np.ndarray:
    center = catch_center(params)
    best_row = min(
        rows,
        key=lambda row: math.dist(
            (float(row["ball_x"]), float(row["ball_y"]), float(row["ball_z"])),
            center,
        ),
    )
    best_point = (
        float(best_row["ball_x"]),
        float(best_row["ball_y"]),
        max(0.0, float(best_row["ball_z"])),
    )
    target_axes = joint_axes_for_target(best_point, params)
    rest = np.asarray([REST_AXIS_BASE, REST_AXIS_ELBOW, REST_AXIS_WRIST], dtype=np.float32)
    target = np.asarray(
        [target_axes.axis_base, target_axes.axis_elbow, target_axes.axis_wrist],
        dtype=np.float32,
    )
    progress = _smoothstep(np.linspace(1.0 / motion_steps, 1.0, motion_steps, dtype=np.float32))
    return (rest[None, :] + progress[:, None] * (target[None, :] - rest[None, :])).astype(np.float32)


def action_to_dict(action: np.ndarray | list[float]) -> dict[str, float]:
    values = [float(value) for value in action]
    return {key: values[index] for index, key in enumerate(ACTION_FEATURES)}


def make_windows(
    params: Any,
    rows: list[dict],
    wind_context: Any | None,
    wind_at: Callable | None,
    history_steps: int,
    horizon_steps: int,
    gravity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequences = []
    contexts = []
    targets = []
    actions = []
    last_start = len(rows) - horizon_steps
    for index in range(history_steps - 1, last_start):
        history = rows[index - history_steps + 1 : index + 1]
        future = rows[index + 1 : index + 1 + horizon_steps]
        sequences.append([sequence_vector(row, params, wind_context, wind_at) for row in history])
        contexts.append(context_vector(params, gravity))
        targets.append(target_vector(future))
        actions.append(plan_from_future(future, params, float(rows[index]["time"])))

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(contexts, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )


def make_controller_windows(
    params: Any,
    rows: list[dict],
    wind_context: Any | None,
    wind_at: Callable | None,
    history_steps: int,
    horizon_steps: int,
    motion_steps: int,
    gravity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequences = []
    contexts = []
    targets = []
    actions = []
    motions = []
    last_start = len(rows) - horizon_steps
    for index in range(history_steps - 1, last_start):
        history = rows[index - history_steps + 1 : index + 1]
        future = rows[index + 1 : index + 1 + horizon_steps]
        sequences.append([sequence_vector(row, params, wind_context, wind_at) for row in history])
        contexts.append(context_vector(params, gravity))
        targets.append(target_vector(future))
        actions.append(plan_from_future(future, params, float(rows[index]["time"])))
        motions.append(motion_from_future(future, params, motion_steps))

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(contexts, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(motions, dtype=np.float32),
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

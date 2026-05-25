"""Synthetic DUM-E catch dataset generation.

This module is intentionally self-contained. The old basketball simulator package
is no longer needed for model training: trajectory physics, wind sampling, and
window generation all live under ``model/`` now.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from model.common.features import make_controller_windows, make_windows
from model.common.kinematics import CATCH_RADIUS, catch_center, catch_event


GRAVITY = 9.81
DT = 0.02
MAX_STEPS = 250
PLAYER_X = 0.0
PLAYER_Y = 0.0

WIND_REGIME_NAMES = {
    0: "vent nul",
    1: "vent constant",
    2: "vent random",
    3: "vent markovien",
    4: "vent cyclique",
    5: "rafales",
    6: "changement brutal",
    7: "mean reversion",
    8: "clustering volatilite",
    9: "jump / news shock",
    10: "pump & fade",
    11: "cascade liquidations",
    12: "liquidity wall",
    13: "squeeze breakout",
    14: "hidden regime switching",
    15: "vent chaotique",
}


@dataclass(frozen=True)
class ShotParams:
    initial_force: float
    vertical_angle: float
    horizontal_angle: float
    distance_to_hoop: float
    boy_height: float
    ball_mass: float
    hoop_height: float
    wind_regime_id: int
    wind_regime: str
    wind_strength: float
    wind_orientation: float
    wind_vertical_orientation: float
    wind_spatial_coupling: int
    wind_field_visible: int
    field_view: int
    past_trail_visible: int
    drag_coeff: float


def clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    t = clamp((value - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def add(a: list[float] | tuple[float, float, float], b: list[float] | tuple[float, float, float]) -> list[float]:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def sub(a: list[float] | tuple[float, float, float], b: list[float] | tuple[float, float, float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def scale(v: list[float] | tuple[float, float, float], scalar: float) -> list[float]:
    return [v[0] * scalar, v[1] * scalar, v[2] * scalar]


def norm(v: list[float] | tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def normalize(v: list[float] | tuple[float, float, float]) -> list[float]:
    length = norm(v)
    if length < 1e-12:
        return [0.0, 0.0, 0.0]
    return [v[0] / length, v[1] / length, v[2] / length]


def deg_to_rad(value: float) -> float:
    return value * math.pi / 180.0


def release_height(params: ShotParams) -> float:
    return params.boy_height * 0.85


def ideal_launch_speed(distance: float, angle_deg: float, release_z: float, target_z: float) -> float | None:
    theta = math.radians(angle_deg)
    denominator = 2.0 * math.cos(theta) ** 2 * (distance * math.tan(theta) - (target_z - release_z))
    if denominator <= 1e-9:
        return None
    return math.sqrt(GRAVITY * distance * distance / denominator)


def random_params(rng: np.random.Generator) -> ShotParams:
    distance = float(rng.uniform(3.2, 9.2))
    child_height = float(rng.uniform(1.25, 1.95))
    catch_height = float(rng.uniform(2.1, 3.65))
    vertical_angle = float(rng.uniform(32.0, 68.0))
    ideal_speed = ideal_launch_speed(distance, vertical_angle, child_height * 0.85, catch_height)
    if ideal_speed is None or not math.isfinite(ideal_speed):
        initial_force = float(rng.uniform(7.0, 18.0))
    else:
        initial_force = float(np.clip(ideal_speed * rng.uniform(0.78, 1.24), 4.0, 28.0))

    wind_regime_id = int(rng.choice([0, 1, 2, 3, 5, 7], p=[0.2, 0.22, 0.2, 0.14, 0.14, 0.1]))
    wind_strength = 0.0 if wind_regime_id == 0 else float(rng.uniform(0.5, 7.5))
    return ShotParams(
        initial_force=initial_force,
        vertical_angle=vertical_angle,
        horizontal_angle=float(rng.normal(0.0, 7.5)),
        distance_to_hoop=distance,
        boy_height=child_height,
        ball_mass=float(rng.uniform(0.45, 0.9)),
        hoop_height=catch_height,
        wind_regime_id=wind_regime_id,
        wind_regime=WIND_REGIME_NAMES[wind_regime_id],
        wind_strength=wind_strength,
        wind_orientation=float(rng.uniform(0.0, 360.0)),
        wind_vertical_orientation=float(rng.uniform(-18.0, 18.0)),
        wind_spatial_coupling=int(rng.integers(0, 2)),
        wind_field_visible=1,
        field_view=0,
        past_trail_visible=1,
        drag_coeff=float(rng.uniform(0.0, 0.08)),
    )


def stable_seed(params: ShotParams) -> int:
    parts = [
        params.initial_force,
        params.vertical_angle,
        params.horizontal_angle,
        params.distance_to_hoop,
        params.boy_height,
        params.ball_mass,
        params.hoop_height,
        params.wind_regime_id,
        params.wind_strength,
        params.wind_orientation,
        params.wind_vertical_orientation,
        params.wind_spatial_coupling,
        params.drag_coeff,
    ]
    text = "|".join(f"{value:.9g}" for value in parts)
    return int.from_bytes(hashlib.blake2s(text.encode("utf-8"), digest_size=8).digest(), "little")


def make_base_wind(params: ShotParams) -> list[float]:
    angle = deg_to_rad(params.wind_orientation)
    elevation = deg_to_rad(params.wind_vertical_orientation)
    horizontal = params.wind_strength * math.cos(elevation)
    return [
        horizontal * math.cos(angle),
        horizontal * math.sin(angle),
        params.wind_strength * math.sin(elevation),
    ]


def create_wind_context(params: ShotParams) -> dict:
    rng = np.random.default_rng(stable_seed(params))
    phases = rng.uniform(0.0, 2.0 * math.pi, size=6)
    modes = rng.normal(0.0, 1.0, size=(6, 3))
    mode_norms = np.linalg.norm(modes, axis=1, keepdims=True)
    modes = modes / np.maximum(mode_norms, 1e-6)

    base = make_base_wind(params)
    markov_states = []
    state = np.asarray(base, dtype=np.float64)
    alpha = math.exp(-DT / 0.75)
    sigma = 0.42 * params.wind_strength * math.sqrt(max(0.0, 1.0 - alpha * alpha))
    for _step in range(MAX_STEPS):
        noise = rng.normal(0.0, sigma, size=3)
        noise[2] *= 0.45
        state = alpha * state + (1.0 - alpha) * np.asarray(base) + noise
        markov_states.append(state.astype(float).tolist())

    gusts = []
    for index in range(4):
        gusts.append(
            {
                "center": 0.48 + index * 0.9 + float(rng.uniform(-0.12, 0.12)),
                "width": float(rng.uniform(0.06, 0.16)),
                "amplitude": float(rng.uniform(0.7, 1.85)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        )

    return {
        "base": base,
        "phases": phases.tolist(),
        "modes": modes.tolist(),
        "markov_states": markov_states,
        "gusts": gusts,
    }


def spectral_noise(x: float, y: float, z: float, t: float, wind_context: dict) -> list[float]:
    total = [0.0, 0.0, 0.0]
    for index, direction in enumerate(wind_context["modes"]):
        phase = wind_context["phases"][index]
        wave = math.sin(0.55 * x * direction[0] + 0.75 * y * direction[1] + 0.42 * z + (0.8 + 0.13 * index) * t + phase)
        total = add(total, scale(direction, wave))
    return scale(total, 1.0 / max(1, len(wind_context["modes"])))


def wind_at(x: float, y: float, z: float, t: float, params: ShotParams, wind_context: dict) -> list[float]:
    strength = params.wind_strength
    if strength <= 1e-9 or params.wind_regime_id == 0:
        return [0.0, 0.0, 0.0]

    base = wind_context["base"]
    angle = deg_to_rad(params.wind_orientation)
    perpendicular = [-math.sin(angle), math.cos(angle), 0.0]
    regime = params.wind_regime_id

    if regime == 1:
        wind = base[:]
    elif regime == 2:
        wind = add(scale(base, 0.76 + 0.24 * math.sin(1.7 * t + 0.35 * x)), scale(spectral_noise(x, y, z, t, wind_context), 1.15 * strength))
    elif regime == 3:
        step = max(0, min(MAX_STEPS - 1, round(t / DT)))
        wind = add(wind_context["markov_states"][step], scale(spectral_noise(x, y, z, t, wind_context), 0.18 * strength))
    elif regime == 5:
        gust_factor = 1.0
        for gust in wind_context["gusts"]:
            time_pulse = math.exp(-((t - gust["center"]) ** 2) / (2.0 * gust["width"] ** 2))
            spatial_pulse = 0.76 + 0.24 * math.sin(0.7 * x - 0.5 * y + gust["phase"])
            gust_factor += gust["amplitude"] * time_pulse * spatial_pulse
        wind = add(scale(base, gust_factor), scale(perpendicular, 0.16 * strength * math.sin(2.8 * t + y)))
    elif regime == 7:
        center = catch_center(params)
        progress = clamp(x / max(params.distance_to_hoop, 1.0), 0.0, 1.15)
        target_z = lerp(release_height(params), center[2], progress)
        mean_pull = [
            -0.3 * strength * math.tanh((x - 0.55 * params.distance_to_hoop) / max(0.35 * params.distance_to_hoop, 1e-6)),
            -0.7 * strength * math.tanh((y - center[1]) / 2.1),
            0.24 * strength * math.tanh((target_z - z) / 1.35),
        ]
        wind = add(scale(base, 0.18 + 0.12 * math.cos(1.4 * t)), mean_pull)
    else:
        wind = add(base, scale(spectral_noise(x, y, z, t, wind_context), 0.5 * strength))

    if params.wind_spatial_coupling:
        coupling = [
            0.16 * strength * math.sin(1.15 * y + 1.22 * t),
            0.16 * strength * math.cos(0.78 * x - 1.05 * t),
            0.11 * strength * math.sin(1.10 * z + 1.72 * t),
        ]
        wind = add(wind, coupling)
    return wind


def acceleration_at(position: list[float], velocity: list[float], t: float, params: ShotParams, wind_context: dict) -> tuple[list[float], list[float]]:
    wind = wind_at(position[0], position[1], position[2], t, params, wind_context)
    relative_velocity = sub(velocity, wind)
    relative_speed = norm(relative_velocity)
    drag_acc = scale(relative_velocity, -(params.drag_coeff * relative_speed) / max(params.ball_mass, 1e-6))
    return wind, [drag_acc[0], drag_acc[1], -GRAVITY + drag_acc[2]]


def derivative(state_vector: list[float], t: float, params: ShotParams, wind_context: dict) -> list[float]:
    position = state_vector[:3]
    velocity = state_vector[3:]
    _wind, acceleration = acceleration_at(position, velocity, t, params, wind_context)
    return [velocity[0], velocity[1], velocity[2], acceleration[0], acceleration[1], acceleration[2]]


def combine_state(state_vector: list[float], derivative_vector: list[float], scalar: float) -> list[float]:
    return [value + derivative_vector[index] * scalar for index, value in enumerate(state_vector)]


def rk4_step(position: list[float], velocity: list[float], t: float, params: ShotParams, wind_context: dict) -> tuple[list[float], list[float]]:
    state_vector = [*position, *velocity]
    k1 = derivative(state_vector, t, params, wind_context)
    k2 = derivative(combine_state(state_vector, k1, DT / 2.0), t + DT / 2.0, params, wind_context)
    k3 = derivative(combine_state(state_vector, k2, DT / 2.0), t + DT / 2.0, params, wind_context)
    k4 = derivative(combine_state(state_vector, k3, DT), t + DT, params, wind_context)
    next_state = [
        value + (DT / 6.0) * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index])
        for index, value in enumerate(state_vector)
    ]
    return next_state[:3], next_state[3:]


def make_row(shot_id: int, step: int, time: float, position: list[float], velocity: list[float], acceleration: list[float], wind: list[float], params: ShotParams) -> dict:
    center = catch_center(params)
    distance = math.dist(position, center)
    horizontal_distance = math.hypot(position[0] - center[0], position[1] - center[1])
    speed = norm(velocity)
    wind_norm = norm(wind)
    return {
        "shot_id": shot_id,
        "timestep": step,
        "time": time,
        "ball_x": position[0],
        "ball_y": position[1],
        "ball_z": position[2],
        "ball_vx": velocity[0],
        "ball_vy": velocity[1],
        "ball_vz": velocity[2],
        "ball_ax": acceleration[0],
        "ball_ay": acceleration[1],
        "ball_az": acceleration[2],
        "wind_x": wind[0],
        "wind_y": wind[1],
        "wind_z": wind[2],
        "wind_norm": wind_norm,
        "distance_ball_to_hoop": distance,
        "horizontal_distance_to_hoop": horizontal_distance,
        "distance_ball_to_catcher": distance,
        "horizontal_distance_to_catcher": horizontal_distance,
        "speed_norm": speed,
    }


def simulate_shot(shot_id: int, params: ShotParams) -> tuple[dict, list[dict], dict, dict]:
    wind_context = create_wind_context(params)
    vertical = deg_to_rad(params.vertical_angle)
    horizontal = deg_to_rad(params.horizontal_angle)
    position = [PLAYER_X, PLAYER_Y, release_height(params)]
    velocity = [
        params.initial_force * math.cos(vertical) * math.cos(horizontal),
        params.initial_force * math.cos(vertical) * math.sin(horizontal),
        params.initial_force * math.sin(vertical),
    ]

    rows = []
    for step in range(MAX_STEPS):
        time = step * DT
        wind, acceleration = acceleration_at(position, velocity, time, params, wind_context)
        rows.append(make_row(shot_id, step, time, position, velocity, acceleration, wind, params))
        if position[2] < -0.2 and velocity[2] < 0.0 and step > 6:
            break
        position, velocity = rk4_step(position, velocity, time, params, wind_context)

    label, capture_time, min_distance = catch_event(rows, params)
    shot_row = {
        "shot_id": shot_id,
        "initial_force": params.initial_force,
        "vertical_angle": params.vertical_angle,
        "horizontal_angle": params.horizontal_angle,
        "distance_to_dum_e": params.distance_to_hoop,
        "distance_to_hoop": params.distance_to_hoop,
        "child_height": params.boy_height,
        "boy_height": params.boy_height,
        "ball_mass": params.ball_mass,
        "catch_height": params.hoop_height,
        "hoop_height": params.hoop_height,
        "catch_radius": CATCH_RADIUS,
        "wind_regime_id": params.wind_regime_id,
        "wind_regime": params.wind_regime,
        "wind_strength": params.wind_strength,
        "wind_orientation": params.wind_orientation,
        "wind_vertical_orientation": params.wind_vertical_orientation,
        "wind_spatial_coupling": params.wind_spatial_coupling,
        "drag_coeff": params.drag_coeff,
        "gravity": GRAVITY,
    }
    label_row = {
        "shot_id": shot_id,
        "label": label,
        "result": "ATTRAPE" if label == 1 else "MANQUE",
        "distance_to_dum_e": params.distance_to_hoop,
        "final_min_catch_distance": min_distance,
        "capture_time": capture_time,
    }
    return shot_row, rows, label_row, wind_context


def annotate_rows(rows: list[dict], params: ShotParams) -> list[dict]:
    label, capture_time, min_distance = catch_event(rows, params)
    annotated = []
    for row in rows:
        next_row = dict(row)
        next_row["label"] = label
        next_row["capture_time"] = capture_time
        next_row["catch_radius"] = CATCH_RADIUS
        next_row["final_min_catch_distance"] = min_distance
        annotated.append(next_row)
    return annotated


def iter_synthetic_shots(count: int, seed: int) -> Iterable[tuple[ShotParams, list[dict], object]]:
    rng = np.random.default_rng(seed)
    for shot_id in range(count):
        params = random_params(rng)
        _shot_row, rows, _label_row, wind_context = simulate_shot(shot_id, params)
        yield params, annotate_rows(rows, params), wind_context


def build_window_dataset(
    shots: int,
    history_steps: int,
    horizon_steps: int,
    seed: int,
    max_windows_per_shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed + 17)
    all_sequences = []
    all_contexts = []
    all_targets = []
    all_actions = []
    used_shots = 0
    positive_windows = 0

    for params, rows, wind_context in iter_synthetic_shots(shots, seed):
        if len(rows) < history_steps + horizon_steps + 1:
            continue
        sequences, contexts, targets, actions = make_windows(
            params=params,
            rows=rows,
            wind_context=wind_context,
            wind_at=wind_at,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
            gravity=GRAVITY,
        )
        if len(sequences) == 0:
            continue
        if max_windows_per_shot > 0 and len(sequences) > max_windows_per_shot:
            indices = rng.choice(len(sequences), size=max_windows_per_shot, replace=False)
            sequences = sequences[indices]
            contexts = contexts[indices]
            targets = targets[indices]
            actions = actions[indices]
        all_sequences.append(sequences)
        all_contexts.append(contexts)
        all_targets.append(targets)
        all_actions.append(actions)
        used_shots += 1
        positive_windows += int(actions[:, -1].sum())

    if not all_sequences:
        raise RuntimeError("No training windows were generated.")

    sequences = np.concatenate(all_sequences, axis=0)
    contexts = np.concatenate(all_contexts, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    metadata = {
        "requested_shots": shots,
        "used_shots": used_shots,
        "windows": int(len(sequences)),
        "positive_windows": positive_windows,
        "positive_fraction": positive_windows / max(1, len(sequences)),
    }
    return sequences, contexts, targets, actions, metadata


def build_controller_dataset(
    shots: int,
    history_steps: int,
    horizon_steps: int,
    motion_steps: int,
    seed: int,
    max_windows_per_shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed + 31)
    all_sequences = []
    all_contexts = []
    all_targets = []
    all_actions = []
    all_motions = []
    used_shots = 0
    positive_windows = 0

    for params, rows, wind_context in iter_synthetic_shots(shots, seed):
        if len(rows) < history_steps + horizon_steps + 1:
            continue
        sequences, contexts, targets, actions, motions = make_controller_windows(
            params=params,
            rows=rows,
            wind_context=wind_context,
            wind_at=wind_at,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
            motion_steps=motion_steps,
            gravity=GRAVITY,
        )
        if len(sequences) == 0:
            continue
        if max_windows_per_shot > 0 and len(sequences) > max_windows_per_shot:
            indices = rng.choice(len(sequences), size=max_windows_per_shot, replace=False)
            sequences = sequences[indices]
            contexts = contexts[indices]
            targets = targets[indices]
            actions = actions[indices]
            motions = motions[indices]
        all_sequences.append(sequences)
        all_contexts.append(contexts)
        all_targets.append(targets)
        all_actions.append(actions)
        all_motions.append(motions)
        used_shots += 1
        positive_windows += int(actions[:, -1].sum())

    if not all_sequences:
        raise RuntimeError("No controller training windows were generated.")

    sequences = np.concatenate(all_sequences, axis=0)
    contexts = np.concatenate(all_contexts, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    motions = np.concatenate(all_motions, axis=0)
    metadata = {
        "requested_shots": shots,
        "used_shots": used_shots,
        "windows": int(len(sequences)),
        "positive_windows": positive_windows,
        "positive_fraction": positive_windows / max(1, len(sequences)),
    }
    return sequences, contexts, targets, actions, motions, metadata

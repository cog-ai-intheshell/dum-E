"""Generate dataset CSV files and the multi-view frame manifest.

Recommended smoke command:
    .venv/bin/python -m basketball_sim.dataset.generate --max-shots 5

By default, generation is capped at 10 shots to avoid producing a huge dataset
by accident. Use --max-shots 0 to generate every combination.

Important: this script does not render training images. Frames should come from
the exact ai-cameras.html renderer so the AI view matches the training input.
"""

from __future__ import annotations

import argparse
import csv
import math
from itertools import islice
from pathlib import Path
from urllib.parse import urlencode

from .config import (
    CAMERA_VIEWS,
    DT,
    GRAVITY,
    HOOP_RADIUS,
    HOOP_Y,
    MAX_STEPS,
    PLAYER_X,
    PLAYER_Y,
    SCENE_BOUNDS,
    WIND_GRID_SIZE,
)
from .grid import (
    ShotParams,
    count_combinations,
    iter_parameter_combinations,
    load_config_module,
)

SHOTS_COLUMNS = [
    "shot_id",
    "initial_force",
    "vertical_angle",
    "horizontal_angle",
    "distance_to_hoop",
    "boy_height",
    "ball_mass",
    "hoop_height",
    "wind_regime_id",
    "wind_regime",
    "wind_strength",
    "wind_orientation",
    "wind_vertical_orientation",
    "wind_spatial_coupling",
    "wind_field_visible",
    "field_view",
    "past_trail_visible",
    "drag_coeff",
    "gravity",
]

TRAJECTORY_COLUMNS = [
    "shot_id",
    "timestep",
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
    "distance_ball_to_hoop",
    "horizontal_distance_to_hoop",
    "speed_norm",
]

WIND_FIELD_COLUMNS = [
    "shot_id",
    "timestep",
    "time",
    "wind_vector_id",
    "wind_grid_x",
    "wind_grid_y",
    "wind_grid_z",
    "wind_x",
    "wind_y",
    "wind_z",
    "wind_norm",
]

LABEL_COLUMNS = [
    "shot_id",
    "label",
    "result",
    "distance_to_hoop",
    "final_min_hoop_distance",
    "collision_time",
]

FRAME_MANIFEST_COLUMNS = [
    "shot_id",
    "timestep",
    "time",
    "camera_view",
    "wind_field_visible",
    "field_view",
    "past_trail_visible",
    "render_query",
    "relative_path",
]


def apply_config(config) -> None:
    global CAMERA_VIEWS
    global DT
    global GRAVITY
    global HOOP_RADIUS
    global HOOP_Y
    global MAX_STEPS
    global PLAYER_X
    global PLAYER_Y
    global SCENE_BOUNDS
    global WIND_GRID_SIZE

    CAMERA_VIEWS = config.CAMERA_VIEWS
    DT = config.DT
    GRAVITY = config.GRAVITY
    HOOP_RADIUS = config.HOOP_RADIUS
    HOOP_Y = config.HOOP_Y
    MAX_STEPS = config.MAX_STEPS
    PLAYER_X = config.PLAYER_X
    PLAYER_Y = config.PLAYER_Y
    SCENE_BOUNDS = config.SCENE_BOUNDS
    WIND_GRID_SIZE = config.WIND_GRID_SIZE


def clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    t = clamp((value - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def scale(v, s: float):
    return [v[0] * s, v[1] * s, v[2] * s]


def norm(v) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def normalize(v):
    length = norm(v)
    if length < 1e-12:
        return [0.0, 0.0, 0.0]
    return [v[0] / length, v[1] / length, v[2] / length]


def vector_limited(v, max_norm: float):
    length = norm(v)
    if length <= max_norm or length < 1e-12:
        return v
    return scale(v, max_norm / length)


def js_number_string(value) -> str:
    number = float(value)
    if math.isfinite(number) and abs(number - round(number)) < 1e-12:
        return str(int(round(number)))
    return f"{number:.15g}"


def js_round(value: float) -> int:
    return math.floor(value + 0.5)


def hash_string(text: str) -> int:
    hash_value = 2166136261
    for char in text:
        hash_value ^= ord(char)
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF
    return hash_value


def imul(a: int, b: int) -> int:
    return ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) & 0xFFFFFFFF


class Mulberry32:
    def __init__(self, seed: int):
        self.seed = seed & 0xFFFFFFFF

    def __call__(self) -> float:
        self.seed = (self.seed + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.seed
        t = imul(t ^ (t >> 15), t | 1)
        t ^= (t + imul(t ^ (t >> 7), t | 61)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0


def normal_sample(random) -> float:
    u1 = max(random(), 1e-12)
    u2 = max(random(), 1e-12)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def deg_to_rad(value: float) -> float:
    return value * math.pi / 180.0


def hoop_center(params: ShotParams):
    return [params.distance_to_hoop, HOOP_Y, params.hoop_height]


def release_height(params: ShotParams) -> float:
    return params.boy_height * 0.85


def build_wind_grid():
    points = []
    nx = WIND_GRID_SIZE["x"]
    ny = WIND_GRID_SIZE["y"]
    nz = WIND_GRID_SIZE["z"]
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                points.append(
                    [
                        lerp(SCENE_BOUNDS["x_min"] + 0.25, SCENE_BOUNDS["x_max"] - 0.25, ix / (nx - 1)),
                        lerp(SCENE_BOUNDS["y_min"] + 0.25, SCENE_BOUNDS["y_max"] - 0.25, iy / (ny - 1)),
                        lerp(0.42, SCENE_BOUNDS["z_max"] - 0.28, iz / (nz - 1)),
                    ]
                )
    return points


def make_base_wind(params: ShotParams):
    angle = deg_to_rad(params.wind_orientation)
    elevation = deg_to_rad(params.wind_vertical_orientation)
    horizontal_strength = params.wind_strength * math.cos(elevation)
    return [
        horizontal_strength * math.cos(angle),
        horizontal_strength * math.sin(angle),
        params.wind_strength * math.sin(elevation),
    ]


def create_modes(random, count: int):
    modes = []
    for _index in range(count):
        direction = normalize([
            random() * 2.0 - 1.0,
            random() * 2.0 - 1.0,
            0.65 * (random() * 2.0 - 1.0),
        ])
        modes.append(
            {
                "k": [
                    (0.35 + 1.15 * random()) * (-1.0 if random() < 0.5 else 1.0),
                    (0.35 + 1.25 * random()) * (-1.0 if random() < 0.5 else 1.0),
                    (0.28 + 0.95 * random()) * (-1.0 if random() < 0.5 else 1.0),
                ],
                "omega": 0.45 + 1.9 * random(),
                "phase": 2.0 * math.pi * random(),
                "direction": direction,
                "amplitude": 0.65 + 0.7 * random(),
            }
        )
    return modes


def spectral_noise(x: float, y: float, z: float, t: float, modes):
    total = [0.0, 0.0, 0.0]
    amplitude_total = 0.0
    for mode in modes:
        phase = (
            mode["k"][0] * x
            + mode["k"][1] * y
            + mode["k"][2] * z
            + mode["omega"] * t
            + mode["phase"]
        )
        wave = math.sin(phase)
        total = add(total, scale(mode["direction"], mode["amplitude"] * wave))
        amplitude_total += mode["amplitude"]
    if amplitude_total <= 1e-12:
        return total
    return scale(total, 1.0 / amplitude_total)


def lorenz_derivative(state):
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    return [
        sigma * (state[1] - state[0]),
        state[0] * (rho - state[2]) - state[1],
        state[0] * state[1] - beta * state[2],
    ]


def lorenz_step(state, dt: float):
    k1 = lorenz_derivative(state)
    k2 = lorenz_derivative(add(state, scale(k1, dt / 2.0)))
    k3 = lorenz_derivative(add(state, scale(k2, dt / 2.0)))
    k4 = lorenz_derivative(add(state, scale(k3, dt)))
    return [
        value + (dt / 6.0) * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index])
        for index, value in enumerate(state)
    ]


def create_chaotic_states(random, strength: float):
    states = []
    state = [
        0.72 + 0.34 * random(),
        0.95 + 0.42 * random(),
        1.08 + 0.38 * random(),
    ]
    lorenz_dt = 0.01

    for _burn in range(140):
        state = lorenz_step(state, lorenz_dt)

    for _step in range(MAX_STEPS):
        for _substep in range(8):
            state = lorenz_step(state, lorenz_dt)
        states.append(
            vector_limited(
                [
                    state[0] / 20.0,
                    state[1] / 27.0,
                    (state[2] - 25.0) / 25.0,
                ],
                1.45 + 0.05 * math.log1p(strength),
            )
        )
    return states


def create_wind_context(params: ShotParams):
    seed_text = "|".join(
        [
            js_number_string(params.initial_force),
            js_number_string(params.vertical_angle),
            js_number_string(params.horizontal_angle),
            js_number_string(params.distance_to_hoop),
            js_number_string(params.boy_height),
            js_number_string(params.ball_mass),
            js_number_string(params.hoop_height),
            js_number_string(params.wind_regime_id),
            js_number_string(params.wind_spatial_coupling),
            js_number_string(params.wind_strength),
            js_number_string(params.wind_orientation),
            js_number_string(params.wind_vertical_orientation),
            js_number_string(params.drag_coeff),
        ]
    )
    random = Mulberry32(hash_string(seed_text))
    modes = create_modes(random, 12)
    base = make_base_wind(params)

    markov_states = []
    tau = 0.82
    alpha = math.exp(-DT / tau)
    sigma = 0.52 * params.wind_strength * math.sqrt(1.0 - alpha * alpha)
    markov_state = base[:]
    for step in range(MAX_STEPS):
        innovation = [
            sigma * normal_sample(random),
            sigma * normal_sample(random),
            0.38 * sigma * normal_sample(random),
        ]
        target = add(base, scale(spectral_noise(-2.0, 0.0, 2.0, step * DT, modes), 0.2 * params.wind_strength))
        markov_state = add(target, add(scale(sub(markov_state, target), alpha), innovation))
        markov_state = vector_limited(markov_state, max(0.1, 2.25 * params.wind_strength))
        markov_states.append(markov_state[:])

    gusts = []
    for index in range(5):
        gusts.append(
            {
                "center": 0.55 + index * 0.86 + (random() - 0.5) * 0.2,
                "width": 0.07 + random() * 0.08,
                "amplitude": 0.85 + random() * 1.45,
                "lateral_phase": 2.0 * math.pi * random(),
            }
        )

    volatility_states = []
    volatility_alpha = math.exp(-DT / 0.9)
    volatility_sigma = math.sqrt(1.0 - volatility_alpha * volatility_alpha)
    volatility_state = 0.0
    for _step in range(MAX_STEPS):
        volatility_state = volatility_alpha * volatility_state + volatility_sigma * normal_sample(random)
        volatility_states.append(clamp(math.exp(0.72 * volatility_state), 0.28, 3.8))

    shocks = []
    for index in range(3):
        shock_angle = deg_to_rad(params.wind_orientation) + (random() - 0.5) * math.pi * 1.35
        shocks.append(
            {
                "center": 0.85 + index * 1.22 + (random() - 0.5) * 0.38,
                "width": 0.045 + random() * 0.105,
                "amplitude": 1.25 + random() * 1.85,
                "direction": normalize([
                    math.cos(shock_angle),
                    math.sin(shock_angle),
                    0.18 * (random() * 2.0 - 1.0),
                ]),
                "spatial_phase": 2.0 * math.pi * random(),
            }
        )

    liquidity_walls = []
    for index in range(3):
        liquidity_walls.append(
            {
                "x": params.distance_to_hoop * (0.34 + index * 0.18 + (random() - 0.5) * 0.08),
                "y": (random() - 0.5) * 3.2,
                "z": 1.05 + random() * 2.95,
                "width_x": 0.24 + random() * 0.22,
                "width_y": 0.65 + random() * 0.65,
                "width_z": 0.75 + random() * 0.65,
                "amplitude": 0.75 + random() * 0.85,
                "polarity": 1.0 if random() < 0.72 else -1.0,
            }
        )

    hidden_states = []
    hidden_state = math.floor(random() * 4.0)
    for step in range(MAX_STEPS):
        if step > 0 and random() < 0.018:
            hidden_state = (hidden_state + 1 + math.floor(random() * 3.0)) % 4
        hidden_states.append(hidden_state)

    return {
        "base": base,
        "modes": modes,
        "markov_states": markov_states,
        "gusts": gusts,
        "volatility_states": volatility_states,
        "shocks": shocks,
        "liquidity_walls": liquidity_walls,
        "hidden_states": hidden_states,
        "chaotic_states": create_chaotic_states(random, params.wind_strength),
        "squeeze_time": 1.35 + random() * 1.45,
    }


def spatial_coupling_wind(x: float, y: float, z: float, t: float, base, perpendicular, strength: float):
    mixed = [
        0.42 * strength * math.sin(1.15 * y + 1.22 * t) + 0.18 * strength * math.cos(0.55 * z + 0.32 * x),
        0.42 * strength * math.cos(0.78 * x - 1.05 * t) + 0.20 * strength * math.sin(0.82 * z + 0.55 * y),
        0.32 * strength * math.sin(1.10 * z + 1.72 * t) + 0.23 * strength * math.cos(0.70 * x + 0.62 * y - t),
    ]
    swirl = [
        -0.18 * strength * math.sin(0.55 * y + t),
        0.18 * strength * math.sin(0.45 * x - 0.7 * t),
        0.12 * strength * math.sin(0.5 * x - 0.4 * y + t),
    ]
    shear = 0.16 * strength * math.sin(0.42 * x + 0.68 * y + 0.55 * z + 0.9 * t)
    return add(add(mixed, swirl), add(scale(base, 0.1 * math.sin(0.65 * z + t)), scale(perpendicular, shear)))


def wind_at(x: float, y: float, z: float, t: float, params: ShotParams, wind_context):
    strength = params.wind_strength
    if strength <= 1e-9 or params.wind_regime_id == 0:
        return [0.0, 0.0, 0.0]

    angle = deg_to_rad(params.wind_orientation)
    base = wind_context["base"]
    perpendicular = [-math.sin(angle), math.cos(angle), 0.0]
    distance_scale = max(params.distance_to_hoop, 1.0)
    base_dir = normalize(base)

    regime = params.wind_regime_id
    if regime == 1:
        wind = base[:]
    elif regime == 2:
        wind = add(
            scale(base, 0.82 + 0.18 * math.sin(1.7 * t + 0.4 * x)),
            scale(spectral_noise(x, y, z, t, wind_context["modes"]), 0.72 * strength),
        )
    elif regime == 3:
        step = max(0, min(MAX_STEPS - 1, js_round(t / DT)))
        wind = add(
            wind_context["markov_states"][step],
            scale(spectral_noise(x, y, z, t, wind_context["modes"]), 0.22 * strength),
        )
    elif regime == 4:
        phase = 2.0 * math.pi * t / 2.35
        wind = add(add(scale(base, math.sin(phase)), scale(perpendicular, 0.34 * strength * math.cos(phase))), [0.0, 0.0, 0.22 * strength * math.sin(2.0 * phase + 0.45 * x)])
    elif regime == 5:
        gust_factor = 1.0
        for gust in wind_context["gusts"]:
            time_pulse = math.exp(-((t - gust["center"]) ** 2) / (2.0 * gust["width"] ** 2))
            spatial_pulse = 0.76 + 0.24 * math.sin(0.7 * x - 0.5 * y + gust["lateral_phase"])
            gust_factor += gust["amplitude"] * time_pulse * spatial_pulse
        wind = add(
            add(scale(base, gust_factor), scale(perpendicular, 0.16 * strength * math.sin(2.8 * t + y))),
            scale(spectral_noise(x, y, z, t, wind_context["modes"]), 0.26 * strength),
        )
    elif regime == 6:
        shear = 0.18 * strength * math.sin(0.9 * z + 0.5 * y)
        wind = add(scale(base, 1.0 if t < MAX_STEPS * DT * 0.44 else -1.0), scale(perpendicular, shear))
    elif regime == 7:
        progress = clamp(x / distance_scale, 0.0, 1.15)
        anchor_x = 0.55 * distance_scale
        target_z = lerp(release_height(params), params.hoop_height, progress)
        mean_pull = [
            -0.32 * strength * math.tanh((x - anchor_x) / (0.35 * distance_scale)),
            -0.78 * strength * math.tanh((y - HOOP_Y) / 2.1),
            0.25 * strength * math.tanh((target_z - z) / 1.35),
        ]
        wind = add(
            add(scale(base, 0.18 + 0.12 * math.cos(1.4 * t)), mean_pull),
            scale(spectral_noise(x, y, z, t, wind_context["modes"]), 0.12 * strength),
        )
    elif regime == 8:
        step = max(0, min(MAX_STEPS - 1, js_round(t / DT)))
        volatility = wind_context["volatility_states"][step]
        clustered_noise = scale(spectral_noise(x, y, z, t, wind_context["modes"]), strength * (0.22 + 0.58 * volatility))
        persistent_drift = scale(base, 0.32 + 0.16 * volatility)
        lateral_chop = scale(perpendicular, 0.14 * strength * volatility * math.sin(2.2 * t + 0.45 * x - 0.3 * y))
        wind = add(add(persistent_drift, clustered_noise), lateral_chop)
    elif regime == 9:
        wind = scale(base, 0.22)
        for shock in wind_context["shocks"]:
            time_pulse = math.exp(-((t - shock["center"]) ** 2) / (2.0 * shock["width"] ** 2))
            spatial_pulse = 0.72 + 0.28 * math.sin(0.65 * x - 0.48 * y + 0.25 * z + shock["spatial_phase"])
            wind = add(wind, scale(shock["direction"], strength * shock["amplitude"] * time_pulse * spatial_pulse))
        wind = add(wind, scale(spectral_noise(x, y, z, t, wind_context["modes"]), 0.1 * strength))
    elif regime == 10:
        pump = smoothstep(0.15, 0.75, t) * (1.0 - smoothstep(1.35, 2.25, t))
        fade = smoothstep(1.75, 3.45, t)
        wind = add(
            add(scale(base, 0.16 + 1.65 * pump - 0.82 * fade), scale(perpendicular, strength * (0.12 + 0.25 * fade) * math.sin(2.3 * t + 0.6 * x))),
            [0.0, 0.0, 0.14 * strength * fade * math.sin(1.6 * t + 0.4 * y)],
        )
    elif regime == 11:
        progress = smoothstep(0.12 * distance_scale, distance_scale, x)
        lateral = math.tanh((y + 0.3 * math.sin(0.9 * x + 1.1 * t)) / 1.2)
        amplification = 0.26 + 0.62 * progress + 0.5 * abs(lateral)
        cascade = scale(perpendicular, strength * lateral * (0.25 + 0.9 * progress))
        lift = [0.0, 0.0, 0.1 * strength * abs(lateral) * math.sin(1.8 * t + 0.35 * x)]
        turbulent = scale(spectral_noise(x, y, z, t, wind_context["modes"]), 0.15 * strength * (1.0 + progress))
        wind = add(add(add(scale(base, amplification), cascade), lift), turbulent)
    elif regime == 12:
        wind = scale(base, 0.22)
        for wall in wind_context["liquidity_walls"]:
            dx = (x - wall["x"]) / wall["width_x"]
            dy = (y - wall["y"]) / wall["width_y"]
            dz = (z - wall["z"]) / wall["width_z"]
            envelope = math.exp(-0.5 * (dx * dx + dy * dy + 0.55 * dz * dz))
            resistance = scale(base_dir, -wall["polarity"] * strength * wall["amplitude"] * envelope)
            lateral_gradient = scale(perpendicular, -0.42 * strength * wall["amplitude"] * dy * envelope)
            vertical_gradient = [0.0, 0.0, -0.22 * strength * wall["amplitude"] * dz * envelope]
            wind = add(wind, add(add(resistance, lateral_gradient), vertical_gradient))
        wind = add(wind, scale(spectral_noise(x, y, z, t, wind_context["modes"]), 0.08 * strength))
    elif regime == 13:
        t0 = wind_context["squeeze_time"]
        compression = 1.0 - smoothstep(t0 - 0.65, t0, t)
        breakout = smoothstep(t0, t0 + 0.35, t)
        directional_factor = 0.08 * compression + (1.65 + 0.22 * math.sin(6.0 * (t - t0) + 0.5 * x)) * breakout
        turbulence_level = 0.06 * compression + 0.35 * breakout
        wind = add(
            add(scale(base, directional_factor), scale(spectral_noise(x, y, z, t, wind_context["modes"]), turbulence_level * strength)),
            [0.0, 0.0, 0.15 * strength * breakout * math.sin(2.1 * t + 0.7 * y)],
        )
    elif regime == 14:
        step = max(0, min(MAX_STEPS - 1, js_round(t / DT)))
        state = wind_context["hidden_states"][step]
        local = spectral_noise(x, y, z, t, wind_context["modes"])
        if state == 0:
            wind = add(scale(base, 0.92), scale(local, 0.1 * strength))
        elif state == 1:
            wind = add(add(scale(base, -0.62), scale(perpendicular, 0.28 * strength * math.sin(1.9 * t + 0.4 * x))), scale(local, 0.12 * strength))
        elif state == 2:
            wind = add(scale(base, 0.15), scale(local, 1.15 * strength))
        else:
            wind = add(
                add(scale(base, 0.22), [0.0, -0.74 * strength * math.tanh((y - HOOP_Y) / 2.0), 0.2 * strength * math.tanh((params.hoop_height - z) / 1.5)]),
                scale(local, 0.16 * strength),
            )
    elif regime == 15:
        step = max(0, min(MAX_STEPS - 1, js_round(t / DT)))
        chaotic = wind_context["chaotic_states"][step]
        local = spectral_noise(
            x + 0.9 * chaotic[1],
            y - 0.7 * chaotic[0],
            z + 0.55 * chaotic[2],
            t * 1.85,
            wind_context["modes"],
        )
        vortex = [
            -math.sin(1.28 * y + 1.7 * t + chaotic[2]) + 0.36 * chaotic[0] * math.cos(0.58 * z + 2.1 * t),
            math.sin(1.06 * x - 1.4 * t + chaotic[0]) + 0.36 * chaotic[1] * math.sin(0.52 * z - 1.6 * t),
            0.58 * math.sin(0.88 * z + 0.55 * x - 0.42 * y + 1.9 * t + chaotic[2]),
        ]
        drift = scale(base, 0.18 + 0.22 * math.sin(2.4 * t + chaotic[0]))
        wind = vector_limited(
            add(add(drift, scale(chaotic, 1.25 * strength)), add(scale(vortex, 0.34 * strength), scale(local, 0.62 * strength))),
            max(0.2, 3.4 * strength),
        )
    else:
        wind = [0.0, 0.0, 0.0]

    if params.wind_spatial_coupling:
        wind = add(wind, spatial_coupling_wind(x, y, z, t, base, perpendicular, strength))
    return wind


def acceleration_at(position, velocity, t: float, params: ShotParams, wind_context):
    wind = wind_at(position[0], position[1], position[2], t, params, wind_context)
    relative_velocity = sub(velocity, wind)
    relative_speed = norm(relative_velocity)
    drag_acc = scale(relative_velocity, -(params.drag_coeff * relative_speed) / max(params.ball_mass, 1e-6))
    return wind, [drag_acc[0], drag_acc[1], -GRAVITY + drag_acc[2]]


def derivative(state_vector, t: float, params: ShotParams, wind_context):
    position = state_vector[:3]
    velocity = state_vector[3:]
    _, acc = acceleration_at(position, velocity, t, params, wind_context)
    return [velocity[0], velocity[1], velocity[2], acc[0], acc[1], acc[2]]


def combine_state(state_vector, derivative_vector, scalar: float):
    return [value + derivative_vector[index] * scalar for index, value in enumerate(state_vector)]


def rk4_step(position, velocity, t: float, params: ShotParams, wind_context):
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


def crossing_result(prev_position, position, prev_velocity, velocity, prev_time, time, params: ShotParams):
    if not prev_position or prev_position[2] < params.hoop_height or position[2] > params.hoop_height:
        return None
    dz = prev_position[2] - position[2]
    if abs(dz) < 1e-9:
        return None
    alpha = clamp((prev_position[2] - params.hoop_height) / dz, 0.0, 1.0)
    crossing_x = lerp(prev_position[0], position[0], alpha)
    crossing_y = lerp(prev_position[1], position[1], alpha)
    crossing_vz = lerp(prev_velocity[2], velocity[2], alpha)
    horizontal_distance = math.hypot(crossing_x - params.distance_to_hoop, crossing_y - HOOP_Y)
    if crossing_vz < 0.0 and horizontal_distance <= HOOP_RADIUS:
        return lerp(prev_time, time, alpha)
    return None


def simulate_shot(shot_id: int, params: ShotParams):
    wind_context = create_wind_context(params)
    vertical = deg_to_rad(params.vertical_angle)
    horizontal = deg_to_rad(params.horizontal_angle)
    position = [PLAYER_X, PLAYER_Y, release_height(params)]
    velocity = [
        params.initial_force * math.cos(vertical) * math.cos(horizontal),
        params.initial_force * math.cos(vertical) * math.sin(horizontal),
        params.initial_force * math.sin(vertical),
    ]

    trajectory_rows = []
    label = 0
    collision_time = None
    final_min_distance = float("inf")
    prev_position = None
    prev_velocity = None
    prev_time = None

    for step in range(MAX_STEPS):
        time = step * DT
        wind, acc = acceleration_at(position, velocity, time, params, wind_context)
        center = hoop_center(params)
        distance = norm(sub(position, center))
        horizontal_distance = math.hypot(position[0] - center[0], position[1] - center[1])
        speed = norm(velocity)
        final_min_distance = min(final_min_distance, distance)

        if label == 0 and prev_position is not None:
            crossing_time = crossing_result(prev_position, position, prev_velocity, velocity, prev_time, time, params)
            if crossing_time is not None:
                label = 1
                collision_time = crossing_time

        trajectory_rows.append(
            {
                "shot_id": shot_id,
                "timestep": step,
                "time": time,
                "ball_x": position[0],
                "ball_y": position[1],
                "ball_z": position[2],
                "ball_vx": velocity[0],
                "ball_vy": velocity[1],
                "ball_vz": velocity[2],
                "ball_ax": acc[0],
                "ball_ay": acc[1],
                "ball_az": acc[2],
                "distance_ball_to_hoop": distance,
                "horizontal_distance_to_hoop": horizontal_distance,
                "speed_norm": speed,
            }
        )

        if position[2] < -0.2 and velocity[2] < 0.0 and step > 6:
            break

        prev_position = position[:]
        prev_velocity = velocity[:]
        prev_time = time
        position, velocity = rk4_step(position, velocity, time, params, wind_context)

    shot_row = {
        "shot_id": shot_id,
        "initial_force": params.initial_force,
        "vertical_angle": params.vertical_angle,
        "horizontal_angle": params.horizontal_angle,
        "distance_to_hoop": params.distance_to_hoop,
        "boy_height": params.boy_height,
        "ball_mass": params.ball_mass,
        "hoop_height": params.hoop_height,
        "wind_regime_id": params.wind_regime_id,
        "wind_regime": params.wind_regime,
        "wind_strength": params.wind_strength,
        "wind_orientation": params.wind_orientation,
        "wind_vertical_orientation": params.wind_vertical_orientation,
        "wind_spatial_coupling": params.wind_spatial_coupling,
        "wind_field_visible": params.wind_field_visible,
        "field_view": params.field_view,
        "past_trail_visible": params.past_trail_visible,
        "drag_coeff": params.drag_coeff,
        "gravity": GRAVITY,
    }
    label_row = {
        "shot_id": shot_id,
        "label": label,
        "result": "MADE" if label == 1 else "MISS",
        "distance_to_hoop": params.distance_to_hoop,
        "final_min_hoop_distance": final_min_distance,
        "collision_time": collision_time,
    }
    return shot_row, trajectory_rows, label_row, wind_context


def build_render_query(params: ShotParams, timestep: int, view: str) -> str:
    return urlencode(
        {
            "camera": view,
            "frame": timestep,
            "initial_force": params.initial_force,
            "vertical_angle": params.vertical_angle,
            "horizontal_angle": params.horizontal_angle,
            "distance_to_hoop": params.distance_to_hoop,
            "boy_height": params.boy_height,
            "ball_mass": params.ball_mass,
            "hoop_height": params.hoop_height,
            "wind_regime_id": params.wind_regime_id,
            "wind_strength": params.wind_strength,
            "wind_orientation": params.wind_orientation,
            "wind_vertical_orientation": params.wind_vertical_orientation,
            "wind_spatial_coupling": params.wind_spatial_coupling,
            "drag_coeff": params.drag_coeff,
            "show_wind_field": params.wind_field_visible,
            "field_view": params.field_view,
            "show_past_trail": params.past_trail_visible,
        }
    )


def write_csv_header(path: Path, columns):
    file = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    return file, writer


def optional_binary(value: str) -> int:
    parsed = int(value)
    if parsed not in (0, 1):
        raise argparse.ArgumentTypeError("Value must be 0 or 1.")
    return parsed


def optional_field_view(value: str) -> int:
    parsed = int(value)
    if parsed not in (0, 1, 2):
        raise argparse.ArgumentTypeError("Value must be 0, 1, or 2.")
    return parsed


def filter_combinations(combinations, args):
    for params in combinations:
        if args.wind_regime_id is not None and params.wind_regime_id != args.wind_regime_id:
            continue
        if args.wind_orientation is not None and params.wind_orientation != args.wind_orientation:
            continue
        if args.wind_vertical_orientation is not None and params.wind_vertical_orientation != args.wind_vertical_orientation:
            continue
        if args.skip_no_wind and params.wind_regime_id == 0:
            continue
        if args.wind_field_visible is not None and params.wind_field_visible != args.wind_field_visible:
            continue
        if args.field_view is not None and params.field_view != args.field_view:
            continue
        if args.past_trail_visible is not None and params.past_trail_visible != args.past_trail_visible:
            continue
        yield params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dataset_config", help="Configuration module to use.")
    parser.add_argument("--output-dir", default="generated_dataset", help="Output directory.")
    parser.add_argument("--max-shots", type=int, default=10, help="Maximum shots. 0 = every combination.")
    parser.add_argument("--frame-stride", type=int, default=5, help="Reference one frame every N timesteps.")
    parser.add_argument("--no-frame-manifest", action="store_true", help="Do not write frames_manifest.csv.")
    parser.add_argument("--wind-field-visible", type=optional_binary, default=None, help="Filter wind-field visibility: 0 or 1.")
    parser.add_argument("--field-view", type=optional_field_view, default=None, help="Filter field view: 0 = vectors, 1 = slice map, 2 = 3D volume.")
    parser.add_argument("--past-trail-visible", type=optional_binary, default=None, help="Filter past-trail visibility: 0 or 1.")
    parser.add_argument("--wind-regime-id", type=int, default=None, help="Filter one wind regime, for example 15 for chaotic wind.")
    parser.add_argument("--wind-orientation", type=float, default=None, help="Filter horizontal wind azimuth in degrees.")
    parser.add_argument("--wind-vertical-orientation", type=float, default=None, help="Filter vertical wind elevation in degrees.")
    parser.add_argument("--skip-no-wind", action="store_true", help="Skip wind regime 0 to guarantee a non-zero wind field.")
    args = parser.parse_args()
    config = load_config_module(args.config)
    apply_config(config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wind_grid = build_wind_grid()

    total = count_combinations(config)
    shot_limit = None if args.max_shots == 0 else args.max_shots
    combinations = filter_combinations(iter_parameter_combinations(config), args)
    if shot_limit is not None:
        combinations = islice(combinations, shot_limit)

    shots_file, shots_writer = write_csv_header(output_dir / "shots.csv", SHOTS_COLUMNS)
    trajectory_file, trajectory_writer = write_csv_header(output_dir / "trajectory.csv", TRAJECTORY_COLUMNS)
    wind_file, wind_writer = write_csv_header(output_dir / "wind_field.csv", WIND_FIELD_COLUMNS)
    labels_file, labels_writer = write_csv_header(output_dir / "labels.csv", LABEL_COLUMNS)
    manifest_file = None
    manifest_writer = None
    if not args.no_frame_manifest:
        manifest_file, manifest_writer = write_csv_header(output_dir / "frames_manifest.csv", FRAME_MANIFEST_COLUMNS)

    generated = 0
    try:
        for shot_id, params in enumerate(combinations):
            shot_row, trajectory_rows, label_row, wind_context = simulate_shot(shot_id, params)
            shots_writer.writerow(shot_row)
            labels_writer.writerow(label_row)
            trajectory_writer.writerows(trajectory_rows)

            for row in trajectory_rows:
                for wind_vector_id, point in enumerate(wind_grid):
                    wind = wind_at(point[0], point[1], point[2], row["time"], params, wind_context)
                    wind_writer.writerow(
                        {
                            "shot_id": shot_id,
                            "timestep": row["timestep"],
                            "time": row["time"],
                            "wind_vector_id": wind_vector_id,
                            "wind_grid_x": point[0],
                            "wind_grid_y": point[1],
                            "wind_grid_z": point[2],
                            "wind_x": wind[0],
                            "wind_y": wind[1],
                            "wind_z": wind[2],
                            "wind_norm": norm(wind),
                        }
                    )

            if manifest_writer is not None:
                for row in trajectory_rows[:: max(1, args.frame_stride)]:
                    for view in CAMERA_VIEWS:
                        relative_path = (
                            f"frames/shot_{shot_id:06d}/"
                            f"{view}/frame_{row['timestep']:06d}.png"
                        )
                        manifest_writer.writerow(
                            {
                                "shot_id": shot_id,
                                "timestep": row["timestep"],
                                "time": row["time"],
                                "camera_view": view,
                                "wind_field_visible": params.wind_field_visible,
                                "field_view": params.field_view,
                                "past_trail_visible": params.past_trail_visible,
                                "render_query": build_render_query(params, row["timestep"], view),
                                "relative_path": relative_path,
                            }
                        )

            generated += 1
            print(f"shot {shot_id:06d} generated ({generated}/{shot_limit or total})")
    finally:
        shots_file.close()
        trajectory_file.close()
        wind_file.close()
        labels_file.close()
        if manifest_file is not None:
            manifest_file.close()

    print()
    print(f"Available observations  : {total:,}".replace(",", " "))
    print(f"Generated shots         : {generated:,}".replace(",", " "))
    print(f"Output directory        : {output_dir.resolve()}")
    if not args.no_frame_manifest:
        print("Frames                  : manifest only")
        print("                           exact rendering should come from ai-cameras.html")
        print(f"Frame render command    : node dataset_builder/render_frames.js --dataset-dir {output_dir}")


if __name__ == "__main__":
    main()

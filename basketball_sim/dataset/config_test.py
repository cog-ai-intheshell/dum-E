"""Tiny test configuration for validating dataset generation.

This configuration produces exactly 2 observations:
- one no-wind observation;
- one chaotic-wind observation.

.venv/bin/python -m basketball_sim.dataset.generate --config dataset_config_test --output-dir generated_dataset_config_test --max-shots 0 --frame-stride 1
node dataset_builder/render_frames.js --dataset-dir generated_dataset_config_test --force

"""

import sys

# Global physics.
GRAVITY = 9.81
DT = 0.02
MAX_STEPS = 250
HOOP_RADIUS = 0.23

# Scene bounds used to sample the wind field.
PLAYER_X = 0.0
PLAYER_Y = 0.0
HOOP_Y = 0.0
SCENE_BOUNDS = {
    "x_min": -0.75,
    "x_max": 13.15,
    "y_min": -4.5,
    "y_max": 4.5,
    "z_min": 0.0,
    "z_max": 5.6,
}
WIND_GRID_SIZE = {
    "x": 18,
    "y": 9,
    "z": 6,
}

# Wind regimes included in this test.
WIND_REGIME_NAMES = {
    0: "no wind",
    15: "chaotic wind",
}
WIND_REGIME_IDS = [0, 15]

# Physical parameter grid.
INITIAL_FORCE_VALUES = [12.0]
VERTICAL_ANGLE_VALUES = [45.0]
HORIZONTAL_ANGLE_VALUES = [0.0]
DISTANCE_TO_HOOP_VALUES = [6.0]
BOY_HEIGHT_VALUES = [1.75]
BALL_MASS_VALUES = [0.62]
HOOP_HEIGHT_VALUES = [3.05]

# Wind parameter grid.
WIND_STRENGTH_VALUES = [4.0]
WIND_ORIENTATION_VALUES = [45.0]
WIND_VERTICAL_ORIENTATION_VALUES = [0.0]
WIND_SPATIAL_COUPLING_VALUES = [1]

# Drag parameter grid.
DRAG_COEFF_VALUES = [0.02]

# Observation parameters: one mode only, so the config yields exactly 2 observations.
WIND_FIELD_VISIBLE_VALUES = [1]
FIELD_VIEW_VALUES = [2]
PAST_TRAIL_VISIBLE_VALUES = [1]

# Camera views exported for the video dataset.
CAMERA_VIEWS = ("profile", "top", "rear", "hoop", "left", "oblique")


def _show(name: str, value) -> None:
    print(f"{name}: {value}")


def print_fields() -> None:
    from .grid import count_combinations, count_observation_modes

    config = sys.modules[__name__]
    print("dataset_config_test fields")
    print("==========================")
    _show("INITIAL_FORCE_VALUES", INITIAL_FORCE_VALUES)
    _show("VERTICAL_ANGLE_VALUES", VERTICAL_ANGLE_VALUES)
    _show("HORIZONTAL_ANGLE_VALUES", HORIZONTAL_ANGLE_VALUES)
    _show("DISTANCE_TO_HOOP_VALUES", DISTANCE_TO_HOOP_VALUES)
    _show("BOY_HEIGHT_VALUES", BOY_HEIGHT_VALUES)
    _show("BALL_MASS_VALUES", BALL_MASS_VALUES)
    _show("HOOP_HEIGHT_VALUES", HOOP_HEIGHT_VALUES)
    _show("WIND_REGIME_IDS", WIND_REGIME_IDS)
    _show("WIND_REGIME_NAMES", WIND_REGIME_NAMES)
    _show("WIND_STRENGTH_VALUES", WIND_STRENGTH_VALUES)
    _show("WIND_ORIENTATION_VALUES", WIND_ORIENTATION_VALUES)
    _show("WIND_VERTICAL_ORIENTATION_VALUES", WIND_VERTICAL_ORIENTATION_VALUES)
    _show("WIND_SPATIAL_COUPLING_VALUES", WIND_SPATIAL_COUPLING_VALUES)
    _show("DRAG_COEFF_VALUES", DRAG_COEFF_VALUES)
    _show("WIND_FIELD_VISIBLE_VALUES", WIND_FIELD_VISIBLE_VALUES)
    _show("FIELD_VIEW_VALUES", FIELD_VIEW_VALUES)
    _show("PAST_TRAIL_VISIBLE_VALUES", PAST_TRAIL_VISIBLE_VALUES)
    _show("CAMERA_VIEWS", CAMERA_VIEWS)
    print()
    _show("Observation modes", count_observation_modes(config))
    _show("Total observations", count_combinations(config))


if __name__ == "__main__":
    print_fields()

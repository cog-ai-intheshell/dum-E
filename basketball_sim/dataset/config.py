"""Default dataset configuration.

Keep this file declarative: only editable constants live here. Combination,
counting, simulation, and export logic lives in dedicated modules.
"""

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

# Wind regimes.
WIND_REGIME_NAMES = {
    0: "no wind",
    1: "constant wind",
    2: "random wind",
    3: "markov wind",
    4: "cyclical wind",
    5: "gusts",
    6: "abrupt shift",
    7: "mean reversion",
    8: "volatility clustering",
    9: "jump / news shock",
    10: "pump & fade",
    11: "liquidation cascade",
    12: "liquidity wall",
    13: "squeeze breakout",
    14: "hidden regime switching",
    15: "chaotic wind",
}
WIND_REGIME_IDS = list(WIND_REGIME_NAMES)

# Physical parameter grid.
INITIAL_FORCE_VALUES = [10.0, 12.0, 14.0]
VERTICAL_ANGLE_VALUES = [40.0, 45.0, 50.0]
HORIZONTAL_ANGLE_VALUES = [-6.0, 0.0, 6.0]
DISTANCE_TO_HOOP_VALUES = [4.5, 6.0, 7.5]
BOY_HEIGHT_VALUES = [1.65, 1.75]
BALL_MASS_VALUES = [0.58, 0.62]
HOOP_HEIGHT_VALUES = [3.05]

# Wind parameter grid.
WIND_STRENGTH_VALUES = [4.0, 8.0]
WIND_ORIENTATION_VALUES = [0.0, 90.0, 180.0, 270.0]
WIND_VERTICAL_ORIENTATION_VALUES = [-20.0, 0.0, 20.0]
WIND_SPATIAL_COUPLING_VALUES = [0, 1]

# Drag parameter grid.
DRAG_COEFF_VALUES = [0.02, 0.05]

# Observation parameters: these control what the AI sees in rendered frames.
WIND_FIELD_VISIBLE_VALUES = [0, 1]
FIELD_VIEW_VALUES = [0, 1, 2]
PAST_TRAIL_VISIBLE_VALUES = [1]

# Camera views exported for the video dataset.
CAMERA_VIEWS = ("profile", "top", "rear", "hoop", "left", "oblique")

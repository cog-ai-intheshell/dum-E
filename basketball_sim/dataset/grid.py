"""Dataset grid construction and summary utilities."""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from itertools import product

from . import config as default_config


CONFIG_ALIASES = {
    "dataset_config": "basketball_sim.dataset.config",
    "dataset_config_test": "basketball_sim.dataset.config_test",
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


def load_config_module(module_name: str):
    if module_name.endswith(".py"):
        module_name = module_name[:-3].replace("/", ".")
    module_name = CONFIG_ALIASES.get(module_name, module_name)
    return importlib.import_module(module_name)


def iter_parameter_combinations(config=default_config):
    """Iterate over all non-redundant parameter combinations."""
    base_grid = product(
        config.INITIAL_FORCE_VALUES,
        config.VERTICAL_ANGLE_VALUES,
        config.HORIZONTAL_ANGLE_VALUES,
        config.DISTANCE_TO_HOOP_VALUES,
        config.BOY_HEIGHT_VALUES,
        config.BALL_MASS_VALUES,
        config.HOOP_HEIGHT_VALUES,
        config.DRAG_COEFF_VALUES,
    )

    for (
        initial_force,
        vertical_angle,
        horizontal_angle,
        distance_to_hoop,
        boy_height,
        ball_mass,
        hoop_height,
        drag_coeff,
    ) in base_grid:
        for wind_regime_id in config.WIND_REGIME_IDS:
            if wind_regime_id == 0:
                wind_grid = [(0.0, 0.0, 0.0, 0)]
            else:
                wind_grid = product(
                    config.WIND_STRENGTH_VALUES,
                    config.WIND_ORIENTATION_VALUES,
                    config.WIND_VERTICAL_ORIENTATION_VALUES,
                    config.WIND_SPATIAL_COUPLING_VALUES,
                )

            for wind_strength, wind_orientation, wind_vertical_orientation, wind_spatial_coupling in wind_grid:
                for wind_field_visible in config.WIND_FIELD_VISIBLE_VALUES:
                    field_view_values = config.FIELD_VIEW_VALUES if wind_field_visible else [0]
                    for field_view in field_view_values:
                        for past_trail_visible in config.PAST_TRAIL_VISIBLE_VALUES:
                            yield ShotParams(
                                initial_force=initial_force,
                                vertical_angle=vertical_angle,
                                horizontal_angle=horizontal_angle,
                                distance_to_hoop=distance_to_hoop,
                                boy_height=boy_height,
                                ball_mass=ball_mass,
                                hoop_height=hoop_height,
                                wind_regime_id=wind_regime_id,
                                wind_regime=config.WIND_REGIME_NAMES[wind_regime_id],
                                wind_strength=wind_strength,
                                wind_orientation=wind_orientation,
                                wind_vertical_orientation=wind_vertical_orientation,
                                wind_spatial_coupling=wind_spatial_coupling,
                                wind_field_visible=wind_field_visible,
                                field_view=field_view,
                                past_trail_visible=past_trail_visible,
                                drag_coeff=drag_coeff,
                            )


def count_combinations(config=default_config) -> int:
    return sum(1 for _ in iter_parameter_combinations(config))


def count_observation_modes(config=default_config) -> int:
    count = 0
    for wind_field_visible in config.WIND_FIELD_VISIBLE_VALUES:
        field_view_values = config.FIELD_VIEW_VALUES if wind_field_visible else [0]
        for _field_view in field_view_values:
            for _past_trail_visible in config.PAST_TRAIL_VISIBLE_VALUES:
                count += 1
    return count


def _fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def print_summary(config=default_config) -> None:
    count = count_combinations(config)
    observation_modes = count_observation_modes(config)
    physical_count = count // observation_modes
    max_rows_trajectory = count * config.MAX_STEPS
    wind_vectors = config.WIND_GRID_SIZE["x"] * config.WIND_GRID_SIZE["y"] * config.WIND_GRID_SIZE["z"]
    max_rows_wind = max_rows_trajectory * wind_vectors

    print("Basketball dataset grid")
    print("=======================")
    print(f"Physical shots                    : {_fmt_int(physical_count)}")
    print(f"Observation modes per shot        : {_fmt_int(observation_modes)}")
    print(f"Total observations                : {_fmt_int(count)}")
    print(f"Max trajectory.csv rows           : {_fmt_int(max_rows_trajectory)}")
    print(f"Wind vectors per timestep         : {_fmt_int(wind_vectors)}")
    print(f"Max wind_field.csv rows           : {_fmt_int(max_rows_wind)}")
    print()
    print("Note: ball_x, ball_y, and ball_z are generated by the simulation.")
    print("They are not grid parameters.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dataset_config", help="Configuration module to use.")
    args = parser.parse_args()
    print_summary(load_config_module(args.config))


if __name__ == "__main__":
    main()

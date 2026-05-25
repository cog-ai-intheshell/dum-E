"""Dataset generation entry points for the DUM-E world model."""

from .synthetic import build_window_dataset, iter_synthetic_shots, random_params

__all__ = ["build_window_dataset", "iter_synthetic_shots", "random_params"]

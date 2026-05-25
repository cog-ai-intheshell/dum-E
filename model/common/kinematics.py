"""Kinematics helpers for the fixed-floor DUM-E robot arm.

The frontend currently renders DUM-E with three meaningful axes:
base rotation, elbow rotation, and wrist rotation. These helpers keep that
robot geometry centralized so dataset generation, training, and serving all
agree on what "catch the ball" means.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


TARGET_Y = 0.0
CATCH_RADIUS = 0.34
ROBOT_BASE_OFFSET = 1.14
ROBOT_SHOULDER_Z = 0.76
UPPER_ARM_LENGTH = 1.55
FOREARM_LENGTH = 1.55
REST_AXIS_BASE = math.pi
REST_AXIS_ELBOW = 0.42
REST_AXIS_WRIST = -1.1


def _read(source: Any, names: tuple[str, ...], default: float) -> float:
    if isinstance(source, Mapping):
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


def distance_to_dum_e(params: Any) -> float:
    return _read(params, ("distance_to_dum_e", "distance_to_hoop", "distanceToHoop"), 6.0)


def catch_height(params: Any) -> float:
    return _read(params, ("catch_height", "hoop_height", "hoopHeight"), 3.05)


def catch_center(params: Any) -> tuple[float, float, float]:
    return (distance_to_dum_e(params), TARGET_Y, catch_height(params))


def robot_base_x(params: Any) -> float:
    return distance_to_dum_e(params) + ROBOT_BASE_OFFSET


def closest_point_on_segment(
    a: tuple[float, float, float] | list[float],
    b: tuple[float, float, float] | list[float],
    point: tuple[float, float, float] | list[float],
) -> tuple[float, tuple[float, float, float], float]:
    segment = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    rel = (point[0] - a[0], point[1] - a[1], point[2] - a[2])
    length_squared = segment[0] ** 2 + segment[1] ** 2 + segment[2] ** 2
    if length_squared < 1e-12:
        alpha = 0.0
    else:
        alpha = max(0.0, min(1.0, sum(rel[i] * segment[i] for i in range(3)) / length_squared))
    closest = (
        a[0] + segment[0] * alpha,
        a[1] + segment[1] * alpha,
        a[2] + segment[2] * alpha,
    )
    distance = math.dist(closest, point)
    return alpha, closest, distance


@dataclass(frozen=True)
class CatchPose:
    catch_x: float
    catch_y: float
    catch_z: float
    axis_base: float
    axis_elbow: float
    axis_wrist: float
    reachable: float
    reach_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class JointAxes:
    axis_base: float
    axis_elbow: float
    axis_wrist: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def rest_joint_axes() -> JointAxes:
    return JointAxes(
        axis_base=REST_AXIS_BASE,
        axis_elbow=REST_AXIS_ELBOW,
        axis_wrist=REST_AXIS_WRIST,
    )


def joint_axes_for_target(target: tuple[float, float, float] | list[float], params: Any) -> JointAxes:
    """Three-axis IK target used by the motor-controller architecture."""

    base_x = robot_base_x(params)
    dx = float(target[0]) - base_x
    dy = float(target[1]) - TARGET_Y
    dz = float(target[2]) - ROBOT_SHOULDER_Z
    horizontal = math.hypot(dx, dy)
    distance = math.hypot(horizontal, dz)
    max_reach = UPPER_ARM_LENGTH + FOREARM_LENGTH - 1e-6
    min_reach = abs(UPPER_ARM_LENGTH - FOREARM_LENGTH) + 1e-6
    clamped_distance = max(min_reach, min(max_reach, distance))
    base = math.atan2(dy, dx)
    target_pitch = math.atan2(dz, max(horizontal, 1e-6))
    shoulder_offset = math.acos(
        max(
            -1.0,
            min(
                1.0,
                (UPPER_ARM_LENGTH**2 + clamped_distance**2 - FOREARM_LENGTH**2)
                / (2.0 * UPPER_ARM_LENGTH * clamped_distance),
            ),
        )
    )
    forearm_offset = math.acos(
        max(
            -1.0,
            min(
                1.0,
                (FOREARM_LENGTH**2 + clamped_distance**2 - UPPER_ARM_LENGTH**2)
                / (2.0 * FOREARM_LENGTH * clamped_distance),
            ),
        )
    )
    elbow_pitch = target_pitch + shoulder_offset
    forearm_pitch = target_pitch - forearm_offset
    wrist_relative = forearm_pitch - elbow_pitch
    return JointAxes(
        axis_base=base,
        axis_elbow=elbow_pitch,
        axis_wrist=wrist_relative,
    )


def forward_kinematics(params: Any, axes: JointAxes) -> dict[str, tuple[float, float, float]]:
    base_x = robot_base_x(params)
    shoulder = (base_x, TARGET_Y, ROBOT_SHOULDER_Z)
    elbow_direction = (
        math.cos(axes.axis_base) * math.cos(axes.axis_elbow),
        math.sin(axes.axis_base) * math.cos(axes.axis_elbow),
        math.sin(axes.axis_elbow),
    )
    forearm_pitch = axes.axis_elbow + axes.axis_wrist
    wrist_direction = (
        math.cos(axes.axis_base) * math.cos(forearm_pitch),
        math.sin(axes.axis_base) * math.cos(forearm_pitch),
        math.sin(forearm_pitch),
    )
    elbow = (
        shoulder[0] + elbow_direction[0] * UPPER_ARM_LENGTH,
        shoulder[1] + elbow_direction[1] * UPPER_ARM_LENGTH,
        shoulder[2] + elbow_direction[2] * UPPER_ARM_LENGTH,
    )
    wrist = (
        elbow[0] + wrist_direction[0] * FOREARM_LENGTH,
        elbow[1] + wrist_direction[1] * FOREARM_LENGTH,
        elbow[2] + wrist_direction[2] * FOREARM_LENGTH,
    )
    palm = (
        wrist[0] + wrist_direction[0] * 0.16,
        wrist[1] + wrist_direction[1] * 0.16,
        wrist[2] + wrist_direction[2] * 0.16,
    )
    return {
        "shoulder": shoulder,
        "elbow": elbow,
        "wrist": wrist,
        "palm": palm,
    }


def inverse_kinematics(target: tuple[float, float, float] | list[float], params: Any) -> CatchPose:
    """Return a simple 3-axis capture pose for DUM-E.

    This is intentionally compact: it is not a physics motor controller yet.
    It gives the learning system a stable latent action target for phase 2.
    """

    base_x = robot_base_x(params)
    dx = float(target[0]) - base_x
    dy = float(target[1]) - TARGET_Y
    dz = float(target[2]) - ROBOT_SHOULDER_Z
    horizontal = math.hypot(dx, dy)
    distance = math.hypot(horizontal, dz)
    max_reach = UPPER_ARM_LENGTH + FOREARM_LENGTH - 1e-6
    min_reach = abs(UPPER_ARM_LENGTH - FOREARM_LENGTH) + 1e-6
    clamped_distance = max(min_reach, min(max_reach, distance))

    base = math.atan2(dy, dx)
    target_pitch = math.atan2(dz, max(horizontal, 1e-6))
    shoulder_offset = math.acos(
        max(
            -1.0,
            min(
                1.0,
                (UPPER_ARM_LENGTH**2 + clamped_distance**2 - FOREARM_LENGTH**2)
                / (2.0 * UPPER_ARM_LENGTH * clamped_distance),
            ),
        )
    )
    shoulder_pitch = target_pitch + shoulder_offset
    elbow_internal = math.acos(
        max(
            -1.0,
            min(
                1.0,
                (UPPER_ARM_LENGTH**2 + FOREARM_LENGTH**2 - clamped_distance**2)
                / (2.0 * UPPER_ARM_LENGTH * FOREARM_LENGTH),
            ),
        )
    )
    elbow = math.pi - elbow_internal
    wrist = -(shoulder_pitch + elbow * 0.5)
    reach_error = max(0.0, distance - max_reach) + max(0.0, min_reach - distance)

    return CatchPose(
        catch_x=float(target[0]),
        catch_y=float(target[1]),
        catch_z=float(target[2]),
        axis_base=base,
        axis_elbow=elbow,
        axis_wrist=wrist,
        reachable=1.0 if reach_error <= 1e-6 else 0.0,
        reach_error=reach_error,
    )


def catch_event(rows: list[dict[str, float]], params: Any) -> tuple[int, float | None, float]:
    """Compute whether a trajectory intersects the DUM-E catch sphere."""

    center = catch_center(params)
    min_distance = float("inf")
    capture_time = None
    previous = None
    previous_time = None

    for row in rows:
        position = (float(row["ball_x"]), float(row["ball_y"]), float(row["ball_z"]))
        distance = math.dist(position, center)
        min_distance = min(min_distance, distance)
        if previous is not None and capture_time is None:
            alpha, _closest, segment_distance = closest_point_on_segment(previous, position, center)
            min_distance = min(min_distance, segment_distance)
            if segment_distance <= CATCH_RADIUS:
                capture_time = previous_time + (float(row["time"]) - previous_time) * alpha
        previous = position
        previous_time = float(row["time"])

    return (1 if capture_time is not None else 0, capture_time, min_distance)

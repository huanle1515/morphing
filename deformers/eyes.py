from __future__ import annotations

import numpy as np

from core.types import Landmarks
from core.masks import make_region_mask
from core.regions import LEFT_EYE_OUTER, RIGHT_EYE_OUTER, LEFT_BROW, RIGHT_BROW, unique


def scale_group(points: np.ndarray, indices: list[int], sx: float, sy: float | None = None) -> None:
    if sy is None:
        sy = sx
    center = np.mean(points[indices], axis=0)
    points[indices, 0] = center[0] + (points[indices, 0] - center[0]) * sx
    points[indices, 1] = center[1] + (points[indices, 1] - center[1]) * sy


def move_group(points: np.ndarray, indices: list[int], dx: float = 0.0, dy: float = 0.0) -> None:
    points[indices, 0] += dx
    points[indices, 1] += dy


def apply_eyes(moved: np.ndarray, lm: Landmarks, eye_size: float, eye_distance: float) -> list[np.ndarray]:
    masks: list[np.ndarray] = []

    # keep eye distance exactly like the earlier working version
    if abs(eye_size) > 1e-6:
        sx = 1.0 + 0.015 * eye_size
        sy = 1.0 + 0.022 * eye_size

        scale_group(moved, LEFT_EYE_OUTER, sx=sx, sy=sy)
        scale_group(moved, RIGHT_EYE_OUTER, sx=sx, sy=sy)

        masks.append(make_region_mask(
            lm,
            unique(LEFT_EYE_OUTER + RIGHT_EYE_OUTER + LEFT_BROW + RIGHT_BROW),
            pad=20,
            blur=39,
        ))

    if abs(eye_distance) > 1e-6:
        move_group(moved, LEFT_EYE_OUTER, dx=-0.6 * eye_distance)
        move_group(moved, RIGHT_EYE_OUTER, dx=0.6 * eye_distance)
        move_group(moved, LEFT_BROW, dx=-0.35 * eye_distance)
        move_group(moved, RIGHT_BROW, dx=0.35 * eye_distance)

        masks.append(make_region_mask(
            lm,
            unique(LEFT_EYE_OUTER + RIGHT_EYE_OUTER + LEFT_BROW + RIGHT_BROW),
            pad=22,
            blur=41,
        ))

    return masks
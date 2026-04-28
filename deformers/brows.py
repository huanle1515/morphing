from __future__ import annotations

import numpy as np

from core.types import Landmarks
from core.masks import make_region_mask
from core.parsing import LEFT_BROW as PARSE_LEFT_BROW
from core.parsing import RIGHT_BROW as PARSE_RIGHT_BROW
from core.parsing import LEFT_EYE as PARSE_LEFT_EYE
from core.parsing import RIGHT_EYE as PARSE_RIGHT_EYE
from core.parsing import ParsingResult, semantic_mask
from core.regions import (
    LEFT_BROW,
    RIGHT_BROW,
    LEFT_BROW_INNER,
    LEFT_BROW_OUTER,
    RIGHT_BROW_INNER,
    RIGHT_BROW_OUTER,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    FOREHEAD,
    unique,
)


def move_group(points: np.ndarray, indices: list[int], dx: float = 0.0, dy: float = 0.0) -> None:
    points[indices, 0] += dx
    points[indices, 1] += dy


def _soft_cap(x: float, cap: float = 6.0) -> float:
    """
    Keep UI range if needed, but cap effective brow motion.
    """
    return float(np.clip(x, -cap, cap))


def apply_brows(
    moved: np.ndarray,
    lm: Landmarks,
    eyebrow_height: float,
    parsing: ParsingResult | None = None,
) -> list[np.ndarray]:
    masks: list[np.ndarray] = []

    if abs(eyebrow_height) < 1e-6:
        return masks

    e = _soft_cap(eyebrow_height, cap=6.0)

    # reduced from previous stronger version
    dy = -0.95 * e
    inner_dy = -0.22 * e
    outer_dy = 0.12 * dy
    eye_support = 0.05 * dy

    move_group(moved, LEFT_BROW, dy=dy)
    move_group(moved, RIGHT_BROW, dy=dy)

    move_group(moved, LEFT_BROW_INNER, dy=inner_dy)
    move_group(moved, RIGHT_BROW_INNER, dy=inner_dy)
    move_group(moved, LEFT_BROW_OUTER, dy=outer_dy)
    move_group(moved, RIGHT_BROW_OUTER, dy=outer_dy)

    move_group(moved, LEFT_EYE_OUTER, dy=eye_support)
    move_group(moved, RIGHT_EYE_OUTER, dy=eye_support)

    sem = semantic_mask(
        parsing,
        [PARSE_LEFT_BROW, PARSE_RIGHT_BROW, PARSE_LEFT_EYE, PARSE_RIGHT_EYE],
        lm.h,
        lm.w,
        blur=23,
        dilate=10,
    )
    if sem is not None:
        masks.append(sem)
    else:
        masks.append(make_region_mask(
            lm,
            unique(
                LEFT_BROW + RIGHT_BROW +
                LEFT_BROW_INNER + LEFT_BROW_OUTER +
                RIGHT_BROW_INNER + RIGHT_BROW_OUTER +
                LEFT_EYE_OUTER + RIGHT_EYE_OUTER +
                FOREHEAD
            ),
            pad=28,
            blur=43,
        ))

    return masks
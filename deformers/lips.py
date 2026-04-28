from __future__ import annotations

import numpy as np

from core.types import Landmarks
from core.masks import make_region_mask
from core.parsing import (
    MOUTH as PARSE_MOUTH,
    UPPER_LIP as PARSE_UPPER_LIP,
    LOWER_LIP as PARSE_LOWER_LIP,
    ParsingResult,
    semantic_mask,
)
from core.regions import (
    MOUTH_OUTER,
    UPPER_LIP,
    LOWER_LIP,
    MOUTH_CORNERS,
    UPPER_LIP_CENTER,
    LOWER_LIP_CENTER,
    CUPID_BOW,
    UPPER_LIP_BULGE,
    LOWER_LIP_BULGE,
    unique,
)


def scale_group(points: np.ndarray, indices: list[int], sx: float, sy: float | None = None) -> None:
    if sy is None:
        sy = sx
    center = np.mean(points[indices], axis=0)
    points[indices, 0] = center[0] + (points[indices, 0] - center[0]) * sx
    points[indices, 1] = center[1] + (points[indices, 1] - center[1]) * sy


def move_group(points: np.ndarray, indices: list[int], dx: float = 0.0, dy: float = 0.0) -> None:
    points[indices, 0] += dx
    points[indices, 1] += dy


def _lips_mask(lm: Landmarks, parsing: ParsingResult | None) -> np.ndarray:
    sem = semantic_mask(
        parsing,
        [PARSE_MOUTH, PARSE_UPPER_LIP, PARSE_LOWER_LIP],
        lm.h,
        lm.w,
        blur=17,
        dilate=5,
    )
    if sem is not None:
        return sem

    return make_region_mask(
        lm,
        unique(MOUTH_OUTER + UPPER_LIP + LOWER_LIP + MOUTH_CORNERS),
        pad=24,
        blur=43,
    )


def apply_lips(
    moved: np.ndarray,
    lm: Landmarks,
    mouth_width: float,
    lip_size: float,
    lip_width: float,
    lip_height: float,
    lip_peak: float,
    parsing: ParsingResult | None = None,
) -> list[np.ndarray]:
    masks: list[np.ndarray] = []

    if abs(mouth_width) > 1e-6:
        mouth_center_x = float(np.mean(lm.px[MOUTH_OUTER, 0]))
        scale = 1.0 + 0.018 * mouth_width

        moved[MOUTH_OUTER, 0] = mouth_center_x + (moved[MOUTH_OUTER, 0] - mouth_center_x) * scale
        moved[UPPER_LIP, 0] = mouth_center_x + (moved[UPPER_LIP, 0] - mouth_center_x) * scale
        moved[LOWER_LIP, 0] = mouth_center_x + (moved[LOWER_LIP, 0] - mouth_center_x) * scale

        moved[MOUTH_CORNERS[0], 0] -= 0.8 * mouth_width
        moved[MOUTH_CORNERS[1], 0] += 0.8 * mouth_width

        masks.append(_lips_mask(lm, parsing))

    if abs(lip_size) > 1e-6:
        sx = 1.0 + 0.0055 * lip_size
        sy = 1.0 + 0.013 * lip_size

        scale_group(moved, MOUTH_OUTER, sx=sx, sy=sy)
        scale_group(moved, UPPER_LIP, sx=sx, sy=sy)
        scale_group(moved, LOWER_LIP, sx=sx, sy=sy)

        masks.append(_lips_mask(lm, parsing))

    if abs(lip_width) > 1e-6:
        mouth_center_x = float(np.mean(moved[MOUTH_OUTER, 0]))
        scale = 1.0 + 0.009 * lip_width

        moved[MOUTH_OUTER, 0] = mouth_center_x + (moved[MOUTH_OUTER, 0] - mouth_center_x) * scale
        moved[UPPER_LIP, 0] = mouth_center_x + (moved[UPPER_LIP, 0] - mouth_center_x) * scale
        moved[LOWER_LIP, 0] = mouth_center_x + (moved[LOWER_LIP, 0] - mouth_center_x) * scale
        moved[MOUTH_CORNERS[0], 0] -= 0.28 * lip_width
        moved[MOUTH_CORNERS[1], 0] += 0.28 * lip_width

        masks.append(_lips_mask(lm, parsing))

    if abs(lip_height) > 1e-6:
        move_group(moved, UPPER_LIP_BULGE, dy=-0.24 * lip_height)
        move_group(moved, LOWER_LIP_BULGE, dy=0.24 * lip_height)
        move_group(moved, UPPER_LIP_CENTER, dy=-0.30 * lip_height)
        move_group(moved, LOWER_LIP_CENTER, dy=0.30 * lip_height)

        masks.append(_lips_mask(lm, parsing))

    if abs(lip_peak) > 1e-6:
        move_group(moved, CUPID_BOW, dy=-0.22 * lip_peak)
        move_group(moved, [39], dx=-0.06 * lip_peak)
        move_group(moved, [269], dx=0.06 * lip_peak)

        masks.append(_lips_mask(lm, parsing))

    return masks
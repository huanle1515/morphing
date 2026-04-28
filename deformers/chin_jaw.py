from __future__ import annotations

import numpy as np

from core.types import Landmarks
from core.masks import make_lower_face_mask, make_full_face_mask
from core.regions import (
    CHIN_CORE,
    CHIN_ARC,
    JAW_LEFT,
    JAW_RIGHT,
    CHEEK_LEFT,
    CHEEK_RIGHT,
    FACE_OVAL,
    MOUTH_OUTER,
    MOUTH_CORNERS,
    unique,
)


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _face_center_x(lm: Landmarks) -> float:
    return float(np.mean(lm.px[FACE_OVAL, 0]))


def _mouth_y(lm: Landmarks) -> float:
    return float(np.mean(lm.px[MOUTH_OUTER, 1]))


def _chin_y(lm: Landmarks) -> float:
    return float(np.mean(lm.px[CHIN_CORE, 1]))


def _lower_face_progress(y: float, mouth_y: float, chin_y: float) -> float:
    denom = max(chin_y - mouth_y, 1.0)
    t = (y - mouth_y) / denom
    return _clip01(t)


def _apply_weighted_vertical_shift(
    moved: np.ndarray,
    lm: Landmarks,
    indices: list[int],
    amount: float,
    power: float = 1.6,
) -> None:
    mouth_y = _mouth_y(lm)
    chin_y = _chin_y(lm)

    for idx in indices:
        y = float(lm.px[idx, 1])
        t = _lower_face_progress(y, mouth_y, chin_y)
        w = t ** power
        moved[idx, 1] += amount * w


def _apply_weighted_horizontal_scale(
    moved: np.ndarray,
    lm: Landmarks,
    indices: list[int],
    scale_amount: float,
    center_x: float,
    power: float = 1.2,
) -> None:
    mouth_y = _mouth_y(lm)
    chin_y = _chin_y(lm)

    for idx in indices:
        x = float(moved[idx, 0])
        y = float(lm.px[idx, 1])
        t = _lower_face_progress(y, mouth_y, chin_y)
        w = t ** power
        s = 1.0 + scale_amount * w
        moved[idx, 0] = center_x + (x - center_x) * s


def _apply_smooth_jaw_contour(
    moved: np.ndarray,
    center_x: float,
    jaw_width: float,
) -> None:
    """
    Broader continuous side-band from lower cheek into jaw.
    This reduces the visible kink.
    """
    left_chain = CHEEK_LEFT[-4:] + JAW_LEFT
    right_chain = CHEEK_RIGHT[-4:] + JAW_RIGHT

    n_left = max(len(left_chain) - 1, 1)
    n_right = max(len(right_chain) - 1, 1)

    for i, idx in enumerate(left_chain):
        t = i / n_left
        # smoother and less abrupt weighting
        w = 0.24 + 0.66 * (t ** 1.0)
        moved[idx, 0] = center_x + (moved[idx, 0] - center_x) * (1.0 + 0.0095 * jaw_width * w)

    for i, idx in enumerate(right_chain):
        t = i / n_right
        w = 0.24 + 0.66 * (t ** 1.0)
        moved[idx, 0] = center_x + (moved[idx, 0] - center_x) * (1.0 + 0.0095 * jaw_width * w)


def _apply_face_width_profile(
    moved: np.ndarray,
    lm: Landmarks,
    face_width: float,
) -> None:
    center_x = _face_center_x(lm)

    cheek_scale = 0.010 * face_width
    jaw_scale = 0.006 * face_width
    chin_arc_scale = 0.002 * face_width

    for idx in CHEEK_LEFT + CHEEK_RIGHT:
        moved[idx, 0] = center_x + (moved[idx, 0] - center_x) * (1.0 + cheek_scale)

    for idx in JAW_LEFT + JAW_RIGHT:
        moved[idx, 0] = center_x + (moved[idx, 0] - center_x) * (1.0 + jaw_scale)

    for idx in CHIN_ARC:
        moved[idx, 0] = center_x + (moved[idx, 0] - center_x) * (1.0 + chin_arc_scale)


def apply_chin_jaw(
    moved: np.ndarray,
    lm: Landmarks,
    chin_length: float,
    jaw_width: float,
    face_width: float,
) -> list[np.ndarray]:
    masks: list[np.ndarray] = []

    # keep current good chin behavior
    if abs(chin_length) > 1e-6:
        center_x = _face_center_x(lm)

        _apply_weighted_vertical_shift(
            moved=moved,
            lm=lm,
            indices=unique(CHIN_CORE + CHIN_ARC + JAW_LEFT + JAW_RIGHT),
            amount=1.55 * chin_length,
            power=1.65,
        )

        for idx in MOUTH_CORNERS:
            moved[idx, 1] += 0.05 * chin_length

        _apply_weighted_horizontal_scale(
            moved=moved,
            lm=lm,
            indices=unique(CHIN_ARC + JAW_LEFT + JAW_RIGHT),
            scale_amount=0.0065 * chin_length,
            center_x=center_x,
            power=1.2,
        )

        for idx in CHIN_CORE:
            moved[idx, 1] += 0.55 * chin_length

        masks.append(make_lower_face_mask(lm, pad=28, blur=51))

    # smoother jaw contour
    if abs(jaw_width) > 1e-6:
        center_x = _face_center_x(lm)

        _apply_smooth_jaw_contour(
            moved=moved,
            center_x=center_x,
            jaw_width=jaw_width,
        )

        # lighter chin-arc coupling
        for idx in CHIN_ARC:
            moved[idx, 0] = center_x + (moved[idx, 0] - center_x) * (1.0 + 0.0022 * jaw_width)

        # reduce vertical kink support
        for idx in JAW_LEFT[-2:] + JAW_RIGHT[-2:]:
            moved[idx, 1] += 0.02 * jaw_width

        masks.append(make_lower_face_mask(lm, pad=26, blur=49))

    if abs(face_width) > 1e-6:
        _apply_face_width_profile(
            moved=moved,
            lm=lm,
            face_width=face_width,
        )
        masks.append(make_full_face_mask(lm, pad=16, blur=41))

    return masks
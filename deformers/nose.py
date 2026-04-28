from __future__ import annotations

import numpy as np

from core.types import Landmarks
from core.masks import make_region_mask
from core.parsing import NOSE as PARSE_NOSE
from core.parsing import ParsingResult, semantic_mask
from core.regions import NOSE_LEFT, NOSE_RIGHT, NOSE_TIP, NOSE_BRIDGE, unique


def apply_nose(
    moved: np.ndarray,
    lm: Landmarks,
    nose_width: float,
    parsing: ParsingResult | None = None,
) -> list[np.ndarray]:
    masks: list[np.ndarray] = []

    if abs(nose_width) < 1e-6:
        return masks

    nose_group = unique(NOSE_LEFT + NOSE_RIGHT + NOSE_TIP + NOSE_BRIDGE)
    nose_center_x = float(np.mean(lm.px[nose_group, 0]))
    scale = 1.0 + 0.018 * nose_width

    moved[NOSE_LEFT, 0] = nose_center_x + (moved[NOSE_LEFT, 0] - nose_center_x) * scale
    moved[NOSE_RIGHT, 0] = nose_center_x + (moved[NOSE_RIGHT, 0] - nose_center_x) * scale
    moved[NOSE_TIP, 0] = nose_center_x + (moved[NOSE_TIP, 0] - nose_center_x) * (1.0 + 0.005 * nose_width)

    sem = semantic_mask(
        parsing,
        [PARSE_NOSE],
        lm.h,
        lm.w,
        blur=17,
        dilate=5,
    )
    if sem is not None:
        masks.append(sem)
    else:
        masks.append(make_region_mask(lm, nose_group, pad=22, blur=41))

    return masks
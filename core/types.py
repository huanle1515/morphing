from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Landmarks:
    px: np.ndarray
    xyz: np.ndarray
    w: int
    h: int
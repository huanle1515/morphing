from __future__ import annotations

import cv2
import numpy as np

from core.types import Landmarks
from core.regions import FACE_OVAL, CHIN_CORE


def make_region_mask(lm: Landmarks, indices: list[int], pad: int = 22, blur: int = 41) -> np.ndarray:
    mask = np.zeros((lm.h, lm.w), dtype=np.uint8)
    pts = np.round(lm.px[indices]).astype(np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    if pad > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    if blur % 2 == 0:
        blur += 1

    alpha = cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32) / 255.0
    return np.clip(alpha, 0.0, 1.0)


def make_lower_face_mask(lm: Landmarks, pad: int = 26, blur: int = 51) -> np.ndarray:
    mask = np.zeros((lm.h, lm.w), dtype=np.uint8)
    oval = np.round(lm.px[FACE_OVAL]).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(oval), 255)

    chin_y = float(np.mean(lm.px[CHIN_CORE, 1]))
    cutoff = int(round(chin_y - 0.26 * lm.h))
    mask[:max(cutoff, 0), :] = 0

    if pad > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    if blur % 2 == 0:
        blur += 1

    alpha = cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32) / 255.0
    return np.clip(alpha, 0.0, 1.0)


def make_full_face_mask(lm: Landmarks, pad: int = 18, blur: int = 41) -> np.ndarray:
    mask = np.zeros((lm.h, lm.w), dtype=np.uint8)
    oval = np.round(lm.px[FACE_OVAL]).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(oval), 255)

    if pad > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    if blur % 2 == 0:
        blur += 1

    alpha = cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32) / 255.0
    return np.clip(alpha, 0.0, 1.0)


def combine_masks(masks: list[np.ndarray], h: int, w: int) -> np.ndarray:
    if not masks:
        return np.zeros((h, w), dtype=np.float32)

    out = np.zeros((h, w), dtype=np.float32)
    for m in masks:
        out = np.maximum(out, m)
    return np.clip(out, 0.0, 1.0)
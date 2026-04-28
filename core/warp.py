from __future__ import annotations

import cv2
import numpy as np
from skimage.transform import ThinPlateSplineTransform, warp

from core.types import Landmarks
from core.regions import (
    FACE_OVAL,
    LEFT_BROW,
    RIGHT_BROW,
    LEFT_BROW_INNER,
    LEFT_BROW_OUTER,
    RIGHT_BROW_INNER,
    RIGHT_BROW_OUTER,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    NOSE_LEFT,
    NOSE_RIGHT,
    NOSE_TIP,
    NOSE_BRIDGE,
    MOUTH_OUTER,
    UPPER_LIP,
    LOWER_LIP,
    MOUTH_CORNERS,
    UPPER_LIP_CENTER,
    LOWER_LIP_CENTER,
    CUPID_BOW,
    CHIN_CORE,
    CHIN_ARC,
    JAW_LEFT,
    JAW_RIGHT,
    CHEEK_LEFT,
    CHEEK_RIGHT,
    STABLE_ALL,
    unique,
)


def build_border_anchors(w: int, h: int, margin: int = 6) -> np.ndarray:
    return np.array([
        [margin, margin],
        [w * 0.25, margin],
        [w * 0.50, margin],
        [w * 0.75, margin],
        [w - 1 - margin, margin],
        [w - 1 - margin, h * 0.25],
        [w - 1 - margin, h * 0.50],
        [w - 1 - margin, h * 0.75],
        [w - 1 - margin, h - 1 - margin],
        [w * 0.75, h - 1 - margin],
        [w * 0.50, h - 1 - margin],
        [w * 0.25, h - 1 - margin],
        [margin, h - 1 - margin],
        [margin, h * 0.75],
        [margin, h * 0.50],
        [margin, h * 0.25],
    ], dtype=np.float32)


def build_outer_background_anchors(w: int, h: int, face_pts: np.ndarray, spacing: int = 42) -> np.ndarray:
    spacing = max(24, int(spacing))
    hull = cv2.convexHull(np.round(face_pts[FACE_OVAL]).astype(np.float32))

    yy, xx = np.mgrid[0:h:spacing, 0:w:spacing]
    grid = np.vstack([xx.ravel(), yy.ravel()]).T.astype(np.float32)

    anchors = []
    for pt in grid:
        dist = cv2.pointPolygonTest(hull, (float(pt[0]), float(pt[1])), True)
        if dist < -18:
            anchors.append(pt)

    if not anchors:
        return np.empty((0, 2), dtype=np.float32)

    return np.asarray(anchors, dtype=np.float32)


def build_control_points(lm: Landmarks, moved_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    control_face = unique(
        FACE_OVAL +
        LEFT_BROW + RIGHT_BROW +
        LEFT_BROW_INNER + LEFT_BROW_OUTER + RIGHT_BROW_INNER + RIGHT_BROW_OUTER +
        LEFT_EYE_OUTER + RIGHT_EYE_OUTER +
        NOSE_LEFT + NOSE_RIGHT + NOSE_TIP + NOSE_BRIDGE +
        MOUTH_OUTER + UPPER_LIP + LOWER_LIP + MOUTH_CORNERS +
        UPPER_LIP_CENTER + LOWER_LIP_CENTER + CUPID_BOW +
        CHIN_CORE + CHIN_ARC + JAW_LEFT + JAW_RIGHT +
        CHEEK_LEFT + CHEEK_RIGHT +
        STABLE_ALL
    )

    src_face = lm.px[control_face]
    dst_face = moved_px[control_face]

    border = build_border_anchors(lm.w, lm.h, margin=6)
    outer_bg = build_outer_background_anchors(lm.w, lm.h, lm.px, spacing=max(lm.w, lm.h) // 18)

    src = np.vstack([src_face, border, outer_bg]).astype(np.float32)
    dst = np.vstack([dst_face, border, outer_bg]).astype(np.float32)
    return src, dst


def tps_warp(image_rgb: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    if src_pts.shape != dst_pts.shape:
        raise ValueError("src_pts and dst_pts must have the same shape.")
    if src_pts.shape[0] < 3:
        raise ValueError("Need at least 3 control points.")

    h, w = image_rgb.shape[:2]
    tps = ThinPlateSplineTransform()
    ok = tps.estimate(dst_pts, src_pts)
    if not ok:
        raise RuntimeError("ThinPlateSplineTransform estimation failed.")

    warped = warp(
        image_rgb.astype(np.float32),
        inverse_map=tps,
        output_shape=(h, w),
        preserve_range=True,
        mode="edge",
    )
    return np.clip(warped, 0, 255).astype(np.uint8)


def alpha_compose(original_rgb: np.ndarray, warped_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None].astype(np.float32)
    out = warped_rgb.astype(np.float32) * a + original_rgb.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_points(image_rgb: np.ndarray, points: np.ndarray, color=(0, 255, 0), radius: int = 1) -> np.ndarray:
    out = image_rgb.copy()
    for x, y in points:
        cv2.circle(out, (int(round(x)), int(round(y))), radius, color, -1)
    return out


def draw_controls(image_rgb: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, max_lines: int = 250) -> np.ndarray:
    out = image_rgb.copy()
    n = min(len(src_pts), max_lines)
    for s, d in zip(src_pts[:n], dst_pts[:n]):
        sx, sy = int(round(s[0])), int(round(s[1]))
        dx, dy = int(round(d[0])), int(round(d[1]))
        cv2.circle(out, (sx, sy), 1, (255, 0, 0), -1)
        cv2.circle(out, (dx, dy), 1, (0, 255, 255), -1)
        cv2.line(out, (sx, sy), (dx, dy), (255, 0, 255), 1)
    return out
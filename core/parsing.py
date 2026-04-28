from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import streamlit as st

from core.regions import FACE_OVAL
from core.types import Landmarks


# CelebAMask-HQ / BiSeNet-style face parsing labels
BACKGROUND = 0
SKIN = 1
LEFT_BROW = 2
RIGHT_BROW = 3
LEFT_EYE = 4
RIGHT_EYE = 5
EYE_GLASSES = 6
LEFT_EAR = 7
RIGHT_EAR = 8
EAR_RING = 9
NOSE = 10
MOUTH = 11
UPPER_LIP = 12
LOWER_LIP = 13
NECK = 14
NECKLACE = 15
CLOTH = 16
HAIR = 17
HAT = 18


@dataclass
class ParsingResult:
    labels_full: np.ndarray  # (H, W) int32
    crop_box: tuple[int, int, int, int]  # x1, y1, x2, y2


def _bbox_from_face(lm: Landmarks, margin_ratio: float = 0.18) -> tuple[int, int, int, int]:
    pts = lm.px[FACE_OVAL]
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))

    w = x2 - x1
    h = y2 - y1
    mx = margin_ratio * w
    my = margin_ratio * h

    x1 = int(max(0, np.floor(x1 - mx)))
    y1 = int(max(0, np.floor(y1 - my)))
    x2 = int(min(lm.w, np.ceil(x2 + mx)))
    y2 = int(min(lm.h, np.ceil(y2 + my)))

    return x1, y1, x2, y2


def _extract_array(output: Any) -> np.ndarray:
    """
    Try to normalize various parser outputs into a label/logit array.
    """
    if output is None:
        raise RuntimeError("Parser returned None.")

    if isinstance(output, dict):
        for key in ("mask", "masks", "segmentation", "seg", "labels", "parsing", "pred"):
            if key in output:
                return np.asarray(output[key])

    if isinstance(output, (list, tuple)):
        if len(output) == 1:
            return _extract_array(output[0])

        # Try first ndarray-looking item
        for item in output:
            if hasattr(item, "shape") or isinstance(item, (list, tuple, dict)):
                try:
                    return _extract_array(item)
                except Exception:
                    pass

    for attr in ("mask", "masks", "segmentation", "seg", "labels", "parsing", "pred"):
        if hasattr(output, attr):
            return np.asarray(getattr(output, attr))

    return np.asarray(output)


def _to_label_map(arr: np.ndarray) -> np.ndarray:
    """
    Convert logits/probabilities/label-map into (H, W) int32 labels.
    """
    arr = np.asarray(arr)

    if arr.ndim == 2:
        return arr.astype(np.int32)

    if arr.ndim == 3:
        # C,H,W
        if arr.shape[0] <= 32:
            return np.argmax(arr, axis=0).astype(np.int32)

        # H,W,C
        if arr.shape[-1] <= 32:
            return np.argmax(arr, axis=-1).astype(np.int32)

        # 1,H,W
        if arr.shape[0] == 1:
            return arr[0].astype(np.int32)

    if arr.ndim == 4:
        arr = arr[0]
        return _to_label_map(arr)

    raise RuntimeError(f"Unsupported parser output shape: {arr.shape}")


@st.cache_resource
def load_face_parser():
    """
    Load UniFace BiSeNet parser.
    """
    try:
        from uniface import BiSeNet  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Could not import UniFace BiSeNet parser. "
            "Make sure `uniface==3.3.0` is installed."
        ) from e

    try:
        parser = BiSeNet()
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize UniFace BiSeNet parser."
        ) from e

    return parser


def run_face_parsing(image_rgb: np.ndarray, lm: Landmarks, parser: Any) -> ParsingResult:
    """
    Parse only the face crop, then place labels back into full-image coordinates.
    """
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = _bbox_from_face(lm, margin_ratio=0.18)
    crop_rgb = image_rgb[y1:y2, x1:x2]

    if crop_rgb.size == 0:
        raise RuntimeError("Empty face crop for parsing.")

    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

    # Try several likely parser entrypoints
    raw = None
    errors = []

    calls = [
        lambda p, img: p.predict(img),
        lambda p, img: p.parse(img),
        lambda p, img: p.segment(img),
        lambda p, img: p(img),
    ]

    for fn in calls:
        try:
            raw = fn(parser, crop_bgr)
            break
        except Exception as e:
            errors.append(str(e))

    if raw is None:
        raise RuntimeError(
            "Could not run parser. Tried predict/parse/segment/call.\n"
            + "\n".join(errors[:4])
        )

    labels_crop = _to_label_map(_extract_array(raw))

    if labels_crop.shape[:2] != crop_rgb.shape[:2]:
        labels_crop = cv2.resize(
            labels_crop.astype(np.uint8),
            (crop_rgb.shape[1], crop_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)

    labels_full = np.zeros((h, w), dtype=np.int32)
    labels_full[y1:y2, x1:x2] = labels_crop

    return ParsingResult(labels_full=labels_full, crop_box=(x1, y1, x2, y2))


def semantic_mask(
    parsing: ParsingResult | None,
    class_ids: list[int],
    h: int,
    w: int,
    blur: int = 21,
    dilate: int = 5,
) -> np.ndarray | None:
    if parsing is None:
        return None

    labels = parsing.labels_full
    mask = np.isin(labels, class_ids).astype(np.uint8) * 255

    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        mask = cv2.dilate(mask, k, iterations=1)

    if blur % 2 == 0:
        blur += 1

    mask = cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32) / 255.0
    return np.clip(mask, 0.0, 1.0)
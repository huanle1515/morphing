from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np

from core.types import Landmarks

MODEL_PATH = Path("models/face_landmarker.task")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def load_landmarker():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}\n"
            f"Put face_landmarker.task inside the models folder."
        )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)


def detect_landmarks(image_rgb: np.ndarray, landmarker) -> Landmarks | None:
    h, w = image_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    face = result.face_landmarks[0]
    xyz = np.array([[lm.x, lm.y, lm.z] for lm in face], dtype=np.float32)
    px = np.empty((xyz.shape[0], 2), dtype=np.float32)
    px[:, 0] = xyz[:, 0] * w
    px[:, 1] = xyz[:, 1] * h

    return Landmarks(px=px, xyz=xyz, w=w, h=h)
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
import traceback

from core.landmarks import load_landmarker, detect_landmarks
from core.masks import combine_masks
from core.parsing import load_face_parser, run_face_parsing
from core.warp import build_control_points, tps_warp, alpha_compose
from deformers.brows import apply_brows
from deformers.lips import apply_lips
from deformers.nose import apply_nose
from deformers.chin_jaw import apply_chin_jaw
from deformers.eyes import apply_eyes

app = Flask(__name__)

# Load models once when the server starts
print("Loading AI Models...")
landmarker = load_landmarker()
try:
    parser = load_face_parser()
except Exception as e:
    print(f"Warning: Could not load UniFace parser: {e}")
    parser = None
print("Models loaded successfully!")

def decode_image(file_bytes: bytes, max_w: int = 900) -> np.ndarray:
    arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode image.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = image_rgb.shape[:2]
    if w0 > max_w:
        scale = max_w / float(w0)
        image_rgb = cv2.resize(image_rgb, (max_w, int(round(h0 * scale))), interpolation=cv2.INTER_AREA)
    return image_rgb

def encode_image(image_rgb: np.ndarray) -> str:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', image_bgr)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

def clamp(val, min_val, max_val):
    """Forces the incoming API data to strictly obey your slider ranges"""
    try:
        v = float(val)
        return max(min_val, min(max_val, v))
    except (TypeError, ValueError):
        return 0.0

@app.route('/api/v1/reshape', methods=['POST'])
def reshape_face():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    # 1. Grab sliders and strictly clamp them to your limits
    params = {
        "eyebrow_height": clamp(request.form.get("eyebrow_height"), -10.0, 10.0),
        "chin_length": clamp(request.form.get("chin_length"), -10.0, 10.0),
        "mouth_width": clamp(request.form.get("mouth_width"), -3.0, 3.0),
        "nose_width": clamp(request.form.get("nose_width"), -10.0, 10.0),
        "jaw_width": clamp(request.form.get("jaw_width"), -8.0, 8.0),
        "face_width": clamp(request.form.get("face_width"), -8.0, 8.0),
        "eye_size": clamp(request.form.get("eye_size"), -10.0, 10.0),
        "eye_distance": clamp(request.form.get("eye_distance"), -10.0, 10.0),
        "lip_size": clamp(request.form.get("lip_size"), -10.0, 10.0),
        "lip_width": clamp(request.form.get("lip_width"), -10.0, 10.0),
        "lip_height": clamp(request.form.get("lip_height"), -10.0, 10.0),
        "lip_peak": clamp(request.form.get("lip_peak"), -10.0, 10.0),
    }

    try:
        image_rgb = decode_image(file.read())
        lm = detect_landmarks(image_rgb, landmarker)

        if lm is None:
            return jsonify({"error": "No face detected."}), 400

        parsing = None
        if parser is not None:
            try:
                parsing = run_face_parsing(image_rgb, lm, parser)
            except Exception as e:
                print(f"Parsing error: {e}")

        moved = lm.px.copy()
        masks = []

        # 2. Apply deformers
        masks += apply_brows(moved, lm, params["eyebrow_height"], parsing)
        masks += apply_lips(moved, lm, params["mouth_width"], params["lip_size"], params["lip_width"], params["lip_height"], params["lip_peak"], parsing)
        masks += apply_nose(moved, lm, params["nose_width"], parsing)
        masks += apply_chin_jaw(moved, lm, params["chin_length"], params["jaw_width"], params["face_width"])
        masks += apply_eyes(moved, lm, params["eye_size"], params["eye_distance"])

        # 3. Clip points to bounds
        moved[:, 0] = np.clip(moved[:, 0], 0, lm.w - 1)
        moved[:, 1] = np.clip(moved[:, 1], 0, lm.h - 1)
        
        alpha = combine_masks(masks, lm.h, lm.w)

        # 4. Build results
        if np.max(alpha) < 1e-6:
            final_rgb = image_rgb.copy()
        else:
            src_pts, dst_pts = build_control_points(lm, moved)
            warped_rgb = tps_warp(image_rgb, src_pts, dst_pts)
            final_rgb = alpha_compose(image_rgb, warped_rgb, alpha)

        # 5. Return the final warped image
        return jsonify({
            "status": "success",
            "image": encode_image(final_rgb)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
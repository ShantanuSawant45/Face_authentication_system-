"""Lightweight helper to run Silent-Face-Anti-Spoofing model and return pass/fail.

This wrapper follows the same prediction aggregation logic used in
`Silent-Face-Anti-Spoofing/test.py` and returns (is_live: bool, score: float).

If the anti-spoofing model directory is missing, the helper logs a warning and
returns (True, 0.0) to avoid blocking registration in environments where the
anti-spoof models are not installed. Adjust this behavior if you prefer fail-closed.
"""
import os
import numpy as np

try:
    from src.anti_spoof_predict import AntiSpoofPredict
    from src.generate_patches import CropImage
    from src.utility import parse_model_name
except Exception:
    # If the silent anti-spoofing package isn't on PYTHONPATH, import will fail.
    # We keep the helper functional but will emit warnings at runtime.
    AntiSpoofPredict = None
    CropImage = None
    parse_model_name = None


def is_live_face(frame, model_dir=None, device_id=0, allow_on_missing_models=True):
    """Return (is_live: bool, score: float).

    - frame: BGR image (as read by OpenCV)
    - model_dir: directory containing anti-spoof model files (default path below)
    - device_id: GPU id (0) or CPU if no CUDA
    - allow_on_missing_models: if True, treat missing model dir as PASS to avoid blocking
    """
    # Default model dir (project relative)
    if model_dir is None:
        model_dir = os.path.join(os.getcwd(), "Silent-Face-Anti-Spoofing", "resources", "anti_spoof_models")

    if AntiSpoofPredict is None or CropImage is None or parse_model_name is None:
        print("Warning: anti-spoof dependencies not importable. Skipping anti-spoof check.")
        return True, 0.0

    if not os.path.isdir(model_dir):
        msg = f"Anti-spoof model directory not found: {model_dir}"
        if allow_on_missing_models:
            print("Warning:", msg, "— allowing registration (adjust allow_on_missing_models to change).")
            return True, 0.0
        else:
            print("Error:", msg)
            return False, 0.0

    model = AntiSpoofPredict(device_id)
    cropper = CropImage()

    # Get bounding box for the face from the full frame
    bbox = model.get_bbox(frame)
    # If bbox indicates no detection, get_bbox may return zeros — treat as failure
    if bbox[0] == 0 and bbox[1] == 0:
        return False, 0.0

    prediction = np.zeros((1, 3))

    # Sum predictions from all files in model_dir that look like model weights
    for model_name in os.listdir(model_dir):
        # skip unrelated files
        if not (model_name.endswith('.pth') or model_name.endswith('.pt') or model_name.endswith('.pkl')):
            continue

        try:
            h_input, w_input, model_type, scale = parse_model_name(model_name)
        except Exception:
            # If parse fails, try a sensible fallback (common sizes)
            h_input, w_input, scale = 64, 64, 2.0

        param = {
            "org_img": frame,
            "bbox": bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False

        try:
            img = cropper.crop(**param)
            res = model.predict(img, os.path.join(model_dir, model_name))
            prediction += res
        except Exception as e:
            print(f"Warning: failed to run anti-spoof model {model_name}: {e}")
            continue

    label = int(np.argmax(prediction))
    score = float(prediction[0][label] / 2.0)
    is_live = (label == 1)
    return is_live, score

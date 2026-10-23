import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parent / "best.onnx"
CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45
INPUT_SIZE = 320

CLASS_NAMES = {
    0: "D00_Longitudinal_crack",
    1: "D10_Transverse_crack",
    2: "D20_Alligator_crack",
    3: "D40_Pothole",
}

_session = None


def get_model():
    global _session

    if _session is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {MODEL_PATH}"
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        _session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    return _session


def detect_damage(image_path):
    session = get_model()

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Unable to read the uploaded image.")

    original_h, original_w = image.shape[:2]

    scale = min(
        INPUT_SIZE / original_w,
        INPUT_SIZE / original_h,
    )
    new_w = max(1, int(round(original_w * scale)))
    new_h = max(1, int(round(original_h * scale)))

    resized = cv2.resize(image, (new_w, new_h))

    pad_w = INPUT_SIZE - new_w
    pad_h = INPUT_SIZE - new_h
    left = pad_w // 2
    top = pad_h // 2

    canvas = np.full(
        (INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8
    )
    canvas[top:top + new_h, left:left + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

    output = session.run(
        None,
        {session.get_inputs()[0].name: tensor},
    )[0]

    predictions = output[0].T

    boxes = []
    scores = []
    class_ids = []

    for row in predictions:
        cx, cy, w, h = row[:4]
        class_scores = row[4:8]

        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x1 = (cx - w / 2 - left) / scale
        y1 = (cy - h / 2 - top) / scale
        x2 = (cx + w / 2 - left) / scale
        y2 = (cy + h / 2 - top) / scale

        x1 = int(np.clip(x1, 0, original_w - 1))
        y1 = int(np.clip(y1, 0, original_h - 1))
        x2 = int(np.clip(x2, 0, original_w - 1))
        y2 = int(np.clip(y2, 0, original_h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(confidence)
        class_ids.append(class_id)

    kept = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONFIDENCE_THRESHOLD,
        IOU_THRESHOLD,
    )

    kept = (
        np.array(kept).flatten().tolist()
        if len(kept) > 0 else []
    )

    detections = []

    for i in kept[:50]:
        x, y, w, h = boxes[i]

        detections.append({
            "class_id": class_ids[i],
            "damage_type": CLASS_NAMES[class_ids[i]],
            "confidence": round(scores[i], 4),
            "bbox_xyxy": [x, y, x + w, y + h],
        })

    return {
        "image": Path(image_path).name,
        "detection_count": len(detections),
        "detections": detections,
    }

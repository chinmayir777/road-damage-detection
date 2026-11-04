from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


MODEL_PATH = Path(__file__).resolve().parent / "best.onnx"

INPUT_SIZE = 640

# Non-Maximum Suppression threshold
IOU_THRESHOLD = 0.35

CLASS_NAMES = {
    0: "D00_Longitudinal_crack",
    1: "D10_Transverse_crack",
    2: "D20_Alligator_crack",
    3: "D40_Pothole",
}

# Increased confidence thresholds to reduce false detections
CLASS_THRESHOLDS = {
    0: 0.20,
    1: 0.20,
    2: 0.20,
    3: 0.30,  # Pothole: increased from 0.05
}

MAX_DETECTIONS_PER_CLASS = 20

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

    # Resize while maintaining aspect ratio
    scale = min(
        INPUT_SIZE / original_w,
        INPUT_SIZE / original_h,
    )

    new_w = max(1, int(round(original_w * scale)))
    new_h = max(1, int(round(original_h * scale)))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    # Letterbox padding
    pad_w = INPUT_SIZE - new_w
    pad_h = INPUT_SIZE - new_h

    left = pad_w // 2
    top = pad_h // 2

    canvas = np.full(
        (INPUT_SIZE, INPUT_SIZE, 3),
        114,
        dtype=np.uint8,
    )

    canvas[
        top:top + new_h,
        left:left + new_w
    ] = resized

    # BGR to RGB
    rgb = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2RGB,
    )

    # Normalize and convert to NCHW
    tensor = rgb.astype(np.float32) / 255.0

    tensor = np.transpose(
        tensor,
        (2, 0, 1),
    )[None, ...]

    # ONNX inference
    output = session.run(
        None,
        {
            session.get_inputs()[0].name: tensor
        },
    )[0]

    predictions = output[0].T

    raw_scores = predictions[:, 4:8]

    # Debug information
    print(
        "[YOLO-DEBUG]",
        {
            "output_shape": list(output.shape),
            "prediction_shape": list(predictions.shape),
            "max_class_confidence": (
                round(float(np.max(raw_scores)), 4)
                if raw_scores.size else None
            ),
            "pothole_candidates_above_010": int(
                np.sum(raw_scores[:, 3] >= 0.10)
            ),
            "pothole_candidates_above_030": int(
                np.sum(raw_scores[:, 3] >= 0.30)
            ),
        },
        flush=True,
    )

    # Separate candidates by damage class
    class_candidates = {
        class_id: []
        for class_id in CLASS_NAMES
    }

    # Process model predictions
    for row in predictions:

        cx, cy, w, h = map(float, row[:4])

        if w <= 0 or h <= 0:
            continue

        # Process each class independently
        for class_id in CLASS_NAMES:

            confidence = float(row[4 + class_id])

            threshold = CLASS_THRESHOLDS[class_id]

            if confidence < threshold:
                continue

            # Convert center coordinates to original image coordinates
            x1 = (cx - w / 2 - left) / scale
            y1 = (cy - h / 2 - top) / scale

            x2 = (cx + w / 2 - left) / scale
            y2 = (cy + h / 2 - top) / scale

            # Clip coordinates
            x1 = int(np.clip(x1, 0, original_w - 1))
            y1 = int(np.clip(y1, 0, original_h - 1))

            x2 = int(np.clip(x2, 0, original_w - 1))
            y2 = int(np.clip(y2, 0, original_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            box = [
                x1,
                y1,
                x2 - x1,
                y2 - y1,
            ]

            class_candidates[class_id].append({
                "box": box,
                "confidence": confidence,
            })

    detections = []

    # Class-wise Non-Maximum Suppression
    for class_id, candidates in class_candidates.items():

        if not candidates:
            continue

        # Sort by confidence
        candidates.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )

        boxes = [
            item["box"]
            for item in candidates
        ]

        scores = [
            item["confidence"]
            for item in candidates
        ]

        threshold = CLASS_THRESHOLDS[class_id]

        # Suppress overlapping detections
        kept = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            threshold,
            IOU_THRESHOLD,
        )

        kept = (
            np.array(kept).flatten().tolist()
            if len(kept) > 0
            else []
        )

        # Limit detections per class
        kept = kept[:MAX_DETECTIONS_PER_CLASS]

        for index in kept:

            x, y, box_w, box_h = boxes[index]

            confidence = scores[index]

            damage_type = CLASS_NAMES[class_id]

            detections.append({
                "class_id": class_id,

                "damage_type": damage_type,

                "candidate_type": damage_type,

                "classification_status": (
                    "Needs human verification"
                    if confidence < 0.45
                    else "Model prediction"
                ),

                "confidence": round(
                    confidence,
                    4,
                ),

                "bbox_xyxy": [
                    x,
                    y,
                    x + box_w,
                    y + box_h,
                ],
            })

    # Sort by confidence
    detections.sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )

    print(
        "[YOLO-RESULT]",
        {
            "total_detections": len(detections),
            "potholes": sum(
                1 for d in detections
                if d["class_id"] == 3
            ),
            "cracks": sum(
                1 for d in detections
                if d["class_id"] != 3
            ),
        },
        flush=True,
    )

    return {
        "image": Path(image_path).name,
        "detection_count": len(detections),
        "detections": detections,
    }

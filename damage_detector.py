from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


MODEL_PATH = Path(__file__).resolve().parent / "best.onnx"

INPUT_SIZE = 640

# Non-Maximum Suppression thresholds
IOU_THRESHOLD = 0.35
CROSS_CLASS_IOU_THRESHOLD = 0.50

CLASS_NAMES = {
    0: "D00_Longitudinal_crack",
    1: "D10_Transverse_crack",
    2: "D20_Alligator_crack",
    3: "D40_Pothole",
}

# Retain conservative thresholds to reduce false detections
CLASS_THRESHOLDS = {
    0: 0.20,
    1: 0.20,
    2: 0.20,
    3: 0.30,
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


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def suppress_cross_class_duplicates(detections):
    """
    Remove highly overlapping predictions of the same physical region.
    Keep the highest-confidence prediction.
    """

    ordered = sorted(
        detections,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    kept = []

    for detection in ordered:
        box = detection["bbox_xyxy"]

        duplicate = False

        for existing in kept:
            existing_box = existing["bbox_xyxy"]

            overlap = calculate_iou(box, existing_box)

            if overlap >= CROSS_CLASS_IOU_THRESHOLD:
                duplicate = True
                break

        if not duplicate:
            kept.append(detection)

    return kept


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

    if predictions.shape[1] != 8:
        raise ValueError(
            f"Unexpected YOLO output shape: {predictions.shape}"
        )

    raw_scores = predictions[:, 4:8]

    print(
        "[YOLO-DEBUG]",
        {
            "output_shape": list(output.shape),
            "prediction_shape": list(predictions.shape),
            "max_class_confidence": (
                round(float(np.max(raw_scores)), 4)
                if raw_scores.size else None
            ),
            "pothole_candidates_above_030": int(
                np.sum(raw_scores[:, 3] >= 0.30)
            ),
        },
        flush=True,
    )

    class_candidates = {
        class_id: []
        for class_id in CLASS_NAMES
    }

    # Process predictions
    for row in predictions:

        cx, cy, w, h = map(float, row[:4])

        if w <= 0 or h <= 0:
            continue

        for class_id in CLASS_NAMES:

            confidence = float(row[4 + class_id])

            if confidence < CLASS_THRESHOLDS[class_id]:
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

    # Class-wise NMS
    for class_id, candidates in class_candidates.items():

        if not candidates:
            continue

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

        kept = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            CLASS_THRESHOLDS[class_id],
            IOU_THRESHOLD,
        )

        kept = (
            np.array(kept).flatten().tolist()
            if len(kept) > 0
            else []
        )

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
                "confidence": round(confidence, 4),
                "bbox_xyxy": [
                    x,
                    y,
                    x + box_w,
                    y + box_h,
                ],
            })

    # Remove highly overlapping predictions across classes
    before_suppression = len(detections)

    detections = suppress_cross_class_duplicates(detections)

    # Sort by confidence
    detections.sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )

    print(
        "[YOLO-RESULT]",
        {
            "before_duplicate_suppression": before_suppression,
            "after_duplicate_suppression": len(detections),
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

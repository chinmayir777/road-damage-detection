from pathlib import Path
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from ultralytics import YOLO

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

_model = None


def find_model():
    current = Path(__file__).resolve().parent

    for folder in (current, *current.parents):
        candidate = folder / "best.pt"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"best.pt not found. Searched from {current} upwards."
    )


MODEL_PATH = find_model()


def get_model():
    global _model

    if _model is None:
        _model = YOLO(str(MODEL_PATH))

    return _model


def intersection_over_smaller(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    return intersection / max(1.0, min(area_a, area_b))


def detect_damage(image_path, confidence_threshold=0.40):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not 0 <= confidence_threshold <= 1:
        raise ValueError(
            "Confidence threshold must be between 0 and 1."
        )

    model = get_model()

    results = model.predict(
        source=str(image_path),
        conf=confidence_threshold,
        iou=0.45,
        imgsz=640,
        max_det=50,
        device="cpu",
        verbose=False,
    )

    raw = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())

            raw.append({
                "class_id": class_id,
                "damage_type": model.names[class_id],
                "confidence": round(float(box.conf.item()), 4),
                "bbox_xyxy": [
                    round(float(v), 2)
                    for v in box.xyxy[0].tolist()
                ],
            })

    # Suppress highly overlapping detections of the same class.
    # Validate this heuristic against labelled images before relying on it.
    raw.sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )

    kept = []

    for item in raw:
        duplicate = any(
            item["class_id"] == prior["class_id"]
            and intersection_over_smaller(
                item["bbox_xyxy"],
                prior["bbox_xyxy"],
            ) >= 0.80
            for prior in kept
        )

        if not duplicate:
            kept.append(item)

    return {
        "image": image_path.name,
        "detection_count": len(kept),
        "detections": kept,
    }

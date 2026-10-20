
from pathlib import Path
from ultralytics import YOLO

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


def detect_damage(image_path, confidence_threshold=0.25):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not 0 <= confidence_threshold <= 1:
        raise ValueError(
            "Confidence threshold must be between 0 and 1."
        )

    model = get_model()

    results = model.predict(
        source=str(image_path),
        conf=confidence_threshold,
        verbose=False
    )

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())

            detections.append({
                "class_id": class_id,
                "damage_type": model.names[class_id],
                "confidence": round(float(box.conf.item()), 4),
                "bbox_xyxy": [
                    round(float(value), 2)
                    for value in box.xyxy[0].tolist()
                ]
            })

    return {
        "image": image_path.name,
        "detection_count": len(detections),
        "detections": detections
    }

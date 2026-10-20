
from pathlib import Path
from ultralytics import YOLO

# best.pt is expected in the same folder as this Python file
MODEL_PATH = Path(__file__).resolve().parent / "best.pt"

_model = None


def get_model():
    """Load the trained YOLO model once and reuse it."""
    global _model

    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model not found at: {MODEL_PATH}"
            )

        _model = YOLO(str(MODEL_PATH))

    return _model


def detect_damage(image_path, confidence_threshold=0.25):
    """Detect road damage and return structured results."""
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
}

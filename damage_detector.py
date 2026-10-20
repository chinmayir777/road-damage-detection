
from pathlib import Path
from ultralytics import YOLO

# Render deploys this file from /opt/render/project/src
MODEL_PATH = Path(__file__).resolve().parent / "best.pt"

_model = None


def get_model():
    global _model

    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )
        _model = YOLO(str(MODEL_PATH))

    return _model


def detect_damage(image_path, confidence_threshold=0.25):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

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
                    round(float(v), 2)
                    for v in box.xyxy[0].tolist()
                ]
            })

    return {
        "image": image_path.name,
        "detection_count": len(detections),
        "detections": detections
    }

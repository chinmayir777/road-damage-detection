import json
import os
import psycopg


def save_report(image_path, latitude, longitude, detections, description=None):
    """
    Save a public road report and its AI detections in one transaction.
    image_path must identify the stored image.
    """
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        try:
            from google.colab import userdata
            database_url = userdata.get("DATABASE_URL")
        except Exception:
            database_url = None

    if not database_url:
        raise RuntimeError("DATABASE_URL is not available to the API process.")

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO road_reports
                    (image_path, latitude, longitude, description)
                VALUES (%s, %s, %s, %s)
                RETURNING report_id, reference_code, submitted_at;
                """,
                (image_path, latitude, longitude, description)
            )
            report_id, reference_code, submitted_at = cur.fetchone()

            for detection in detections:
                cur.execute(
                    """
                    INSERT INTO damage_detections
                        (report_id, damage_type, confidence, bbox_xyxy)
                    VALUES (%s, %s, %s, %s::jsonb);
                    """,
                    (
                        report_id,
                        detection["damage_type"],
                        detection["confidence"],
                        json.dumps(detection["bbox_xyxy"])
                    )
                )

    return {
        "report_id": report_id,
        "reference_code": str(reference_code),
        "submitted_at": submitted_at.isoformat()
    }

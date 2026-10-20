
from pathlib import Path
import sys
import uuid
import tempfile
import shutil
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# All Python files are in the repository root
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from damage_detector import detect_damage
from report_service import save_report


app = FastAPI(
    title="AI-Powered Road Damage System",
    description="Road damage reporting and AI detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp"
}
MAX_FILE_SIZE = 10 * 1024 * 1024

# Temporary upload directory suitable for Render
UPLOAD_DIR = Path(tempfile.gettempdir()) / "road_damage_uploads"


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "AI-Powered Road Damage System"
    }


@app.post("/api/reports")
async def submit_report(
    latitude: float = Form(...),
    longitude: float = Form(...),
    description: str = Form(""),
    file: UploadFile = File(...)
):
    if not -90 <= latitude <= 90:
        raise HTTPException(status_code=400, detail="Invalid latitude.")

    if not -180 <= longitude <= 180:
        raise HTTPException(status_code=400, detail="Invalid longitude.")

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, or WebP image."
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(status_code=400, detail="The image is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be 10 MB or smaller."
        )

    temp_path = None
    saved_image_path = None

    try:
        # Save the upload temporarily for AI inference
        with tempfile.NamedTemporaryFile(
            suffix=ALLOWED_TYPES[file.content_type],
            delete=False
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        # Run road damage detection
        ai_result = detect_damage(temp_path)

        # Save a copy of the image
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        extension = ALLOWED_TYPES[file.content_type]
        saved_image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"

        shutil.copy2(temp_path, saved_image_path)

        # Save report and detections to the database
        report = save_report(
            image_path=str(saved_image_path),
            latitude=latitude,
            longitude=longitude,
            description=description.strip() or None,
            detections=ai_result["detections"]
        )

        return {
            "message": "Report submitted successfully",
            "reference_code": str(report["reference_code"]),
            "submitted_at": (
                report["submitted_at"].isoformat()
                if hasattr(report["submitted_at"], "isoformat")
                else report["submitted_at"]
            ),
            "detection_count": ai_result["detection_count"],
            "detections": ai_result["detections"]
        }

    except HTTPException:
        raise

    except Exception as e:
        print("REPORT SUBMISSION ERROR:", repr(e), flush=True)

        # Remove the copied image if saving the report failed
        if saved_image_path is not None:
            saved_image_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Report submission failed. Please check server logs."
        )

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

        await file.close()


@app.get("/api/reports")
def list_reports():
    import psycopg

    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        raise HTTPException(
            status_code=500,
            detail="Database configuration unavailable."
        )

    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT report_id, reference_code, latitude, longitude,
                           description, submitted_at, report_status
                    FROM road_reports
                    ORDER BY submitted_at DESC
                    LIMIT 100;
                """)
                rows = cur.fetchall()

        return [
            {
                "report_id": r[0],
                "reference_code": str(r[1]),
                "latitude": r[2],
                "longitude": r[3],
                "description": r[4],
                "submitted_at": r[5].isoformat() if r[5] else None,
                "report_status": r[6]
            }
            for r in rows
        ]

    except Exception as e:
        print("REPORT LIST ERROR:", repr(e), flush=True)
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve reports."
        )

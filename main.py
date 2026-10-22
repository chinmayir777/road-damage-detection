from pathlib import Path
import json
import os
import shutil
import tempfile
import time
import uuid

import psycopg
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from damage_detector import detect_damage
from report_service import save_report

app = FastAPI(
    title="AI-Powered Road Damage Detection and Maintenance Prioritization System",
    description="YOLO detection, damage features, severity proxy, uncertainty-aware routing and TOPSIS-style ranking",
    version="1.3.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024
VALID_STATUSES = {"Reported", "Under Review", "Repair Scheduled", "In Progress", "Repaired"}


def db_url():
    value = os.environ.get("DATABASE_URL")
    if not value:
        raise HTTPException(status_code=500, detail="Database configuration unavailable.")
    return value


def ensure_columns(conn):
    # Additive migration only; existing report data is retained.
    conn.execute("""
        ALTER TABLE road_reports
        ADD COLUMN IF NOT EXISTS priority TEXT NOT NULL DEFAULT 'Pending',
        ADD COLUMN IF NOT EXISTS priority_score INTEGER,
        ADD COLUMN IF NOT EXISTS severity TEXT,
        ADD COLUMN IF NOT EXISTS avg_confidence DOUBLE PRECISION,
        ADD COLUMN IF NOT EXISTS uncertainty_level TEXT,
        ADD COLUMN IF NOT EXISTS manual_verification_required BOOLEAN NOT NULL DEFAULT TRUE,
        ADD COLUMN IF NOT EXISTS decision_route TEXT NOT NULL DEFAULT 'Review',
        ADD COLUMN IF NOT EXISTS detection_count INTEGER NOT NULL DEFAULT 0,
        ADD COLUMN IF NOT EXISTS detections JSONB NOT NULL DEFAULT '[]'::jsonb,
        ADD COLUMN IF NOT EXISTS damage_size_ratio DOUBLE PRECISION,
        ADD COLUMN IF NOT EXISTS damage_extent DOUBLE PRECISION,
        ADD COLUMN IF NOT EXISTS damage_density DOUBLE PRECISION,
        ADD COLUMN IF NOT EXISTS traffic_level TEXT,
        ADD COLUMN IF NOT EXISTS road_importance TEXT,
        ADD COLUMN IF NOT EXISTS location_type TEXT,
        ADD COLUMN IF NOT EXISTS context_status TEXT NOT NULL DEFAULT 'Pending verification',
        ADD COLUMN IF NOT EXISTS conventional_score DOUBLE PRECISION,
        ADD COLUMN IF NOT EXISTS proposed_score DOUBLE PRECISION
    """)


def extract_features(detections):
    """Image-relative bounding-box features. Full-image dimensions are not available here,
    so size ratio is an approximation relative to the union of detected boxes, not the image."""
    if not detections:
        return {"damage_size_ratio": 0.0, "damage_extent": 0.0, "damage_density": 0.0}
    boxes = [d.get("bbox_xyxy") or [] for d in detections]
    valid = [b for b in boxes if len(b) == 4]
    if not valid:
        return {"damage_size_ratio": 0.0, "damage_extent": 0.0, "damage_density": float(len(detections))}
    areas = [max(0.0, float(b[2])-float(b[0])) * max(0.0, float(b[3])-float(b[1])) for b in valid]
    total_area = sum(areas)
    min_x = min(float(b[0]) for b in valid)
    min_y = min(float(b[1]) for b in valid)
    max_x = max(float(b[2]) for b in valid)
    max_y = max(float(b[3]) for b in valid)
    union_box_area = max(1.0, (max_x-min_x) * (max_y-min_y))
    return {
        "damage_size_ratio": round(max(areas) / max(1.0, total_area), 4),
        "damage_extent": round(min(1.0, total_area / union_box_area), 4),
        "damage_density": round(len(valid) / max(1.0, union_box_area / 10000.0), 4),
    }


def assess_severity(detections, features):
    # Transparent proxy until a separately trained and evaluated severity classifier is supplied.
    if not detections:
        return "Unconfirmed"
    names = [str(d.get("damage_type", "")).lower() for d in detections]
    confs = [max(0.0, min(1.0, float(d.get("confidence", 0) or 0))) for d in detections]
    class_points = sum(3 if "pothole" in n else 2 if ("crack" in n or "alligator" in n) else 1 for n in names)
    size_points = min(3.0, features["damage_density"] * 0.25)
    score = class_points + size_points + sum(confs) / len(confs)
    return "Severe" if score >= 7 else "Moderate" if score >= 3.5 else "Minor"


def confidence_uncertainty(detections):
    # This is a provisional uncertainty heuristic, not statistical calibration.
    if not detections:
        return None, "High", True, "Human verification"
    values = [max(0.0, min(1.0, float(d.get("confidence", 0) or 0))) for d in detections]
    mean = sum(values) / len(values)
    low_fraction = sum(v < 0.40 for v in values) / len(values)
    level = "High" if mean < 0.40 or low_fraction >= 0.5 else "Medium" if mean < 0.65 else "Low"
    if mean >= 0.75 and low_fraction == 0:
        route = "Automatic candidate"
        verify = False
    elif mean >= 0.45:
        route = "Review"
        verify = True
    else:
        route = "Human verification"
        verify = True
    return round(mean, 4), level, verify, route


def topsis_rank(report_rows, context_verified=False):
    """TOPSIS-style multi-criteria ranking over current reports.
    Criteria: severity, context/risk, confidence/reliability and damage extent.
    Context remains neutral until verified by the authority/system.
    """
    if not report_rows:
        return {}
    import math
    criteria = []
    for r in report_rows:
        # Row indices: id, severity, context_points, confidence, uncertainty penalty, extent, route
        criteria.append([
            float(r[1] or 0),
            float(r[2] or 0),
            float(r[3] or 0) * (0.7 if r[4] else 1.0),
            float(r[5] or 0),
        ])
    weights = [0.35, 0.30, 0.20, 0.15]
    norms = []
    for j in range(4):
        denom = math.sqrt(sum(row[j] ** 2 for row in criteria))
        norms.append([row[j] / denom if denom else 0.0 for row in criteria])
    weighted = [[norms[j][i] * weights[j] for j in range(4)] for i in range(len(criteria))]  
    ideal_best = [max(row[j] for row in weighted) for j in range(4)]
    ideal_worst = [min(row[j] for row in weighted) for j in range(4)]
    out = {}
    for i, row in enumerate(weighted):
        d_best = math.sqrt(sum((row[j] - ideal_best[j]) ** 2 for j in range(4)))
        d_worst = math.sqrt(sum((row[j] - ideal_worst[j]) ** 2 for j in range(4)))
        closeness = d_worst / (d_best + d_worst) if (d_best + d_worst) else 0.0
        out[report_rows[i][0]] = round(closeness * 100)
    return out


def refresh_priorities(conn):
    rows = conn.execute("""
        SELECT report_id,
               CASE severity WHEN 'Minor' THEN 1 WHEN 'Moderate' THEN 2 WHEN 'Severe' THEN 3 ELSE 0 END AS severity_value,
               CASE
                 WHEN traffic_level IS NULL OR road_importance IS NULL OR location_type IS NULL THEN 0
                 ELSE
                   (CASE traffic_level WHEN 'High' THEN 3 WHEN 'Medium' THEN 2 WHEN 'Low' THEN 1 ELSE 0 END)
                   + (CASE road_importance WHEN 'Major' THEN 3 WHEN 'Collector' THEN 2 WHEN 'Local' THEN 1 ELSE 0 END)
                   + (CASE location_type WHEN 'School/Hospital' THEN 3 WHEN 'Junction/Transit' THEN 2 WHEN 'General' THEN 1 ELSE 0 END)
               END AS context_points,
               COALESCE(avg_confidence, 0), CASE WHEN manual_verification_required THEN 1 ELSE 0 END,
               COALESCE(damage_extent, 0), decision_route
        FROM road_reports
        WHERE report_status <> 'Repaired'
    """).fetchall()
    scores = topsis_rank(rows)
    for row in rows:
        score = scores.get(row[0], 0)
        priority = "High" if score >= 67 else "Medium" if score >= 34 else "Low"
        # A low-confidence result cannot be auto-actioned even if ranked high.
        route = row[6] or "Review"
        if route != "Automatic candidate":
            route = "Human verification" if route == "Human verification" else "Review"
        conn.execute(
            "UPDATE road_reports SET priority_score=%s, proposed_score=%s, priority=%s, decision_route=%s WHERE report_id=%s",
            (score, score, priority, route, row[0]),
        )


@app.get("/", response_class=HTMLResponse)
def dashboard():
    path = Path(__file__).resolve().parent / "dashboard.html"
    if not path.is_file():
        return HTMLResponse("<h2>Dashboard file not found</h2><p>Place dashboard.html beside main.py.</p>", status_code=500)
    return path.read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AI-Powered Road Damage Detection and Maintenance Prioritization System"}


@app.post("/api/reports")
async def submit_report(
    latitude: float = Form(...),
    longitude: float = Form(...),
    description: str = Form(""),
    file: UploadFile = File(...),
):
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise HTTPException(status_code=400, detail="Invalid location coordinates.")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WebP image.")
    content = await file.read(MAX_FILE_SIZE + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The image is empty.")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    temp_path = None
    request_started = time.perf_counter()
    print("[ROAD-REPORT] Upload request received", flush=True)
    try:
        with tempfile.NamedTemporaryFile(suffix=ALLOWED_TYPES[file.content_type], delete=False) as temp:
            temp.write(content)
            temp_path = Path(temp.name)
        inference_started = time.perf_counter()
        print("[ROAD-REPORT] Starting YOLO inference", flush=True)
        result = await run_in_threadpool(detect_damage, temp_path)
        print(
            f"[ROAD-REPORT] YOLO finished in {time.perf_counter() - inference_started:.2f}s",
            flush=True,
        )
        detections = result["detections"]
        features = extract_features(detections)
        severity = assess_severity(detections, features)
        avg_conf, uncertainty, manual_review, route = confidence_uncertainty(detections)
        upload_dir = Path(tempfile.gettempdir()) / "road_damage_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        saved = upload_dir / f"{uuid.uuid4().hex}{ALLOWED_TYPES[file.content_type]}"
        shutil.copy2(temp_path, saved)
        print("[ROAD-REPORT] Saving report record", flush=True)
        save_started = time.perf_counter()
        report = await run_in_threadpool(
            lambda: save_report(
                image_path=str(saved), latitude=latitude, longitude=longitude,
                description=description.strip() or None, detections=detections,
            )
        )
        print(
            f"[ROAD-REPORT] Report record saved in {time.perf_counter() - save_started:.2f}s",
            flush=True,
        )
        print("[ROAD-REPORT] Updating report and calculating priorities", flush=True)
        with psycopg.connect(db_url(), connect_timeout=10) as conn:
            ensure_columns(conn)
            conn.execute("""
                UPDATE road_reports SET detections=%s::jsonb, detection_count=%s,
                    severity=%s, avg_confidence=%s, uncertainty_level=%s,
                    manual_verification_required=%s, decision_route=%s,
                    damage_size_ratio=%s, damage_extent=%s, damage_density=%s,
                    context_status='Pending authority/system verification'
                WHERE reference_code::text=%s
            """, (
                json.dumps(detections), len(detections), severity, avg_conf,
                uncertainty, manual_review, route, features["damage_size_ratio"],
                features["damage_extent"], features["damage_density"], str(report["reference_code"]),
            ))
            # Baseline is damage-only; proposed ranking becomes meaningful as road context is verified.
            conn.execute("""
                UPDATE road_reports SET conventional_score = %s
                WHERE reference_code::text=%s
            """, (min(100, len(detections) * 20 + (avg_conf or 0) * 40), str(report["reference_code"])))
            refresh_priorities(conn)
            current = conn.execute("""
                SELECT priority, priority_score, conventional_score, proposed_score
                FROM road_reports WHERE reference_code::text=%s
            """, (str(report["reference_code"]),)).fetchone()
        print(
            f"[ROAD-REPORT] Upload completed in {time.perf_counter() - request_started:.2f}s",
            flush=True,
        )
        return {
            "message": "Report submitted successfully",
            "reference_code": str(report["reference_code"]),
            "submitted_at": report["submitted_at"],
            "detection_count": len(detections), "detections": detections,
            "features": features, "severity": severity,
            "avg_confidence": avg_conf, "uncertainty_level": uncertainty,
            "manual_verification_required": manual_review, "decision_route": route,
            "priority": current[0], "priority_score": current[1],
            "conventional_score": current[2], "proposed_score": current[3],
            "context_status": "Pending authority/system verification",
        }
    except HTTPException:
        raise
    except Exception as e:
        print(
            f"[ROAD-REPORT] SUBMISSION ERROR after {time.perf_counter() - request_started:.2f}s: {e!r}",
            flush=True,
        )
        raise HTTPException(status_code=500, detail="Report submission failed. Check Render logs.") from e
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.get("/api/reports")
def list_reports():
    try:
        with psycopg.connect(db_url(), connect_timeout=10) as conn:
            ensure_columns(conn)
            rows = conn.execute("""
                SELECT report_id, reference_code, latitude, longitude, description,
                       submitted_at, report_status, priority, detection_count, detections,
                       priority_score, severity, avg_confidence, uncertainty_level,
                       manual_verification_required, decision_route, traffic_level,
                       road_importance, location_type, context_status, damage_size_ratio,
                       damage_extent, damage_density, conventional_score, proposed_score
                FROM road_reports ORDER BY submitted_at DESC LIMIT 100
            """).fetchall()
        return [{
            "report_id": r[0], "reference_code": str(r[1]), "latitude": r[2], "longitude": r[3],
            "description": r[4], "submitted_at": r[5].isoformat() if r[5] else None,
            "report_status": r[6], "priority": r[7], "detection_count": r[8], "detections": r[9],
            "priority_score": r[10], "severity": r[11], "avg_confidence": r[12],
            "uncertainty_level": r[13], "manual_verification_required": r[14],
            "decision_route": r[15], "traffic_level": r[16], "road_importance": r[17],
            "location_type": r[18], "context_status": r[19], "damage_size_ratio": r[20],
            "damage_extent": r[21], "damage_density": r[22], "conventional_score": r[23],
            "proposed_score": r[24],
        } for r in rows]
    except HTTPException:
        raise
    except Exception as e:
        print("REPORT LIST ERROR:", repr(e), flush=True)
        raise HTTPException(status_code=500, detail="Could not retrieve reports.")


class StatusUpdate(BaseModel):
    report_status: str


class ContextUpdate(BaseModel):
    traffic_level: str
    road_importance: str
    location_type: str


@app.patch("/api/reports/{report_id}/context")
def update_road_context(report_id: int, update: ContextUpdate, x_authority_pin: str | None = Header(default=None)):
    pin = os.environ.get("AUTHORITY_PIN")
    if not pin:
        raise HTTPException(status_code=503, detail="Authority updates are not configured. Set AUTHORITY_PIN in Render.")
    if not x_authority_pin or x_authority_pin != pin:
        raise HTTPException(status_code=403, detail="Incorrect authority PIN.")
    if update.traffic_level not in {"Low", "Medium", "High"}:
        raise HTTPException(status_code=400, detail="Traffic must be Low, Medium, or High.")
    if update.road_importance not in {"Local", "Collector", "Major"}:
        raise HTTPException(status_code=400, detail="Road importance must be Local, Collector, or Major.")
    if update.location_type not in {"General", "Junction/Transit", "School/Hospital"}:
        raise HTTPException(status_code=400, detail="Invalid location type.")
    try:
        with psycopg.connect(db_url(), connect_timeout=10) as conn:
            ensure_columns(conn)
            found = conn.execute("""
                UPDATE road_reports SET traffic_level=%s, road_importance=%s,
                    location_type=%s, context_status='Verified by authority'
                WHERE report_id=%s RETURNING report_id
            """, (update.traffic_level, update.road_importance, update.location_type, report_id)).fetchone()
            if not found:
                raise HTTPException(status_code=404, detail="Report not found.")
            refresh_priorities(conn)
        return {"message": "Road context saved and priority ranking refreshed", "report_id": report_id}
    except HTTPException:
        raise
    except Exception as e:
        print("CONTEXT UPDATE ERROR:", repr(e), flush=True)
        raise HTTPException(status_code=500, detail="Could not save road context.")


@app.patch("/api/reports/{report_id}/status")
def update_status(report_id: int, update: StatusUpdate, x_authority_pin: str | None = Header(default=None)):
    pin = os.environ.get("AUTHORITY_PIN")
    if not pin:
        raise HTTPException(status_code=503, detail="Authority updates are not configured. Set AUTHORITY_PIN in Render.")
    if not x_authority_pin or x_authority_pin != pin:
        raise HTTPException(status_code=403, detail="Incorrect authority PIN.")
    if update.report_status not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid report status.")
    try:
        with psycopg.connect(db_url(), connect_timeout=10) as conn:
            ensure_columns(conn)
            found = conn.execute(
                "UPDATE road_reports SET report_status=%s WHERE report_id=%s RETURNING report_id",
                (update.report_status, report_id),
            ).fetchone()
            if not found:
                raise HTTPException(status_code=404, detail="Report not found.")
            refresh_priorities(conn)
        return {"message": "Report status updated", "report_id": report_id, "report_status": update.report_status}
    except HTTPException:
        raise
    except Exception as e:
        print("STATUS UPDATE ERROR:", repr(e), flush=True)
        raise HTTPException(status_code=500, detail="Could not update report status.")

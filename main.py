from pathlib import Path
import json
import os
import shutil
import tempfile
import time
import uuid
import urllib.parse
import urllib.request
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
    description=(
        "YOLO detection, severity assessment, context retrieval, "
        "uncertainty-aware review and TOPSIS-style prioritization"
    ),
    version="1.5.0",
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
    "image/webp": ".webp",
}

MAX_FILE_SIZE = 10 * 1024 * 1024

VALID_STATUSES = {
    "Reported",
    "Under Review",
    "Repair Scheduled",
    "In Progress",
    "Repaired",
}


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def db_url():
    value = os.environ.get("DATABASE_URL")

    if not value:
        raise HTTPException(
            status_code=500,
            detail="Database configuration unavailable.",
        )

    return value


def ensure_columns(conn):
    """
    Add missing columns without deleting existing records.
    This is an additive database migration.
    """

    conn.execute("""
        ALTER TABLE road_reports
        ADD COLUMN IF NOT EXISTS priority
            TEXT NOT NULL DEFAULT 'Pending',

        ADD COLUMN IF NOT EXISTS priority_score
            INTEGER,

        ADD COLUMN IF NOT EXISTS severity
            TEXT,

        ADD COLUMN IF NOT EXISTS avg_confidence
            DOUBLE PRECISION,

        ADD COLUMN IF NOT EXISTS uncertainty_level
            TEXT,

        ADD COLUMN IF NOT EXISTS manual_verification_required
            BOOLEAN NOT NULL DEFAULT TRUE,

        ADD COLUMN IF NOT EXISTS decision_route
            TEXT NOT NULL DEFAULT 'Review',

        ADD COLUMN IF NOT EXISTS detection_count
            INTEGER NOT NULL DEFAULT 0,

        ADD COLUMN IF NOT EXISTS detections
            JSONB NOT NULL DEFAULT '[]'::jsonb,

        ADD COLUMN IF NOT EXISTS damage_size_ratio
            DOUBLE PRECISION,

        ADD COLUMN IF NOT EXISTS damage_extent
            DOUBLE PRECISION,

        ADD COLUMN IF NOT EXISTS damage_density
            DOUBLE PRECISION,

        ADD COLUMN IF NOT EXISTS traffic_level
            TEXT,

        ADD COLUMN IF NOT EXISTS road_importance
            TEXT,

        ADD COLUMN IF NOT EXISTS location_type
            TEXT,

        ADD COLUMN IF NOT EXISTS context_status
            TEXT NOT NULL DEFAULT 'Automatic lookup pending',

        ADD COLUMN IF NOT EXISTS road_name
            TEXT,

        ADD COLUMN IF NOT EXISTS road_class
            TEXT,

        ADD COLUMN IF NOT EXISTS context_source
            TEXT,

        ADD COLUMN IF NOT EXISTS context_retrieved_at
            TIMESTAMPTZ,

        ADD COLUMN IF NOT EXISTS conventional_score
            DOUBLE PRECISION,

        ADD COLUMN IF NOT EXISTS proposed_score
            DOUBLE PRECISION
    """)


# --------------------------------------------------
# AUTOMATIC ROAD CONTEXT RETRIEVAL
# --------------------------------------------------

def _nominatim_request(params):
    """Call Nominatim with a descriptive User-Agent."""
    url = "https://nominatim.openstreetmap.org/" + params.pop("_endpoint") + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "RoadDamagePrioritizationProject/1.0 (educational prototype)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def geocode_address(address):
    """Convert a user-entered address/landmark into coordinates."""
    query = (address or "").strip()
    if len(query) < 3:
        raise HTTPException(status_code=400, detail="Enter a more complete address or landmark.")

    if not hasattr(geocode_address, "cache"):
        geocode_address.cache = {}
    key = query.casefold()
    if key in geocode_address.cache:
        return dict(geocode_address.cache[key])

    try:
        results = _nominatim_request({
            "_endpoint": "search",
            "q": query,
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 1,
            "extratags": 1,
        })
        if not results:
            raise HTTPException(status_code=404, detail="Address not found. Try adding the area, city, or a nearby landmark.")

        item = results[0]
        result = {
            "latitude": float(item["lat"]),
            "longitude": float(item["lon"]),
            "display_name": item.get("display_name") or query,
            "source": "OpenStreetMap Nominatim",
        }
        geocode_address.cache[key] = dict(result)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        print("[ROAD-GEOCODE] Lookup failed:", repr(exc), flush=True)
        raise HTTPException(status_code=502, detail="Address lookup is temporarily unavailable. Use current GPS or try again.") from exc


def retrieve_road_context(latitude, longitude):
    """Retrieve mapped road/address attributes; never fabricate traffic volume."""
    result = {
        "road_name": None,
        "road_class": None,
        "road_importance": None,
        "location_type": None,
        "traffic_level": None,
        "context_source": None,
        "context_status": "Context lookup failed; retry required",
    }

    key = (round(float(latitude), 5), round(float(longitude), 5))
    if not hasattr(retrieve_road_context, "cache"):
        retrieve_road_context.cache = {}
    if key in retrieve_road_context.cache:
        return dict(retrieve_road_context.cache[key])

    try:
        data = _nominatim_request({
            "_endpoint": "reverse",
            "format": "jsonv2",
            "lat": float(latitude),
            "lon": float(longitude),
            "zoom": 18,
            "addressdetails": 1,
            "extratags": 1,
        })
        address = data.get("address") or {}
        tags = data.get("extratags") or {}

        result["road_name"] = (
            address.get("road") or address.get("pedestrian")
            or address.get("residential") or address.get("path")
            or address.get("footway") or address.get("cycleway")
        )

        # Reverse results may expose highway as an extra tag or result type.
        road_class = tags.get("highway")
        if not road_class and data.get("category") == "highway":
            road_class = data.get("type")
        result["road_class"] = road_class

        # These are broad map-class proxies, not measured risk or traffic.
        if road_class in {"motorway", "trunk", "primary"}:
            result["road_importance"] = "Major"
        elif road_class in {"secondary", "tertiary"}:
            result["road_importance"] = "Collector"
        elif road_class in {"residential", "living_street", "service", "unclassified"}:
            result["road_importance"] = "Local"

        # Use an explicit mapped amenity if present; don't infer proximity.
        amenity = tags.get("amenity") or address.get("amenity")
        if amenity in {"school", "hospital", "clinic"}:
            result["location_type"] = "School/Hospital"
        elif amenity in {"bus_station", "bus_stop", "parking"}:
            result["location_type"] = "Junction/Transit"

        result["context_source"] = "OpenStreetMap Nominatim"
        result["context_status"] = (
            "Road details retrieved; traffic data unavailable"
            if result["road_name"] or result["road_class"]
            else "Map provider returned no road details for these coordinates"
        )
        retrieve_road_context.cache[key] = dict(result)

        print("[ROAD-CONTEXT] " + json.dumps({
            "coordinates": key,
            "road_name": result["road_name"],
            "road_class": result["road_class"],
            "status": result["context_status"],
        }), flush=True)
    except Exception as exc:
        print("[ROAD-CONTEXT] Lookup failed:", repr(exc), flush=True)

    return result


# --------------------------------------------------
# DAMAGE FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(detections):
    """
    Calculate approximate bounding-box features.

    Note: full original image dimensions are not available here.
    These features are proxies, not physical measurements.
    """

    if not detections:
        return {
            "damage_size_ratio": 0.0,
            "damage_extent": 0.0,
            "damage_density": 0.0,
        }

    boxes = [
        detection.get("bbox_xyxy") or []
        for detection in detections
    ]

    valid_boxes = [
        box for box in boxes
        if len(box) == 4
    ]

    if not valid_boxes:
        return {
            "damage_size_ratio": 0.0,
            "damage_extent": 0.0,
            "damage_density": float(len(detections)),
        }

    areas = [
        max(0.0, float(box[2]) - float(box[0]))
        * max(0.0, float(box[3]) - float(box[1]))
        for box in valid_boxes
    ]

    total_area = sum(areas)

    min_x = min(float(box[0]) for box in valid_boxes)
    min_y = min(float(box[1]) for box in valid_boxes)
    max_x = max(float(box[2]) for box in valid_boxes)
    max_y = max(float(box[3]) for box in valid_boxes)

    bounding_region_area = max(
        1.0,
        (max_x - min_x) * (max_y - min_y),
    )

    return {
        "damage_size_ratio": round(
            max(areas) / max(1.0, total_area),
            4,
        ),

        "damage_extent": round(
            min(1.0, total_area / bounding_region_area),
            4,
        ),

        "damage_density": round(
            len(valid_boxes)
            / max(1.0, bounding_region_area / 10000.0),
            4,
        ),
    }


# --------------------------------------------------
# SEVERITY ASSESSMENT
# --------------------------------------------------

def assess_severity(detections, features):
    """
    Heuristic severity estimate.
    This is not a trained severity classifier.
    """

    if not detections:
        return "Unconfirmed"

    names = [
        str(d.get("damage_type", "")).lower()
        for d in detections
    ]

    confidences = [
        max(
            0.0,
            min(
                1.0,
                float(d.get("confidence", 0) or 0),
            ),
        )
        for d in detections
    ]

    class_points = sum(
        3 if "pothole" in name
        else 2 if (
            "crack" in name or "alligator" in name
        )
        else 1
        for name in names
    )

    size_points = min(
        3.0,
        features["damage_density"] * 0.25,
    )

    average_confidence = (
        sum(confidences) / len(confidences)
    )

    score = (
        class_points
        + size_points
        + average_confidence
    )

    if score >= 7:
        return "Severe"

    if score >= 3.5:
        return "Moderate"

    return "Minor"


# --------------------------------------------------
# CONFIDENCE AND UNCERTAINTY
# --------------------------------------------------

def confidence_uncertainty(detections):
    """
    Provisional uncertainty heuristic.
    Confidence scores are not statistically calibrated.
    """

    if not detections:
        return (
            None,
            "High",
            True,
            "Human verification",
        )

    values = [
        max(
            0.0,
            min(
                1.0,
                float(d.get("confidence", 0) or 0),
            ),
        )
        for d in detections
    ]

    mean_confidence = sum(values) / len(values)

    low_fraction = (
        sum(value < 0.40 for value in values)
        / len(values)
    )

    if mean_confidence < 0.40 or low_fraction >= 0.5:
        uncertainty = "High"
    elif mean_confidence < 0.65:
        uncertainty = "Medium"
    else:
        uncertainty = "Low"

    if mean_confidence >= 0.75 and low_fraction == 0:
        route = "Automatic candidate"
        verification_required = False

    elif mean_confidence >= 0.45:
        route = "Review"
        verification_required = True

    else:
        route = "Human verification"
        verification_required = True

    return (
        round(mean_confidence, 4),
        uncertainty,
        verification_required,
        route,
    )


# --------------------------------------------------
# TOPSIS-STYLE PRIORITY RANKING
# --------------------------------------------------

def topsis_rank(report_rows):
    """
    TOPSIS-style ranking.

    Criteria:
      1. Severity
      2. Available context/risk information
      3. Confidence adjusted for verification uncertainty
      4. Approximate damage extent

    Scores are prototype-relative rankings, not validated risk scores.
    """

    if not report_rows:
        return {}

    import math

    criteria = []

    for row in report_rows:
        criteria.append([
            float(row[1] or 0),
            float(row[2] or 0),
            float(row[3] or 0)
            * (0.7 if row[4] else 1.0),
            float(row[5] or 0),
        ])

    weights = [0.35, 0.30, 0.20, 0.15]

    normalized = []

    for column in range(4):
        denominator = math.sqrt(
            sum(
                row[column] ** 2
                for row in criteria
            )
        )

        normalized.append([
            row[column] / denominator
            if denominator else 0.0
            for row in criteria
        ])

    weighted = [
        [
            normalized[column][index] * weights[column]
            for column in range(4)
        ]
        for index in range(len(criteria))
    ]

    ideal_best = [
        max(row[column] for row in weighted)
        for column in range(4)
    ]

    ideal_worst = [
        min(row[column] for row in weighted)
        for column in range(4)
    ]

    scores = {}

    for index, row in enumerate(weighted):
        distance_best = math.sqrt(
            sum(
                (row[column] - ideal_best[column]) ** 2
                for column in range(4)
            )
        )

        distance_worst = math.sqrt(
            sum(
                (row[column] - ideal_worst[column]) ** 2
                for column in range(4)
            )
        )

        denominator = distance_best + distance_worst

        closeness = (
            distance_worst / denominator
            if denominator else 0.0
        )

        scores[report_rows[index][0]] = round(
            closeness * 100
        )

    return scores


def refresh_priorities(conn):
    """
    Recalculate rankings for all reports that are not repaired.
    Missing context is treated as unknown, not fabricated.
    """

    rows = conn.execute("""
        SELECT
            report_id,

            CASE severity
                WHEN 'Minor' THEN 1
                WHEN 'Moderate' THEN 2
                WHEN 'Severe' THEN 3
                ELSE 0
            END AS severity_value,

            CASE
                WHEN traffic_level IS NULL
                  OR road_importance IS NULL
                  OR location_type IS NULL
                THEN 0

                ELSE
                    (
                        CASE traffic_level
                            WHEN 'High' THEN 3
                            WHEN 'Medium' THEN 2
                            WHEN 'Low' THEN 1
                            ELSE 0
                        END
                    )
                    +
                    (
                        CASE road_importance
                            WHEN 'Major' THEN 3
                            WHEN 'Collector' THEN 2
                            WHEN 'Local' THEN 1
                            ELSE 0
                        END
                    )
                    +
                    (
                        CASE location_type
                            WHEN 'School/Hospital' THEN 3
                            WHEN 'Junction/Transit' THEN 2
                            WHEN 'General' THEN 1
                            ELSE 0
                        END
                    )
            END AS context_points,

            COALESCE(avg_confidence, 0),

            CASE
                WHEN manual_verification_required THEN 1
                ELSE 0
            END,

            COALESCE(damage_extent, 0),

            decision_route

        FROM road_reports
        WHERE report_status <> 'Repaired'
    """).fetchall()

    scores = topsis_rank(rows)

    for row in rows:
        score = scores.get(row[0], 0)

        if score >= 67:
            priority = "High"
        elif score >= 34:
            priority = "Medium"
        else:
            priority = "Low"

        route = row[6] or "Review"

        if route != "Automatic candidate":
            route = (
                "Human verification"
                if route == "Human verification"
                else "Review"
            )

        conn.execute("""
            UPDATE road_reports
            SET
                priority_score = %s,
                proposed_score = %s,
                priority = %s,
                decision_route = %s
            WHERE report_id = %s
        """, (
            score,
            score,
            priority,
            route,
            row[0],
        ))


# --------------------------------------------------
# HTML DASHBOARDS
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def dashboard():
    path = Path(__file__).resolve().parent / "dashboard.html"

    if not path.is_file():
        return HTMLResponse(
            "<h2>Dashboard file not found</h2>"
            "<p>Place dashboard.html beside main.py.</p>",
            status_code=500,
        )

    return path.read_text(encoding="utf-8")


@app.get("/authority", response_class=HTMLResponse)
def authority_dashboard():
    path = Path(__file__).resolve().parent / "authority.html"

    if not path.is_file():
        return HTMLResponse(
            "<h2>Authority dashboard file not found</h2>"
            "<p>Place authority.html beside main.py.</p>",
            status_code=500,
        )

    return path.read_text(encoding="utf-8")


# --------------------------------------------------
# AUTHORITY AUTHENTICATION
# --------------------------------------------------

def require_authority_pin(
    x_authority_pin: str | None,
):
    configured_pin = os.environ.get("AUTHORITY_PIN")

    if not configured_pin:
        raise HTTPException(
            status_code=503,
            detail=(
                "Authority access is not configured. "
                "Set AUTHORITY_PIN in Render."
            ),
        )

    if (
        not x_authority_pin
        or x_authority_pin != configured_pin
    ):
        raise HTTPException(
            status_code=403,
            detail="Incorrect authority PIN.",
        )


@app.post("/api/authority/verify")
def verify_authority(
    x_authority_pin: str | None = Header(default=None),
):
    require_authority_pin(x_authority_pin)

    return {
        "authenticated": True,
        "message": "Authority access verified.",
    }


@app.get("/api/geocode")
async def geocode_endpoint(address: str):
    """Public address-to-coordinate lookup for the reporting form."""
    return await run_in_threadpool(geocode_address, address)


# --------------------------------------------------
# PUBLIC REPORT TRACKING
# --------------------------------------------------

@app.get("/api/track/{reference_code}")
def track_report(reference_code: str):
    try:
        with psycopg.connect(
            db_url(),
            connect_timeout=10,
        ) as conn:
            row = conn.execute("""
                SELECT
                    reference_code,
                    report_status,
                    submitted_at
                FROM road_reports
                WHERE reference_code::text = %s
            """, (reference_code,)).fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail="No report found for that reference ID.",
            )

        return {
            "reference_code": str(row[0]),
            "status": row[1] or "Reported",
            "submitted_at": (
                row[2].isoformat()
                if row[2] else None
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(
            "PUBLIC TRACK ERROR:",
            repr(exc),
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail="Could not check report status.",
        )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": (
            "AI-Powered Road Damage Detection "
            "and Maintenance Prioritization System"
        ),
    }


# --------------------------------------------------
# SUBMIT A ROAD DAMAGE REPORT
# --------------------------------------------------

@app.post("/api/reports")
async def submit_report(
    latitude: float = Form(...),
    longitude: float = Form(...),
    description: str = Form(""),
    file: UploadFile = File(...),
):
    if not -90 <= latitude <= 90:
        raise HTTPException(
            status_code=400,
            detail="Invalid latitude.",
        )

    if not -180 <= longitude <= 180:
        raise HTTPException(
            status_code=400,
            detail="Invalid longitude.",
        )

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, or WebP image.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The image is empty.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be 10 MB or smaller.",
        )

    temp_path = None
    request_started = time.perf_counter()

    print(
        "[ROAD-REPORT] Upload request received",
        flush=True,
    )

    try:
        with tempfile.NamedTemporaryFile(
            suffix=ALLOWED_TYPES[file.content_type],
            delete=False,
        ) as temp:
            temp.write(content)
            temp_path = Path(temp.name)

        # YOLO detection
        inference_started = time.perf_counter()

        print(
            "[ROAD-REPORT] Starting YOLO inference",
            flush=True,
        )

        result = await run_in_threadpool(
            detect_damage,
            temp_path,
        )

        print(
            "[ROAD-REPORT] YOLO finished in "
            f"{time.perf_counter() - inference_started:.2f}s",
            flush=True,
        )

        detections = result["detections"]

        # Feature extraction and heuristic severity
        features = extract_features(detections)

        severity = assess_severity(
            detections,
            features,
        )

        (
            avg_confidence,
            uncertainty,
            verification_required,
            decision_route,
        ) = confidence_uncertainty(detections)

        # Retrieve road name; other context remains unknown unless
        # a reliable provider supplies it.
        road_context = await run_in_threadpool(
            retrieve_road_context,
            latitude,
            longitude,
        )

        upload_dir = (
            Path(tempfile.gettempdir())
            / "road_damage_uploads"
        )

        upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        saved_image = upload_dir / (
            f"{uuid.uuid4().hex}"
            f"{ALLOWED_TYPES[file.content_type]}"
        )

        shutil.copy2(
            temp_path,
            saved_image,
        )

        print(
            "[ROAD-REPORT] Saving report record",
            flush=True,
        )

        save_started = time.perf_counter()

        report = await run_in_threadpool(
            lambda: save_report(
                image_path=str(saved_image),
                latitude=latitude,
                longitude=longitude,
                description=description.strip() or None,
                detections=detections,
            )
        )

        print(
            "[ROAD-REPORT] Report record saved in "
            f"{time.perf_counter() - save_started:.2f}s",
            flush=True,
        )

        print(
            "[ROAD-REPORT] Updating report and calculating priorities",
            flush=True,
        )

        with psycopg.connect(
            db_url(),
            connect_timeout=10,
        ) as conn:

            ensure_columns(conn)

            conn.execute("""
                UPDATE road_reports
                SET
                    detections = %s::jsonb,
                    detection_count = %s,
                    severity = %s,
                    avg_confidence = %s,
                    uncertainty_level = %s,
                    manual_verification_required = %s,
                    decision_route = %s,
                    damage_size_ratio = %s,
                    damage_extent = %s,
                    damage_density = %s,
                    road_name = %s,
                    road_class = %s,
                    road_importance = %s,
                    location_type = %s,
                    traffic_level = %s,
                    context_source = %s,
                    context_retrieved_at =
                    CASE WHEN %s::text IS NOT NULL THEN NOW()
                    ELSE NULL END,
                    context_status = %s
                WHERE reference_code::text = %s
            """, (
                json.dumps(detections),
                len(detections),
                severity,
                avg_confidence,
                uncertainty,
                verification_required,
                decision_route,
                features["damage_size_ratio"],
                features["damage_extent"],
                features["damage_density"],
                road_context["road_name"],
                road_context.get("road_class"),
                road_context.get("road_importance"),
                road_context.get("location_type"),
                road_context.get("traffic_level"),
                road_context["context_source"],
                road_context["context_source"],
                road_context["context_status"],
                str(report["reference_code"]),
            ))

            # Conventional baseline uses detection count and confidence.
            conventional_score = min(
                100,
                len(detections) * 20
                + (avg_confidence or 0) * 40,
            )

            conn.execute("""
                UPDATE road_reports
                SET conventional_score = %s
                WHERE reference_code::text = %s
            """, (
                conventional_score,
                str(report["reference_code"]),
            ))

            refresh_priorities(conn)

        print(
            "[ROAD-REPORT] Upload completed in "
            f"{time.perf_counter() - request_started:.2f}s",
            flush=True,
        )

        # Public users receive only a submission receipt.
        # AI analysis and priority details remain authority-only.
        return {
            "message": "Report submitted successfully",
            "reference_code": str(report["reference_code"]),
            "submitted_at": report["submitted_at"],
            "status": "Reported",
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(
            "[ROAD-REPORT] SUBMISSION ERROR after "
            f"{time.perf_counter() - request_started:.2f}s: "
            f"{exc!r}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Report submission failed. "
                "Check Render logs."
            ),
        ) from exc

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


# --------------------------------------------------
# AUTHORITY REPORT LIST
# --------------------------------------------------

@app.get("/api/reports")
def list_reports(
    x_authority_pin: str | None = Header(default=None),
):
    require_authority_pin(x_authority_pin)

    try:
        with psycopg.connect(
            db_url(),
            connect_timeout=10,
        ) as conn:

            ensure_columns(conn)

            rows = conn.execute("""
                SELECT
                    report_id,
                    reference_code,
                    latitude,
                    longitude,
                    description,
                    submitted_at,
                    report_status,
                    priority,
                    detection_count,
                    detections,
                    priority_score,
                    severity,
                    avg_confidence,
                    uncertainty_level,
                    manual_verification_required,
                    decision_route,
                    traffic_level,
                    road_importance,
                    location_type,
                    context_status,
                    damage_size_ratio,
                    damage_extent,
                    damage_density,
                    conventional_score,
                    proposed_score,
                    road_name,
                    road_class,
                    context_source,
                    context_retrieved_at
                FROM road_reports
                ORDER BY submitted_at DESC
                LIMIT 100
            """).fetchall()

        reports = []

        for row in rows:
            reports.append({
                "report_id": row[0],
                "reference_code": str(row[1]),
                "latitude": row[2],
                "longitude": row[3],
                "description": row[4],
                "submitted_at": (
                    row[5].isoformat()
                    if row[5] else None
                ),
                "report_status": row[6],
                "priority": row[7],
                "detection_count": row[8],
                "detections": row[9],
                "priority_score": row[10],
                "severity": row[11],
                "avg_confidence": row[12],
                "uncertainty_level": row[13],
                "manual_verification_required": row[14],
                "decision_route": row[15],
                "traffic_level": row[16],
                "road_importance": row[17],
                "location_type": row[18],
                "context_status": row[19],
                "damage_size_ratio": row[20],
                "damage_extent": row[21],
                "damage_density": row[22],
                "conventional_score": row[23],
                "proposed_score": row[24],
                "road_name": row[25],
                "road_class": row[26],
                "context_source": row[27],
                "context_retrieved_at": (
                    row[28].isoformat()
                    if row[28] else None
                ),
            })

        return reports

    except HTTPException:
        raise

    except Exception as exc:
        print(
            "REPORT LIST ERROR:",
            repr(exc),
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail="Could not retrieve reports.",
        )


# --------------------------------------------------
# AUTHORITY ROAD-CONTEXT REFRESH FOR EXISTING REPORTS
# --------------------------------------------------

@app.post("/api/reports/{report_id}/refresh-context")
def refresh_report_context(
    report_id: int,
    x_authority_pin: str | None = Header(default=None),
):
    require_authority_pin(x_authority_pin)
    try:
        with psycopg.connect(db_url(), connect_timeout=10) as conn:
            ensure_columns(conn)
            row = conn.execute(
                "SELECT latitude, longitude FROM road_reports WHERE report_id = %s",
                (report_id,),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Report not found.")

            latitude, longitude = row
            key = (round(float(latitude), 5), round(float(longitude), 5))
            getattr(retrieve_road_context, "cache", {}).pop(key, None)
            context = retrieve_road_context(latitude, longitude)

            conn.execute("""
                UPDATE road_reports
                SET road_name = %s,
                    road_class = %s,
                    road_importance = COALESCE(%s, road_importance),
                    location_type = COALESCE(%s, location_type),
                    traffic_level = COALESCE(%s, traffic_level),
                    context_source = %s,
                    context_retrieved_at =
                        CASE WHEN %s IS NOT NULL THEN NOW()
                             ELSE context_retrieved_at END,
                    context_status = %s
                WHERE report_id = %s
            """, (
                context["road_name"], context.get("road_class"),
                context.get("road_importance"), context.get("location_type"),
                context.get("traffic_level"), context.get("context_source"),
                context.get("context_source"), context.get("context_status"),
                report_id,
            ))
            refresh_priorities(conn)

        return {"report_id": report_id, **context}
    except HTTPException:
        raise
    except Exception as exc:
        print("[ROAD-CONTEXT] Refresh endpoint failed:", repr(exc), flush=True)
        raise HTTPException(status_code=500, detail="Could not refresh road context. Check Render logs.") from exc


# --------------------------------------------------
# AUTHORITY MAINTENANCE STATUS UPDATE
# --------------------------------------------------

class StatusUpdate(BaseModel):
    report_status: str


@app.patch("/api/reports/{report_id}/status")
def update_status(
    report_id: int,
    update: StatusUpdate,
    x_authority_pin: str | None = Header(default=None),
):
    require_authority_pin(x_authority_pin)

    if update.report_status not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail="Invalid report status.",
        )

    try:
        with psycopg.connect(
            db_url(),
            connect_timeout=10,
        ) as conn:

            ensure_columns(conn)

            found = conn.execute("""
                UPDATE road_reports
                SET report_status = %s
                WHERE report_id = %s
                RETURNING report_id
            """, (
                update.report_status,
                report_id,
            )).fetchone()

            if not found:
                raise HTTPException(
                    status_code=404,
                    detail="Report not found.",
                )

            refresh_priorities(conn)

        return {
            "message": "Report status updated",
            "report_id": report_id,
            "report_status": update.report_status,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(
            "STATUS UPDATE ERROR:",
            repr(exc),
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail="Could not update report status.",
        )

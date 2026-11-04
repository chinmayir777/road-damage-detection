from pathlib import Path
import json
import math
import os
import shutil
import tempfile
import time
import threading
import urllib.error
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

        ADD COLUMN IF NOT EXISTS nearby_facility_name
            TEXT,

        ADD COLUMN IF NOT EXISTS nearby_facility_category
            TEXT,

        ADD COLUMN IF NOT EXISTS nearby_facility_distance_m
            DOUBLE PRECISION,

        ADD COLUMN IF NOT EXISTS nearby_facility_osm_url
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

# Shared Nominatim request protection
_nominatim_lock = threading.Lock()
_nominatim_last_request = 0.0
_nominatim_blocked_until = 0.0


def _nominatim_request(params):
    """Call Nominatim with rate limiting and a descriptive User-Agent."""

    global _nominatim_last_request, _nominatim_blocked_until

    with _nominatim_lock:
        now = time.monotonic()

        if now < _nominatim_blocked_until:
            raise HTTPException(
                status_code=503,
                detail="Address lookup is temporarily rate-limited. Please try again later or use GPS.",
            )

        wait = 1.1 - (now - _nominatim_last_request)

        if wait > 0:
            time.sleep(wait)

        _nominatim_last_request = time.monotonic()

    endpoint = params.pop("_endpoint")

    url = (
        "https://nominatim.openstreetmap.org/"
        + endpoint + "?"
        + urllib.parse.urlencode(params)
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "RoadWatch/1.0 "
                "(contact: chinmayir777@gmail.com)"
            ),
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            return json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            with _nominatim_lock:
                _nominatim_blocked_until = time.monotonic() + 60

            print(
                "[ROAD-GEOCODE] Nominatim rate limit reached. "
                "Pausing requests for 60 seconds.",
                flush=True,
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "Address lookup is temporarily rate-limited. "
                    "Please wait before trying again or use GPS."
                ),
            ) from exc

        raise
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


def _overpass_road_context(latitude, longitude):
    """
    Retrieve mapped road ways and nearby facilities from OpenStreetMap.
    Tries more than one public Overpass instance because individual instances
    can be temporarily unavailable or rate-limited. This is not live traffic.
    """
    query = f"""
    [out:json][timeout:15];
    (
      way(around:400,{latitude},{longitude})["highway"~"^(motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|unclassified|residential|living_street|service|road)$"];
      nwr(around:400,{latitude},{longitude})["amenity"~"^(school|hospital|clinic|bus_station|bus_stop|parking)$"];
      nwr(around:400,{latitude},{longitude})["public_transport"~"^(station|platform|stop_position)$"];
      nwr(around:400,{latitude},{longitude})["railway"~"^(station|halt|tram_stop)$"];
      node(around:400,{latitude},{longitude})["highway"~"^(traffic_signals|crossing|stop)$"];
    );
    out center tags;
    """

    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
    ]
    errors = []

    for endpoint in endpoints:
        url = endpoint + "?" + urllib.parse.urlencode({"data": query})
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "RoadWatch/1.0 (contact: chinmayir777@gmail.com)",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=14) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload.get("elements", [])
        except Exception as exc:
            errors.append(f"{endpoint}: {exc!r}")
            print(
                "[ROAD-CONTEXT] Overpass endpoint failed:",
                endpoint,
                repr(exc),
                flush=True,
            )

    raise RuntimeError("All Overpass endpoints failed: " + " | ".join(errors))


def _photon_lookup(params, timeout=10):
    """Query Photon (OpenStreetMap-based geocoder) as an Overpass fallback."""
    url = "https://photon.komoot.io/" + ("reverse" if "lat" in params and "lon" in params and "q" not in params else "api/")
    url += "?" + urllib.parse.urlencode(params, doseq=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "RoadWatch/1.0 (contact: chinmayir777@gmail.com)",
            "Accept": "application/geo+json, application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def retrieve_road_context(latitude, longitude):
    """Retrieve road details and the nearest named OSM facility.

    Traffic is not inferred from map data. Authority-assessed traffic remains
    stored separately and is preserved by the refresh endpoint.
    """
    latitude = float(latitude)
    longitude = float(longitude)
    result = {
        "road_name": "Not available",
        "road_class": "Not available",
        "road_importance": "Unknown",
        "location_type": "General",
        "nearby_facility_name": "No mapped facility found",
        "nearby_facility_category": "Not available",
        "nearby_facility_distance_m": None,
        "nearby_facility_osm_url": None,
        "traffic_level": "Unavailable",
        "context_source": "OpenStreetMap-based geocoding",
        "context_status": "Road context lookup pending",
    }

    cache_key = (round(latitude, 5), round(longitude, 5))
    if not hasattr(retrieve_road_context, "cache"):
        retrieve_road_context.cache = {}
    if cache_key in retrieve_road_context.cache:
        return dict(retrieve_road_context.cache[cache_key])

    road_classes = {
        "motorway", "motorway_link", "trunk", "trunk_link",
        "primary", "primary_link", "secondary", "secondary_link",
        "tertiary", "tertiary_link", "unclassified", "residential",
        "living_street", "service", "road", "pedestrian", "cycleway",
        "footway", "path",
    }
    errors = []
    address = {}
    facility_candidates = []

    def distance_m(lat, lon):
        dy = (float(lat) - latitude) * 111320.0
        dx = (float(lon) - longitude) * 111320.0 * max(
            0.01, abs(math.cos(math.radians(latitude)))
        )
        return math.hypot(dx, dy)

    def classify_facility(props):
        key = str(props.get("osm_key") or "").lower()
        value = str(props.get("osm_value") or "").lower()
        if key == "amenity" and value in {
            "school", "kindergarten", "college", "university"
        }:
            return "Education", "School/Hospital"
        if key == "amenity" and value in {
            "hospital", "clinic", "doctors"
        }:
            return "Healthcare", "School/Hospital"
        if (key == "highway" and value == "bus_stop") or (
            key == "amenity" and value in {"bus_station", "taxi"}
        ) or (key == "public_transport" and value in {
            "station", "platform", "stop_position"
        }):
            return "Public transport", "Junction/Transit"
        if key == "railway" and value in {"station", "halt", "tram_stop"}:
            return "Rail/Metro station", "Junction/Transit"
        if key == "highway" and value in {
            "traffic_signals", "crossing", "stop"
        }:
            return "Traffic control", "Junction/Transit"
        return None, None

    def add_photon_features(payload, source_label):
        for feature in payload.get("features") or []:
            props = feature.get("properties") or {}
            category, location_group = classify_facility(props)
            if not category:
                continue
            name = str(props.get("name") or "").strip()
            if not name:
                # The user asked for a specific facility. Do not manufacture a name.
                continue
            coords = (feature.get("geometry") or {}).get("coordinates") or []
            if len(coords) < 2:
                continue
            distance = distance_m(coords[1], coords[0])
            if distance > 1500:
                continue
            osm_type = str(props.get("osm_type") or "").upper()
            osm_id = props.get("osm_id")
            osm_kind = {"N": "node", "W": "way", "R": "relation"}.get(osm_type)
            osm_url = (
                f"https://www.openstreetmap.org/{osm_kind}/{osm_id}"
                if osm_kind and osm_id else None
            )
            facility_candidates.append({
                "distance": distance,
                "name": name,
                "category": category,
                "location_group": location_group,
                "osm_url": osm_url,
                "source": source_label,
            })

    # 1. Nominatim: road name and any available highway tags.
    try:
        reverse_data = _nominatim_request({
            "_endpoint": "reverse", "format": "jsonv2",
            "lat": latitude, "lon": longitude, "zoom": 18,
            "addressdetails": 1, "extratags": 1,
        }) or {}
        address = reverse_data.get("address") or {}
        for field in ("road", "residential", "pedestrian", "path", "footway", "cycleway"):
            if address.get(field):
                result["road_name"] = str(address[field])
                break
        tags = reverse_data.get("extratags") or {}
        candidate_class = str(tags.get("highway") or reverse_data.get("type") or "").lower()
        category = str(reverse_data.get("category") or reverse_data.get("class") or "").lower()
        if candidate_class in road_classes and (category in {"highway", ""} or tags.get("highway")):
            result["road_class"] = candidate_class
    except Exception as exc:
        errors.append("Nominatim reverse")
        print("[ROAD-CONTEXT] Nominatim reverse failed:", repr(exc), flush=True)

    # 2. Photon reverse: road classification and any named nearby POIs returned.
    try:
        photon = _photon_lookup({"lat": latitude, "lon": longitude, "lang": "en", "limit": 20})
        for feature in photon.get("features") or []:
            props = feature.get("properties") or {}
            osm_key = str(props.get("osm_key") or "").lower()
            osm_value = str(props.get("osm_value") or "").lower()
            if osm_key == "highway" and osm_value in road_classes:
                if result["road_name"] == "Not available" and props.get("name"):
                    result["road_name"] = str(props["name"])
                if result["road_class"] == "Not available":
                    result["road_class"] = osm_value
        add_photon_features(photon, "Photon reverse")
    except Exception as exc:
        errors.append("Photon reverse")
        print("[ROAD-CONTEXT] Photon reverse failed:", repr(exc), flush=True)

    # 3. If needed, search the named road to recover its highway classification.
    if result["road_name"] != "Not available" and result["road_class"] == "Not available":
        try:
            city_hint = ", ".join(
                str(address.get(k)) for k in ("city", "town", "suburb", "state")
                if address.get(k)
            )
            query_text = result["road_name"] + (", " + city_hint if city_hint else "")
            matches = _nominatim_request({
                "_endpoint": "search", "q": query_text, "format": "jsonv2",
                "limit": 5, "addressdetails": 1, "extratags": 1,
            }) or []
            for item in matches:
                tags = item.get("extratags") or {}
                highway = str(tags.get("highway") or item.get("type") or "").lower()
                item_category = str(item.get("category") or item.get("class") or "").lower()
                if highway in road_classes and (item_category in {"highway", ""} or tags.get("highway")):
                    result["road_class"] = highway
                    if item.get("name"):
                        result["road_name"] = str(item["name"])
                    break
        except Exception as exc:
            errors.append("Nominatim road search")
            print("[ROAD-CONTEXT] Nominatim road search failed:", repr(exc), flush=True)

    # 4. Overpass is attempted for named facilities, but is not required.
    elements = []
    try:
        elements = _overpass_road_context(latitude, longitude)
    except Exception as exc:
        errors.append("Overpass")
        print("[ROAD-CONTEXT] Overpass unavailable; using Photon facility lookup:", repr(exc), flush=True)

    for element in elements:
        tags = element.get("tags") or {}
        center = element.get("center") or {}
        lat = element.get("lat", center.get("lat"))
        lon = element.get("lon", center.get("lon"))
        name = str(tags.get("name") or "").strip()
        if lat is None or lon is None or not name:
            continue
        props = {
            "osm_key": next((k for k in ("amenity", "highway", "railway", "public_transport") if tags.get(k)), ""),
            "osm_value": next((tags.get(k) for k in ("amenity", "highway", "railway", "public_transport") if tags.get(k)), ""),
            "osm_type": str(element.get("type") or "").upper(),
            "osm_id": element.get("id"),
            "name": name,
        }
        facility_category, location_group = classify_facility(props)
        if facility_category:
            kind = {"N": "node", "W": "way", "R": "relation"}.get(props["osm_type"])
            osm_url = f"https://www.openstreetmap.org/{kind}/{props['osm_id']}" if kind and props["osm_id"] else None
            facility_candidates.append({
                "distance": distance_m(lat, lon),
                "name": name,
                "category": facility_category,
                "location_group": location_group,
                "osm_url": osm_url,
                "source": "Overpass",
            })

    # 5. Photon reverse lookup with OSM tag filters: request named nearby
    # facilities by category, rather than relying on broad text search ranking.
    photon_tag_groups = (
        ["amenity:school", "amenity:kindergarten", "amenity:college", "amenity:university"],
        ["amenity:hospital", "amenity:clinic", "amenity:doctors"],
        ["highway:bus_stop", "amenity:bus_station", "railway:station", "railway:halt", "railway:tram_stop", "public_transport:station", "public_transport:platform"],
        ["highway:traffic_signals", "highway:crossing", "highway:stop"],
    )
    for tag_group in photon_tag_groups:
        try:
            payload = _photon_lookup({
                "lat": latitude,
                "lon": longitude,
                "radius": 1.5,
                "limit": 20,
                "lang": "en",
                "osm_tag": tag_group,
            }, timeout=8)
            add_photon_features(payload, "Photon nearby search")
        except Exception as exc:
            errors.append("Photon nearby search")
            print("[ROAD-CONTEXT] Photon nearby facility lookup failed:", repr(exc), flush=True)

    # Derive importance only from a retrieved OSM highway classification.
    road_class = result["road_class"]
    if road_class in {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"}:
        result["road_importance"] = "Major"
    elif road_class in {"secondary", "secondary_link", "tertiary", "tertiary_link"}:
        result["road_importance"] = "Collector"
    elif road_class in {"unclassified", "residential", "living_street", "service", "road", "pedestrian", "cycleway", "footway", "path"}:
        result["road_importance"] = "Local"

    # Select the nearest named facility. Never fabricate a facility name.
    if facility_candidates:
        facility_candidates.sort(key=lambda item: item["distance"])
        nearest = facility_candidates[0]
        result["nearby_facility_name"] = nearest["name"]
        result["nearby_facility_category"] = nearest["category"]
        result["nearby_facility_distance_m"] = round(nearest["distance"])
        result["nearby_facility_osm_url"] = nearest["osm_url"]
        result["location_type"] = nearest["location_group"]

    result["traffic_level"] = "Unavailable"
    used = ["Nominatim"]
    if result["road_class"] != "Not available" or facility_candidates:
        used.append("Photon/OSM fallback")
    if elements:
        used.append("Overpass")
    result["context_source"] = " + ".join(used)

    if result["road_name"] != "Not available" or result["road_class"] != "Not available":
        result["context_status"] = "Road details retrieved; nearest named mapped facility checked; traffic assessed separately by authority"
    else:
        result["context_status"] = "Road details not found in available map data; traffic assessed separately by authority"
    if result["nearby_facility_name"] == "No mapped facility found":
        result["context_status"] += "; no named facility returned within 1.5 km"
    if errors:
        result["context_status"] += "; one or more map providers were unavailable"

    print("[ROAD-CONTEXT] " + json.dumps({
        "coordinates": cache_key,
        "road_name": result["road_name"],
        "road_class": result["road_class"],
        "road_importance": result["road_importance"],
        "nearby_facility_name": result["nearby_facility_name"],
        "nearby_facility_category": result["nearby_facility_category"],
        "nearby_facility_distance_m": result["nearby_facility_distance_m"],
        "location_type": result["location_type"],
        "traffic_level": result["traffic_level"],
        "status": result["context_status"],
    }), flush=True)

    retrieve_road_context.cache[cache_key] = dict(result)
    return result


# --------------------------------------------------
# DAMAGE FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(detections):
    """Calculate approximate damage features after suppressing duplicate boxes."""
    if not detections:
        return {
            "damage_size_ratio": 0.0,
            "damage_extent": 0.0,
            "damage_density": 0.0,
        }

    candidates = []
    for detection in detections:
        box = detection.get("bbox_xyxy") or []
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = map(float, box)
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append({
            "box": [x1, y1, x2, y2],
            "confidence": float(detection.get("confidence", 0) or 0),
        })

    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    unique = []

    for item in candidates:
        x1, y1, x2, y2 = item["box"]
        duplicate = False
        for kept in unique:
            a1, b1, a2, b2 = kept["box"]
            ix1, iy1 = max(x1, a1), max(y1, b1)
            ix2, iy2 = min(x2, a2), min(y2, b2)
            intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (a2 - a1) * (b2 - b1)
            union = area1 + area2 - intersection
            if union > 0 and intersection / union >= 0.50:
                duplicate = True
                break
        if not duplicate:
            unique.append(item)

    if not unique:
        return {
            "damage_size_ratio": 0.0,
            "damage_extent": 0.0,
            "damage_density": 0.0,
        }

    boxes = [item["box"] for item in unique]
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    total_area = sum(areas)
    min_x = min(b[0] for b in boxes)
    min_y = min(b[1] for b in boxes)
    max_x = max(b[2] for b in boxes)
    max_y = max(b[3] for b in boxes)
    region_area = max(1.0, (max_x - min_x) * (max_y - min_y))

    return {
        "damage_size_ratio": round(max(areas) / max(1.0, total_area), 4),
        "damage_extent": round(min(1.0, total_area / region_area), 4),
        "damage_density": round(len(unique) / max(1.0, region_area / 10000.0), 4),
    }



def assess_severity(detections, features):
    """
    Conservative severity assessment.
    No valid detections = Unconfirmed.
    Low-confidence detections = Minor.
    Higher severity requires stronger model evidence.
    """

    if not detections:
        return "Unconfirmed"

    # Sort by confidence, highest first
    ordered = sorted(
        detections,
        key=lambda d: float(d.get("confidence", 0) or 0),
        reverse=True,
    )

    # Remove overlapping duplicate detections
    unique = []

    for detection in ordered:
        box = detection.get("bbox_xyxy") or []

        if len(box) != 4:
            continue

        x1, y1, x2, y2 = map(float, box)

        if x2 <= x1 or y2 <= y1:
            continue

        duplicate = False

        for existing in unique:
            other = existing.get("bbox_xyxy") or []

            if len(other) != 4:
                continue

            a1, b1, a2, b2 = map(float, other)

            intersection = (
                max(0, min(x2, a2) - max(x1, a1))
                * max(0, min(y2, b2) - max(y1, b1))
            )

            area1 = (x2 - x1) * (y2 - y1)
            area2 = (a2 - a1) * (b2 - b1)
            union = area1 + area2 - intersection

            if union > 0 and intersection / union >= 0.50:
                duplicate = True
                break

        if not duplicate:
            unique.append(detection)

    if not unique:
        return "Unconfirmed"

    # Low-confidence detections cannot justify higher severity
    confidences = [
        float(d.get("confidence", 0) or 0)
        for d in unique
    ]

    average_confidence = sum(confidences) / len(confidences)

    if average_confidence < 0.25:
        return "Minor"

    # Only reasonably confident detections count toward
    # Moderate or Severe classifications.
    confirmed = [
        d for d in unique
        if float(d.get("confidence", 0) or 0) >= 0.45
    ]

    if not confirmed:
        return "Minor"

    pothole_count = sum(
        "pothole" in str(
            d.get("damage_type", "")
        ).lower()
        for d in confirmed
    )

    if len(confirmed) >= 3 and pothole_count >= 1:
        return "Severe"

    if pothole_count >= 1 or len(confirmed) >= 2:
        return "Moderate"

    return "Minor"

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
    """Recalculate severity and priority, applying confidence safety limits."""
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
                    (CASE traffic_level
                        WHEN 'High' THEN 3 WHEN 'Medium' THEN 2
                        WHEN 'Low' THEN 1 ELSE 0 END)
                    +
                    (CASE road_importance
                        WHEN 'Major' THEN 3 WHEN 'Collector' THEN 2
                        WHEN 'Local' THEN 1 ELSE 0 END)
                    +
                    (CASE location_type
                        WHEN 'School/Hospital' THEN 3
                        WHEN 'Junction/Transit' THEN 2
                        WHEN 'General' THEN 1 ELSE 0 END)
            END AS context_points,
            COALESCE(avg_confidence, 0),
            CASE WHEN manual_verification_required THEN 1 ELSE 0 END,
            COALESCE(damage_extent, 0),
            decision_route,
            detections
        FROM road_reports
        WHERE report_status <> 'Repaired'
    """).fetchall()

    ranking_rows = []

    for row in rows:
        report_id = row[0]
        detections = row[7] or []
        severity = assess_severity(detections, {})
        severity_value = {
            "Unconfirmed": 0,
            "Minor": 1,
            "Moderate": 2,
            "Severe": 3,
        }.get(severity, 0)

        conn.execute(
            "UPDATE road_reports SET severity = %s WHERE report_id = %s",
            (severity, report_id),
        )

        ranking_rows.append([
            report_id,
            severity_value,
            float(row[2] or 0),
            float(row[3] or 0),
            bool(row[4]),
            float(row[5] or 0),
            row[6] or "Review",
            severity,
        ])

    scores = topsis_rank(ranking_rows)

    for row in ranking_rows:
        report_id = row[0]
        confidence = row[3]
        verification_required = row[4]
        severity = row[7]
        score = scores.get(report_id, 0)

        # Relative TOPSIS rank cannot override these absolute safety rules.
        if severity == "Unconfirmed" or confidence < 0.15:
            priority = "Low"
            score = min(score, 33)
        elif verification_required or confidence < 0.65:
            priority = "Medium" if confidence >= 0.25 else "Low"
            score = min(score, 66 if priority == "Medium" else 33)
        elif severity == "Severe":
            priority = "High"
        elif severity == "Moderate":
            priority = "Medium"
            score = min(score, 66)
        else:
            priority = "Low"
            score = min(score, 33)

        route = row[6] or "Review"
        if route != "Automatic candidate":
            route = (
                "Human verification"
                if route == "Human verification"
                else "Review"
            )

        conn.execute("""
            UPDATE road_reports
            SET priority_score = %s,
                proposed_score = %s,
                priority = %s,
                decision_route = %s
            WHERE report_id = %s
        """, (score, score, priority, route, report_id))
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
                    nearby_facility_name = %s,
                    nearby_facility_category = %s,
                    nearby_facility_distance_m = %s,
                    nearby_facility_osm_url = %s,
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
                road_context.get("nearby_facility_name"),
                road_context.get("nearby_facility_category"),
                road_context.get("nearby_facility_distance_m"),
                road_context.get("nearby_facility_osm_url"),
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
                    nearby_facility_name,
                    nearby_facility_category,
                    nearby_facility_distance_m,
                    nearby_facility_osm_url,
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
                "nearby_facility_name": row[27],
                "nearby_facility_category": row[28],
                "nearby_facility_distance_m": row[29],
                "nearby_facility_osm_url": row[30],
                "context_source": row[31],
                "context_retrieved_at": (
                    row[32].isoformat()
                    if row[32] else None
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
                SET road_name = CASE
                        WHEN %s = 'Not available' THEN road_name
                        ELSE %s
                    END,
                    road_class = CASE
                        WHEN %s = 'Not available' THEN road_class
                        ELSE %s
                    END,
                    road_importance = CASE
                        WHEN %s = 'Unknown' THEN road_importance
                        ELSE %s
                    END,
                    location_type = CASE
                        WHEN %s = 'General' THEN location_type
                        ELSE %s
                    END,
                    nearby_facility_name = CASE
                        WHEN %s = 'No mapped facility found' THEN nearby_facility_name
                        ELSE %s
                    END,
                    nearby_facility_category = CASE
                        WHEN %s = 'Not available' THEN nearby_facility_category
                        ELSE %s
                    END,
                    nearby_facility_distance_m = COALESCE(%s, nearby_facility_distance_m),
                    nearby_facility_osm_url = COALESCE(%s, nearby_facility_osm_url),
                    traffic_level = CASE
                        WHEN %s = 'Unavailable' THEN traffic_level
                        ELSE COALESCE(%s, traffic_level)
                    END,
                    context_source = %s,
                    context_retrieved_at =
                        CASE WHEN %s::text IS NOT NULL THEN NOW()
                             ELSE context_retrieved_at END,
                    context_status = %s
                WHERE report_id = %s
            """, (
                context["road_name"], context["road_name"],
                context.get("road_class"), context.get("road_class"),
                context.get("road_importance"), context.get("road_importance"),
                context.get("location_type"), context.get("location_type"),
                context.get("nearby_facility_name"), context.get("nearby_facility_name"),
                context.get("nearby_facility_category"), context.get("nearby_facility_category"),
                context.get("nearby_facility_distance_m"), context.get("nearby_facility_osm_url"),
                context.get("traffic_level"), context.get("traffic_level"),
                context.get("context_source"), context.get("context_source"),
                context.get("context_status"), report_id,
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

class TrafficUpdate(BaseModel):
    traffic_level: str


@app.patch("/api/reports/{report_id}/traffic")
def update_traffic_level(
    report_id: int,
    update: TrafficUpdate,
    x_authority_pin: str | None = Header(default=None),
):
    """Save an authority-assessed traffic level for a reported location."""
    require_authority_pin(x_authority_pin)

    if update.traffic_level not in {"High", "Medium", "Low"}:
        raise HTTPException(
            status_code=400,
            detail="Traffic level must be High, Medium, or Low.",
        )

    try:
        with psycopg.connect(db_url(), connect_timeout=10) as conn:
            ensure_columns(conn)

            found = conn.execute("""
                UPDATE road_reports
                SET traffic_level = %s,
                    context_status = CASE
                        WHEN context_status IS NULL OR context_status = ''
                        THEN 'Traffic level assessed by authority'
                        ELSE context_status || '; traffic level assessed by authority'
                    END
                WHERE report_id = %s
                RETURNING report_id
            """, (update.traffic_level, report_id)).fetchone()

            if not found:
                raise HTTPException(
                    status_code=404,
                    detail="Report not found.",
                )

            # Recalculate TOPSIS-style scores using the newly assigned traffic.
            refresh_priorities(conn)

        return {
            "message": "Authority-assessed traffic level saved.",
            "report_id": report_id,
            "traffic_level": update.traffic_level,
        }

    except HTTPException:
        raise
    except Exception as exc:
        print("[TRAFFIC] Update failed:", repr(exc), flush=True)
        raise HTTPException(
            status_code=500,
            detail="Could not save traffic level.",
        ) from exc


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

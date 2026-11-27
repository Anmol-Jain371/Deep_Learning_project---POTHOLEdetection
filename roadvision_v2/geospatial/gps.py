"""
Phase 11: Genuine Optional External GPS Telemetry Ingestion & Video-Timestamp Synchronization.

Supports CSV, GPX (with strict XXE protection), and GeoJSON telemetry with:
- Strict WGS84 coordinate bounds checking (lat in [-90, 90], lon in [-180, 180]).
- Explicit ISO-8601 timezone-aware timestamps or explicit video-relative numeric timestamps.
- Duplicate timestamp handling and out-of-order monotonic tracking.
- Source-video timestamp interpolation within `max_sync_gap_sec` (never wall-clock time;
  never extrapolates outside telemetry bounds by default).
- Honest provenance distinguishing approximate vehicle/camera GPS from exact hazard location.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from roadvision_v2.config import GeospatialConfig

from .matching import initial_bearing_deg

_SUPPORTED_EXTS = {".csv", ".gpx", ".geojson", ".json"}


def _parse_iso_epoch(raw_ts: str) -> float:
    text = str(raw_ts or "").strip()
    if not text:
        raise ValueError("Empty timestamp string")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError(
            f"ISO-8601 timestamp '{raw_ts}' is missing explicit timezone offset (e.g., 'Z' or '+05:30')."
        )
    return float(dt.astimezone(timezone.utc).timestamp())


def _safe_float(val: Any) -> Optional[float]:
    if val is None or isinstance(val, bool):
        return None
    text = str(val).strip()
    if text == "":
        return None
    f = float(text)
    if not math.isfinite(f):
        return None
    return f


def _validate_coord(lat: Optional[float], lon: Optional[float], row_idx: int) -> Tuple[float, float]:
    if lat is None or lon is None:
        raise ValueError(f"Row {row_idx}: missing latitude or longitude")
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Row {row_idx}: latitude {lat} out of WGS84 bounds [-90, 90]")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Row {row_idx}: longitude {lon} out of WGS84 bounds [-180, 180]")
    return float(lat), float(lon)


def _parse_raw_timestamp(
    raw_val: Any,
    time_mode: str,
    row_idx: int,
) -> Tuple[str, float, str]:
    """Return (resolved_mode, numeric_or_epoch_val, raw_str)."""
    raw_str = str(raw_val or "").strip()
    if not raw_str:
        raise ValueError(f"Row {row_idx}: missing timestamp")

    if time_mode == "video_relative":
        val = _safe_float(raw_str)
        if val is None:
            raise ValueError(f"Row {row_idx}: expected numeric video-relative timestamp, got {raw_str!r}")
        return "video_relative", float(val), raw_str

    if time_mode == "iso8601":
        return "iso8601", _parse_iso_epoch(raw_str), raw_str

    # auto mode
    try:
        num = float(raw_str)
        if math.isfinite(num):
            return "video_relative", float(num), raw_str
    except ValueError:
        pass
    return "iso8601", _parse_iso_epoch(raw_str), raw_str


def _extract_csv_rows(text: str, time_mode: str, max_rows: int) -> Tuple[List[Dict[str, Any]], str]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("GPS CSV file has no header row")

    col_map = {str(c).strip().lower(): c for c in reader.fieldnames if c}

    def _pick(*names: str) -> Optional[str]:
        for n in names:
            if n in col_map:
                return col_map[n]
        return None

    ts_col = _pick("timestamp", "time", "timestamp_sec", "time_sec", "utc_time")
    lat_col = _pick("latitude", "lat")
    lon_col = _pick("longitude", "lon", "lng")
    alt_col = _pick("altitude", "alt", "ele", "elevation")
    spd_col = _pick("speed", "speed_mps")
    acc_col = _pick("horizontal_accuracy", "accuracy", "accuracy_m", "hdop_m")
    hdg_col = _pick("heading", "heading_deg", "bearing", "course")

    if not ts_col or not lat_col or not lon_col:
        raise ValueError(
            "GPS CSV must contain timestamp, latitude, and longitude columns."
        )

    detected_mode: Optional[str] = None
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(reader, start=1):
        if idx > max_rows:
            raise ValueError(f"GPS CSV exceeds max_gps_rows={max_rows}")
        lat, lon = _validate_coord(_safe_float(row.get(lat_col)), _safe_float(row.get(lon_col)), idx)
        r_mode, t_val, raw_str = _parse_raw_timestamp(row.get(ts_col), time_mode, idx)
        if detected_mode is None:
            detected_mode = r_mode
        elif r_mode != detected_mode:
            raise ValueError(f"Row {idx}: mixed timestamp formats ({detected_mode} vs {r_mode})")
        rows.append(
            {
                "raw_timestamp": raw_str,
                "raw_time_val": t_val,
                "lat": lat,
                "lon": lon,
                "altitude": _safe_float(row.get(alt_col)) if alt_col else None,
                "speed_mps": _safe_float(row.get(spd_col)) if spd_col else None,
                "horizontal_accuracy_m": _safe_float(row.get(acc_col)) if acc_col else None,
                "heading_deg": _safe_float(row.get(hdg_col)) if hdg_col else None,
            }
        )
    if not rows:
        raise ValueError("GPS CSV contains no data rows")
    return rows, detected_mode or "video_relative"


def _extract_gpx_rows(text: str, time_mode: str, max_rows: int) -> Tuple[List[Dict[str, Any]], str]:
    low = text.lower()
    if "<!doctype" in low or "<!entity" in low:
        raise ValueError("GPX XML DOCTYPE/ENTITY declarations are forbidden (XXE protection)")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"Malformed GPX XML: {exc}") from exc

    def _find_child(elem: ET.Element, local_name: str) -> Optional[ET.Element]:
        for child in elem:
            tag = child.tag.split("}", 1)[-1]
            if tag == local_name:
                return child
        return None

    pts: List[ET.Element] = []
    for elem in root.iter():
        tag = elem.tag.split("}", 1)[-1]
        if tag in ("trkpt", "rtept", "wpt"):
            pts.append(elem)
        if len(pts) > max_rows:
            raise ValueError(f"GPX exceeds max_gps_rows={max_rows}")

    if not pts:
        raise ValueError("GPX file contains no track/waypoint elements (<trkpt>, <rtept>, or <wpt>)")

    rows: List[Dict[str, Any]] = []
    detected_mode: Optional[str] = None
    for idx, pt in enumerate(pts, start=1):
        lat, lon = _validate_coord(_safe_float(pt.attrib.get("lat")), _safe_float(pt.attrib.get("lon")), idx)
        time_el = _find_child(pt, "time")
        if time_el is None or not (time_el.text or "").strip():
            raise ValueError(f"GPX point {idx} is missing required <time> element")
        r_mode, t_val, raw_str = _parse_raw_timestamp(time_el.text, time_mode, idx)
        if detected_mode is None:
            detected_mode = r_mode
        elif r_mode != detected_mode:
            raise ValueError(f"GPX point {idx}: mixed timestamp formats")

        ele_el = _find_child(pt, "ele")
        spd_el = _find_child(pt, "speed")
        hdop_el = _find_child(pt, "hdop")
        rows.append(
            {
                "raw_timestamp": raw_str,
                "raw_time_val": t_val,
                "lat": lat,
                "lon": lon,
                "altitude": _safe_float(ele_el.text) if ele_el is not None else None,
                "speed_mps": _safe_float(spd_el.text) if spd_el is not None else None,
                "horizontal_accuracy_m": _safe_float(hdop_el.text) if hdop_el is not None else None,
                "heading_deg": None,
            }
        )
    return rows, detected_mode or "iso8601"


def _extract_geojson_rows(text: str, time_mode: str, max_rows: int) -> Tuple[List[Dict[str, Any]], str]:
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("GeoJSON root must be an object")
    features = obj.get("features") if obj.get("type") == "FeatureCollection" else ([obj] if obj.get("type") == "Feature" else [])
    if not features:
        raise ValueError("GeoJSON contains no features")

    rows: List[Dict[str, Any]] = []
    detected_mode: Optional[str] = None
    idx = 0
    for feat in features:
        geom = feat.get("geometry") or {}
        props = feat.get("properties") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if gtype == "Point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
            idx += 1
            if idx > max_rows:
                raise ValueError(f"GeoJSON exceeds max_gps_rows={max_rows}")
            lon, lat = _validate_coord(_safe_float(coords[1]), _safe_float(coords[0]), idx)
            alt = _safe_float(coords[2]) if len(coords) >= 3 else None
            ts_raw = props.get("timestamp") or props.get("time") or props.get("timestamp_sec")
            r_mode, t_val, raw_str = _parse_raw_timestamp(ts_raw, time_mode, idx)
            detected_mode = detected_mode or r_mode
            rows.append(
                {
                    "raw_timestamp": raw_str,
                    "raw_time_val": t_val,
                    "lat": lat,
                    "lon": lon,
                    "altitude": alt,
                    "speed_mps": _safe_float(props.get("speed_mps") or props.get("speed")),
                    "horizontal_accuracy_m": _safe_float(props.get("horizontal_accuracy_m") or props.get("horizontal_accuracy")),
                    "heading_deg": _safe_float(props.get("heading_deg") or props.get("heading")),
                }
            )
        elif gtype == "LineString" and isinstance(coords, list):
            ts_arr = props.get("timestamps") or props.get("coordTimes") or []
            if len(ts_arr) != len(coords):
                raise ValueError("GeoJSON LineString requires 'timestamps' or 'coordTimes' matching coordinates length")
            for c, t_raw in zip(coords, ts_arr):
                idx += 1
                if idx > max_rows:
                    raise ValueError(f"GeoJSON exceeds max_gps_rows={max_rows}")
                lon, lat = _validate_coord(_safe_float(c[1]), _safe_float(c[0]), idx)
                alt = _safe_float(c[2]) if len(c) >= 3 else None
                r_mode, t_val, raw_str = _parse_raw_timestamp(t_raw, time_mode, idx)
                detected_mode = detected_mode or r_mode
                rows.append(
                    {
                        "raw_timestamp": raw_str,
                        "raw_time_val": t_val,
                        "lat": lat,
                        "lon": lon,
                        "altitude": alt,
                        "speed_mps": None,
                        "horizontal_accuracy_m": _safe_float(props.get("horizontal_accuracy_m")),
                        "heading_deg": None,
                    }
                )
    if not rows:
        raise ValueError("GeoJSON contained no valid timestamped Point or LineString features")
    return rows, detected_mode or "iso8601"


def parse_gps_telemetry(
    raw_bytes: bytes,
    filename: str,
    *,
    time_mode: str = "auto",
    sync_offset_sec: float = 0.0,
    video_start_iso: Optional[str] = None,
    max_file_bytes: int = 10 * 1024 * 1024,
    max_rows: int = 100000,
) -> Dict[str, Any]:
    """Parse and validate CSV, GPX, or GeoJSON external GPS telemetry."""
    if not raw_bytes:
        raise ValueError("Uploaded GPS telemetry file is empty")
    if len(raw_bytes) > max_file_bytes:
        raise ValueError(f"GPS file exceeds max size of {max_file_bytes} bytes")

    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in _SUPPORTED_EXTS:
        raise ValueError(f"Unsupported GPS file extension '{ext}'; expected {sorted(_SUPPORTED_EXTS)}")

    if time_mode not in ("auto", "iso8601", "video_relative"):
        raise ValueError(f"Invalid time_mode '{time_mode}'")

    text = raw_bytes.decode("utf-8-sig")
    if ext == ".csv":
        raw_rows, resolved_mode = _extract_csv_rows(text, time_mode, max_rows)
        fmt = "csv"
    elif ext == ".gpx":
        raw_rows, resolved_mode = _extract_gpx_rows(text, time_mode, max_rows)
        fmt = "gpx"
    else:
        raw_rows, resolved_mode = _extract_geojson_rows(text, time_mode, max_rows)
        fmt = "geojson"

    # Normalize timestamps onto video-relative seconds
    base_epoch: Optional[float] = None
    if resolved_mode == "iso8601":
        if video_start_iso:
            base_epoch = _parse_iso_epoch(video_start_iso)
        else:
            base_epoch = min(r["raw_time_val"] for r in raw_rows)

    out_of_order_count = 0
    prev_t: Optional[float] = None
    normalized: List[Dict[str, Any]] = []
    for r in raw_rows:
        if resolved_mode == "iso8601":
            ts_sec = (r["raw_time_val"] - float(base_epoch)) + float(sync_offset_sec)
        else:
            ts_sec = r["raw_time_val"] + float(sync_offset_sec)
        if prev_t is not None and ts_sec < prev_t:
            out_of_order_count += 1
        prev_t = ts_sec
        normalized.append(
            {
                "timestamp_sec": round(float(ts_sec), 6),
                "raw_timestamp": r["raw_timestamp"],
                "lat": round(r["lat"], 7),
                "lon": round(r["lon"], 7),
                "altitude": round(r["altitude"], 3) if r["altitude"] is not None else None,
                "speed_mps": round(r["speed_mps"], 3) if r["speed_mps"] is not None else None,
                "horizontal_accuracy_m": (
                    round(r["horizontal_accuracy_m"], 3)
                    if r["horizontal_accuracy_m"] is not None
                    else None
                ),
                "heading_deg": round(r["heading_deg"] % 360.0, 2) if r["heading_deg"] is not None else None,
            }
        )

    # Sort monotonically and deduplicate exact duplicate timestamps deterministically
    normalized.sort(key=lambda s: s["timestamp_sec"])
    deduped: List[Dict[str, Any]] = []
    duplicate_count = 0
    for s in normalized:
        if deduped and abs(s["timestamp_sec"] - deduped[-1]["timestamp_sec"]) <= 1e-6:
            duplicate_count += 1
            # Keep the sample with better (smaller) accuracy if available, otherwise latest
            prev_acc = deduped[-1].get("horizontal_accuracy_m")
            curr_acc = s.get("horizontal_accuracy_m")
            if curr_acc is not None and (prev_acc is None or curr_acc <= prev_acc):
                deduped[-1] = s
        else:
            deduped.append(s)

    # Compute headings from consecutive GPS positions when heading_deg is missing
    for idx, s in enumerate(deduped):
        if s["heading_deg"] is None and len(deduped) >= 2:
            if idx < len(deduped) - 1:
                nxt = deduped[idx + 1]
                s["heading_deg"] = round(initial_bearing_deg(s["lat"], s["lon"], nxt["lat"], nxt["lon"]), 2)
            else:
                prv = deduped[idx - 1]
                s["heading_deg"] = round(initial_bearing_deg(prv["lat"], prv["lon"], s["lat"], s["lon"]), 2)
        s["sample_idx"] = idx

    return {
        "source_format": fmt,
        "time_mode": resolved_mode,
        "base_epoch_sec": base_epoch,
        "sync_offset_sec": float(sync_offset_sec),
        "sample_count": len(deduped),
        "duplicate_timestamps_merged": duplicate_count,
        "out_of_order_count": out_of_order_count,
        "min_ts_sec": deduped[0]["timestamp_sec"] if deduped else None,
        "max_ts_sec": deduped[-1]["timestamp_sec"] if deduped else None,
        "samples": deduped,
    }


def _lerp(a: Optional[float], b: Optional[float], alpha: float) -> Optional[float]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return float(a + alpha * (b - a))


def _lerp_angle_deg(a: Optional[float], b: Optional[float], alpha: float) -> Optional[float]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    diff = ((b - a + 180.0) % 360.0) - 180.0
    return (a + alpha * diff) % 360.0


def synchronize_timestamp_to_gps(
    target_ts_sec: float,
    gps_samples: Sequence[Dict[str, Any]],
    *,
    cfg: Optional[GeospatialConfig] = None,
    sync_offset_sec: float = 0.0,
    gps_source: str = "external_telemetry",
) -> Dict[str, Any]:
    """
    Associate a single video-relative timestamp against monotonic GPS samples.
    Never extrapolates outside [first_ts, last_ts] when cfg.allow_extrapolation is False.
    """
    cfg = cfg or GeospatialConfig()
    if not gps_samples:
        return {
            "available": False,
            "lat": None,
            "lon": None,
            "horizontal_accuracy_m": None,
            "heading_deg": None,
            "original_gps_timestamp": None,
            "sync_offset_sec": float(sync_offset_sec),
            "interpolation_status": "UNAVAILABLE",
            "time_diff_sec": None,
            "sync_quality": "UNAVAILABLE",
            "gps_source": gps_source,
            "provenance_label": "UNAVAILABLE",
            "reason": "no_gps_telemetry",
        }

    ts_list = [float(s["timestamp_sec"]) for s in gps_samples]
    t = float(target_ts_sec)

    if t < ts_list[0] or t > ts_list[-1]:
        edge = gps_samples[0] if t < ts_list[0] else gps_samples[-1]
        dt = abs(t - float(edge["timestamp_sec"]))
        if not cfg.allow_extrapolation or dt > cfg.max_sync_gap_sec:
            return {
                "available": False,
                "lat": None,
                "lon": None,
                "horizontal_accuracy_m": None,
                "heading_deg": None,
                "original_gps_timestamp": edge.get("raw_timestamp"),
                "sync_offset_sec": float(sync_offset_sec),
                "interpolation_status": "UNAVAILABLE",
                "time_diff_sec": round(dt, 4),
                "sync_quality": "OUT_OF_RANGE",
                "gps_source": gps_source,
                "provenance_label": "UNAVAILABLE",
                "reason": "timestamp_outside_gps_bounds",
            }
        return {
            "available": True,
            "lat": edge["lat"],
            "lon": edge["lon"],
            "horizontal_accuracy_m": edge.get("horizontal_accuracy_m"),
            "heading_deg": edge.get("heading_deg"),
            "original_gps_timestamp": edge.get("raw_timestamp"),
            "sync_offset_sec": float(sync_offset_sec),
            "interpolation_status": "EXTRAPOLATED",
            "time_diff_sec": round(dt, 4),
            "sync_quality": "LOW",
            "gps_source": gps_source,
            "provenance_label": "APPROXIMATE_VEHICLE_GPS",
            "reason": None,
        }

    # Find bracketing interval via binary search
    import bisect

    idx = bisect.bisect_left(ts_list, t)
    if idx < len(gps_samples) and abs(ts_list[idx] - t) <= cfg.exact_match_threshold_sec:
        s = gps_samples[idx]
        return {
            "available": True,
            "lat": s["lat"],
            "lon": s["lon"],
            "horizontal_accuracy_m": s.get("horizontal_accuracy_m"),
            "heading_deg": s.get("heading_deg"),
            "original_gps_timestamp": s.get("raw_timestamp"),
            "sync_offset_sec": float(sync_offset_sec),
            "interpolation_status": "EXACT",
            "time_diff_sec": round(abs(ts_list[idx] - t), 4),
            "sync_quality": "HIGH",
            "gps_source": gps_source,
            "provenance_label": "APPROXIMATE_VEHICLE_GPS",
            "reason": None,
        }
    if idx > 0 and abs(t - ts_list[idx - 1]) <= cfg.exact_match_threshold_sec:
        s = gps_samples[idx - 1]
        return {
            "available": True,
            "lat": s["lat"],
            "lon": s["lon"],
            "horizontal_accuracy_m": s.get("horizontal_accuracy_m"),
            "heading_deg": s.get("heading_deg"),
            "original_gps_timestamp": s.get("raw_timestamp"),
            "sync_offset_sec": float(sync_offset_sec),
            "interpolation_status": "EXACT",
            "time_diff_sec": round(abs(t - ts_list[idx - 1]), 4),
            "sync_quality": "HIGH",
            "gps_source": gps_source,
            "provenance_label": "APPROXIMATE_VEHICLE_GPS",
            "reason": None,
        }

    s0 = gps_samples[max(0, idx - 1)]
    s1 = gps_samples[min(len(gps_samples) - 1, idx)]
    gap = float(s1["timestamp_sec"]) - float(s0["timestamp_sec"])
    nearest_dt = min(abs(t - float(s0["timestamp_sec"])), abs(float(s1["timestamp_sec"]) - t))

    if gap > cfg.max_sync_gap_sec:
        return {
            "available": False,
            "lat": None,
            "lon": None,
            "horizontal_accuracy_m": None,
            "heading_deg": None,
            "original_gps_timestamp": s0.get("raw_timestamp"),
            "sync_offset_sec": float(sync_offset_sec),
            "interpolation_status": "UNAVAILABLE",
            "time_diff_sec": round(nearest_dt, 4),
            "sync_quality": "GAP_EXCEEDED",
            "gps_source": gps_source,
            "provenance_label": "UNAVAILABLE",
            "reason": f"gps_gap_{gap:.2f}s_exceeds_limit_{cfg.max_sync_gap_sec:.2f}s",
        }

    alpha = (t - float(s0["timestamp_sec"])) / max(gap, 1e-9)
    lat = round(_lerp(s0["lat"], s1["lat"], alpha), 7)
    lon = round(_lerp(s0["lon"], s1["lon"], alpha), 7)
    acc = _lerp(s0.get("horizontal_accuracy_m"), s1.get("horizontal_accuracy_m"), alpha)
    hdg = _lerp_angle_deg(s0.get("heading_deg"), s1.get("heading_deg"), alpha)
    sync_qual = "HIGH" if nearest_dt <= 0.5 else "MODERATE"

    return {
        "available": True,
        "lat": lat,
        "lon": lon,
        "horizontal_accuracy_m": round(acc, 3) if acc is not None else None,
        "heading_deg": round(hdg, 2) if hdg is not None else None,
        "original_gps_timestamp": s0.get("raw_timestamp"),
        "sync_offset_sec": float(sync_offset_sec),
        "interpolation_status": "INTERPOLATED",
        "time_diff_sec": round(nearest_dt, 4),
        "sync_quality": sync_qual,
        "gps_source": gps_source,
        "provenance_label": "APPROXIMATE_VEHICLE_GPS",
        "reason": None,
    }

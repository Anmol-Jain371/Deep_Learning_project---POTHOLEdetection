"""
Flask Blueprint routes for RoadVision AI 2.0 (Phases 8B.1–8B.4).
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import uuid
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from flask import Blueprint, Response, current_app, request

from roadvision_v2.geospatial import parse_gps_telemetry
from roadvision_v2.maintenance import render_inspection_report_csv, render_inspection_report_html
from roadvision_v2.schemas import (
    INSPECTION_PROVENANCE_CATEGORIES,
    MEASUREMENT_STATES,
)

from .artifacts import resolve_artifact_entry, serve_artifact_response
from .configuration import APIConfig, PROJECT_ROOT
from .errors import (
    BadRequestError,
    InvalidJobStateError,
    NotFoundError,
    ServiceUnavailableError,
    register_v2_error_handlers,
)
from .geospatial_store import GeospatialStore
from .gpu_lock import is_gpu_busy
from .job_manager import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JobManager,
    validate_uuid,
)
from .maintenance_store import MaintenanceStore
from .schemas import (
    PIPELINE_VERSION,
    SCIENTIFIC_MEASUREMENT_DISCLAIMER,
    compute_measurement_coverage,
    sanitize_json_value,
)
from .validation import parse_optional_bool, save_and_validate_upload, validate_job_options

v2_bp = Blueprint("roadvision_v2_api", __name__)
register_v2_error_handlers(v2_bp)


def _strict_json_response(payload: Any, status_code: int = 200) -> Response:
    """Serialize payload to strict RFC 8259 JSON (NaN/Infinity converted to null)."""
    clean = sanitize_json_value(payload)
    body = json.dumps(clean, allow_nan=False)
    return Response(body, status=status_code, mimetype="application/json")


def _get_runtime() -> Dict[str, Any]:
    rt = current_app.extensions.get("roadvision_v2")
    if not rt:
        raise ServiceUnavailableError("RoadVision V2 runtime is not initialized")
    return rt


@v2_bp.after_request
def _enforce_v2_cors_and_headers(resp: Response) -> Response:
    """Restrict V2 CORS origins strictly to allowed_cors_origins (never wildcard *)."""
    rt = current_app.extensions.get("roadvision_v2")
    if rt:
        cfg: APIConfig = rt["config"]
        origin = request.headers.get("Origin")
        resp.headers.pop("Access-Control-Allow-Origin", None)
        if origin and origin in cfg.allowed_cors_origins:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PATCH, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Range"
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    return resp


def _get_pagination_params(cfg: APIConfig) -> Tuple[int, int]:
    raw_page = request.args.get("page", "1")
    raw_size = request.args.get("page_size", str(cfg.default_page_size))
    try:
        page = int(raw_page)
        page_size = int(raw_size)
    except ValueError as exc:
        raise BadRequestError("page and page_size must be integers", error_code="INVALID_PAGINATION") from exc
    if page < 1:
        raise BadRequestError("page must be >= 1", error_code="INVALID_PAGINATION")
    if page_size < 1:
        raise BadRequestError("page_size must be >= 1", error_code="INVALID_PAGINATION")
    page_size = min(page_size, cfg.max_page_size)
    return page, page_size


def _parse_provenance_scope(default_scope: str = "all") -> Tuple[Optional[List[str]], bool, str]:
    """Parse ?provenance=... and ?scope=genuine|exploratory|all cleanly."""
    raw_prov = request.args.get("provenance")
    if raw_prov is not None and str(raw_prov).strip() != "":
        prov_norm = str(raw_prov).strip().upper()
        if prov_norm not in INSPECTION_PROVENANCE_CATEGORIES:
            raise BadRequestError(
                f"Unsupported provenance filter '{prov_norm}'",
                error_code="INVALID_FILTER",
                details={"allowed_provenances": list(INSPECTION_PROVENANCE_CATEGORIES)},
            )
        return [prov_norm], False, prov_norm

    scope = (request.args.get("scope") or default_scope).strip().lower()
    if scope == "genuine":
        return ["USER_INSPECTION"], True, "genuine"
    if scope == "exploratory":
        return ["USER_INSPECTION", "RESEARCH_EXPLORATORY"], False, "exploratory"
    if scope == "all":
        return None, False, "all"
    raise BadRequestError(
        f"Unsupported provenance scope '{scope}'; expected genuine, exploratory, or all",
        error_code="INVALID_FILTER",
    )


def _classify_geometry_snapshot_artifact(snap: Dict[str, Any]) -> Tuple[str, Optional[str], int]:
    """Inspect reference_mask_grid without fabricating any mask cells."""
    ref_grid = snap.get("reference_mask_grid") or []
    ref_true = sum(1 for row in ref_grid for cell in (row or []) if bool(cell))
    schema_ver = str(snap.get("geometry_schema_version") or "")
    explicit_status = str(snap.get("reference_mask_status") or "")
    if ref_true > 0:
        return "CURRENT_VALID_REFERENCE_MASK", None, ref_true
    if schema_ver == "10b" or explicit_status in ("POPULATED", "EMPTY"):
        return (
            "GENUINE_EMPTY_REFERENCE_MASK",
            "Reference ring mask is genuinely empty due to scene occlusion or boundary constraints; no synthetic mask cells were added.",
            0,
        )
    return (
        "HISTORICAL_EMPTY_REFERENCE_MASK",
        "Historical geometry snapshot saved prior to the Phase 10B reference-mask fix; annular reference mask is unavailable in this stored artifact.",
        0,
    )


def _format_job_status_payload(
    job: Dict[str, Any],
    geo_store: Optional[GeospatialStore] = None,
) -> Dict[str, Any]:
    """Build polling-safe job status response without large observation arrays."""
    jid = job["job_id"]
    total_obs = int(job.get("detection_observation_count") or 0)
    state_counts = job.get("measurement_state_counts") or {}
    coverage = compute_measurement_coverage(state_counts, total_obs)
    avail = job.get("artifact_availability") or {}
    gps_avail = False
    if geo_store is not None:
        try:
            gps_avail = geo_store.has_gps_for_job(jid)
        except Exception:
            gps_avail = False
    return {
        "job_id": jid,
        "original_filename": job.get("original_filename"),
        "status": job["status"],
        "mode": job["mode"],
        "provenance": job.get("provenance") or "USER_INSPECTION",
        "provenance_reason": job.get("provenance_reason"),
        "created_at": job["created_at"],
        "started_at": job.get("started_at"),
        "updated_at": job["updated_at"],
        "completed_at": job.get("completed_at"),
        "frames_processed": job.get("processed_frames", 0),
        "total_frames": job.get("total_frames"),
        "progress_percentage": job.get("progress_percentage"),
        "throughput_fps": job.get("throughput_fps"),
        "candidate_observation_count": total_obs,
        "unique_candidate_track_count": job.get("unique_candidate_track_count", 0),
        "depth_invocation_count": job.get("depth_invocation_count", 0),
        "scheduler_request_count": job.get("scheduler_request_count", 0),
        "failed_depth_count": job.get("failed_depth_count", 0),
        "measurement_state_counts": coverage["counts_by_state"],
        "measurement_coverage": coverage,
        "cancel_requested": bool(job.get("cancel_requested", False)),
        "latest_warning": job.get("latest_warning"),
        "error": (
            {"code": job["error_code"], "message": job["error_message"]}
            if job.get("error_code") or job.get("error_message")
            else None
        ),
        "artifact_availability": avail,
        "geometry_available": bool(avail.get("geometry_manifest_json", False)),
        "gps_available": gps_avail,
        "pipeline_version": job.get("pipeline_version", PIPELINE_VERSION),
        "scientific_disclaimer": SCIENTIFIC_MEASUREMENT_DISCLAIMER,
    }


def _require_completed_artifact_path(
    job_id: str,
    artifact_id: str,
    allow_cancelled: bool = False,
) -> Tuple[Dict[str, Any], str]:
    rt = _get_runtime()
    jm: JobManager = rt["job_manager"]
    cfg: APIConfig = rt["config"]
    job = jm.get_job(job_id, include_internal_paths=True)
    status = job["status"]

    allowed_statuses = (JOB_STATUS_COMPLETED, JOB_STATUS_CANCELLED) if allow_cancelled else (JOB_STATUS_COMPLETED,)
    if status not in allowed_statuses:
        raise InvalidJobStateError(
            f"Results are not available while job status is '{status}'",
            error_code="RESULTS_UNAVAILABLE",
            details={"status": status},
        )
    _, path = resolve_artifact_entry(job, artifact_id, cfg.resolved_artifact_dir)
    return job, path


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------

@v2_bp.route("/health", methods=["GET"])
def health():
    rt = _get_runtime()
    cfg: APIConfig = rt["config"]
    jm: JobManager = rt["job_manager"]
    worker = rt.get("worker")
    genuine_counts = jm.count_jobs_by_status(exclude_non_genuine=True)
    all_counts = jm.count_jobs_by_status(exclude_non_genuine=False)
    prov_counts = jm.count_jobs_by_provenance()
    total_all = sum(all_counts.values())
    total_genuine = sum(genuine_counts.values())
    return _strict_json_response(
        {
            "status": "online",
            "api_version": PIPELINE_VERSION,
            "api_enabled": cfg.enabled,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "worker_running": bool(worker and worker.is_running),
            "active_job_id": worker.active_job_id if worker else None,
            "gpu_busy": is_gpu_busy(),
            "queue_counts": genuine_counts,
            "all_queue_counts": all_counts,
            "provenance_counts": prov_counts,
            "non_genuine_excluded_count": max(0, total_all - total_genuine),
        },
        200,
    )


@v2_bp.route("/modes", methods=["GET"])
def modes():
    cfg_path = os.path.join(PROJECT_ROOT, "config", "roadvision_v2.yaml")
    if not os.path.exists(cfg_path):
        raise ServiceUnavailableError("RoadVision V2 mode configuration file is missing")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    modes_block = data.get("modes", {}) if isinstance(data, dict) else {}
    return _strict_json_response(
        {
            "modes": modes_block,
            "scientific_disclaimer": SCIENTIFIC_MEASUREMENT_DISCLAIMER,
        },
        200,
    )


@v2_bp.route("/capabilities", methods=["GET"])
def capabilities():
    rt = _get_runtime()
    cfg: APIConfig = rt["config"]
    gpu_avail = torch.cuda.is_available()
    det_ckpt = os.path.join(
        PROJECT_ROOT, "runs", "detect", "pothole_unified_yolov8n_100e_sgd", "weights", "best.pt"
    )
    return _strict_json_response(
        {
            "gpu_available": gpu_avail,
            "gpu_name": torch.cuda.get_device_name(0) if gpu_avail else None,
            "detector_checkpoint_available": os.path.isfile(det_ckpt),
            "models": {
                "detector": "yolov8n_phase2_controlled",
                "midas_small": True,
                "depth_anything_v2_small": True,
            },
            "supported_modes": ["live", "balanced", "survey"],
            "features": {
                "quality_assessment": True,
                "geometry_snapshots": True,
                "geospatial_telemetry": True,
                "cross_trip_matching": True,
                "maintenance_intelligence": True,
            },
            "limits": {
                "maximum_upload_size_bytes": cfg.maximum_upload_size,
                "allowed_video_formats": cfg.allowed_video_formats,
                "maximum_video_resolution": list(cfg.maximum_video_resolution),
                "maximum_video_duration_sec": cfg.maximum_video_duration,
                "maximum_processed_frames": cfg.maximum_processed_frames,
                "max_queued_jobs": cfg.max_queued_jobs,
                "max_gpu_concurrency": cfg.max_gpu_concurrency,
                "default_page_size": cfg.default_page_size,
                "max_page_size": cfg.max_page_size,
            },
            "security_note": (
                "Local research prototype API without user authentication; any client with access "
                "to localhost can query local jobs."
            ),
        },
        200,
    )


# ---------------------------------------------------------------------------
# Job endpoints
# ---------------------------------------------------------------------------

@v2_bp.route("/jobs", methods=["POST"])
def submit_job():
    rt = _get_runtime()
    cfg: APIConfig = rt["config"]
    jm: JobManager = rt["job_manager"]
    worker = rt.get("worker")

    if "video" not in request.files:
        raise BadRequestError("Missing 'video' file field in multipart upload", error_code="MISSING_VIDEO")

    validated_opts = validate_job_options(request.form, cfg, check_assets=True)
    job_id = str(uuid.uuid4())
    media = save_and_validate_upload(request.files["video"], job_id, cfg)
    output_dir = os.path.join(cfg.resolved_artifact_dir, job_id)
    os.makedirs(output_dir, exist_ok=True)

    try:
        effective_total = media.estimated_total_frames
        if effective_total is not None and validated_opts.max_frames is not None:
            effective_total = min(effective_total, validated_opts.max_frames)

        job = jm.create_job(
            job_id=job_id,
            original_filename=media.original_filename,
            validated_input_path=media.validated_input_path,
            mode=validated_opts.mode,
            config_snapshot=validated_opts.v2_config_snapshot,
            output_dir=output_dir,
            total_frames=effective_total,
            pipeline_version=PIPELINE_VERSION,
            provenance=validated_opts.provenance,
            provenance_reason="API upload",
        )
    except Exception:
        shutil.rmtree(media.upload_job_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    if worker is not None:
        worker.notify()

    return _strict_json_response(
        {
            "job_id": job["job_id"],
            "status": job["status"],
            "mode": job["mode"],
            "provenance": job.get("provenance", "USER_INSPECTION"),
            "created_at": job["created_at"],
            "status_url": f"/api/v2/jobs/{job['job_id']}",
            "original_filename": job["original_filename"],
            "total_frames": job["total_frames"],
        },
        202,
    )


@v2_bp.route("/jobs", methods=["GET"])
def list_jobs_route():
    rt = _get_runtime()
    cfg: APIConfig = rt["config"]
    jm: JobManager = rt["job_manager"]
    geo_store: Optional[GeospatialStore] = rt.get("geospatial_store")
    page, page_size = _get_pagination_params(cfg)
    status_filter = request.args.get("status")
    include_provs, excl_non_genuine, scope_label = _parse_provenance_scope(default_scope="all")
    offset = (page - 1) * page_size
    res = jm.list_jobs(
        status=status_filter,
        include_provenances=include_provs,
        exclude_non_genuine=excl_non_genuine,
        limit=page_size,
        offset=offset,
    )
    prov_counts = jm.count_jobs_by_provenance(status=status_filter.upper().strip() if status_filter else None)
    return _strict_json_response(
        {
            "jobs": [_format_job_status_payload(j, geo_store=geo_store) for j in res["jobs"]],
            "total": res["total"],
            "scope": scope_label,
            "provenance_counts": prov_counts,
            "page": page,
            "page_size": page_size,
        },
        200,
    )


@v2_bp.route("/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    rt = _get_runtime()
    jm: JobManager = rt["job_manager"]
    geo_store: Optional[GeospatialStore] = rt.get("geospatial_store")
    job = jm.get_job(job_id, include_internal_paths=False)
    return _strict_json_response(_format_job_status_payload(job, geo_store=geo_store), 200)


@v2_bp.route("/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id: str):
    rt = _get_runtime()
    jm: JobManager = rt["job_manager"]
    updated = jm.request_cancellation(job_id)
    return _strict_json_response(
        {
            "job_id": updated["job_id"],
            "status": updated["status"],
            "cancel_requested": updated["cancel_requested"],
            "updated_at": updated["updated_at"],
        },
        202,
    )


# ---------------------------------------------------------------------------
# Result endpoints
# ---------------------------------------------------------------------------

@v2_bp.route("/jobs/<job_id>/summary", methods=["GET"])
def get_job_summary(job_id: str):
    rt = _get_runtime()
    jm: JobManager = rt["job_manager"]
    job = jm.get_job(job_id, include_internal_paths=False)
    if job["status"] != JOB_STATUS_COMPLETED:
        raise InvalidJobStateError(
            f"Summary is only available after completion; current status is '{job['status']}'",
            error_code="RESULTS_UNAVAILABLE",
            details={"status": job["status"]},
        )
    raw_summary = dict(job.get("summary") or {})
    # Never expose server filesystem source path
    if "source" in raw_summary:
        raw_summary["source"] = job.get("original_filename")
    state_counts = job.get("measurement_state_counts") or raw_summary.get("measurement_state_counts") or {}
    total_obs = int(job.get("detection_observation_count") or raw_summary.get("total_hazard_observations") or 0)
    coverage = compute_measurement_coverage(state_counts, total_obs)
    return _strict_json_response(
        {
            "job_id": job["job_id"],
            "status": job["status"],
            "mode": job["mode"],
            "original_filename": job.get("original_filename"),
            "session_summary": raw_summary,
            "quality_summary": raw_summary.get("quality_summary"),
            "measurement_coverage": coverage,
            "telemetry": job.get("telemetry"),
            "scientific_disclaimer": SCIENTIFIC_MEASUREMENT_DISCLAIMER,
        },
        200,
    )


@v2_bp.route("/jobs/<job_id>/observations", methods=["GET"])
def get_job_observations(job_id: str):
    rt = _get_runtime()
    cfg: APIConfig = rt["config"]
    page, page_size = _get_pagination_params(cfg)
    _, frames_path = _require_completed_artifact_path(job_id, "frames_json", allow_cancelled=False)

    # Parse optional filters cleanly
    track_id_filter: Optional[int] = None
    if request.args.get("track_id") is not None:
        try:
            track_id_filter = int(request.args["track_id"])
        except ValueError as exc:
            raise BadRequestError("track_id must be an integer", error_code="INVALID_FILTER") from exc

    min_frame: Optional[int] = None
    max_frame: Optional[int] = None
    if request.args.get("min_frame") is not None:
        try:
            min_frame = int(request.args["min_frame"])
        except ValueError as exc:
            raise BadRequestError("min_frame must be an integer", error_code="INVALID_FILTER") from exc
    if request.args.get("max_frame") is not None:
        try:
            max_frame = int(request.args["max_frame"])
        except ValueError as exc:
            raise BadRequestError("max_frame must be an integer", error_code="INVALID_FILTER") from exc

    state_filter = request.args.get("measurement_state")
    if state_filter is not None:
        state_filter = state_filter.strip().upper()
        if state_filter not in MEASUREMENT_STATES:
            raise BadRequestError(
                f"Invalid measurement_state '{state_filter}'",
                error_code="INVALID_FILTER",
                details={"allowed_states": list(MEASUREMENT_STATES)},
            )

    min_conf: Optional[float] = None
    max_conf: Optional[float] = None
    if request.args.get("min_conf") is not None:
        try:
            min_conf = float(request.args["min_conf"])
        except ValueError as exc:
            raise BadRequestError("min_conf must be numeric", error_code="INVALID_FILTER") from exc
    if request.args.get("max_conf") is not None:
        try:
            max_conf = float(request.args["max_conf"])
        except ValueError as exc:
            raise BadRequestError("max_conf must be numeric", error_code="INVALID_FILTER") from exc

    with open(frames_path, "r", encoding="utf-8") as fh:
        all_obs = json.load(fh)
    if not isinstance(all_obs, list):
        all_obs = []

    filtered: List[Dict[str, Any]] = []
    for obs in all_obs:
        fc = obs.get("frame_context") or {}
        det = obs.get("detection") or {}
        meas = obs.get("measurement") or {}
        if track_id_filter is not None and det.get("track_id") != track_id_filter:
            continue
        fidx = fc.get("frame_idx")
        if min_frame is not None and (fidx is None or fidx < min_frame):
            continue
        if max_frame is not None and (fidx is None or fidx > max_frame):
            continue
        if state_filter is not None and meas.get("measurement_state") != state_filter:
            continue
        conf = det.get("detector_confidence")
        if min_conf is not None and (conf is None or conf < min_conf):
            continue
        if max_conf is not None and (conf is None or conf > max_conf):
            continue
        filtered.append(obs)

    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = filtered[start:end]

    return _strict_json_response(
        {
            "job_id": validate_uuid(job_id),
            "observations": page_items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if total > 0 else 0,
            "scientific_disclaimer": SCIENTIFIC_MEASUREMENT_DISCLAIMER,
        },
        200,
    )


@v2_bp.route("/jobs/<job_id>/tracks", methods=["GET"])
def get_job_tracks(job_id: str):
    rt = _get_runtime()
    cfg: APIConfig = rt["config"]
    page, page_size = _get_pagination_params(cfg)
    _, tracks_path = _require_completed_artifact_path(job_id, "tracks_json", allow_cancelled=False)

    with open(tracks_path, "r", encoding="utf-8") as fh:
        raw_tracks = json.load(fh)
    track_list = list(raw_tracks.values()) if isinstance(raw_tracks, dict) else []
    track_list.sort(key=lambda t: int(t.get("track_id", 0)))

    total = len(track_list)
    start = (page - 1) * page_size
    page_items = track_list[start : start + page_size]

    return _strict_json_response(
        {
            "job_id": validate_uuid(job_id),
            "tracks": page_items,
            "unique_session_track_count": total,
            "note": (
                "Track entries reflect unique session ByteTrack IDs, not verified independent "
                "physical potholes."
            ),
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if total > 0 else 0,
        },
        200,
    )


@v2_bp.route("/jobs/<job_id>/performance", methods=["GET"])
def get_job_performance(job_id: str):
    rt = _get_runtime()
    cfg: APIConfig = rt["config"]
    page, page_size = _get_pagination_params(cfg)
    job = rt["job_manager"].get_job(job_id, include_internal_paths=False)
    if job["status"] != JOB_STATUS_COMPLETED:
        raise InvalidJobStateError(
            f"Performance data is only available after completion; current status is '{job['status']}'",
            error_code="RESULTS_UNAVAILABLE",
            details={"status": job["status"]},
        )

    per_frame_rows: List[Dict[str, Any]] = []
    total_rows = 0
    avail = job.get("artifact_availability") or {}
    if avail.get("performance_csv"):
        _, csv_path = _require_completed_artifact_path(job_id, "performance_csv", allow_cancelled=False)
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            parsed_rows: List[Dict[str, Any]] = []
            for row in reader:
                item: Dict[str, Any] = {}
                for k, v in row.items():
                    if k in ("frame_id", "mode"):
                        item[k] = v
                    elif k == "frame_idx":
                        item[k] = int(v)
                    else:
                        try:
                            item[k] = float(v)
                        except (TypeError, ValueError):
                            item[k] = None
                parsed_rows.append(item)
        total_rows = len(parsed_rows)
        start = (page - 1) * page_size
        per_frame_rows = parsed_rows[start : start + page_size]

    return _strict_json_response(
        {
            "job_id": job["job_id"],
            "mode": job["mode"],
            "telemetry": job.get("telemetry"),
            "frames": per_frame_rows,
            "total": total_rows,
            "page": page,
            "page_size": page_size,
        },
        200,
    )


# ---------------------------------------------------------------------------
# Artifact endpoints
# ---------------------------------------------------------------------------

@v2_bp.route("/jobs/<job_id>/artifacts", methods=["GET"])
def list_job_artifacts(job_id: str):
    rt = _get_runtime()
    jm: JobManager = rt["job_manager"]
    job = jm.get_job(job_id, include_internal_paths=False)
    manifest = job.get("artifact_manifest") or {"job_id": job["job_id"], "partial": False, "artifacts": []}
    return _strict_json_response(
        {
            "job_id": job["job_id"],
            "status": job["status"],
            "partial": bool(manifest.get("partial", False)),
            "artifacts": manifest.get("artifacts", []),
            "artifact_availability": job.get("artifact_availability") or {},
        },
        200,
    )


@v2_bp.route("/jobs/<job_id>/artifacts/<artifact_name>", methods=["GET"])
def download_job_artifact(job_id: str, artifact_name: str):
    rt = _get_runtime()
    jm: JobManager = rt["job_manager"]
    cfg: APIConfig = rt["config"]
    job = jm.get_job(job_id, include_internal_paths=True)
    if job["status"] not in (JOB_STATUS_COMPLETED, JOB_STATUS_CANCELLED, JOB_STATUS_FAILED):
        raise InvalidJobStateError(
            f"Artifacts are not published while job status is '{job['status']}'",
            error_code="ARTIFACTS_NOT_READY",
            details={"status": job["status"]},
        )
    entry, file_path = resolve_artifact_entry(job, artifact_name, cfg.resolved_artifact_dir)
    as_attachment = request.args.get("download", "0").lower() in ("1", "true", "yes")
    return serve_artifact_response(entry, file_path, as_attachment=as_attachment)


# ---------------------------------------------------------------------------
# Phase 10: Relative 2.5D Geometry Snapshot endpoints
# ---------------------------------------------------------------------------

_SAFE_SNAPSHOT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,80}$")
_MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024  # 4 MiB safety bound


@v2_bp.route("/jobs/<job_id>/geometry", methods=["GET"])
def list_job_geometry_snapshots(job_id: str):
    rt = _get_runtime()
    jm: JobManager = rt["job_manager"]
    cfg: APIConfig = rt["config"]
    job = jm.get_job(job_id, include_internal_paths=True)

    if job["status"] != JOB_STATUS_COMPLETED:
        raise InvalidJobStateError(
            f"Geometry snapshots are only available after completion; current status is '{job['status']}'",
            error_code="RESULTS_UNAVAILABLE",
            details={"status": job["status"]},
        )

    track_filter: Optional[int] = None
    if request.args.get("track_id") is not None and str(request.args.get("track_id")).strip() != "":
        try:
            track_filter = int(request.args["track_id"])
        except ValueError as exc:
            raise BadRequestError("track_id must be an integer", error_code="INVALID_FILTER") from exc

    avail = job.get("artifact_availability") or {}
    if not avail.get("geometry_manifest_json"):
        return _strict_json_response(
            {
                "job_id": job["job_id"],
                "geometry_available": False,
                "snapshot_count": 0,
                "snapshots": [],
                "reason": "Geometry snapshot recording was not enabled for this inspection.",
                "scientific_disclaimer": SCIENTIFIC_MEASUREMENT_DISCLAIMER,
            },
            200,
        )

    _, manifest_path = resolve_artifact_entry(job, "geometry_manifest_json", cfg.resolved_artifact_dir)
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    snaps = manifest.get("snapshots") or []
    if track_filter is not None:
        snaps = [s for s in snaps if int(s.get("track_id", -1)) == track_filter]

    real_job_dir = os.path.realpath(job.get("output_dir") or "")
    enriched_snaps = []
    historical_empty_ref_count = 0
    for s in snaps:
        item = dict(s)
        art_status = "UNKNOWN"
        if real_job_dir and item.get("relative_path"):
            cand_p = os.path.realpath(os.path.join(real_job_dir, item["relative_path"]))
            if os.path.commonpath([real_job_dir, cand_p]) == real_job_dir and os.path.isfile(cand_p):
                try:
                    with open(cand_p, "r", encoding="utf-8") as sfh:
                        snap_obj = json.load(sfh)
                    art_status, _, ref_cnt = _classify_geometry_snapshot_artifact(snap_obj)
                    item["reference_mask_cell_count"] = ref_cnt
                except Exception:
                    art_status = "UNKNOWN"
        item["geometry_artifact_status"] = art_status
        if art_status == "HISTORICAL_EMPTY_REFERENCE_MASK":
            historical_empty_ref_count += 1
        enriched_snaps.append(item)

    return _strict_json_response(
        {
            "job_id": job["job_id"],
            "geometry_available": True,
            "snapshot_count": len(enriched_snaps),
            "historical_empty_reference_mask_count": historical_empty_ref_count,
            "total_job_snapshots": manifest.get("snapshot_count", len(enriched_snaps)),
            "max_grid_size": manifest.get("max_grid_size"),
            "residual_polarity": manifest.get("residual_polarity"),
            "residual_units": manifest.get("residual_units"),
            "global_min_residual": manifest.get("global_min_residual"),
            "global_max_residual": manifest.get("global_max_residual"),
            "snapshots": enriched_snaps,
            "scientific_disclaimer": manifest.get("scientific_disclaimer") or SCIENTIFIC_MEASUREMENT_DISCLAIMER,
        },
        200,
    )


@v2_bp.route("/jobs/<job_id>/geometry/<snapshot_id>", methods=["GET"])
def get_job_geometry_snapshot(job_id: str, snapshot_id: str):
    rt = _get_runtime()
    jm: JobManager = rt["job_manager"]
    cfg: APIConfig = rt["config"]

    if not snapshot_id or not _SAFE_SNAPSHOT_ID_RE.match(snapshot_id):
        raise BadRequestError("Invalid snapshot_id format", error_code="INVALID_SNAPSHOT_ID")

    job = jm.get_job(job_id, include_internal_paths=True)
    if job["status"] != JOB_STATUS_COMPLETED:
        raise InvalidJobStateError(
            f"Geometry snapshot unavailable while job status is '{job['status']}'",
            error_code="RESULTS_UNAVAILABLE",
            details={"status": job["status"]},
        )

    avail = job.get("artifact_availability") or {}
    if not avail.get("geometry_manifest_json"):
        raise NotFoundError("Geometry snapshots unavailable for this job", error_code="GEOMETRY_NOT_FOUND")

    _, manifest_path = resolve_artifact_entry(job, "geometry_manifest_json", cfg.resolved_artifact_dir)
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    matched = next((s for s in (manifest.get("snapshots") or []) if s.get("snapshot_id") == snapshot_id), None)
    if matched is None:
        raise NotFoundError(f"Geometry snapshot '{snapshot_id}' not found", error_code="SNAPSHOT_NOT_FOUND")

    real_root = os.path.realpath(cfg.resolved_artifact_dir)
    real_job_dir = os.path.realpath(job.get("output_dir") or "")
    if not real_job_dir or os.path.commonpath([real_root, real_job_dir]) != real_root:
        raise NotFoundError("Snapshot directory outside approved root", error_code="SNAPSHOT_FORBIDDEN")

    raw_snap_path = os.path.join(real_job_dir, matched["relative_path"])
    if os.path.islink(raw_snap_path):
        raise NotFoundError("Symlink geometry snapshots are forbidden", error_code="SNAPSHOT_FORBIDDEN")

    real_snap_path = os.path.realpath(raw_snap_path)
    if os.path.commonpath([real_job_dir, real_snap_path]) != real_job_dir or not os.path.isfile(real_snap_path):
        raise NotFoundError("Snapshot file missing or escapes job directory", error_code="SNAPSHOT_NOT_FOUND")

    if os.path.getsize(real_snap_path) > _MAX_SNAPSHOT_BYTES:
        raise ServiceUnavailableError("Geometry snapshot exceeds maximum allowed response size")

    with open(real_snap_path, "r", encoding="utf-8") as fh:
        snap_payload = json.load(fh)
    snap_payload["global_min_residual"] = manifest.get("global_min_residual")
    snap_payload["global_max_residual"] = manifest.get("global_max_residual")
    artifact_status, artifact_warning, ref_true_count = _classify_geometry_snapshot_artifact(snap_payload)
    snap_payload["geometry_artifact_status"] = artifact_status
    snap_payload["geometry_artifact_warning"] = artifact_warning
    snap_payload["reference_mask_cell_count"] = ref_true_count
    return _strict_json_response(snap_payload, 200)


# ---------------------------------------------------------------------------
# Phase 11: Geospatial Telemetry, Routes, Locations & Cross-Trip Matching
# ---------------------------------------------------------------------------


def _get_geo_store() -> GeospatialStore:
    rt = _get_runtime()
    gs = rt.get("geospatial_store")
    if gs is None:
        gs = GeospatialStore(db=rt["db"], api_cfg=rt["config"])
        rt["geospatial_store"] = gs
    return gs


@v2_bp.route("/jobs/<job_id>/gps", methods=["POST"])
def upload_job_gps(job_id: str):
    rt = _get_runtime()
    jm: JobManager = rt["job_manager"]
    gs = _get_geo_store()
    job = jm.get_job(job_id, include_internal_paths=True)

    if "gps" not in request.files:
        raise BadRequestError("Missing 'gps' file field in multipart upload", error_code="MISSING_GPS_FILE")

    f_storage = request.files["gps"]
    raw_name = getattr(f_storage, "filename", "") or ""
    if not raw_name.strip():
        raise BadRequestError("Uploaded GPS filename is empty", error_code="EMPTY_FILENAME")

    time_mode = str(request.form.get("time_mode") or "auto").strip().lower()
    raw_offset = request.form.get("sync_offset_sec") or "0.0"
    raw_max_gap = request.form.get("max_sync_gap_sec") or "2.0"
    video_start_iso = request.form.get("video_start_iso") or None

    try:
        sync_offset_sec = float(raw_offset)
        max_sync_gap_sec = float(raw_max_gap)
    except ValueError as exc:
        raise BadRequestError("sync_offset_sec and max_sync_gap_sec must be numeric", error_code="INVALID_PARAMETER") from exc
    if max_sync_gap_sec <= 0.0 or max_sync_gap_sec > 60.0:
        raise BadRequestError("max_sync_gap_sec must be in (0, 60]", error_code="INVALID_PARAMETER")

    max_bytes = gs.default_geo_cfg.max_gps_file_bytes
    raw_bytes = f_storage.read(max_bytes + 1)
    if len(raw_bytes) > max_bytes:
        raise BadRequestError(
            f"GPS telemetry file exceeds maximum size of {max_bytes} bytes",
            error_code="GPS_FILE_TOO_LARGE",
        )

    try:
        parsed = parse_gps_telemetry(
            raw_bytes,
            raw_name,
            time_mode=time_mode,
            sync_offset_sec=sync_offset_sec,
            video_start_iso=video_start_iso,
            max_file_bytes=max_bytes,
            max_rows=gs.default_geo_cfg.max_gps_rows,
        )
    except ValueError as exc:
        raise BadRequestError(str(exc), error_code="INVALID_GPS_TELEMETRY") from exc

    safe_name = os.path.basename(raw_name.replace("\\", "/"))[:160] or "telemetry"
    gps_is_synth = parse_optional_bool(request.form.get("is_synthetic"), None, "is_synthetic")
    gps_prov = request.form.get("gps_provenance")
    route_payload = gs.save_job_gps_telemetry(
        job["job_id"],
        parsed,
        original_filename=safe_name,
        max_sync_gap_sec=max_sync_gap_sec,
        is_synthetic=gps_is_synth,
        gps_provenance=gps_prov,
    )

    synced_count = 0
    if job["status"] == JOB_STATUS_COMPLETED:
        synced_rows = gs.sync_job_tracks_with_gps(job)
        synced_count = sum(1 for r in synced_rows if r.get("lat") is not None)

    return _strict_json_response(
        {
            **route_payload,
            "synchronized_track_count": synced_count,
        },
        201,
    )


@v2_bp.route("/jobs/<job_id>/route", methods=["GET"])
def get_job_route(job_id: str):
    rt = _get_runtime()
    jm: JobManager = rt["job_manager"]
    gs = _get_geo_store()
    job = jm.get_job(job_id, include_internal_paths=False)
    return _strict_json_response(gs.get_job_route(job["job_id"]), 200)


@v2_bp.route("/jobs/<job_id>/locations", methods=["GET"])
def get_job_locations(job_id: str):
    rt = _get_runtime()
    jm: JobManager = rt["job_manager"]
    gs = _get_geo_store()
    job = jm.get_job(job_id, include_internal_paths=True)

    state_filter = request.args.get("measurement_state")
    if state_filter:
        state_filter = state_filter.strip().upper()
    qual_filter = request.args.get("quality_category")
    if qual_filter:
        qual_filter = qual_filter.strip().upper()
    fmt = (request.args.get("format") or "json").strip().lower()

    res = gs.get_job_locations(
        job,
        measurement_state=state_filter,
        quality_category=qual_filter,
        include_geojson=(fmt == "geojson" or request.args.get("include_geojson") in ("1", "true")),
    )
    if fmt == "geojson":
        return _strict_json_response(res.get("geojson") or {"type": "FeatureCollection", "features": []}, 200)
    return _strict_json_response(res, 200)


@v2_bp.route("/geospatial/match", methods=["POST"])
def run_geospatial_match():
    rt = _get_runtime()
    jm: JobManager = rt["job_manager"]
    gs = _get_geo_store()
    body = request.get_json(silent=True) or {}
    job_ids = body.get("job_ids")
    if job_ids is not None and not isinstance(job_ids, list):
        raise BadRequestError("job_ids must be an array of UUID strings", error_code="INVALID_PARAMETER")
    thresholds = body.get("thresholds")
    if thresholds is not None and not isinstance(thresholds, dict):
        raise BadRequestError("thresholds must be an object", error_code="INVALID_PARAMETER")

    try:
        res = gs.run_candidate_matching(
            job_manager=jm,
            job_ids=job_ids,
            config_overrides=thresholds,
        )
    except (ValueError, TypeError) as exc:
        raise BadRequestError(str(exc), error_code="INVALID_MATCH_PARAMETERS") from exc
    return _strict_json_response(res, 200)


@v2_bp.route("/geospatial/clusters", methods=["GET"])
def list_geospatial_clusters():
    rt = _get_runtime()
    cfg: APIConfig = rt["config"]
    gs = _get_geo_store()
    page, page_size = _get_pagination_params(cfg)
    status_filter = request.args.get("match_status")
    if status_filter:
        status_filter = status_filter.strip().upper()
    exclude_synth = bool(parse_optional_bool(request.args.get("exclude_synthetic"), False, "exclude_synthetic"))
    return _strict_json_response(
        gs.list_clusters(
            match_status=status_filter,
            exclude_synthetic=exclude_synth,
            page=page,
            page_size=page_size,
        ),
        200,
    )


@v2_bp.route("/geospatial/clusters/<cluster_id>", methods=["GET"])
def get_geospatial_cluster(cluster_id: str):
    gs = _get_geo_store()
    return _strict_json_response(gs.get_cluster(cluster_id), 200)


@v2_bp.route("/geospatial/clusters/<cluster_id>/review", methods=["POST"])
def review_geospatial_cluster(cluster_id: str):
    gs = _get_geo_store()
    body = request.get_json(silent=True) or {}
    review_status = body.get("review_status")
    return _strict_json_response(gs.update_cluster_review(cluster_id, review_status), 200)


# ---------------------------------------------------------------------------
# Phase 12: Road Maintenance Intelligence, Reviews, Segments & Reports
# ---------------------------------------------------------------------------


def _get_maint_store() -> MaintenanceStore:
    rt = _get_runtime()
    ms = rt.get("maintenance_store")
    if ms is None:
        gs = _get_geo_store()
        ms = MaintenanceStore(
            db=rt["db"],
            job_manager=rt["job_manager"],
            geospatial_store=gs,
            api_cfg=rt["config"],
        )
        rt["maintenance_store"] = ms
    return ms


@v2_bp.route("/maintenance/overview", methods=["GET"])
def maintenance_overview():
    ms = _get_maint_store()
    include_provs, excl_non_genuine, _ = _parse_provenance_scope(default_scope="genuine")
    return _strict_json_response(
        ms.get_overview(include_provenances=include_provs, exclude_non_genuine=excl_non_genuine),
        200,
    )


@v2_bp.route("/maintenance/review-queue", methods=["GET"])
def maintenance_review_queue():
    rt = _get_runtime()
    cfg: APIConfig = rt["config"]
    ms = _get_maint_store()
    page, page_size = _get_pagination_params(cfg)
    job_id = request.args.get("job_id") or None
    priority_cat = (request.args.get("priority_category") or "").strip().upper() or None
    review_status = (request.args.get("review_status") or "").strip().upper() or None
    quality_cat = (request.args.get("quality_category") or "").strip().upper() or None
    include_provs, excl_non_genuine, _ = _parse_provenance_scope(default_scope="genuine")
    return _strict_json_response(
        ms.get_review_queue(
            job_id=job_id,
            priority_category=priority_cat,
            review_status=review_status,
            quality_category=quality_cat,
            include_provenances=include_provs,
            exclude_non_genuine=excl_non_genuine,
            page=page,
            page_size=page_size,
        ),
        200,
    )


@v2_bp.route("/maintenance/reviews", methods=["GET", "POST"])
def maintenance_reviews_collection():
    rt = _get_runtime()
    cfg: APIConfig = rt["config"]
    ms = _get_maint_store()
    if request.method == "GET":
        page, page_size = _get_pagination_params(cfg)
        job_id = request.args.get("job_id") or None
        review_status = (request.args.get("review_status") or "").strip().upper() or None
        return _strict_json_response(
            ms.list_reviews(
                job_id=job_id,
                review_status=review_status,
                page=page,
                page_size=page_size,
            ),
            200,
        )

    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        raise BadRequestError("Request body must be a JSON object", error_code="INVALID_REQUEST")
    if "job_id" not in body or "track_id" not in body:
        raise BadRequestError("job_id and track_id are required", error_code="MISSING_FIELDS")

    created = ms.create_review(
        job_id=body["job_id"],
        track_id=body["track_id"],
        track_generation_id=body.get("track_generation_id", 0),
        review_status=body.get("review_status", "PENDING_REVIEW"),
        reviewer_label=body.get("reviewer_label"),
        review_notes=body.get("review_notes"),
        cluster_id=body.get("cluster_id"),
    )
    return _strict_json_response(created, 201)


@v2_bp.route("/maintenance/reviews/<review_id>", methods=["GET", "PATCH"])
def maintenance_review_item(review_id: str):
    ms = _get_maint_store()
    if request.method == "GET":
        return _strict_json_response(ms.get_review(review_id), 200)

    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        raise BadRequestError("Request body must be a JSON object", error_code="INVALID_REQUEST")
    updated = ms.update_review(
        review_id,
        review_status=body.get("review_status"),
        reviewer_label=body.get("reviewer_label"),
        review_notes=body.get("review_notes"),
    )
    return _strict_json_response(updated, 200)


@v2_bp.route("/jobs/<job_id>/maintenance", methods=["GET"])
def job_maintenance(job_id: str):
    ms = _get_maint_store()
    return _strict_json_response(ms.get_job_maintenance(job_id), 200)


@v2_bp.route("/jobs/<job_id>/segments", methods=["GET"])
def job_segments(job_id: str):
    ms = _get_maint_store()
    raw_seg = request.args.get("segment_length_m", "50.0")
    try:
        seg_len = float(raw_seg)
    except ValueError as exc:
        raise BadRequestError("segment_length_m must be numeric", error_code="INVALID_PARAMETER") from exc
    if seg_len < 10.0 or seg_len > 1000.0:
        raise BadRequestError("segment_length_m must be in [10, 1000]", error_code="INVALID_PARAMETER")
    return _strict_json_response(ms.get_job_segments(job_id, segment_length_m=seg_len), 200)


@v2_bp.route("/maintenance/compare", methods=["GET"])
def maintenance_compare():
    ms = _get_maint_store()
    raw_ids = request.args.get("job_ids", "")
    job_ids = [s.strip() for s in raw_ids.split(",") if s.strip()]
    return _strict_json_response(ms.compare_jobs(job_ids), 200)


@v2_bp.route("/jobs/<job_id>/report", methods=["GET"])
def job_report(job_id: str):
    ms = _get_maint_store()
    gs = _get_geo_store()
    jm: JobManager = _get_runtime()["job_manager"]

    fmt = (request.args.get("format") or "json").strip().lower()
    if fmt not in ("json", "html", "csv", "geojson"):
        raise BadRequestError(
            f"Unsupported report format '{fmt}'; expected json, html, csv, or geojson",
            error_code="INVALID_REPORT_FORMAT",
        )
    as_download = request.args.get("download", "0").lower() in ("1", "true", "yes")
    short_id = validate_uuid(job_id)[:8]

    if fmt == "geojson":
        job = jm.get_job(job_id, include_internal_paths=True)
        if job["status"] != JOB_STATUS_COMPLETED:
            raise InvalidJobStateError(
                f"Report unavailable while job status is '{job['status']}'",
                error_code="RESULTS_UNAVAILABLE",
            )
        res = gs.get_job_locations(job, include_geojson=True)
        gj = sanitize_json_value(res.get("geojson") or {"type": "FeatureCollection", "features": []})
        resp = Response(json.dumps(gj, indent=2, allow_nan=False), status=200, mimetype="application/geo+json")
        if as_download:
            resp.headers["Content-Disposition"] = f'attachment; filename="roadvision_report_{short_id}.geojson"'
        return resp

    payload = ms.build_job_report(job_id)
    if fmt == "html":
        html_str = render_inspection_report_html(payload)
        resp = Response(html_str, status=200, mimetype="text/html; charset=utf-8")
        if as_download:
            resp.headers["Content-Disposition"] = f'attachment; filename="roadvision_report_{short_id}.html"'
        return resp
    if fmt == "csv":
        csv_str = render_inspection_report_csv(payload)
        resp = Response(csv_str, status=200, mimetype="text/csv; charset=utf-8")
        if as_download:
            resp.headers["Content-Disposition"] = f'attachment; filename="roadvision_report_{short_id}.csv"'
        return resp

    clean = sanitize_json_value(payload)
    resp = Response(json.dumps(clean, indent=2, allow_nan=False), status=200, mimetype="application/json")
    if as_download:
        resp.headers["Content-Disposition"] = f'attachment; filename="roadvision_report_{short_id}.json"'
    return resp

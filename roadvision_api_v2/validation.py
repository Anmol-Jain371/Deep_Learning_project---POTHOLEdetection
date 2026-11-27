"""
Upload and job-parameter validation for RoadVision AI 2.0.

Enforces strict multipart upload bounds, server-generated filenames inside isolated
job directories, actual OpenCV decodability checks, resolution/duration limits, and
rejection of unsupported config overrides.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import cv2

from roadvision_v2.config import VALID_MODES, RoadVisionV2Config, load_v2_config

from .configuration import APIConfig
from .errors import (
    BadRequestError,
    ServiceUnavailableError,
    UnsupportedMediaError,
    UploadTooLargeError,
)

ALLOWED_FORM_FIELDS = {
    "mode",
    "enable_depth",
    "enable_geometry",
    "diagnostics",
    "save_video",
    "save_json",
    "save_csv",
    "max_frames",
}

FORBIDDEN_OVERRIDE_FIELDS = {
    "checkpoint_path",
    "detector_checkpoint",
    "depth_checkpoint",
    "source",
    "output_dir",
    "device",
    "weights",
    "config",
    "config_path",
    "model_path",
    "url",
    "source_url",
}

_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class ValidatedJobRequest:
    mode: str
    enable_depth: Optional[bool]
    enable_geometry: bool
    diagnostics: bool
    save_video: bool
    save_json: bool
    save_csv: bool
    max_frames: Optional[int]
    v2_config_snapshot: Dict[str, Any]


@dataclass
class ValidatedUploadMedia:
    original_filename: str
    validated_input_path: str
    upload_job_dir: str
    size_bytes: int
    width: int
    height: int
    source_fps: Optional[float]
    estimated_total_frames: Optional[int]
    estimated_duration_sec: Optional[float]


def sanitize_display_filename(raw_name: str) -> str:
    """Produce a safe display-only filename without directory traversal components."""
    base = os.path.basename((raw_name or "").replace("\\", "/")).strip()
    cleaned = _UNSAFE_CHARS_RE.sub("_", base).strip("._")
    if not cleaned:
        return "uploaded_video.mp4"
    return cleaned[:160]


def parse_optional_bool(raw: Optional[str], default: Optional[bool], field_name: str) -> Optional[bool]:
    if raw is None or str(raw).strip() == "":
        return default
    norm = str(raw).strip().lower()
    if norm in ("1", "true", "yes", "on"):
        return True
    if norm in ("0", "false", "no", "off"):
        return False
    raise BadRequestError(
        f"Invalid boolean value for '{field_name}': expected true/false",
        error_code="INVALID_PARAMETER",
        details={"field": field_name},
    )


def parse_optional_max_frames(raw: Optional[str], hard_cap: int) -> Optional[int]:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        val = int(str(raw).strip())
    except ValueError as exc:
        raise BadRequestError(
            "Invalid integer value for 'max_frames'",
            error_code="INVALID_PARAMETER",
            details={"field": "max_frames"},
        ) from exc
    if val < 1 or val > hard_cap:
        raise BadRequestError(
            f"'max_frames' must be between 1 and {hard_cap}",
            error_code="INVALID_PARAMETER",
            details={"field": "max_frames", "min": 1, "max": hard_cap},
        )
    return val


def validate_job_options(
    form: Mapping[str, Any],
    api_cfg: APIConfig,
    check_assets: bool = True,
) -> ValidatedJobRequest:
    """Validate multipart job options and verify effective V2 configuration before upload save."""
    unknown_fields = [k for k in form.keys() if k not in ALLOWED_FORM_FIELDS]
    if unknown_fields:
        for k in unknown_fields:
            if k in FORBIDDEN_OVERRIDE_FIELDS:
                raise BadRequestError(
                    f"Parameter '{k}' is not permitted via the public API",
                    error_code="FORBIDDEN_OVERRIDE",
                    details={"field": k},
                )
        raise BadRequestError(
            f"Unsupported form field(s): {', '.join(sorted(unknown_fields))}",
            error_code="UNSUPPORTED_PARAMETER",
            details={"fields": sorted(unknown_fields)},
        )

    mode = str(form.get("mode") or "balanced").strip().lower()
    if mode not in VALID_MODES:
        raise BadRequestError(
            f"Unsupported mode '{mode}'; expected one of {list(VALID_MODES)}",
            error_code="INVALID_MODE",
            details={"allowed_modes": list(VALID_MODES)},
        )

    enable_depth = parse_optional_bool(form.get("enable_depth"), None, "enable_depth")
    enable_geometry = bool(parse_optional_bool(form.get("enable_geometry"), False, "enable_geometry"))
    diagnostics = bool(parse_optional_bool(form.get("diagnostics"), False, "diagnostics"))
    save_video = bool(parse_optional_bool(form.get("save_video"), True, "save_video"))
    save_json = bool(parse_optional_bool(form.get("save_json"), True, "save_json"))
    save_csv = bool(parse_optional_bool(form.get("save_csv"), True, "save_csv"))
    max_frames = parse_optional_max_frames(form.get("max_frames"), api_cfg.maximum_processed_frames)

    overrides: Optional[Dict[str, Any]] = None
    if diagnostics or enable_geometry:
        overrides = {}
        if diagnostics:
            overrides["diagnostics"] = {
                "enable_horizon_proxy": True,
                "show_rejection_reasons": True,
                "visualize_annular": True,
            }
        if enable_geometry:
            overrides["geometry_snapshot"] = {"enabled": True}

    try:
        v2_cfg: RoadVisionV2Config = load_v2_config(
            mode=mode,
            enable_depth=enable_depth,
            overrides=overrides,
            check_assets=check_assets,
        )
    except FileNotFoundError as exc:
        raise ServiceUnavailableError(
            "Required V2 model checkpoint is unavailable on this server.",
            error_code="MODEL_CHECKPOINT_UNAVAILABLE",
        ) from exc
    except ValueError as exc:
        raise BadRequestError(str(exc), error_code="INVALID_MODE_CONFIG") from exc

    snap = v2_cfg.to_dict()
    snap.pop("project_root", None)
    # Strip internal checkpoint paths from public snapshot
    if "detector" in snap and isinstance(snap["detector"], dict):
        snap["detector"]["checkpoint_path"] = os.path.basename(
            str(snap["detector"].get("checkpoint_path") or "")
        )
        snap["detector"]["tracker_config"] = os.path.basename(
            str(snap["detector"].get("tracker_config") or "")
        )
    if "depth" in snap and isinstance(snap["depth"], dict) and snap["depth"].get("checkpoint_path"):
        snap["depth"]["checkpoint_path"] = os.path.basename(str(snap["depth"]["checkpoint_path"]))
    snap["execution_options"] = {
        "diagnostics": diagnostics,
        "enable_geometry": enable_geometry,
        "save_video": save_video,
        "save_json": save_json,
        "save_csv": save_csv,
        "max_frames": max_frames,
    }

    return ValidatedJobRequest(
        mode=mode,
        enable_depth=enable_depth,
        enable_geometry=enable_geometry,
        diagnostics=diagnostics,
        save_video=save_video,
        save_json=save_json,
        save_csv=save_csv,
        max_frames=max_frames,
        v2_config_snapshot=snap,
    )


def save_and_validate_upload(
    file_storage: Any,
    job_id: str,
    api_cfg: APIConfig,
) -> ValidatedUploadMedia:
    """
    Stream uploaded video into an isolated server-side directory, enforce byte size,
    verify real OpenCV frame decodability, and validate resolution/duration limits.
    Cleans up any partial/rejected files before raising.
    """
    if file_storage is None:
        raise BadRequestError("Missing 'video' file field in multipart request", error_code="MISSING_VIDEO")

    raw_filename = getattr(file_storage, "filename", "") or ""
    if not raw_filename.strip():
        raise BadRequestError("Uploaded video filename is empty", error_code="EMPTY_FILENAME")

    display_name = sanitize_display_filename(raw_filename)
    _, ext = os.path.splitext(display_name)
    ext_low = ext.lower()
    if ext_low not in api_cfg.allowed_video_formats:
        raise UnsupportedMediaError(
            f"Unsupported video format '{ext_low or 'unknown'}'. Allowed: {api_cfg.allowed_video_formats}",
            details={"allowed_video_formats": api_cfg.allowed_video_formats},
        )

    upload_job_dir = os.path.join(api_cfg.resolved_upload_dir, job_id)
    os.makedirs(upload_job_dir, exist_ok=True)
    server_path = os.path.join(upload_job_dir, f"{job_id}_input{ext_low}")

    total_bytes = 0
    chunk_size = 1024 * 1024  # 1 MiB
    try:
        stream = file_storage.stream
        with open(server_path, "wb") as out_fh:
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > api_cfg.maximum_upload_size:
                    raise UploadTooLargeError(
                        f"Upload exceeds maximum allowed size of {api_cfg.maximum_upload_size} bytes",
                        details={"maximum_upload_size": api_cfg.maximum_upload_size},
                    )
                out_fh.write(chunk)

        if total_bytes <= 0:
            raise BadRequestError("Uploaded video file is empty (0 bytes)", error_code="EMPTY_UPLOAD")

        cap = cv2.VideoCapture(server_path)
        try:
            if not cap.isOpened():
                raise UnsupportedMediaError("Uploaded file could not be decoded as a valid video")
            ok, first_frame = cap.read()
            if not ok or first_frame is None or first_frame.size == 0:
                raise UnsupportedMediaError("Uploaded video contains no readable frames or is corrupt")

            height, width = int(first_frame.shape[0]), int(first_frame.shape[1])
            max_w, max_h = api_cfg.maximum_video_resolution
            if width <= 0 or height <= 0 or width > max_w or height > max_h:
                raise BadRequestError(
                    f"Video resolution {width}x{height} exceeds limit {max_w}x{max_h}",
                    error_code="RESOLUTION_LIMIT_EXCEEDED",
                    details={"width": width, "height": height, "max_width": max_w, "max_height": max_h},
                )

            fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            source_fps = fps_raw if (0.0 < fps_raw <= 240.0) else None
            fc_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            est_total = fc_raw if fc_raw > 0 else None

            est_duration: Optional[float] = None
            if source_fps is not None and est_total is not None:
                est_duration = round(est_total / source_fps, 3)
                if est_duration > api_cfg.maximum_video_duration:
                    raise BadRequestError(
                        f"Video duration {est_duration:.1f}s exceeds limit {api_cfg.maximum_video_duration:.1f}s",
                        error_code="DURATION_LIMIT_EXCEEDED",
                        details={
                            "estimated_duration_sec": est_duration,
                            "maximum_video_duration": api_cfg.maximum_video_duration,
                        },
                    )
        finally:
            cap.release()

        return ValidatedUploadMedia(
            original_filename=display_name,
            validated_input_path=server_path,
            upload_job_dir=upload_job_dir,
            size_bytes=total_bytes,
            width=width,
            height=height,
            source_fps=source_fps,
            estimated_total_frames=est_total,
            estimated_duration_sec=est_duration,
        )
    except Exception:
        shutil.rmtree(upload_job_dir, ignore_errors=True)
        raise


def cleanup_job_upload_dir(upload_job_dir: str) -> None:
    """Remove an isolated job upload directory safely."""
    if upload_job_dir and os.path.isdir(upload_job_dir):
        shutil.rmtree(upload_job_dir, ignore_errors=True)

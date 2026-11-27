"""
Verified artifact manifest builder and secure HTTP Range file serving for RoadVision AI 2.0.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from flask import Response, request, send_file

from .errors import BadRequestError, NotFoundError
from .job_manager import JOB_STATUS_CANCELLED, JOB_STATUS_FAILED, validate_uuid

_SAFE_ARTIFACT_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_RANGE_HEADER_RE = re.compile(r"bytes=(\d*)-(\d*)")

ARTIFACT_DEFS = (
    ("annotated_video", "annotated_video", "_annotated.mp4", "video/mp4"),
    ("frames_json", "observations_json", "_frames.json", "application/json"),
    ("tracks_json", "tracks_json", "_tracks.json", "application/json"),
    ("performance_csv", "performance_csv", "_performance.csv", "text/csv"),
    ("session_summary_json", "session_summary_json", "_session_summary.json", "application/json"),
    ("geometry_manifest_json", "geometry_manifest_json", "_geometry_manifest.json", "application/json"),
)


def build_artifact_manifest(
    job_id: str,
    mode: str,
    stem: str,
    output_dir: str,
    job_status: str,
    video_codec: Optional[str] = None,
    video_browser_compatible: Optional[bool] = None,
) -> Tuple[Dict[str, Any], Dict[str, bool]]:
    """
    Inspect the job's isolated output directory for actual completed output files and build
    a verified manifest + boolean availability map.
    """
    prefix = f"phase8a_{mode}_{stem}"
    is_partial = job_status in (JOB_STATUS_CANCELLED, JOB_STATUS_FAILED)
    real_out = os.path.realpath(output_dir)
    items: List[Dict[str, Any]] = []
    availability: Dict[str, bool] = {}

    for artifact_id, artifact_type, suffix, mime_type in ARTIFACT_DEFS:
        filename = f"{prefix}{suffix}"
        candidate = os.path.realpath(os.path.join(real_out, filename))
        exists = (
            os.path.isfile(candidate)
            and os.path.commonpath([real_out, candidate]) == real_out
            and not os.path.islink(os.path.join(real_out, filename))
        )
        availability[artifact_id] = bool(exists)
        if not exists:
            continue
        st = os.stat(candidate)
        created_iso = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        entry: Dict[str, Any] = {
            "artifact_id": artifact_id,
            "filename": filename,
            "artifact_type": artifact_type,
            "mime_type": mime_type,
            "size_bytes": int(st.st_size),
            "created_at": created_iso,
            "download_url": f"/api/v2/jobs/{job_id}/artifacts/{artifact_id}",
            "available": True,
            "partial": bool(is_partial),
        }
        if artifact_id == "annotated_video":
            entry["codec"] = video_codec
            entry["browser_compatible"] = video_browser_compatible
        items.append(entry)

    manifest = {
        "job_id": job_id,
        "partial": bool(is_partial),
        "artifacts": items,
    }
    return manifest, availability


def resolve_artifact_entry(
    job_record: Dict[str, Any],
    artifact_name: str,
    allowed_artifact_root: str,
) -> Tuple[Dict[str, Any], str]:
    """
    Validate requested artifact against the job's verified manifest and resolve
    its real path strictly within both the job output directory and allowed_artifact_root.
    """
    jid = validate_uuid(job_record["job_id"])
    if not artifact_name or "/" in artifact_name or "\\" in artifact_name or ".." in artifact_name:
        raise BadRequestError("Invalid artifact identifier", error_code="INVALID_ARTIFACT_NAME")
    if not _SAFE_ARTIFACT_NAME_RE.match(artifact_name):
        raise BadRequestError("Invalid artifact identifier", error_code="INVALID_ARTIFACT_NAME")

    manifest = job_record.get("artifact_manifest") or {}
    artifacts = manifest.get("artifacts") or []
    matched: Optional[Dict[str, Any]] = None
    for item in artifacts:
        if item.get("artifact_id") == artifact_name or item.get("filename") == artifact_name:
            matched = item
            break

    if matched is None or not matched.get("available"):
        raise NotFoundError(
            f"Artifact '{artifact_name}' not available for job '{jid}'",
            error_code="ARTIFACT_NOT_FOUND",
        )

    output_dir = job_record.get("output_dir") or ""
    if not output_dir:
        raise NotFoundError("Job output directory unavailable", error_code="ARTIFACT_NOT_FOUND")

    real_root = os.path.realpath(allowed_artifact_root)
    real_job_dir = os.path.realpath(output_dir)
    if os.path.commonpath([real_root, real_job_dir]) != real_root:
        raise NotFoundError("Artifact path outside approved root", error_code="ARTIFACT_FORBIDDEN")

    raw_file = os.path.join(real_job_dir, matched["filename"])
    if os.path.islink(raw_file):
        raise NotFoundError("Symlink artifacts are not permitted", error_code="ARTIFACT_FORBIDDEN")

    real_file = os.path.realpath(raw_file)
    if os.path.commonpath([real_job_dir, real_file]) != real_job_dir:
        raise NotFoundError("Artifact escapes job directory", error_code="ARTIFACT_FORBIDDEN")
    if not os.path.isfile(real_file):
        raise NotFoundError("Artifact file missing on disk", error_code="ARTIFACT_NOT_FOUND")

    return matched, real_file


def serve_artifact_response(entry: Dict[str, Any], file_path: str, as_attachment: bool = False) -> Response:
    """Serve an artifact file with HTTP Byte-Range support for video playback."""
    mime_type = entry.get("mime_type") or "application/octet-stream"
    filename = entry.get("filename") or os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("Range")

    if range_header and not as_attachment:
        match = _RANGE_HEADER_RE.match(range_header.strip())
        if not match:
            return Response(status=416, headers={"Content-Range": f"bytes */{file_size}"})
        start_str, end_str = match.groups()
        if start_str == "" and end_str == "":
            return Response(status=416, headers={"Content-Range": f"bytes */{file_size}"})
        if start_str != "":
            start = int(start_str)
            end = int(end_str) if end_str != "" else file_size - 1
        else:
            suffix_len = int(end_str)
            start = max(0, file_size - suffix_len)
            end = file_size - 1

        if start >= file_size or end < start:
            return Response(status=416, headers={"Content-Range": f"bytes */{file_size}"})
        end = min(end, file_size - 1)
        length = end - start + 1

        def _stream_chunk():
            with open(file_path, "rb") as fh:
                fh.seek(start)
                remaining = length
                buf_size = 64 * 1024
                while remaining > 0:
                    data = fh.read(min(buf_size, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        resp = Response(_stream_chunk(), status=206, mimetype=mime_type, direct_passthrough=True)
        resp.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        resp.headers["Accept-Ranges"] = "bytes"
        resp.headers["Content-Length"] = str(length)
        resp.headers["Content-Disposition"] = f'inline; filename="{filename}"'
        return resp

    resp = send_file(
        file_path,
        mimetype=mime_type,
        as_attachment=as_attachment,
        download_name=filename,
    )
    resp.headers["Accept-Ranges"] = "bytes"
    return resp

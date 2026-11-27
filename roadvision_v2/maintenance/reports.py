"""
Phase 12: Structured Inspection Report Generation (JSON, Print-Friendly HTML, CSV).

Constructs reports strictly from persisted inspection outputs, manual reviews,
geospatial metadata, and verified thumbnails without fabricating measurements.
"""

from __future__ import annotations

import csv
import html
import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from .analytics import MAINTENANCE_SCIENTIFIC_DISCLAIMER


def build_inspection_report_payload(
    *,
    job: Dict[str, Any],
    coverage: Dict[str, Any],
    maintenance: Dict[str, Any],
    segments: Optional[Dict[str, Any]] = None,
    reviews: Optional[Sequence[Dict[str, Any]]] = None,
    geometry_manifest: Optional[Dict[str, Any]] = None,
    representative_crops: Optional[Sequence[Dict[str, Any]]] = None,
    observations_sample: Optional[Sequence[Dict[str, Any]]] = None,
    pipeline_version: str = "2.0.0-phase12",
) -> Dict[str, Any]:
    """Assemble structured, strictly factual inspection report payload."""
    summary = job.get("summary") or {}
    cfg_snap = job.get("config_snapshot") or {}
    quality_summary = summary.get("quality_summary") or {}

    return {
        "report_version": "roadvision_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": job.get("pipeline_version") or pipeline_version,
        "scoring_version": maintenance.get("scoring_version"),
        "inspection_metadata": {
            "job_id": job.get("job_id"),
            "original_filename": job.get("original_filename"),
            "status": job.get("status"),
            "mode": job.get("mode"),
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "frames_processed": job.get("processed_frames") or summary.get("total_frames") or 0,
            "total_frames": job.get("total_frames"),
            "source_fps": summary.get("source_fps"),
        },
        "model_configuration": {
            "detector": cfg_snap.get("detector") or {},
            "depth": cfg_snap.get("depth") or {},
            "cavity": cfg_snap.get("cavity") or {},
            "scheduler": cfg_snap.get("scheduler") or {},
            "geometry_recording_enabled": bool(summary.get("geometry_recording_enabled", False)),
        },
        "detection_and_measurement_summary": {
            "total_hazard_observations": maintenance.get("total_observations", 0),
            "unique_session_tracks": maintenance.get("candidate_count", 0),
            "measurement_coverage": coverage,
            "quality_summary": quality_summary,
        },
        "geometry_availability": {
            "geometry_available": bool(maintenance.get("geometry_available", False)),
            "snapshot_count": (geometry_manifest or {}).get("snapshot_count", 0),
            "global_min_residual": (geometry_manifest or {}).get("global_min_residual"),
            "global_max_residual": (geometry_manifest or {}).get("global_max_residual"),
        },
        "geospatial_summary": segments or {
            "gps_available": False,
            "geographical_analysis_unavailable": True,
        },
        "maintenance_summary": {
            "priority_distribution": maintenance.get("priority_distribution") or {},
            "review_status_distribution": maintenance.get("review_status_distribution") or {},
            "quality_distribution": maintenance.get("quality_distribution") or {},
            "scoring_config": maintenance.get("scoring_config") or {},
        },
        "track_candidates": maintenance.get("candidates") or [],
        "manual_reviews": list(reviews or []),
        "representative_images": list(representative_crops or []),
        "observation_sample": list(observations_sample or []),
        "processing_telemetry": {
            "elapsed_sec": summary.get("elapsed_sec"),
            "end_to_end_fps": job.get("throughput_fps") or summary.get("end_to_end_fps"),
            "model_init_sec": summary.get("model_init_sec"),
            "depth_invocation_count": job.get("depth_invocation_count"),
            "scheduler_request_count": job.get("scheduler_request_count"),
            "failed_depth_count": job.get("failed_depth_count"),
            "depth_invocation_rate": summary.get("depth_invocation_rate"),
            "peak_allocated_gpu_mb": summary.get("peak_allocated_gpu_mb"),
            "peak_reserved_gpu_mb": summary.get("peak_reserved_gpu_mb"),
        },
        "scientific_limitations": [
            "Relative inverse-depth residuals are uncalibrated research indicators, NOT metric cm/mm depth or physical volume.",
            "Session track counts reflect ByteTrack IDs, NOT independently verified unique physical potholes.",
            "Heuristic quality and review-priority indices are NOT calibrated probabilities, Pavement Condition Index (PCI) values, or repair urgency scores.",
            "GPS positions (where available) represent approximate vehicle/camera coordinates, NOT surveyed surface geometry.",
            "Human visual confirmation indicates visual appearance only unless physical field inspection is separately performed.",
        ],
        "scientific_disclaimer": MAINTENANCE_SCIENTIFIC_DISCLAIMER,
    }


def _esc(val: Any) -> str:
    if val is None:
        return "Unavailable"
    return html.escape(str(val))


def render_inspection_report_csv(payload: Dict[str, Any]) -> str:
    """Export track candidates & manual review states to UTF-8 CSV."""
    buf = io.StringIO()
    fieldnames = [
        "job_id",
        "track_id",
        "track_generation_id",
        "observation_count",
        "first_seen_frame_idx",
        "last_seen_frame_idx",
        "detector_confidence_median",
        "fresh_valid_count",
        "cached_valid_count",
        "fresh_invalid_count",
        "expired_count",
        "not_evaluated_count",
        "quality_category",
        "median_fresh_quality_index",
        "evidence_sufficiency",
        "median_relative_score",
        "review_priority_category",
        "review_priority_index",
        "review_status",
        "reviewer_label",
        "review_notes",
        "lat",
        "lon",
        "gps_sync_quality",
        "cluster_id",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for c in payload.get("track_candidates") or []:
        row = {k: c.get(k) for k in fieldnames}
        writer.writerow(row)
    return buf.getvalue()


def render_inspection_report_html(payload: Dict[str, Any]) -> str:
    """Render a standalone, print-friendly HTML5 inspection report (browser Save-as-PDF ready)."""
    meta = payload.get("inspection_metadata") or {}
    det_meas = payload.get("detection_and_measurement_summary") or {}
    cov = det_meas.get("measurement_coverage") or {}
    sc = cov.get("counts_by_state") or {}
    maint = payload.get("maintenance_summary") or {}
    geo = payload.get("geospatial_summary") or {}
    telem = payload.get("processing_telemetry") or {}
    model_cfg = payload.get("model_configuration") or {}
    depth_cfg = model_cfg.get("depth") or {}

    track_rows_html = []
    for c in payload.get("track_candidates") or []:
        reasons_txt = "; ".join(c.get("priority_reasons") or [])
        track_rows_html.append(
            "<tr>"
            f"<td>#{_esc(c.get('track_id'))} (g{_esc(c.get('track_generation_id'))})</td>"
            f"<td>{_esc(c.get('observation_count'))}</td>"
            f"<td>{_esc(c.get('detector_confidence_median'))}</td>"
            f"<td>{_esc(c.get('quality_category'))} ({_esc(c.get('median_fresh_quality_index'))})</td>"
            f"<td>{_esc(c.get('median_relative_score'))}</td>"
            f"<td><strong>{_esc(c.get('review_priority_category'))}</strong> ({_esc(c.get('review_priority_index'))})</td>"
            f"<td>{_esc(c.get('review_status'))}</td>"
            f"<td>{_esc(c.get('review_notes'))}</td>"
            f"<td>{_esc(reasons_txt)}</td>"
            "</tr>"
        )

    rep_imgs_html = []
    for img in payload.get("representative_images") or []:
        if img.get("source_crop_data_url"):
            rep_imgs_html.append(
                "<div class='thumb-card'>"
                f"<img src='{html.escape(str(img['source_crop_data_url']), quote=True)}' alt='Track {_esc(img.get('track_id'))} crop' />"
                f"<div class='thumb-cap'>Track #{_esc(img.get('track_id'))} · Frame {_esc(img.get('frame_idx'))} ({_esc(img.get('timestamp_sec'))}s) · {_esc(img.get('snapshot_id'))}</div>"
                "</div>"
            )

    lims_html = "".join(f"<li>{_esc(item)}</li>" for item in (payload.get("scientific_limitations") or []))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>RoadVision AI 2.0 Inspection Report — {_esc(meta.get('job_id'))}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color: #0f172a; background: #f8fafc; margin: 0; padding: 24px; line-height: 1.45; }}
    .container {{ max-width: 1120px; margin: 0 auto; background: #ffffff; padding: 28px; border: 1px solid #cbd5e1; border-radius: 8px; }}
    h1 {{ margin: 0 0 4px; font-size: 22px; color: #0f172a; }}
    h2 {{ margin: 22px 0 10px; font-size: 15px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; color: #0e7490; }}
    .sub {{ font-size: 12px; color: #475569; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; }}
    .kv {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 10px; }}
    .kv .k {{ font-size: 11px; color: #64748b; display: block; }}
    .kv .v {{ font-size: 13px; font-weight: 600; color: #0f172a; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; font-weight: 600; }}
    .disclaimer {{ margin-top: 18px; padding: 12px; background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; font-size: 12px; color: #78350f; }}
    .thumbs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .thumb-card {{ border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px; background: #f8fafc; }}
    .thumb-card img {{ width: 100%; max-height: 170px; object-fit: contain; background: #020617; border-radius: 4px; }}
    .thumb-cap {{ font-size: 11px; color: #475569; margin-top: 6px; }}
    @media print {{
      body {{ background: #ffffff; padding: 0; }}
      .container {{ border: none; padding: 8px; }}
      .no-print {{ display: none; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
      <div>
        <h1>RoadVision AI 2.0 — Maintenance &amp; Inspection Report</h1>
        <div class="sub">
          Generated: {_esc(payload.get('generated_at'))} · Pipeline: {_esc(payload.get('pipeline_version'))} · Scoring: {_esc(payload.get('scoring_version'))}
        </div>
      </div>
      <button class="no-print" onclick="window.print()" style="padding:7px 14px;cursor:pointer;">Print / Save as PDF</button>
    </div>

    <h2>1. Inspection Metadata &amp; Configuration</h2>
    <div class="grid">
      <div class="kv"><span class="k">Job UUID</span><span class="v">{_esc(meta.get('job_id'))}</span></div>
      <div class="kv"><span class="k">Source Video</span><span class="v">{_esc(meta.get('original_filename'))}</span></div>
      <div class="kv"><span class="k">Mode / Status</span><span class="v">{_esc(meta.get('mode')).upper()} · {_esc(meta.get('status'))}</span></div>
      <div class="kv"><span class="k">Processed Frames</span><span class="v">{_esc(meta.get('frames_processed'))} / {_esc(meta.get('total_frames'))}</span></div>
      <div class="kv"><span class="k">Depth Model</span><span class="v">{_esc(depth_cfg.get('model_id')) if depth_cfg.get('enabled') else 'Disabled'} ({_esc(depth_cfg.get('input_size'))}px)</span></div>
      <div class="kv"><span class="k">Sustained Throughput</span><span class="v">{_esc(telem.get('end_to_end_fps'))} FPS · Elapsed {_esc(telem.get('elapsed_sec'))}s</span></div>
    </div>

    <h2>2. Measurement Coverage &amp; Maintenance Summary</h2>
    <div class="grid">
      <div class="kv"><span class="k">Session Candidate Tracks</span><span class="v">{_esc(det_meas.get('unique_session_tracks'))}</span></div>
      <div class="kv"><span class="k">Total Observations</span><span class="v">{_esc(det_meas.get('total_hazard_observations'))}</span></div>
      <div class="kv"><span class="k">Fresh / Cached Valid</span><span class="v">{_esc(sc.get('FRESH_VALID', 0))} / {_esc(sc.get('CACHED_VALID', 0))}</span></div>
      <div class="kv"><span class="k">Invalid / Expired / Unevaluated</span><span class="v">{_esc(sc.get('FRESH_INVALID', 0))} / {_esc(sc.get('EXPIRED', 0))} / {_esc(sc.get('NOT_EVALUATED', 0))}</span></div>
      <div class="kv"><span class="k">Priority Distribution</span><span class="v">{_esc(maint.get('priority_distribution'))}</span></div>
      <div class="kv"><span class="k">Manual Review Outcomes</span><span class="v">{_esc(maint.get('review_status_distribution'))}</span></div>
    </div>

    <h2>3. Geospatial &amp; Geometry Availability</h2>
    <div class="grid">
      <div class="kv"><span class="k">GPS Available</span><span class="v">{'Yes' if geo.get('gps_available') else 'No'}</span></div>
      <div class="kv"><span class="k">Observed Route Length</span><span class="v">{_esc(geo.get('observed_route_length_m'))} m</span></div>
      <div class="kv"><span class="k">Geometry Snapshots</span><span class="v">{_esc((payload.get('geometry_availability') or {}).get('snapshot_count'))}</span></div>
    </div>

    <h2>4. Session Candidate Track Summary &amp; Review Decisions</h2>
    <table>
      <thead>
        <tr>
          <th>Track</th>
          <th>Obs</th>
          <th>Conf Med</th>
          <th>Quality</th>
          <th>Rel Score (Uncal.)</th>
          <th>Review Priority</th>
          <th>Manual Status</th>
          <th>Reviewer Notes</th>
          <th>Priority Explanation</th>
        </tr>
      </thead>
      <tbody>
        {''.join(track_rows_html) if track_rows_html else '<tr><td colspan="9">No candidate tracks recorded.</td></tr>'}
      </tbody>
    </table>

    {('<h2>5. Verified Geometry ROI Thumbnails</h2><div class="thumbs">' + ''.join(rep_imgs_html) + '</div>') if rep_imgs_html else ''}

    <div class="disclaimer">
      <strong>Scientific &amp; Engineering Limitations:</strong>
      <ul style="margin:6px 0 0 18px;padding:0;">{lims_html}</ul>
    </div>
  </div>
</body>
</html>"""

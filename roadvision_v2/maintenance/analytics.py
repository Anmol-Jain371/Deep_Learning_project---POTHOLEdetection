"""
Phase 12: Explainable Road Maintenance Intelligence Analytics.

Synthesizes existing observation, track, Phase 9 quality, Phase 10 geometry,
and Phase 11 geospatial outputs into inspection-review queues, distance-based
road segment summaries, and compatible historical comparisons without running
additional inference or altering original model outputs.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from roadvision_v2.config import MaintenancePriorityConfig
from roadvision_v2.geospatial.matching import haversine_distance_m

MAINTENANCE_SCORING_VERSION = "heuristic_review_priority_v1"

REVIEW_STATUSES = (
    "PENDING_REVIEW",
    "IN_REVIEW",
    "CONFIRMED_VISUALLY",
    "REJECTED_FALSE_POSITIVE",
    "NEEDS_FIELD_INSPECTION",
)

REVIEW_PRIORITY_CATEGORIES = (
    "PRIORITY_REVIEW",
    "STANDARD_REVIEW",
    "INSUFFICIENT_EVIDENCE",
    "MANUAL_VERIFICATION_REQUIRED",
    "REVIEWED",
)

MAINTENANCE_SCIENTIFIC_DISCLAIMER = (
    "Phase 12 Maintenance Intelligence provides heuristic inspection-review prioritization "
    "and candidate summaries ONLY. Categories do NOT represent calibrated civil-engineering "
    "repair urgency, structural damage severity, or Pavement Condition Index (PCI) ratings. "
    "Human review confirms visual appearance only unless physical field inspection is recorded."
)


def _safe_float(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _median(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(float(x) for x in vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _extract_depth_signature(config_snapshot: Optional[Dict[str, Any]], summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = config_snapshot or {}
    dcfg = cfg.get("depth") if isinstance(cfg.get("depth"), dict) else {}
    ccfg = cfg.get("cavity") if isinstance(cfg.get("cavity"), dict) else {}
    summ = summary or {}
    depth_enabled = bool(summ.get("depth_enabled") if "depth_enabled" in summ else dcfg.get("enabled", False))
    return {
        "depth_enabled": depth_enabled,
        "depth_model_id": summ.get("depth_model_id") or dcfg.get("model_id") or ("none" if not depth_enabled else "unknown"),
        "depth_input_size": summ.get("depth_input_size") or dcfg.get("input_size"),
        "inference_precision": dcfg.get("precision"),
        "cavity_strategy": ccfg.get("strategy"),
    }


def compute_review_priority(
    candidate: Dict[str, Any],
    cfg: Optional[MaintenancePriorityConfig] = None,
) -> Dict[str, Any]:
    """
    Compute explainable inspection-review priority separately from evidence sufficiency.
    Unavailable/invalid measurements never automatically downgrade a persistent candidate to harmless.
    """
    cfg = cfg or MaintenancePriorityConfig()
    obs_count = int(candidate.get("observation_count") or 0)
    conf_med = _safe_float(candidate.get("detector_confidence_median"))
    fresh_valid = int(candidate.get("fresh_valid_count") or 0)
    cached_valid = int(candidate.get("cached_valid_count") or 0)
    fresh_invalid = int(candidate.get("fresh_invalid_count") or 0)
    expired = int(candidate.get("expired_count") or 0)
    not_eval = int(candidate.get("not_evaluated_count") or 0)

    qual_cat = str(candidate.get("quality_category") or "INSUFFICIENT_EVIDENCE")
    qual_idx = _safe_float(candidate.get("median_fresh_quality_index"))
    if qual_idx is None:
        qual_idx = _safe_float(candidate.get("median_quality_index"))

    evidence_sufficiency = str(candidate.get("evidence_sufficiency") or "INSUFFICIENT")
    rel_score = _safe_float(candidate.get("median_relative_score"))
    temporal_var = _safe_float(candidate.get("relative_depth_variability"))
    depth_enabled = bool(candidate.get("depth_enabled", False))

    distinct_trips = int(candidate.get("cluster_distinct_trip_count") or 0)
    contains_synth_gps = bool(candidate.get("contains_synthetic_gps", False))
    gps_sync_q = candidate.get("gps_sync_quality")
    review_status = str(candidate.get("review_status") or "PENDING_REVIEW")

    # Individual normalized sub-components (None when genuinely unavailable)
    c_persistence = _clamp01(obs_count / max(1, cfg.persistence_full_obs)) if obs_count > 0 else None
    c_det_conf = _clamp01((conf_med or 0.0) / 0.75) if conf_med is not None else None
    c_fresh = _clamp01(fresh_valid / max(1, cfg.fresh_full_count)) if depth_enabled else None
    c_quality = _clamp01(qual_idx) if qual_idx is not None else None
    c_rel = (
        _clamp01(max(0.0, rel_score) / max(cfg.relative_signal_ref_scale, 1e-4))
        if (depth_enabled and rel_score is not None and fresh_valid > 0)
        else None
    )
    c_temp = _clamp01(1.0 - min(1.0, temporal_var)) if temporal_var is not None else None
    if distinct_trips >= 2 and not contains_synth_gps:
        c_cross = _clamp01((distinct_trips - 1) / 2.0)
    elif candidate.get("has_gps"):
        c_cross = 0.0
    else:
        c_cross = None

    if gps_sync_q in ("EXACT", "GOOD"):
        c_gps = 1.0
    elif gps_sync_q == "ACCEPTABLE":
        c_gps = 0.65
    elif gps_sync_q in ("LOW_QUALITY", "GAP_REJECTED", "UNAVAILABLE"):
        c_gps = 0.20 if candidate.get("has_gps") else None
    else:
        c_gps = None

    components = {
        "track_persistence": round(c_persistence, 4) if c_persistence is not None else None,
        "detector_confidence": round(c_det_conf, 4) if c_det_conf is not None else None,
        "fresh_measurement_support": round(c_fresh, 4) if c_fresh is not None else None,
        "measurement_quality": round(c_quality, 4) if c_quality is not None else None,
        "relative_cavity_signal": round(c_rel, 4) if c_rel is not None else None,
        "temporal_stability": round(c_temp, 4) if c_temp is not None else None,
        "cross_trip_repetition": round(c_cross, 4) if c_cross is not None else None,
        "gps_sync_quality": round(c_gps, 4) if c_gps is not None else None,
    }
    weights = {
        "track_persistence": cfg.w_persistence,
        "detector_confidence": cfg.w_detector_confidence,
        "fresh_measurement_support": cfg.w_fresh_coverage,
        "measurement_quality": cfg.w_quality_index,
        "relative_cavity_signal": cfg.w_relative_signal,
        "temporal_stability": cfg.w_temporal_stability,
        "cross_trip_repetition": cfg.w_cross_trip,
        "gps_sync_quality": cfg.w_gps_sync,
    }

    avail_w = sum(weights[k] for k, v in components.items() if v is not None)
    priority_index: Optional[float] = None
    if avail_w > 0:
        raw_sum = sum(weights[k] * float(v) for k, v in components.items() if v is not None)
        priority_index = round(_clamp01(raw_sum / avail_w), 4)

    reasons: List[str] = []
    if review_status in ("CONFIRMED_VISUALLY", "REJECTED_FALSE_POSITIVE"):
        category = "REVIEWED"
        reasons.append(f"Manual review status is {review_status}; heuristic index preserved for audit traceability.")
    elif review_status == "NEEDS_FIELD_INSPECTION":
        category = "MANUAL_VERIFICATION_REQUIRED"
        reasons.append("Human reviewer flagged candidate as NEEDS_FIELD_INSPECTION.")
    else:
        persistent_detection = obs_count >= 3 and (conf_med is not None and conf_med >= 0.45)
        has_measurement_failure = (
            qual_cat == "INVALID"
            or (depth_enabled and fresh_invalid > fresh_valid)
            or expired > 0
        )
        missing_or_weak_depth = (
            not depth_enabled
            or fresh_valid == 0
            or qual_cat in ("LOW_QUALITY", "INSUFFICIENT_EVIDENCE")
            or evidence_sufficiency != "SUFFICIENT"
        )

        if persistent_detection and (has_measurement_failure or missing_or_weak_depth):
            category = "MANUAL_VERIFICATION_REQUIRED"
            if has_measurement_failure:
                reasons.append(
                    f"Persistent candidate (obs={obs_count}, conf_med={conf_med:.2f}) has invalid or expired depth observations "
                    f"(fresh_invalid={fresh_invalid}, expired={expired}); unavailable/failed measurements are not treated as low risk."
                )
            else:
                reasons.append(
                    f"Persistent candidate (obs={obs_count}, conf_med={conf_med:.2f}) lacks sufficient valid depth evidence "
                    f"(quality={qual_cat}, fresh_valid={fresh_valid}); manual visual verification required."
                )
        elif obs_count < 2 and fresh_valid == 0 and (conf_med is None or conf_med < 0.45):
            category = "INSUFFICIENT_EVIDENCE"
            reasons.append(
                f"Minimal track persistence (obs={obs_count}), low/unavailable confidence, and zero fresh depth support."
            )
        elif (
            (priority_index is not None and priority_index >= cfg.priority_review_min)
            or (distinct_trips >= 2 and not contains_synth_gps)
        ):
            category = "PRIORITY_REVIEW"
            reasons.append(
                f"Composite review-priority index ({priority_index}) meets PRIORITY_REVIEW threshold ({cfg.priority_review_min})."
            )
            if distinct_trips >= 2 and not contains_synth_gps:
                reasons.append(f"Repeated candidate across {distinct_trips} GPS-matched inspection sessions.")
        else:
            category = "STANDARD_REVIEW"
            reasons.append(
                f"Candidate has moderate inspection evidence (obs={obs_count}, fresh_valid={fresh_valid}, cached_valid={cached_valid})."
            )

    if not depth_enabled:
        reasons.append("Depth inference disabled in this mode; ranking relies on detection persistence & confidence only.")
    elif not_eval == obs_count and obs_count > 0:
        reasons.append("No depth frames evaluated on this track (scheduler/horizon skipped).")
    if contains_synth_gps:
        reasons.append("Associated geospatial cluster includes synthetic GPS fixture data; excluded from automated cross-trip boost.")

    return {
        "review_priority_category": category,
        "review_priority_index": priority_index,
        "evidence_sufficiency": evidence_sufficiency,
        "priority_reasons": reasons,
        "component_scores": components,
        "scoring_version": cfg.scoring_version,
        "scoring_config": asdict(cfg),
        "is_metric_severity": False,
        "disclaimer": MAINTENANCE_SCIENTIFIC_DISCLAIMER,
    }


def aggregate_job_maintenance(
    *,
    job: Dict[str, Any],
    tracks: Sequence[Dict[str, Any]],
    observations: Sequence[Dict[str, Any]],
    geolocations: Optional[Sequence[Dict[str, Any]]] = None,
    clusters: Optional[Sequence[Dict[str, Any]]] = None,
    geometry_manifest: Optional[Dict[str, Any]] = None,
    manual_reviews_by_key: Optional[Dict[Tuple[int, int], Dict[str, Any]]] = None,
    cfg: Optional[MaintenancePriorityConfig] = None,
) -> Dict[str, Any]:
    """Build deterministic track-level maintenance review candidates & job summary from stored artifacts."""
    cfg = cfg or MaintenancePriorityConfig()
    job_id = str(job["job_id"])
    summary = job.get("summary") or {}
    depth_sig = _extract_depth_signature(job.get("config_snapshot"), summary)

    # Index observations by (track_id, track_generation_id)
    obs_by_track: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for obs in observations:
        det = obs.get("detection") or {}
        tid = int(det.get("track_id", -1))
        tgen = int(det.get("track_generation_id", 0))
        if tid < 0:
            continue
        obs_by_track.setdefault((tid, tgen), []).append(obs)

    # Index geolocations by (track_id, track_generation_id)
    geo_by_track: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for g in geolocations or []:
        geo_by_track[(int(g["track_id"]), int(g.get("track_generation_id", 0)))] = g

    # Index clusters by member (job_id, track_id, track_generation_id)
    cluster_by_track: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for cl in clusters or []:
        for m in cl.get("members") or []:
            if str(m.get("job_id")) == job_id:
                cluster_by_track[(int(m["track_id"]), int(m.get("track_generation_id", 0)))] = cl

    # Index representative geometry snapshots by (track_id, track_generation_id)
    snap_by_track: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for sm in (geometry_manifest or {}).get("snapshots") or []:
        key = (int(sm.get("track_id", -1)), int(sm.get("track_generation_id", 0)))
        if key not in snap_by_track:
            snap_by_track[key] = sm

    # Build canonical track list (fallback to obs_by_track keys if tracks JSON missing)
    track_entries: List[Dict[str, Any]] = list(tracks or [])
    seen_keys = {(int(t["track_id"]), int(t.get("track_generation_id", 0))) for t in track_entries}
    for tkey, t_obs in sorted(obs_by_track.items()):
        if tkey not in seen_keys:
            track_entries.append(
                {
                    "source_session_id": job_id,
                    "track_id": tkey[0],
                    "track_generation_id": tkey[1],
                    "first_seen_frame_idx": (t_obs[0].get("frame_context") or {}).get("frame_idx", 0),
                    "last_seen_frame_idx": (t_obs[-1].get("frame_context") or {}).get("frame_idx", 0),
                    "observation_count": len(t_obs),
                    "box_xyxy": (t_obs[0].get("detection") or {}).get("box_xyxy"),
                    "latest_state": (t_obs[-1].get("measurement") or {}).get("measurement_state", "NOT_EVALUATED"),
                    "quality_summary": None,
                }
            )

    candidates: List[Dict[str, Any]] = []
    priority_counts: Counter = Counter()
    review_status_counts: Counter = Counter()
    quality_counts: Counter = Counter()

    for tr in track_entries:
        tid = int(tr["track_id"])
        tgen = int(tr.get("track_generation_id", 0))
        tkey = (tid, tgen)
        t_obs = obs_by_track.get(tkey, [])
        qsum = tr.get("quality_summary") or {}

        confs = [
            _safe_float((o.get("detection") or {}).get("detector_confidence"))
            for o in t_obs
        ]
        confs_clean = [c for c in confs if c is not None]
        conf_med = round(_median(confs_clean), 4) if confs_clean else _safe_float((qsum.get("component_medians") or {}).get("detection_quality"))
        conf_max = round(max(confs_clean), 4) if confs_clean else conf_med

        state_ctr = Counter(
            str((o.get("measurement") or {}).get("measurement_state") or "NOT_EVALUATED")
            for o in t_obs
        )
        obs_count = int(tr.get("observation_count") or len(t_obs))
        fresh_valid = int(qsum.get("fresh_valid_count") if "fresh_valid_count" in qsum else state_ctr.get("FRESH_VALID", 0))
        fresh_invalid = int(qsum.get("fresh_invalid_count") if "fresh_invalid_count" in qsum else state_ctr.get("FRESH_INVALID", 0))
        cached_valid = int(qsum.get("cached_measurement_count") if "cached_measurement_count" in qsum else state_ctr.get("CACHED_VALID", 0))
        expired = int(qsum.get("expired_count") if "expired_count" in qsum else state_ctr.get("EXPIRED", 0))
        not_eval = int(qsum.get("not_evaluated_count") if "not_evaluated_count" in qsum else state_ctr.get("NOT_EVALUATED", 0))

        first_f = int(tr.get("first_seen_frame_idx") or 0)
        last_f = int(tr.get("last_seen_frame_idx") or first_f)
        first_ts = _safe_float((t_obs[0].get("frame_context") or {}).get("timestamp_sec")) if t_obs else None
        last_ts = _safe_float((t_obs[-1].get("frame_context") or {}).get("timestamp_sec")) if t_obs else None

        # Choose best representative frame (prefer geometry snapshot frame, else highest fresh quality/confidence)
        snap = snap_by_track.get(tkey)
        rep_obs = None
        if t_obs:
            rep_obs = max(
                t_obs,
                key=lambda o: (
                    1 if (o.get("measurement") or {}).get("measurement_state") == "FRESH_VALID" else 0,
                    _safe_float(((o.get("measurement") or {}).get("quality_assessment") or {}).get("quality_index")) or -1.0,
                    _safe_float((o.get("detection") or {}).get("detector_confidence")) or 0.0,
                ),
            )
        rep_frame_idx = (
            int(snap["frame_idx"])
            if snap is not None
            else (int((rep_obs.get("frame_context") or {}).get("frame_idx", first_f)) if rep_obs else first_f)
        )
        rep_ts = (
            _safe_float(snap.get("timestamp_sec"))
            if snap is not None
            else (_safe_float((rep_obs.get("frame_context") or {}).get("timestamp_sec")) if rep_obs else first_ts)
        )

        geo = geo_by_track.get(tkey)
        cl = cluster_by_track.get(tkey)
        mrev = (manual_reviews_by_key or {}).get(tkey)

        has_gps = bool(geo and geo.get("lat") is not None and geo.get("lon") is not None)
        qual_cat = qsum.get("track_quality_category") or (
            "NOT_EVALUATED" if not depth_sig["depth_enabled"] else "INSUFFICIENT_EVIDENCE"
        )
        med_rel = _safe_float(qsum.get("median_relative_score"))

        cand: Dict[str, Any] = {
            "job_id": job_id,
            "original_filename": job.get("original_filename"),
            "mode": job.get("mode"),
            "track_id": tid,
            "track_generation_id": tgen,
            "observation_count": obs_count,
            "frame_span": max(1, last_f - first_f + 1),
            "first_seen_frame_idx": first_f,
            "last_seen_frame_idx": last_f,
            "first_seen_timestamp_sec": first_ts,
            "last_seen_timestamp_sec": last_ts,
            "representative_frame_idx": rep_frame_idx,
            "representative_timestamp_sec": rep_ts,
            "representative_snapshot_id": snap.get("snapshot_id") if snap else None,
            "box_xyxy": tr.get("box_xyxy"),
            "latest_state": tr.get("latest_state"),
            "detector_confidence_median": conf_med,
            "detector_confidence_max": conf_max,
            "fresh_valid_count": fresh_valid,
            "cached_valid_count": cached_valid,
            "fresh_invalid_count": fresh_invalid,
            "expired_count": expired,
            "not_evaluated_count": not_eval,
            "fresh_measurement_coverage": round(fresh_valid / obs_count, 4) if obs_count > 0 else None,
            "effective_measurement_coverage": round((fresh_valid + cached_valid) / obs_count, 4) if obs_count > 0 else None,
            "quality_category": qual_cat,
            "median_quality_index": _safe_float(qsum.get("median_quality_index")),
            "median_fresh_quality_index": _safe_float(qsum.get("median_fresh_quality_index")),
            "evidence_sufficiency": qsum.get("evidence_sufficiency") or "INSUFFICIENT",
            "relative_depth_variability": _safe_float(qsum.get("relative_depth_variability")),
            "median_relative_score": med_rel if depth_sig["depth_enabled"] else None,
            "relative_score_units": (
                "uncalibrated_relative_inverse_depth_residual"
                if (depth_sig["depth_enabled"] and med_rel is not None)
                else None
            ),
            "depth_enabled": depth_sig["depth_enabled"],
            "depth_provenance": depth_sig,
            "has_gps": has_gps,
            "lat": geo.get("lat") if has_gps else None,
            "lon": geo.get("lon") if has_gps else None,
            "horizontal_accuracy_m": geo.get("horizontal_accuracy_m") if geo else None,
            "gps_sync_quality": geo.get("sync_quality") if geo else None,
            "interpolation_status": geo.get("interpolation_status") if geo else None,
            "gps_provenance_label": geo.get("provenance_label") if geo else None,
            "cluster_id": cl.get("cluster_id") if cl else None,
            "cluster_distinct_trip_count": int(cl.get("distinct_trip_count", 0)) if cl else 0,
            "cluster_match_status": cl.get("match_status") if cl else None,
            "contains_synthetic_gps": bool(cl.get("contains_synthetic", False)) if cl else False,
            "review_id": mrev.get("review_id") if mrev else None,
            "review_status": mrev.get("review_status") if mrev else "PENDING_REVIEW",
            "reviewer_label": mrev.get("reviewer_label") if mrev else None,
            "review_notes": mrev.get("review_notes") if mrev else None,
            "review_updated_at": mrev.get("updated_at") if mrev else None,
            "dominant_rejection_reasons": qsum.get("dominant_rejection_reasons") or [],
            "quality_summary": qsum if qsum else None,
        }

        prio = compute_review_priority(cand, cfg=cfg)
        cand.update(
            {
                "review_priority_category": prio["review_priority_category"],
                "review_priority_index": prio["review_priority_index"],
                "priority_reasons": prio["priority_reasons"],
                "priority_component_scores": prio["component_scores"],
            }
        )

        priority_counts[cand["review_priority_category"]] += 1
        review_status_counts[cand["review_status"]] += 1
        quality_counts[cand["quality_category"]] += 1
        candidates.append(cand)

    # Sort highest review priority index first (with MANUAL_VERIFICATION_REQUIRED & PRIORITY_REVIEW first)
    _cat_order = {
        "PRIORITY_REVIEW": 0,
        "MANUAL_VERIFICATION_REQUIRED": 1,
        "STANDARD_REVIEW": 2,
        "INSUFFICIENT_EVIDENCE": 3,
        "REVIEWED": 4,
    }
    candidates.sort(
        key=lambda c: (
            _cat_order.get(c["review_priority_category"], 9),
            -(c["review_priority_index"] if c["review_priority_index"] is not None else -1.0),
            -c["observation_count"],
            c["track_id"],
        )
    )

    return {
        "job_id": job_id,
        "original_filename": job.get("original_filename"),
        "mode": job.get("mode"),
        "status": job.get("status"),
        "depth_provenance": depth_sig,
        "candidate_count": len(candidates),
        "total_observations": sum(c["observation_count"] for c in candidates),
        "gps_available": any(c["has_gps"] for c in candidates) or bool(geolocations),
        "geometry_available": bool((geometry_manifest or {}).get("snapshot_count", 0) > 0),
        "priority_distribution": {cat: int(priority_counts.get(cat, 0)) for cat in REVIEW_PRIORITY_CATEGORIES},
        "review_status_distribution": {st: int(review_status_counts.get(st, 0)) for st in REVIEW_STATUSES},
        "quality_distribution": dict(sorted(quality_counts.items())),
        "scoring_version": cfg.scoring_version,
        "scoring_config": asdict(cfg),
        "candidates": candidates,
        "scientific_disclaimer": MAINTENANCE_SCIENTIFIC_DISCLAIMER,
    }


def build_road_segment_analytics(
    *,
    job: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    route_points: Optional[Sequence[Dict[str, Any]]] = None,
    route_metadata: Optional[Dict[str, Any]] = None,
    segment_length_m: float = 50.0,
) -> Dict[str, Any]:
    """
    Group synchronized candidate tracks into configurable route-distance road segments.
    Never divides by zero/unreliable route length, never fabricates coordinates, and returns
    clear unavailable status when GPS telemetry is absent.
    """
    seg_len = max(10.0, min(1000.0, float(segment_length_m)))
    job_id = str(job["job_id"])
    pts = [p for p in (route_points or []) if p.get("lat") is not None and p.get("lon") is not None]

    session_summary = {
        "session_track_count": len(candidates),
        "total_candidate_observations": sum(int(c.get("observation_count") or 0) for c in candidates),
        "review_status_distribution": dict(Counter(c.get("review_status", "PENDING_REVIEW") for c in candidates)),
        "priority_distribution": dict(Counter(c.get("review_priority_category", "INSUFFICIENT_EVIDENCE") for c in candidates)),
        "quality_distribution": dict(Counter(c.get("quality_category", "INSUFFICIENT_EVIDENCE") for c in candidates)),
        "note": "Session-level candidate tracks only (ByteTrack IDs, NOT verified physical pothole counts).",
    }

    if len(pts) == 0:
        return {
            "job_id": job_id,
            "gps_available": False,
            "geographical_analysis_unavailable": True,
            "reason": "No genuine GPS telemetry is associated with this inspection.",
            "segment_length_m": seg_len,
            "observed_route_length_m": None,
            "route_length_reliable": False,
            "geolocated_track_count": 0,
            "segments": [],
            "session_level_summary": session_summary,
            "scientific_disclaimer": MAINTENANCE_SCIENTIFIC_DISCLAIMER,
        }

    # Compute cumulative geodesic distance along sorted GPS trajectory
    pts_sorted = sorted(pts, key=lambda p: float(p.get("timestamp_sec") or 0.0))
    cum_dists = [0.0]
    for i in range(1, len(pts_sorted)):
        d = haversine_distance_m(
            float(pts_sorted[i - 1]["lat"]),
            float(pts_sorted[i - 1]["lon"]),
            float(pts_sorted[i]["lat"]),
            float(pts_sorted[i]["lon"]),
        )
        cum_dists.append(cum_dists[-1] + d)

    total_route_m = round(cum_dists[-1], 2)
    route_reliable = total_route_m >= 1.0

    # Match each GPS-enabled candidate to nearest route point to assign cumulative route chainage
    geolocated_cands = [c for c in candidates if c.get("has_gps") and c.get("lat") is not None and c.get("lon") is not None]
    seg_buckets: Dict[int, Dict[str, Any]] = {}

    if not route_reliable:
        # Stationary or degenerate route: put geolocated candidates into a single unsegmented bucket (index 0)
        seg_buckets[0] = {
            "segment_idx": 0,
            "start_m": 0.0,
            "end_m": total_route_m,
            "observed_length_m": total_route_m,
            "route_points": list(pts_sorted),
            "candidates": list(geolocated_cands),
        }
    else:
        for pt, cd in zip(pts_sorted, cum_dists):
            sidx = int(cd // seg_len)
            b = seg_buckets.setdefault(
                sidx,
                {
                    "segment_idx": sidx,
                    "start_m": round(sidx * seg_len, 2),
                    "end_m": round(min(total_route_m, (sidx + 1) * seg_len), 2),
                    "observed_length_m": 0.0,
                    "route_points": [],
                    "candidates": [],
                },
            )
            b["route_points"].append(pt)

        for cand in geolocated_cands:
            clat, clon = float(cand["lat"]), float(cand["lon"])
            best_idx = 0
            best_d = float("inf")
            for idx, pt in enumerate(pts_sorted):
                d = haversine_distance_m(clat, clon, float(pt["lat"]), float(pt["lon"]))
                if d < best_d:
                    best_d = d
                    best_idx = idx
            sidx = int(cum_dists[best_idx] // seg_len)
            b = seg_buckets.setdefault(
                sidx,
                {
                    "segment_idx": sidx,
                    "start_m": round(sidx * seg_len, 2),
                    "end_m": round(min(total_route_m, (sidx + 1) * seg_len), 2),
                    "observed_length_m": 0.0,
                    "route_points": [],
                    "candidates": [],
                },
            )
            b["candidates"].append(cand)

    segments_out: List[Dict[str, Any]] = []
    for sidx in sorted(seg_buckets.keys()):
        b = seg_buckets[sidx]
        cands_in_seg: List[Dict[str, Any]] = b["candidates"]
        rpts: List[Dict[str, Any]] = b["route_points"]
        span_m = max(0.0, float(b["end_m"]) - float(b["start_m"]))
        coords_lat = [float(p["lat"]) for p in rpts] or [float(c["lat"]) for c in cands_in_seg]
        coords_lon = [float(p["lon"]) for p in rpts] or [float(c["lon"]) for c in cands_in_seg]
        centroid_lat = round(sum(coords_lat) / len(coords_lat), 7) if coords_lat else None
        centroid_lon = round(sum(coords_lon) / len(coords_lon), 7) if coords_lon else None

        obs_total = sum(int(c.get("observation_count") or 0) for c in cands_in_seg)
        fresh_sum = sum(int(c.get("fresh_valid_count") or 0) for c in cands_in_seg)
        eff_sum = sum(int(c.get("fresh_valid_count") or 0) + int(c.get("cached_valid_count") or 0) for c in cands_in_seg)
        accs = [_safe_float(c.get("horizontal_accuracy_m")) for c in cands_in_seg]
        accs_clean = [a for a in accs if a is not None]
        cluster_ids = sorted({str(c["cluster_id"]) for c in cands_in_seg if c.get("cluster_id")})

        segments_out.append(
            {
                "segment_id": f"seg_{sidx:04d}",
                "segment_idx": sidx,
                "start_m": b["start_m"],
                "end_m": b["end_m"],
                "observed_length_m": round(span_m, 2),
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "gps_sample_count": len(rpts),
                "session_candidate_track_count": len(cands_in_seg),
                "candidate_observation_count": obs_total,
                "tracks_per_km_approximate": (
                    round((len(cands_in_seg) / span_m) * 1000.0, 2)
                    if (route_reliable and span_m >= 5.0)
                    else None
                ),
                "measurement_availability_rate": round(eff_sum / obs_total, 4) if obs_total > 0 else None,
                "fresh_measurement_coverage": round(fresh_sum / obs_total, 4) if obs_total > 0 else None,
                "review_status_distribution": dict(Counter(c.get("review_status", "PENDING_REVIEW") for c in cands_in_seg)),
                "priority_distribution": dict(Counter(c.get("review_priority_category", "INSUFFICIENT_EVIDENCE") for c in cands_in_seg)),
                "quality_distribution": dict(Counter(c.get("quality_category", "INSUFFICIENT_EVIDENCE") for c in cands_in_seg)),
                "cluster_ids": cluster_ids,
                "max_horizontal_accuracy_m": round(max(accs_clean), 2) if accs_clean else None,
                "track_ids": [int(c["track_id"]) for c in cands_in_seg],
                "provenance_note": (
                    "Approximate vehicle-GPS corridor segment; counts reflect session ByteTrack candidate "
                    "tracks, NOT verified physical pothole density or Pavement Condition Index."
                ),
            }
        )

    return {
        "job_id": job_id,
        "gps_available": True,
        "geographical_analysis_unavailable": False,
        "segment_length_m": seg_len,
        "observed_route_length_m": total_route_m,
        "route_length_reliable": route_reliable,
        "route_length_warning": (
            None
            if route_reliable
            else "Observed GPS route span is < 1.0 m; segment length normalization is suppressed."
        ),
        "gps_metadata": route_metadata,
        "geolocated_track_count": len(geolocated_cands),
        "unlocated_track_count": max(0, len(candidates) - len(geolocated_cands)),
        "segment_count": len(segments_out),
        "segments": segments_out,
        "session_level_summary": session_summary,
        "scientific_disclaimer": MAINTENANCE_SCIENTIFIC_DISCLAIMER,
    }


def compare_inspections(
    inspection_bundles: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compare 2+ completed inspections with explicit exposure normalization and
    depth-configuration / route compatibility warnings.
    """
    items: List[Dict[str, Any]] = []
    depth_sigs: List[Tuple[Any, ...]] = []
    gps_flags: List[bool] = []
    all_cluster_ids: List[set] = []

    for b in inspection_bundles:
        job = b["job"]
        maint = b["maintenance"]
        summ = job.get("summary") or {}
        cov = b.get("coverage") or {}
        processed_frames = int(job.get("processed_frames") or summ.get("total_frames") or 0)
        src_fps = _safe_float(summ.get("source_fps"))
        duration_sec = (
            round(processed_frames / src_fps, 2)
            if (processed_frames > 0 and src_fps is not None and src_fps > 0)
            else None
        )

        obs_count = int(maint.get("total_observations") or job.get("detection_observation_count") or 0)
        track_count = int(maint.get("candidate_count") or job.get("unique_candidate_track_count") or 0)
        sc = cov.get("counts_by_state") or job.get("measurement_state_counts") or {}
        fresh_valid = int(sc.get("FRESH_VALID", 0))
        cached_valid = int(sc.get("CACHED_VALID", 0))
        eff_rate = round((fresh_valid + cached_valid) / obs_count, 4) if obs_count > 0 else None

        dsig = maint.get("depth_provenance") or _extract_depth_signature(job.get("config_snapshot"), summ)
        sig_tuple = (
            bool(dsig.get("depth_enabled")),
            dsig.get("depth_model_id"),
            dsig.get("depth_input_size"),
            dsig.get("inference_precision"),
            dsig.get("cavity_strategy"),
        )
        depth_sigs.append(sig_tuple)

        has_gps = bool(maint.get("gps_available", False))
        gps_flags.append(has_gps)
        cids = {str(c["cluster_id"]) for c in (maint.get("candidates") or []) if c.get("cluster_id")}
        all_cluster_ids.append(cids)

        rel_vals = [
            float(c["median_relative_score"])
            for c in (maint.get("candidates") or [])
            if c.get("median_relative_score") is not None
        ]

        items.append(
            {
                "job_id": job["job_id"],
                "original_filename": job.get("original_filename"),
                "mode": job.get("mode"),
                "created_at": job.get("created_at"),
                "completed_at": job.get("completed_at"),
                "processed_frames": processed_frames,
                "source_fps": src_fps,
                "video_duration_sec": duration_sec,
                "candidate_observation_count": obs_count,
                "session_candidate_track_count": track_count,
                "tracks_per_100_frames": round((track_count / processed_frames) * 100.0, 2) if processed_frames > 0 else None,
                "observations_per_100_frames": round((obs_count / processed_frames) * 100.0, 2) if processed_frames > 0 else None,
                "tracks_per_minute_video": round((track_count / duration_sec) * 60.0, 2) if (duration_sec and duration_sec > 0) else None,
                "observations_per_minute_video": round((obs_count / duration_sec) * 60.0, 2) if (duration_sec and duration_sec > 0) else None,
                "fresh_valid_rate_over_observations": cov.get("fresh_valid_rate_over_observations"),
                "fresh_valid_rate_over_keyframes": cov.get("fresh_valid_rate_over_keyframes"),
                "effective_measurement_availability": eff_rate,
                "throughput_fps": job.get("throughput_fps") or summ.get("end_to_end_fps"),
                "depth_invocation_rate": summ.get("depth_invocation_rate"),
                "depth_provenance": dsig,
                "quality_distribution": maint.get("quality_distribution") or {},
                "review_status_distribution": maint.get("review_status_distribution") or {},
                "priority_distribution": maint.get("priority_distribution") or {},
                "gps_available": has_gps,
                "median_track_relative_score": round(_median(rel_vals), 5) if rel_vals else None,
            }
        )

    warnings: List[str] = []
    unique_depth_sigs = set(depth_sigs)
    all_depth_on = all(sig[0] for sig in unique_depth_sigs)
    relative_depth_compatible = len(unique_depth_sigs) == 1 and all_depth_on

    if not all_depth_on:
        warnings.append(
            "One or more selected inspections ran with depth inference disabled; relative-depth distributions are unavailable/incomparable."
        )
    elif len(unique_depth_sigs) > 1:
        warnings.append(
            "Incompatible depth configurations detected across selected inspections (model, input resolution, precision, or cavity strategy differ). Raw relative-depth scores MUST NOT be compared directly."
        )

    shared_clusters: set = set()
    if len(all_cluster_ids) >= 2:
        shared_clusters = set.intersection(*all_cluster_ids) if all(all_cluster_ids) else set()

    if not all(gps_flags):
        warnings.append(
            "GPS telemetry is missing on at least one inspection; routes may be different corridors/viewpoints. Do NOT interpret candidate-count differences as road deterioration or improvement."
        )
    elif not shared_clusters:
        warnings.append(
            "No overlapping cross-trip geospatial candidate clusters link these inspections yet; verify repeated-route alignment before inferring physical change."
        )

    if not relative_depth_compatible:
        for it in items:
            it["comparable_median_relative_score"] = None
    else:
        for it in items:
            it["comparable_median_relative_score"] = it["median_track_relative_score"]

    return {
        "compared_count": len(items),
        "relative_depth_compatible": relative_depth_compatible,
        "shared_geospatial_cluster_count": len(shared_clusters),
        "shared_geospatial_cluster_ids": sorted(shared_clusters),
        "compatibility_warnings": warnings,
        "inspections": items,
        "scientific_disclaimer": MAINTENANCE_SCIENTIFIC_DISCLAIMER,
    }

"""
SQLite persistence & analytics orchestration bridge for Phase 12 Road Maintenance Intelligence.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from roadvision_v2.config import MaintenancePriorityConfig
from roadvision_v2.maintenance import (
    MAINTENANCE_SCIENTIFIC_DISCLAIMER,
    REVIEW_PRIORITY_CATEGORIES,
    REVIEW_STATUSES,
    aggregate_job_maintenance,
    build_inspection_report_payload,
    build_road_segment_analytics,
    compare_inspections,
)

from .artifacts import resolve_artifact_entry
from .configuration import APIConfig
from .errors import APIError, BadRequestError, InvalidJobStateError, NotFoundError
from .geospatial_store import GeospatialStore
from .job_manager import JOB_STATUS_COMPLETED, JobManager, validate_uuid
from .schemas import PIPELINE_VERSION, compute_measurement_coverage, sanitize_json_value
from .storage import JobDatabase


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_optional_text(val: Any, max_len: int, field_name: str) -> Optional[str]:
    if val is None:
        return None
    if not isinstance(val, str):
        raise BadRequestError(f"{field_name} must be a string", error_code="INVALID_PARAMETER")
    cleaned = val.strip()
    if not cleaned:
        return None
    if len(cleaned) > max_len:
        raise BadRequestError(
            f"{field_name} exceeds maximum length of {max_len} characters",
            error_code="INVALID_PARAMETER",
        )
    return cleaned


class MaintenanceStore:
    """Manages persistent human-in-the-loop reviews, review queues, road segments, and reports."""

    def __init__(
        self,
        db: JobDatabase,
        job_manager: JobManager,
        geospatial_store: GeospatialStore,
        api_cfg: APIConfig,
        priority_cfg: Optional[MaintenancePriorityConfig] = None,
    ) -> None:
        self.db = db
        self.job_manager = job_manager
        self.geo_store = geospatial_store
        self.api_cfg = api_cfg
        self.priority_cfg = priority_cfg or MaintenancePriorityConfig()

    def _row_to_review_dict(self, row: Any, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        d = dict(row)
        evidence = json.loads(d.pop("automated_evidence_json") or "{}")
        reasons = json.loads(d.pop("priority_reasons_json") or "[]")
        out = {
            "review_id": d["review_id"],
            "job_id": d["job_id"],
            "track_id": d["track_id"],
            "track_generation_id": d["track_generation_id"],
            "cluster_id": d["cluster_id"],
            "review_status": d["review_status"],
            "reviewer_label": d["reviewer_label"],
            "review_notes": d["review_notes"],
            "priority_category": d["priority_category"],
            "priority_index": d["priority_index"],
            "priority_reasons": reasons,
            "scoring_version": d["scoring_version"],
            "automated_evidence": evidence,
            "created_at": d["created_at"],
            "updated_at": d["updated_at"],
            "prototype_auth_note": "Local prototype record without authenticated reviewer identity.",
        }
        if history is not None:
            out["history"] = history
        return sanitize_json_value(out)

    def _get_reviews_for_job(self, job_id: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
        jid = validate_uuid(job_id)
        with self.db.read_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM manual_reviews WHERE job_id = ? ORDER BY updated_at DESC",
                (jid,),
            ).fetchall()
        out: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for r in rows:
            item = self._row_to_review_dict(r)
            out[(int(item["track_id"]), int(item["track_generation_id"]))] = item
        return out

    def _load_json_artifact_safe(self, job: Dict[str, Any], artifact_id: str, default: Any) -> Any:
        avail = job.get("artifact_availability") or {}
        if not avail.get(artifact_id):
            return default
        try:
            _, path = resolve_artifact_entry(job, artifact_id, self.api_cfg.resolved_artifact_dir)
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return default

    def _load_geometry_snapshot_crops(
        self,
        job: Dict[str, Any],
        manifest: Optional[Dict[str, Any]],
        max_items: int = 4,
    ) -> List[Dict[str, Any]]:
        if not manifest or not manifest.get("snapshots"):
            return []
        real_root = os.path.realpath(self.api_cfg.resolved_artifact_dir)
        real_job_dir = os.path.realpath(job.get("output_dir") or "")
        if not real_job_dir or os.path.commonpath([real_root, real_job_dir]) != real_root:
            return []
        crops: List[Dict[str, Any]] = []
        for sm in (manifest.get("snapshots") or [])[:max_items]:
            rel = sm.get("relative_path")
            if not rel:
                continue
            raw_path = os.path.join(real_job_dir, rel)
            if os.path.islink(raw_path):
                continue
            real_path = os.path.realpath(raw_path)
            if os.path.commonpath([real_job_dir, real_path]) != real_job_dir or not os.path.isfile(real_path):
                continue
            try:
                with open(real_path, "r", encoding="utf-8") as fh:
                    snap_data = json.load(fh)
                if snap_data.get("source_crop_data_url"):
                    crops.append(
                        {
                            "snapshot_id": snap_data.get("snapshot_id"),
                            "track_id": snap_data.get("track_id"),
                            "frame_idx": snap_data.get("frame_idx"),
                            "timestamp_sec": snap_data.get("timestamp_sec"),
                            "source_crop_data_url": snap_data["source_crop_data_url"],
                        }
                    )
            except Exception:
                continue
        return crops

    def get_job_maintenance(self, job_id: str) -> Dict[str, Any]:
        job = self.job_manager.get_job(job_id, include_internal_paths=True)
        if job["status"] != JOB_STATUS_COMPLETED:
            raise InvalidJobStateError(
                f"Maintenance analytics are only available after job completion; current status is '{job['status']}'",
                error_code="RESULTS_UNAVAILABLE",
                details={"status": job["status"]},
            )

        raw_tracks = self._load_json_artifact_safe(job, "tracks_json", {})
        tracks = list(raw_tracks.values()) if isinstance(raw_tracks, dict) else (raw_tracks if isinstance(raw_tracks, list) else [])
        observations = self._load_json_artifact_safe(job, "frames_json", [])
        if not isinstance(observations, list):
            observations = []
        geom_manifest = self._load_json_artifact_safe(job, "geometry_manifest_json", None)

        geo_loc_payload = self.geo_store.get_job_locations(job, include_geojson=False)
        geolocations = geo_loc_payload.get("locations") or []
        clusters_resp = self.geo_store.list_clusters(page=1, page_size=200)
        cluster_details: List[Dict[str, Any]] = []
        for cl in clusters_resp.get("clusters") or []:
            try:
                cluster_details.append(self.geo_store.get_cluster(cl["cluster_id"]))
            except Exception:
                continue

        manual_reviews = self._get_reviews_for_job(job["job_id"])

        return aggregate_job_maintenance(
            job=job,
            tracks=tracks,
            observations=observations,
            geolocations=geolocations,
            clusters=cluster_details,
            geometry_manifest=geom_manifest,
            manual_reviews_by_key=manual_reviews,
            cfg=self.priority_cfg,
        )

    def get_job_segments(self, job_id: str, segment_length_m: float = 50.0) -> Dict[str, Any]:
        maint = self.get_job_maintenance(job_id)
        job = self.job_manager.get_job(job_id, include_internal_paths=False)
        route_payload = self.geo_store.get_job_route(job["job_id"])
        return build_road_segment_analytics(
            job=job,
            candidates=maint.get("candidates") or [],
            route_points=route_payload.get("route_points") or [],
            route_metadata=route_payload.get("metadata"),
            segment_length_m=segment_length_m,
        )

    def get_overview(
        self,
        *,
        include_provenances: Optional[Sequence[str]] = None,
        exclude_non_genuine: bool = True,
    ) -> Dict[str, Any]:
        provenance_counts = self.job_manager.count_jobs_by_provenance(status=JOB_STATUS_COMPLETED)
        if include_provenances:
            eff_provs: Optional[List[str]] = list(include_provenances)
            scope_label = ",".join(eff_provs)
        elif exclude_non_genuine:
            eff_provs = ["USER_INSPECTION"]
            scope_label = "USER_INSPECTION_ONLY"
        else:
            eff_provs = None
            scope_label = "ALL"

        res = self.job_manager.list_jobs(
            status=JOB_STATUS_COMPLETED,
            include_provenances=eff_provs,
            limit=100,
            offset=0,
        )
        completed_jobs = res.get("jobs") or []

        total_tracks = 0
        total_obs = 0
        jobs_with_gps = 0
        priority_totals: Counter = Counter()
        review_status_totals: Counter = Counter()
        quality_totals: Counter = Counter()
        session_rows: List[Dict[str, Any]] = []

        for j in completed_jobs:
            jid = j["job_id"]
            try:
                maint = self.get_job_maintenance(jid)
            except Exception:
                continue
            cand_cnt = int(maint.get("candidate_count") or 0)
            obs_cnt = int(maint.get("total_observations") or 0)
            total_tracks += cand_cnt
            total_obs += obs_cnt
            if maint.get("gps_available"):
                jobs_with_gps += 1
            for k, v in (maint.get("priority_distribution") or {}).items():
                priority_totals[k] += int(v)
            for k, v in (maint.get("review_status_distribution") or {}).items():
                review_status_totals[k] += int(v)
            for k, v in (maint.get("quality_distribution") or {}).items():
                quality_totals[k] += int(v)

            session_rows.append(
                {
                    "job_id": jid,
                    "original_filename": j.get("original_filename"),
                    "mode": j.get("mode"),
                    "provenance": j.get("provenance") or "USER_INSPECTION",
                    "completed_at": j.get("completed_at"),
                    "candidate_count": cand_cnt,
                    "total_observations": obs_cnt,
                    "gps_available": bool(maint.get("gps_available")),
                    "geometry_available": bool(maint.get("geometry_available")),
                    "priority_distribution": maint.get("priority_distribution"),
                    "review_status_distribution": maint.get("review_status_distribution"),
                }
            )

        total_completed_all = sum(provenance_counts.values())
        return {
            "completed_inspection_count": len(completed_jobs),
            "provenance_scope": scope_label,
            "completed_counts_by_provenance": provenance_counts,
            "total_completed_all_provenances": total_completed_all,
            "excluded_completed_count": max(0, total_completed_all - len(completed_jobs)),
            "exploratory_completed_count": int(provenance_counts.get("RESEARCH_EXPLORATORY", 0)),
            "jobs_with_gps_count": jobs_with_gps,
            "total_session_candidate_tracks": total_tracks,
            "total_candidate_observations": total_obs,
            "priority_distribution": {cat: int(priority_totals.get(cat, 0)) for cat in REVIEW_PRIORITY_CATEGORIES},
            "review_status_distribution": {st: int(review_status_totals.get(st, 0)) for st in REVIEW_STATUSES},
            "quality_distribution": dict(sorted(quality_totals.items())),
            "inspections": session_rows,
            "prototype_note": "Local prototype without authenticated user accounts.",
            "scientific_disclaimer": MAINTENANCE_SCIENTIFIC_DISCLAIMER,
        }

    def get_review_queue(
        self,
        *,
        job_id: Optional[str] = None,
        priority_category: Optional[str] = None,
        review_status: Optional[str] = None,
        quality_category: Optional[str] = None,
        include_provenances: Optional[Sequence[str]] = None,
        exclude_non_genuine: bool = True,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        if priority_category and priority_category not in REVIEW_PRIORITY_CATEGORIES:
            raise BadRequestError(
                f"Invalid priority_category '{priority_category}'",
                error_code="INVALID_FILTER",
                details={"allowed_categories": list(REVIEW_PRIORITY_CATEGORIES)},
            )
        if review_status and review_status not in REVIEW_STATUSES:
            raise BadRequestError(
                f"Invalid review_status '{review_status}'",
                error_code="INVALID_FILTER",
                details={"allowed_statuses": list(REVIEW_STATUSES)},
            )

        if job_id:
            target_jobs = [self.job_manager.get_job(job_id, include_internal_paths=False)]
            if target_jobs[0]["status"] != JOB_STATUS_COMPLETED:
                raise InvalidJobStateError(
                    f"Review queue unavailable while job status is '{target_jobs[0]['status']}'",
                    error_code="RESULTS_UNAVAILABLE",
                )
        else:
            eff_provs = list(include_provenances) if include_provenances else (["USER_INSPECTION"] if exclude_non_genuine else None)
            target_jobs = (
                self.job_manager.list_jobs(
                    status=JOB_STATUS_COMPLETED,
                    include_provenances=eff_provs,
                    limit=50,
                    offset=0,
                )
            ).get("jobs") or []

        all_candidates: List[Dict[str, Any]] = []
        for j in target_jobs:
            try:
                maint = self.get_job_maintenance(j["job_id"])
                for cand in maint.get("candidates") or []:
                    cand["job_provenance"] = j.get("provenance") or "USER_INSPECTION"
                    all_candidates.append(cand)
            except Exception:
                continue

        filtered: List[Dict[str, Any]] = []
        for c in all_candidates:
            if priority_category and c.get("review_priority_category") != priority_category:
                continue
            if review_status and c.get("review_status") != review_status:
                continue
            if quality_category and c.get("quality_category") != quality_category:
                continue
            filtered.append(c)

        _cat_order = {
            "PRIORITY_REVIEW": 0,
            "MANUAL_VERIFICATION_REQUIRED": 1,
            "STANDARD_REVIEW": 2,
            "INSUFFICIENT_EVIDENCE": 3,
            "REVIEWED": 4,
        }
        filtered.sort(
            key=lambda c: (
                _cat_order.get(c.get("review_priority_category", ""), 9),
                -(c["review_priority_index"] if c.get("review_priority_index") is not None else -1.0),
                -int(c.get("observation_count") or 0),
                str(c.get("job_id")),
                int(c.get("track_id") or 0),
            )
        )

        total = len(filtered)
        start = (page - 1) * page_size
        page_items = filtered[start : start + page_size]

        return {
            "items": page_items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if total > 0 else 0,
            "scoring_version": self.priority_cfg.scoring_version,
            "scientific_disclaimer": MAINTENANCE_SCIENTIFIC_DISCLAIMER,
        }

    def list_reviews(
        self,
        *,
        job_id: Optional[str] = None,
        review_status: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        clauses: List[str] = []
        params: List[Any] = []
        if job_id:
            clauses.append("job_id = ?")
            params.append(validate_uuid(job_id))
        if review_status:
            if review_status not in REVIEW_STATUSES:
                raise BadRequestError(
                    f"Invalid review_status '{review_status}'",
                    error_code="INVALID_FILTER",
                    details={"allowed_statuses": list(REVIEW_STATUSES)},
                )
            clauses.append("review_status = ?")
            params.append(review_status)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        offset = (page - 1) * page_size
        with self.db.read_connection() as conn:
            total = int(
                conn.execute(f"SELECT COUNT(*) FROM manual_reviews {where}", tuple(params)).fetchone()[0]
            )
            rows = conn.execute(
                f"SELECT * FROM manual_reviews {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                tuple([*params, page_size, offset]),
            ).fetchall()

        return {
            "reviews": [self._row_to_review_dict(r) for r in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if total > 0 else 0,
        }

    def get_review(self, review_id: str) -> Dict[str, Any]:
        rid = validate_uuid(review_id)
        with self.db.read_connection() as conn:
            row = conn.execute("SELECT * FROM manual_reviews WHERE review_id = ?", (rid,)).fetchone()
            if row is None:
                raise NotFoundError(f"Review '{rid}' not found", error_code="REVIEW_NOT_FOUND")
            hist_rows = conn.execute(
                "SELECT * FROM manual_review_history WHERE review_id = ? ORDER BY history_id ASC",
                (rid,),
            ).fetchall()
        history = [dict(h) for h in hist_rows]
        return self._row_to_review_dict(row, history=history)

    def create_review(
        self,
        *,
        job_id: str,
        track_id: int,
        track_generation_id: int = 0,
        review_status: str = "PENDING_REVIEW",
        reviewer_label: Optional[str] = None,
        review_notes: Optional[str] = None,
        cluster_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        jid = validate_uuid(job_id)
        try:
            tid = int(track_id)
            tgen = int(track_generation_id)
        except (TypeError, ValueError) as exc:
            raise BadRequestError("track_id and track_generation_id must be integers", error_code="INVALID_PARAMETER") from exc

        status_clean = str(review_status or "PENDING_REVIEW").strip().upper()
        if status_clean not in REVIEW_STATUSES:
            raise BadRequestError(
                f"Invalid review_status '{status_clean}'",
                error_code="INVALID_REVIEW_STATUS",
                details={"allowed_statuses": list(REVIEW_STATUSES)},
            )
        label_clean = _clean_optional_text(reviewer_label, 80, "reviewer_label")
        notes_clean = _clean_optional_text(review_notes, 1000, "review_notes")
        cid_clean = validate_uuid(cluster_id) if cluster_id else None

        maint = self.get_job_maintenance(jid)
        matched = next(
            (
                c
                for c in (maint.get("candidates") or [])
                if int(c["track_id"]) == tid and int(c.get("track_generation_id", 0)) == tgen
            ),
            None,
        )
        if matched is None:
            raise NotFoundError(
                f"Candidate track ({tid}, gen={tgen}) not found in completed inspection '{jid}'",
                error_code="TRACK_NOT_FOUND",
            )

        # Freeze original automated evidence at review creation time (never overwritten by updates)
        frozen_evidence = {
            "observation_count": matched.get("observation_count"),
            "first_seen_frame_idx": matched.get("first_seen_frame_idx"),
            "last_seen_frame_idx": matched.get("last_seen_frame_idx"),
            "representative_frame_idx": matched.get("representative_frame_idx"),
            "representative_timestamp_sec": matched.get("representative_timestamp_sec"),
            "representative_snapshot_id": matched.get("representative_snapshot_id"),
            "detector_confidence_median": matched.get("detector_confidence_median"),
            "quality_category": matched.get("quality_category"),
            "median_fresh_quality_index": matched.get("median_fresh_quality_index"),
            "median_relative_score": matched.get("median_relative_score"),
            "depth_provenance": matched.get("depth_provenance"),
            "quality_summary": matched.get("quality_summary"),
            "has_gps": matched.get("has_gps"),
            "lat": matched.get("lat"),
            "lon": matched.get("lon"),
            "gps_sync_quality": matched.get("gps_sync_quality"),
        }

        rid = str(uuid.uuid4())
        now = _utc_now()
        try:
            with self.db.transaction(immediate=True) as conn:
                conn.execute(
                    """
                    INSERT INTO manual_reviews (
                        review_id, job_id, track_id, track_generation_id, cluster_id,
                        review_status, reviewer_label, review_notes,
                        automated_evidence_json, priority_category, priority_index,
                        priority_reasons_json, scoring_version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rid,
                        jid,
                        tid,
                        tgen,
                        cid_clean or matched.get("cluster_id"),
                        status_clean,
                        label_clean,
                        notes_clean,
                        json.dumps(sanitize_json_value(frozen_evidence), allow_nan=False),
                        matched["review_priority_category"],
                        matched["review_priority_index"],
                        json.dumps(matched["priority_reasons"], allow_nan=False),
                        maint["scoring_version"],
                        now,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO manual_review_history (
                        review_id, previous_status, new_status, reviewer_label, review_notes, changed_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (rid, None, status_clean, label_clean, notes_clean, now),
                )
        except sqlite3.IntegrityError as exc:
            with self.db.read_connection() as conn:
                existing = conn.execute(
                    "SELECT review_id FROM manual_reviews WHERE job_id = ? AND track_id = ? AND track_generation_id = ?",
                    (jid, tid, tgen),
                ).fetchone()
            raise APIError(
                f"A manual review record already exists for track #{tid} (gen {tgen}) in inspection {jid}.",
                status_code=409,
                error_code="DUPLICATE_REVIEW",
                details={"existing_review_id": existing["review_id"] if existing else None},
            ) from exc

        return self.get_review(rid)

    def update_review(
        self,
        review_id: str,
        *,
        review_status: Optional[str] = None,
        reviewer_label: Optional[str] = None,
        review_notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        rid = validate_uuid(review_id)

        with self.db.transaction(immediate=True) as conn:
            row = conn.execute("SELECT * FROM manual_reviews WHERE review_id = ?", (rid,)).fetchone()
            if row is None:
                raise NotFoundError(f"Review '{rid}' not found", error_code="REVIEW_NOT_FOUND")
            current = dict(row)

            next_status = current["review_status"]
            if review_status is not None:
                cand_st = str(review_status).strip().upper()
                if cand_st not in REVIEW_STATUSES:
                    raise BadRequestError(
                        f"Invalid review_status '{cand_st}'",
                        error_code="INVALID_REVIEW_STATUS",
                        details={"allowed_statuses": list(REVIEW_STATUSES)},
                    )
                next_status = cand_st

            next_label = (
                _clean_optional_text(reviewer_label, 80, "reviewer_label")
                if reviewer_label is not None
                else current["reviewer_label"]
            )
            next_notes = (
                _clean_optional_text(review_notes, 1000, "review_notes")
                if review_notes is not None
                else current["review_notes"]
            )
            now = _utc_now()

            # Update ONLY human-review fields; original automated_evidence_json remains immutable
            conn.execute(
                """
                UPDATE manual_reviews
                SET review_status = ?, reviewer_label = ?, review_notes = ?, updated_at = ?
                WHERE review_id = ?
                """,
                (next_status, next_label, next_notes, now, rid),
            )
            conn.execute(
                """
                INSERT INTO manual_review_history (
                    review_id, previous_status, new_status, reviewer_label, review_notes, changed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (rid, current["review_status"], next_status, next_label, next_notes, now),
            )

        return self.get_review(rid)

    def compare_jobs(self, job_ids: Sequence[str]) -> Dict[str, Any]:
        if len(job_ids) < 2:
            raise BadRequestError(
                "At least 2 completed inspection job IDs are required for historical comparison",
                error_code="INVALID_PARAMETER",
            )
        if len(job_ids) > 8:
            raise BadRequestError(
                "At most 8 inspection job IDs can be compared at once",
                error_code="INVALID_PARAMETER",
            )

        bundles: List[Dict[str, Any]] = []
        seen: set = set()
        for raw_jid in job_ids:
            jid = validate_uuid(raw_jid)
            if jid in seen:
                continue
            seen.add(jid)
            job = self.job_manager.get_job(jid, include_internal_paths=False)
            if job["status"] != JOB_STATUS_COMPLETED:
                raise InvalidJobStateError(
                    f"Job '{jid}' has status '{job['status']}'; only COMPLETED inspections can be compared",
                    error_code="RESULTS_UNAVAILABLE",
                )
            maint = self.get_job_maintenance(jid)
            summ = job.get("summary") or {}
            sc = job.get("measurement_state_counts") or summ.get("measurement_state_counts") or {}
            total_obs = int(job.get("detection_observation_count") or summ.get("total_hazard_observations") or 0)
            cov = compute_measurement_coverage(sc, total_obs)
            bundles.append({"job": job, "maintenance": maint, "coverage": cov})

        if len(bundles) < 2:
            raise BadRequestError(
                "Provide at least 2 distinct completed inspection job IDs",
                error_code="INVALID_PARAMETER",
            )
        return compare_inspections(bundles)

    def build_job_report(
        self,
        job_id: str,
        *,
        segment_length_m: float = 50.0,
        include_images: bool = True,
    ) -> Dict[str, Any]:
        job = self.job_manager.get_job(job_id, include_internal_paths=True)
        if job["status"] != JOB_STATUS_COMPLETED:
            raise InvalidJobStateError(
                f"Reports are only available after job completion; current status is '{job['status']}'",
                error_code="RESULTS_UNAVAILABLE",
                details={"status": job["status"]},
            )
        maint = self.get_job_maintenance(job["job_id"])
        segments = self.get_job_segments(job["job_id"], segment_length_m=segment_length_m)
        reviews = self.list_reviews(job_id=job["job_id"], page=1, page_size=200).get("reviews") or []

        summ = job.get("summary") or {}
        sc = job.get("measurement_state_counts") or summ.get("measurement_state_counts") or {}
        total_obs = int(job.get("detection_observation_count") or summ.get("total_hazard_observations") or 0)
        coverage = compute_measurement_coverage(sc, total_obs)

        geom_manifest = self._load_json_artifact_safe(job, "geometry_manifest_json", None)
        rep_crops = self._load_geometry_snapshot_crops(job, geom_manifest, max_items=4) if include_images else []

        obs_list = self._load_json_artifact_safe(job, "frames_json", [])
        obs_sample = obs_list[:50] if isinstance(obs_list, list) else []

        # Strip internal paths before embedding job metadata in report
        clean_job = {
            k: v
            for k, v in job.items()
            if k not in ("validated_input_path", "output_dir")
        }
        return build_inspection_report_payload(
            job=clean_job,
            coverage=coverage,
            maintenance=maint,
            segments=segments,
            reviews=reviews,
            geometry_manifest=geom_manifest,
            representative_crops=rep_crops,
            observations_sample=obs_sample,
            pipeline_version=clean_job.get("pipeline_version") or PIPELINE_VERSION,
        )

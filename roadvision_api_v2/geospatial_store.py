"""
SQLite persistence & synchronization bridge for Phase 11 Geospatial Intelligence.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from roadvision_v2.config import GeospatialConfig
from roadvision_v2.geospatial import (
    REVIEW_STATUSES,
    cluster_geospatial_candidates,
    export_locations_geojson,
    synchronize_timestamp_to_gps,
)

from .artifacts import resolve_artifact_entry
from .configuration import APIConfig
from .errors import BadRequestError, NotFoundError
from .job_manager import JOB_STATUS_COMPLETED, JobManager, validate_uuid
from .schemas import sanitize_json_value
from .storage import JobDatabase


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GeospatialStore:
    """Manages external GPS telemetry, track location synchronization, and candidate clusters."""

    def __init__(self, db: JobDatabase, api_cfg: APIConfig, default_geo_cfg: Optional[GeospatialConfig] = None) -> None:
        self.db = db
        self.api_cfg = api_cfg
        self.default_geo_cfg = default_geo_cfg or GeospatialConfig()

    def save_job_gps_telemetry(
        self,
        job_id: str,
        parsed_track: Dict[str, Any],
        original_filename: str,
        max_sync_gap_sec: float = 2.0,
    ) -> Dict[str, Any]:
        jid = validate_uuid(job_id)
        now = _utc_now()
        samples = parsed_track.get("samples") or []
        with self.db.transaction(immediate=True) as conn:
            conn.execute("DELETE FROM inspection_gps_samples WHERE job_id = ?", (jid,))
            conn.execute("DELETE FROM inspection_gps_tracks WHERE job_id = ?", (jid,))
            conn.execute("DELETE FROM track_geolocations WHERE job_id = ?", (jid,))

            conn.execute(
                """
                INSERT INTO inspection_gps_tracks (
                    job_id, source_format, original_filename, time_mode,
                    sync_offset_sec, max_sync_gap_sec, sample_count,
                    duplicate_timestamps_merged, out_of_order_count,
                    min_ts_sec, max_ts_sec, uploaded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    jid,
                    parsed_track["source_format"],
                    original_filename,
                    parsed_track["time_mode"],
                    float(parsed_track.get("sync_offset_sec", 0.0)),
                    float(max_sync_gap_sec),
                    int(parsed_track.get("sample_count", len(samples))),
                    int(parsed_track.get("duplicate_timestamps_merged", 0)),
                    int(parsed_track.get("out_of_order_count", 0)),
                    parsed_track.get("min_ts_sec"),
                    parsed_track.get("max_ts_sec"),
                    now,
                ),
            )
            if samples:
                conn.executemany(
                    """
                    INSERT INTO inspection_gps_samples (
                        job_id, sample_idx, timestamp_sec, raw_timestamp,
                        lat, lon, altitude, speed_mps, horizontal_accuracy_m, heading_deg
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            jid,
                            int(s.get("sample_idx", idx)),
                            float(s["timestamp_sec"]),
                            str(s["raw_timestamp"]),
                            float(s["lat"]),
                            float(s["lon"]),
                            s.get("altitude"),
                            s.get("speed_mps"),
                            s.get("horizontal_accuracy_m"),
                            s.get("heading_deg"),
                        )
                        for idx, s in enumerate(samples)
                    ],
                )
        return self.get_job_route(jid)

    def has_gps_for_job(self, job_id: str) -> bool:
        jid = validate_uuid(job_id)
        with self.db.read_connection() as conn:
            row = conn.execute(
                "SELECT sample_count FROM inspection_gps_tracks WHERE job_id = ?", (jid,)
            ).fetchone()
            return bool(row and int(row["sample_count"]) > 0)

    def sync_job_tracks_with_gps(self, job: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Synchronize tracks from a completed job with persisted GPS samples using source-video
        timestamps and write track_geolocations deterministically.
        """
        jid = validate_uuid(job["job_id"])
        if job.get("status") != JOB_STATUS_COMPLETED:
            return []

        with self.db.read_connection() as conn:
            meta_row = conn.execute(
                "SELECT * FROM inspection_gps_tracks WHERE job_id = ?", (jid,)
            ).fetchone()
            if not meta_row:
                return []
            sample_rows = conn.execute(
                "SELECT * FROM inspection_gps_samples WHERE job_id = ? ORDER BY timestamp_sec ASC",
                (jid,),
            ).fetchall()

        meta = dict(meta_row)
        samples = [dict(r) for r in sample_rows]
        if not samples:
            return []

        cfg = GeospatialConfig(
            max_sync_gap_sec=float(meta.get("max_sync_gap_sec") or self.default_geo_cfg.max_sync_gap_sec),
            exact_match_threshold_sec=self.default_geo_cfg.exact_match_threshold_sec,
            allow_extrapolation=self.default_geo_cfg.allow_extrapolation,
        )

        # Load tracks.json and frames.json from verified artifact manifest
        try:
            _, tracks_path = resolve_artifact_entry(job, "tracks_json", self.api_cfg.resolved_artifact_dir)
            with open(tracks_path, "r", encoding="utf-8") as fh:
                raw_tracks = json.load(fh)
        except Exception:
            raw_tracks = {}

        obs_by_track: Dict[int, List[Dict[str, Any]]] = {}
        try:
            _, frames_path = resolve_artifact_entry(job, "frames_json", self.api_cfg.resolved_artifact_dir)
            with open(frames_path, "r", encoding="utf-8") as fh:
                all_obs = json.load(fh)
            if isinstance(all_obs, list):
                for obs in all_obs:
                    det = obs.get("detection") or {}
                    tid = det.get("track_id")
                    if tid is not None:
                        obs_by_track.setdefault(int(tid), []).append(obs)
        except Exception:
            pass

        track_list = list(raw_tracks.values()) if isinstance(raw_tracks, dict) else []
        synced_rows: List[Dict[str, Any]] = []
        source_label = f"{meta['source_format']}:{meta['original_filename']}"

        for trk in track_list:
            tid = int(trk["track_id"])
            gen_id = int(trk.get("track_generation_id", 0))
            t_obs = obs_by_track.get(tid, [])

            # Pick representative video observation timestamp: prefer first FRESH_VALID, else midpoint/first obs
            rep_obs = next(
                (o for o in t_obs if (o.get("measurement") or {}).get("measurement_state") == "FRESH_VALID"),
                None,
            )
            if rep_obs is None and t_obs:
                rep_obs = t_obs[len(t_obs) // 2]

            if rep_obs is not None:
                fc = rep_obs.get("frame_context") or {}
                meas = rep_obs.get("measurement") or {}
                obs_ts = float(fc.get("timestamp_sec") or 0.0)
                depth_model_id = meas.get("model_identity")
                depth_res = meas.get("depth_input_resolution")
            else:
                # Fallback from frame index & source FPS if frames_json disabled
                src_fps = float((job.get("summary") or {}).get("source_fps") or 25.0)
                mid_f = 0.5 * (float(trk.get("first_seen_frame_idx", 0)) + float(trk.get("last_seen_frame_idx", 0)))
                obs_ts = mid_f / max(src_fps, 1.0)
                depth_model_id = (job.get("summary") or {}).get("depth_model_id")
                depth_res = (job.get("summary") or {}).get("depth_input_size")

            sync_res = synchronize_timestamp_to_gps(
                obs_ts,
                samples,
                cfg=cfg,
                sync_offset_sec=float(meta.get("sync_offset_sec") or 0.0),
                gps_source=source_label,
            )

            qs = trk.get("quality_summary") if isinstance(trk.get("quality_summary"), dict) else {}
            q_cat = qs.get("track_quality_category")
            q_idx = qs.get("median_fresh_quality_index")
            if q_idx is None:
                q_idx = qs.get("median_quality_index")
            med_score = qs.get("median_relative_score")

            synced_rows.append(
                {
                    "job_id": jid,
                    "track_id": tid,
                    "track_generation_id": gen_id,
                    "observation_timestamp_sec": round(obs_ts, 4),
                    "lat": sync_res["lat"],
                    "lon": sync_res["lon"],
                    "horizontal_accuracy_m": sync_res["horizontal_accuracy_m"],
                    "heading_deg": sync_res["heading_deg"],
                    "original_gps_timestamp": sync_res["original_gps_timestamp"],
                    "sync_offset_sec": sync_res["sync_offset_sec"],
                    "interpolation_status": sync_res["interpolation_status"],
                    "time_diff_sec": sync_res["time_diff_sec"],
                    "sync_quality": sync_res["sync_quality"],
                    "gps_source": sync_res["gps_source"],
                    "provenance_label": sync_res["provenance_label"],
                    "quality_category": q_cat,
                    "quality_index": q_idx,
                    "median_relative_score": med_score,
                    "depth_model_id": depth_model_id,
                    "depth_input_resolution": depth_res,
                    "measurement_state": trk.get("latest_state"),
                }
            )

        with self.db.transaction(immediate=True) as conn:
            conn.execute("DELETE FROM track_geolocations WHERE job_id = ?", (jid,))
            if synced_rows:
                conn.executemany(
                    """
                    INSERT INTO track_geolocations (
                        job_id, track_id, track_generation_id, observation_timestamp_sec,
                        lat, lon, horizontal_accuracy_m, heading_deg,
                        original_gps_timestamp, sync_offset_sec, interpolation_status,
                        time_diff_sec, sync_quality, gps_source, provenance_label,
                        quality_category, quality_index, median_relative_score,
                        depth_model_id, depth_input_resolution, measurement_state
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            r["job_id"],
                            r["track_id"],
                            r["track_generation_id"],
                            r["observation_timestamp_sec"],
                            r["lat"],
                            r["lon"],
                            r["horizontal_accuracy_m"],
                            r["heading_deg"],
                            r["original_gps_timestamp"],
                            r["sync_offset_sec"],
                            r["interpolation_status"],
                            r["time_diff_sec"],
                            r["sync_quality"],
                            r["gps_source"],
                            r["provenance_label"],
                            r["quality_category"],
                            r["quality_index"],
                            r["median_relative_score"],
                            r["depth_model_id"],
                            r["depth_input_resolution"],
                            r["measurement_state"],
                        )
                        for r in synced_rows
                    ],
                )
        return synced_rows

    def get_job_route(self, job_id: str) -> Dict[str, Any]:
        jid = validate_uuid(job_id)
        with self.db.read_connection() as conn:
            meta_row = conn.execute(
                "SELECT * FROM inspection_gps_tracks WHERE job_id = ?", (jid,)
            ).fetchone()
            if not meta_row:
                return sanitize_json_value(
                    {
                        "job_id": jid,
                        "gps_available": False,
                        "metadata": None,
                        "route_points": [],
                        "note": "No external GPS telemetry has been uploaded for this inspection.",
                    }
                )
            sample_rows = conn.execute(
                "SELECT sample_idx, timestamp_sec, raw_timestamp, lat, lon, altitude, speed_mps, horizontal_accuracy_m, heading_deg "
                "FROM inspection_gps_samples WHERE job_id = ? ORDER BY timestamp_sec ASC",
                (jid,),
            ).fetchall()
        meta = dict(meta_row)
        pts = [dict(r) for r in sample_rows]
        return sanitize_json_value(
            {
                "job_id": jid,
                "gps_available": len(pts) > 0,
                "metadata": meta,
                "route_points": pts,
                "provenance_note": (
                    "External GPS telemetry represents the approximate camera/vehicle path, "
                    "not independently surveyed road-surface coordinates."
                ),
            }
        )

    def get_job_locations(
        self,
        job: Dict[str, Any],
        *,
        measurement_state: Optional[str] = None,
        quality_category: Optional[str] = None,
        include_geojson: bool = False,
    ) -> Dict[str, Any]:
        jid = validate_uuid(job["job_id"])
        has_gps = self.has_gps_for_job(jid)
        if not has_gps:
            empty_res = {
                "job_id": jid,
                "gps_available": False,
                "locations": [],
                "synchronized_count": 0,
                "unavailable_count": 0,
                "note": "No external GPS telemetry has been uploaded for this inspection.",
            }
            if include_geojson:
                empty_res["geojson"] = export_locations_geojson([], [])
            return sanitize_json_value(empty_res)

        with self.db.read_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM track_geolocations WHERE job_id = ? ORDER BY track_id ASC",
                (jid,),
            ).fetchall()

        if not rows and job.get("status") == JOB_STATUS_COMPLETED:
            self.sync_job_tracks_with_gps(job)
            with self.db.read_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM track_geolocations WHERE job_id = ? ORDER BY track_id ASC",
                    (jid,),
                ).fetchall()

        # Attach cluster membership info if matched
        with self.db.read_connection() as conn:
            member_rows = conn.execute(
                """
                SELECT m.track_id, m.track_generation_id, m.cluster_id, c.match_status, c.review_status
                FROM geospatial_cluster_members m
                JOIN geospatial_clusters c ON c.cluster_id = m.cluster_id
                WHERE m.job_id = ?
                """,
                (jid,),
            ).fetchall()
        cluster_map = {
            (int(r["track_id"]), int(r["track_generation_id"])): {
                "cluster_id": r["cluster_id"],
                "match_status": r["match_status"],
                "review_status": r["review_status"],
            }
            for r in member_rows
        }

        locations: List[Dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            if measurement_state and item.get("measurement_state") != measurement_state:
                continue
            if quality_category and item.get("quality_category") != quality_category:
                continue
            cinfo = cluster_map.get((int(item["track_id"]), int(item["track_generation_id"])))
            item["cluster"] = cinfo
            locations.append(item)

        synced_cnt = sum(1 for l in locations if l.get("lat") is not None and l.get("lon") is not None)
        res: Dict[str, Any] = {
            "job_id": jid,
            "gps_available": True,
            "locations": locations,
            "total": len(locations),
            "synchronized_count": synced_cnt,
            "unavailable_count": len(locations) - synced_cnt,
            "provenance_note": (
                "Track locations represent approximate vehicle/camera GPS positions at observation "
                "timestamp, NOT surveyed physical pothole coordinates."
            ),
        }
        if include_geojson:
            route = self.get_job_route(jid)
            res["geojson"] = export_locations_geojson(locations, route.get("route_points", []))
        return sanitize_json_value(res)

    def run_candidate_matching(
        self,
        *,
        job_manager: JobManager,
        job_ids: Optional[Sequence[str]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg_kwargs = {
            "max_match_distance_m": self.default_geo_cfg.max_match_distance_m,
            "max_cluster_diameter_m": self.default_geo_cfg.max_cluster_diameter_m,
            "max_horizontal_accuracy_m": self.default_geo_cfg.max_horizontal_accuracy_m,
            "max_heading_diff_deg": self.default_geo_cfg.max_heading_diff_deg,
            "min_quality_index_for_auto_match": self.default_geo_cfg.min_quality_index_for_auto_match,
            "relative_depth_diff_limit": self.default_geo_cfg.relative_depth_diff_limit,
        }
        if config_overrides:
            for k in list(cfg_kwargs.keys()):
                if k in config_overrides and config_overrides[k] is not None:
                    cfg_kwargs[k] = float(config_overrides[k])
        if cfg_kwargs["max_cluster_diameter_m"] < cfg_kwargs["max_match_distance_m"]:
            cfg_kwargs["max_cluster_diameter_m"] = cfg_kwargs["max_match_distance_m"]
        geo_cfg = GeospatialConfig(**cfg_kwargs)

        # Ensure any completed GPS jobs requested are synced
        target_ids: Optional[List[str]] = None
        if job_ids:
            target_ids = [validate_uuid(j) for j in job_ids]
            for jid in target_ids:
                job = job_manager.get_job(jid, include_internal_paths=True)
                if self.has_gps_for_job(jid):
                    with self.db.read_connection() as conn:
                        cnt = conn.execute(
                            "SELECT COUNT(*) AS c FROM track_geolocations WHERE job_id = ?", (jid,)
                        ).fetchone()["c"]
                    if cnt == 0:
                        self.sync_job_tracks_with_gps(job)

        with self.db.read_connection() as conn:
            if target_ids:
                placeholders = ",".join("?" for _ in target_ids)
                rows = conn.execute(
                    f"SELECT * FROM track_geolocations WHERE job_id IN ({placeholders})",
                    tuple(target_ids),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM track_geolocations").fetchall()

        candidates = [dict(r) for r in rows]
        result = cluster_geospatial_candidates(candidates, cfg=geo_cfg)
        now = _utc_now()

        with self.db.transaction(immediate=True) as conn:
            conn.execute("DELETE FROM geospatial_cluster_members")
            conn.execute("DELETE FROM geospatial_clusters")

            for cl in result["clusters"]:
                diag_blob = json.dumps(
                    {
                        "match_reasons": cl["match_reasons"],
                        "thresholds": result["thresholds"],
                    },
                    allow_nan=False,
                )
                conn.execute(
                    """
                    INSERT INTO geospatial_clusters (
                        cluster_id, created_at, updated_at, centroid_lat, centroid_lon,
                        diameter_m, match_status, review_status, member_count,
                        distinct_trip_count, contains_synthetic, diagnostics_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cl["cluster_id"],
                        now,
                        now,
                        cl["centroid_lat"],
                        cl["centroid_lon"],
                        cl["diameter_m"],
                        cl["match_status"],
                        cl["review_status"],
                        cl["member_count"],
                        cl["distinct_trip_count"],
                        1 if cl.get("contains_synthetic") else 0,
                        diag_blob,
                    ),
                )
                for m in cl["members"]:
                    conn.execute(
                        """
                        INSERT INTO geospatial_cluster_members (
                            cluster_id, job_id, track_id, track_generation_id,
                            distance_to_centroid_m, member_metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            cl["cluster_id"],
                            m["job_id"],
                            int(m["track_id"]),
                            int(m.get("track_generation_id", 0)),
                            m.get("distance_to_centroid_m"),
                            json.dumps(sanitize_json_value(m), allow_nan=False),
                        ),
                    )
        return sanitize_json_value(result)

    def list_clusters(
        self,
        *,
        match_status: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        with self.db.read_connection() as conn:
            if match_status:
                total = conn.execute(
                    "SELECT COUNT(*) AS c FROM geospatial_clusters WHERE match_status = ?",
                    (match_status,),
                ).fetchone()["c"]
                rows = conn.execute(
                    """
                    SELECT * FROM geospatial_clusters
                    WHERE match_status = ?
                    ORDER BY member_count DESC, updated_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (match_status, page_size, (page - 1) * page_size),
                ).fetchall()
            else:
                total = conn.execute("SELECT COUNT(*) AS c FROM geospatial_clusters").fetchone()["c"]
                rows = conn.execute(
                    """
                    SELECT * FROM geospatial_clusters
                    ORDER BY member_count DESC, updated_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (page_size, (page - 1) * page_size),
                ).fetchall()

        clusters = []
        for r in rows:
            d = dict(r)
            diag = json.loads(d.pop("diagnostics_json") or "{}")
            d["contains_synthetic"] = bool(d.get("contains_synthetic"))
            d["match_reasons"] = diag.get("match_reasons") or []
            clusters.append(d)

        return sanitize_json_value(
            {
                "clusters": clusters,
                "total": int(total),
                "page": page,
                "page_size": page_size,
                "scientific_disclaimer": (
                    "Candidate clusters group nearby approximate vehicle GPS observations across trips; "
                    "they are NOT independently confirmed unique physical potholes."
                ),
            }
        )

    def get_cluster(self, cluster_id: str) -> Dict[str, Any]:
        cid = validate_uuid(cluster_id)
        with self.db.read_connection() as conn:
            row = conn.execute(
                "SELECT * FROM geospatial_clusters WHERE cluster_id = ?", (cid,)
            ).fetchone()
            if not row:
                raise NotFoundError(f"Unknown cluster '{cid}'", error_code="CLUSTER_NOT_FOUND")
            member_rows = conn.execute(
                "SELECT * FROM geospatial_cluster_members WHERE cluster_id = ? ORDER BY job_id ASC, track_id ASC",
                (cid,),
            ).fetchall()

        d = dict(row)
        diag = json.loads(d.pop("diagnostics_json") or "{}")
        d["contains_synthetic"] = bool(d.get("contains_synthetic"))
        d["match_reasons"] = diag.get("match_reasons") or []
        d["thresholds"] = diag.get("thresholds") or {}
        members = []
        for mr in member_rows:
            md = json.loads(mr["member_metadata_json"] or "{}")
            md["distance_to_centroid_m"] = mr["distance_to_centroid_m"]
            members.append(md)
        d["members"] = members
        return sanitize_json_value(d)

    def update_cluster_review(self, cluster_id: str, review_status: str) -> Dict[str, Any]:
        cid = validate_uuid(cluster_id)
        norm = str(review_status or "").strip().upper()
        if norm not in REVIEW_STATUSES:
            raise BadRequestError(
                f"Invalid review_status '{norm}'; expected {REVIEW_STATUSES}",
                error_code="INVALID_REVIEW_STATUS",
            )
        now = _utc_now()
        with self.db.transaction(immediate=True) as conn:
            cur = conn.execute(
                "UPDATE geospatial_clusters SET review_status = ?, updated_at = ? WHERE cluster_id = ?",
                (norm, now, cid),
            )
            if cur.rowcount == 0:
                raise NotFoundError(f"Unknown cluster '{cid}'", error_code="CLUSTER_NOT_FOUND")
        return self.get_cluster(cid)

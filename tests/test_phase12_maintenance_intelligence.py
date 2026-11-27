"""
tests/test_phase12_maintenance_intelligence.py — Unit, persistence, migration,
analytics, API, and report tests for Phase 12 Road Maintenance Intelligence.
"""

import csv
import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
import uuid

from flask import Flask

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roadvision_api_v2 import register_v2_api
from roadvision_api_v2.artifacts import build_artifact_manifest
from roadvision_api_v2.configuration import APIConfig
from roadvision_api_v2.job_manager import JOB_STATUS_COMPLETED, JobManager
from roadvision_api_v2.storage import JobDatabase, SCHEMA_SQL
from roadvision_v2.maintenance import (
    aggregate_job_maintenance,
    build_road_segment_analytics,
    compare_inspections,
    compute_review_priority,
    render_inspection_report_csv,
    render_inspection_report_html,
)


def _sample_track_and_obs(job_id: str):
    tracks = [
        {
            "source_session_id": job_id,
            "track_id": 1,
            "track_generation_id": 0,
            "first_seen_frame_idx": 0,
            "last_seen_frame_idx": 9,
            "observation_count": 10,
            "box_xyxy": [100.0, 200.0, 180.0, 260.0],
            "latest_state": "FRESH_VALID",
            "quality_summary": {
                "track_quality_category": "HIGH_QUALITY",
                "median_quality_index": 0.86,
                "median_fresh_quality_index": 0.88,
                "fresh_valid_count": 6,
                "fresh_invalid_count": 0,
                "cached_measurement_count": 4,
                "expired_count": 0,
                "not_evaluated_count": 0,
                "evidence_sufficiency": "SUFFICIENT",
                "median_relative_score": 0.14,
                "relative_depth_variability": 0.12,
                "component_medians": {"detection_quality": 0.85, "geometric_reference_quality": 0.90},
            },
        },
        {
            "source_session_id": job_id,
            "track_id": 2,
            "track_generation_id": 0,
            "first_seen_frame_idx": 12,
            "last_seen_frame_idx": 17,
            "observation_count": 6,
            "box_xyxy": [210.0, 220.0, 270.0, 270.0],
            "latest_state": "EXPIRED",
            "quality_summary": {
                "track_quality_category": "INVALID",
                "median_quality_index": None,
                "median_fresh_quality_index": None,
                "fresh_valid_count": 0,
                "fresh_invalid_count": 2,
                "cached_measurement_count": 1,
                "expired_count": 3,
                "not_evaluated_count": 0,
                "evidence_sufficiency": "INSUFFICIENT",
                "median_relative_score": None,
                "relative_depth_variability": None,
                "dominant_rejection_reasons": [{"code": "low_inlier_ratio", "count": 2, "title": "Low inlier ratio", "explanation": "Noisy surround."}],
            },
        },
        {
            "source_session_id": job_id,
            "track_id": 3,
            "track_generation_id": 0,
            "first_seen_frame_idx": 20,
            "last_seen_frame_idx": 20,
            "observation_count": 1,
            "box_xyxy": [50.0, 230.0, 80.0, 250.0],
            "latest_state": "NOT_EVALUATED",
            "quality_summary": {
                "track_quality_category": "INSUFFICIENT_EVIDENCE",
                "median_quality_index": None,
                "median_fresh_quality_index": None,
                "fresh_valid_count": 0,
                "fresh_invalid_count": 0,
                "cached_measurement_count": 0,
                "expired_count": 0,
                "not_evaluated_count": 1,
                "evidence_sufficiency": "INSUFFICIENT",
            },
        },
    ]
    observations = []
    for i in range(10):
        observations.append({
            "frame_context": {"frame_idx": i, "timestamp_sec": i * 0.04, "source_session_id": job_id},
            "detection": {"track_id": 1, "track_generation_id": 0, "detector_confidence": 0.78, "box_xyxy": [100.0, 200.0, 180.0, 260.0]},
            "measurement": {"measurement_state": "FRESH_VALID" if i < 6 else "CACHED_VALID", "measurement_validity": True, "quality_assessment": {"quality_index": 0.88}},
        })
    for i in range(12, 18):
        observations.append({
            "frame_context": {"frame_idx": i, "timestamp_sec": i * 0.04, "source_session_id": job_id},
            "detection": {"track_id": 2, "track_generation_id": 0, "detector_confidence": 0.68, "box_xyxy": [210.0, 220.0, 270.0, 270.0]},
            "measurement": {"measurement_state": "EXPIRED" if i >= 15 else "FRESH_INVALID", "measurement_validity": False},
        })
    observations.append({
        "frame_context": {"frame_idx": 20, "timestamp_sec": 0.80, "source_session_id": job_id},
        "detection": {"track_id": 3, "track_generation_id": 0, "detector_confidence": 0.30, "box_xyxy": [50.0, 230.0, 80.0, 250.0]},
        "measurement": {"measurement_state": "NOT_EVALUATED", "measurement_validity": False},
    })
    return tracks, observations


class TestPhase12MaintenanceAnalytics(unittest.TestCase):
    def test_review_priority_separates_sufficiency_and_flags_invalid_or_missing(self):
        jid = str(uuid.uuid4())
        tracks, obs = _sample_track_and_obs(jid)
        job = {
            "job_id": jid,
            "original_filename": "clip.mp4",
            "mode": "balanced",
            "status": "COMPLETED",
            "config_snapshot": {
                "depth": {"enabled": True, "model_id": "depth_anything_v2_small", "input_size": 336, "precision": "fp16_weights"},
                "cavity": {"strategy": "elliptical_core"},
            },
            "summary": {"depth_enabled": True, "depth_model_id": "depth_anything_v2_small", "depth_input_size": 336},
        }
        agg = aggregate_job_maintenance(job=job, tracks=tracks, observations=obs)
        by_tid = {c["track_id"]: c for c in agg["candidates"]}

        # Track 1: strong persistent, high quality, fresh support -> PRIORITY_REVIEW
        self.assertEqual(by_tid[1]["review_priority_category"], "PRIORITY_REVIEW")
        self.assertGreaterEqual(by_tid[1]["review_priority_index"], 0.65)
        self.assertTrue(len(by_tid[1]["priority_reasons"]) >= 1)

        # Track 2: persistent detection but EXPIRED/FRESH_INVALID measurements -> MANUAL_VERIFICATION_REQUIRED
        self.assertEqual(by_tid[2]["review_priority_category"], "MANUAL_VERIFICATION_REQUIRED")
        self.assertEqual(by_tid[2]["evidence_sufficiency"], "INSUFFICIENT")

        # Track 3: single low-confidence frame with no depth -> INSUFFICIENT_EVIDENCE
        self.assertEqual(by_tid[3]["review_priority_category"], "INSUFFICIENT_EVIDENCE")

        # Manual confirmation switches category to REVIEWED
        reviewed_cand = dict(by_tid[1], review_status="CONFIRMED_VISUALLY")
        prio_after = compute_review_priority(reviewed_cand)
        self.assertEqual(prio_after["review_priority_category"], "REVIEWED")
        self.assertFalse(prio_after["is_metric_severity"])

    def test_segments_missing_gps_zero_length_and_real_route(self):
        jid = str(uuid.uuid4())
        tracks, obs = _sample_track_and_obs(jid)
        job = {"job_id": jid, "original_filename": "clip.mp4", "mode": "balanced", "status": "COMPLETED"}
        agg = aggregate_job_maintenance(job=job, tracks=tracks, observations=obs)

        # Missing GPS -> unavailable without fabricating segments
        seg_no_gps = build_road_segment_analytics(job=job, candidates=agg["candidates"], route_points=[])
        self.assertFalse(seg_no_gps["gps_available"])
        self.assertTrue(seg_no_gps["geographical_analysis_unavailable"])
        self.assertEqual(seg_no_gps["segments"], [])
        self.assertIsNone(seg_no_gps["observed_route_length_m"])
        self.assertEqual(seg_no_gps["session_level_summary"]["session_track_count"], 3)

        # Stationary / zero-length GPS -> never divides by zero
        stationary_pts = [
            {"timestamp_sec": 0.0, "lat": 12.9716, "lon": 77.5946},
            {"timestamp_sec": 1.0, "lat": 12.9716, "lon": 77.5946},
        ]
        geo_cands = [dict(c, has_gps=True, lat=12.9716, lon=77.5946) for c in agg["candidates"]]
        seg_zero = build_road_segment_analytics(job=job, candidates=geo_cands, route_points=stationary_pts, segment_length_m=50.0)
        self.assertTrue(seg_zero["gps_available"])
        self.assertFalse(seg_zero["route_length_reliable"])
        self.assertIsNone(seg_zero["segments"][0]["tracks_per_km_approximate"])

        # Moving ~120m North route -> multiple 50m segments
        moving_pts = [
            {"timestamp_sec": float(k), "lat": 12.9716 + k * 0.00018, "lon": 77.5946}
            for k in range(7)
        ]
        cands_moving = [
            dict(agg["candidates"][0], has_gps=True, lat=12.97165, lon=77.5946),
            dict(agg["candidates"][1], has_gps=True, lat=12.97260, lon=77.5946),
        ]
        seg_real = build_road_segment_analytics(job=job, candidates=cands_moving, route_points=moving_pts, segment_length_m=50.0)
        self.assertTrue(seg_real["route_length_reliable"])
        self.assertGreater(seg_real["observed_route_length_m"], 100.0)
        self.assertGreaterEqual(seg_real["segment_count"], 2)

    def test_historical_comparison_compatibility_and_normalization(self):
        jid1, jid2 = str(uuid.uuid4()), str(uuid.uuid4())
        t1, o1 = _sample_track_and_obs(jid1)
        t2, o2 = _sample_track_and_obs(jid2)
        j1 = {
            "job_id": jid1,
            "original_filename": "run1.mp4",
            "mode": "balanced",
            "status": "COMPLETED",
            "processed_frames": 100,
            "detection_observation_count": len(o1),
            "unique_candidate_track_count": len(t1),
            "config_snapshot": {"depth": {"enabled": True, "model_id": "depth_anything_v2_small", "input_size": 336, "precision": "fp16_weights"}, "cavity": {"strategy": "elliptical_core"}},
            "summary": {"total_frames": 100, "source_fps": 25.0, "depth_enabled": True, "depth_model_id": "depth_anything_v2_small", "depth_input_size": 336},
        }
        j2 = {
            "job_id": jid2,
            "original_filename": "run2.mp4",
            "mode": "survey",
            "status": "COMPLETED",
            "processed_frames": 50,
            "detection_observation_count": len(o2),
            "unique_candidate_track_count": len(t2),
            "config_snapshot": {"depth": {"enabled": True, "model_id": "depth_anything_v2_small", "input_size": 518, "precision": "fp16_weights"}, "cavity": {"strategy": "elliptical_core"}},
            "summary": {"total_frames": 50, "source_fps": 25.0, "depth_enabled": True, "depth_model_id": "depth_anything_v2_small", "depth_input_size": 518},
        }
        m1 = aggregate_job_maintenance(job=j1, tracks=t1, observations=o1)
        m2 = aggregate_job_maintenance(job=j2, tracks=t2, observations=o2)
        cmp_res = compare_inspections([
            {"job": j1, "maintenance": m1, "coverage": {"fresh_valid_rate_over_observations": 0.35}},
            {"job": j2, "maintenance": m2, "coverage": {"fresh_valid_rate_over_observations": 0.40}},
        ])
        self.assertFalse(cmp_res["relative_depth_compatible"])
        self.assertTrue(len(cmp_res["compatibility_warnings"]) >= 1)
        self.assertIsNone(cmp_res["inspections"][0]["comparable_median_relative_score"])
        self.assertAlmostEqual(cmp_res["inspections"][0]["tracks_per_100_frames"], 3.0)
        self.assertAlmostEqual(cmp_res["inspections"][1]["tracks_per_100_frames"], 6.0)


class TestPhase12MigrationAndAPIPersistence(unittest.TestCase):
    def test_migration_on_preexisting_db_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "legacy_v2.db")
            # Simulate an older pre-Phase-12 SQLite DB without manual_reviews
            conn = sqlite3.connect(db_path)
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            conn.close()

            db1 = JobDatabase(db_path)
            db2 = JobDatabase(db_path)  # Must run idempotently twice
            with db2.read_connection() as rconn:
                migs = [r[0] for r in rconn.execute("SELECT migration_id FROM schema_migrations").fetchall()]
            self.assertIn("001_phase12_manual_reviews", migs)
            self.assertEqual(len(migs), 1)

    def test_api_review_persistence_duplicate_prevention_and_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = APIConfig(
                enabled=True,
                upload_directory=os.path.join(tmpdir, "uploads"),
                artifact_directory=os.path.join(tmpdir, "artifacts"),
                database_path=os.path.join(tmpdir, "jobs.db"),
            )
            cfg.ensure_runtime_dirs()

            app = Flask(__name__)
            rt = register_v2_api(app, api_cfg=cfg)
            jm: JobManager = rt["job_manager"]

            jid = str(uuid.uuid4())
            out_dir = os.path.join(cfg.resolved_artifact_dir, jid)
            os.makedirs(out_dir, exist_ok=True)
            tracks, obs = _sample_track_and_obs(jid)
            tracks_path = os.path.join(out_dir, "phase8a_balanced_clip_tracks.json")
            frames_path = os.path.join(out_dir, "phase8a_balanced_clip_frames.json")
            with open(tracks_path, "w", encoding="utf-8") as fh:
                json.dump({str(t["track_id"]): t for t in tracks}, fh)
            with open(frames_path, "w", encoding="utf-8") as fh:
                json.dump(obs, fh)

            jm.create_job(
                job_id=jid,
                original_filename="clip.mp4",
                validated_input_path=os.path.join(tmpdir, "uploads", "clip.mp4"),
                mode="balanced",
                config_snapshot={"depth": {"enabled": True, "model_id": "depth_anything_v2_small", "input_size": 336, "precision": "fp16_weights"}, "cavity": {"strategy": "elliptical_core"}},
                output_dir=out_dir,
                total_frames=25,
                pipeline_version="2.0.0-phase12",
            )
            manifest, avail = build_artifact_manifest(jid, "balanced", "clip", out_dir, "COMPLETED")
            jm.claim_next_job()
            jm.mark_running(jid, total_frames=25)
            jm.complete_job(
                jid,
                processed_frames=25,
                total_frames=25,
                detection_observation_count=len(obs),
                unique_candidate_track_count=3,
                depth_invocation_count=8,
                scheduler_request_count=8,
                failed_depth_count=0,
                measurement_state_counts={"FRESH_VALID": 6, "CACHED_VALID": 5, "FRESH_INVALID": 2, "EXPIRED": 3, "NOT_EVALUATED": 1},
                throughput_fps=20.8,
                summary={"total_frames": 25, "source_fps": 25.0, "elapsed_sec": 1.2, "end_to_end_fps": 20.8, "depth_enabled": True, "depth_model_id": "depth_anything_v2_small", "depth_input_size": 336, "unique_tracks": 3, "total_hazard_observations": len(obs), "measurement_state_counts": {"FRESH_VALID": 6, "CACHED_VALID": 5, "FRESH_INVALID": 2, "EXPIRED": 3, "NOT_EVALUATED": 1}},
                telemetry={"elapsed_sec": 1.2},
                artifact_availability=avail,
                artifact_manifest=manifest,
            )

            client = app.test_client()

            # Overview & Queue
            ov = client.get("/api/v2/maintenance/overview")
            self.assertEqual(ov.status_code, 200)
            self.assertEqual(ov.get_json()["completed_inspection_count"], 1)

            q = client.get(f"/api/v2/maintenance/review-queue?job_id={jid}")
            self.assertEqual(q.status_code, 200)
            self.assertEqual(q.get_json()["total"], 3)

            # Invalid filters return 400
            bad_f = client.get("/api/v2/maintenance/review-queue?priority_category=FAKE")
            self.assertEqual(bad_f.status_code, 400)

            # Create review -> 201
            cr = client.post(
                "/api/v2/maintenance/reviews",
                json={"job_id": jid, "track_id": 1, "review_status": "IN_REVIEW", "reviewer_label": "Eng-A", "review_notes": "Initial pass"},
            )
            self.assertEqual(cr.status_code, 201)
            rev_id = cr.get_json()["review_id"]
            orig_evidence = cr.get_json()["automated_evidence"]

            # Duplicate create blocked by UNIQUE constraint -> 409
            dup = client.post(
                "/api/v2/maintenance/reviews",
                json={"job_id": jid, "track_id": 1, "review_status": "CONFIRMED_VISUALLY"},
            )
            self.assertEqual(dup.status_code, 409)

            # Update review preserves automated_evidence and records full history
            up = client.patch(
                f"/api/v2/maintenance/reviews/{rev_id}",
                json={"review_status": "CONFIRMED_VISUALLY", "review_notes": "Visible depression on pavement."},
            )
            self.assertEqual(up.status_code, 200)
            up_json = up.get_json()
            self.assertEqual(up_json["review_status"], "CONFIRMED_VISUALLY")
            self.assertEqual(up_json["automated_evidence"], orig_evidence)
            self.assertEqual(len(up_json["history"]), 2)
            self.assertEqual(up_json["history"][0]["new_status"], "IN_REVIEW")
            self.assertEqual(up_json["history"][1]["new_status"], "CONFIRMED_VISUALLY")

            # Restart / new Flask app instance against same DB verifies persistence
            app2 = Flask(__name__)
            register_v2_api(app2, api_cfg=cfg)
            client2 = app2.test_client()
            persisted = client2.get(f"/api/v2/maintenance/reviews/{rev_id}")
            self.assertEqual(persisted.status_code, 200)
            self.assertEqual(persisted.get_json()["review_status"], "CONFIRMED_VISUALLY")

            # Segments with missing GPS
            seg_resp = client2.get(f"/api/v2/jobs/{jid}/segments")
            self.assertEqual(seg_resp.status_code, 200)
            self.assertTrue(seg_resp.get_json()["geographical_analysis_unavailable"])

            # Bad segment bounds
            self.assertEqual(client2.get(f"/api/v2/jobs/{jid}/segments?segment_length_m=1").status_code, 400)

            # Reports: JSON, HTML, CSV, GeoJSON, and invalid format 400
            rep_json = client2.get(f"/api/v2/jobs/{jid}/report?format=json")
            self.assertEqual(rep_json.status_code, 200)
            self.assertNotIn(tmpdir, rep_json.get_data(as_text=True))

            rep_html = client2.get(f"/api/v2/jobs/{jid}/report?format=html")
            self.assertEqual(rep_html.status_code, 200)
            self.assertIn("@media print", rep_html.get_data(as_text=True))
            self.assertIn("CONFIRMED_VISUALLY", rep_html.get_data(as_text=True))

            rep_csv = client2.get(f"/api/v2/jobs/{jid}/report?format=csv")
            self.assertEqual(rep_csv.status_code, 200)
            rows = list(csv.DictReader(io.StringIO(rep_csv.get_data(as_text=True))))
            self.assertEqual(len(rows), 3)

            rep_geo = client2.get(f"/api/v2/jobs/{jid}/report?format=geojson")
            self.assertEqual(rep_geo.status_code, 200)
            self.assertEqual(rep_geo.get_json()["type"], "FeatureCollection")

            self.assertEqual(client2.get(f"/api/v2/jobs/{jid}/report?format=exe").status_code, 400)


if __name__ == "__main__":
    unittest.main()

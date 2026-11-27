"""
tests/test_phase11_geospatial_intelligence.py — Focused tests for Phase 11 GPS ingestion,
timestamp synchronization, Haversine distance, conservative candidate matching,
SQLite migration preservation, GeoJSON export, and API endpoints.
"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest

from flask import Flask

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roadvision_api_v2 import register_v2_api
from roadvision_api_v2.configuration import APIConfig
from roadvision_api_v2.storage import JobDatabase
from roadvision_v2.config import GeospatialConfig
from roadvision_v2.geospatial import (
    cluster_geospatial_candidates,
    export_locations_geojson,
    haversine_distance_m,
    parse_gps_telemetry,
    synchronize_timestamp_to_gps,
)


class TestPhase11GeospatialCore(unittest.TestCase):
    def test_valid_csv_and_gpx_parsing_and_xxe_blocked(self):
        csv_bytes = (
            "timestamp,latitude,longitude,altitude,speed,horizontal_accuracy\n"
            "2026-11-27T10:00:00Z,12.971600,77.594600,920.5,8.2,4.5\n"
            "2026-11-27T10:00:01Z,12.971690,77.594600,920.6,8.3,4.2\n"
            "2026-11-27T10:00:02Z,12.971780,77.594600,920.4,8.1,4.8\n"
        ).encode("utf-8")
        parsed_csv = parse_gps_telemetry(csv_bytes, "trip1.csv")
        self.assertEqual(parsed_csv["source_format"], "csv")
        self.assertEqual(parsed_csv["time_mode"], "iso8601")
        self.assertEqual(parsed_csv["sample_count"], 3)
        self.assertEqual(parsed_csv["samples"][0]["timestamp_sec"], 0.0)
        self.assertEqual(parsed_csv["samples"][2]["timestamp_sec"], 2.0)

        gpx_bytes = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<gpx version="1.1"><trk><trkseg>'
            '<trkpt lat="12.971600" lon="77.594600"><ele>920.0</ele><time>2026-11-27T10:00:00Z</time></trkpt>'
            '<trkpt lat="12.971700" lon="77.594600"><ele>920.2</ele><time>2026-11-27T10:00:02Z</time></trkpt>'
            "</trkseg></trk></gpx>"
        ).encode("utf-8")
        parsed_gpx = parse_gps_telemetry(gpx_bytes, "trip1.gpx")
        self.assertEqual(parsed_gpx["source_format"], "gpx")
        self.assertEqual(parsed_gpx["sample_count"], 2)

        # XXE / DOCTYPE must be rejected strictly
        xxe_gpx = b'<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><gpx></gpx>'
        with self.assertRaises(ValueError):
            parse_gps_telemetry(xxe_gpx, "bad.gpx")

    def test_invalid_coords_naive_ts_duplicates_and_video_relative(self):
        # Out-of-bounds latitude
        with self.assertRaises(ValueError):
            parse_gps_telemetry(b"timestamp,latitude,longitude\n0.0,95.0,77.0\n", "bad.csv")

        # Naive ISO timestamp without timezone rejected
        with self.assertRaises(ValueError):
            parse_gps_telemetry(b"timestamp,latitude,longitude\n2026-11-27T10:00:00,12.9,77.5\n", "naive.csv")

        # Video-relative numeric timestamps + duplicate timestamp merging + offset
        dup_csv = (
            "timestamp,latitude,longitude,horizontal_accuracy\n"
            "1.0,12.971600,77.594600,12.0\n"
            "1.0,12.971605,77.594605,3.5\n"
            "2.0,12.971700,77.594600,4.0\n"
        ).encode("utf-8")
        parsed = parse_gps_telemetry(dup_csv, "rel.csv", time_mode="video_relative", sync_offset_sec=0.5)
        self.assertEqual(parsed["time_mode"], "video_relative")
        self.assertEqual(parsed["duplicate_timestamps_merged"], 1)
        self.assertEqual(parsed["sample_count"], 2)
        self.assertEqual(parsed["samples"][0]["timestamp_sec"], 1.5)
        self.assertEqual(parsed["samples"][0]["horizontal_accuracy_m"], 3.5)

    def test_gps_interpolation_gaps_and_no_extrapolation(self):
        csv_bytes = (
            "timestamp,latitude,longitude,horizontal_accuracy\n"
            "0.0,12.971600,77.594600,4.0\n"
            "1.0,12.971700,77.594600,4.0\n"
            "6.0,12.972200,77.594600,4.0\n"
        ).encode("utf-8")
        parsed = parse_gps_telemetry(csv_bytes, "sync.csv")
        cfg = GeospatialConfig(max_sync_gap_sec=2.0, allow_extrapolation=False)

        # Interpolated at t=0.5
        s_mid = synchronize_timestamp_to_gps(0.5, parsed["samples"], cfg=cfg)
        self.assertTrue(s_mid["available"])
        self.assertEqual(s_mid["interpolation_status"], "INTERPOLATED")
        self.assertEqual(s_mid["provenance_label"], "APPROXIMATE_VEHICLE_GPS")
        self.assertAlmostEqual(s_mid["lat"], 12.97165, places=5)

        # Gap exceeded at t=3.0 (gap between 1.0 and 6.0 is 5.0s > 2.0s)
        s_gap = synchronize_timestamp_to_gps(3.0, parsed["samples"], cfg=cfg)
        self.assertFalse(s_gap["available"])
        self.assertEqual(s_gap["interpolation_status"], "UNAVAILABLE")
        self.assertEqual(s_gap["sync_quality"], "GAP_EXCEEDED")

        # Out of range (no extrapolation) at t=7.0
        s_oob = synchronize_timestamp_to_gps(7.0, parsed["samples"], cfg=cfg)
        self.assertFalse(s_oob["available"])
        self.assertEqual(s_oob["sync_quality"], "OUT_OF_RANGE")

        # Missing GPS returns unavailable cleanly
        s_none = synchronize_timestamp_to_gps(0.5, [], cfg=cfg)
        self.assertFalse(s_none["available"])
        self.assertIsNone(s_none["lat"])

    def test_haversine_matching_low_acc_rejection_and_no_chain_merging(self):
        # Verify Haversine 0.0001 deg lat ~ 11.12 m
        d = haversine_distance_m(12.97160, 77.59460, 12.97170, 77.59460)
        self.assertAlmostEqual(d, 11.12, delta=0.2)

        cfg = GeospatialConfig(
            max_match_distance_m=15.0,
            max_cluster_diameter_m=22.0,
            max_horizontal_accuracy_m=20.0,
            min_quality_index_for_auto_match=0.5,
        )
        # Synthetic labeled fixture:
        # A (job1) and B (job2) ~ 6m apart, high quality -> POTENTIAL_MATCH
        # C (job3) is 12m from B (so within 15m of B) AND ~18m from A, plus D (job4) 14m beyond C (>22m diameter from A) -> prevents chain merge!
        # E (job5) poor GPS accuracy (35m) -> UNMATCHED
        candidates = [
            {
                "job_id": "11111111-1111-1111-1111-111111111111",
                "track_id": 1,
                "track_generation_id": 0,
                "lat": 12.971600,
                "lon": 77.594600,
                "horizontal_accuracy_m": 4.0,
                "heading_deg": 0.0,
                "interpolation_status": "EXACT",
                "quality_index": 0.85,
                "median_relative_score": 0.05,
                "depth_model_id": "depth_anything_v2_small",
                "depth_input_resolution": 336,
                "is_synthetic": True,
            },
            {
                "job_id": "22222222-2222-2222-2222-222222222222",
                "track_id": 2,
                "track_generation_id": 0,
                "lat": 12.971654,  # ~6m north of A
                "lon": 77.594600,
                "horizontal_accuracy_m": 5.0,
                "heading_deg": 5.0,
                "interpolation_status": "INTERPOLATED",
                "quality_index": 0.82,
                "median_relative_score": 0.052,
                "depth_model_id": "depth_anything_v2_small",
                "depth_input_resolution": 336,
                "is_synthetic": True,
            },
            {
                "job_id": "33333333-3333-3333-3333-333333333333",
                "track_id": 3,
                "track_generation_id": 0,
                "lat": 12.971762,  # ~12m north of B, ~18m from A (still <= 22m diameter)
                "lon": 77.594600,
                "horizontal_accuracy_m": 5.0,
                "heading_deg": 2.0,
                "interpolation_status": "EXACT",
                "quality_index": 0.80,
                "median_relative_score": 0.051,
                "depth_model_id": "depth_anything_v2_small",
                "depth_input_resolution": 336,
                "is_synthetic": True,
            },
            {
                "job_id": "44444444-4444-4444-4444-444444444444",
                "track_id": 4,
                "track_generation_id": 0,
                "lat": 12.971880,  # ~13m north of C, ~31m from A (> max_cluster_diameter_m 22m!)
                "lon": 77.594600,
                "horizontal_accuracy_m": 5.0,
                "heading_deg": 0.0,
                "interpolation_status": "EXACT",
                "quality_index": 0.85,
                "is_synthetic": True,
            },
            {
                "job_id": "55555555-5555-5555-5555-555555555555",
                "track_id": 5,
                "track_generation_id": 0,
                "lat": 12.971610,
                "lon": 77.594600,
                "horizontal_accuracy_m": 35.0,  # poor accuracy rejected
                "heading_deg": 0.0,
                "interpolation_status": "EXACT",
                "quality_index": 0.90,
                "is_synthetic": True,
            },
        ]
        res = cluster_geospatial_candidates(candidates, cfg=cfg)
        self.assertEqual(res["total_candidates"], 5)
        self.assertEqual(res["ineligible_candidates_unmatched"], 1)
        # All 5 original tracks remain represented in clusters (no silent deletion)
        total_members = sum(c["member_count"] for c in res["clusters"])
        self.assertEqual(total_members, 5)
        # No cluster may exceed max_cluster_diameter_m (chain merge prevented!)
        for c in res["clusters"]:
            if c["diameter_m"] is not None:
                self.assertLessEqual(c["diameter_m"], 22.0)

    def test_geojson_export_excludes_unavailable_locations(self):
        gj = export_locations_geojson(
            [
                {"job_id": "j1", "track_id": 1, "lat": 12.97, "lon": 77.59, "provenance_label": "APPROXIMATE_VEHICLE_GPS"},
                {"job_id": "j1", "track_id": 2, "lat": None, "lon": None, "interpolation_status": "UNAVAILABLE"},
            ],
            route_points=[{"lat": 12.97, "lon": 77.59}, {"lat": 12.98, "lon": 77.60}],
        )
        self.assertEqual(gj["type"], "FeatureCollection")
        self.assertEqual(len(gj["features"]), 2)  # 1 LineString + 1 valid Point


class TestPhase11MigrationAndAPI(unittest.TestCase):
    def test_database_migration_preserves_existing_jobs_and_apis_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "legacy_jobs.db")
            # Simulate an older Phase 8B DB with only inspection_jobs table and an existing row
            conn = sqlite3.connect(db_path)
            conn.executescript(
                """
                CREATE TABLE inspection_jobs (
                    job_id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    validated_input_path TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    config_snapshot_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    updated_at TEXT NOT NULL,
                    total_frames INTEGER,
                    processed_frames INTEGER NOT NULL DEFAULT 0,
                    detection_observation_count INTEGER NOT NULL DEFAULT 0,
                    unique_candidate_track_count INTEGER NOT NULL DEFAULT 0,
                    depth_invocation_count INTEGER NOT NULL DEFAULT 0,
                    scheduler_request_count INTEGER NOT NULL DEFAULT 0,
                    failed_depth_count INTEGER NOT NULL DEFAULT 0,
                    measurement_state_counts_json TEXT NOT NULL,
                    progress_percentage REAL,
                    throughput_fps REAL,
                    output_dir TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT,
                    error_message TEXT,
                    latest_warning TEXT,
                    artifact_manifest_json TEXT,
                    artifact_availability_json TEXT NOT NULL,
                    summary_json TEXT,
                    telemetry_json TEXT,
                    pipeline_version TEXT NOT NULL
                );
                """
            )
            old_jid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
            out_dir = os.path.join(tmpdir, "artifacts", old_jid)
            os.makedirs(out_dir, exist_ok=True)
            tracks_path = os.path.join(out_dir, "phase8a_balanced_clip_tracks.json")
            frames_path = os.path.join(out_dir, "phase8a_balanced_clip_frames.json")
            with open(tracks_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "1": {
                            "source_session_id": old_jid,
                            "track_id": 1,
                            "track_generation_id": 0,
                            "first_seen_frame_idx": 0,
                            "last_seen_frame_idx": 10,
                            "observation_count": 11,
                            "box_xyxy": [10, 20, 50, 60],
                            "latest_state": "FRESH_VALID",
                        }
                    },
                    fh,
                )
            with open(frames_path, "w", encoding="utf-8") as fh:
                json.dump(
                    [
                        {
                            "frame_context": {"frame_idx": 5, "timestamp_sec": 0.5},
                            "detection": {"track_id": 1},
                            "measurement": {"measurement_state": "FRESH_VALID", "model_identity": "depth_anything_v2_small", "depth_input_resolution": 336},
                        }
                    ],
                    fh,
                )
            manifest = {
                "job_id": old_jid,
                "partial": False,
                "artifacts": [
                    {
                        "artifact_id": "tracks_json",
                        "filename": "phase8a_balanced_clip_tracks.json",
                        "artifact_type": "tracks_json",
                        "mime_type": "application/json",
                        "size_bytes": os.path.getsize(tracks_path),
                        "available": True,
                    },
                    {
                        "artifact_id": "frames_json",
                        "filename": "phase8a_balanced_clip_frames.json",
                        "artifact_type": "observations_json",
                        "mime_type": "application/json",
                        "size_bytes": os.path.getsize(frames_path),
                        "available": True,
                    },
                ],
            }
            avail = {"tracks_json": True, "frames_json": True}
            conn.execute(
                """
                INSERT INTO inspection_jobs VALUES (
                    ?, 'clip.mp4', 'in.mp4', 'balanced', '{}', 'COMPLETED',
                    '2026-11-27T00:00:00Z', '2026-11-27T00:00:01Z', '2026-11-27T00:00:05Z', '2026-11-27T00:00:05Z',
                    11, 11, 11, 1, 3, 3, 0, '{"FRESH_VALID": 11}', 100.0, 15.0, ?, 0, NULL, NULL, NULL,
                    ?, ?, '{"mode":"balanced"}', '{}', '2.0.0-phase8b'
                )
                """,
                (old_jid, out_dir, json.dumps(manifest), json.dumps(avail)),
            )
            conn.commit()
            conn.close()

            # Initialize JobDatabase & API on existing DB without data loss
            cfg = APIConfig(
                enabled=True,
                database_path=db_path,
                upload_directory=os.path.join(tmpdir, "uploads"),
                artifact_directory=os.path.join(tmpdir, "artifacts"),
            )
            app = Flask(__name__)
            register_v2_api(app, cfg)
            client = app.test_client()

            # Existing job preserved
            r_job = client.get(f"/api/v2/jobs/{old_jid}")
            self.assertEqual(r_job.status_code, 200)
            self.assertFalse(r_job.get_json()["gps_available"])

            # Geometry unavailable cleanly on old job
            r_geom = client.get(f"/api/v2/jobs/{old_jid}/geometry")
            self.assertEqual(r_geom.status_code, 200)
            self.assertFalse(r_geom.get_json()["geometry_available"])

            # Path traversal blocked on geometry snapshot endpoint
            r_trav = client.get(f"/api/v2/jobs/{old_jid}/geometry/..%2Fsecret")
            self.assertIn(r_trav.status_code, (400, 404))
            r_bad_id = client.get(f"/api/v2/jobs/{old_jid}/geometry/..bad.json")
            self.assertEqual(r_bad_id.status_code, 400)

            # Route returns gps_available=false before upload
            r_route0 = client.get(f"/api/v2/jobs/{old_jid}/route")
            self.assertEqual(r_route0.status_code, 200)
            self.assertFalse(r_route0.get_json()["gps_available"])

            # Upload valid CSV GPS telemetry
            import io

            gps_csv = (
                "timestamp,latitude,longitude,horizontal_accuracy\n"
                "0.0,12.97160,77.59460,4.0\n"
                "1.0,12.97170,77.59460,4.0\n"
            ).encode("utf-8")
            r_up = client.post(
                f"/api/v2/jobs/{old_jid}/gps",
                data={"gps": (io.BytesIO(gps_csv), "trip_a.csv"), "time_mode": "video_relative"},
                content_type="multipart/form-data",
            )
            self.assertEqual(r_up.status_code, 201)
            self.assertEqual(r_up.get_json()["synchronized_track_count"], 1)

            r_locs = client.get(f"/api/v2/jobs/{old_jid}/locations?format=geojson")
            self.assertEqual(r_locs.status_code, 200)
            gj = r_locs.get_json()
            self.assertEqual(gj["type"], "FeatureCollection")
            self.assertEqual(len(gj["features"]), 2)

            # Run cluster matching (single job -> UNMATCHED cleanly)
            r_match = client.post("/api/v2/geospatial/match", json={"job_ids": [old_jid]})
            self.assertEqual(r_match.status_code, 200)
            self.assertEqual(r_match.get_json()["status_counts"]["UNMATCHED"], 1)

            r_cl = client.get("/api/v2/geospatial/clusters")
            self.assertEqual(r_cl.status_code, 200)
            self.assertEqual(r_cl.get_json()["total"], 1)


if __name__ == "__main__":
    unittest.main()

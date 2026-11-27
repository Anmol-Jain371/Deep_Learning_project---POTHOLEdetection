"""
SQLite storage backend for RoadVision AI 2.0 inspection jobs.

Uses Python's built-in `sqlite3` module with WAL journaling, parameterized queries,
explicit thread-safe locking, and atomic `BEGIN IMMEDIATE` transactions.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS inspection_jobs (
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

CREATE INDEX IF NOT EXISTS idx_jobs_status ON inspection_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON inspection_jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON inspection_jobs(status, created_at ASC);

CREATE TABLE IF NOT EXISTS inspection_gps_tracks (
    job_id TEXT PRIMARY KEY REFERENCES inspection_jobs(job_id) ON DELETE CASCADE,
    source_format TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    time_mode TEXT NOT NULL,
    sync_offset_sec REAL NOT NULL DEFAULT 0.0,
    max_sync_gap_sec REAL NOT NULL DEFAULT 2.0,
    sample_count INTEGER NOT NULL DEFAULT 0,
    duplicate_timestamps_merged INTEGER NOT NULL DEFAULT 0,
    out_of_order_count INTEGER NOT NULL DEFAULT 0,
    min_ts_sec REAL,
    max_ts_sec REAL,
    uploaded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inspection_gps_samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES inspection_jobs(job_id) ON DELETE CASCADE,
    sample_idx INTEGER NOT NULL,
    timestamp_sec REAL NOT NULL,
    raw_timestamp TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    altitude REAL,
    speed_mps REAL,
    horizontal_accuracy_m REAL,
    heading_deg REAL
);

CREATE INDEX IF NOT EXISTS idx_gps_samples_job_ts ON inspection_gps_samples(job_id, timestamp_sec ASC);

CREATE TABLE IF NOT EXISTS track_geolocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES inspection_jobs(job_id) ON DELETE CASCADE,
    track_id INTEGER NOT NULL,
    track_generation_id INTEGER NOT NULL DEFAULT 0,
    observation_timestamp_sec REAL NOT NULL,
    lat REAL,
    lon REAL,
    horizontal_accuracy_m REAL,
    heading_deg REAL,
    original_gps_timestamp TEXT,
    sync_offset_sec REAL NOT NULL DEFAULT 0.0,
    interpolation_status TEXT NOT NULL,
    time_diff_sec REAL,
    sync_quality TEXT NOT NULL,
    gps_source TEXT NOT NULL,
    provenance_label TEXT NOT NULL,
    quality_category TEXT,
    quality_index REAL,
    median_relative_score REAL,
    depth_model_id TEXT,
    depth_input_resolution INTEGER,
    measurement_state TEXT,
    UNIQUE(job_id, track_id, track_generation_id)
);

CREATE INDEX IF NOT EXISTS idx_track_geo_job ON track_geolocations(job_id);
CREATE INDEX IF NOT EXISTS idx_track_geo_coords ON track_geolocations(lat, lon);

CREATE TABLE IF NOT EXISTS geospatial_clusters (
    cluster_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    centroid_lat REAL,
    centroid_lon REAL,
    diameter_m REAL,
    match_status TEXT NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'PENDING_REVIEW',
    member_count INTEGER NOT NULL DEFAULT 1,
    distinct_trip_count INTEGER NOT NULL DEFAULT 1,
    contains_synthetic INTEGER NOT NULL DEFAULT 0,
    diagnostics_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_geo_clusters_status ON geospatial_clusters(match_status);

CREATE TABLE IF NOT EXISTS geospatial_cluster_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id TEXT NOT NULL REFERENCES geospatial_clusters(cluster_id) ON DELETE CASCADE,
    job_id TEXT NOT NULL REFERENCES inspection_jobs(job_id) ON DELETE CASCADE,
    track_id INTEGER NOT NULL,
    track_generation_id INTEGER NOT NULL DEFAULT 0,
    distance_to_centroid_m REAL,
    member_metadata_json TEXT NOT NULL,
    UNIQUE(cluster_id, job_id, track_id, track_generation_id)
);

CREATE INDEX IF NOT EXISTS idx_geo_members_cluster ON geospatial_cluster_members(cluster_id);
CREATE INDEX IF NOT EXISTS idx_geo_members_job ON geospatial_cluster_members(job_id);
"""

MIGRATIONS: tuple = (
    (
        "001_phase12_manual_reviews",
        """
        CREATE TABLE IF NOT EXISTS manual_reviews (
            review_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES inspection_jobs(job_id) ON DELETE CASCADE,
            track_id INTEGER NOT NULL,
            track_generation_id INTEGER NOT NULL DEFAULT 0,
            cluster_id TEXT,
            review_status TEXT NOT NULL DEFAULT 'PENDING_REVIEW',
            reviewer_label TEXT,
            review_notes TEXT,
            automated_evidence_json TEXT NOT NULL,
            priority_category TEXT NOT NULL,
            priority_index REAL,
            priority_reasons_json TEXT NOT NULL,
            scoring_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(job_id, track_id, track_generation_id)
        );

        CREATE INDEX IF NOT EXISTS idx_manual_reviews_job_status ON manual_reviews(job_id, review_status);
        CREATE INDEX IF NOT EXISTS idx_manual_reviews_status_updated ON manual_reviews(review_status, updated_at DESC);

        CREATE TABLE IF NOT EXISTS manual_review_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id TEXT NOT NULL REFERENCES manual_reviews(review_id) ON DELETE CASCADE,
            previous_status TEXT,
            new_status TEXT NOT NULL,
            reviewer_label TEXT,
            review_notes TEXT,
            changed_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_manual_review_history_rid ON manual_review_history(review_id, changed_at DESC);
        """,
    ),
)


class JobDatabase:
    """Thread-safe SQLite wrapper configured with WAL mode and atomic transactions."""

    def __init__(self, db_path: str) -> None:
        self.db_path = os.path.abspath(db_path)
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._lock = threading.RLock()
        self.journal_mode = "unknown"
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=10.0,
            isolation_level=None,  # Explicit transaction control
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                mode_row = conn.execute("PRAGMA journal_mode=WAL;").fetchone()
                if mode_row is not None:
                    self.journal_mode = str(mode_row[0]).lower()
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.executescript(SCHEMA_SQL)
                self._apply_migrations(conn)
            finally:
                conn.close()

    def _apply_migrations(self, conn: sqlite3.Connection) -> None:
        """Apply additive, idempotent SQLite schema migrations safely."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                migration_id TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            """
        )
        existing = {
            str(r[0]) for r in conn.execute("SELECT migration_id FROM schema_migrations").fetchall()
        }
        now = datetime.now(timezone.utc).isoformat()
        for mid, sql_block in MIGRATIONS:
            if mid not in existing:
                conn.executescript(sql_block)
                conn.execute(
                    "INSERT INTO schema_migrations (migration_id, applied_at) VALUES (?, ?)",
                    (mid, now),
                )

    @contextmanager
    def transaction(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """
        Provide an atomic SQLite transaction protected by a process-local RLock.
        Set immediate=True for write transactions that read-then-modify state atomically.
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
                yield conn
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        """Provide a read-only connection under the storage lock."""
        with self._lock:
            conn = self._connect()
            try:
                yield conn
            finally:
                conn.close()

    def verify_wal_mode(self) -> str:
        """Query the current SQLite journal_mode pragma."""
        with self.read_connection() as conn:
            row = conn.execute("PRAGMA journal_mode;").fetchone()
            self.journal_mode = str(row[0]).lower() if row else "unknown"
            return self.journal_mode

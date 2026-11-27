"""RoadVision AI 2.0 Flask API package initialization (side-effect-free on import)."""

from __future__ import annotations

import atexit
from typing import Any, Dict, Optional

from flask import Flask

from .configuration import APIConfig, load_api_config
from .geospatial_store import GeospatialStore
from .gpu_lock import gpu_inference_lock
from .job_manager import JobManager
from .maintenance_store import MaintenanceStore
from .routes import v2_bp
from .storage import JobDatabase
from .worker import BoundedGPUWorker

__all__ = [
    "APIConfig",
    "BoundedGPUWorker",
    "JobDatabase",
    "JobManager",
    "gpu_inference_lock",
    "load_api_config",
    "register_v2_api",
    "start_v2_worker",
    "stop_v2_worker",
    "v2_bp",
]


def register_v2_api(
    app: Flask,
    api_cfg: Optional[APIConfig] = None,
    pipeline_factory=None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Register the V2 Blueprint and initialize SQLite JobDatabase, JobManager, and
    BoundedGPUWorker on `app.extensions['roadvision_v2']`.
    Does NOT automatically spawn background worker threads on import.
    """
    cfg = api_cfg or load_api_config()
    cfg.ensure_runtime_dirs()

    db = JobDatabase(cfg.resolved_database_path)
    jm = JobManager(
        db=db,
        max_queued_jobs=cfg.max_queued_jobs,
        progress_update_interval=cfg.progress_update_interval,
    )
    geo_store = GeospatialStore(db=db, api_cfg=cfg)
    maint_store = MaintenanceStore(
        db=db,
        job_manager=jm,
        geospatial_store=geo_store,
        api_cfg=cfg,
    )
    worker = BoundedGPUWorker(
        job_manager=jm,
        api_cfg=cfg,
        pipeline_factory=pipeline_factory,
        device=device,
        geospatial_store=geo_store,
    )

    runtime = {
        "config": cfg,
        "db": db,
        "job_manager": jm,
        "geospatial_store": geo_store,
        "maintenance_store": maint_store,
        "worker": worker,
    }
    app.extensions["roadvision_v2"] = runtime

    if "roadvision_v2_api" not in app.blueprints:
        app.register_blueprint(v2_bp, url_prefix="/api/v2")

    return runtime


def start_v2_worker(app: Flask, enforce_process_lock: bool = True) -> bool:
    """Recover interrupted jobs and start the bounded GPU worker explicitly."""
    rt = app.extensions.get("roadvision_v2")
    if not rt:
        return False
    worker: BoundedGPUWorker = rt["worker"]
    started = worker.start(enforce_process_lock=enforce_process_lock)
    if started:
        atexit.register(stop_v2_worker, app)
    return started


def stop_v2_worker(app: Flask, timeout: Optional[float] = None) -> bool:
    """Cooperatively stop the V2 worker thread if running."""
    rt = app.extensions.get("roadvision_v2")
    if not rt:
        return True
    worker: BoundedGPUWorker = rt["worker"]
    return worker.stop(timeout=timeout)

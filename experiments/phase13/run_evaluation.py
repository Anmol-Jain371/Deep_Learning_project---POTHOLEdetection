"""
Phase 13 Unified Reproducible Evaluation & Ablation Runner.

Usage:
  python -m experiments.phase13.run_evaluation --mode smoke --run-id phase13_smoke_01
  python -m experiments.phase13.run_evaluation --mode full  --run-id phase13_full_01 --resume
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.phase13.dataset_audit import run_dataset_audit
from experiments.phase13.depth_cavity_ablation import run_depth_and_cavity_ablation
from experiments.phase13.detection_evaluation import run_detection_evaluation
from experiments.phase13.generate_figures import generate_all_figures
from experiments.phase13.geometry_ablation import run_geometry_ablation
from experiments.phase13.quality_geo_maint_eval import run_quality_geo_maint_and_negatives
from experiments.phase13.scheduler_temporal_ablation import run_scheduler_and_temporal_ablation

ALL_MODULES = (
    "dataset_audit",
    "detection",
    "geometry",
    "depth_cavity",
    "scheduler_temporal",
    "quality_geo_maint",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(512 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _export_tables_and_raw_csv(metrics: Dict[str, Any], out_dir: str) -> None:
    tbl_dir = os.path.join(out_dir, "tables")
    os.makedirs(tbl_dir, exist_ok=True)
    raw_rows: List[Dict[str, Any]] = []

    # 1. Detection table
    det_rows: List[Dict[str, Any]] = []
    for mname, mdata in (metrics.get("detection", {}).get("models") or {}).items():
        rt = mdata.get("runtime_benchmark") or {}
        for sp_id, sp in (mdata.get("splits") or {}).items():
            row = {
                "model": mname,
                "split": sp_id,
                "independent": sp.get("independent"),
                "precision": sp.get("precision"),
                "recall": sp.get("recall"),
                "f1_score": sp.get("f1_score"),
                "map50": sp.get("map50"),
                "map50_95": sp.get("map50_95"),
                "mean_ms": rt.get("mean_ms"),
                "fps": rt.get("throughput_fps"),
                "peak_vram_mb": rt.get("peak_allocated_mb"),
            }
            det_rows.append(row)
            raw_rows.append({"experiment": "detection", "arm": f"{mname}/{sp_id}", "metric": "map50", "value": sp.get("map50"), "note": "exploratory"})
    if det_rows:
        with open(os.path.join(tbl_dir, "detection_comparison.csv"), "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(det_rows[0].keys()))
            w.writeheader()
            w.writerows(det_rows)

    # 2. Ref-plane ablation table
    ref_ab = metrics.get("geometry", {}).get("reference_plane_ablation") or {}
    if ref_ab:
        rows = [{"method": k, **v} for k, v in ref_ab.items()]
        with open(os.path.join(tbl_dir, "reference_plane_ablation.csv"), "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        for k, v in ref_ab.items():
            raw_rows.append({"experiment": "reference_plane", "arm": k, "metric": "outlier_mean_plane_rmse", "value": v.get("outlier_mean_plane_rmse"), "note": "synthetic"})

    # 3. Depth ablation table
    d_ab = metrics.get("depth_cavity", {}).get("depth_model_and_resolution_ablation") or {}
    if d_ab:
        rows = [{"arm": k, **v} for k, v in d_ab.items()]
        with open(os.path.join(tbl_dir, "depth_resolution_ablation.csv"), "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        for k, v in d_ab.items():
            raw_rows.append({"experiment": "depth", "arm": k, "metric": "inference_ms_mean", "value": v.get("inference_ms_mean"), "note": "exploratory_rtx2050"})

    # 4. Scheduler ablation table
    s_ab = metrics.get("scheduler_temporal", {}).get("scheduling_ablation") or {}
    if s_ab:
        rows = [{"strategy": k, **v} for k, v in s_ab.items()]
        with open(os.path.join(tbl_dir, "scheduler_ablation.csv"), "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        for k, v in s_ab.items():
            raw_rows.append({"experiment": "scheduler", "arm": k, "metric": "depth_call_reduction_pct", "value": v.get("depth_call_reduction_vs_always_on_pct"), "note": "frozen_stream"})

    with open(os.path.join(out_dir, "raw_metrics.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["experiment", "arm", "metric", "value", "note"])
        w.writeheader()
        w.writerows(raw_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 13 Scientific Evaluation Harness")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--experiments", default="all", help="Comma-separated modules or 'all'")
    parser.add_argument("--resume", action="store_true", help="Skip already COMPLETED modules in an existing run_id")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cfg_path = os.path.join(os.path.dirname(__file__), "evaluation_config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    run_id = args.run_id or f"phase13_{args.mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = os.path.join(project_root, "runs", "evaluation", "phase13b", run_id)
    if os.path.exists(out_dir) and not args.resume:
        raise FileExistsError(f"Output directory '{out_dir}' already exists; pass --resume or choose a unique --run-id.")
    os.makedirs(out_dir, exist_ok=True)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.get("device", "auto") == "auto" else cfg["device"]
    smoke = args.mode == "smoke"
    selected = list(ALL_MODULES) if args.experiments == "all" else [s.strip() for s in args.experiments.split(",") if s.strip()]

    # Record environment, config snapshot, and checkpoint hashes
    ckpt_hashes = {}
    for name, rel in {**(cfg.get("detector_checkpoints") or {}), **(cfg.get("segmentation_checkpoint") or {})}.items():
        fp = os.path.join(project_root, rel)
        if os.path.isfile(fp):
            ckpt_hashes[name] = {"path": rel, "sha256": _sha256_file(fp), "bytes": os.path.getsize(fp)}

    _write_json(
        os.path.join(out_dir, "environment.json"),
        {
            "run_id": run_id,
            "mode": args.mode,
            "started_at": _utc_now(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "checkpoint_hashes": ckpt_hashes,
        },
    )
    with open(os.path.join(out_dir, "config_snapshot.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    manifest_path = os.path.join(out_dir, "experiment_manifest.json")
    agg_path = os.path.join(out_dir, "aggregate_metrics.json")

    if args.resume and os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    else:
        manifest = {
            "run_id": run_id,
            "mode": args.mode,
            "modules": {m: {"status": "NOT_RUN", "updated_at": _utc_now(), "reason": None} for m in ALL_MODULES},
        }

    if args.resume and os.path.isfile(agg_path):
        with open(agg_path, "r", encoding="utf-8") as fh:
            aggregate = json.load(fh)
    else:
        aggregate = {"run_id": run_id, "mode": args.mode, "device": device}

    def run_mod(mod_name: str, fn):
        if args.resume and manifest["modules"].get(mod_name, {}).get("status") == "COMPLETED":
            print(f"[{mod_name}] Resumed: already COMPLETED, skipping.")
            return
        if mod_name not in selected:
            return
        print(f"[{mod_name}] RUNNING...")
        manifest["modules"][mod_name] = {"status": "RUNNING", "updated_at": _utc_now(), "reason": None}
        _write_json(manifest_path, manifest)
        try:
            res = fn()
            status = res.get("status", "COMPLETED") if isinstance(res, dict) else "COMPLETED"
            reason = res.get("reason") if isinstance(res, dict) else None
            aggregate[mod_name] = res
            manifest["modules"][mod_name] = {"status": status, "updated_at": _utc_now(), "reason": reason}
            _write_json(os.path.join(out_dir, f"result_{mod_name}.json"), res)
            print(f"[{mod_name}] {status}")
        except Exception as exc:
            manifest["modules"][mod_name] = {"status": "FAILED", "updated_at": _utc_now(), "reason": f"{type(exc).__name__}: {exc}"}
            _write_json(manifest_path, manifest)
            raise
        _write_json(manifest_path, manifest)
        _write_json(agg_path, aggregate)

    run_mod("dataset_audit", lambda: run_dataset_audit(project_root, out_dir))
    run_mod("detection", lambda: run_detection_evaluation(project_root, cfg, smoke=smoke, device=device))
    run_mod("geometry", lambda: run_geometry_ablation(smoke=smoke, seed=int(cfg.get("seed", 42))))
    run_mod("depth_cavity", lambda: run_depth_and_cavity_ablation(project_root, cfg, smoke=smoke, device=device))
    run_mod("scheduler_temporal", lambda: run_scheduler_and_temporal_ablation(project_root, cfg, smoke=smoke, device=device))
    run_mod("quality_geo_maint", lambda: run_quality_geo_maint_and_negatives(project_root, out_dir))

    _export_tables_and_raw_csv(aggregate, out_dir)
    figs = generate_all_figures(aggregate, out_dir)
    aggregate["generated_figures"] = figs
    aggregate["completed_at"] = _utc_now()
    _write_json(agg_path, aggregate)
    _write_json(manifest_path, manifest)
    print(f"Phase 13 evaluation complete: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import json
import math
import os
import sys
import time
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from roadvision_engine import RoadVisionPipeline
from roadvision_api_v2 import (
    gpu_inference_lock,
    load_api_config,
    register_v2_api,
    start_v2_worker,
    stop_v2_worker,
)

api_cfg = load_api_config()
allowed_legacy_origins = list(api_cfg.allowed_cors_origins)

app = Flask(__name__)
# Restrict legacy CORS strictly to explicit localhost/loopback origins and non-V2 routes
# so app.after_request can never override V2 Blueprint origin restrictions.
CORS(
    app,
    resources={
        r"/api/health": {"origins": allowed_legacy_origins},
        r"/api/benchmark": {"origins": allowed_legacy_origins},
        r"/api/camera-config": {"origins": allowed_legacy_origins},
        r"/api/detect": {"origins": allowed_legacy_origins},
        r"/outputs/.*": {"origins": allowed_legacy_origins},
    },
)

if api_cfg.enabled:
    register_v2_api(app, api_cfg)
    print("✅ RoadVision V2 API registered under /api/v2 (worker starts on server run)")
else:
    print("ℹ️ RoadVision V2 API disabled (feature flag off)")

# Base directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "roadvision-dashboard", "public", "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Deep Learning Engine (Legacy V1 compatibility checkpoint selection)
UPGRADED_RDD_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "pothole_rdd1666_50e", "weights", "best.pt")
PROD_V2_50E = os.path.join(BASE_DIR, "runs", "detect", "pothole_combined_v2_50e", "weights", "best.pt")
PROD_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "pothole_combined_v2", "weights", "best.pt")
BASELINE_WEIGHTS = os.path.join(BASE_DIR, "pothole_best.pt")


def _describe_checkpoint_arch(weights_path: str) -> str:
    if not os.path.exists(weights_path):
        return "architecture unavailable (checkpoint missing)"
    try:
        from ultralytics import YOLO
        m = YOLO(weights_path)
        n_params = sum(int(p.numel()) for p in m.model.parameters())
        arch_hint = "YOLOv8n-scale" if n_params < 5_000_000 else ("YOLOv8s-scale" if n_params < 18_000_000 else "YOLOv8-variant")
        return f"{arch_hint}, {n_params / 1e6:.2f}M params"
    except Exception:
        return "architecture verification unavailable"


if os.path.exists(UPGRADED_RDD_WEIGHTS):
    WEIGHTS_PATH = UPGRADED_RDD_WEIGHTS
    MODEL_LABEL = f"Legacy V1 (pothole_rdd1666_50e, {_describe_checkpoint_arch(UPGRADED_RDD_WEIGHTS)} — historical benchmark verification unavailable)"
elif os.path.exists(PROD_V2_50E):
    WEIGHTS_PATH = PROD_V2_50E
    MODEL_LABEL = f"Legacy V1 (pothole_combined_v2_50e, {_describe_checkpoint_arch(PROD_V2_50E)} — historical benchmark verification unavailable)"
elif os.path.exists(PROD_WEIGHTS):
    WEIGHTS_PATH = PROD_WEIGHTS
    MODEL_LABEL = f"Legacy V1 (pothole_combined_v2, {_describe_checkpoint_arch(PROD_WEIGHTS)})"
else:
    WEIGHTS_PATH = BASELINE_WEIGHTS
    MODEL_LABEL = f"Legacy V1 Baseline (pothole_best.pt, {_describe_checkpoint_arch(BASELINE_WEIGHTS)})"

print(f"🚀 Initializing Python Flask Backend Server...")
print(f"📦 Production Detector Checkpoint: {WEIGHTS_PATH}")
print(f"✨ Active Model: {MODEL_LABEL}")

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = RoadVisionPipeline(yolo_weights=WEIGHTS_PATH, device=device)
print(f"✅ RoadVision AI Engine Online on [{device.upper()}]!")

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "device": device,
        "detector_model": MODEL_LABEL,
        "weights": WEIGHTS_PATH,
        "tracker": "road_tracker.yaml (fuse_score: false)",
        "timestamp": time.time()
    })

def _opt_round(val, ndigits: int):
    """Round genuinely computed values while preserving 0.0 and returning None when missing/non-finite."""
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, ndigits)


LEGACY_PHYSICAL_DISCLAIMER = (
    "Uncalibrated experimental estimate — physical accuracy not established."
)


def build_legacy_audit_records(records: dict, display_id_map: dict, has_gps: bool) -> list:
    """Format legacy V1 smoother records without substituting artificial numeric defaults."""
    audit_records = []
    for tid, data in records.items():
        disp_id = display_id_map.get(tid, tid)
        sev = data.get("severity")
        is_water = (data.get("water_votes") or 0) > 0

        if is_water or (isinstance(sev, str) and "WATER" in sev):
            status = "Aquaplaning Risk / Drain Clearing"
        elif isinstance(sev, str) and "SEVERE" in sev:
            status = "Experimental Legacy Severe Tier (Unvalidated)"
        elif isinstance(sev, str) and "MODERATE" in sev:
            status = "Experimental Legacy Moderate Tier (Unvalidated)"
        else:
            status = "Experimental Legacy Logged Entry (Unvalidated)"

        raw_lat = data.get("lat")
        raw_lon = data.get("lon")
        record = {
            "id": f"P-{disp_id:03d}" if isinstance(disp_id, int) else f"P-{disp_id}",
            "timestamp": _opt_round(data.get("timestamp"), 1),
            "severity": sev,
            "depth_cm": _opt_round(data.get("smooth_cm"), 1),
            "width_cm": _opt_round(data.get("smooth_width_cm"), 1),
            "length_cm": _opt_round(data.get("smooth_length_cm"), 1),
            "volume_liters": _opt_round(data.get("volume_liters"), 2),
            "asphalt_kg": _opt_round(data.get("asphalt_kg"), 1),
            "confidence": _opt_round(data.get("confidence"), 2),
            "water": is_water,
            "lat": _opt_round(raw_lat, 5) if (has_gps and raw_lat is not None) else None,
            "lng": _opt_round(raw_lon, 5) if (has_gps and raw_lon is not None) else None,
            "status": status,
            "physical_accuracy_validated": False,
            "physical_measurement_status": LEGACY_PHYSICAL_DISCLAIMER,
        }
        audit_records.append(record)
    return audit_records


@app.route("/api/benchmark", methods=["GET"])
def get_benchmark():
    benchmark_file = os.path.join(BASE_DIR, "runs", "benchmark", "yolo_rdd1666_vs_faster_rcnn.json")
    if not os.path.exists(benchmark_file):
        benchmark_file = os.path.join(BASE_DIR, "runs", "benchmark", "yolo_vs_faster_rcnn.json")

    if os.path.exists(benchmark_file):
        with open(benchmark_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return jsonify({
            "archive_notice": "Legacy V1 research archive — historical benchmark verification unavailable.",
            "verified": False,
            "model_metadata_note": (
                "Checkpoint pothole_rdd1666_50e (~3.01M params, YOLOv8n architecture) was previously mislabeled as YOLOv8s in raw historical JSON."
            ),
            "historical_raw": raw,
        })
    return jsonify({"error": "Benchmark data not generated yet"}), 404

@app.route("/api/camera-config", methods=["GET"])
def get_camera_config():
    cfg_file = os.path.join(BASE_DIR, "config", "camera_params.json")
    if os.path.exists(cfg_file):
        import json
        with open(cfg_file, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Camera configuration file not found"}), 404

@app.route("/api/detect", methods=["POST"])
def process_video_upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    vehicle_profile = request.form.get("vehicle_profile") or request.args.get("vehicle_profile", "sedan_dashcam")

    filename = f"upload_{int(time.time())}_{file.filename.replace(' ', '_')}"
    input_path = os.path.join(UPLOAD_DIR, filename)
    file.save(input_path)

    output_filename = f"analyzed_{filename}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    csv_path = os.path.join(OUTPUT_DIR, f"audit_{int(time.time())}.csv")

    print(f"\n🎬 Received Video Upload: {filename}")
    print(f"🚗 Selected Vehicle Geometry Profile: [{vehicle_profile}]")
    print(f"⏳ Running PyTorch YOLOv8 + MiDaS Detection Pipeline...")

    try:
        with gpu_inference_lock():
            pipeline.smoother.records = {}
            pipeline.display_id_map = {}
            results = pipeline.process_video(input_path, output_path, csv_path, vehicle_profile=vehicle_profile)
        
        # Read generated audit records with sequential display IDs
        audit_records = []
        verified = {
            tid: data for tid, data in pipeline.smoother.records.items()
            if data.get("observations", 1) >= 3
        }
        if not verified:
            verified = {
                tid: data for tid, data in pipeline.smoother.records.items()
                if data.get("observations", 1) >= 2
            }
        if not verified:
            verified = pipeline.smoother.records

        has_gps = bool(results.get("has_gps", False))
        audit_records = build_legacy_audit_records(verified, pipeline.display_id_map, has_gps)

        print(f"✅ PyTorch Inference Completed! {len(audit_records)} hazards registered.")
        print(f"🎥 Output video saved to: {output_path}")

        return jsonify({
            "success": True,
            "filename": file.filename,
            "videoUrl": f"/outputs/{output_filename}",
            "stats": results,
            "auditRecords": audit_records,
            "hasGPS": has_gps,
            "physical_accuracy_validated": False,
            "physical_measurement_status": LEGACY_PHYSICAL_DISCLAIMER,
        })

    except Exception as e:
        print(f"❌ Error during video processing: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    as_attachment = request.args.get("download", "0").lower() in ["1", "true", "yes"]
    download_name = request.args.get("name", "roadvision_annotated_survey.mp4")
    return send_from_directory(
        OUTPUT_DIR,
        filename,
        as_attachment=as_attachment,
        download_name=download_name
    )

if __name__ == "__main__":
    default_host = api_cfg.host if api_cfg.enabled else "0.0.0.0"
    default_port = api_cfg.port if api_cfg.enabled else 5000
    host = os.environ.get("HOST", default_host)
    port = int(os.environ.get("PORT", default_port))
    # Disable Flask debug mode for normal V2 operation to avoid reloader & surface exposure
    debug = False if api_cfg.enabled else (os.environ.get("DEBUG", "false").lower() in ["1", "true", "yes"])
    if api_cfg.enabled and (not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true"):
        start_v2_worker(app)
    print(f"🚀 Starting RoadVision Server on http://{host}:{port} (debug={debug})...")
    try:
        app.run(host=host, port=port, debug=debug)
    finally:
        if api_cfg.enabled:
            stop_v2_worker(app)


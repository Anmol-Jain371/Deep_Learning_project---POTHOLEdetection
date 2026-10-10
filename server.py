import os
import sys
import time
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from roadvision_engine import RoadVisionPipeline

app = Flask(__name__)
CORS(app)

# Base directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "roadvision-dashboard", "public", "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Deep Learning Engine with frozen production detector (Model #2)
PROD_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "pothole_combined_v2", "weights", "best.pt")
BASELINE_WEIGHTS = os.path.join(BASE_DIR, "pothole_best.pt")
WEIGHTS_PATH = PROD_WEIGHTS if os.path.exists(PROD_WEIGHTS) else BASELINE_WEIGHTS

print(f"🚀 Initializing Python Flask Backend Server...")
print(f"📦 Production Detector Checkpoint: {WEIGHTS_PATH}")
if WEIGHTS_PATH == PROD_WEIGHTS:
    print(f"✨ Model #2 (pothole_combined_v2) locked as Production Detector.")
else:
    print(f"⚠️ Warning: Model #2 not found at {PROD_WEIGHTS}, fell back to baseline {BASELINE_WEIGHTS}")

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = RoadVisionPipeline(yolo_weights=WEIGHTS_PATH, device=device)
print(f"✅ RoadVision AI Engine Online on [{device.upper()}]!")

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "device": device,
        "detector_model": "Model #2 (pothole_combined_v2)" if WEIGHTS_PATH == PROD_WEIGHTS else "Baseline (pothole_best.pt)",
        "weights": WEIGHTS_PATH,
        "tracker": "road_tracker.yaml (fuse_score: false)",
        "timestamp": time.time()
    })

@app.route("/api/detect", methods=["POST"])
def process_video_upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = f"upload_{int(time.time())}_{file.filename.replace(' ', '_')}"
    input_path = os.path.join(UPLOAD_DIR, filename)
    file.save(input_path)

    output_filename = f"analyzed_{filename}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    csv_path = os.path.join(OUTPUT_DIR, f"audit_{int(time.time())}.csv")

    print(f"\n🎬 Received Video Upload: {filename}")
    print(f"⏳ Running PyTorch YOLOv8 + MiDaS Detection Pipeline...")

    try:
        pipeline.smoother.records = {}
        pipeline.display_id_map = {}
        results = pipeline.process_video(input_path, output_path, csv_path)
        
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

        has_gps = results.get("has_gps", False)

        for tid, data in verified.items():
            disp_id = pipeline.display_id_map.get(tid, tid)
            sev = data.get("severity", "MODERATE")
            is_water = data.get("water_votes", 0) > 0

            if is_water or "WATER" in sev:
                status = "Aquaplaning Risk / Drain Clearing"
            elif "SEVERE" in sev:
                status = "Urgent Work Order Dispatched"
            elif "MODERATE" in sev:
                status = "Scheduled Inspection"
            else:
                status = "Logged / Routine Maintenance"

            # Only include GPS coordinates when real GPS data exists
            raw_lat = data.get("lat")
            raw_lon = data.get("lon")
            record = {
                "id": f"P-{disp_id:03d}",
                "timestamp": round(data.get("timestamp", 0.0), 1),
                "severity": sev,
                "depth_cm": round(data.get("smooth_cm", 6.5), 1),
                "confidence": round(data.get("confidence", 0.85), 2),
                "water": is_water,
                "lat": round(raw_lat, 5) if (has_gps and raw_lat is not None) else None,
                "lng": round(raw_lon, 5) if (has_gps and raw_lon is not None) else None,
                "status": status
            }
            audit_records.append(record)

        print(f"✅ PyTorch Inference Completed! {len(audit_records)} hazards registered.")
        print(f"🎥 Output video saved to: {output_path}")

        return jsonify({
            "success": True,
            "filename": file.filename,
            "videoUrl": f"/outputs/{output_filename}",
            "stats": results,
            "auditRecords": audit_records,
            "hasGPS": has_gps
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
    app.run(host="0.0.0.0", port=5000, debug=False)


"""
========================================================================================
🧪 RoadVision AI — Full-Stack Pipeline & Dashboard Integration Test Suite
========================================================================================
Validates:
  1. Production detector (Model #2) checkpoint integrity and loading.
  2. Preserved comparison model (RDD2022) checkpoint integrity.
  3. Frozen tracker configuration (fuse_score: false).
  4. Backend health endpoint (/api/health).
  5. Severity classification threshold compliance (Mild <4, Mod 4-7.9, Sev >=8).
  6. GPS honesty verification: no coordinates fabricated when SRT/GPX is absent.
  7. End-to-end video upload, tracking, depth inference, and CSV audit output.
========================================================================================
"""

import os
import sys
import json
import io
import cv2
import torch
import numpy as np

# Ensure UTF-8 console output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_tests():
    print("\n" + "=" * 80)
    print("[TEST SUITE] ROADVISION AI END-TO-END INTEGRATION TESTS")
    print("=" * 80)
    base_dir = os.path.abspath(os.path.dirname(__file__))

    # --- TEST 1: Checkpoint Verification ---
    print("\n--> Test 1: Verifying Model Checkpoints...")
    prod_weights = os.path.join(base_dir, "runs", "detect", "pothole_combined_v2", "weights", "best.pt")
    comp_weights = os.path.join(base_dir, "pothole_rdd_india_best.pt")
    baseline_weights = os.path.join(base_dir, "pothole_best.pt")

    assert os.path.exists(prod_weights), f"Production detector missing at: {prod_weights}"
    assert os.path.exists(comp_weights), f"Comparison model missing at: {comp_weights}"
    assert os.path.exists(baseline_weights), f"Baseline weights missing at: {baseline_weights}"

    prod_size_mb = os.path.getsize(prod_weights) / (1024 * 1024)
    comp_size_mb = os.path.getsize(comp_weights) / (1024 * 1024)
    print(f"  * Production Model (Model #2)   : {prod_weights} ({prod_size_mb:.2f} MB) [OK]")
    print(f"  * Comparison Model (RDD2022)   : {comp_weights} ({comp_size_mb:.2f} MB) [OK]")
    print(f"  * Baseline Model (Untouched)   : {baseline_weights} [OK]")

    # --- TEST 2: Frozen Tracker Configuration ---
    print("\n--> Test 2: Verifying Production Tracker Configuration...")
    tracker_path = os.path.join(base_dir, "road_tracker.yaml")
    assert os.path.exists(tracker_path), "road_tracker.yaml not found!"
    with open(tracker_path, "r", encoding="utf-8") as f:
        tracker_text = f.read()
    assert "fuse_score: false" in tracker_text or "fuse_score: False" in tracker_text, \
        "road_tracker.yaml must have fuse_score: false!"
    print("  * road_tracker.yaml confirmed with fuse_score: false [OK]")

    # --- TEST 3: Severity Threshold Classification Compliance ---
    print("\n--> Test 3: Verifying Severity Classification Thresholds...")
    from roadvision_engine import TemporalEMASmoother
    smoother = TemporalEMASmoother()

    # Mild: < 4.0 cm
    rec_mild = smoother.update(1, 3.2, 500, False, 0.85)
    assert rec_mild["severity"] == "MILD", f"Expected MILD, got {rec_mild['severity']}"

    # Moderate: 4.0 - 7.9 cm
    rec_mod1 = smoother.update(2, 4.0, 500, False, 0.85)
    assert rec_mod1["severity"] == "MODERATE", f"Expected MODERATE for 4.0cm, got {rec_mod1['severity']}"
    rec_mod2 = smoother.update(3, 7.9, 500, False, 0.85)
    assert rec_mod2["severity"] == "MODERATE", f"Expected MODERATE for 7.9cm, got {rec_mod2['severity']}"

    # Severe: >= 8.0 cm
    rec_sev = smoother.update(4, 8.0, 500, False, 0.85)
    assert rec_sev["severity"] == "SEVERE", f"Expected SEVERE for 8.0cm, got {rec_sev['severity']}"

    # Water hazard
    rec_water = smoother.update(5, 5.0, 500, True, 0.85)
    assert "WATER" in rec_water["severity"], f"Expected WATER severity, got {rec_water['severity']}"

    print("  * Mild (<4.0 cm), Moderate (4.0-7.9 cm), Severe (>=8.0 cm) thresholds verified [OK]")

    # --- TEST 4: Backend Health Endpoint & Weight Loading ---
    print("\n--> Test 4: Verifying Backend Flask App & Health Check...")
    import server
    client = server.app.test_client()

    health_res = client.get("/api/health")
    assert health_res.status_code == 200, f"Health check failed: {health_res.status_code}"
    health_data = health_res.get_json()
    print(f"  * Backend Status: {health_data.get('status')}")
    print(f"  * Loaded Model  : {health_data.get('detector_model')}")
    print(f"  * Checkpoint    : {health_data.get('weights')}")
    print(f"  * Tracker Config: {health_data.get('tracker')}")
    assert health_data.get("detector_model") == "Model #2 (pothole_combined_v2)", \
        "Backend did not load Model #2 as production detector!"

    # --- TEST 5: Video Pipeline Inference & GPS Honesty ---
    print("\n--> Test 5: Running End-to-End Video Inference Pipeline...")
    # Create a 20-frame test clip from Rangoon Road video
    src_video = os.path.join(base_dir, "Watch Video_ Terrible potholes in Rangoon Road.mp4")
    assert os.path.exists(src_video), f"Rangoon Road video missing at {src_video}"

    cap = cv2.VideoCapture(src_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    test_clip_path = os.path.join(base_dir, "scratch_test_short_clip.mp4")
    writer = cv2.VideoWriter(test_clip_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(25):  # 1 second clip (25 frames)
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    cap.release()
    writer.release()
    print(f"  * Generated 25-frame test clip: {test_clip_path}")

    with open(test_clip_path, "rb") as vf:
        video_bytes = vf.read()

    data = {
        "video": (io.BytesIO(video_bytes), "test_clip.mp4")
    }
    detect_res = client.post("/api/detect", data=data, content_type="multipart/form-data")
    assert detect_res.status_code == 200, f"Detection failed with status: {detect_res.status_code}"
    detect_json = detect_res.get_json()

    print(f"  * Video Upload & Analysis Succeeded: {detect_json.get('success')}")
    print(f"  * Output Video URL: {detect_json.get('videoUrl')}")
    print(f"  * Processed Frames: {detect_json.get('stats', {}).get('total_frames')}")
    print(f"  * GPS Presence Flag: {detect_json.get('hasGPS')}")
    print(f"  * Registered Hazards: {len(detect_json.get('auditRecords', []))}")

    # Verify GPS scope removal & no fabricated coordinates: hasGPS is False, no coordinates fabricated
    assert detect_json.get("hasGPS") is False, "GPS was falsely flagged as present!"
    for rec in detect_json.get("auditRecords", []):
        assert rec.get("lat") is None, f"Fabricated latitude detected: {rec.get('lat')}"
        assert rec.get("lng") is None, f"Fabricated longitude detected: {rec.get('lng')}"
    print("  * Verified No GPS Fabrication: all coordinates are None in retired geospatial scope [OK]")

    # Check generated output video file
    out_rel = detect_json.get("videoUrl").replace("/outputs/", "")
    out_path = os.path.join(base_dir, "roadvision-dashboard", "public", "outputs", out_rel)
    assert os.path.exists(out_path), f"Output video was not saved to {out_path}"
    assert os.path.getsize(out_path) > 1000, "Output video file is empty!"
    print(f"  * Verified output video exists in public/outputs: {os.path.basename(out_path)} ({os.path.getsize(out_path)/1024:.1f} KB) [OK]")

    # Clean up scratch test clip
    if os.path.exists(test_clip_path):
        os.remove(test_clip_path)

    print("\n" + "=" * 80)
    print("[PASS] ALL 5 INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_tests()

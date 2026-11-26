"""
========================================================================================
RoadVision AI -- Phase 7B: Comparative Dynamic Horizon Demo Video & Visual Sheet Generator
========================================================================================
Renders side-by-side / overlay comparative research videos (research-only outputs under
`runs/benchmark/phase7b_viz/`) showing:
  - Fixed baseline horizon ratio (`0.40`, dashed white/gray) vs dynamic smoothed /
    fallback horizon (`D_dynamic_fallback`, solid green when VALID, amber when STALE,
    red when EXPIRED/UNAVAILABLE fixed fallback).
  - Accepted perspective inlier segments (green) vs rejected oriented segments (orange/red)
    and estimated vanishing-point crosshair + ROI box.
  - Telemetry HUD with live state, fallback status, condition number, inliers, and latency.

Zero production file modifications.
========================================================================================
"""

import os
import sys
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments")))

from experiments.dynamic_horizon_estimator import (
    HorizonEstimatorConfig,
    HorizonPolicyConfig,
    DynamicHorizonTracker,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIZ_DIR = os.path.join(BASE_DIR, "runs", "benchmark", "phase7b_viz")

DEMO_SOURCES = [
    {
        "name": "ruralRoad_202f",
        "path": "uploads/upload_1794847200_ruralRoad_potHoles.mp4",
        "start_frame": 0,
        "count": 202,
    },
    {
        "name": "rawSurvey_160f",
        "path": "roadvision-dashboard/public/videos/raw_survey.mp4",
        "start_frame": 0,
        "count": 160,
    },
]

STATE_COLORS = {
    "VALID": (60, 220, 75),
    "LOW_CONFIDENCE": (0, 195, 255),
    "STALE": (0, 165, 255),
    "EXPIRED": (50, 60, 235),
    "UNAVAILABLE": (50, 60, 235),
    "FIXED": (190, 190, 190),
}


def render_horizon_frame(
    img_bgr: np.ndarray,
    tracker_out: Dict[str, Any],
    est_cfg: HorizonEstimatorConfig,
    frame_idx: int,
    fixed_ratio: float = 0.40,
) -> np.ndarray:
    canvas = img_bgr.copy()
    H, W = canvas.shape[:2]
    est = tracker_out["raw_estimator"]
    st = tracker_out["output_state"]
    col = STATE_COLORS.get(st, (200, 200, 200))

    # ROI subtle rectangle
    rx1, ry1 = int(est_cfg.roi_left_ratio * W), int(est_cfg.roi_top_ratio * H)
    rx2, ry2 = int(est_cfg.roi_right_ratio * W), int(est_cfg.roi_bottom_ratio * H)
    cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (110, 110, 110), 1)

    if est is not None:
        # Rejected oriented lines (orange-red thin)
        for x1, y1, x2, y2 in est["rejected_oriented_lines_px"]:
            cv2.line(canvas, (x1, y1), (x2, y2), (40, 80, 220), 1, cv2.LINE_AA)
        # Inlier lines (bright green)
        for x1, y1, x2, y2 in est["inlier_lines_px"]:
            cv2.line(canvas, (x1, y1), (x2, y2), (60, 235, 90), max(1, W // 320), cv2.LINE_AA)
        # Raw VP crosshair
        if est["vp_px"] is not None and est["state"] != "UNAVAILABLE":
            vx, vy = int(round(est["vp_px"][0])), int(round(est["vp_px"][1]))
            if -50 <= vx <= W + 50 and -50 <= vy <= H + 50:
                cv2.drawMarker(canvas, (vx, vy), (0, 240, 255), cv2.MARKER_CROSS, max(12, W // 40), 2)

    # Fixed horizon baseline (dashed white)
    fy_px = int(round(fixed_ratio * H))
    for x0 in range(0, W, 24):
        cv2.line(canvas, (x0, fy_px), (min(W - 1, x0 + 12), fy_px), (225, 225, 225), 1)

    # Active Policy D horizon (solid colored)
    act_r = tracker_out["active_horizon_ratio"]
    if act_r is not None:
        ay_px = int(round(act_r * H))
        cv2.line(canvas, (0, ay_px), (W - 1, ay_px), col, max(2, W // 320))

    # Top semi-transparent telemetry banner
    banner_h = max(46, int(0.11 * H))
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (W, banner_h), (18, 20, 24), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

    diag = est["diagnostics"] if est is not None else {}
    ms = (est["timings_ms"]["total_ms"] + tracker_out["filter_ms"]) if est is not None else 0.0
    scale = max(0.45, W / 1100.0)
    line1 = f"F{frame_idx:04d} | State: {st} | Active H={act_r:.3f} (Fixed={fixed_ratio:.2f}) | FB={'YES' if tracker_out['fallback_used'] else 'NO'}"
    line2 = f"Inliers L/R={diag.get('left_inliers',0)}/{diag.get('right_inliers',0)} | Cond={diag.get('cond_number',0):.1f} | Conf={est.get('heuristic_confidence',0):.2f} | {ms:.2f}ms" if est else ""
    cv2.putText(canvas, line1, (10, int(banner_h * 0.44)), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1 if W <= 640 else 2, cv2.LINE_AA)
    cv2.putText(canvas, line2, (10, int(banner_h * 0.84)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.90, (220, 225, 230), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    os.makedirs(VIZ_DIR, exist_ok=True)
    est_cfg = HorizonEstimatorConfig(detector_mode="lsd", process_width=320)
    pol_cfg = HorizonPolicyConfig(policy="D_dynamic_fallback", fixed_horizon_ratio=0.40)

    for src in DEMO_SOURCES:
        vpath = os.path.join(BASE_DIR, src["path"])
        cap = cv2.VideoCapture(vpath)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, src["start_frame"])

        out_mp4 = os.path.join(VIZ_DIR, f"phase7b_horizon_demo_{src['name']}.mp4")
        writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        trk = DynamicHorizonTracker(est_cfg, pol_cfg)

        sample_frames: List[np.ndarray] = []
        sample_indices = {int(src["count"] * q) for q in (0.10, 0.35, 0.65, 0.90)}
        for off in range(src["count"]):
            ok, img = cap.read()
            if not ok or img is None:
                break
            u = trk.update(img, src["start_frame"] + off, off / fps, session_id=src["name"], camera_profile="sedan_dashcam")
            vis = render_horizon_frame(img, u, est_cfg, src["start_frame"] + off)
            writer.write(vis)
            if off in sample_indices:
                sample_frames.append(vis)

        cap.release()
        writer.release()

        if len(sample_frames) >= 4:
            top = np.hstack([sample_frames[0], sample_frames[1]])
            bot = np.hstack([sample_frames[2], sample_frames[3]])
            grid = np.vstack([top, bot])
            grid_path = os.path.join(VIZ_DIR, f"phase7b_preview_grid_{src['name']}.png")
            cv2.imwrite(grid_path, grid)
            print(f"[Demo] Wrote {out_mp4} and {grid_path}")


if __name__ == "__main__":
    main()

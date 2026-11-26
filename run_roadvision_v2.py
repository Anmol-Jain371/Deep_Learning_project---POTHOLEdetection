#!/usr/bin/env python3
"""
run_roadvision_v2.py — RoadVision AI 2.0 Standalone CLI Entry Point (Phase 8A)

Usage examples:
  # LIVE mode (detection + ByteTrack, no depth)
  python run_roadvision_v2.py --source uploads/upload_1794847200_ruralRoad_potHoles.mp4 --mode live --save-video --save-json --output-dir runs/v2_outputs/

  # BALANCED mode (DAv2 336 + budget scheduler)
  python run_roadvision_v2.py --source uploads/upload_1794847200_ruralRoad_potHoles.mp4 --mode balanced --enable-depth --save-video --save-json --save-csv

  # SURVEY mode (DAv2 518 + full diagnostics)
  python run_roadvision_v2.py --source roadvision-dashboard/public/videos/raw_survey.mp4 --mode survey --enable-depth --diagnostics --save-video --save-json --save-csv
"""

import argparse
import json
import logging
import os
import sys
import time

# Ensure project root on path
_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from roadvision_v2.config import load_v2_config, VALID_MODES
from roadvision_v2.pipeline import RoadVisionV2Pipeline


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RoadVision AI 2.0 — Phase 8A standalone prototype",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", required=True, help="Path to input video file")
    p.add_argument("--mode", choices=list(VALID_MODES), default="balanced",
                   help="Pipeline operating mode: live | balanced | survey")
    p.add_argument("--config", default=None,
                   help="Path to config YAML (default: config/roadvision_v2.yaml)")
    p.add_argument("--output-dir", default="runs/v2_outputs",
                   help="Directory for all output files")
    p.add_argument("--device", default=None,
                   help="Compute device: cuda or cpu (auto-detected if omitted)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Maximum frames to process (None = all)")
    p.add_argument("--save-video", action="store_true",
                   help="Write annotated output video (.mp4)")
    p.add_argument("--save-json", action="store_true",
                   help="Write per-frame and per-track JSON outputs")
    p.add_argument("--save-csv", action="store_true",
                   help="Write per-frame timing CSV")
    p.add_argument("--diagnostics", action="store_true",
                   help="Enable horizon proxy diagnostics and verbose rejection reasons")
    p.add_argument("--enable-depth", action="store_true", default=None,
                   help="Override config to enable depth inference")
    p.add_argument("--no-depth", action="store_true",
                   help="Override config to disable depth inference")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("run_roadvision_v2")

    # Resolve depth override
    enable_depth_override = None
    if args.enable_depth:
        enable_depth_override = True
    if args.no_depth:
        enable_depth_override = False

    # Diagnostics override
    overrides = {}
    if args.diagnostics:
        overrides["diagnostics"] = {
            "enable_horizon_proxy": True,
            "show_rejection_reasons": True,
            "visualize_annular": True,
        }

    logger.info("Loading config | mode=%s", args.mode)
    cfg = load_v2_config(
        config_path=args.config,
        mode=args.mode,
        enable_depth=enable_depth_override,
        overrides=overrides if overrides else None,
        check_assets=True,
    )

    logger.info(
        "Config loaded | mode=%s depth=%s model=%s input_size=%d",
        cfg.mode,
        cfg.depth.enabled,
        cfg.depth.model_id if cfg.depth.enabled else "none",
        cfg.depth.input_size if cfg.depth.enabled else 0,
    )

    pipeline = RoadVisionV2Pipeline(cfg=cfg, device=args.device)

    logger.info(
        "Processing: %s | save_video=%s save_json=%s save_csv=%s max_frames=%s",
        args.source, args.save_video, args.save_json, args.save_csv, args.max_frames,
    )

    t0 = time.perf_counter()
    summary = pipeline.process_video(
        source=args.source,
        output_dir=args.output_dir,
        save_video=args.save_video,
        save_json=args.save_json,
        save_csv=args.save_csv,
        diagnostics=args.diagnostics,
        max_frames=args.max_frames,
    )
    elapsed = time.perf_counter() - t0

    # Print summary
    print("\n" + "=" * 60)
    print("RoadVision AI 2.0 — Phase 8A Run Complete")
    print("=" * 60)
    print(f"  Mode            : {summary['mode']}")
    print(f"  Source          : {summary['source']}")
    print(f"  Frames          : {summary['total_frames']}")
    print(f"  Elapsed         : {elapsed:.2f}s")
    print(f"  End-to-end FPS  : {summary['end_to_end_fps']:.2f}")
    print(f"  Depth enabled   : {summary['depth_enabled']}")
    if summary['depth_enabled']:
        print(f"  Depth model     : {summary['depth_model_id']} @ {summary['depth_input_size']}px")
        print(f"  Scheduler       : {summary['scheduler_strategy']}")
        print(f"  Depth calls     : {summary['depth_calls']}/{summary['total_frames']} "
              f"({summary['depth_invocation_rate']*100:.1f}%)")
    print(f"  Unique tracks   : {summary['unique_tracks']}")
    print(f"  Hazard records  : {summary['total_hazard_observations']}")
    print(f"  Output dir      : {args.output_dir}")
    print("=" * 60)

    # Print summary JSON for scripting
    logger.debug("Session summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

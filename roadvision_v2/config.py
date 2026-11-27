"""
Configuration loading and strict validation for RoadVision AI 2.0 (Phase 8A Task 4).
Validates operating modes (`live`, `balanced`, `survey`), depth/precision pairings,
scheduler bounds, and required model checkpoint paths before initializing models.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import yaml

VALID_MODES = ("live", "balanced", "survey")
VALID_DEPTH_MODELS = ("none", "midas_small", "depth_anything_v2_small")
VALID_PRECISIONS = ("none", "fp32", "fp16_autocast", "fp16_weights")
VALID_STRATEGIES = (
    "A_always_on",
    "B_candidate_gated",
    "C_fixed_interval",
    "D_track_aware",
    "E_budget_aware",
)
VALID_CAVITY_STRATEGIES = ("full_box", "rect_trim_10", "rect_trim_20", "rect_trim_30", "elliptical_core")


@dataclass
class DetectorConfig:
    model_id: str = "yolov8n_phase2_controlled"
    checkpoint_path: str = "runs/detect/pothole_unified_yolov8n_100e_sgd/weights/best.pt"
    tracker_config: str = "road_tracker.yaml"
    input_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.50
    horizon_cutoff_ratio: float = 0.15


@dataclass
class DepthConfig:
    enabled: bool = False
    model_id: str = "midas_small"
    input_size: int = 256
    precision: str = "fp16_autocast"
    checkpoint_path: Optional[str] = None


@dataclass
class RansacConfig:
    sampling_mode: str = "stratified_quadrant"
    max_ref_samples: int = 512
    min_iters: int = 16
    max_iters: int = 128
    batch_size: int = 16
    confidence: float = 0.99
    enable_early_stop: bool = True
    seed: int = 42
    min_ref_samples: int = 40
    min_quadrant_samples: int = 5
    min_populated_quadrants: int = 2
    max_single_quadrant_frac: float = 0.80
    min_annular_retention: float = 0.25
    min_inlier_ratio: float = 0.45
    min_positive_depress_frac: float = 0.35


@dataclass
class CavityConfig:
    strategy: str = "elliptical_core"
    reject_water_patches: bool = True


@dataclass
class SchedulerConfig:
    strategy: str = "E_budget_aware"
    fixed_interval_k: int = 3
    refresh_interval_sec: float = 0.16
    max_measurement_age_sec: float = 0.45
    area_change_threshold: float = 0.30
    centroid_shift_threshold: float = 0.35
    min_box_iou: float = 0.48
    invalid_retry_base_sec: float = 0.12
    invalid_retry_max_sec: float = 0.36
    min_ref_inlier_hold: float = 0.55
    scene_change_l1_thresh: float = 38.0
    budget_target_ema_ms: float = 38.0
    budget_min_gap_sec: float = 0.08
    starvation_override_sec: float = 0.40


@dataclass
class CacheTemporalConfig:
    max_cache_age_sec: float = 0.45
    max_invalid_stale_sec: float = 0.25
    max_absence_reset_sec: float = 0.60
    max_reuse_centroid_norm: float = 1.75
    tau_sec: float = 0.16
    rolling_window: int = 5
    fixed_alpha: float = 0.35


@dataclass
class DiagnosticsConfig:
    enable_horizon_proxy: bool = False
    visualize_annular: bool = False
    visualize_ransac_inliers: bool = False
    visualize_residual_heatmap: bool = False
    show_rejection_reasons: bool = True
    show_timing_hud: bool = True


@dataclass
class QualityConfig:
    enabled: bool = True
    min_fresh_samples_for_stability: int = 3
    max_cache_age_sec: float = 0.45
    cached_freshness_penalty_max: float = 0.35
    min_confidence_for_full_score: float = 0.70
    min_ref_samples: int = 40
    ideal_ref_samples: int = 256
    min_cavity_px: int = 25
    ideal_cavity_px: int = 250
    min_inlier_ratio: float = 0.45
    ideal_inlier_ratio: float = 0.85
    mad_reference_scale: float = 0.08
    rel_cv_reference_scale: float = 0.50
    w_detection: float = 0.15
    w_geometric_ref: float = 0.30
    w_cavity_extraction: float = 0.20
    w_temporal_consistency: float = 0.15
    w_freshness: float = 0.10
    w_environmental: float = 0.10
    high_quality_min: float = 0.75
    moderate_quality_min: float = 0.50
    min_component_coverage_for_index: float = 0.50


@dataclass
class GeometrySnapshotConfig:
    enabled: bool = False
    max_grid_size: int = 48
    max_snapshots_per_track: int = 3
    max_snapshots_per_job: int = 24
    include_source_crop: bool = True
    min_quality_for_snapshot: float = 0.0


@dataclass
class GeospatialConfig:
    max_sync_gap_sec: float = 2.0
    exact_match_threshold_sec: float = 0.05
    allow_extrapolation: bool = False
    max_gps_file_bytes: int = 10485760
    max_gps_rows: int = 100000
    max_match_distance_m: float = 15.0
    max_cluster_diameter_m: float = 25.0
    max_horizontal_accuracy_m: float = 25.0
    max_heading_diff_deg: float = 45.0
    min_quality_index_for_auto_match: float = 0.45
    relative_depth_diff_limit: float = 0.50


@dataclass
class MaintenancePriorityConfig:
    scoring_version: str = "heuristic_review_priority_v1"
    w_persistence: float = 0.18
    w_detector_confidence: float = 0.16
    w_fresh_coverage: float = 0.12
    w_quality_index: float = 0.16
    w_relative_signal: float = 0.14
    w_temporal_stability: float = 0.10
    w_cross_trip: float = 0.08
    w_gps_sync: float = 0.06
    priority_review_min: float = 0.65
    persistence_full_obs: int = 10
    fresh_full_count: int = 5
    relative_signal_ref_scale: float = 0.15
    default_segment_length_m: float = 50.0


@dataclass
class RoadVisionV2Config:
    mode: str = "balanced"
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    ransac: RansacConfig = field(default_factory=RansacConfig)
    cavity: CavityConfig = field(default_factory=CavityConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    cache_temporal: CacheTemporalConfig = field(default_factory=CacheTemporalConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    geometry_snapshot: GeometrySnapshotConfig = field(default_factory=GeometrySnapshotConfig)
    geospatial: GeospatialConfig = field(default_factory=GeospatialConfig)
    maintenance: MaintenancePriorityConfig = field(default_factory=MaintenancePriorityConfig)
    project_root: str = ""

    def resolve_path(self, rel_or_abs: Optional[str]) -> Optional[str]:
        if rel_or_abs is None:
            return None
        if os.path.isabs(rel_or_abs):
            return rel_or_abs
        root = self.project_root or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.join(root, rel_or_abs)

    def validate(self, check_assets: bool = True) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode '{self.mode}'; expected one of {VALID_MODES}")
        if self.detector.input_size <= 0 or self.detector.input_size % 32 != 0:
            raise ValueError(f"detector.input_size must be positive multiple of 32, got {self.detector.input_size}")
        if not (0.0 < self.detector.conf_threshold < 1.0):
            raise ValueError(f"detector.conf_threshold must be in (0, 1), got {self.detector.conf_threshold}")
        if not (0.0 < self.detector.iou_threshold <= 1.0):
            raise ValueError(f"detector.iou_threshold must be in (0, 1], got {self.detector.iou_threshold}")
        if not (0.0 <= self.detector.horizon_cutoff_ratio < 0.6):
            raise ValueError(f"detector.horizon_cutoff_ratio must be in [0, 0.6), got {self.detector.horizon_cutoff_ratio}")

        if self.depth.model_id not in VALID_DEPTH_MODELS:
            raise ValueError(f"Invalid depth.model_id '{self.depth.model_id}'; expected {VALID_DEPTH_MODELS}")
        if self.depth.precision not in VALID_PRECISIONS:
            raise ValueError(f"Invalid depth.precision '{self.depth.precision}'; expected {VALID_PRECISIONS}")
        if self.depth.enabled:
            if self.depth.model_id == "none" or self.depth.precision == "none":
                raise ValueError("depth.enabled=True requires a concrete depth.model_id and precision != 'none'")
            if self.depth.model_id == "midas_small":
                if self.depth.input_size != 256:
                    raise ValueError(f"midas_small requires input_size=256, got {self.depth.input_size}")
                if self.depth.precision == "fp16_weights":
                    raise ValueError("midas_small supports 'fp32' or 'fp16_autocast', not 'fp16_weights'")
            if self.depth.model_id == "depth_anything_v2_small" and self.depth.input_size % 14 != 0:
                raise ValueError(f"DAv2 input_size must be a multiple of 14 (e.g., 336 or 518), got {self.depth.input_size}")

        if self.ransac.min_iters < 1 or self.ransac.max_iters < self.ransac.min_iters:
            raise ValueError("Invalid ransac iteration bounds (require 1 <= min_iters <= max_iters)")
        if self.ransac.max_ref_samples < 40:
            raise ValueError("ransac.max_ref_samples must be >= 40")
        if self.cavity.strategy not in VALID_CAVITY_STRATEGIES:
            raise ValueError(f"Invalid cavity.strategy '{self.cavity.strategy}'; expected {VALID_CAVITY_STRATEGIES}")
        if self.scheduler.strategy not in VALID_STRATEGIES:
            raise ValueError(f"Invalid scheduler.strategy '{self.scheduler.strategy}'; expected {VALID_STRATEGIES}")
        if self.cache_temporal.max_cache_age_sec <= 0.0 or self.cache_temporal.tau_sec <= 0.0:
            raise ValueError("cache_temporal.max_cache_age_sec and tau_sec must be positive")
        if self.quality.min_fresh_samples_for_stability < 2:
            raise ValueError("quality.min_fresh_samples_for_stability must be >= 2")
        if not (0.0 <= self.quality.moderate_quality_min <= self.quality.high_quality_min <= 1.0):
            raise ValueError("quality thresholds must satisfy 0 <= moderate_quality_min <= high_quality_min <= 1")
        if not (16 <= self.geometry_snapshot.max_grid_size <= 128):
            raise ValueError("geometry_snapshot.max_grid_size must be in [16, 128]")
        if self.geometry_snapshot.max_snapshots_per_track < 1 or self.geometry_snapshot.max_snapshots_per_job < 1:
            raise ValueError("geometry_snapshot snapshot limits must be >= 1")
        if self.geospatial.max_sync_gap_sec <= 0.0 or self.geospatial.max_match_distance_m <= 0.0:
            raise ValueError("geospatial.max_sync_gap_sec and max_match_distance_m must be > 0")
        if self.geospatial.max_cluster_diameter_m < self.geospatial.max_match_distance_m:
            raise ValueError("geospatial.max_cluster_diameter_m must be >= max_match_distance_m")

        if check_assets:
            det_ckpt = self.resolve_path(self.detector.checkpoint_path)
            if not det_ckpt or not os.path.exists(det_ckpt):
                raise FileNotFoundError(
                    f"Required detector checkpoint missing: '{det_ckpt}'. Configure detector.checkpoint_path explicitly."
                )
            if self.depth.enabled and self.depth.checkpoint_path:
                d_ckpt = self.resolve_path(self.depth.checkpoint_path)
                if not d_ckpt or not os.path.exists(d_ckpt):
                    raise FileNotFoundError(f"Configured depth checkpoint missing: '{d_ckpt}'.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_v2_config(
    config_path: Optional[str] = None,
    mode: str = "balanced",
    enable_depth: Optional[bool] = None,
    overrides: Optional[Dict[str, Any]] = None,
    check_assets: bool = True,
) -> RoadVisionV2Config:
    """Load YAML configuration for `live`, `balanced`, or `survey`, apply CLI overrides, and validate."""
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode '{mode}'; expected {VALID_MODES}")

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_file = config_path or os.path.join(root, "config", "roadvision_v2.yaml")
    if not os.path.exists(cfg_file):
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    modes_block = raw.get("modes", {})
    if mode not in modes_block:
        raise ValueError(f"Mode '{mode}' not defined in {cfg_file}")
    mraw = modes_block[mode]

    det_cfg = DetectorConfig(**(raw.get("detector") or {}))
    ransac_cfg = RansacConfig(**(raw.get("ransac") or {}))
    cache_cfg = CacheTemporalConfig(**(raw.get("cache_temporal") or {}))
    quality_cfg = QualityConfig(**(raw.get("quality") or {}))
    geom_cfg = GeometrySnapshotConfig(**(raw.get("geometry_snapshot") or {}))
    geo_cfg = GeospatialConfig(**(raw.get("geospatial") or {}))
    depth_cfg = DepthConfig(**(mraw.get("depth") or {}))
    cav_cfg = CavityConfig(**(mraw.get("cavity") or {}))
    sched_cfg = SchedulerConfig(**(mraw.get("scheduler") or {}))
    diag_cfg = DiagnosticsConfig(**(mraw.get("diagnostics") or {}))

    if enable_depth is not None:
        depth_cfg.enabled = bool(enable_depth)

    if overrides:
        for sec_name, sec_vals in overrides.items():
            target = {
                "detector": det_cfg,
                "depth": depth_cfg,
                "ransac": ransac_cfg,
                "cavity": cav_cfg,
                "scheduler": sched_cfg,
                "cache_temporal": cache_cfg,
                "diagnostics": diag_cfg,
                "quality": quality_cfg,
                "geometry_snapshot": geom_cfg,
                "geospatial": geo_cfg,
            }.get(sec_name)
            if target is not None and isinstance(sec_vals, dict):
                for k, v in sec_vals.items():
                    if hasattr(target, k):
                        setattr(target, k, v)

    # Keep scheduler expiry synced with cache expiry unless explicitly diverged
    if sched_cfg.max_measurement_age_sec != cache_cfg.max_cache_age_sec:
        sched_cfg.max_measurement_age_sec = cache_cfg.max_cache_age_sec

    cfg = RoadVisionV2Config(
        mode=mode,
        detector=det_cfg,
        depth=depth_cfg,
        ransac=ransac_cfg,
        cavity=cav_cfg,
        scheduler=sched_cfg,
        cache_temporal=cache_cfg,
        diagnostics=diag_cfg,
        quality=quality_cfg,
        geometry_snapshot=geom_cfg,
        geospatial=geo_cfg,
        project_root=root,
    )
    cfg.validate(check_assets=check_assets)
    return cfg

"""
Unified typed data contracts for RoadVision AI 2.0 (Phase 8A Task 5).
Every unavailable value is explicitly `None` (`null` in JSON) — never `0.0`.
No physical metric cm or physical severity categories are assigned from relative scores.
"""

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

MEASUREMENT_STATES = (
    "FRESH_VALID",
    "FRESH_INVALID",
    "CACHED_VALID",
    "EXPIRED",
    "NOT_EVALUATED",
)

INSPECTION_PROVENANCE_CATEGORIES = (
    "USER_INSPECTION",
    "RESEARCH_EXPLORATORY",
    "TEST_FIXTURE",
    "SYNTHETIC_EXPERIMENT",
    "DEMONSTRATION",
    "UNKNOWN",
)

NON_GENUINE_INSPECTION_PROVENANCE = (
    "TEST_FIXTURE",
    "SYNTHETIC_EXPERIMENT",
    "DEMONSTRATION",
    "UNKNOWN",
)

GPS_PROVENANCE_CATEGORIES = (
    "USER_SUPPLIED_UNVERIFIED",
    "VERIFIED_FIELD_LOG",
    "SYNTHETIC",
    "UNKNOWN",
)


def _clean_json(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        return _clean_json(obj.item())
    return obj


@dataclass
class FrameContext:
    source_session_id: str
    frame_id: str
    frame_idx: int
    timestamp_sec: float
    frame_wh: Tuple[int, int]
    mode: str
    camera_profile: str = "sedan_dashcam"
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    has_gps: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))


@dataclass
class DetectionObservation:
    track_id: int
    track_generation_id: int
    box_xyxy: Tuple[float, float, float, float]
    detector_confidence: float
    class_id: int
    model_identity: str
    detector_input_size: int
    is_below_horizon_cutoff: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))


@dataclass
class TrackObservation:
    source_session_id: str
    track_id: int
    track_generation_id: int
    first_seen_frame_idx: int
    last_seen_frame_idx: int
    observation_count: int
    box_xyxy: Tuple[float, float, float, float]
    latest_state: str
    quality_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))


@dataclass
class ReferenceFitDiagnostics:
    used_ref_samples: Optional[int] = None
    available_ref_samples: Optional[int] = None
    populated_quadrants: Optional[int] = None
    inlier_ratio: Optional[float] = None
    ref_fit_mad: Optional[float] = None
    ref_fit_rmse: Optional[float] = None
    annular_retention_ratio: Optional[float] = None
    plane_degenerate: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))


@dataclass
class SchedulingDecision:
    frame_id: str
    frame_idx: int
    timestamp_sec: float
    strategy: str
    run_depth: bool
    frame_reasons: List[str] = field(default_factory=list)
    per_track_reasons: Dict[int, str] = field(default_factory=dict)
    shared_across_tracks: int = 0
    scheduler_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))


@dataclass
class RelativeDepthMeasurement:
    source_session_id: str
    frame_id: str
    frame_idx: int
    timestamp_sec: float
    track_id: int
    track_generation_id: int
    box_xyxy: Tuple[float, float, float, float]
    detector_confidence: float
    model_identity: str
    depth_input_resolution: Optional[int]
    inference_precision: Optional[str]
    raw_relative_cavity_score: Optional[float]
    normalized_relative_score: Optional[float]
    smoothed_relative_score: Optional[float]
    measurement_validity: bool
    measurement_state: str
    measurement_age_sec: Optional[float]
    ref_diagnostics: ReferenceFitDiagnostics = field(default_factory=ReferenceFitDiagnostics)
    failure_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_water_suspected: bool = False
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    quality_assessment: Optional[Dict[str, Any]] = None
    geometry_snapshot_id: Optional[str] = None
    geometry_snapshot_timestamp_sec: Optional[float] = None

    def __post_init__(self) -> None:
        if self.measurement_state not in MEASUREMENT_STATES:
            raise ValueError(f"Invalid measurement_state '{self.measurement_state}'; expected {MEASUREMENT_STATES}")
        if not self.measurement_validity:
            self.raw_relative_cavity_score = None
            self.normalized_relative_score = None
            if self.measurement_state != "CACHED_VALID":
                self.smoothed_relative_score = None

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))


@dataclass
class HazardObservation:
    frame_context: FrameContext
    detection: DetectionObservation
    measurement: RelativeDepthMeasurement
    horizon_diag: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))


@dataclass
class PipelinePerformance:
    frame_id: str
    frame_idx: int
    mode: str
    decode_ms: float = 0.0
    detect_track_ms: float = 0.0
    scheduler_ms: float = 0.0
    depth_pre_ms: float = 0.0
    depth_inf_ms: float = 0.0
    depth_resize_ms: float = 0.0
    extract_ms: float = 0.0
    horizon_diag_ms: float = 0.0
    visualize_ms: float = 0.0
    total_frame_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))

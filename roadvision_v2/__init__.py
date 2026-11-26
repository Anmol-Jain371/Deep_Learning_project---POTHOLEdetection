"""
RoadVision AI 2.0 -- Modular Prototype Package (Phase 8A).
Strictly parallel to legacy `roadvision_engine.py` and `server.py`; outputs relative
uncalibrated inverse-depth scores (`null` when invalid) and never emits physical cm/severity.
"""

from roadvision_v2.config import (
    RoadVisionV2Config,
    VALID_MODES,
    load_v2_config,
)
from roadvision_v2.schemas import (
    MEASUREMENT_STATES,
    FrameContext,
    DetectionObservation,
    TrackObservation,
    ReferenceFitDiagnostics,
    SchedulingDecision,
    RelativeDepthMeasurement,
    HazardObservation,
    PipelinePerformance,
)
from roadvision_v2.pipeline import RoadVisionV2Pipeline

__all__ = [
    "RoadVisionV2Config",
    "VALID_MODES",
    "load_v2_config",
    "MEASUREMENT_STATES",
    "FrameContext",
    "DetectionObservation",
    "TrackObservation",
    "ReferenceFitDiagnostics",
    "SchedulingDecision",
    "RelativeDepthMeasurement",
    "HazardObservation",
    "PipelinePerformance",
    "RoadVisionV2Pipeline",
]

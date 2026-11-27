"""Phase 10: Relative 2.5D cavity surface visualization and snapshot exports."""

from .surface import (
    RESIDUAL_POLARITY_DOC,
    GeometrySnapshotManager,
    build_relative_surface_snapshot,
    compute_cross_section_profiles,
)

__all__ = [
    "RESIDUAL_POLARITY_DOC",
    "GeometrySnapshotManager",
    "build_relative_surface_snapshot",
    "compute_cross_section_profiles",
]

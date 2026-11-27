"""
Phase 12: Road Maintenance Intelligence, Explainable Review Prioritization,
Road Segment Analytics, Historical Comparison, and Structured Inspection Reports.
"""

from .analytics import (
    MAINTENANCE_SCIENTIFIC_DISCLAIMER,
    MAINTENANCE_SCORING_VERSION,
    REVIEW_PRIORITY_CATEGORIES,
    REVIEW_STATUSES,
    aggregate_job_maintenance,
    build_road_segment_analytics,
    compare_inspections,
    compute_review_priority,
)
from .reports import (
    build_inspection_report_payload,
    render_inspection_report_csv,
    render_inspection_report_html,
)

__all__ = [
    "MAINTENANCE_SCIENTIFIC_DISCLAIMER",
    "MAINTENANCE_SCORING_VERSION",
    "REVIEW_PRIORITY_CATEGORIES",
    "REVIEW_STATUSES",
    "aggregate_job_maintenance",
    "build_inspection_report_payload",
    "build_road_segment_analytics",
    "compare_inspections",
    "compute_review_priority",
    "render_inspection_report_csv",
    "render_inspection_report_html",
]

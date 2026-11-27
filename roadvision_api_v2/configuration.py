"""Configuration loader and strict validator for RoadVision AI 2.0 Backend API."""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "roadvision_api_v2.yaml")
CONFIG_PATH = DEFAULT_CONFIG_PATH  # Backward-compatible alias

VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ENV_ENABLED_KEYS = ("ROADVISION_V2_ENABLED", "ROADVISION_V2_API_ENABLED")


def _parse_bool_env(val: Optional[str]) -> Optional[bool]:
    if val is None:
        return None
    norm = val.strip().lower()
    if norm in ("1", "true", "yes", "on"):
        return True
    if norm in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean environment value: {val!r}")


def _is_localhost(host: str) -> bool:
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass
class APIConfig:
    """Validated runtime configuration for the RoadVision AI 2.0 REST backend."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 5000
    database_path: str = "runs/v2_api/roadvision_v2_jobs.db"
    upload_directory: str = "uploads_v2"
    artifact_directory: str = "artifacts_v2"
    maximum_upload_size: int = 268435456  # 256 MiB
    allowed_video_formats: List[str] = field(
        default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"]
    )
    maximum_video_resolution: Tuple[int, int] = (1920, 1080)
    maximum_video_duration: float = 300.0
    maximum_processed_frames: int = 9000
    max_queued_jobs: int = 5
    max_gpu_concurrency: int = 1
    progress_update_interval: float = 0.5
    worker_shutdown_timeout: float = 15.0
    artifact_retention_policy: str = "retain_latest_50"
    allowed_cors_origins: List[str] = field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )
    default_page_size: int = 100
    max_page_size: int = 500
    enable_h264_transcode: bool = True
    project_root: str = PROJECT_ROOT

    def resolve_path(self, path_val: str) -> str:
        """Resolve a path relative to project_root cleanly."""
        if os.path.isabs(path_val):
            return os.path.normpath(path_val)
        return os.path.normpath(os.path.join(self.project_root, path_val))

    @property
    def resolved_database_path(self) -> str:
        return self.resolve_path(self.database_path)

    @property
    def resolved_upload_dir(self) -> str:
        return self.resolve_path(self.upload_directory)

    @property
    def resolved_artifact_dir(self) -> str:
        return self.resolve_path(self.artifact_directory)

    def ensure_runtime_dirs(self) -> None:
        """Create required runtime directories for database, uploads, and artifacts."""
        db_dir = os.path.dirname(self.resolved_database_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        os.makedirs(self.resolved_upload_dir, exist_ok=True)
        os.makedirs(self.resolved_artifact_dir, exist_ok=True)

    def validate(self) -> None:
        """Validate all configuration bounds and safety invariants."""
        if not isinstance(self.enabled, bool):
            raise ValueError(f"enabled must be a boolean, got {self.enabled!r}")
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host must be a non-empty string")
        if not _is_localhost(self.host.strip()):
            raise ValueError(
                f"V2 research backend must bind to localhost/loopback only, got host={self.host!r}"
            )
        if not isinstance(self.port, int) or isinstance(self.port, bool) or not (1 <= self.port <= 65535):
            raise ValueError(f"port must be an integer in [1, 65535], got {self.port!r}")

        for label, pval in (
            ("database_path", self.database_path),
            ("upload_directory", self.upload_directory),
            ("artifact_directory", self.artifact_directory),
        ):
            if not isinstance(pval, str) or not pval.strip():
                raise ValueError(f"{label} must be a non-empty path string")

        if isinstance(self.maximum_upload_size, bool) or not isinstance(self.maximum_upload_size, int) or self.maximum_upload_size <= 0:
            raise ValueError("maximum_upload_size must be a positive integer (bytes)")

        if not isinstance(self.allowed_video_formats, list) or not self.allowed_video_formats:
            raise ValueError("allowed_video_formats must be a non-empty list")
        normalized_exts: List[str] = []
        for ext in self.allowed_video_formats:
            if not isinstance(ext, str) or not ext.startswith("."):
                raise ValueError(f"Invalid video format extension: {ext!r}")
            low = ext.lower().strip()
            if low not in VALID_VIDEO_EXTENSIONS:
                raise ValueError(
                    f"Unsupported video format {ext!r}; supported set: {sorted(VALID_VIDEO_EXTENSIONS)}"
                )
            if low not in normalized_exts:
                normalized_exts.append(low)
        self.allowed_video_formats = normalized_exts

        if (
            not isinstance(self.maximum_video_resolution, (list, tuple))
            or len(self.maximum_video_resolution) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in self.maximum_video_resolution)
        ):
            raise ValueError("maximum_video_resolution must be a 2-tuple of positive ints [width, height]")
        self.maximum_video_resolution = (
            int(self.maximum_video_resolution[0]),
            int(self.maximum_video_resolution[1]),
        )

        if isinstance(self.maximum_video_duration, bool) or float(self.maximum_video_duration) <= 0.0:
            raise ValueError("maximum_video_duration must be > 0 seconds")
        self.maximum_video_duration = float(self.maximum_video_duration)

        if isinstance(self.maximum_processed_frames, bool) or not isinstance(self.maximum_processed_frames, int) or self.maximum_processed_frames <= 0:
            raise ValueError("maximum_processed_frames must be a positive integer")

        if isinstance(self.max_queued_jobs, bool) or not isinstance(self.max_queued_jobs, int) or not (1 <= self.max_queued_jobs <= 50):
            raise ValueError("max_queued_jobs must be an integer in [1, 50]")

        if isinstance(self.max_gpu_concurrency, bool) or int(self.max_gpu_concurrency) != 1:
            raise ValueError(
                f"max_gpu_concurrency must be 1 for bounded 4 GB VRAM safety, got {self.max_gpu_concurrency!r}"
            )
        self.max_gpu_concurrency = 1

        if isinstance(self.progress_update_interval, bool) or float(self.progress_update_interval) < 0.05:
            raise ValueError("progress_update_interval must be >= 0.05 seconds")
        self.progress_update_interval = float(self.progress_update_interval)

        if isinstance(self.worker_shutdown_timeout, bool) or float(self.worker_shutdown_timeout) <= 0.0:
            raise ValueError("worker_shutdown_timeout must be > 0 seconds")
        self.worker_shutdown_timeout = float(self.worker_shutdown_timeout)

        if not isinstance(self.artifact_retention_policy, str) or not self.artifact_retention_policy.strip():
            raise ValueError("artifact_retention_policy must be a non-empty string")

        if not isinstance(self.allowed_cors_origins, list):
            raise ValueError("allowed_cors_origins must be a list of origin strings")
        for origin in self.allowed_cors_origins:
            if not isinstance(origin, str) or not origin.strip() or origin.strip() == "*":
                raise ValueError(
                    f"Wildcard or invalid CORS origin not permitted for V2 local API: {origin!r}"
                )

        if not isinstance(self.default_page_size, int) or not isinstance(self.max_page_size, int):
            raise ValueError("default_page_size and max_page_size must be integers")
        if self.default_page_size <= 0 or self.max_page_size < self.default_page_size:
            raise ValueError("Require 0 < default_page_size <= max_page_size")

    def to_safe_dict(self) -> Dict[str, Any]:
        """Return non-sensitive config summary without exposing absolute filesystem paths."""
        d = asdict(self)
        d.pop("project_root", None)
        d["database_path"] = os.path.basename(self.database_path)
        d["upload_directory"] = os.path.basename(self.upload_directory)
        d["artifact_directory"] = os.path.basename(self.artifact_directory)
        return d


def load_api_config(
    config_path: Optional[str] = None,
    ensure_dirs: bool = True,
    apply_env_overrides: bool = True,
) -> APIConfig:
    """Load, override (env), validate, and return APIConfig."""
    cfg_file = config_path or DEFAULT_CONFIG_PATH
    data: Dict[str, Any] = {}
    if os.path.exists(cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ValueError(f"API config file {cfg_file} must parse to a YAML mapping")
            data = loaded

    valid_keys = set(APIConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in data.items() if k in valid_keys}

    # Support legacy alias keys gracefully if present in custom test configs
    if "upload_dir" in data and "upload_directory" not in filtered:
        filtered["upload_directory"] = data["upload_dir"]
    if "artifact_dir" in data and "artifact_directory" not in filtered:
        filtered["artifact_directory"] = data["artifact_dir"]
    if "max_upload_size_mb" in data and "maximum_upload_size" not in filtered:
        filtered["maximum_upload_size"] = int(data["max_upload_size_mb"]) * 1024 * 1024
    if "allowed_formats" in data and "allowed_video_formats" not in filtered:
        filtered["allowed_video_formats"] = data["allowed_formats"]
    if "cors_origins" in data and "allowed_cors_origins" not in filtered:
        filtered["allowed_cors_origins"] = data["cors_origins"]

    if apply_env_overrides:
        for env_key in ENV_ENABLED_KEYS:
            if env_key in os.environ:
                parsed = _parse_bool_env(os.environ.get(env_key))
                if parsed is not None:
                    filtered["enabled"] = parsed
                break

    cfg = APIConfig(**filtered)
    cfg.validate()
    if ensure_dirs:
        cfg.ensure_runtime_dirs()
    return cfg

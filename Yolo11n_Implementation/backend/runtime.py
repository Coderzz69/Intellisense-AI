"""Locates the AI package and builds the pipeline the dashboard streams from.

The dashboard runs the *same* AI stack as the standalone Flask UI: one canonical
``traffic/`` package holding ``traffic_pipeline`` (YOLO + tracking) and
``traffic_control`` (the trained DQN). Only the detector backend is swappable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from backend.settings import (
    AI_DIR_ENV_VAR,
    DEFAULT_VIDEO_ENV_VAR,
    INFERENCE_PROVIDER_ENV,
    PROJECT_ROOT,
    resolve_sagemaker_endpoint_name,
)


def discover_ai_dir() -> Path:
    """Find the directory that holds ``traffic_pipeline`` and ``traffic_control``."""
    candidates: list[Path] = []
    configured = os.environ.get(AI_DIR_ENV_VAR)
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(PROJECT_ROOT / "traffic")

    checked: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        checked.append(resolved)
        if (
            (resolved / "traffic_pipeline" / "__init__.py").exists()
            and (resolved / "traffic_control" / "__init__.py").exists()
            and (resolved / "configs" / "runtime.yaml").exists()
        ):
            return resolved

    raise FileNotFoundError(
        "Could not locate the traffic AI package (expected 'traffic_pipeline', "
        "'traffic_control' and 'configs/runtime.yaml' side by side). Looked in: "
        + ", ".join(str(path) for path in checked)
        + f". Set {AI_DIR_ENV_VAR} to override."
    )


def discover_default_video_path() -> Optional[Path]:
    configured_video = os.environ.get(DEFAULT_VIDEO_ENV_VAR)
    candidates: list[Path] = []
    if configured_video:
        candidates.append(Path(configured_video).expanduser())

    candidates.extend(sorted((PROJECT_ROOT / "samples").glob("*.mp4")))
    candidates.extend(sorted(PROJECT_ROOT.glob("*.mp4")))
    candidates.extend(sorted((Path.home() / "Downloads").glob("*.mp4")))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


AI_DIR = discover_ai_dir()
DEFAULT_CONFIG_PATH = AI_DIR / "configs" / "runtime.yaml"
DEFAULT_VIDEO_PATH = discover_default_video_path()
DEFAULT_PROCESS_FPS = 10.0
DEFAULT_EMIT_INTERVAL_SEC = 1.0
DEFAULT_REALTIME = True

# Kept under the old name so existing callers/scripts keep working.
BUNDLE_DIR = AI_DIR

if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from traffic_control import AITrafficPipeline  # noqa: E402
from traffic_pipeline import AppConfig, TrafficDetector, load_config  # noqa: E402


def get_inference_provider() -> str:
    """Which detector backend to use: ``local`` (bundled YOLO weights) or ``sagemaker``."""
    provider = os.environ.get(INFERENCE_PROVIDER_ENV, "").strip().lower()
    if provider:
        if provider not in {"local", "sagemaker"}:
            raise ValueError(
                f"Unsupported inference provider: {provider!r}. Use 'local' or 'sagemaker'."
            )
        return provider
    return "sagemaker" if resolve_sagemaker_endpoint_name() else "local"


def resolve_app_config(config_path: Path) -> AppConfig:
    """Load the pipeline config, forcing CPU when no CUDA device is present."""
    config = load_config(str(config_path))
    if str(config.detector.device).lower() not in {"cpu", ""}:
        try:
            import torch
        except ModuleNotFoundError:
            config.detector.device = "cpu"
        else:
            if not torch.cuda.is_available():
                config.detector.device = "cpu"
    return config


def build_detector(config: AppConfig) -> Optional[TrafficDetector]:
    """Return the configured detector, or ``None`` to let the pipeline build the local one."""
    if get_inference_provider() == "sagemaker":
        from backend.cloud.sagemaker import SageMakerDetector

        return SageMakerDetector(config.vehicle_keywords)
    return None


def load_processing_pipeline(config_path: Path) -> AITrafficPipeline:
    """Build the full frame -> YOLO -> tracking -> DQN pipeline."""
    config = resolve_app_config(config_path)
    return AITrafficPipeline(
        config_path,
        app_config=config,
        detector=build_detector(config),
    )

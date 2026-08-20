from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from backend.settings import (
    BUNDLE_ENV_VAR,
    DEFAULT_VIDEO_ENV_VAR,
    INFERENCE_PROVIDER_ENV,
    PROJECT_ROOT,
    resolve_sagemaker_endpoint_name,
)


def discover_bundle_dir() -> Path:
    candidates: list[Path] = []
    configured_bundle = os.environ.get(BUNDLE_ENV_VAR)
    if configured_bundle:
        candidates.append(Path(configured_bundle).expanduser())

    candidates.extend(sorted(PROJECT_ROOT.glob("traffic_main_files_bundle*"), reverse=True))
    candidates.append(PROJECT_ROOT / "traffic_model_bundle")
    candidates.append(Path.home() / "Downloads" / "traffic_model_bundle")

    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        resolved_key = str(resolved).lower()
        if resolved_key in seen:
            continue
        seen.add(resolved_key)

        if (
            (resolved / "traffic_pipeline" / "__init__.py").exists()
            and (resolved / "configs" / "runtime.yaml").exists()
        ):
            return resolved

    raise FileNotFoundError(
        "Could not locate a traffic model bundle. "
        f"Set {BUNDLE_ENV_VAR} or add a bundle folder under the project root."
    )


def discover_default_video_path() -> Optional[Path]:
    configured_video = os.environ.get(DEFAULT_VIDEO_ENV_VAR)
    candidates: list[Path] = []
    if configured_video:
        candidates.append(Path(configured_video).expanduser())

    candidates.append(Path.home() / "Downloads" / "5927708-hd_1080_1920_30fps.mp4")
    candidates.extend(sorted(PROJECT_ROOT.glob("*.mp4")))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


BUNDLE_DIR = discover_bundle_dir()
DEFAULT_CONFIG_PATH = BUNDLE_DIR / "configs" / "runtime.yaml"
DEFAULT_VIDEO_PATH = discover_default_video_path()
DEFAULT_PROCESS_FPS = 10.0
DEFAULT_EMIT_INTERVAL_SEC = 1.0
DEFAULT_REALTIME = True

if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from traffic_pipeline import TrafficPipeline, load_config  # noqa: E402


def load_local_pipeline(config_path: Path) -> TrafficPipeline:
    config = load_config(str(config_path))
    weights_path = Path(config.detector.weights_path)
    if not weights_path.is_absolute():
        config_relative = (config_path.parent / weights_path).resolve()
        bundle_relative = (BUNDLE_DIR / weights_path).resolve()
        resolved_weights_path = config_relative if config_relative.exists() else bundle_relative
        config.detector.weights_path = str(resolved_weights_path)
    if str(config.detector.device).lower() != "cpu":
        try:
            import torch
        except ModuleNotFoundError:
            pass
        else:
            if not torch.cuda.is_available():
                config.detector.device = "cpu"
    return TrafficPipeline(config)


def get_inference_provider() -> str:
    provider = os.environ.get(INFERENCE_PROVIDER_ENV, "local").strip().lower()
    if provider == "local" and resolve_sagemaker_endpoint_name():
        return "sagemaker"
    return provider or "local"


def load_processing_pipeline(config_path: Path):
    provider = get_inference_provider()
    if provider == "sagemaker":
        from backend.cloud.sagemaker import SageMakerTrafficPipeline

        return SageMakerTrafficPipeline(config_path)
    if provider != "local":
        raise ValueError(f"Unsupported inference provider: {provider}")
    return load_local_pipeline(config_path)

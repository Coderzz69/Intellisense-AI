from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_ENV_VAR = "TRAFFIC_MODEL_BUNDLE_DIR"
DEFAULT_VIDEO_ENV_VAR = "TRAFFIC_DEFAULT_VIDEO_PATH"
INFERENCE_PROVIDER_ENV = "TRAFFIC_INFERENCE_PROVIDER"
PRIMARY_SAGEMAKER_ENDPOINT_ENV = "SAGEMAKER_ENDPOINT_NAME"
LEGACY_SAGEMAKER_ENDPOINT_ENV = "TRAFFIC_SAGEMAKER_ENDPOINT_NAME"
SAGEMAKER_ENDPOINT_ENVS = (
    PRIMARY_SAGEMAKER_ENDPOINT_ENV,
    LEGACY_SAGEMAKER_ENDPOINT_ENV,
)
ULTRALYTICS_CONFIG_DIR = PROJECT_ROOT / ".ultralytics"


def ensure_runtime_paths() -> None:
    ULTRALYTICS_CONFIG_DIR.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def resolve_sagemaker_endpoint_name() -> str | None:
    for env_name in SAGEMAKER_ENDPOINT_ENVS:
        value = os.environ.get(env_name)
        if value and value.strip():
            return value.strip()
    return None


ensure_runtime_paths()

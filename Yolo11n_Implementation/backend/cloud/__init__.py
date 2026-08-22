from .storage import S3AnalyticsSink
from .traffic_nodes import build_congestion_summary, build_traffic_nodes

__all__ = [
    "S3AnalyticsSink",
    "SageMakerDetector",
    "build_congestion_summary",
    "build_traffic_nodes",
]


def __getattr__(name: str):
    """Defer the SageMaker import so boto3 stays optional for local runs."""
    if name == "SageMakerDetector":
        from .sagemaker import SageMakerDetector

        return SageMakerDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

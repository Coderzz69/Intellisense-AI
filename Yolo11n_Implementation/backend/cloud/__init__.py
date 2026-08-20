from .storage import S3AnalyticsSink
from .traffic_nodes import build_congestion_summary, build_traffic_nodes

__all__ = [
    "S3AnalyticsSink",
    "SageMakerTrafficPipeline",
    "build_congestion_summary",
    "build_traffic_nodes",
]


def __getattr__(name: str):
    if name == "SageMakerTrafficPipeline":
        from .sagemaker import SageMakerTrafficPipeline

        return SageMakerTrafficPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

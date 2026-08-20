from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

S3_BUCKET_ENV = "TRAFFIC_S3_BUCKET"
S3_PREFIX_ENV = "TRAFFIC_S3_PREFIX"
S3_FLUSH_INTERVAL_ENV = "TRAFFIC_S3_FLUSH_INTERVAL"
CAMERA_ID_ENV = "TRAFFIC_CAMERA_ID"
AWS_REGION_ENV = "AWS_REGION"


class S3AnalyticsSink:
    def __init__(self, bucket: str, prefix: str, camera_id: str, flush_interval: int = 10) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "boto3 is required for S3 analytics uploads. Install backend cloud dependencies first."
            ) from exc

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.camera_id = camera_id
        self.flush_interval = max(1, flush_interval)
        self.buffer: List[Dict[str, object]] = []
        self.session_id = uuid4().hex
        self.chunk_index = 0
        self.s3 = boto3.client(
            "s3",
            region_name=os.environ.get(AWS_REGION_ENV),
            config=Config(
                retries={"max_attempts": 5, "mode": "standard"},
            ),
        )

    @classmethod
    def from_env(cls) -> Optional["S3AnalyticsSink"]:
        bucket = os.environ.get(S3_BUCKET_ENV)
        if not bucket:
            return None

        prefix = os.environ.get(S3_PREFIX_ENV, "traffic-analytics/events")
        camera_id = os.environ.get(CAMERA_ID_ENV, "camera-001")
        flush_interval = int(os.environ.get(S3_FLUSH_INTERVAL_ENV, "10"))
        return cls(bucket=bucket, prefix=prefix, camera_id=camera_id, flush_interval=flush_interval)

    def record(self, payload: Dict[str, object]) -> None:
        self.buffer.append(payload)
        if len(self.buffer) >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return

        now = datetime.now(timezone.utc)
        key = (
            f"{self.prefix}/camera_id={self.camera_id}/date={now:%Y-%m-%d}/hour={now:%H}/"
            f"session={self.session_id}/chunk-{self.chunk_index:05d}.jsonl"
        )
        body = "\n".join(json.dumps(item, separators=(",", ":")) for item in self.buffer).encode("utf-8")
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/x-ndjson",
        )
        self.buffer.clear()
        self.chunk_index += 1

    def close(self) -> None:
        self.flush()

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

import boto3
from botocore.exceptions import ClientError

DEFAULT_DEPLOYMENT_DIR = REPO_ROOT / "deployment"
DEFAULT_INSTANCE_TYPE = "ml.g4dn.xlarge"
DEFAULT_FRAMEWORK_VERSION = "2.2.0"
DEFAULT_PY_VERSION = "py310"
DEFAULT_VARIANT_NAME = "AllTraffic"
CODE_ONLY_FILENAMES = {"inference.py", "requirements.txt"}
CODE_EXCLUDE_FILENAMES = {"README.md", "model.tar.gz"}
MODEL_EXCLUDE_FILENAMES = {"inference.py", "requirements.txt", "README.md", "model.tar.gz"}


def _iter_bundle_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix == ".pyc":
            continue
        yield path


def _normalize_s3_key(prefix: str, filename: str) -> str:
    normalized_prefix = prefix.strip("/")
    return f"{normalized_prefix}/{filename}" if normalized_prefix else filename


def _resolve_deployment_dir(raw_path: str | None) -> Path:
    deployment_dir = Path(raw_path).expanduser().resolve() if raw_path else DEFAULT_DEPLOYMENT_DIR.resolve()
    if not deployment_dir.exists():
        raise FileNotFoundError(f"Deployment directory not found: {deployment_dir}")
    if not deployment_dir.is_dir():
        raise NotADirectoryError(f"Deployment path must be a directory: {deployment_dir}")
    for required_name in CODE_ONLY_FILENAMES:
        required_path = deployment_dir / required_name
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required deployment file: {required_path}")
    return deployment_dir


def _resolve_model_path(deployment_dir: Path, raw_model_path: str | None) -> Path:
    if raw_model_path:
        model_path = Path(raw_model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return model_path

    for candidate_name in ("model.pt", "best.pt"):
        candidate = deployment_dir / candidate_name
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find model weights in the deployment directory. "
        "Expected deployment/model.pt or pass --model-path."
    )


def _create_model_artifact(deployment_dir: Path, model_path: Path, temp_dir: Path) -> Path:
    model_tar = temp_dir / "model.tar.gz"
    with tarfile.open(model_tar, "w:gz") as archive:
        archive.add(model_path, arcname="model.pt")
        for path in _iter_bundle_files(deployment_dir):
            relative_path = path.relative_to(deployment_dir)
            if relative_path.name in MODEL_EXCLUDE_FILENAMES:
                continue
            if path.resolve() == model_path.resolve():
                continue
            archive.add(path, arcname=relative_path.as_posix())
    return model_tar


def _create_code_bundle(deployment_dir: Path, temp_dir: Path) -> Path:
    code_dir = temp_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    for path in _iter_bundle_files(deployment_dir):
        relative_path = path.relative_to(deployment_dir)
        if relative_path.name in CODE_EXCLUDE_FILENAMES:
            continue
        if path.suffix == ".pt":
            continue
        destination = code_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    inference_path = code_dir / "inference.py"
    requirements_path = code_dir / "requirements.txt"
    if not inference_path.exists():
        raise FileNotFoundError(f"Missing inference entry point in code bundle: {inference_path}")
    if not requirements_path.exists():
        raise FileNotFoundError(f"Missing requirements file in code bundle: {requirements_path}")
    return code_dir


def _upload_model_artifact(model_tar: Path, bucket: str, prefix: str, boto_session: boto3.Session) -> str:
    s3_client = boto_session.client("s3")
    s3_key = _normalize_s3_key(prefix, model_tar.name)
    s3_client.upload_file(str(model_tar), bucket, s3_key)
    return f"s3://{bucket}/{s3_key}"


def _load_sagemaker_sdk() -> tuple[type, type]:
    shadow_paths = {
        str(REPO_ROOT.resolve()).lower(),
        str((REPO_ROOT / "sagemaker").resolve()).lower(),
    }
    original_sys_path = list(sys.path)
    try:
        filtered_sys_path: list[str] = []
        for entry in original_sys_path:
            resolved_entry = Path(entry or ".").resolve()
            if str(resolved_entry).lower() in shadow_paths:
                continue
            filtered_sys_path.append(entry)
        sys.path = filtered_sys_path

        pytorch_module = importlib.import_module("sagemaker.pytorch")
        session_module = importlib.import_module("sagemaker.session")
        return pytorch_module.PyTorchModel, session_module.Session
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "AWS SageMaker SDK is not installed in this interpreter. "
            "Install it with `py -3.12 -m pip install -r sagemaker\\requirements-deploy.txt` "
            "and run the script with `py -3.12 sagemaker\\deploy.py ...`."
        ) from exc
    finally:
        sys.path = original_sys_path


def _endpoint_exists(endpoint_name: str, boto_session: boto3.Session) -> bool:
    sagemaker_client = boto_session.client("sagemaker")
    try:
        sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code == "ValidationException":
            return False
        raise


def _configure_autoscaling(
    endpoint_name: str,
    variant_name: str,
    min_capacity: int,
    max_capacity: int,
    boto_session: boto3.Session,
) -> None:
    application_autoscaling = boto_session.client("application-autoscaling")
    resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"
    application_autoscaling.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=min_capacity,
        MaxCapacity=max_capacity,
    )
    application_autoscaling.put_scaling_policy(
        PolicyName=f"{endpoint_name}-invocations-scaling",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 70.0,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
            },
            "ScaleInCooldown": 300,
            "ScaleOutCooldown": 60,
        },
    )


def _validate_capacity_args(args: argparse.Namespace) -> None:
    if args.initial_instance_count < 1:
        raise ValueError("--initial-instance-count must be at least 1.")
    if args.min_capacity < 1:
        raise ValueError("--min-capacity must be at least 1.")
    if args.max_capacity < args.min_capacity:
        raise ValueError("--max-capacity must be greater than or equal to --min-capacity.")
    if not args.disable_autoscaling and not (args.min_capacity <= args.initial_instance_count <= args.max_capacity):
        raise ValueError(
            "--initial-instance-count must be within the autoscaling range when autoscaling is enabled."
        )


def deploy(args: argparse.Namespace) -> None:
    _validate_capacity_args(args)

    deployment_dir = _resolve_deployment_dir(args.deployment_dir)
    model_path = _resolve_model_path(deployment_dir, args.model_path)
    boto_session = boto3.Session(region_name=args.region)
    PyTorchModel, Session = _load_sagemaker_sdk()

    if _endpoint_exists(args.endpoint_name, boto_session) and not args.update_endpoint:
        raise ValueError(
            f"Endpoint '{args.endpoint_name}' already exists. "
            "Use --update-endpoint to update it in place."
        )

    sagemaker_session = Session(boto_session=boto_session)

    with tempfile.TemporaryDirectory(prefix="traffic-sagemaker-") as temp_path:
        temp_dir = Path(temp_path)
        model_artifact = _create_model_artifact(deployment_dir, model_path, temp_dir)
        code_dir = _create_code_bundle(deployment_dir, temp_dir)
        model_data = _upload_model_artifact(model_artifact, args.bucket, args.prefix, boto_session)

        print(f"Using deployment directory: {deployment_dir}")
        print(f"Using model file: {model_path}")
        print(f"Uploaded model artifact: {model_data}")

        model = PyTorchModel(
            model_data=model_data,
            role=args.role_arn,
            entry_point="inference.py",
            source_dir=str(code_dir),
            framework_version=args.framework_version,
            py_version=args.py_version,
            sagemaker_session=sagemaker_session,
            env={
                "YOLO_CONFIDENCE": str(args.confidence),
                "YOLO_IOU": str(args.iou),
                "YOLO_IMGSZ": str(args.imgsz),
                "YOLO_DEVICE": args.device,
            },
        )

        predictor = model.deploy(
            initial_instance_count=args.initial_instance_count,
            instance_type=args.instance_type,
            endpoint_name=args.endpoint_name,
            update_endpoint=args.update_endpoint,
            serializer=None,
            deserializer=None,
        )

    if not args.disable_autoscaling:
        _configure_autoscaling(
            endpoint_name=args.endpoint_name,
            variant_name=args.variant_name,
            min_capacity=args.min_capacity,
            max_capacity=args.max_capacity,
            boto_session=boto_session,
        )
        print(
            "Configured autoscaling "
            f"for variant '{args.variant_name}' with range {args.min_capacity}-{args.max_capacity}."
        )

    print(f"Endpoint deployed: {predictor.endpoint_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy the YOLO detector bundle to a SageMaker real-time endpoint."
    )
    parser.add_argument(
        "--deployment-dir",
        default=str(DEFAULT_DEPLOYMENT_DIR),
        help="Directory containing inference.py, requirements.txt, and the model assets.",
    )
    parser.add_argument(
        "--model-path",
        "--weights-path",
        dest="model_path",
        default=None,
        help="Optional override for the YOLO .pt file. Defaults to deployment/model.pt.",
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket used for the uploaded model artifact.")
    parser.add_argument(
        "--prefix",
        default="traffic-sagemaker/model",
        help="S3 key prefix for the uploaded model artifact.",
    )
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN.")
    parser.add_argument("--region", required=True, help="AWS region for the endpoint.")
    parser.add_argument("--endpoint-name", required=True, help="SageMaker endpoint name.")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE, help="Endpoint instance type.")
    parser.add_argument(
        "--initial-instance-count",
        type=int,
        default=1,
        help="Initial number of endpoint instances.",
    )
    parser.add_argument(
        "--variant-name",
        default=DEFAULT_VARIANT_NAME,
        help="Production variant name used for autoscaling configuration.",
    )
    parser.add_argument("--min-capacity", type=int, default=1, help="Minimum autoscaling capacity.")
    parser.add_argument("--max-capacity", type=int, default=4, help="Maximum autoscaling capacity.")
    parser.add_argument(
        "--disable-autoscaling",
        action="store_true",
        help="Skip Application Auto Scaling configuration after deploy.",
    )
    parser.add_argument(
        "--update-endpoint",
        action="store_true",
        help="Update an existing endpoint instead of failing when the endpoint name already exists.",
    )
    parser.add_argument(
        "--framework-version",
        default=DEFAULT_FRAMEWORK_VERSION,
        help="PyTorch inference container framework version.",
    )
    parser.add_argument(
        "--py-version",
        default=DEFAULT_PY_VERSION,
        help="Python version for the PyTorch inference container.",
    )
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", default="cuda", help="YOLO inference device inside the endpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    deploy(parse_args())

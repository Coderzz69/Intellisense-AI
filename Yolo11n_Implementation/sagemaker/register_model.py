from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DEPLOYMENT_DIR = REPO_ROOT / "deployment"
DEFAULT_FRAMEWORK_VERSION = "2.2.0"
DEFAULT_PY_VERSION = "py310"
DEFAULT_INFERENCE_INSTANCE = "ml.g4dn.xlarge"
DEFAULT_TRANSFORM_INSTANCE = "ml.g4dn.xlarge"
DEFAULT_APPROVAL_STATUS = "PendingManualApproval"
DEFAULT_PREFIX = "traffic-sagemaker/model-registry"
DEFAULT_DOMAIN = "COMPUTER_VISION"
DEFAULT_TASK = "OBJECT_DETECTION"
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


def _slugify(value: str) -> str:
    slug = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
    slug = slug.strip("-_")
    return slug or "model"


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


def _create_source_archive(deployment_dir: Path, temp_dir: Path) -> Path:
    source_root = temp_dir / "source_dir"
    source_root.mkdir(parents=True, exist_ok=True)

    for path in _iter_bundle_files(deployment_dir):
        relative_path = path.relative_to(deployment_dir)
        if relative_path.name in CODE_EXCLUDE_FILENAMES:
            continue
        if path.suffix == ".pt":
            continue
        destination = source_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    inference_path = source_root / "inference.py"
    requirements_path = source_root / "requirements.txt"
    if not inference_path.exists():
        raise FileNotFoundError(f"Missing inference entry point in source archive: {inference_path}")
    if not requirements_path.exists():
        raise FileNotFoundError(f"Missing requirements file in source archive: {requirements_path}")

    source_tar = temp_dir / "sourcedir.tar.gz"
    with tarfile.open(source_tar, "w:gz") as archive:
        for path in _iter_bundle_files(source_root):
            archive.add(path, arcname=path.relative_to(source_root).as_posix())
    return source_tar


def _upload_file(local_path: Path, bucket: str, prefix: str, boto_session: boto3.Session) -> str:
    s3_client = boto_session.client("s3")
    s3_key = _normalize_s3_key(prefix, local_path.name)
    s3_client.upload_file(str(local_path), bucket, s3_key)
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
            "and run the script with `py -3.12 sagemaker\\register_model.py ...`."
        ) from exc
    finally:
        sys.path = original_sys_path


def _model_package_group_exists(model_package_group_name: str, boto_session: boto3.Session) -> bool:
    sagemaker_client = boto_session.client("sagemaker")
    try:
        sagemaker_client.describe_model_package_group(ModelPackageGroupName=model_package_group_name)
        return True
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code == "ValidationException":
            return False
        raise


def _ensure_model_package_group(
    model_package_group_name: str,
    description: str | None,
    boto_session: boto3.Session,
) -> bool:
    if _model_package_group_exists(model_package_group_name, boto_session):
        return False

    sagemaker_client = boto_session.client("sagemaker")
    payload = {"ModelPackageGroupName": model_package_group_name}
    if description:
        payload["ModelPackageGroupDescription"] = description
    sagemaker_client.create_model_package_group(**payload)
    return True


def _parse_metadata_pairs(raw_pairs: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw_pair in raw_pairs:
        if "=" not in raw_pair:
            raise ValueError(
                f"Invalid metadata entry '{raw_pair}'. Use the form key=value."
            )
        key, value = raw_pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(
                f"Invalid metadata entry '{raw_pair}'. Both key and value are required."
            )
        metadata[key] = value
    return metadata


def register_model(args: argparse.Namespace) -> None:
    deployment_dir = _resolve_deployment_dir(args.deployment_dir)
    model_path = _resolve_model_path(deployment_dir, args.model_path)
    metadata = _parse_metadata_pairs(args.metadata)
    boto_session = boto3.Session(region_name=args.region)
    PyTorchModel, Session = _load_sagemaker_sdk()

    created_group = _ensure_model_package_group(
        model_package_group_name=args.model_package_group_name,
        description=args.model_package_group_description,
        boto_session=boto_session,
    )

    sagemaker_session = Session(boto_session=boto_session)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    upload_prefix = _normalize_s3_key(
        args.prefix,
        f"{_slugify(args.model_package_group_name)}/{timestamp}",
    )

    with tempfile.TemporaryDirectory(prefix="traffic-sagemaker-register-") as temp_path:
        temp_dir = Path(temp_path)
        model_artifact = _create_model_artifact(deployment_dir, model_path, temp_dir)
        source_archive = _create_source_archive(deployment_dir, temp_dir)
        model_data_url = _upload_file(model_artifact, args.bucket, upload_prefix, boto_session)
        source_dir_url = _upload_file(source_archive, args.bucket, upload_prefix, boto_session)

        print(f"Using deployment directory: {deployment_dir}")
        print(f"Using model file: {model_path}")
        print(f"Uploaded model artifact: {model_data_url}")
        print(f"Uploaded source archive: {source_dir_url}")

        model = PyTorchModel(
            model_data=model_data_url,
            role=args.role_arn,
            entry_point="inference.py",
            source_dir=source_dir_url,
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
        container_definition = model.prepare_container_def(
            instance_type=args.inference_instance_types[0]
        )
        container_definition.setdefault("Framework", "PYTORCH")
        container_definition.setdefault("FrameworkVersion", args.framework_version)

    inference_specification = {
        "Containers": [container_definition],
        "SupportedContentTypes": args.content_types,
        "SupportedResponseMIMETypes": args.response_types,
        "SupportedRealtimeInferenceInstanceTypes": args.inference_instance_types,
        "SupportedTransformInstanceTypes": args.transform_instance_types,
    }

    create_payload = {
        "ModelPackageGroupName": args.model_package_group_name,
        "ModelApprovalStatus": args.approval_status,
        "ModelPackageDescription": args.model_package_description,
        "InferenceSpecification": inference_specification,
        "Domain": args.domain,
        "Task": args.task,
    }
    if metadata:
        create_payload["CustomerMetadataProperties"] = metadata

    sagemaker_client = boto_session.client("sagemaker")
    response = sagemaker_client.create_model_package(**create_payload)

    if created_group:
        print(f"Created model package group: {args.model_package_group_name}")
    print(f"Registered model package ARN: {response['ModelPackageArn']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register the YOLO detector bundle in SageMaker Model Registry."
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
    parser.add_argument("--bucket", required=True, help="S3 bucket used for uploaded model and code bundles.")
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="S3 key prefix for the uploaded model and source archives.",
    )
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN.")
    parser.add_argument("--region", required=True, help="AWS region for the model package.")
    parser.add_argument(
        "--model-package-group-name",
        required=True,
        help="SageMaker Model Package Group name for versioned registration.",
    )
    parser.add_argument(
        "--model-package-group-description",
        default="Traffic YOLO detector package group.",
        help="Description to use if the model package group must be created.",
    )
    parser.add_argument(
        "--model-package-description",
        default="YOLO traffic detector registered from the deployment bundle.",
        help="Description for the registered model package version.",
    )
    parser.add_argument(
        "--approval-status",
        choices=["Approved", "Rejected", "PendingManualApproval"],
        default=DEFAULT_APPROVAL_STATUS,
        help="Initial model approval status.",
    )
    parser.add_argument(
        "--content-types",
        nargs="+",
        default=["application/x-image"],
        help="Supported request content types for inference.",
    )
    parser.add_argument(
        "--response-types",
        nargs="+",
        default=["application/json"],
        help="Supported response MIME types for inference.",
    )
    parser.add_argument(
        "--inference-instance-types",
        nargs="+",
        default=[DEFAULT_INFERENCE_INSTANCE],
        help="Supported real-time inference instance types.",
    )
    parser.add_argument(
        "--transform-instance-types",
        nargs="+",
        default=[DEFAULT_TRANSFORM_INSTANCE],
        help="Supported batch transform instance types.",
    )
    parser.add_argument(
        "--domain",
        default=DEFAULT_DOMAIN,
        help="Optional SageMaker model domain metadata.",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Optional SageMaker model task metadata.",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional custom metadata pair. Repeat for multiple entries.",
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
    parser.add_argument("--device", default="cuda", help="YOLO inference device inside the container.")
    return parser.parse_args()


if __name__ == "__main__":
    register_model(parse_args())

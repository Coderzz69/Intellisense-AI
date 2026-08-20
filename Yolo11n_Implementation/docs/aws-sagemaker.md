# AWS SageMaker Upgrade

## Suggested Folder Structure

```text
backend/
  cloud/
    __init__.py
    sagemaker.py
    storage.py
    traffic_nodes.py
  requirements-cloud.txt
  server.py
sagemaker/
  analytics_job.py
  deploy.py
  inference.py
  requirements.txt
src/
  components/
    TrafficMap.js
  utils/
    trafficNodes.js
```

## Runtime Flow

1. `backend/server.py` captures video frames from the current source.
2. If `TRAFFIC_INFERENCE_PROVIDER=sagemaker`, the backend sends JPEG-encoded frames to the SageMaker endpoint.
3. The SageMaker endpoint returns raw YOLO detections.
4. The backend reuses the existing tracker + signal-control logic to compute:
   - vehicle list
   - wait time
   - halting state
   - emergency state
   - lane metrics
   - traffic nodes for the map
5. The backend streams the structured payload to the React frontend over SSE.
6. The backend optionally writes JSONL analytics chunks to S3 for batch analysis.

## Env Vars To Configure

### Backend / AWS

- `TRAFFIC_INFERENCE_PROVIDER`
  - `local` or `sagemaker`
- `TRAFFIC_SAGEMAKER_ENDPOINT_NAME`
  - Real-time SageMaker endpoint name
- `AWS_REGION`
  - AWS region for SageMaker Runtime and S3
- `AWS_ACCESS_KEY_ID`
  - IAM access key for local execution if not using an attached role/profile
- `AWS_SECRET_ACCESS_KEY`
  - IAM secret key
- `AWS_SESSION_TOKEN`
  - Optional session token
- `TRAFFIC_S3_BUCKET`
  - S3 bucket for analytics events
- `TRAFFIC_S3_PREFIX`
  - S3 prefix, e.g. `traffic-analytics/events`
- `TRAFFIC_CAMERA_ID`
  - Stable camera/source identifier
- `TRAFFIC_NODE_CENTER_LAT`
  - Base latitude used for map nodes
- `TRAFFIC_NODE_CENTER_LNG`
  - Base longitude used for map nodes
- `TRAFFIC_S3_FLUSH_INTERVAL`
  - Number of events per uploaded JSONL chunk
- `TRAFFIC_SAGEMAKER_JPEG_QUALITY`
  - JPEG quality when sending frames to the endpoint
- `TRAFFIC_SAGEMAKER_CONNECT_TIMEOUT_SEC`
  - Runtime client connect timeout
- `TRAFFIC_SAGEMAKER_READ_TIMEOUT_SEC`
  - Runtime client read timeout
- `TRAFFIC_SAGEMAKER_MAX_RETRIES`
  - Runtime invoke retries

### Frontend

- `REACT_APP_API_BASE`
  - Backend base URL
- `REACT_APP_MAPBOX_ACCESS_TOKEN`
  - Mapbox token

## Deploy The YOLO Endpoint

Install the deployment-side SDK first:

```powershell
py -3.12 -m pip install -r sagemaker\requirements-deploy.txt
```

```powershell
py -3.12 sagemaker\deploy.py `
  --deployment-dir deployment `
  --bucket <your-s3-bucket> `
  --prefix traffic-sagemaker/model `
  --role-arn <your-sagemaker-role-arn> `
  --region <your-aws-region> `
  --endpoint-name traffic-yolo-realtime
```

Notes:

- The deployment script now treats [deployment](/C:/Users/seera/Desktop/Exploration/vnr/intelligent_Traffic_management/deployment) as the source of truth.
- It uploads `deployment/model.pt` as the SageMaker model artifact.
- It uses `deployment/inference.py` and `deployment/requirements.txt` as the endpoint code package.
- If your weights live somewhere else, pass `--model-path <path-to-model.pt>`.
- On this machine, use Python 3.12 for deployment because the repo `.venv` does not have `pip` and the currently active `python` resolves to Python 3.13.

Recommended endpoint instance:

- `ml.g4dn.xlarge`

## Register The Model In SageMaker Model Registry

Use the registry script when you want a versioned model package in a SageMaker Model Package Group instead of deploying an endpoint directly.

```powershell
py -3.12 sagemaker\register_model.py `
  --deployment-dir deployment `
  --bucket <your-s3-bucket> `
  --prefix traffic-sagemaker/model-registry `
  --role-arn <your-sagemaker-role-arn> `
  --region <your-aws-region> `
  --model-package-group-name traffic-yolo
```

Notes:

- The script packages `deployment/model.pt` into `model.tar.gz`.
- It packages `deployment/inference.py` and `deployment/requirements.txt` into `sourcedir.tar.gz`.
- Both archives are uploaded to S3 under a timestamped prefix before registration.
- If the model package group does not exist yet, the script creates it automatically.
- Override the weights path with `--model-path <path-to-model.pt>` when needed.
- Add custom registry metadata with repeated `--metadata key=value` flags.

## Backend Cloud Dependencies

Install the runtime AWS client for the backend before enabling SageMaker inference:

```powershell
python -m pip install -r backend\requirements-cloud.txt
```

## Analytics Job

The `sagemaker/analytics_job.py` script aggregates the JSONL records uploaded to S3 and produces hourly summaries for:

- average vehicle count
- congestion score
- peak traffic windows

This script is suitable for a SageMaker Processing or scheduled batch job.

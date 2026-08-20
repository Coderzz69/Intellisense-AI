# Deployment Artifact

This folder is the minimal SageMaker-style model artifact layout for the YOLO detector.

```text
deployment/
├── model.pt
├── inference.py
├── requirements.txt
└── README.md
```

## Files

- `model.pt`
  - your trained YOLO weights
- `inference.py`
  - SageMaker inference entry point with `model_fn`, `input_fn`, `predict_fn`, `output_fn`
- `requirements.txt`
  - Python dependencies for the inference container

## Packaging

To bundle this folder into a model artifact:

```powershell
tar -czf model.tar.gz -C deployment .
```

That `model.tar.gz` can then be uploaded to S3 and used by the SageMaker deployment flow.

## Deploy With The Repo Script

From the repo root:

```powershell
py -3.12 sagemaker\deploy.py `
  --deployment-dir deployment `
  --bucket <your-s3-bucket> `
  --prefix traffic-sagemaker/model `
  --role-arn <your-sagemaker-role-arn> `
  --region <your-aws-region> `
  --endpoint-name traffic-yolo-realtime
```

The deploy script packages `model.pt` into the SageMaker model artifact and uses
`inference.py` plus `requirements.txt` as the endpoint code bundle.

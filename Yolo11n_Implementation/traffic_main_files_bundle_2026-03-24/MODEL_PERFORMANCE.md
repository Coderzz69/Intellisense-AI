# Model Performance Summary

## Model
- Detector: Ultralytics YOLO11n (fine-tuned from `yolo11n.pt`)
- Classes (6): ambulance, auto-rikshaw, bike, bus, car, truck

## Training Data
- Dataset config: `configs/all_sources_no_police_dataset.yaml`
- Data path: `/home/anuj/Desktop/image-processing/traffic/prepared_data/all_sources_no_police`
- Train/val split: `train/images`, `valid/images`

## Training Run
- Run name: `ambulance-priority-all-sources-no-police-960-70`
- Image size: 960
- Batch size: 4
- Epochs: 70
- Device: GPU 0

## Validation Metrics
Final epoch (70):
- Precision: **0.76275**
- Recall: **0.63088**
- mAP50: **0.70830**
- mAP50-95: **0.42651**

Best mAP50-95:
- **0.42795** at epoch **56**  

Notes:
- `best.pt` is selected by Ultralytics based on validation fitness (driven by mAP50-95), so it should align with the best epoch above.
- Full history is stored in `runs/detect/runs/train/ambulance-priority-all-sources-no-police-960-70/results.csv`.

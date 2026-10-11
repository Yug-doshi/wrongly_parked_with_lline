# YOLOv8 Smart Parking Detection System

Detects **free parking slots (GREEN)**, **correctly parked cars (BLUE)**, and **incorrectly parked cars (RED)** using YOLOv8 detections plus configurable vehicle–slot **overlap-ratio** analysis.

## Dataset findings (do not skip)

Source: Roboflow **PKLot v2 640** at  
`D:\RMS 4\Parking_dataset (1)\Parking_dataset`

| Item | Value |
|---|---|
| Images | 12,416 JPG @ 640×640 |
| Splits | train 8691 / valid 2483 / test 1242 |
| Format | YOLOv8 `.txt` normalized boxes |
| Class `0` | `space-empty` — free valid slot |
| Class `1` | `space-occupied` — occupied slot (**not** “correct parking”) |
| Boxes | ~57 per image; empty≈376k, occupied≈336k |
| Vehicles / wrong-parking labels | **None** |
| Polygons | **None** (axis-aligned boxes approximate original rotated PKLot slots) |

**What this dataset can train directly:** free vs occupied parking slots.

**What it cannot train alone:** a true vehicle detector or a wrong-parking class. Vehicles are handled by (1) remapping occupied slots → `car` for approximate training, and/or (2) COCO-pretrained vehicle classes. Wrong parking is decided only by geometry.

## Architecture

1. **Slot detector (Approach A)** — YOLOv8 on `space-empty` / `space-occupied`.
2. **Vehicle detector (Approach B)** — YOLOv8 `car` model from remapped occupied boxes, or COCO vehicles.
3. **Overlap classifier (Approach C)** — IoU, \(R_{vehicle}\), \(R_{slot}\), \(R_{outside}\), multi-slot intrusion → FREE / CORRECT / WRONG / UNCERTAIN.

## Setup (Windows PowerShell)

```powershell
cd "D:\RMS 4\parking-detection"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Step-by-step commands

### 1) Inspect dataset

```powershell
python src\data_inspection.py --dataset "D:\RMS 4\Parking_dataset (1)\Parking_dataset" --out reports
```

### 2) Prepare training subsets (original files untouched)

```powershell
# CPU-friendly subsample (800 train images). Use --max-train 0 for full data.
python src\prepare_data.py --src "D:\RMS 4\Parking_dataset (1)\Parking_dataset" --max-train 800 --seed 42
```

### 3) Train slot detector

```powershell
python src\train.py --data dataset\space\data.yaml --model yolov8n.pt --name slot_det --epochs 20 --batch 4 --device cpu
```

### 4) Train approximate car detector

```powershell
python src\train.py --data dataset\car\data.yaml --model yolov8n.pt --name car_det --epochs 20 --batch 4 --device cpu
```

> This machine reports **CUDA unavailable**. **Train on Kaggle GPU for 50 epochs** — see `KAGGLE_TRAINING.md` and `notebooks/ParkingDetection_Kaggle.ipynb`.

### 5) Validate detection (mAP)

```powershell
python src\evaluation.py --weights weights\slot_det_best.pt --data dataset\space\data.yaml --split test --device cpu
```

### 6) Tune overlap thresholds (validation only)

```powershell
python src\tune_thresholds.py --max-images 40
```

### 7) Inference on images

```powershell
# With trained weights:
python src\inference.py --source dataset\space\test\images --out outputs\inference --slot-weights weights\slot_det_best.pt --vehicle-weights weights\car_det_best.pt

# Geometry demo using ground-truth labels (no weights required):
python src\inference.py --source "D:\RMS 4\Parking_dataset (1)\Parking_dataset\test\images" --labels-dir "D:\RMS 4\Parking_dataset (1)\Parking_dataset\test\labels" --use-gt-slots --use-gt-vehicles --out outputs\inference\gt_demo
```

### 8) Video / webcam

```powershell
python src\video_processor.py --source path\to\video.mp4 --out outputs\inference\video_out.mp4 --slot-weights weights\slot_det_best.pt --vehicle-weights weights\car_det_best.pt
python src\video_processor.py --source 0 --display --max-frames 100
```

### 9) Unit tests

```powershell
pytest tests -q
```

## Color legend

| Status | Color | Label |
|---|---|---|
| Free parking slot | GREEN | `FREE SLOT \| Slot-01` |
| Correctly parked car | BLUE | `CORRECT PARKING \| Slot-01 \| Inside: 96%` |
| Incorrectly parked car | RED | `WRONG PARKING \| Outside: 18%` |

## Thresholds

Editable in `configs/thresholds.yaml`:

- `vehicle_inside_min: 0.85` (tuned on validation)
- `outside_max: 0.10`
- `slot_assign_min: 0.30`
- `minimum_second_slot_intrusion_ratio: 0.15` measures the share of a vehicle footprint extending into a second slot beyond the tolerance band; it does not depend on the second slot's area.
- `boundary_crossing_tolerance: 0.05` ignores a narrow boundary band equal to 5% of the vehicle box's shorter dimension.
- `vehicle_detection_conf_min: 0.10` keeps lower-confidence boxes visible as uncertain; `vehicle_conf_min` still controls whether they can be classified.

When significant vehicle area penetrates a second slot beyond that band, the vehicle is labeled `WRONG PARKING | Straddles Slot-A / Slot-B`, even if it remains inside the combined outer boundary. The tolerance and intrusion ratio are initial configurable values, not camera-calibrated thresholds; the available PKLot annotations do not label actual vehicle footprints or parking violations, so calibration requires manually verified examples from the target camera.

The slot detector's `space-occupied` result is used as an occupancy signal. If it says a slot is occupied but no vehicle can be confidently matched, the slot is shown as **occupied, unverified** rather than incorrectly counted as free or as correctly parked. Wrong-parking accuracy still requires camera-specific vehicle detections and labeled wrong-parking examples; the PKLot data does not provide those labels.

### Local CPU demo weights (5 epochs, 400-image subset)

| Model | Weights | Test/Val mAP@0.50 |
|---|---|---|
| Slot (`space-empty` / `space-occupied`) | `weights/slot_det_best.pt` | **0.984** (test) |
| Car (occupied remapped) | `weights/car_det_best.pt` | **0.969** (val) |

Replace these with Kaggle **50-epoch** `yolov8s` weights when available (`KAGGLE_TRAINING.md`).

## Honesty / limitations

- Learned detections ≠ geometric decisions ≠ evaluation metrics.
- Occupied slot ≠ correctly parked vehicle.
- Bounding-box overlap can disagree with the true car footprint under perspective.
- Wrong-parking precision/recall needs a manually labeled evaluation set (not present in PKLot Roboflow export).
- Rectangular slots approximate angled parking; polygons + calibration would improve reliability.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: ultralytics` | `pip install -r requirements.txt` |
| Invalid dataset path | Confirm `Parking_dataset (1)\Parking_dataset\data.yaml` exists |
| Invalid YOLO labels | Re-run `data_inspection.py`; fix empty/orphan labels |
| CUDA unavailable | Train on Kaggle GPU (50 epochs) — see `KAGGLE_TRAINING.md` |
| Missing `best.pt` | Run `train.py` or use `--use-gt-slots --use-gt-vehicles` for geometry demos |
| Symlink errors on Windows | Re-run `prepare_data.py --copy` |

## Citation

Almeida et al., PKLot, Expert Systems with Applications, 2015. Dataset via Roboflow CC BY 4.0.

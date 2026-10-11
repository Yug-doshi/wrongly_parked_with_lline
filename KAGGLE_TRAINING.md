# Train on Kaggle GPU — 50 epochs

Local CPU training is only a quick demo. For the research-quality run, train on **Kaggle GPU for 50 epochs** using this guide and `notebooks/ParkingDetection_Kaggle.ipynb`.

## What you will train

| Model | Dataset | Classes | Purpose |
|---|---|---|---|
| `slot_det` | `dataset/space` | `0=space-empty`, `1=space-occupied` | Free / occupied parking slots |
| `car_det` | `dataset/car` | `0=car` | Approximate vehicle boxes (occupied → car) |

Wrong parking is **not** trained. It is decided by overlap geometry in `src/inference.py` (GREEN / BLUE / RED).

## Recommended settings (this project)

| Setting | Value |
|---|---|
| Model | `yolov8s.pt` |
| Image size | `640` |
| **Epochs** | **`50`** |
| Patience | `15` |
| Batch | `16` (use `8` or `4` if OOM) |
| Device | `0` (GPU) |
| Workers | `2` |
| Seed | `42` |
| Data | **full** (`--max-train 0`) |

Config file: `configs/train_kaggle.yaml`.

---

## Part A — Upload to Kaggle (once)

### A1. PKLot dataset

Local path: `D:\RMS 4\Parking_dataset (1)\Parking_dataset`

Zip so the archive contains:

```text
train/images  train/labels
valid/images  valid/labels
test/images   test/labels
data.yaml
```

Kaggle → **Datasets** → **New Dataset** → upload → name e.g. `pklot-yolo-parking`.

### A2. Code dataset

From PowerShell:

```powershell
cd "D:\RMS 4\parking-detection"
powershell -ExecutionPolicy Bypass -File scripts\make_kaggle_code_zip.ps1
```

Upload `parking-detection-code.zip` as a Kaggle dataset named e.g. `parking-detection-code`.

### A3. Notebook

1. Kaggle → **Code** → **New Notebook**
2. Settings: **GPU** accelerator, **Internet On**
3. Add Input: both datasets from A1 and A2
4. File → Import notebook → upload `notebooks/ParkingDetection_Kaggle.ipynb`  
   **or** copy cells from that file
5. Edit `CODE_INPUT` / `DATA_INPUT` paths to match `/kaggle/input/...` names shown in the first cell

---

## Part B — Run order (50 epochs)

The notebook does this automatically:

1. Install Ultralytics + verify CUDA  
2. Copy code to `/kaggle/working/parking-detection`  
3. `prepare_data.py --max-train 0 --copy` (full space + car datasets)  
4. Train **slot_det** — **50 epochs**  
5. Train **car_det** — **50 epochs**  
6. Evaluate both on the **test** split  

### Manual equivalent commands

```bash
python src/prepare_data.py --src /kaggle/input/<pklot>/... --out dataset --max-train 0 --seed 42 --copy

python src/train.py --data dataset/space/data.yaml --model yolov8s.pt --name slot_det \
  --epochs 50 --batch 16 --imgsz 640 --device 0 --workers 2 --patience 15 --seed 42 \
  --project /kaggle/working/parking-detection/runs

python src/train.py --data dataset/car/data.yaml --model yolov8s.pt --name car_det \
  --epochs 50 --batch 16 --imgsz 640 --device 0 --workers 2 --patience 15 --seed 42 \
  --project /kaggle/working/parking-detection/runs
```

Expected outputs:

```text
weights/slot_det_best.pt
weights/car_det_best.pt
runs/slot_det/results.csv
runs/car_det/results.csv
```

---

## Part C — Download and use locally

Download at least:

```text
weights/slot_det_best.pt
weights/car_det_best.pt
runs/*/results.csv
outputs/evaluation/*.json
```

Place weights here:

```text
D:\RMS 4\parking-detection\weights\slot_det_best.pt
D:\RMS 4\parking-detection\weights\car_det_best.pt
```

```powershell
cd "D:\RMS 4\parking-detection"
python src\inference.py `
  --source "D:\RMS 4\Parking_dataset (1)\Parking_dataset\test\images" `
  --slot-weights weights\slot_det_best.pt `
  --vehicle-weights weights\car_det_best.pt `
  --out outputs\inference\kaggle_50ep
```

---

## Checklist

- [ ] GPU accelerator enabled  
- [ ] Internet on  
- [ ] Both datasets attached  
- [ ] Notebook found `train/valid/test`  
- [ ] `--max-train 0` (full data)  
- [ ] **50 epochs** for both models  
- [ ] Class mapping unchanged: `0=space-empty`, `1=space-occupied`  
- [ ] Both `best.pt` downloaded  
- [ ] YOLO mAP not claimed as wrong-parking accuracy  

## Troubleshooting

| Problem | Fix |
|---|---|
| No CUDA | Settings → Accelerator → GPU → restart |
| OOM | `BATCH = 8` or `4` in the notebook config cell |
| Cannot find `src/train.py` | Fix `CODE_INPUT`; list `/kaggle/input` |
| Wrong dataset nesting | Point `DATA_INPUT` at the folder that contains `train/` |
| Pip fails | Internet = On |

## Time (approx., T4)

| Step | Time |
|---|---|
| Prepare full data | a few minutes |
| Slot 50 epochs yolov8s | ~1–2.5 h |
| Car 50 epochs yolov8s | ~1–2 h |

Early stopping (`patience=15`) may finish before all 50 epochs if validation plateaus.

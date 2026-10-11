"""Parking-slot detection via trained YOLOv8 space-empty / space-occupied model."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
from ultralytics import YOLO

from geometry import ParkingSlot, yolo_norm_to_xyxy

# Preserve original dataset class mapping
SLOT_CLASS_NAMES = {0: "space-empty", 1: "space-occupied"}


class SlotDetector:
    def __init__(self, weights: Union[str, Path], conf: float = 0.25, iou: float = 0.45, device: str = "cpu"):
        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        self.device = device

    def detect(self, image: np.ndarray) -> List[ParkingSlot]:
        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        slots: List[ParkingSlot] = []
        if not results:
            return slots
        r0 = results[0]
        h, w = image.shape[:2]
        if r0.boxes is None:
            return slots
        for i, box in enumerate(r0.boxes):
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = SLOT_CLASS_NAMES.get(cls_id, str(cls_id))
            slots.append(
                ParkingSlot(
                    slot_id=f"Slot-{i + 1:02d}",
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                    source_class=name,
                )
            )
        return slots


def slots_from_yolo_labels(
    label_path: Union[str, Path],
    width: int,
    height: int,
    class_filter: Optional[Sequence[int]] = None,
) -> List[ParkingSlot]:
    """Build stable slot geometry from ground-truth YOLO labels (preferred when available)."""
    path = Path(label_path)
    slots: List[ParkingSlot] = []
    if not path.is_file():
        return slots
    lines = path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    idx = 0
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        if class_filter is not None and cid not in class_filter:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        bbox = yolo_norm_to_xyxy(cx, cy, bw, bh, width, height)
        idx += 1
        slots.append(
            ParkingSlot(
                slot_id=f"Slot-{idx:02d}",
                bbox=bbox,
                confidence=1.0,
                source_class=SLOT_CLASS_NAMES.get(cid, str(cid)),
            )
        )
    return slots

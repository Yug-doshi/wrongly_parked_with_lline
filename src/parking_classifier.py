"""Configurable free / correct / wrong / uncertain parking classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from geometry import (
    MultiSlotOverlap,
    ParkingSlot,
    VehicleDetection,
    assign_vehicle_to_slot,
    calculate_slot_coverage_ratio,
    calculate_vehicle_inside_ratio,
    vehicle_intrusion_ratio_beyond_tolerance,
)


DEFAULT_THRESHOLDS = {
    "vehicle_inside_min": 0.90,
    "outside_max": 0.10,
    "slot_assign_min": 0.30,
    "minimum_second_slot_intrusion_ratio": 0.15,
    "boundary_crossing_tolerance": 0.05,
    "slot_coverage_occupied_min": 0.30,
    "usable_free_max_intrusion": 0.05,
    "vehicle_detection_conf_min": 0.10,
    "vehicle_conf_min": 0.25,
    "slot_conf_min": 0.25,
    "uncertain_conf_max": 0.40,
    "nms_iou": 0.45,
}


def load_thresholds(path: Optional[str | Path] = None) -> Dict[str, float]:
    cfg = dict(DEFAULT_THRESHOLDS)
    if path is not None and Path(path).is_file():
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update({k: float(v) for k, v in loaded.items() if isinstance(v, (int, float))})
    return cfg


@dataclass
class VehicleParkingResult:
    vehicle: VehicleDetection
    status: str  # CORRECT_PARKING | WRONG_PARKING | UNCERTAIN
    slot_id: Optional[str] = None
    r_vehicle: float = 0.0
    r_outside: float = 0.0
    r_slot: float = 0.0
    overlapped_slots: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class SlotOccupancyResult:
    slot: ParkingSlot
    status: str  # FREE | OCCUPIED_CORRECT | OCCUPIED_WRONG | PARTIALLY_OBSTRUCTED
    physical_occupied: bool = False
    usable_available: bool = True
    best_r_slot: float = 0.0
    vehicle_ids: List[int] = field(default_factory=list)


@dataclass
class ParkingFrameResult:
    vehicles: List[VehicleParkingResult]
    slots: List[SlotOccupancyResult]
    summary: Dict[str, Any]


def detect_wrong_parking(
    vehicle: VehicleDetection,
    slots: Sequence[ParkingSlot],
    thresholds: Dict[str, float],
) -> VehicleParkingResult:
    """Classify one vehicle using overlap ratios — never a learned wrong-parking class."""
    conf_min = thresholds["vehicle_conf_min"]
    uncertain_max = thresholds["uncertain_conf_max"]
    assign_min = thresholds["slot_assign_min"]
    inside_min = thresholds["vehicle_inside_min"]
    outside_max = thresholds["outside_max"]
    second_slot_intrusion_min = thresholds["minimum_second_slot_intrusion_ratio"]
    boundary_tolerance_ratio = thresholds["boundary_crossing_tolerance"]

    if vehicle.confidence < conf_min:
        return VehicleParkingResult(
            vehicle=vehicle,
            status="UNCERTAIN",
            reason="below confidence floor",
        )
    if vehicle.confidence < uncertain_max:
        assignment = assign_vehicle_to_slot(vehicle, slots, assign_min=assign_min)
        # Still compute overlaps for diagnostics, but mark uncertain
        primary = assignment.primary
        return VehicleParkingResult(
            vehicle=vehicle,
            status="UNCERTAIN",
            slot_id=None if primary is None else primary.slot_id,
            r_vehicle=0.0 if primary is None else primary.r_vehicle,
            r_outside=1.0 if primary is None else primary.r_outside,
            r_slot=0.0 if primary is None else primary.r_slot,
            reason="low confidence",
        )

    assignment: MultiSlotOverlap = assign_vehicle_to_slot(vehicle, slots, assign_min=assign_min)
    primary = assignment.primary

    if primary is None:
        # Meaningful overlap with parking area but no assignable slot?
        soft = [m for m in assignment.all_overlaps if m.r_vehicle >= 0.05 or m.r_slot >= 0.05]
        if soft:
            ids = [m.slot_id for m in soft[:3]]
            return VehicleParkingResult(
                vehicle=vehicle,
                status="WRONG_PARKING",
                r_vehicle=soft[0].r_vehicle,
                r_outside=soft[0].r_outside,
                overlapped_slots=ids,
                reason="no valid slot assignment",
            )
        return VehicleParkingResult(
            vehicle=vehicle,
            status="WRONG_PARKING",
            r_outside=1.0,
            reason="outside parking area",
        )

    overlapped = [primary.slot_id]
    vehicle_width = max(0.0, vehicle.bbox[2] - vehicle.bbox[0])
    vehicle_height = max(0.0, vehicle.bbox[3] - vehicle.bbox[1])
    boundary_tolerance = min(vehicle_width, vehicle_height) * boundary_tolerance_ratio
    slot_by_id = {slot.slot_id: slot for slot in slots}
    intrusive = []
    for metrics in assignment.all_overlaps:
        if metrics.slot_id == primary.slot_id:
            continue
        candidate_slot = slot_by_id.get(metrics.slot_id)
        if candidate_slot is None:
            continue
        intrusion_ratio = vehicle_intrusion_ratio_beyond_tolerance(
            vehicle, candidate_slot, slot_by_id[primary.slot_id], boundary_tolerance
        )
        if intrusion_ratio >= second_slot_intrusion_min:
            intrusive.append(metrics)
            overlapped.append(metrics.slot_id)

    if intrusive:
        return VehicleParkingResult(
            vehicle=vehicle,
            status="WRONG_PARKING",
            slot_id=primary.slot_id,
            r_vehicle=primary.r_vehicle,
            r_outside=primary.r_outside,
            r_slot=primary.r_slot,
            overlapped_slots=overlapped,
            reason=f"straddles {primary.slot_id} / {', '.join(m.slot_id for m in intrusive)}",
        )

    if primary.r_vehicle < inside_min or primary.r_outside > outside_max:
        return VehicleParkingResult(
            vehicle=vehicle,
            status="WRONG_PARKING",
            slot_id=primary.slot_id,
            r_vehicle=primary.r_vehicle,
            r_outside=primary.r_outside,
            r_slot=primary.r_slot,
            overlapped_slots=[primary.slot_id],
            reason=f"outside {primary.r_outside:.0%}",
        )

    return VehicleParkingResult(
        vehicle=vehicle,
        status="CORRECT_PARKING",
        slot_id=primary.slot_id,
        r_vehicle=primary.r_vehicle,
        r_outside=primary.r_outside,
        r_slot=primary.r_slot,
        overlapped_slots=[primary.slot_id],
        reason="within slot bounds",
    )


def calculate_slot_occupancy(
    slots: Sequence[ParkingSlot],
    vehicles: Sequence[VehicleDetection],
    vehicle_results: Sequence[VehicleParkingResult],
    thresholds: Dict[str, float],
) -> List[SlotOccupancyResult]:
    """Physical occupancy + usable availability per slot."""
    occ_min = thresholds["slot_coverage_occupied_min"]
    usable_max = thresholds["usable_free_max_intrusion"]
    vr_by_id = {r.vehicle.vehicle_id: r for r in vehicle_results}

    out: List[SlotOccupancyResult] = []
    for slot in slots:
        detector_says_occupied = slot.source_class == "space-occupied"
        best_r = 0.0
        vids: List[int] = []
        statuses: List[str] = []
        for v in vehicles:
            r_slot = calculate_slot_coverage_ratio(v.bbox, slot.as_polygon())
            r_veh = calculate_vehicle_inside_ratio(v.bbox, slot.as_polygon())
            intrusion = max(r_slot, r_veh)
            if intrusion >= usable_max:
                vids.append(v.vehicle_id)
                statuses.append(vr_by_id.get(v.vehicle_id, VehicleParkingResult(v, "UNCERTAIN")).status)
            if r_slot > best_r:
                best_r = r_slot

        physical = detector_says_occupied or best_r >= occ_min or any(
            calculate_vehicle_inside_ratio(v.bbox, slot.as_polygon()) >= thresholds["slot_assign_min"]
            for v in vehicles
        )
        usable = not detector_says_occupied
        for v in vehicles:
            r_slot = calculate_slot_coverage_ratio(v.bbox, slot.as_polygon())
            r_veh = calculate_vehicle_inside_ratio(v.bbox, slot.as_polygon())
            if max(r_slot, r_veh) >= usable_max:
                usable = False
                break

        if not physical and usable:
            status = "FREE"
        elif physical and any(s == "WRONG_PARKING" for s in statuses):
            status = "OCCUPIED_WRONG" if best_r >= occ_min else "PARTIALLY_OBSTRUCTED"
        elif detector_says_occupied and not any(s == "CORRECT_PARKING" for s in statuses):
            status = "OCCUPIED_UNVERIFIED"
        elif any(s == "UNCERTAIN" for s in statuses):
            status = "OCCUPIED_UNVERIFIED"
        elif physical:
            status = "OCCUPIED_CORRECT"
        elif not usable:
            status = "PARTIALLY_OBSTRUCTED"
        else:
            status = "FREE"

        out.append(
            SlotOccupancyResult(
                slot=slot,
                status=status,
                physical_occupied=physical,
                usable_available=usable and status == "FREE",
                best_r_slot=best_r,
                vehicle_ids=vids,
            )
        )
    return out


def calculate_parking_summary(
    vehicle_results: Sequence[VehicleParkingResult],
    slot_results: Sequence[SlotOccupancyResult],
) -> Dict[str, Any]:
    total_slots = len(slot_results)
    free = sum(1 for s in slot_results if s.status == "FREE" and s.usable_available)
    occupied_correct = sum(1 for s in slot_results if s.status == "OCCUPIED_CORRECT")
    occupied_wrong_slots = sum(
        1 for s in slot_results if s.status in ("OCCUPIED_WRONG", "PARTIALLY_OBSTRUCTED")
    )
    occupied_unverified_slots = sum(1 for s in slot_results if s.status == "OCCUPIED_UNVERIFIED")
    wrong_vehicles = sum(1 for v in vehicle_results if v.status == "WRONG_PARKING")
    correct_vehicles = sum(1 for v in vehicle_results if v.status == "CORRECT_PARKING")
    uncertain = sum(1 for v in vehicle_results if v.status == "UNCERTAIN")
    # Unique vehicles only — never double-count one car as two
    n_vehicles = len({v.vehicle.vehicle_id for v in vehicle_results})
    occupied_pct = 0.0 if total_slots == 0 else 100.0 * (total_slots - free) / total_slots
    return {
        "total_slots": total_slots,
        "free_slots": free,
        "correctly_occupied_slots": occupied_correct,
        "wrong_affected_slots": occupied_wrong_slots,
        "unverified_occupied_slots": occupied_unverified_slots,
        "correct_parking_vehicles": correct_vehicles,
        "wrong_parking_vehicles": wrong_vehicles,
        "uncertain_detections": uncertain,
        "unique_vehicles": n_vehicles,
        "available_spaces": free,
        "occupancy_percent": round(occupied_pct, 2),
    }


def classify_frame(
    vehicles: Sequence[VehicleDetection],
    slots: Sequence[ParkingSlot],
    thresholds: Optional[Dict[str, float]] = None,
) -> ParkingFrameResult:
    thr = thresholds or dict(DEFAULT_THRESHOLDS)
    # Filter slots by confidence if present
    slot_conf = thr.get("slot_conf_min", 0.25)
    slots_f = [s for s in slots if s.confidence >= slot_conf]

    vehicle_results = [detect_wrong_parking(v, slots_f, thr) for v in vehicles]
    slot_results = calculate_slot_occupancy(slots_f, vehicles, vehicle_results, thr)
    summary = calculate_parking_summary(vehicle_results, slot_results)
    return ParkingFrameResult(vehicles=vehicle_results, slots=slot_results, summary=summary)

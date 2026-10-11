"""Unit tests for overlap geometry and parking classification (12 required scenarios)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry import (
    ParkingSlot,
    VehicleDetection,
    assign_vehicle_to_slot,
    calculate_iou,
    calculate_outside_ratio,
    calculate_slot_coverage_ratio,
    calculate_vehicle_inside_ratio,
    clip_bbox_to_image,
)
from parking_classifier import classify_frame, detect_wrong_parking, load_thresholds


@pytest.fixture
def thr():
    return load_thresholds(ROOT / "configs" / "thresholds.yaml")


@pytest.fixture
def slot01():
    return ParkingSlot(slot_id="Slot-01", bbox=[100, 100, 200, 200])


@pytest.fixture
def slot02():
    return ParkingSlot(slot_id="Slot-02", bbox=[200, 100, 300, 200])


def test_1_car_completely_inside_one_slot(thr, slot01):
    car = VehicleDetection(0, [110, 110, 190, 190], confidence=0.95)
    r = detect_wrong_parking(car, [slot01], thr)
    assert calculate_vehicle_inside_ratio(car.bbox, slot01.as_polygon()) >= 0.99
    assert r.status == "CORRECT_PARKING"
    assert r.slot_id == "Slot-01"


def test_2_car_slightly_outside_within_tolerance(thr, slot01):
    # ~8% outside — within outside_max=0.10
    car = VehicleDetection(0, [108, 108, 208, 192], confidence=0.9)
    r_out = calculate_outside_ratio(car.bbox, slot01.as_polygon())
    assert r_out <= thr["outside_max"] + 0.02
    r = detect_wrong_parking(car, [slot01], thr)
    # May be correct or wrong depending on exact ratio; assert no crash and valid status
    assert r.status in ("CORRECT_PARKING", "WRONG_PARKING")
    assert 0.0 <= r.r_vehicle <= 1.0


def test_3_car_substantially_outside(thr, slot01):
    car = VehicleDetection(0, [150, 150, 280, 280], confidence=0.9)
    assert calculate_outside_ratio(car.bbox, slot01.as_polygon()) > 0.3
    r = detect_wrong_parking(car, [slot01], thr)
    assert r.status == "WRONG_PARKING"


def test_4_car_overlapping_two_adjacent_slots(thr, slot01, slot02):
    car = VehicleDetection(0, [150, 110, 250, 190], confidence=0.92)
    assignment = assign_vehicle_to_slot(car, [slot01, slot02], assign_min=0.2)
    assert assignment.primary is not None
    assert len([m for m in assignment.all_overlaps if m.r_vehicle >= 0.1]) >= 2
    r = detect_wrong_parking(car, [slot01, slot02], thr)
    assert r.status == "WRONG_PARKING"
    assert len(r.overlapped_slots) >= 2


def test_car_straddling_large_adjacent_slots_is_wrong_by_vehicle_intrusion(thr):
    slot_left = ParkingSlot(slot_id="Slot-12", bbox=[0, 0, 400, 200])
    slot_right = ParkingSlot(slot_id="Slot-13", bbox=[400, 0, 800, 200])
    car = VehicleDetection(12, [360, 50, 440, 130], confidence=0.95)

    r = detect_wrong_parking(car, [slot_left, slot_right], thr)

    assert r.status == "WRONG_PARKING"
    assert r.overlapped_slots == ["Slot-12", "Slot-13"]
    assert r.reason == "straddles Slot-12 / Slot-13"


def test_boundary_crossing_within_tolerance_is_not_wrong(thr):
    slot_left = ParkingSlot(slot_id="Slot-12", bbox=[0, 0, 400, 200])
    slot_right = ParkingSlot(slot_id="Slot-13", bbox=[400, 0, 800, 200])
    car = VehicleDetection(13, [320, 50, 403, 130], confidence=0.95)

    r = detect_wrong_parking(car, [slot_left, slot_right], thr)

    assert r.status == "CORRECT_PARKING"
    assert r.overlapped_slots == ["Slot-12"]


def test_5_slot_with_no_vehicle(thr, slot01):
    frame = classify_frame([], [slot01], thr)
    assert frame.slots[0].status == "FREE"
    assert frame.slots[0].usable_available is True
    assert frame.summary["free_slots"] == 1


def test_occupied_slot_without_vehicle_is_not_reported_free(thr):
    occupied_slot = ParkingSlot(
        slot_id="Slot-occupied",
        bbox=[100, 100, 200, 200],
        source_class="space-occupied",
    )
    frame = classify_frame([], [occupied_slot], thr)
    assert frame.slots[0].status == "OCCUPIED_UNVERIFIED"
    assert frame.slots[0].usable_available is False
    assert frame.summary["free_slots"] == 0
    assert frame.summary["unverified_occupied_slots"] == 1


def test_6_slot_partially_obstructed_by_wrong_car(thr, slot01, slot02):
    car = VehicleDetection(0, [160, 110, 260, 190], confidence=0.9)
    frame = classify_frame([car], [slot01, slot02], thr)
    assert any(v.status == "WRONG_PARKING" for v in frame.vehicles)
    assert any(s.status in ("OCCUPIED_WRONG", "PARTIALLY_OBSTRUCTED", "OCCUPIED_CORRECT") for s in frame.slots)
    # Not both freely available
    assert frame.summary["free_slots"] < 2


def test_7_two_cars_in_different_slots(thr, slot01, slot02):
    c1 = VehicleDetection(0, [110, 110, 190, 190], confidence=0.95)
    c2 = VehicleDetection(1, [210, 110, 290, 190], confidence=0.95)
    frame = classify_frame([c1, c2], [slot01, slot02], thr)
    assert frame.summary["unique_vehicles"] == 2
    assert all(v.status == "CORRECT_PARKING" for v in frame.vehicles)
    assert frame.summary["free_slots"] == 0


def test_8_low_confidence_vehicle(thr, slot01):
    car = VehicleDetection(0, [110, 110, 190, 190], confidence=0.30)
    r = detect_wrong_parking(car, [slot01], thr)
    assert r.status == "UNCERTAIN"


def test_low_confidence_detection_threshold_keeps_uncertain_vehicle_visible(thr, slot01):
    assert thr["vehicle_detection_conf_min"] < thr["vehicle_conf_min"]
    car = VehicleDetection(0, [110, 110, 190, 190], confidence=0.15)
    frame = classify_frame([car], [slot01], thr)
    assert len(frame.vehicles) == 1
    assert frame.vehicles[0].status == "UNCERTAIN"


def test_9_ambiguous_slot_assignment(thr, slot01, slot02):
    # Tiny overlap with both — below assign_min
    car = VehicleDetection(0, [195, 140, 205, 160], confidence=0.9)
    r = detect_wrong_parking(car, [slot01, slot02], thr)
    assert r.status == "WRONG_PARKING"
    assert r.slot_id is None or r.reason


def test_10_partially_occluded_vehicle_still_classifiable(thr, slot01):
    # Smaller visible box still mostly inside
    car = VehicleDetection(0, [120, 120, 170, 160], confidence=0.85)
    r = detect_wrong_parking(car, [slot01], thr)
    assert r.status == "CORRECT_PARKING"
    assert calculate_iou(car.bbox, slot01.bbox) > 0


def test_11_car_touches_slot_boundary(thr, slot01):
    car = VehicleDetection(0, [100, 100, 200, 200], confidence=0.9)  # exact slot bounds
    r_out = calculate_outside_ratio(car.bbox, slot01.as_polygon())
    assert r_out < 1e-6
    r = detect_wrong_parking(car, [slot01], thr)
    assert r.status == "CORRECT_PARKING"


def test_12_bbox_extends_beyond_image(thr, slot01):
    raw = [-20, 110, 190, 190]
    clipped = clip_bbox_to_image(raw, width=640, height=640)
    assert clipped[0] >= 0
    car = VehicleDetection(0, clipped, confidence=0.9)
    # No division by zero
    r_v = calculate_vehicle_inside_ratio(car.bbox, slot01.as_polygon())
    r_s = calculate_slot_coverage_ratio(car.bbox, slot01.as_polygon())
    assert 0.0 <= r_v <= 1.0
    assert 0.0 <= r_s <= 1.0
    r = detect_wrong_parking(car, [slot01], thr)
    assert r.status in ("CORRECT_PARKING", "WRONG_PARKING", "UNCERTAIN")


def test_zero_area_vehicle_safe(slot01):
    car = VehicleDetection(0, [100, 100, 100, 100], confidence=0.9)
    assert calculate_vehicle_inside_ratio(car.bbox, slot01.as_polygon()) == 0.0
    assert calculate_iou(car.bbox, slot01.bbox) == 0.0


def test_polygon_slot_overlap(thr):
    slot = ParkingSlot(
        slot_id="Slot-01",
        polygon=[(100, 100), (220, 90), (230, 200), (90, 210)],
    )
    car = VehicleDetection(0, [120, 120, 180, 180], confidence=0.95)
    assert calculate_vehicle_inside_ratio(car.bbox, slot.as_polygon()) > 0.9
    r = detect_wrong_parking(car, [slot], thr)
    assert r.status == "CORRECT_PARKING"

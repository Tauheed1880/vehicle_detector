"""
Configuration settings for Vehicle Tracking and FastALPR Pipeline.
All thresholds, paths, model names, smart frame gating, and timing parameters are defined here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class PipelineConfig:
    # -------------------------------------------------------------
    # Camera & System Identification
    # -------------------------------------------------------------
    camera_id: str = "CAM-00001"

    # -------------------------------------------------------------
    # Vehicle Detector & Tracker Configuration
    # NOTE ON LICENSE:
    # Ultralytics YOLO models (YOLOv8, YOLO11, etc.) are licensed under AGPL-3.0.
    # If used in commercial proprietary software without open-sourcing,
    # an enterprise commercial license from Ultralytics is required.
    # -------------------------------------------------------------
    vehicle_model_name: str = "yolov8n.pt"  # Lightweight nano model
    vehicle_conf_thresh: float = 0.40       # Confidence threshold for vehicles
    # Target COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
    vehicle_classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])
    tracker_config: str = "bytetrack.yaml"  # Built-in ByteTrack tracker

    # -------------------------------------------------------------
    # FastALPR License Plate Detection & OCR Configuration
    # Uses ONNX models under MIT / open-source licenses
    # -------------------------------------------------------------
    alpr_detector_model: str = "yolo-v9-t-384-license-plate-end2end"
    alpr_detector_conf_thresh: float = 0.35
    alpr_ocr_model: str = "cct-xs-v2-global-model"
    # ALPR frame interval: Run ALPR every K frames to save compute
    alpr_frame_interval: int = 2

    # -------------------------------------------------------------
    # Adaptive Smart Frame & Quality Gating (Enterprise Optimization)
    # -------------------------------------------------------------
    enable_smart_gating: bool = True
    # Minimum vehicle bounding box area (in pixels^2) before triggering ALPR
    # Filters out far-away vehicles at the horizon where plates are unreadable (< 80x80px)
    min_vehicle_area_for_alpr: float = 6400.0
    # Minimum sharpness score (Laplacian variance) to reject motion-blurred plates
    min_plate_sharpness: float = 35.0

    # -------------------------------------------------------------
    # Single Detection & Event Emission Rules (STRICT)
    # -------------------------------------------------------------
    # Minimum frames a vehicle must be tracked before emitting an event
    min_track_frames: int = 5

    # Frames of absence before a lost track is considered exited/dead
    max_disappeared_frames: int = 30

    # Minimum OCR confidence to consider a plate confident and stop ALPR on this vehicle
    min_confident_plate_conf: float = 0.75

    # Duplicate plate suppression window in seconds (to handle track ID switches)
    duplicate_plate_window_sec: float = 30.0

    # Bounding box smoothing factor (EMA: Exponential Moving Average, 0.0 to 1.0)
    # Higher = faster response to new position, Lower = smoother box
    box_smoothing_factor: float = 0.65

    # -------------------------------------------------------------
    # Paths & Storage
    # -------------------------------------------------------------
    base_output_dir: Path = Path("output")
    snapshots_dir: Path = Path("output/snapshots")
    events_json_path: Path = Path("output/events.json")
    events_jsonl_path: Path = Path("output/events.jsonl")

    def __post_init__(self) -> None:
        self.base_output_dir = Path(self.base_output_dir)
        self.snapshots_dir = Path(self.snapshots_dir)
        self.events_json_path = Path(self.events_json_path)
        self.events_jsonl_path = Path(self.events_jsonl_path)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

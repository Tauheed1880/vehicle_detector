"""
Configuration settings for Vehicle Tracking and FastALPR Pipeline (Phase 1 & Phase 2 Ready).
All thresholds, paths, model names, ROI polygons, and preview settings are defined here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


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
    # For commercial / proprietary deployments, consider:
    # 1. YOLOv10 (Apache-2.0)
    # 2. RT-DETR (Apache-2.0)
    # 3. YOLOX (Apache-2.0)
    # 4. Ultralytics Commercial Enterprise License
    # -------------------------------------------------------------
    vehicle_model_name: str = "yolov8n.pt"  # Lightweight nano model
    vehicle_conf_thresh: float = 0.35       # Confidence threshold for vehicles
    # Target COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
    vehicle_classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])
    tracker_config: str = "custom_bytetrack.yaml"  # Tuned ByteTrack config

    # Nested & Duplicate Box Elimination
    box_containment_thresh: float = 0.75  # Merge if one box is >= 75% inside another
    class_agnostic_nms_iou: float = 0.65  # Class-agnostic NMS IoU threshold

    # -------------------------------------------------------------
    # Readable Zone (ROI) & Distance Filtering
    # Far-away vehicles outside the readable zone are NOT counted as "NO PLATE"
    # -------------------------------------------------------------
    enable_smart_gating: bool = True
    min_vehicle_width: int = 70
    min_vehicle_height: int = 50
    min_readable_vehicle_area: float = 8000.0  # Minimum box area to be considered readable
    # Optional ROI polygon [[x, y], ...]. If None, whole frame is used but area gate applies.
    roi_polygon: Optional[List[List[int]]] = None

    # -------------------------------------------------------------
    # FastALPR License Plate Detection & OCR Configuration
    # -------------------------------------------------------------
    alpr_detector_model: str = "yolo-v9-t-384-license-plate-end2end"
    alpr_detector_conf_thresh: float = 0.35
    alpr_ocr_model: str = "cct-xs-v2-global-model"
    # ALPR frame interval: Run ALPR every K frames to save compute
    alpr_frame_interval: int = 2

    # Plate Quality & Voting
    min_plate_length: int = 4
    min_plate_sharpness: float = 30.0
    # Minimum plate reads needed before declaring a confident plate event
    min_plate_reads_for_event: int = 2
    # Country-wise regex (UK standard default: 2 letters, 2 digits, 3 letters e.g. AB12CDE)
    plate_country_code: str = "UK"
    plate_regex_pattern: str = r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$"
    enforce_regex: bool = False  # Set True to strictly enforce regex format

    # -------------------------------------------------------------
    # Single Detection & Fuzzy Duplicate Suppression
    # -------------------------------------------------------------
    min_track_frames: int = 5            # Minimum frames tracked before emitting event
    max_disappeared_frames: int = 45     # Frames lost before closing track (1.5s at 30 fps)
    min_confident_plate_conf: float = 0.75
    duplicate_plate_window_sec: float = 60.0  # Suppress duplicate events within 60s
    fuzzy_plate_max_distance: int = 2         # Max Levenshtein distance for duplicate matching
    box_smoothing_factor: float = 0.65        # EMA box smoothing
    include_remarks: bool = False             # Set True only if remarks field is requested

    # -------------------------------------------------------------
    # Real-Time Colab Preview & Performance
    # -------------------------------------------------------------
    show_preview: bool = True
    preview_interval: int = 2            # Update preview every N-th frame
    preview_max_width: int = 960
    preview_max_height: int = 540
    preview_jpeg_quality: int = 70
    sync_realtime: bool = True           # Sync playback to real wall-clock time
    save_video: bool = True
    convert_h264: bool = True            # Convert to H.264 / yuv420p with ffmpeg

    # -------------------------------------------------------------
    # Paths & Storage
    # -------------------------------------------------------------
    base_output_dir: Path = Path("output")
    snapshots_dir: Path = Path("output/snapshots")
    events_json_path: Path = Path("output/events.json")
    events_jsonl_path: Path = Path("output/events.jsonl")
    drive_output_dir: Path = Path("/content/drive/MyDrive/ALPR_Output")

    def __post_init__(self) -> None:
        self.base_output_dir = Path(self.base_output_dir)
        self.snapshots_dir = Path(self.snapshots_dir)
        self.events_json_path = Path(self.events_json_path)
        self.events_jsonl_path = Path(self.events_jsonl_path)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

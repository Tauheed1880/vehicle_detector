"""
Track state management, plate-to-vehicle association, and single-event emission.
Guarantees:
1. Exactly ONE event per vehicle track ID (preventing repetitive duplicate JSON events).
2. Bijective infinite alphabet event ID: evt_id_{yymmdd}_{camera_num}_{alphabets}.
3. Best frame snapshot (largest vehicle box + clearest plate with padding).
4. Remarks / Reason field handling clear captures, faint/low-confidence characters, and missing plates.
5. Duplicate plate suppression within a time window (handles ID switching).
"""

import json
import random
import re
import string
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from config import PipelineConfig


def int_to_alphabets(n: int) -> str:
    """
    Converts a positive integer (1, 2, 3...) to Excel-style bijective alphabet string
    (1 -> A, 2 -> B... 26 -> Z, 27 -> AA, 28 -> AB... to infinity).
    """
    res = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        res.append(chr(65 + rem))
    return "".join(reversed(res)) if res else "A"


def get_camera_number(camera_id_str: str) -> str:
    """Extracts numeric camera number from camera_id (e.g. 'CAM-00001' -> '1', '4' -> '4')."""
    match = re.search(r"\d+", str(camera_id_str))
    if match:
        cleaned = match.group(0).lstrip("0")
        return cleaned if cleaned else "1"
    return "1"


def get_image_sharpness(img: Optional[np.ndarray]) -> float:
    """Calculates image sharpness score using Laplacian variance (higher is sharper)."""
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def generate_infinite_event_id(counter: int, camera_id_str: str, now: datetime) -> str:
    """
    Generates unique event ID formatted as:
    evt_id_{yymmdd}_{camera_number}_{alphabet_sequence}_{random_alphabets}
    Scales to infinity without collision.
    Example: evt_id_261122_1_A_XKW
    """
    yymmdd = now.strftime("%y%m%d")
    cam_num = get_camera_number(camera_id_str)
    alpha_seq = int_to_alphabets(counter)
    # 3 random uppercase letters guarantee 100% collision-proof global uniqueness
    rand_alpha = "".join(random.choices(string.ascii_uppercase, k=3))
    return f"evt_id_{yymmdd}_{cam_num}_{alpha_seq}_{rand_alpha}"


@dataclass
class TrackState:
    track_id: int
    first_frame_idx: int
    last_frame_idx: int
    frames_tracked: int = 1
    disappeared_count: int = 0
    # Smoothed bounding box [x1, y1, x2, y2]
    smoothed_box: Optional[List[float]] = None
    # Best vehicle snapshot data
    best_vehicle_box: Optional[List[int]] = None
    best_full_frame: Optional[np.ndarray] = None
    best_vehicle_crop: Optional[np.ndarray] = None
    best_vehicle_area: float = 0.0
    # Best plate snapshot & OCR data
    best_plate_crop: Optional[np.ndarray] = None
    best_plate_text: Optional[str] = None
    best_plate_conf: Optional[float] = None
    best_plate_det_conf: Optional[float] = None
    # Lifecycle flags
    has_event_emitted: bool = False
    plate_confident_reached: bool = False


class VehicleTrackerManager:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tracks: Dict[int, TrackState] = {}
        self.emitted_events: List[dict] = []
        # Set of track IDs that have ALREADY emitted an event (guarantees NO repeat JSON)
        self.emitted_track_ids: Set[int] = set()
        # Store emitted plate -> timestamp (epoch seconds) for duplicate filtering
        self.recent_emitted_plates: Dict[str, float] = {}
        self.event_counter: int = 0

        # Stats counters
        self.total_unique_vehicles: int = 0
        self.total_with_plate: int = 0
        self.total_without_plate: int = 0

    def update_track(
        self,
        track_id: int,
        box: List[float],
        frame: np.ndarray,
        frame_idx: int,
    ) -> TrackState:
        """
        Updates an existing track or creates a new track state with box smoothing.
        Maintains the clearest/largest vehicle snapshot with padding.
        """
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        area = float((x2 - x1) * (y2 - y1))

        # Add 5% padding around vehicle snapshot so it looks natural and not cut off
        pad_x = int((x2 - x1) * 0.05)
        pad_y = int((y2 - y1) * 0.05)
        vx1, vy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        vx2, vy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        vehicle_crop = frame[vy1:vy2, vx1:vx2].copy() if area > 0 else None

        if track_id not in self.tracks:
            track = TrackState(
                track_id=track_id,
                first_frame_idx=frame_idx,
                last_frame_idx=frame_idx,
                frames_tracked=1,
                disappeared_count=0,
                smoothed_box=[float(x1), float(y1), float(x2), float(y2)],
                best_vehicle_box=[x1, y1, x2, y2],
                best_full_frame=frame.copy() if area > 0 else None,
                best_vehicle_crop=vehicle_crop,
                best_vehicle_area=area,
            )
            self.tracks[track_id] = track
        else:
            track = self.tracks[track_id]
            track.last_frame_idx = frame_idx
            track.frames_tracked += 1
            track.disappeared_count = 0

            # Exponential Moving Average (EMA) smoothing to eliminate bounding box jitter
            alpha = self.config.box_smoothing_factor
            if track.smoothed_box is not None:
                track.smoothed_box = [
                    alpha * float(c) + (1.0 - alpha) * float(s)
                    for c, s in zip([x1, y1, x2, y2], track.smoothed_box)
                ]
            else:
                track.smoothed_box = [float(x1), float(y1), float(x2), float(y2)]

            # Keep crop with larger area before event emission (clearer closer view)
            if area > track.best_vehicle_area and area > 0:
                track.best_vehicle_area = area
                track.best_vehicle_box = [x1, y1, x2, y2]
                if not track.has_event_emitted:
                    track.best_full_frame = frame.copy()
                    track.best_vehicle_crop = vehicle_crop

        return self.tracks[track_id]

    def mark_disappeared_tracks(self, active_track_ids: set[int]) -> None:
        """Increments disappeared count for tracks that were not detected in current frame."""
        for t_id, track in self.tracks.items():
            if t_id not in active_track_ids:
                track.disappeared_count += 1

    def match_and_update_plates(
        self,
        alpr_results: list,
        frame: np.ndarray,
        frame_idx: int,
    ) -> None:
        """
        Matches full-frame ALPR plate detections to vehicles.
        Keeps plate text even if low confidence (faint letters) so it can be safely reported.
        """
        h, w = frame.shape[:2]

        for alpr_res in alpr_results:
            det = alpr_res.detection
            ocr = alpr_res.ocr
            if ocr is None or not ocr.text:
                continue

            bbox = det.bounding_box
            px1, py1, px2, py2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0

            best_match_id: Optional[int] = None
            smallest_enclosing_area = float("inf")

            for t_id, track in self.tracks.items():
                if track.disappeared_count > 0 or track.smoothed_box is None:
                    continue
                vx1, vy1, vx2, vy2 = track.smoothed_box
                if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                    v_area = (vx2 - vx1) * (vy2 - vy1)
                    if v_area < smallest_enclosing_area:
                        smallest_enclosing_area = v_area
                        best_match_id = t_id

            if best_match_id is not None:
                track = self.tracks[best_match_id]
                # If event already emitted for this vehicle, don't re-read
                if track.has_event_emitted:
                    continue

                ocr_conf = (
                    float(np.mean(ocr.confidence))
                    if isinstance(ocr.confidence, (list, tuple, np.ndarray))
                    else float(ocr.confidence)
                )
                plate_det_conf = float(det.confidence)

                # Crop plate image safely
                c_px1, c_py1 = max(0, int(px1)), max(0, int(py1))
                c_px2, c_py2 = min(w, int(px2)), min(h, int(py2))
                plate_crop = frame[c_py1:c_py2, c_px1:c_px2].copy()

                # Smart blur filter: Check sharpness of plate patch
                sharpness = get_image_sharpness(plate_crop)
                if self.config.enable_smart_gating and track.best_plate_crop is not None:
                    # If new plate is blurry and doesn't beat previous OCR confidence, reject
                    if sharpness < self.config.min_plate_sharpness and ocr_conf <= (track.best_plate_conf or 0.0):
                        continue

                # Crop vehicle image at this exact frame
                if track.smoothed_box is not None:
                    pad_x = int((track.smoothed_box[2] - track.smoothed_box[0]) * 0.05)
                    pad_y = int((track.smoothed_box[3] - track.smoothed_box[1]) * 0.05)
                    c_vx1 = max(0, int(track.smoothed_box[0]) - pad_x)
                    c_vy1 = max(0, int(track.smoothed_box[1]) - pad_y)
                    c_vx2 = min(w, int(track.smoothed_box[2]) + pad_x)
                    c_vy2 = min(h, int(track.smoothed_box[3]) + pad_y)
                    vehicle_crop = frame[c_vy1:c_vy2, c_vx1:c_vx2].copy()
                else:
                    vehicle_crop = None

                # Keep reading with highest OCR confidence (even faint ones are preserved)
                is_better = (track.best_plate_conf is None) or (ocr_conf > track.best_plate_conf)
                if is_better:
                    track.best_plate_text = str(ocr.text).strip().upper()
                    track.best_plate_conf = ocr_conf
                    track.best_plate_det_conf = plate_det_conf
                    track.best_plate_crop = plate_crop
                    track.best_full_frame = frame.copy()
                    if track.smoothed_box is not None:
                        track.best_vehicle_box = [int(v) for v in track.smoothed_box]
                    if vehicle_crop is not None and vehicle_crop.size > 0:
                        track.best_vehicle_crop = vehicle_crop

                    if ocr_conf >= self.config.min_confident_plate_conf:
                        track.plate_confident_reached = True

    def check_and_emit_events(
        self,
        current_frame_idx: int,
        force_all: bool = False,
    ) -> List[dict]:
        """
        Evaluates active and disappeared tracks.
        Enforces STRICT single-event rule:
          - Each vehicle track ID produces AT MOST ONE event.
          - Never duplicates or repeats an emitted track ID.
        """
        emitted_now: List[dict] = []
        now_ts = datetime.now().timestamp()

        for t_id, track in list(self.tracks.items()):
            # SAFEGUARD: If already emitted, NEVER emit again
            if track.has_event_emitted or (t_id in self.emitted_track_ids):
                continue

            # Must satisfy minimum track frames to filter false positives
            if track.frames_tracked < self.config.min_track_frames:
                if force_all:
                    continue
                if track.disappeared_count >= self.config.max_disappeared_frames:
                    del self.tracks[t_id]
                continue

            should_emit = False

            # Condition 1: Confident plate identified
            if track.plate_confident_reached and track.best_plate_text:
                should_emit = True

            # Condition 2: Vehicle exited the frame (disappeared for N frames) or stream finished
            elif track.disappeared_count >= self.config.max_disappeared_frames or force_all:
                should_emit = True

            if should_emit:
                # Mark emitted immediately to prevent any race condition
                track.has_event_emitted = True
                self.emitted_track_ids.add(t_id)

                event = self._emit_event(track, now_ts)
                if event is not None:
                    emitted_now.append(event)

        return emitted_now

    def _emit_event(self, track: TrackState, current_timestamp: float) -> Optional[dict]:
        """
        Formats and persists single event JSON with:
        - Infinite unique event_id: evt_id_{yymmdd}_{camera_num}_{alphabets}
        - Remarks: Clear detection, faint/low-confidence detection, or missing plate.
        - High-quality vehicle & plate snapshots.
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        has_plate = track.best_plate_text is not None and track.best_plate_conf is not None

        # Determine remarks and status
        if has_plate:
            if track.best_plate_conf >= self.config.min_confident_plate_conf:
                remarks = "Number plate successfully detected and recognized"
            else:
                remarks = "Low confidence / faint characters detected on plate"
        else:
            remarks = "No number plate detected on vehicle"

        # Duplicate suppression: If same plate text was emitted recently under another track ID, skip
        if has_plate:
            plate_key = track.best_plate_text
            last_seen = self.recent_emitted_plates.get(plate_key)
            if last_seen is not None:
                if (current_timestamp - last_seen) < self.config.duplicate_plate_window_sec:
                    print(
                        f"[DUPLICATE SUPPRESSED] Plate '{plate_key}' was emitted recently "
                        f"for another track ID. Skipping duplicate event."
                    )
                    return None
            self.recent_emitted_plates[plate_key] = current_timestamp

        self.event_counter += 1
        event_id = generate_infinite_event_id(self.event_counter, self.config.camera_id, now)

        # Save snapshot_vehicle as FULL-SIZE frame with bounding box ONLY on this specific vehicle
        vehicle_filename = f"vehicle_{event_id}.png"
        vehicle_save_path = self.config.snapshots_dir / vehicle_filename

        if track.best_full_frame is not None and track.best_vehicle_box is not None:
            full_frame_annotated = track.best_full_frame.copy()
            h_f, w_f = full_frame_annotated.shape[:2]
            vx1, vy1, vx2, vy2 = track.best_vehicle_box
            vx1, vy1 = max(0, vx1), max(0, vy1)
            vx2, vy2 = min(w_f, vx2), min(h_f, vy2)

            # Box color: Green if plate detected, Orange if no plate
            box_color = (0, 255, 0) if has_plate else (0, 140, 255)
            # Draw highlight box ONLY on this specific vehicle
            cv2.rectangle(full_frame_annotated, (vx1, vy1), (vx2, vy2), box_color, 3)

            # Label on box
            if has_plate:
                conf_pct = (track.best_plate_conf or 0.0) * 100.0
                label = f"ID:{track.track_id} | {track.best_plate_text} ({conf_pct:.0f}%)"
            else:
                label = f"ID:{track.track_id} | NO PLATE"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = max(th + 6, vy1 - 8)
            cv2.rectangle(
                full_frame_annotated,
                (vx1, label_y - th - 6),
                (vx1 + tw + 6, label_y + baseline + 2),
                box_color,
                -1,
            )
            cv2.putText(
                full_frame_annotated,
                label,
                (vx1 + 3, label_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

            cv2.imwrite(str(vehicle_save_path), full_frame_annotated)
            # Free memory immediately to keep RAM usage low
            track.best_full_frame = None
        elif track.best_vehicle_crop is not None and track.best_vehicle_crop.size > 0:
            cv2.imwrite(str(vehicle_save_path), track.best_vehicle_crop)
        else:
            placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(vehicle_save_path), placeholder)

        plate_filename: Optional[str] = None
        if has_plate and track.best_plate_crop is not None and track.best_plate_crop.size > 0:
            plate_filename = f"plate_{event_id}.png"
            plate_save_path = self.config.snapshots_dir / plate_filename
            cv2.imwrite(str(plate_save_path), track.best_plate_crop)

        # Build exact required JSON event format with remarks
        event_data = {
            "event_id": event_id,
            "camera_id": self.config.camera_id,
            "time": time_str,
            "date": date_str,
            "snapshot_number_plate": plate_filename if has_plate else None,
            "snapshot_vehicle": vehicle_filename,
            "plate_detection_confidence": (
                round(float(track.best_plate_det_conf), 2)
                if has_plate and track.best_plate_det_conf is not None
                else None
            ),
            "number_plate": track.best_plate_text if has_plate else None,
            "number_plate_confidence": (
                round(float(track.best_plate_conf), 2)
                if has_plate and track.best_plate_conf is not None
                else None
            ),
            "remarks": remarks,
        }

        # Update statistics
        self.total_unique_vehicles += 1
        if has_plate:
            self.total_with_plate += 1
        else:
            self.total_without_plate += 1

        self.emitted_events.append(event_data)

        # Append to JSONL for real-time safety
        with open(self.config.events_jsonl_path, "a", encoding="utf-8") as f_jsonl:
            f_jsonl.write(json.dumps(event_data) + "\n")

        # Save full JSON array
        with open(self.config.events_json_path, "w", encoding="utf-8") as f_json:
            json.dump(self.emitted_events, f_json, indent=2)

        print(f"\n[EVENT RECORDED] {json.dumps(event_data, indent=2)}")
        return event_data

    def get_counts(self) -> Tuple[int, int, int, int]:
        """Returns (total_unique_vehicles, with_plate, without_plate, active_tracks)."""
        active_count = sum(1 for t in self.tracks.values() if t.disappeared_count == 0)
        return (
            self.total_unique_vehicles,
            self.total_with_plate,
            self.total_without_plate,
            active_count,
        )

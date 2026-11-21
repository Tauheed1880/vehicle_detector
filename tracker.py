"""
Track state management, box deduplication, plate voting, and single-event emission.
Phase 1 Enhancements:
1. Global Unique Track IDs: {camera_id}:{track_id}.
2. Class-agnostic NMS and Nested / Containment box suppression.
3. Readable Zone (ROI) & Distance Gate (Far-away horizon vehicles are completely ignored).
4. Incomplete plate filter (length + regex) and character-level confidence-weighted voting.
5. Fuzzy duplicate suppression (Levenshtein distance <= 2 within 60s).
6. Configurable 'remarks' field (default off as requested).
"""

import json
import random
import re
import string
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from config import PipelineConfig


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings in pure Python."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def int_to_alphabets(n: int) -> str:
    """Bijective base-26 numbering (1 -> A, 2 -> B... 26 -> Z, 27 -> AA, 28 -> AB...)."""
    res = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        res.append(chr(65 + rem))
    return "".join(reversed(res)) if res else "A"


def get_camera_number(camera_id_str: str) -> str:
    """Extracts numeric camera number from camera_id (e.g. 'CAM-00001' -> '1')."""
    match = re.search(r"\d+", str(camera_id_str))
    if match:
        cleaned = match.group(0).lstrip("0")
        return cleaned if cleaned else "1"
    return "1"


def generate_infinite_event_id(counter: int, camera_id_str: str, now: datetime) -> str:
    """Generates unique event ID formatted as: evt_id_{yymmdd}_{camera_num}_{alpha_seq}_{random_letters}."""
    yymmdd = now.strftime("%y%m%d")
    cam_num = get_camera_number(camera_id_str)
    alpha_seq = int_to_alphabets(counter)
    rand_alpha = "".join(random.choices(string.ascii_uppercase, k=3))
    return f"evt_id_{yymmdd}_{cam_num}_{alpha_seq}_{rand_alpha}"


def get_image_sharpness(img: Optional[np.ndarray]) -> float:
    """Calculates image sharpness score using Laplacian variance."""
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def filter_nested_boxes(
    boxes: List[List[float]],
    scores: List[float],
    containment_thresh: float = 0.75,
) -> Tuple[List[List[float]], List[float]]:
    """
    Suppresses duplicate / nested bounding boxes where one box is almost entirely
    contained within another (e.g. car detected twice or sub-part detected as separate box).
    """
    if len(boxes) <= 1:
        return boxes, scores

    keep_indices: Set[int] = set(range(len(boxes)))

    for i in range(len(boxes)):
        if i not in keep_indices:
            continue
        ax1, ay1, ax2, ay2 = boxes[i]
        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        if area_a <= 0:
            keep_indices.discard(i)
            continue

        for j in range(len(boxes)):
            if i == j or j not in keep_indices:
                continue
            bx1, by1, bx2, by2 = boxes[j]
            area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
            if area_b <= 0:
                continue

            # Compute intersection area
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            intersection = iw * ih

            # Check if Box A is substantially inside Box B
            containment_a = intersection / area_a
            containment_b = intersection / area_b

            if containment_a >= containment_thresh or containment_b >= containment_thresh:
                # If one is inside the other, keep the one with higher confidence score
                # or larger area if scores are very close
                if scores[i] < scores[j]:
                    keep_indices.discard(i)
                    break
                else:
                    keep_indices.discard(j)

    kept_boxes = [boxes[k] for k in sorted(keep_indices)]
    kept_scores = [scores[k] for k in sorted(keep_indices)]
    return kept_boxes, kept_scores


def character_level_voting(
    readings: List[Tuple[str, float]],
    min_length: int = 4,
    regex_pattern: Optional[str] = None,
    enforce_regex: bool = False,
) -> Optional[Tuple[str, float]]:
    """
    Performs character-level confidence-weighted frequency voting across multiple plate readings.
    Filters out flickering/incomplete reads (like 'DMV' or 'HNI4') and finds true consensus.
    """
    if not readings:
        return None

    valid_readings: List[Tuple[str, float]] = []
    regex = re.compile(regex_pattern) if regex_pattern else None

    for text, conf in readings:
        clean_text = re.sub(r"[^A-Z0-9]", "", text.upper())
        if len(clean_text) < min_length:
            continue
        if enforce_regex and regex and not regex.match(clean_text):
            continue
        valid_readings.append((clean_text, conf))

    if not valid_readings:
        # Fallback to longest available clean reading if strict filter leaves none
        for text, conf in readings:
            clean_text = re.sub(r"[^A-Z0-9]", "", text.upper())
            if len(clean_text) >= 3:
                valid_readings.append((clean_text, conf))
        if not valid_readings:
            return None

    # Group by length to find the most frequent plate length
    length_votes = defaultdict(float)
    for text, conf in valid_readings:
        length_votes[len(text)] += conf

    best_len = max(length_votes.keys(), key=lambda l: length_votes[l])
    same_len_readings = [r for r in valid_readings if len(r[0]) == best_len]

    # Character-level position-wise voting
    consensus_chars = []
    char_confs = []

    for pos in range(best_len):
        pos_votes = defaultdict(float)
        for text, conf in same_len_readings:
            char = text[pos]
            pos_votes[char] += conf

        # Pick character with highest accumulated confidence
        best_char = max(pos_votes.keys(), key=lambda c: pos_votes[c])
        consensus_chars.append(best_char)

        total_pos_weight = sum(pos_votes.values())
        winning_weight = pos_votes[best_char]
        char_confs.append(min(1.0, winning_weight / max(1e-5, len(same_len_readings))))

    consensus_text = "".join(consensus_chars)
    avg_confidence = float(np.mean(char_confs)) if char_confs else float(same_len_readings[0][1])
    return consensus_text, avg_confidence


@dataclass
class TrackState:
    track_id: int
    global_id: str
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
    # Readable Zone Gate
    entered_readable_zone: bool = False
    # Plate readings history: list of (text, confidence) for consensus voting
    plate_readings: List[Tuple[str, float]] = field(default_factory=list)
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
        # Store emitted plate -> (timestamp, event_id) for fuzzy duplicate filtering
        self.recent_emitted_plates: Dict[str, Tuple[float, str]] = {}
        self.event_counter: int = 0

        # Stats counters
        self.total_unique_vehicles: int = 0
        self.total_with_plate: int = 0
        self.total_without_plate: int = 0

    def is_in_roi(self, box: List[float], frame_shape: Tuple[int, int]) -> bool:
        """
        Determines if the vehicle box is inside the configured ROI polygon.
        If roi_polygon is None, checks if box is within the lower 85% of frame (readable region).
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        bottom_cx = (x1 + x2) / 2.0
        bottom_cy = y2

        if self.config.roi_polygon is not None and len(self.config.roi_polygon) >= 3:
            pts = np.array(self.config.roi_polygon, dtype=np.int32)
            dist = cv2.pointPolygonTest(pts, (float(bottom_cx), float(bottom_cy)), measureDist=False)
            return dist >= 0

        # Default: readable zone is any vehicle whose bottom is within 15% to 98% of the frame height
        # and has minimum box dimensions (filters out extreme horizon)
        return (0.15 * h <= bottom_cy <= 0.99 * h)

    def update_track(
        self,
        track_id: int,
        box: List[float],
        frame: np.ndarray,
        frame_idx: int,
    ) -> TrackState:
        """
        Updates an existing track or initializes a new track state.
        Enforces readable zone entry and box smoothing.
        """
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = max(0, x2 - x1), max(0, y2 - y1)
        area = float(bw * bh)

        in_roi = self.is_in_roi([float(x1), float(y1), float(x2), float(y2)], (h, w))
        is_readable_size = (
            bw >= self.config.min_vehicle_width
            and bh >= self.config.min_vehicle_height
            and area >= self.config.min_readable_vehicle_area
        )

        global_id = f"{self.config.camera_id}:{track_id}"

        # Crop vehicle with 5% padding
        pad_x = int(bw * 0.05)
        pad_y = int(bh * 0.05)
        vx1, vy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        vx2, vy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        vehicle_crop = frame[vy1:vy2, vx1:vx2].copy() if area > 0 else None

        if track_id not in self.tracks:
            track = TrackState(
                track_id=track_id,
                global_id=global_id,
                first_frame_idx=frame_idx,
                last_frame_idx=frame_idx,
                frames_tracked=1,
                disappeared_count=0,
                smoothed_box=[float(x1), float(y1), float(x2), float(y2)],
                best_vehicle_box=[x1, y1, x2, y2],
                best_full_frame=frame.copy() if area > 0 else None,
                best_vehicle_crop=vehicle_crop,
                best_vehicle_area=area,
                entered_readable_zone=(in_roi and is_readable_size),
            )
            self.tracks[track_id] = track
        else:
            track = self.tracks[track_id]
            track.last_frame_idx = frame_idx
            track.frames_tracked += 1
            track.disappeared_count = 0

            # Once vehicle enters readable zone with sufficient size, flag stays True
            if in_roi and is_readable_size:
                track.entered_readable_zone = True

            # EMA box smoothing
            alpha = self.config.box_smoothing_factor
            if track.smoothed_box is not None:
                track.smoothed_box = [
                    alpha * float(c) + (1.0 - alpha) * float(s)
                    for c, s in zip([x1, y1, x2, y2], track.smoothed_box)
                ]
            else:
                track.smoothed_box = [float(x1), float(y1), float(x2), float(y2)]

            # Keep full frame and crop with larger vehicle area before event is emitted
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
        Matches full-frame ALPR plate detections to vehicle tracks.
        Accumulates plate readings and performs character-level consensus voting.
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
                if track.has_event_emitted:
                    continue

                raw_text = str(ocr.text).strip().upper()
                clean_text = re.sub(r"[^A-Z0-9]", "", raw_text)

                ocr_conf = (
                    float(np.mean(ocr.confidence))
                    if isinstance(ocr.confidence, (list, tuple, np.ndarray))
                    else float(ocr.confidence)
                )
                plate_det_conf = float(det.confidence)

                # Plate length check: filter out incomplete reads (like 'DMV', 'HNI4')
                if len(clean_text) < self.config.min_plate_length:
                    continue

                # Crop plate image safely
                c_px1, c_py1 = max(0, int(px1)), max(0, int(py1))
                c_px2, c_py2 = min(w, int(px2)), min(h, int(py2))
                plate_crop = frame[c_py1:c_py2, c_px1:c_px2].copy()

                # Smart blur filter: check sharpness score
                sharpness = get_image_sharpness(plate_crop)
                if sharpness < self.config.min_plate_sharpness and track.best_plate_crop is not None:
                    continue

                # Add reading to track history for consensus voting
                track.plate_readings.append((clean_text, ocr_conf))

                # Update best crop and frame
                is_better = (track.best_plate_conf is None) or (ocr_conf > track.best_plate_conf)
                if is_better:
                    track.best_plate_crop = plate_crop
                    track.best_plate_det_conf = plate_det_conf
                    track.best_full_frame = frame.copy()
                    if track.smoothed_box is not None:
                        track.best_vehicle_box = [int(v) for v in track.smoothed_box]

                # Run consensus voting across accumulated readings
                consensus_result = character_level_voting(
                    readings=track.plate_readings,
                    min_length=self.config.min_plate_length,
                    regex_pattern=self.config.plate_regex_pattern,
                    enforce_regex=self.config.enforce_regex,
                )

                if consensus_result is not None:
                    voted_text, voted_conf = consensus_result
                    track.best_plate_text = voted_text
                    track.best_plate_conf = voted_conf

                    # Confident condition requires minimum number of consistent reads
                    if (
                        voted_conf >= self.config.min_confident_plate_conf
                        and len(track.plate_readings) >= self.config.min_plate_reads_for_event
                    ):
                        track.plate_confident_reached = True

    def check_and_emit_events(
        self,
        current_frame_idx: int,
        force_all: bool = False,
    ) -> List[dict]:
        """
        Evaluates active and disappeared tracks.
        Enforces STRICT single-event rule:
          1. Far-away horizon vehicles that never entered readable zone are DROPPED (no count, no event).
          2. Exactly 1 event per valid vehicle track.
          3. Fuzzy duplicate suppression (Levenshtein distance <= 2 within 60s).
        """
        emitted_now: List[dict] = []
        now_ts = datetime.now().timestamp()

        for t_id, track in list(self.tracks.items()):
            if track.has_event_emitted or (t_id in self.emitted_track_ids):
                continue

            # -------------------------------------------------------------
            # Distance / Zone Gate: Far-away small cars outside readable zone
            # must be ignored completely (neither counted nor emitted).
            # -------------------------------------------------------------
            if not track.entered_readable_zone and track.best_plate_text is None:
                if track.disappeared_count >= self.config.max_disappeared_frames or force_all:
                    del self.tracks[t_id]
                continue

            # Minimum track frames filter to eliminate false-positive flickers
            if track.frames_tracked < self.config.min_track_frames:
                if force_all:
                    del self.tracks[t_id]
                    continue
                if track.disappeared_count >= self.config.max_disappeared_frames:
                    del self.tracks[t_id]
                continue

            should_emit = False

            # Condition 1: Plate confidently resolved with character-level consensus
            if track.plate_confident_reached and track.best_plate_text:
                should_emit = True

            # Condition 2: Vehicle exited the readable zone / frame
            elif track.disappeared_count >= self.config.max_disappeared_frames or force_all:
                should_emit = True

            if should_emit:
                track.has_event_emitted = True
                self.emitted_track_ids.add(t_id)

                event = self._emit_event(track, now_ts)
                if event is not None:
                    emitted_now.append(event)

        return emitted_now

    def _emit_event(self, track: TrackState, current_timestamp: float) -> Optional[dict]:
        """
        Formats single event JSON and saves best frame snapshots.
        Applies fuzzy duplicate suppression (Levenshtein distance <= 2 within 60s).
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        has_plate = track.best_plate_text is not None and track.best_plate_conf is not None

        # -------------------------------------------------------------
        # Fuzzy Duplicate Suppression (Levenshtein distance <= 2 within 60s)
        # Prevents ID churn from creating duplicate events for the same vehicle
        # -------------------------------------------------------------
        if has_plate:
            plate_candidate = track.best_plate_text
            for existing_plate, (last_seen_ts, prev_event_id) in list(self.recent_emitted_plates.items()):
                if (current_timestamp - last_seen_ts) <= self.config.duplicate_plate_window_sec:
                    dist = levenshtein_distance(plate_candidate, existing_plate)
                    if dist <= self.config.fuzzy_plate_max_distance:
                        print(
                            f"[FUZZY DUPLICATE SUPPRESSED] Plate '{plate_candidate}' is within "
                            f"edit distance {dist} of recently recorded '{existing_plate}' "
                            f"(Event: {prev_event_id}). Suppressing new event."
                        )
                        return None

            # Register plate in recent memory
            self.recent_emitted_plates[plate_candidate] = (current_timestamp, f"EVT_{self.event_counter + 1}")

        self.event_counter += 1
        event_id = generate_infinite_event_id(self.event_counter, self.config.camera_id, now)
        if has_plate:
            self.recent_emitted_plates[track.best_plate_text] = (current_timestamp, event_id)

        # -------------------------------------------------------------
        # Snapshot Generation
        # Full-size context frame with highlight box ONLY on target vehicle
        # -------------------------------------------------------------
        vehicle_filename = f"vehicle_{event_id}.png"
        vehicle_save_path = self.config.snapshots_dir / vehicle_filename

        if track.best_full_frame is not None and track.best_vehicle_box is not None:
            full_frame_annotated = track.best_full_frame.copy()
            h_f, w_f = full_frame_annotated.shape[:2]
            vx1, vy1, vx2, vy2 = track.best_vehicle_box
            vx1, vy1 = max(0, vx1), max(0, vy1)
            vx2, vy2 = min(w_f, vx2), min(h_f, vy2)

            box_color = (0, 255, 0) if has_plate else (0, 140, 255)
            cv2.rectangle(full_frame_annotated, (vx1, vy1), (vx2, vy2), box_color, 3)

            label = (
                f"{track.global_id} | {track.best_plate_text}"
                if has_plate
                else f"{track.global_id} | NO PLATE"
            )

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
            track.best_full_frame = None  # Free memory immediately
        elif track.best_vehicle_crop is not None and track.best_vehicle_crop.size > 0:
            cv2.imwrite(str(vehicle_save_path), track.best_vehicle_crop)
        else:
            placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(vehicle_save_path), placeholder)

        # Plate cropped snapshot
        plate_filename: Optional[str] = None
        if has_plate and track.best_plate_crop is not None and track.best_plate_crop.size > 0:
            plate_filename = f"plate_{event_id}.png"
            plate_save_path = self.config.snapshots_dir / plate_filename
            cv2.imwrite(str(plate_save_path), track.best_plate_crop)

        # -------------------------------------------------------------
        # Exact JSON Event Output Schema
        # -------------------------------------------------------------
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
        }

        # Optional remarks field (only included if config.include_remarks is True)
        if self.config.include_remarks:
            if has_plate:
                if track.best_plate_conf >= self.config.min_confident_plate_conf:
                    event_data["remarks"] = "Number plate successfully detected and recognized"
                else:
                    event_data["remarks"] = "Low confidence / faint characters detected on plate"
            else:
                event_data["remarks"] = "No number plate detected on vehicle"

        # Update statistics
        self.total_unique_vehicles += 1
        if has_plate:
            self.total_with_plate += 1
        else:
            self.total_without_plate += 1

        self.emitted_events.append(event_data)

        # Append to JSONL for crash safety
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

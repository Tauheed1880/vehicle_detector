"""
Track state management, box deduplication, plate voting, and single-event emission.
Phase 1 Enterprise Enhancements:
1. Stage 2 (Analyzer) Event Interface & Queue.
2. Confidence-squared ($conf^2$) weighted character voting with UK syntax correction.
3. Fast Async Snapshot Saver (Level 1 lossless PNG compression for 4x disk speed).
4. Spatial Jump & ID Switch Guard (prevents identity bleed).
5. Clean console output (events written to JSON without terminal spam).
"""

import json
import math
import queue
import random
import re
import string
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from config import PipelineConfig


class Stage2EventInterface(ABC):
    """
    Abstract interface for downstream Stage 2 Analyzers
    (Vehicle Make/Model/Color, Speed estimation, Driver face, Blacklist lookup).
    """

    @abstractmethod
    def on_event(
        self,
        event_id: str,
        camera_id: str,
        number_plate: Optional[str],
        vehicle_snapshot_path: Path,
        plate_snapshot_path: Optional[Path],
        event_data: dict,
    ) -> None:
        """Called automatically whenever a new verified vehicle event is finalized."""
        pass


class AsyncSnapshotSaver:
    """Non-blocking background worker for saving snapshots with fast lossless compression."""

    def __init__(self):
        self.queue: queue.Queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, path: Path, image: np.ndarray) -> None:
        if self.running and image is not None and image.size > 0:
            self.queue.put((str(path), image))

    def _worker(self) -> None:
        # Fast PNG compression parameters (Level 1: 4x faster, 100% lossless)
        png_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]

        while self.running or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            path_str, img = item
            try:
                cv2.imwrite(path_str, img, png_params)
            except Exception as e:
                print(f"[SNAPSHOT SAVE ERROR] Could not save {path_str}: {e}")
            finally:
                self.queue.task_done()

    def stop(self) -> None:
        self.running = False
        self.queue.join()


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
    containment_thresh: float = 0.70,
) -> Tuple[List[List[float]], List[float]]:
    """
    Suppresses duplicate / nested bounding boxes where one box is substantially
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

            containment_a = intersection / area_a
            containment_b = intersection / area_b

            if containment_a >= containment_thresh or containment_b >= containment_thresh:
                if scores[i] < scores[j]:
                    keep_indices.discard(i)
                    break
                else:
                    keep_indices.discard(j)

    kept_boxes = [boxes[k] for k in sorted(keep_indices)]
    kept_scores = [scores[k] for k in sorted(keep_indices)]
    return kept_boxes, kept_scores


def normalize_pakistan_plate_syntax(plate: str) -> str:
    """
    Normalizes standard Pakistani license plate formats (e.g. LEA-18-5001, ICT-VS-221, MNP-123, FD-1234).
    Resolves optical character confusion between digits and letters based on Pakistani prefix/suffix layout:
    - Leading 2 to 4 characters are alphabetic jurisdiction codes (e.g. LEA, ICT, LHR, KHI, FD, MNP, ISB).
    - Trailing characters are predominantly digits.
    """
    clean = re.sub(r"[^A-Z0-9]", "", plate.upper())
    if len(clean) < 3:
        return clean

    digit_to_letter = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}
    letter_to_digit = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"}

    chars = list(clean)

    # Pakistani prefixes: check if first 2-3 characters look like digits mistaken for letters
    for i in range(min(2, len(chars))):
        if chars[i] in digit_to_letter and any(c.isalpha() for c in chars[:3]):
            chars[i] = digit_to_letter[chars[i]]

    # Trailing part: if the tail (last 2-4 characters) is numeric, fix letter confusions
    for i in range(max(2, len(chars) - 4), len(chars)):
        if chars[i] in letter_to_digit and any(c.isdigit() for c in chars[-3:]):
            chars[i] = letter_to_digit[chars[i]]

    return "".join(chars)


def normalize_uk_plate_syntax(plate: str) -> str:
    """
    Normalizes standard UK plate format (2 letters, 2 digits, 3 letters e.g. AB12CDE).
    Resolves optical character confusion (0 vs O, 1 vs I, 5 vs S, 8 vs B).
    """
    if len(plate) != 7:
        return plate

    digit_to_letter = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}
    letter_to_digit = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"}

    chars = list(plate)
    # Positions 0 and 1: Must be letters
    for i in (0, 1):
        if chars[i] in digit_to_letter:
            chars[i] = digit_to_letter[chars[i]]

    # Positions 2 and 3: Must be digits
    for i in (2, 3):
        if chars[i] in letter_to_digit:
            chars[i] = letter_to_digit[chars[i]]

    # Positions 4, 5, 6: Must be letters
    for i in (4, 5, 6):
        if chars[i] in digit_to_letter:
            chars[i] = digit_to_letter[chars[i]]

    return "".join(chars)


def character_level_voting(
    readings: List[Tuple[str, float]],
    min_length: int = 4,
    regex_pattern: Optional[str] = None,
    uk_syntax_correction: bool = False,
    plate_format: str = "PAKISTAN",
) -> Optional[Tuple[str, float]]:
    """
    Performs confidence-squared ($conf^2$) weighted character voting across multiple plate readings.
    High confidence reads dominate, eliminating flickering run-to-run discrepancies.
    Supports Pakistani, UK, and General plate syntax.
    """
    if not readings:
        return None

    valid_readings: List[Tuple[str, float]] = []
    regex = re.compile(regex_pattern) if regex_pattern else None

    for text, conf in readings:
        clean_text = re.sub(r"[^A-Z0-9]", "", text.upper())
        if plate_format == "UK" and uk_syntax_correction and len(clean_text) == 7:
            clean_text = normalize_uk_plate_syntax(clean_text)
        elif plate_format == "PAKISTAN":
            clean_text = normalize_pakistan_plate_syntax(clean_text)

        if len(clean_text) < min_length:
            continue
        if regex and not regex.match(clean_text):
            continue
        valid_readings.append((clean_text, conf))

    if not valid_readings:
        return None

    # Group by length to find majority plate length
    length_votes = defaultdict(float)
    for text, conf in valid_readings:
        length_votes[len(text)] += (conf ** 2)

    best_len = max(length_votes.keys(), key=lambda l: length_votes[l])
    same_len_readings = [r for r in valid_readings if len(r[0]) == best_len]

    consensus_chars = []
    char_confs = []

    for pos in range(best_len):
        pos_votes = defaultdict(float)
        # Extra priority for leading area code letters (pos 0 and 1)
        pos_multiplier = 1.35 if pos in (0, 1) else 1.0

        for text, conf in same_len_readings:
            char = text[pos]
            weight = (conf ** 2) * pos_multiplier
            pos_votes[char] += weight

        best_char = max(pos_votes.keys(), key=lambda c: pos_votes[c])
        consensus_chars.append(best_char)

        total_weight = sum(pos_votes.values())
        winning_weight = pos_votes[best_char]
        char_confs.append(min(1.0, math.sqrt(winning_weight / max(1e-5, total_weight))))

    raw_consensus = "".join(consensus_chars)
    if plate_format == "UK" and uk_syntax_correction and len(raw_consensus) == 7:
        raw_consensus = normalize_uk_plate_syntax(raw_consensus)
    elif plate_format == "PAKISTAN":
        raw_consensus = normalize_pakistan_plate_syntax(raw_consensus)

    avg_confidence = float(np.mean(char_confs)) if char_confs else float(same_len_readings[0][1])
    return raw_consensus, avg_confidence


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
    # Previous box center for spatial jump tracking
    last_center: Optional[Tuple[float, float]] = None
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
        self.emitted_track_ids: Set[int] = set()
        self.recent_emitted_plates: Dict[str, Tuple[float, str]] = {}
        self.event_counter: int = 0

        # Async background snapshot saver
        self.snapshot_saver = AsyncSnapshotSaver()

        # Stage 2 (Analyzer) downstream interface queue & handlers
        self.stage2_queue: queue.Queue = queue.Queue(maxsize=self.config.stage2_queue_maxsize)
        self.stage2_handlers: List[Stage2EventInterface] = []

        # Stats counters
        self.total_unique_vehicles: int = 0
        self.total_with_plate: int = 0
        self.total_without_plate: int = 0
        self.total_filtered_too_small: int = 0

    def register_stage2_handler(self, handler: Stage2EventInterface) -> None:
        """Registers a downstream Stage 2 analyzer callback."""
        self.stage2_handlers.append(handler)

    def is_in_roi(self, box: List[float], frame_shape: Tuple[int, int]) -> bool:
        """
        Determines if the vehicle box is inside the readable zone.
        Excludes the upper horizon (top 32% of frame) where vehicles are too far away.
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        bottom_cx = (x1 + x2) / 2.0
        bottom_cy = y2

        if self.config.roi_polygon is not None and len(self.config.roi_polygon) >= 3:
            pts = np.array(self.config.roi_polygon, dtype=np.int32)
            dist = cv2.pointPolygonTest(pts, (float(bottom_cx), float(bottom_cy)), measureDist=False)
            return dist >= 0

        horizon_y = self.config.roi_top_cutoff_ratio * h
        return (bottom_cy >= horizon_y) and (y1 >= horizon_y * 0.7)

    def update_track(
        self,
        track_id: int,
        box: List[float],
        frame: np.ndarray,
        frame_idx: int,
    ) -> TrackState:
        """
        Updates an existing track or initializes a new track state.
        Detects spatial jumps (ID switches) and enforces the readable zone gate.
        """
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = max(0, x2 - x1), max(0, y2 - y1)
        area = float(bw * bh)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

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
                last_center=(cx, cy),
                best_vehicle_box=[x1, y1, x2, y2],
                best_full_frame=frame.copy() if area > 0 else None,
                best_vehicle_crop=vehicle_crop,
                best_vehicle_area=area,
                entered_readable_zone=(in_roi and is_readable_size),
            )
            self.tracks[track_id] = track
        else:
            track = self.tracks[track_id]

            # Spatial Jump Check: Detect ID switches / tracker teleportation
            if track.last_center is not None and track.frames_tracked > 2:
                prev_cx, prev_cy = track.last_center
                jump_dist = math.hypot(cx - prev_cx, cy - prev_cy)
                if jump_dist > self.config.max_spatial_jump_px:
                    track.plate_readings.clear()
                    track.best_plate_text = None
                    track.best_plate_conf = None
                    track.plate_confident_reached = False
                    track.best_vehicle_area = area

            track.last_center = (cx, cy)
            track.last_frame_idx = frame_idx
            track.frames_tracked += 1
            track.disappeared_count = 0

            if in_roi and is_readable_size:
                track.entered_readable_zone = True

            alpha = self.config.box_smoothing_factor
            if track.smoothed_box is not None:
                track.smoothed_box = [
                    alpha * float(c) + (1.0 - alpha) * float(s)
                    for c, s in zip([x1, y1, x2, y2], track.smoothed_box)
                ]
            else:
                track.smoothed_box = [float(x1), float(y1), float(x2), float(y2)]

            if area > track.best_vehicle_area and area > 0:
                track.best_vehicle_area = area
                track.best_vehicle_box = [x1, y1, x2, y2]
                if not track.has_event_emitted:
                    track.best_full_frame = frame.copy()
                    track.best_vehicle_crop = vehicle_crop

        return self.tracks[track_id]

    def mark_disappeared_tracks(self, active_track_ids: set[int]) -> None:
        """Increments disappeared count for tracks absent from current frame."""
        for t_id, track in self.tracks.items():
            if t_id not in active_track_ids:
                track.disappeared_count += 1

    def update_track_plate(
        self,
        track_id: int,
        raw_text: str,
        ocr_conf: float,
        det_conf: float,
        plate_crop: np.ndarray,
        frame: np.ndarray,
    ) -> None:
        """
        Directly associates a plate prediction (from vehicle crop or full frame) to a track.
        Enforces plate mutability lock (prevents ID 19 plate jump) and consensus voting.
        """
        track = self.tracks.get(track_id)
        if track is None or track.has_event_emitted:
            return

        clean_text = re.sub(r"[^A-Z0-9]", "", str(raw_text).strip().upper())
        if self.config.plate_format == "UK" and self.config.uk_syntax_correction and len(clean_text) == 7:
            clean_text = normalize_uk_plate_syntax(clean_text)
        elif self.config.plate_format == "PAKISTAN":
            clean_text = normalize_pakistan_plate_syntax(clean_text)

        if len(clean_text) < self.config.min_plate_length:
            return

        # Plate Mutability Lock: Do not let a conflicting plate overwrite established text
        if track.plate_confident_reached and track.best_plate_text:
            dist = levenshtein_distance(clean_text, track.best_plate_text)
            if dist > 3:
                return

        sharpness = get_image_sharpness(plate_crop)
        if sharpness < self.config.min_plate_sharpness and track.best_plate_crop is not None:
            return

        track.plate_readings.append((clean_text, ocr_conf))

        is_better = (track.best_plate_conf is None) or (ocr_conf > track.best_plate_conf)
        if is_better:
            track.best_plate_crop = plate_crop.copy() if plate_crop is not None else None
            track.best_plate_det_conf = det_conf
            track.best_full_frame = frame.copy()
            if track.smoothed_box is not None:
                track.best_vehicle_box = [int(v) for v in track.smoothed_box]

        consensus_result = character_level_voting(
            readings=track.plate_readings,
            min_length=self.config.min_plate_length,
            regex_pattern=self.config.plate_regex_pattern if self.config.enforce_regex else None,
            uk_syntax_correction=self.config.uk_syntax_correction,
            plate_format=self.config.plate_format,
        )

        if consensus_result is not None:
            voted_text, voted_conf = consensus_result
            track.best_plate_text = voted_text
            track.best_plate_conf = voted_conf

            if (
                voted_conf >= self.config.min_confident_plate_conf
                and len(track.plate_readings) >= self.config.min_plate_reads_for_event
            ):
                track.plate_confident_reached = True

    def match_and_update_plates(
        self,
        alpr_results: list,
        frame: np.ndarray,
        frame_idx: int,
    ) -> None:
        """Matches full-frame ALPR plate detections to vehicle tracks."""
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
                c_px1, c_py1 = max(0, int(px1)), max(0, int(py1))
                c_px2, c_py2 = min(w, int(px2)), min(h, int(py2))
                plate_crop = frame[c_py1:c_py2, c_px1:c_px2].copy()

                ocr_conf = (
                    float(np.mean(ocr.confidence))
                    if isinstance(ocr.confidence, (list, tuple, np.ndarray))
                    else float(ocr.confidence)
                )
                self.update_track_plate(
                    track_id=best_match_id,
                    raw_text=str(ocr.text),
                    ocr_conf=ocr_conf,
                    det_conf=float(det.confidence),
                    plate_crop=plate_crop,
                    frame=frame,
                )

    def check_and_emit_events(
        self,
        current_frame_idx: int,
        force_all: bool = False,
    ) -> List[dict]:
        """Evaluates active and disappeared tracks. Enforces single-event and flush criteria."""
        emitted_now: List[dict] = []
        now_ts = datetime.now().timestamp()

        for t_id, track in list(self.tracks.items()):
            if track.has_event_emitted or (t_id in self.emitted_track_ids):
                continue

            # Distance / Zone Gate: horizon vehicles dropped without event
            if not track.entered_readable_zone:
                if track.disappeared_count >= self.config.max_disappeared_frames or force_all:
                    self.total_filtered_too_small += 1
                    del self.tracks[t_id]
                continue

            # Minimum Track Frames Filter
            if track.frames_tracked < self.config.min_track_frames:
                if force_all or (track.disappeared_count >= self.config.max_disappeared_frames):
                    self.total_filtered_too_small += 1
                    del self.tracks[t_id]
                continue

            should_emit = False

            if track.plate_confident_reached and track.best_plate_text:
                should_emit = True
            elif track.disappeared_count >= self.config.max_disappeared_frames:
                should_emit = True
            elif force_all and track.entered_readable_zone and track.frames_tracked >= self.config.min_track_frames:
                should_emit = True

            if should_emit:
                track.has_event_emitted = True
                self.emitted_track_ids.add(t_id)

                event = self._emit_event(track, now_ts)
                if event is not None:
                    emitted_now.append(event)

        return emitted_now

    def _emit_event(self, track: TrackState, current_timestamp: float) -> Optional[dict]:
        """Formats single event JSON, submits async snapshots, and dispatches to Stage 2."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        has_plate = track.best_plate_text is not None and track.best_plate_conf is not None

        # Partial Plate Policy: Reject incomplete or invalid plate reads
        if has_plate:
            plate_candidate = track.best_plate_text
            is_valid = False

            if self.config.plate_format == "UK":
                uk_regex = re.compile(self.config.plate_regex_pattern or r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")
                is_valid = bool(uk_regex.match(plate_candidate)) and len(plate_candidate) >= self.config.min_plate_length
            elif self.config.plate_format == "PAKISTAN":
                # Pakistani plates: 3 to 10 alphanumeric characters (e.g. ICT123, LEA185001, MNP123, FD1234)
                has_letter = any(c.isalpha() for c in plate_candidate)
                has_digit = any(c.isdigit() for c in plate_candidate)
                is_valid = (
                    len(plate_candidate) >= self.config.min_plate_length
                    and len(plate_candidate) <= 10
                    and (has_letter or has_digit)
                )
                if self.config.enforce_regex and self.config.plate_regex_pattern:
                    pk_regex = re.compile(self.config.plate_regex_pattern)
                    is_valid = is_valid and bool(pk_regex.match(plate_candidate))
            else:  # GENERAL format
                is_valid = len(plate_candidate) >= self.config.min_plate_length
                if self.config.enforce_regex and self.config.plate_regex_pattern:
                    gen_regex = re.compile(self.config.plate_regex_pattern)
                    is_valid = is_valid and bool(gen_regex.match(plate_candidate))

            if not is_valid:
                if self.config.partial_plate_policy == "drop":
                    has_plate = False
                    track.best_plate_text = None
                    track.best_plate_conf = None
                    track.best_plate_det_conf = None
                    track.best_plate_crop = None
                elif self.config.partial_plate_policy == "keep_low_conf":
                    track.best_plate_conf = min(0.35, float(track.best_plate_conf or 0.35))

        # Fuzzy Duplicate Suppression
        if has_plate:
            plate_candidate = track.best_plate_text
            for existing_plate, (last_seen_ts, prev_event_id) in list(self.recent_emitted_plates.items()):
                if (current_timestamp - last_seen_ts) <= self.config.duplicate_plate_window_sec:
                    dist = levenshtein_distance(plate_candidate, existing_plate)
                    if dist <= self.config.fuzzy_plate_max_distance:
                        return None

            self.recent_emitted_plates[plate_candidate] = (current_timestamp, f"EVT_{self.event_counter + 1}")

        self.event_counter += 1
        event_id = generate_infinite_event_id(self.event_counter, self.config.camera_id, now)
        if has_plate:
            self.recent_emitted_plates[track.best_plate_text] = (current_timestamp, event_id)

        # Full-size vehicle snapshot with box ONLY on target vehicle
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

            self.snapshot_saver.submit(vehicle_save_path, full_frame_annotated)
            track.best_full_frame = None
        elif track.best_vehicle_crop is not None and track.best_vehicle_crop.size > 0:
            self.snapshot_saver.submit(vehicle_save_path, track.best_vehicle_crop)
        else:
            placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
            self.snapshot_saver.submit(vehicle_save_path, placeholder)

        # Cropped Plate snapshot
        plate_filename: Optional[str] = None
        plate_save_path: Optional[Path] = None
        if has_plate and track.best_plate_crop is not None and track.best_plate_crop.size > 0:
            plate_filename = f"plate_{event_id}.png"
            plate_save_path = self.config.snapshots_dir / plate_filename
            self.snapshot_saver.submit(plate_save_path, track.best_plate_crop)

        # JSON Event Output
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

        # Write to JSONL
        with open(self.config.events_jsonl_path, "a", encoding="utf-8") as f_jsonl:
            f_jsonl.write(json.dumps(event_data) + "\n")

        # Rewrite full JSON
        with open(self.config.events_json_path, "w", encoding="utf-8") as f_json:
            json.dump(self.emitted_events, f_json, indent=2)

        # Console print only if explicitly requested
        if self.config.print_events_to_terminal:
            print(f"\n[EVENT RECORDED] {json.dumps(event_data, indent=2)}")

        # -------------------------------------------------------------
        # Stage 2 (Analyzer) Dispatch: Submit package to queue & handlers
        # -------------------------------------------------------------
        if self.config.enable_stage2_interface:
            stage2_pkg = {
                "event_id": event_id,
                "camera_id": self.config.camera_id,
                "number_plate": track.best_plate_text if has_plate else None,
                "vehicle_snapshot_path": vehicle_save_path,
                "plate_snapshot_path": plate_save_path,
                "event_data": event_data,
            }
            if not self.stage2_queue.full():
                try:
                    self.stage2_queue.put_nowait(stage2_pkg)
                except queue.Full:
                    pass

            for handler in self.stage2_handlers:
                try:
                    handler.on_event(
                        event_id=event_id,
                        camera_id=self.config.camera_id,
                        number_plate=track.best_plate_text if has_plate else None,
                        vehicle_snapshot_path=vehicle_save_path,
                        plate_snapshot_path=plate_save_path,
                        event_data=event_data,
                    )
                except Exception as e:
                    print(f"[STAGE 2 HANDLER ERROR] {e}")

        return event_data

    def get_counts(self) -> Tuple[int, int, int, int, int]:
        """Returns (total_unique, with_plate, without_plate, active, filtered_too_small)."""
        active_count = sum(1 for t in self.tracks.values() if t.disappeared_count == 0)
        return (
            self.total_unique_vehicles,
            self.total_with_plate,
            self.total_without_plate,
            active_count,
            self.total_filtered_too_small,
        )

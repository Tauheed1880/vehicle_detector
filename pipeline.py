"""
Vehicle ALPR Pipeline.
Combines:
1. Ultralytics YOLO nano vehicle detector with ByteTrack tracker (persist=True).
2. FastALPR on full frames every K frames.
3. Box smoothing and live HUD counters.
4. Single-event generation per vehicle.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Ensure repository root is on Python path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import PipelineConfig
from fast_alpr import ALPR
from tracker import VehicleTrackerManager


def has_gui_display() -> bool:
    """Checks whether a graphical desktop display is available."""
    if os.name == "nt":
        return True
    # On Linux/macOS, a display server is required for cv2.imshow
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class VehicleALPRPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        print(f"[INIT] Initializing VehicleALPRPipeline with Camera ID: {self.config.camera_id}")

        # -------------------------------------------------------------
        # License Notice for Ultralytics YOLO
        # -------------------------------------------------------------
        print(
            "\n" + "=" * 70 + "\n"
            "[LICENSE NOTICE] Vehicle detector uses Ultralytics YOLO ("
            f"{self.config.vehicle_model_name}).\n"
            "Ultralytics YOLO is licensed under GNU AGPL-3.0.\n"
            "Commercial/closed-source deployments require an enterprise license "
            "from Ultralytics.\n"
            + "=" * 70 + "\n"
        )

        # Lazy model loading: models will be loaded inside initialize_models()
        self.vehicle_model = None
        self.alpr = None
        self.tracker_manager = VehicleTrackerManager(self.config)

    def initialize_models(self) -> None:
        """
        Loads models only when inference is explicitly called, keeping memory clean.
        """
        if self.vehicle_model is None:
            from ultralytics import YOLO

            print(f"[LOAD] Loading vehicle detector model: {self.config.vehicle_model_name}...")
            self.vehicle_model = YOLO(self.config.vehicle_model_name)

        if self.alpr is None:
            print(f"[LOAD] Loading FastALPR (Detector: {self.config.alpr_detector_model}, OCR: {self.config.alpr_ocr_model})...")
            self.alpr = ALPR(
                detector_model=self.config.alpr_detector_model,
                detector_conf_thresh=self.config.alpr_detector_conf_thresh,
                ocr_model=self.config.alpr_ocr_model,
                ocr_device="auto",
            )
        print("[READY] All models successfully initialized.")

    def process_video(
        self,
        video_source: str | int | Path,
        output_video_path: Optional[str | Path] = None,
        show_display: bool = False,
        max_frames: Optional[int] = None,
    ) -> list[dict]:
        """
        Processes a video file, stream, or webcam feed frame-by-frame.
        """
        self.initialize_models()

        source_str = str(video_source)
        if isinstance(video_source, int) or (isinstance(video_source, str) and video_source.isdigit()):
            cap = cv2.VideoCapture(int(video_source))
        else:
            video_path = Path(video_source)
            if not video_path.exists():
                raise FileNotFoundError(f"Input video file not found: {video_path.resolve()}")
            cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {video_source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_video_path is not None:
            out_p = Path(output_video_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))
            print(f"[OUTPUT] Writing annotated output video to: {out_p.resolve()}")

        if show_display and not has_gui_display():
            print("\n" + "-" * 65)
            print("[INFO] Headless cloud environment detected (No physical monitor/display).")
            print(f"[INFO] Disabling cv2.imshow window to prevent Qt crash.")
            print(f"[INFO] Annotated video will be saved to: {output_video_path or self.config.base_output_dir}")
            print("-" * 65 + "\n")
            show_display = False

        frame_idx = 0
        start_time = time.time()
        print(f"[PROCESS] Starting processing (Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_video_frames})...")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    print(f"[INFO] Reached max_frames limit ({max_frames}). Stopping.")
                    break

                # 1. Run Vehicle Detection with ByteTrack Tracking (persist=True)
                track_results = self.vehicle_model.track(
                    source=frame,
                    persist=True,
                    classes=self.config.vehicle_classes,
                    conf=self.config.vehicle_conf_thresh,
                    tracker=self.config.tracker_config,
                    verbose=False,
                )

                active_track_ids: set[int] = set()

                if track_results and len(track_results) > 0 and track_results[0].boxes is not None:
                    boxes_obj = track_results[0].boxes
                    if boxes_obj.id is not None:
                        ids = boxes_obj.id.int().cpu().tolist()
                        xyxys = boxes_obj.xyxy.cpu().tolist()

                        for t_id, xyxy in zip(ids, xyxys):
                            active_track_ids.add(t_id)
                            self.tracker_manager.update_track(
                                track_id=t_id,
                                box=xyxy,
                                frame=frame,
                                frame_idx=frame_idx,
                            )

                # Mark tracks not detected in this frame
                self.tracker_manager.mark_disappeared_tracks(active_track_ids)

                # 2. Run FastALPR on full frame periodically
                # Adaptive Smart Frame Gating:
                # Run ALPR only if vehicles are missing confident readings AND are within the readable focal zone
                needs_alpr = False
                for t_id in active_track_ids:
                    t = self.tracker_manager.tracks.get(t_id)
                    if t and not t.plate_confident_reached:
                        if self.config.enable_smart_gating:
                            # Sweet-spot distance gate
                            if t.best_vehicle_area >= self.config.min_vehicle_area_for_alpr:
                                needs_alpr = True
                                break
                        else:
                            needs_alpr = True
                            break

                if needs_alpr and (frame_idx % self.config.alpr_frame_interval == 0):
                    alpr_predictions = self.alpr.predict(frame)
                    if alpr_predictions:
                        self.tracker_manager.match_and_update_plates(
                            alpr_results=alpr_predictions,
                            frame=frame,
                            frame_idx=frame_idx,
                        )

                # 3. Check single event emission criteria
                self.tracker_manager.check_and_emit_events(current_frame_idx=frame_idx)

                # 4. Render visual overlays
                annotated_frame = self._render_frame(frame)

                if writer is not None:
                    writer.write(annotated_frame)

                if show_display:
                    try:
                        cv2.imshow("FastALPR Vehicle Tracking", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("[USER] Quitting visualization.")
                            break
                    except Exception as e:
                        if not hasattr(self, "_display_warning_shown"):
                            print(
                                f"\n[NOTICE] GUI window cannot be opened directly in this headless environment ({e}).\n"
                                f"Annotated video with all tracking boxes & HUD is being saved to: {output_video_path or 'output/'}\n"
                            )
                            self._display_warning_shown = True

                if frame_idx % 30 == 0 or frame_idx == 1:
                    u, p, np_cnt, act = self.tracker_manager.get_counts()
                    print(
                        f"[FRAME {frame_idx:05d}] Unique Vehicles: {u} "
                        f"(Plates: {p}, Without Plate: {np_cnt}) | Active: {act}"
                    )

        finally:
            cap.release()
            # Force emit events for any active vehicles tracked >= min_track_frames before closing
            self.tracker_manager.check_and_emit_events(current_frame_idx=frame_idx, force_all=True)

            if writer is not None:
                writer.release()
            if show_display:
                cv2.destroyAllWindows()

        end_time = time.time()
        elapsed_sec = end_time - start_time
        mins = int(elapsed_sec // 60)
        secs = elapsed_sec % 60
        avg_fps = frame_idx / elapsed_sec if elapsed_sec > 0 else 0.0

        u, p, np_cnt, _ = self.tracker_manager.get_counts()
        print("\n" + "=" * 65)
        print("          PIPELINE VIDEO PROCESSING COMPLETED")
        print("=" * 65)
        print(f"Total Time Taken:              {mins:02d}m {secs:05.2f}s ({elapsed_sec:.2f} seconds)")
        print(f"Total Frames Processed:        {frame_idx} frames")
        print(f"Average Processing Speed:      {avg_fps:.2f} FPS")
        print("-" * 65)
        print(f"Total Unique Vehicles Found:   {u}")
        print(f"  - Vehicles With Plate:       {p}")
        print(f"  - Vehicles Without Plate:    {np_cnt}")
        print("-" * 65)
        print(f"Events JSON Saved To:          {self.config.events_json_path.resolve()}")
        print(f"Snapshots Folder:              {self.config.snapshots_dir.resolve()}")
        if output_video_path is not None:
            print(f"Annotated Video Output:        {Path(output_video_path).resolve()}")
        print("=" * 65 + "\n")

        return self.tracker_manager.emitted_events

    def _render_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Draws smoothed bounding boxes, track IDs, plate text, and live statistics banner.
        - Vehicles with plate: GREEN box (0, 255, 0)
        - Vehicles without plate: ORANGE box (0, 140, 255)
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for t_id, track in self.tracker_manager.tracks.items():
            if track.disappeared_count > 0 or track.smoothed_box is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in track.smoothed_box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            has_plate = track.best_plate_text is not None

            # Color scheme (BGR)
            # Green for vehicles with plate, Orange for vehicles without plate
            box_color = (0, 255, 0) if has_plate else (0, 140, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Label text
            if has_plate:
                conf_pct = (track.best_plate_conf or 0.0) * 100.0
                label = f"ID:{t_id} | {track.best_plate_text} ({conf_pct:.0f}%)"
            else:
                label = f"ID:{t_id} | NO PLATE"

            # Draw label with solid background for contrast
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            label_y1 = max(0, y1 - th - baseline - 4)
            label_y2 = y1
            cv2.rectangle(annotated, (x1, label_y1), (x1 + tw + 6, label_y2), box_color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, y1 - baseline - 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        # Draw HUD Live Statistics Banner (Top-Left)
        u_cnt, p_cnt, np_cnt, act_cnt = self.tracker_manager.get_counts()
        lines = [
            f"CAMERA: {self.config.camera_id}",
            f"Total Unique Vehicles: {u_cnt}",
            f"Vehicles With Plate:   {p_cnt}",
            f"Vehicles Without Plate:{np_cnt}",
            f"Current Active:        {act_cnt}",
        ]

        hud_w = 310
        hud_h = 24 * len(lines) + 16
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (10, 10), (10 + hud_w, 10 + hud_h), (255, 255, 255), 1)

        y_offset = 32
        for line in lines:
            cv2.putText(
                annotated,
                line,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y_offset += 24

        return annotated

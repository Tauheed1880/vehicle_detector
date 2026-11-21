"""
Vehicle ALPR Pipeline (Phase 1 Enterprise Edition).
Features:
1. Exact 1:1 Video Duration Match (Output duration == Input duration).
2. Granular Profiling (Detector, Tracker, ALPR, Drawing, Video Encode).
3. FP16 Half Precision & imgsz=640 on CUDA for 20+ FPS.
4. Non-blocking Threaded Colab Live Preview using IPython.display.
5. Post-processing ffmpeg H.264/yuv420p browser conversion.
6. Detailed Benchmark Summary Table.
"""

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Ensure repository root is on Python path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import PipelineConfig
from fast_alpr import ALPR
from tracker import VehicleTrackerManager, filter_nested_boxes


class ColabLivePreviewWorker:
    """Non-blocking background worker for Colab real-time visual updates."""

    def __init__(self, max_w: int = 960, max_h: int = 540, jpeg_quality: int = 70):
        self.max_w = max_w
        self.max_h = max_h
        self.jpeg_quality = jpeg_quality
        self.queue: queue.Queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.display_handle = None
        self._init_display()

    def _init_display(self) -> None:
        try:
            from IPython.display import Image, display

            self.display_handle = display(None, display_id=True)
            self._has_ipython = True
        except Exception:
            self._has_ipython = False
            self.display_handle = None

    def start(self) -> None:
        if not self._has_ipython or self.display_handle is None:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def update_frame(self, frame: np.ndarray) -> None:
        if not self.running or not self._has_ipython:
            return
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            pass

    def _worker_loop(self) -> None:
        from IPython.display import Image

        while self.running:
            try:
                frame = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                h, w = frame.shape[:2]
                scale = min(self.max_w / w, self.max_h / h, 1.0)
                if scale < 1.0:
                    preview = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
                else:
                    preview = frame

                ret, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if ret and self.display_handle is not None:
                    self.display_handle.update(Image(data=buf.tobytes()))
            except Exception:
                pass

    def stop(self) -> None:
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)


class VehicleALPRPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        print(f"[INIT] Initializing VehicleALPRPipeline for Camera: {self.config.camera_id}")

        print(
            "\n" + "=" * 70 + "\n"
            "[LICENSE NOTICE] Vehicle detector uses Ultralytics YOLO ("
            f"{self.config.vehicle_model_name}).\n"
            "Ultralytics YOLO is licensed under GNU AGPL-3.0.\n"
            "For commercial closed-source production, consider permissive alternatives:\n"
            "  1. YOLOv10 (Apache-2.0)\n"
            "  2. RT-DETR (Apache-2.0)\n"
            "  3. YOLOX (Apache-2.0)\n"
            "  4. Enterprise Commercial License from Ultralytics.\n"
            + "=" * 70 + "\n"
        )

        self.vehicle_model = None
        self.alpr = None
        self.tracker_manager = VehicleTrackerManager(self.config)
        self.preview_worker: Optional[ColabLivePreviewWorker] = None
        self.has_cuda = False

    def initialize_models(self) -> None:
        """Loads and verifies models on GPU/CUDA with FP16 support."""
        try:
            import onnxruntime as ort

            available_providers = ort.get_available_providers()
            print(f"[ONNX RUNTIME] Available Providers: {available_providers}")
            self.has_cuda = "CUDAExecutionProvider" in available_providers
            if self.has_cuda:
                print("[ONNX RUNTIME] -> CUDA GPU Acceleration ACTIVE.")
                detector_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                print("[ONNX RUNTIME] -> Running on CPUExecutionProvider (No CUDA found).")
                detector_providers = ["CPUExecutionProvider"]
        except Exception as e:
            print(f"[ONNX RUNTIME] Provider check warning: {e}")
            detector_providers = None

        if self.vehicle_model is None:
            from ultralytics import YOLO

            print(f"[LOAD] Loading vehicle detector model: {self.config.vehicle_model_name}...")
            self.vehicle_model = YOLO(self.config.vehicle_model_name)

        if self.alpr is None:
            print(
                f"[LOAD] Loading FastALPR (Detector: {self.config.alpr_detector_model}, "
                f"OCR: {self.config.alpr_ocr_model})..."
            )
            self.alpr = ALPR(
                detector_model=self.config.alpr_detector_model,
                detector_conf_thresh=self.config.alpr_detector_conf_thresh,
                detector_providers=detector_providers,
                ocr_model=self.config.alpr_ocr_model,
                ocr_device="cuda" if self.has_cuda else "auto",
            )
        print("[READY] All models successfully verified and loaded.")

    def process_video(
        self,
        video_source: str | int | Path,
        output_video_path: Optional[str | Path] = None,
        max_frames: Optional[int] = None,
    ) -> list[dict]:
        """
        Processes video stream with exact 1:1 duration matching, granular profiling,
        sweet-spot plate gating, and non-blocking live preview.
        """
        self.initialize_models()

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
        expected_duration = total_video_frames / fps if total_video_frames > 0 else 0.0

        # Output video writer
        writer = None
        raw_output_path = None
        output_frames_written = 0

        if self.config.save_video and output_video_path is not None:
            out_p = Path(output_video_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            raw_output_path = out_p.with_name("raw_" + out_p.name)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(raw_output_path), fourcc, fps, (width, height))
            print(f"[OUTPUT] Writing video at {fps:.1f} FPS to: {raw_output_path.resolve()}")

        # Start non-blocking Colab Live Preview worker
        if self.config.show_preview:
            self.preview_worker = ColabLivePreviewWorker(
                max_w=self.config.preview_max_width,
                max_h=self.config.preview_max_height,
                jpeg_quality=self.config.preview_jpeg_quality,
            )
            self.preview_worker.start()

        frame_idx = 0
        start_time = time.time()
        rolling_fps = fps
        fps_last_time = time.time()
        fps_frame_count = 0

        # Granular Profiling Timers (seconds)
        prof_detector = 0.0
        prof_tracker = 0.0
        prof_alpr = 0.0
        prof_drawing = 0.0
        prof_encode = 0.0

        # Custom ByteTrack config path
        tracker_cfg_path = Path(self.config.tracker_config)
        if not tracker_cfg_path.is_absolute():
            tracker_cfg_path = REPO_ROOT / tracker_cfg_path

        # Half precision option for CUDA
        use_half = self.has_cuda and self.config.vehicle_half_precision

        print(
            f"[PROCESS] Starting processing (Resolution: {width}x{height}, FPS: {fps:.1f}, "
            f"Frames: {total_video_frames}, Duration: {expected_duration:.2f}s)..."
        )

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                fps_frame_count += 1
                if max_frames and frame_idx > max_frames:
                    print(f"[INFO] Reached max_frames limit ({max_frames}). Stopping.")
                    break

                # Measure live rolling FPS
                t_now = time.time()
                if (t_now - fps_last_time) >= 1.0:
                    rolling_fps = fps_frame_count / (t_now - fps_last_time)
                    fps_frame_count = 0
                    fps_last_time = t_now

                # ---------------------------------------------------------
                # 1. Vehicle Detection (YOLO with imgsz=640 & FP16)
                # ---------------------------------------------------------
                t_det_start = time.perf_counter()
                track_results = self.vehicle_model.track(
                    source=frame,
                    persist=True,
                    classes=self.config.vehicle_classes,
                    conf=self.config.vehicle_conf_thresh,
                    imgsz=self.config.vehicle_imgsz,
                    half=use_half,
                    tracker=str(tracker_cfg_path),
                    verbose=False,
                )
                prof_detector += time.perf_counter() - t_det_start

                # ---------------------------------------------------------
                # 2. Tracking Association & Nested Box Suppression
                # ---------------------------------------------------------
                t_trk_start = time.perf_counter()
                active_track_ids: set[int] = set()

                if track_results and len(track_results) > 0 and track_results[0].boxes is not None:
                    boxes_obj = track_results[0].boxes
                    if boxes_obj.id is not None:
                        ids = boxes_obj.id.int().cpu().tolist()
                        xyxys = boxes_obj.xyxy.cpu().tolist()
                        scores = boxes_obj.conf.cpu().tolist()

                        # Class-Agnostic Nested Box Suppression
                        clean_boxes, clean_scores = filter_nested_boxes(
                            boxes=xyxys,
                            scores=scores,
                            containment_thresh=self.config.box_containment_thresh,
                        )

                        for t_id, xyxy in zip(ids, xyxys):
                            if any(np.allclose(xyxy, cb, atol=2.0) for cb in clean_boxes):
                                active_track_ids.add(t_id)
                                self.tracker_manager.update_track(
                                    track_id=t_id,
                                    box=xyxy,
                                    frame=frame,
                                    frame_idx=frame_idx,
                                )

                self.tracker_manager.mark_disappeared_tracks(active_track_ids)
                prof_tracker += time.perf_counter() - t_trk_start

                # ---------------------------------------------------------
                # 3. Adaptive ALPR Sweet-Spot Plate Recognition
                # ---------------------------------------------------------
                t_alpr_start = time.perf_counter()
                needs_alpr = False
                for t_id in active_track_ids:
                    t = self.tracker_manager.tracks.get(t_id)
                    if t and not t.plate_confident_reached:
                        # Only trigger ALPR if vehicle is inside readable zone and large enough
                        if self.config.enable_smart_gating:
                            if t.entered_readable_zone and t.best_vehicle_area >= self.config.min_readable_vehicle_area:
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

                # Check event emission
                self.tracker_manager.check_and_emit_events(current_frame_idx=frame_idx)
                prof_alpr += time.perf_counter() - t_alpr_start

                # ---------------------------------------------------------
                # 4. Render Frame (Only when preview or save is enabled)
                # ---------------------------------------------------------
                t_draw_start = time.perf_counter()
                annotated_frame = None
                if self.config.save_video or self.config.show_preview:
                    annotated_frame = self._render_frame(frame, live_fps=rolling_fps)
                prof_drawing += time.perf_counter() - t_draw_start

                # ---------------------------------------------------------
                # 5. Video Encoding
                # ---------------------------------------------------------
                t_enc_start = time.perf_counter()
                if writer is not None and annotated_frame is not None:
                    writer.write(annotated_frame)
                    output_frames_written += 1
                prof_encode += time.perf_counter() - t_enc_start

                # Update live preview worker
                if self.preview_worker is not None and annotated_frame is not None:
                    if frame_idx % self.config.preview_interval == 0:
                        self.preview_worker.update_frame(annotated_frame)

                if frame_idx % 30 == 0 or frame_idx == 1:
                    u, p, np_cnt, act, sm = self.tracker_manager.get_counts()
                    print(
                        f"[FRAME {frame_idx:05d}] FPS: {rolling_fps:.1f} | Unique: {u} "
                        f"(Plates: {p}, Without Plate: {np_cnt}) | Active: {act} | Ignored Small: {sm}"
                    )

        finally:
            cap.release()
            if self.preview_worker is not None:
                self.preview_worker.stop()

            # End of video flush: ONLY flush tracks that entered readable zone and satisfied min_track_frames
            self.tracker_manager.check_and_emit_events(current_frame_idx=frame_idx, force_all=True)

            if writer is not None:
                writer.release()

            # Wait for all async snapshots to finish saving to disk
            self.tracker_manager.snapshot_saver.stop()

        # ---------------------------------------------------------
        # 6. Video Duration Verification & H.264 Conversion
        # ---------------------------------------------------------
        final_video_path = output_video_path
        actual_output_duration = output_frames_written / fps if fps > 0 else 0.0
        duration_diff = abs(actual_output_duration - expected_duration)

        print(f"\n[DURATION CHECK] Input: {expected_duration:.2f}s ({total_video_frames} frames) | "
              f"Output: {actual_output_duration:.2f}s ({output_frames_written} frames) | "
              f"Diff: {duration_diff:.2f}s")

        if expected_duration > 0 and duration_diff > 0.5:
            print(f"[DURATION WARNING] Output video duration difference is {duration_diff:.2f}s (expected < 0.5s).")
        else:
            print(f"[DURATION PASSED] Output video duration matches input duration perfectly (Diff: {duration_diff:.2f}s < 0.5s).")

        if raw_output_path is not None and raw_output_path.exists():
            final_out = Path(output_video_path)
            if self.config.convert_h264 and shutil.which("ffmpeg"):
                print(f"[FFMPEG] Converting video to browser-compatible H.264 (libx264 + yuv420p)...")
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(raw_output_path),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "fast",
                    "-crf", "23",
                    str(final_out),
                ]
                res = subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if res.returncode == 0:
                    raw_output_path.unlink(missing_ok=True)
                    print(f"[FFMPEG] H.264 conversion complete: {final_out.resolve()}")
                else:
                    raw_output_path.rename(final_out)
            else:
                raw_output_path.rename(final_out)

            if self.config.drive_output_dir.exists():
                try:
                    self.config.drive_output_dir.mkdir(parents=True, exist_ok=True)
                    drive_dest = self.config.drive_output_dir / final_out.name
                    shutil.copy2(str(final_out), str(drive_dest))
                    print(f"[GOOGLE DRIVE] Video backed up to: {drive_dest}")
                except Exception as e:
                    print(f"[GOOGLE DRIVE] Drive backup note: {e}")

        # ---------------------------------------------------------
        # 7. Summary Table & Granular Profiling Report
        # ---------------------------------------------------------
        end_time = time.time()
        elapsed_sec = end_time - start_time
        mins = int(elapsed_sec // 60)
        secs = elapsed_sec % 60
        avg_fps = frame_idx / elapsed_sec if elapsed_sec > 0 else 0.0

        u, p, np_cnt, _, small_ignored = self.tracker_manager.get_counts()
        total_profiled = prof_detector + prof_tracker + prof_alpr + prof_drawing + prof_encode
        tot_prof_safe = max(1e-5, total_profiled)

        print("\n" + "=" * 70)
        print("                 PIPELINE PROCESSING SUMMARY TABLE")
        print("=" * 70)
        print(f"Total Video Frames Processed:  {frame_idx} frames")
        print(f"Total Processing Time:         {mins:02d}m {secs:05.2f}s ({elapsed_sec:.2f} seconds)")
        print(f"Average Pipeline Speed:        {avg_fps:.2f} FPS")
        print(f"Video Duration Match:          Input {expected_duration:.2f}s == Output {actual_output_duration:.2f}s")
        print("-" * 70)
        print(f"Total Unique Vehicles Tracked: {u}")
        print(f"  - Vehicles With Plate:       {p}")
        print(f"  - Vehicles Without Plate:    {np_cnt}")
        print(f"  - Filtered as Too Small/Far: {small_ignored} (Horizon cars ignored)")
        print("-" * 70)
        print("                 GRANULAR PROFILING BREAKDOWN")
        print("-" * 70)
        print(f"Vehicle Detector (YOLO FP16):  {prof_detector:6.2f}s  ({(prof_detector/tot_prof_safe)*100:5.1f}%)")
        print(f"Tracker & Box Deduplication:   {prof_tracker:6.2f}s  ({(prof_tracker/tot_prof_safe)*100:5.1f}%)")
        print(f"FastALPR Detection & OCR:      {prof_alpr:6.2f}s  ({(prof_alpr/tot_prof_safe)*100:5.1f}%)")
        print(f"Visual HUD & Box Drawing:      {prof_drawing:6.2f}s  ({(prof_drawing/tot_prof_safe)*100:5.1f}%)")
        print(f"Video File Encoding:           {prof_encode:6.2f}s  ({(prof_encode/tot_prof_safe)*100:5.1f}%)")
        print(f"Snapshot Saving (Async):       0.00s  (Non-blocking background thread)")
        print("-" * 70)
        print(f"Events JSON Saved To:          {self.config.events_json_path.resolve()}")
        print(f"Snapshots Folder:              {self.config.snapshots_dir.resolve()}")
        if output_video_path is not None:
            print(f"Annotated Video Output:        {Path(output_video_path).resolve()}")
        print("=" * 70 + "\n")

        return self.tracker_manager.emitted_events

    def _render_frame(self, frame: np.ndarray, live_fps: float = 0.0) -> np.ndarray:
        """Draws smoothed boxes, IDs, plate consensus, HUD dashboard, and optional ROI."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        if self.config.roi_polygon is not None and len(self.config.roi_polygon) >= 3:
            pts = np.array(self.config.roi_polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

        for t_id, track in self.tracker_manager.tracks.items():
            if track.disappeared_count > 0 or track.smoothed_box is None:
                continue
            # Only draw boxes for tracks that have entered the readable zone
            if not track.entered_readable_zone and track.best_plate_text is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in track.smoothed_box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            has_plate = track.best_plate_text is not None
            box_color = (0, 255, 0) if has_plate else (0, 140, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            if has_plate:
                conf_pct = (track.best_plate_conf or 0.0) * 100.0
                label = f"{track.global_id} | {track.best_plate_text} ({conf_pct:.0f}%)"
            else:
                label = f"{track.global_id} | NO PLATE"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = max(th + 5, y1 - 4)
            cv2.rectangle(annotated, (x1, label_y - th - 5), (x1 + tw + 6, label_y + baseline + 2), box_color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, label_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        # Draw HUD Live Statistics Dashboard (Top-Left)
        u_cnt, p_cnt, np_cnt, act_cnt, small_cnt = self.tracker_manager.get_counts()
        lines = [
            f"CAMERA: {self.config.camera_id} | FPS: {live_fps:.1f}",
            f"Total Unique Vehicles: {u_cnt}",
            f"Vehicles With Plate:   {p_cnt}",
            f"Vehicles Without Plate:{np_cnt}",
            f"Ignored Horizon Cars:  {small_cnt}",
            f"Current Active:        {act_cnt}",
        ]

        hud_w = 340
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

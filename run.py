"""
CLI Entry point to run Vehicle ALPR Tracking Pipeline (Phase 1).
Example:
    python run.py --source input_video.mp4 --output-video output/annotated.mp4
"""

import argparse
import json
from pathlib import Path

from config import PipelineConfig
from pipeline import VehicleALPRPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle ALPR Single-Detection Pipeline (Phase 1)")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input video file (e.g. stream.mp4), RTSP stream, or camera index (e.g. 0).",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="output/annotated_output.mp4",
        help="Path where annotated output video will be saved.",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="CAM-00001",
        help="Camera ID string for event logging (e.g. CAM-00001).",
    )
    parser.add_argument(
        "--vehicle-conf",
        type=float,
        default=0.35,
        help="Confidence threshold for vehicle detection (default: 0.35).",
    )
    parser.add_argument(
        "--alpr-interval",
        type=int,
        default=2,
        help="Run ALPR every K frames to optimize compute (default: 2).",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=5,
        help="Minimum frames a vehicle must be tracked before emitting event (default: 5).",
    )
    parser.add_argument(
        "--disappeared-frames",
        type=int,
        default=45,
        help="Frames before an untracked vehicle is considered disappeared (default: 45).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=8000.0,
        help="Minimum vehicle box area to be considered readable (default: 8000).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        default=True,
        help="Enable Colab real-time visual preview via IPython.display (default: enabled).",
    )
    parser.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        help="Disable live visual preview.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=True,
        help="Save annotated output video file (default: enabled).",
    )
    parser.add_argument(
        "--no-save-video",
        dest="save_video",
        action="store_false",
        help="Do not save output video file to disk.",
    )
    parser.add_argument(
        "--sync-realtime",
        action="store_true",
        default=True,
        help="Synchronize playback to real wall-clock time (default: enabled).",
    )
    parser.add_argument(
        "--no-sync-realtime",
        dest="sync_realtime",
        action="store_false",
        help="Process video as fast as possible without real-time pacing.",
    )
    parser.add_argument(
        "--enforce-regex",
        action="store_true",
        default=False,
        help="Strictly enforce country regex validation on plate reads.",
    )
    parser.add_argument(
        "--include-remarks",
        action="store_true",
        default=False,
        help="Include optional 'remarks' field in JSON event output.",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="Optional JSON string of ROI polygon coordinates, e.g. '[[100,200],[1100,200],[1200,700],[50,700]]'",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    roi_poly = None
    if args.roi:
        try:
            roi_poly = json.loads(args.roi)
        except Exception as e:
            print(f"[CONFIG WARNING] Could not parse --roi JSON: {e}")

    config = PipelineConfig(
        camera_id=args.camera_id,
        vehicle_conf_thresh=args.vehicle_conf,
        alpr_frame_interval=args.alpr_interval,
        min_track_frames=args.min_track_frames,
        max_disappeared_frames=args.disappeared_frames,
        min_readable_vehicle_area=args.min_area,
        show_preview=args.preview,
        save_video=args.save_video,
        sync_realtime=args.sync_realtime,
        enforce_regex=args.enforce_regex,
        include_remarks=args.include_remarks,
        roi_polygon=roi_poly,
    )

    pipeline = VehicleALPRPipeline(config=config)
    pipeline.process_video(
        video_source=args.source,
        output_video_path=args.output_video,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

"""
CLI Entry point to run Vehicle ALPR Tracking Pipeline.
Example:
    python run.py --source input_video.mp4 --output-video output/annotated.mp4
"""

import argparse
from pathlib import Path

from config import PipelineConfig
from pipeline import VehicleALPRPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle ALPR Single-Detection Pipeline")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input video file (e.g. video.mp4), image, or camera index (e.g. 0).",
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
        default=0.40,
        help="Confidence threshold for vehicle detection (default: 0.40).",
    )
    parser.add_argument(
        "--alpr-interval",
        type=int,
        default=2,
        help="Run ALPR every K frames to optimize CPU usage (default: 2).",
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
        default=30,
        help="Frames before an untracked vehicle is considered disappeared (default: 30).",
    )
    parser.add_argument(
        "--vehicle-model",
        type=str,
        default="yolov8n.pt",
        help="Vehicle detector model weight file or name (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live GUI window with cv2.imshow (do not use in headless Colab).",
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

    config = PipelineConfig(
        camera_id=args.camera_id,
        vehicle_model_name=args.vehicle_model,
        vehicle_conf_thresh=args.vehicle_conf,
        alpr_frame_interval=args.alpr_interval,
        min_track_frames=args.min_track_frames,
        max_disappeared_frames=args.disappeared_frames,
    )

    pipeline = VehicleALPRPipeline(config=config)
    pipeline.process_video(
        video_source=args.source,
        output_video_path=args.output_video,
        show_display=args.show,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

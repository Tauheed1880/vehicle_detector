"""
Lightweight benchmark harness for the Vehicle ALPR pipeline.

It runs the EXISTING pipeline unchanged on a test video, tees the console output,
parses the pipeline's own profiling table and the emitted events JSON, then writes a
machine-readable result file so a baseline run and an optimized run can be compared.

This harness does not modify, re-implement or accelerate the pipeline - it only measures
it. No new runtime dependency is introduced.

Usage:
    python tools/benchmark.py --source stream.mp4 --label baseline
    python tools/benchmark.py --source stream.mp4 --label optimized

Outputs:
    output/benchmark_<label>.json   (machine-readable metrics)
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repository root is importable when the script is run directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import PipelineConfig  # noqa: E402
from pipeline import VehicleALPRPipeline  # noqa: E402


class _Tee:
    """Duplicates stdout to the console and a capture buffer simultaneously."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass


def _extract(pattern: str, text: str, cast=float) -> Optional[Any]:
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        value = match.group(1)
        return cast(value)
    except Exception:
        return None


def _parse_metrics(captured: str) -> Dict[str, Any]:
    """Parses the pipeline's profiling table into a metrics dict."""
    metrics: Dict[str, Any] = {}

    # --- Source video properties ---
    metrics["source_fps"] = _extract(r"FPS:\s*([\d.]+),\s*Frames", captured, float)
    metrics["total_frames_processed"] = _extract(
        r"Total Video Frames Processed:\s*(\d+)", captured, int
    )

    # --- Speed ---
    metrics["ai_processing_fps"] = _extract(r"Average Pipeline Speed:\s*([\d.]+) FPS", captured, float)
    metrics["total_wall_clock_sec"] = _extract(
        r"Total Wall-Clock Time:\s*\d+m\s*([\d.]+)s", captured, float
    )

    # --- Inference / ALPR timing (seconds, cumulative) ---
    metrics["yolo_inference_sec"] = _extract(
        r"Vehicle Detector \(YOLO\):\s*([\d.]+)s", captured, float
    )
    metrics["fastalpr_sec"] = _extract(r"FastALPR \([^)]+\):\s*([\d.]+)s", captured, float)
    metrics["hud_drawing_sec"] = _extract(r"HUD Visual Drawing:\s*([\d.]+)s", captured, float)
    metrics["video_encode_sec"] = _extract(r"Video File Encoding \(MP4\):\s*([\d.]+)s", captured, float)

    # --- Async I/O accounting ---
    metrics["decoder_decode_sec"] = _extract(r"Decoder thread decode:\s*([\d.]+)s", captured, float)
    metrics["encoder_encode_sec"] = _extract(r"Encoder thread encode:\s*([\d.]+)s", captured, float)
    metrics["frames_dropped_decoder"] = _extract(r"Frames Dropped:\s*decoder (\d+)", captured, int)
    metrics["frames_dropped_encoder"] = _extract(r"encoder (\d+)", captured, int)

    # --- Vehicle / plate counts ---
    metrics["unique_vehicles_tracked"] = _extract(
        r"Total Unique Vehicles Tracked:\s*(\d+)", captured, int
    )
    metrics["vehicles_with_plate"] = _extract(r"Vehicles With Plate:\s*(\d+)", captured, int)
    metrics["vehicles_without_plate"] = _extract(r"Vehicles Without Plate:\s*(\d+)", captured, int)

    # --- Ground truth (UK format only) ---
    metrics["ground_truth_matched"] = _extract(r"Match Score:\s*(\d+)/\d+", captured, int)
    metrics["ground_truth_missed"] = _extract(r"Missed:\s*(\d+)", captured, int)

    # --- Event counts from the pipeline report ---
    metrics["total_events"] = _extract(r"Total Number of Events:\s*(\d+)", captured, int)
    metrics["partial_plate_events"] = _extract(
        r"Events with Partial Plates \('\?'\):\s*(\d+)", captured, int
    )

    # --- Durability / duration match ---
    metrics["duration_diff_sec"] = _extract(r"Diff:\s*([\d.]+)s", captured, float)

    # End-to-end event latency is not directly instrumented by the existing pipeline.
    metrics["end_to_end_event_latency_sec"] = None

    return metrics


def _event_stats(events_json_path: Path) -> Dict[str, Any]:
    """Reads the emitted events JSON and derives duplicate-event statistics."""
    stats: Dict[str, Any] = {
        "events_file": str(events_json_path),
        "events_in_file": 0,
        "events_with_plate": 0,
        "duplicate_plate_events": 0,
    }
    if not events_json_path.exists():
        return stats
    try:
        with open(events_json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return stats

    if not isinstance(data, list):
        return stats

    stats["events_in_file"] = len(data)
    seen: Dict[str, int] = {}
    for event in data:
        plate = event.get("number_plate")
        if plate:
            seen[str(plate)] = seen.get(str(plate), 0) + 1
    stats["events_with_plate"] = sum(1 for e in data if e.get("number_plate"))
    stats["duplicate_plate_events"] = sum(count - 1 for count in seen.values() if count > 1)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark harness for the Vehicle ALPR pipeline.")
    parser.add_argument("--source", type=str, required=True, help="Path to the test video file.")
    parser.add_argument("--label", type=str, default="run", help="Label for this run (e.g. baseline / optimized).")
    parser.add_argument("--camera-id", type=str, default="CAM-00001", help="Camera ID for the run.")
    parser.add_argument(
        "--plate-format",
        type=str,
        choices=["PAKISTAN", "GENERAL", "UK"],
        default="UK",
        help="License plate format standard (default: UK, matching the ground-truth report).",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="output/annotated_output.mp4",
        help="Where the annotated output video is written.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process (for quick short runs).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        default=False,
        help="Enable the Colab live preview during the benchmark (default: disabled).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[BENCHMARK ERROR] Video source not found: {source_path.resolve()}")
        sys.exit(1)

    config = PipelineConfig(
        camera_id=args.camera_id,
        plate_format=args.plate_format,
        show_preview=args.preview,
        save_video=True,
    )

    print("=" * 75)
    print(f"                 BENCHMARK RUN: '{args.label}'")
    print("=" * 75)
    print(f"Source video : {source_path.resolve()}")
    print(f"Plate format : {config.plate_format}")
    print(f"ALPR interval: every {config.alpr_frame_interval} frames")
    print(f"Async I/O    : {config.async_io}")
    print("=" * 75)

    pipeline = VehicleALPRPipeline(config=config)

    buffer = io.StringIO()
    tee = _Tee(sys.stdout, buffer)
    original_stdout = sys.stdout
    run_error: Optional[str] = None
    try:
        sys.stdout = tee
        pipeline.process_video(
            video_source=str(source_path),
            output_video_path=args.output_video,
            max_frames=args.max_frames,
        )
    except Exception as exc:  # keep the harness honest: record failures, do not hide them
        run_error = f"{type(exc).__name__}: {exc}"
    finally:
        sys.stdout = original_stdout

    captured = buffer.getvalue()
    metrics = _parse_metrics(captured)
    metrics["label"] = args.label
    metrics["status"] = "FAILED" if run_error else "OK"
    metrics["error"] = run_error
    metrics.update(_event_stats(Path(config.events_json_path)))

    results_path = Path(config.base_output_dir) / f"benchmark_{args.label}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
    except Exception as exc:
        print(f"[BENCHMARK WARNING] Could not write results file: {exc}")

    print("\n" + "=" * 75)
    print(f"                 BENCHMARK RESULT: '{args.label}'")
    print("=" * 75)
    for key, value in metrics.items():
        print(f"  {key:<32}: {value}")
    print("=" * 75)
    print(f"  Machine-readable result: {results_path.resolve()}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

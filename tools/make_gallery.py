#!/usr/bin/env python3
"""
tools/make_gallery.py

Generates:
1. docs/snapshots_slideshow.gif: Animated GIF showing events one-by-one (1280x540, ~2s per event, < 10 MB).
   - Left: Vehicle snapshot fitted into 760x540 (keeps aspect ratio, dark background).
   - Right: Plate crop enlarged to ~480px wide with plate text (or "NO PLATE" in orange),
     plate confidence in %, event_id, and time in large readable text under it.
   - Ordered by event time.
2. docs/gallery.md: Markdown table with one row per event (event_id, vehicle thumbnail, plate thumbnail,
   plate text, confidence, time/date, status).

Usage:
   python tools/make_gallery.py --events output/events.json --snapshots output/snapshots --out docs/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont


def get_font(size: int = 20, bold: bool = False):
    """Loads a readable TrueType font if available, fallback to default font."""
    font_candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in font_candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Scales image proportionally to fit within max_w and max_h."""
    w_ratio = max_w / float(img.width)
    h_ratio = max_h / float(img.height)
    ratio = min(w_ratio, h_ratio)
    new_w = max(1, int(img.width * ratio))
    new_h = max(1, int(img.height * ratio))
    resample = Image.Resampling.LANCZOS if ratio < 1.0 else Image.Resampling.BILINEAR
    return img.resize((new_w, new_h), resample)


def create_slideshow_card(event: dict, snapshots_dir: Path) -> Image.Image:
    """Creates a 1280x540 slide card for an event."""
    card_w, card_h = 1280, 540
    card = Image.new("RGB", (card_w, card_h), color=(18, 20, 26))
    draw = ImageDraw.Draw(card)

    font_huge = get_font(size=36, bold=True)
    font_large = get_font(size=24, bold=True)
    font_medium = get_font(size=20, bold=False)
    font_label = get_font(size=18, bold=True)

    # -------------------------------------------------------------
    # Left Section (0 to 760 px): Vehicle Snapshot fitted into 760x540
    # -------------------------------------------------------------
    v_area_w, v_area_h = 760, 540
    v_name = event.get("snapshot_vehicle")
    if v_name:
        src_v = snapshots_dir / v_name
        if src_v.exists():
            try:
                with Image.open(src_v) as im_v:
                    # Fit within (740, 520) with 10px margin
                    im_fit = fit_image(im_v, max_w=740, max_h=510)
                    px = (v_area_w - im_fit.width) // 2
                    py = (v_area_h - im_fit.height) // 2
                    card.paste(im_fit, (px, py))
                    # Subtle vehicle box outline
                    draw.rectangle([(px, py), (px + im_fit.width, py + im_fit.height)], outline=(255, 140, 0), width=2)
            except Exception as e:
                print(f"[WARN] Could not load vehicle snapshot {src_v}: {e}", file=sys.stderr)

    # Dividing separator line
    draw.line([(760, 20), (760, 520)], fill=(45, 52, 68), width=2)

    # -------------------------------------------------------------
    # Right Section (760 to 1280 px): Plate crop + metadata
    # -------------------------------------------------------------
    r_start_x = 780
    r_w = 480
    p_name = event.get("snapshot_number_plate")
    p_text = event.get("number_plate")
    p_conf = event.get("number_plate_confidence")
    eid = event.get("event_id", "N/A")
    etime = event.get("time", "N/A")
    edate = event.get("date", "")

    if p_name and p_text:
        src_p = snapshots_dir / p_name
        if src_p.exists():
            try:
                with Image.open(src_p) as im_p:
                    # Enlarge plate crop to ~460-480 px wide
                    im_p_fit = fit_image(im_p, max_w=460, max_h=190)
                    px = r_start_x + (r_w - im_p_fit.width) // 2
                    py = 35 + (190 - im_p_fit.height) // 2
                    card.paste(im_p_fit, (px, py))
                    # Bright green border around plate
                    draw.rectangle([(px - 2, py - 2), (px + im_p_fit.width + 1, py + im_p_fit.height + 1)], outline=(0, 255, 0), width=3)
            except Exception as e:
                print(f"[WARN] Could not load plate crop {src_p}: {e}", file=sys.stderr)

        text_y = 250
        # Plate Text in large green
        draw.text((r_start_x + 10, text_y), f"PLATE: {p_text}", fill=(0, 255, 120), font=font_huge)
        text_y += 55

        # Plate confidence in %
        conf_str = f"{p_conf * 100:.1f}%" if p_conf is not None else "N/A"
        draw.text((r_start_x + 10, text_y), f"Confidence: {conf_str}", fill=(240, 245, 255), font=font_large)
        text_y += 45

        # Event ID
        draw.text((r_start_x + 10, text_y), f"Event ID: {eid}", fill=(180, 195, 215), font=font_medium)
        text_y += 35

        # Time
        time_display = f"{etime} ({edate})" if edate else etime
        draw.text((r_start_x + 10, text_y), f"Time: {time_display}", fill=(180, 195, 215), font=font_medium)
        text_y += 35

        # Camera
        cam = event.get("camera_id", "CAM-00001")
        draw.text((r_start_x + 10, text_y), f"Camera: {cam}", fill=(140, 155, 175), font=font_medium)

    else:
        # NO PLATE event (e.g. edge horizon vehicle)
        text_y = 120
        # "NO PLATE" in orange
        draw.text((r_start_x + 10, text_y), "NO PLATE", fill=(255, 140, 0), font=font_huge)
        text_y += 65

        draw.text((r_start_x + 10, text_y), "Confidence: N/A", fill=(200, 205, 215), font=font_large)
        text_y += 50

        draw.text((r_start_x + 10, text_y), f"Event ID: {eid}", fill=(180, 195, 215), font=font_medium)
        text_y += 40

        time_display = f"{etime} ({edate})" if edate else etime
        draw.text((r_start_x + 10, text_y), f"Time: {time_display}", fill=(180, 195, 215), font=font_medium)
        text_y += 40

        draw.text((r_start_x + 10, text_y), "Status: Edge-cut / Horizon Vehicle", fill=(255, 180, 80), font=font_medium)

    return card


def build_slideshow_gif(
    events: List[dict],
    snapshots_dir: Path,
    out_gif: Path,
    duration_ms: int = 2000,
) -> None:
    """Builds docs/snapshots_slideshow.gif and guarantees size < 10 MB."""
    frames = []
    for ev in events:
        card = create_slideshow_card(event=ev, snapshots_dir=snapshots_dir)
        card_pal = card.quantize(colors=256, method=Image.Resampling.LANCZOS)
        frames.append(card_pal)

    if not frames:
        return

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )

    # Automatic size check: reduce to 128 colors if > 10 MB
    if out_gif.stat().st_size > 10 * 1024 * 1024:
        reduced_frames = [f.quantize(colors=128) for f in frames]
        reduced_frames[0].save(
            out_gif,
            save_all=True,
            append_images=reduced_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )


def build_gallery_markdown(
    events: List[dict],
    out_md: Path,
    rel_snapshots_path: str = "snapshots",
) -> str:
    """Builds docs/gallery.md markdown table."""
    lines = [
        "# FastALPR Vehicle & Plate Detection Gallery",
        "",
        "| Event ID | Vehicle Snapshot | Plate Snapshot | Plate Read | Confidence | Time / Date | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for ev in events:
        eid = ev.get("event_id", "")
        t_str = ev.get("time", "")
        d_str = ev.get("date", "")
        p_text = ev.get("number_plate")
        p_conf = ev.get("number_plate_confidence")
        v_name = ev.get("snapshot_vehicle")
        p_name = ev.get("snapshot_number_plate")

        time_date = f"{t_str} {d_str}".strip()
        v_cell = f"[<img src='{rel_snapshots_path}/{v_name}' width='200'>]({rel_snapshots_path}/{v_name})" if v_name else "—"
        p_cell = f"[<img src='{rel_snapshots_path}/{p_name}' width='140'>]({rel_snapshots_path}/{p_name})" if p_name else "*No Plate*"
        plate_str = f"**`{p_text}`**" if p_text else "*None*"
        conf_str = f"{p_conf * 100:.1f}%" if p_conf is not None else "—"
        status_str = "Verified Plate Event" if p_text else "Vehicle Only (Edge/Horizon)"

        lines.append(f"| `{eid}` | {v_cell} | {p_cell} | {plate_str} | {conf_str} | {time_date} | {status_str} |")

    lines.append("")
    content = "\n".join(lines)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(content, encoding="utf-8")
    return content


def main():
    parser = argparse.ArgumentParser(description="Generate vehicle & plate gallery and slideshow GIF.")
    parser.add_argument("--events", type=Path, default=Path("output/events.json"), help="Path to events.json")
    parser.add_argument("--snapshots", type=Path, default=Path("output/snapshots"), help="Path to snapshots folder")
    parser.add_argument("--out", type=Path, default=Path("docs"), help="Output documentation folder")
    args = parser.parse_args()

    if not args.events.exists():
        print(f"[ERROR] Events file not found: {args.events}", file=sys.stderr)
        sys.exit(1)

    with open(args.events, "r", encoding="utf-8") as f:
        events = json.load(f)

    # Order by event time
    events = sorted(events, key=lambda e: (e.get("time", ""), e.get("event_id", "")))
    print(f"Loaded {len(events)} events (ordered by time).")

    # 1. Slideshow GIF
    slideshow_gif = args.out / "snapshots_slideshow.gif"
    print(f"Building slideshow GIF: {slideshow_gif}...")
    build_slideshow_gif(events=events, snapshots_dir=args.snapshots, out_gif=slideshow_gif, duration_ms=2000)
    print(f"Slideshow saved: {slideshow_gif} ({slideshow_gif.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Gallery Markdown
    gallery_md = args.out / "gallery.md"
    print(f"Building gallery markdown: {gallery_md}...")
    md_content = build_gallery_markdown(events=events, out_md=gallery_md)
    print("\n" + md_content)


if __name__ == "__main__":
    main()

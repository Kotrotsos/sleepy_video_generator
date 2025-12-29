#!/usr/bin/env python3
"""
Video Composer for Sleepy Videos
Combines audio narration and images into a video with Ken Burns effect.

Usage:
    python compose_video.py                              # Use most recent run
    python compose_video.py runs/run_xxx                 # Use specific run
    python compose_video.py runs/run_xxx --from 1 --to 53  # Only sections 1-53
"""

import os
import sys
import json
import argparse
import subprocess
import random
from pathlib import Path
from datetime import datetime


def load_settings(settings_path: str = "video_settings.json") -> dict:
    """Load video settings from JSON file."""
    with open(settings_path, "r") as f:
        return json.load(f)


def get_audio_duration(audio_path: Path) -> float:
    """Get duration of audio file in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def get_ken_burns_filter(
    duration: float,
    fps: int,
    width: int,
    height: int,
    settings: dict,
    segment_index: int
) -> str:
    """
    Generate FFmpeg zoompan filter for Ken Burns effect.

    Creates smooth, cinematic zoom and pan movements using
    eased interpolation for natural-looking motion.
    """
    kb = settings["ken_burns"]
    total_frames = int(duration * fps)
    max_zoom = kb["max_zoom"]
    pan_speed = kb["pan_speed"]

    # Randomize zoom and pan direction
    if kb["randomize_direction"]:
        random.seed(segment_index)
        zoom_in = random.choice([True, False])
        pan_x_dir = random.choice([-1, 0, 1])
        pan_y_dir = random.choice([-1, 0, 1])
    else:
        zoom_in = True
        pan_x_dir = 1
        pan_y_dir = 0

    # Smooth zoom using linear interpolation based on frame number
    # This creates consistent, smooth zoom throughout the clip
    if zoom_in:
        # Zoom from 1.0 to max_zoom
        start_zoom = 1.0
        end_zoom = max_zoom
    else:
        # Zoom from max_zoom to 1.0
        start_zoom = max_zoom
        end_zoom = 1.0

    # Linear interpolation: start + (end - start) * (frame / total_frames)
    zoom_range = end_zoom - start_zoom
    zoom_expr = f"{start_zoom}+{zoom_range}*(on/{total_frames})"

    # Smooth pan using linear interpolation
    # Pan range is proportional to how much "extra" image we have at current zoom
    # Center base: (iw - iw/zoom) / 2

    if pan_x_dir == 0:
        x_expr = "(iw-iw/zoom)/2"
    else:
        # Drift from center towards edge over duration
        drift_factor = pan_speed * pan_x_dir * 0.25
        x_expr = f"(iw-iw/zoom)/2 + {drift_factor}*(iw-iw/zoom)*(on/{total_frames})"

    if pan_y_dir == 0:
        y_expr = "(ih-ih/zoom)/2"
    else:
        drift_factor = pan_speed * pan_y_dir * 0.25
        y_expr = f"(ih-ih/zoom)/2 + {drift_factor}*(ih-ih/zoom)*(on/{total_frames})"

    # Build zoompan filter with smooth interpolation
    filter_str = (
        f"zoompan="
        f"z='{zoom_expr}':"
        f"x='{x_expr}':"
        f"y='{y_expr}':"
        f"d={total_frames}:"
        f"s={width}x{height}:"
        f"fps={fps}"
    )

    return filter_str


def get_title_filter(title: str, settings: dict) -> str:
    """
    Generate FFmpeg drawtext filter for title overlay with fade out.
    """
    title_settings = settings.get("title", {})

    if not title_settings.get("enabled", False):
        return ""

    display_duration = title_settings.get("display_duration", 10.0)
    fade_duration = title_settings.get("fade_out_duration", 2.0)
    font_size = title_settings.get("font_size", 56)
    font_color = title_settings.get("font_color", "white")
    font = title_settings.get("font", "Arial")
    shadow = title_settings.get("shadow", True)
    shadow_color = title_settings.get("shadow_color", "black@0.6")

    # Escape special characters for FFmpeg
    escaped_title = title.replace("'", "'\\''").replace(":", "\\:")

    # Calculate fade start time (when to start fading out)
    fade_start = display_duration - fade_duration

    # Alpha expression: full opacity until fade_start, then fade to 0
    alpha_expr = f"if(lt(t\\,{fade_start})\\,1\\,max(0\\,({display_duration}-t)/{fade_duration}))"

    # Build drawtext filter
    filter_parts = [
        f"drawtext=text='{escaped_title}'",
        f"fontsize={font_size}",
        f"fontcolor={font_color}",
        "x=(w-text_w)/2",
        "y=(h-text_h)/2",
        f"alpha='{alpha_expr}'",
    ]

    if shadow:
        filter_parts.extend([
            f"shadowcolor={shadow_color}",
            "shadowx=3",
            "shadowy=3"
        ])

    return ":".join(filter_parts)


def get_video_duration(video_path: Path) -> float:
    """Get duration of video file in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def create_segment_from_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    settings: dict,
    segment_index: int,
    title: str = None
) -> Path:
    """Create a video segment from video clip and audio with optional title overlay."""
    output_settings = settings["output"]
    width, height = map(int, output_settings["resolution"].split("x"))

    # Get audio and video durations
    audio_duration = get_audio_duration(audio_path)
    video_duration = get_video_duration(video_path)
    print(f"    Audio: {audio_duration:.1f}s, Video clip: {video_duration:.1f}s")

    fps = output_settings["fps"]

    # Build filter chain
    filters = []

    # Scale video to output resolution and normalize frame rate
    filters.append(f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")

    # Handle duration mismatch
    if video_duration < audio_duration:
        # Loop the video to match audio duration
        loop_count = int(audio_duration / video_duration) + 1
        # Use setpts to reset timestamps after loop
        filters.insert(0, f"loop=loop={loop_count}:size=32767:start=0,setpts=N/FRAME_RATE/TB")
        print(f"    Looping video {loop_count}x to match audio")

    # Add title overlay if provided
    if title:
        title_filter = get_title_filter(title, settings)
        if title_filter:
            filters.append(title_filter)

    video_filter = ",".join(filters)

    # FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-filter_complex", f"[0:v]{video_filter}[v]",
        "-map", "[v]",
        "-map", "1:a",
        "-c:v", output_settings["codec"],
        "-preset", output_settings["preset"],
        "-crf", str(output_settings["crf"]),
        "-c:a", output_settings["audio_codec"],
        "-b:a", output_settings["audio_bitrate"],
        "-pix_fmt", output_settings["pixel_format"],
        "-t", str(audio_duration),  # Limit to audio duration
        "-movflags", "+faststart",
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FFmpeg error: {result.stderr}")
        raise RuntimeError(f"Failed to create segment: {result.stderr}")

    return output_path


def create_segment_video(
    image_path: Path,
    audio_path: Path,
    output_path: Path,
    settings: dict,
    segment_index: int,
    title: str = None
) -> Path:
    """Create a video segment from image and audio with optional title overlay."""
    output_settings = settings["output"]
    width, height = map(int, output_settings["resolution"].split("x"))
    fps = output_settings["fps"]

    # Get audio duration
    duration = get_audio_duration(audio_path)
    print(f"    Duration: {duration:.1f}s")

    # Build Ken Burns filter
    kb_filter = get_ken_burns_filter(
        duration, fps, width, height, settings, segment_index
    )

    # Scale and pad the image to fit output resolution
    # First scale to cover the frame (may crop), then apply zoompan
    scale_filter = f"scale={width*2}:{height*2}:force_original_aspect_ratio=increase,crop={width*2}:{height*2}"

    # Build filter chain
    filters = [scale_filter, kb_filter]

    # Add title overlay if provided
    if title:
        title_filter = get_title_filter(title, settings)
        if title_filter:
            filters.append(title_filter)

    video_filter = ",".join(filters)

    # FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-loop", "1",  # Loop image
        "-i", str(image_path),
        "-i", str(audio_path),
        "-filter_complex", video_filter,
        "-c:v", output_settings["codec"],
        "-preset", output_settings["preset"],
        "-crf", str(output_settings["crf"]),
        "-c:a", output_settings["audio_codec"],
        "-b:a", output_settings["audio_bitrate"],
        "-pix_fmt", output_settings["pixel_format"],
        "-shortest",  # End when audio ends
        "-movflags", "+faststart",
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FFmpeg error: {result.stderr}")
        raise RuntimeError(f"Failed to create segment: {result.stderr}")

    return output_path


def concatenate_videos_simple(
    segment_paths: list[Path],
    output_path: Path,
    settings: dict
) -> Path:
    """Simple concatenation without transitions."""
    output_settings = settings["output"]

    concat_file = output_path.parent / "concat_list.txt"
    with open(concat_file, "w") as f:
        for path in segment_paths:
            f.write(f"file '{path.absolute()}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c:v", output_settings["codec"],
        "-preset", output_settings["preset"],
        "-crf", str(output_settings["crf"]),
        "-c:a", output_settings["audio_codec"],
        "-b:a", output_settings["audio_bitrate"],
        "-pix_fmt", output_settings["pixel_format"],
        "-movflags", "+faststart",
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg concat error: {result.stderr}")
        raise RuntimeError(f"Failed to concatenate: {result.stderr}")

    concat_file.unlink()
    return output_path


def concatenate_videos(
    segment_paths: list[Path],
    output_path: Path,
    settings: dict
) -> Path:
    """Concatenate videos with crossfade transitions."""
    output_settings = settings["output"]
    fade_duration = settings["transitions"]["fade_duration"]

    # If no fade or single video, use simple concat
    if fade_duration <= 0 or len(segment_paths) <= 1:
        return concatenate_videos_simple(segment_paths, output_path, settings)

    print(f"  Applying {fade_duration}s crossfade transitions...")

    # Get durations of all segments
    durations = []
    for path in segment_paths:
        dur = get_audio_duration(path)
        durations.append(dur)

    # Build input arguments
    inputs = []
    for path in segment_paths:
        inputs.extend(["-i", str(path)])

    # Build xfade filter chain for video
    # Each xfade takes two inputs and produces one output
    # Offset is cumulative duration minus fade overlaps
    video_filters = []
    audio_filters = []

    n = len(segment_paths)

    # Calculate offsets for each transition
    # offset[i] = sum of durations[0:i+1] - (i+1)*fade_duration
    cumulative = 0
    offsets = []
    for i in range(n - 1):
        cumulative += durations[i]
        offset = cumulative - (i + 1) * fade_duration
        offsets.append(offset)

    # Build video xfade chain
    if n == 2:
        # Simple case: just one xfade
        video_filters.append(
            f"[0:v][1:v]xfade=transition=fade:duration={fade_duration}:offset={offsets[0]}[outv]"
        )
        audio_filters.append(
            f"[0:a][1:a]acrossfade=d={fade_duration}:c1=tri:c2=tri[outa]"
        )
    else:
        # Chain multiple xfades
        # First xfade
        video_filters.append(
            f"[0:v][1:v]xfade=transition=fade:duration={fade_duration}:offset={offsets[0]}[v1]"
        )
        audio_filters.append(
            f"[0:a][1:a]acrossfade=d={fade_duration}:c1=tri:c2=tri[a1]"
        )

        # Middle xfades
        for i in range(2, n - 1):
            video_filters.append(
                f"[v{i-1}][{i}:v]xfade=transition=fade:duration={fade_duration}:offset={offsets[i-1]}[v{i}]"
            )
            audio_filters.append(
                f"[a{i-1}][{i}:a]acrossfade=d={fade_duration}:c1=tri:c2=tri[a{i}]"
            )

        # Last xfade
        video_filters.append(
            f"[v{n-2}][{n-1}:v]xfade=transition=fade:duration={fade_duration}:offset={offsets[n-2]}[outv]"
        )
        audio_filters.append(
            f"[a{n-2}][{n-1}:a]acrossfade=d={fade_duration}:c1=tri:c2=tri[outa]"
        )

    # Combine all filters
    filter_complex = ";".join(video_filters + audio_filters)

    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", output_settings["codec"],
        "-preset", output_settings["preset"],
        "-crf", str(output_settings["crf"]),
        "-c:a", output_settings["audio_codec"],
        "-b:a", output_settings["audio_bitrate"],
        "-pix_fmt", output_settings["pixel_format"],
        "-movflags", "+faststart",
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg crossfade error: {result.stderr}")
        # Fall back to simple concat
        print("  Falling back to simple concatenation...")
        return concatenate_videos_simple(segment_paths, output_path, settings)

    return output_path


def compose_video(
    run_path: str | Path,
    output_name: str = None,
    from_section: int = None,
    to_section: int = None
):
    """
    Main function to compose video from a generation run.

    Args:
        run_path: Path to the run folder (e.g., runs/run_20231224_101053)
        output_name: Optional output filename (default: sleepy_video.mp4)
        from_section: Start from this section (1-based, inclusive)
        to_section: End at this section (1-based, inclusive)
    """
    run_path = Path(run_path)

    print(f"Loading settings...")
    settings = load_settings()

    # Load manifest
    manifest_path = run_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found in {run_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    all_sections = manifest["sections"]
    print(f"Found {len(all_sections)} sections in manifest")

    # Apply section range filter
    start_idx = (from_section - 1) if from_section else 0
    end_idx = to_section if to_section else len(all_sections)

    sections = all_sections[start_idx:end_idx]

    if from_section or to_section:
        print(f"Processing sections {start_idx + 1} to {end_idx} ({len(sections)} sections)")

    # Create segments directory
    segments_dir = run_path / "segments"
    segments_dir.mkdir(exist_ok=True)

    segment_paths = []

    # Process each section
    for i, section in enumerate(sections):
        section_num = section.get("index", i + start_idx + 1)
        title = section.get("title_clean", section.get("title", f"Section {section_num}"))
        print(f"\n[{i + 1}/{len(sections)}] Processing: {title}")

        # Check for audio error
        if "audio_error" in section:
            print(f"  Skipping - audio error: {section['audio_error']}")
            continue

        # Check for visual (either video or image)
        has_video = "video" in section and "video_error" not in section
        has_image = "image" in section and "image_error" not in section

        if not has_video and not has_image:
            error = section.get("video_error") or section.get("image_error") or "no visual"
            print(f"  Skipping - {error}")
            continue

        audio_path = run_path / section["audio"]
        if not audio_path.exists():
            print(f"  Skipping - audio not found: {audio_path}")
            continue
        if audio_path.stat().st_size == 0:
            print(f"  Skipping - audio file is empty: {audio_path}")
            continue

        # Create segment video
        segment_output = segments_dir / f"{section_num:03d}_segment.mp4"

        # Skip if segment already exists and is valid
        if segment_output.exists() and segment_output.stat().st_size > 0:
            print(f"  Segment exists, skipping...")
            segment_paths.append(segment_output)
            continue

        try:
            if has_video:
                # Use video clip from Pixabay
                video_path = run_path / section["video"]
                if not video_path.exists():
                    print(f"  Skipping - video not found: {video_path}")
                    continue
                print(f"  Creating segment from video clip...")
                create_segment_from_video(
                    video_path,
                    audio_path,
                    segment_output,
                    settings,
                    section_num,
                    title=title
                )
            else:
                # Use image with Ken Burns effect
                image_path = run_path / section["image"]
                if not image_path.exists():
                    print(f"  Skipping - image not found: {image_path}")
                    continue
                print(f"  Creating segment from image...")
                create_segment_video(
                    image_path,
                    audio_path,
                    segment_output,
                    settings,
                    section_num,
                    title=title
                )

            segment_paths.append(segment_output)
            print(f"  Segment saved: {segment_output.name}")
        except Exception as e:
            print(f"  Error creating segment: {e}")
            continue

    if not segment_paths:
        raise RuntimeError("No segments were created successfully")

    # Concatenate all segments
    print(f"\nConcatenating {len(segment_paths)} segments...")

    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"sleepy_video_{timestamp}.mp4"

    # If output_name contains a path separator, use it as-is; otherwise join with run_path
    output_path = Path(output_name)
    if output_path.is_absolute() or "/" in output_name:
        final_output = output_path
    else:
        final_output = run_path / output_name
    concatenate_videos(segment_paths, final_output, settings)

    # Get final video info
    duration = get_audio_duration(final_output)
    file_size = final_output.stat().st_size / (1024 * 1024)  # MB

    print(f"\nVideo composition complete!")
    print(f"  Output: {final_output}")
    print(f"  Duration: {duration/60:.1f} minutes")
    print(f"  Size: {file_size:.1f} MB")

    return final_output


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compose video from generation run with Ken Burns effects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compose_video.py                              # Use most recent run
  python compose_video.py runs/run_xxx                 # Use specific run
  python compose_video.py runs/run_xxx --from 1 --to 53  # Only sections 1-53
  python compose_video.py --output my_video.mp4        # Custom output name
        """
    )
    parser.add_argument(
        "run_path",
        nargs="?",
        help="Path to run folder (default: most recent)"
    )
    parser.add_argument(
        "--from",
        dest="from_section",
        type=int,
        help="Start from section number (1-based)"
    )
    parser.add_argument(
        "--to",
        dest="to_section",
        type=int,
        help="End at section number (1-based, inclusive)"
    )
    parser.add_argument(
        "--output", "-o",
        dest="output_name",
        help="Output filename"
    )

    args = parser.parse_args()

    # Determine run path
    if args.run_path:
        run_path = args.run_path
    else:
        runs_dir = Path("runs")
        if not runs_dir.exists():
            print("No runs/ directory found. Specify a run path.")
            sys.exit(1)

        runs = sorted(runs_dir.glob("run_*"), reverse=True)
        if not runs:
            print("No runs found in runs/ directory")
            sys.exit(1)

        run_path = runs[0]
        print(f"Using most recent run: {run_path}")

    compose_video(
        run_path,
        output_name=args.output_name,
        from_section=args.from_section,
        to_section=args.to_section
    )


if __name__ == "__main__":
    main()

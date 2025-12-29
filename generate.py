#!/usr/bin/env python3
"""
Sleepy Video Generator
Generates audio narration and images for knowledge videos.

Usage:
    python generate.py                           # Full generation, new run
    python generate.py --run runs/run_xxx        # Continue in existing run folder
    python generate.py --from 53                 # Start from section 53
    python generate.py --from 53 --to 60         # Generate sections 53-60
    python generate.py --from 53 --audio-only    # Only regenerate audio from section 53
    python generate.py --from 53 --images-only   # Only regenerate images from section 53
    python generate.py --run runs/run_xxx --from 53 --audio-only  # Fix audio in existing run
"""

import os
import re
import json
import time
import argparse
import struct
import requests
from datetime import datetime
from typing import Optional
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

# Import APIs
try:
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    ElevenLabs = None

from google import genai
from google.genai import types
from PIL import Image


def load_settings(settings_path: str = "settings.json") -> dict:
    """Load settings from JSON file."""
    with open(settings_path, "r") as f:
        return json.load(f)


def parse_narration(content: str, delimiter: str = "## ") -> list[dict]:
    """
    Parse narration markdown file into sections.
    Each section contains title and content.
    """
    sections = []

    # Split by ## headers
    parts = re.split(r'\n(?=## )', content)

    for part in parts:
        part = part.strip()
        if not part or not part.startswith("## "):
            continue

        lines = part.split("\n", 1)
        title_line = lines[0].replace("## ", "").strip()
        content_text = lines[1].strip() if len(lines) > 1 else ""

        # Clean up the title (remove Roman numerals prefix like "I. ", "II. ")
        title_clean = re.sub(r'^[IVXLC]+\.\s*', '', title_line)

        sections.append({
            "title": title_line,
            "title_clean": title_clean,
            "content": content_text
        })

    return sections


def extract_narration_text(content: str) -> str:
    """
    Extract plain narration text, removing SSML break tags and markdown.
    """
    # Remove <break time="Xs" /> tags
    text = re.sub(r'<break\s+time="[^"]+"\s*/>', '', content)
    # Remove --- separators
    text = re.sub(r'\n---\n', '\n', text)
    # Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def generate_image_prompt(title: str, content: str, style: str) -> str:
    """
    Generate an image prompt based on the section content.
    """
    # Extract key visual elements from the narration
    prompt = f"{style}. Scene: {title}. "

    # Add context from content (first few sentences for context)
    clean_content = extract_narration_text(content)
    sentences = clean_content.split('.')[:3]
    context = '. '.join(sentences)

    prompt += f"Inspired by: {context}"

    return prompt


def create_run_folder(base_dir: str, visual_provider: str = "imagen") -> Path:
    """
    Create a timestamped folder for this run.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = Path(base_dir) / f"run_{timestamp}"
    run_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (run_path / "audio").mkdir(exist_ok=True)
    if visual_provider == "pixabay":
        (run_path / "videos").mkdir(exist_ok=True)
    else:
        (run_path / "images").mkdir(exist_ok=True)

    return run_path


def generate_audio_elevenlabs(
    client: ElevenLabs,
    text: str,
    output_path: Path,
    settings: dict
) -> Path:
    """
    Generate audio using ElevenLabs TTS.
    """
    el_settings = settings["elevenlabs"]

    audio = client.text_to_speech.convert(
        text=text,
        voice_id=el_settings["voice_id"],
        model_id=el_settings["model_id"],
        output_format=el_settings["output_format"],
        voice_settings={
            "stability": el_settings.get("stability", 0.5),
            "similarity_boost": el_settings.get("similarity_boost", 0.75),
            "style": el_settings.get("style", 0.0),
            "use_speaker_boost": el_settings.get("use_speaker_boost", True)
        }
    )

    # Write audio to file
    with open(output_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    return output_path


def parse_audio_mime_type(mime_type: str) -> dict:
    """
    Parse bits per sample and rate from an audio MIME type string.
    """
    bits_per_sample = 16
    rate = 24000

    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass

    return {"bits_per_sample": bits_per_sample, "rate": rate}


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """
    Convert raw PCM audio data to WAV format.
    """
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size
    )
    return header + audio_data


def generate_audio_google(
    client: genai.Client,
    text: str,
    output_path: Path,
    settings: dict
) -> Path:
    """
    Generate audio using Google Gemini TTS.
    """
    google_tts = settings.get("google_tts", {})
    model = google_tts.get("model", "gemini-2.5-flash-preview-tts")
    voice_name = google_tts.get("voice_name", "Kore")

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=text)],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name
                )
            )
        ),
    )

    # Collect all audio chunks
    audio_chunks = []
    mime_type = None

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue

        part = chunk.candidates[0].content.parts[0]
        if part.inline_data and part.inline_data.data:
            audio_chunks.append(part.inline_data.data)
            if mime_type is None:
                mime_type = part.inline_data.mime_type

    if not audio_chunks:
        raise ValueError("No audio generated in response")

    # Combine all chunks
    combined_audio = b"".join(audio_chunks)

    # Convert to WAV
    wav_data = convert_to_wav(combined_audio, mime_type or "audio/L16;rate=24000")

    # Save as WAV (change extension)
    wav_path = output_path.with_suffix(".wav")
    with open(wav_path, "wb") as f:
        f.write(wav_data)

    return wav_path


def generate_image(
    client: genai.Client,
    prompt: str,
    output_path: Path,
    model: str = "gemini-2.5-flash-image-preview"
) -> Path:
    """
    Generate image using Google Gemini image generation.
    """
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    # Extract image parts from response
    image_parts = [
        part.inline_data.data
        for part in response.candidates[0].content.parts
        if part.inline_data
    ]

    if image_parts:
        image = Image.open(BytesIO(image_parts[0]))
        image.save(output_path)
        return output_path

    raise ValueError("No image generated in response")


def search_pixabay_videos(query: str, settings: dict) -> Optional[dict]:
    """
    Search Pixabay for videos matching the query.
    Returns the first matching video info or None.
    """
    pixabay = settings.get("pixabay", {})
    api_key = os.getenv("PIXABAY_API_KEY")

    if not api_key:
        raise ValueError("PIXABAY_API_KEY environment variable not set")

    params = {
        "key": api_key,
        "q": query,
        "video_type": pixabay.get("video_type", "film"),
        "min_width": pixabay.get("min_width", 1920),
        "min_height": pixabay.get("min_height", 1080),
        "safesearch": "true" if pixabay.get("safesearch", True) else "false",
        "editors_choice": "true" if pixabay.get("editors_choice", False) else "false",
        "per_page": 3,
        "order": "popular"
    }

    response = requests.get("https://pixabay.com/api/videos/", params=params)
    response.raise_for_status()

    data = response.json()

    if data.get("hits") and len(data["hits"]) > 0:
        video = data["hits"][0]
        quality = pixabay.get("quality", "medium")
        video_data = video.get("videos", {}).get(quality, {})

        return {
            "id": video.get("id"),
            "url": video_data.get("url"),
            "width": video_data.get("width"),
            "height": video_data.get("height"),
            "duration": video.get("duration"),
            "tags": video.get("tags"),
            "user": video.get("user")
        }

    return None


def download_pixabay_video(video_info: dict, output_path: Path) -> Path:
    """
    Download a video from Pixabay to the specified path.
    """
    if not video_info or not video_info.get("url"):
        raise ValueError("No video URL provided")

    response = requests.get(video_info["url"], stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return output_path


def generate_video_search_terms(title: str, content: str) -> list[str]:
    """
    Generate search terms for finding relevant videos.
    Returns a list of terms to try in order of specificity.
    """
    # Clean the title
    clean_title = re.sub(r'^[IVXLC]+\.\s*', '', title)
    clean_title = re.sub(r"['\"]", '', clean_title)

    # Extract key words from title (remove common words)
    stop_words = {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'is', 'are', 'was', 'were'}
    title_words = [w.lower() for w in clean_title.split() if w.lower() not in stop_words]

    search_terms = []

    # Full title (cleaned)
    if clean_title:
        search_terms.append(clean_title)

    # Key words combined
    if len(title_words) >= 2:
        search_terms.append(' '.join(title_words[:3]))

    # Individual key words (for fallback)
    for word in title_words[:2]:
        if len(word) > 3:
            search_terms.append(word)

    return search_terms


def sanitize_filename(name: str) -> str:
    """
    Convert title to safe filename.
    """
    # Remove special characters
    safe = re.sub(r'[^\w\s-]', '', name)
    # Replace spaces with underscores
    safe = re.sub(r'\s+', '_', safe)
    return safe.lower()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate audio and images for knowledge videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py                           # Full generation, new run
  python generate.py --run runs/run_xxx        # Continue in existing run folder
  python generate.py --from 53                 # Start from section 53
  python generate.py --from 53 --to 60         # Generate sections 53-60
  python generate.py --from 53 --audio-only    # Only regenerate audio from section 53
  python generate.py --from 53 --images-only   # Only regenerate images from section 53
  python generate.py --run runs/run_xxx --from 53 --audio-only  # Fix audio in existing run
        """
    )
    parser.add_argument(
        "--run",
        type=str,
        help="Path to existing run folder to continue generation"
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
        "--audio-only",
        action="store_true",
        help="Only generate audio files"
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Only generate image files"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=0,
        help="Delay in seconds between sections (for rate limiting)"
    )
    return parser.parse_args()


def load_manifest(run_path: Path) -> Optional[dict]:
    """Load existing manifest from run folder."""
    manifest_path = run_path / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            return json.load(f)
    return None


def main():
    """
    Main generation pipeline.
    """
    args = parse_args()

    # Validate mutually exclusive options
    if args.audio_only and args.images_only:
        print("Error: Cannot use both --audio-only and --images-only")
        return

    print("Loading settings...")
    settings = load_settings()

    # Determine TTS provider
    tts_provider = settings.get("tts_provider", "google")
    print(f"TTS Provider: {tts_provider}")

    # Determine visual provider
    visual_provider = settings.get("visual_provider", "imagen")
    print(f"Visual Provider: {visual_provider}")

    # Initialize clients based on what we need to generate
    print("Initializing API clients...")
    elevenlabs_client = None
    google_client = None

    if not args.images_only:
        if tts_provider == "elevenlabs":
            if not ELEVENLABS_AVAILABLE:
                print("Error: ElevenLabs not installed. Use 'pip install elevenlabs' or switch to Google TTS.")
                return
            elevenlabs_client = ElevenLabs(
                api_key=os.getenv("ELEVENLABS_API_KEY")
            )
        else:
            # Google TTS uses the same client as image generation
            google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    if not args.audio_only:
        # Need Google client for Gemini images (not for Pixabay videos)
        if visual_provider == "imagen" and google_client is None:
            google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    # Load narration
    print(f"Loading narration from {settings['narration']['source_file']}...")
    with open(settings["narration"]["source_file"], "r") as f:
        narration_content = f.read()

    # Parse sections
    sections = parse_narration(
        narration_content,
        settings["narration"]["section_delimiter"]
    )
    print(f"Found {len(sections)} sections")

    # Determine run folder
    if args.run:
        run_path = Path(args.run)
        if not run_path.exists():
            print(f"Error: Run folder does not exist: {run_path}")
            return
        print(f"Using existing run folder: {run_path}")
        # Load existing manifest
        manifest = load_manifest(run_path)
        if manifest is None:
            manifest = {
                "created": datetime.now().isoformat(),
                "settings": settings,
                "sections": []
            }
    else:
        run_path = create_run_folder(settings["output"]["base_dir"], visual_provider)
        print(f"Output folder: {run_path}")
        manifest = {
            "created": datetime.now().isoformat(),
            "settings": settings,
            "sections": []
        }

    # Determine section range
    start_section = args.from_section if args.from_section else 1
    end_section = args.to_section if args.to_section else len(sections)

    # Validate range
    if start_section < 1 or start_section > len(sections):
        print(f"Error: --from {start_section} is out of range (1-{len(sections)})")
        return
    if end_section < start_section or end_section > len(sections):
        print(f"Error: --to {end_section} is out of range ({start_section}-{len(sections)})")
        return

    if start_section > 1 or end_section < len(sections):
        print(f"Processing sections {start_section} to {end_section}")

    # Build a lookup of existing manifest sections by index
    existing_sections = {}
    if manifest.get("sections"):
        for sec in manifest["sections"]:
            existing_sections[sec["index"]] = sec

    # Process each section in range
    for i, section in enumerate(sections, 1):
        # Skip sections outside the range
        if i < start_section or i > end_section:
            continue

        print(f"\n[{i}/{len(sections)}] Processing: {section['title']}")

        safe_name = sanitize_filename(section['title_clean'])

        # Get or create section data
        section_data = existing_sections.get(i, {
            "index": i,
            "title": section["title"],
            "title_clean": section["title_clean"]
        })
        section_data["index"] = i
        section_data["title"] = section["title"]
        section_data["title_clean"] = section["title_clean"]

        # Generate audio (unless images-only)
        if not args.images_only:
            print(f"  Generating audio ({tts_provider})...")
            # Set extension based on provider
            audio_ext = ".wav" if tts_provider == "google" else ".mp3"
            audio_path = run_path / "audio" / f"{i:02d}_{safe_name}{audio_ext}"
            try:
                if tts_provider == "elevenlabs":
                    generate_audio_elevenlabs(
                        elevenlabs_client,
                        section["content"],
                        audio_path,
                        settings
                    )
                else:
                    audio_path = generate_audio_google(
                        google_client,
                        section["content"],
                        audio_path,
                        settings
                    )
                section_data["audio"] = str(audio_path.relative_to(run_path))
                section_data.pop("audio_error", None)  # Clear any previous error
                print(f"  Audio saved: {audio_path.name}")
            except Exception as e:
                print(f"  Audio generation failed: {e}")
                section_data["audio_error"] = str(e)

        # Generate visual (image or video, unless audio-only)
        if not args.audio_only:
            if visual_provider == "pixabay":
                # Download video from Pixabay
                print(f"  Searching for video...")
                video_path = run_path / "videos" / f"{i:02d}_{safe_name}.mp4"

                # Generate search terms from title
                search_terms = generate_video_search_terms(
                    section["title_clean"],
                    section["content"]
                )

                video_info = None
                used_query = None
                for query in search_terms:
                    try:
                        video_info = search_pixabay_videos(query, settings)
                        if video_info and video_info.get("url"):
                            used_query = query
                            break
                    except Exception as e:
                        print(f"    Search '{query}' failed: {e}")
                        continue

                if video_info and video_info.get("url"):
                    try:
                        print(f"  Downloading video (query: '{used_query}')...")
                        download_pixabay_video(video_info, video_path)
                        section_data["video"] = str(video_path.relative_to(run_path))
                        section_data["video_query"] = used_query
                        section_data["video_duration"] = video_info.get("duration")
                        section_data["video_source"] = f"pixabay:{video_info.get('id')}"
                        section_data.pop("video_error", None)
                        print(f"  Video saved: {video_path.name} ({video_info.get('duration')}s)")
                    except Exception as e:
                        print(f"  Video download failed: {e}")
                        section_data["video_error"] = str(e)
                else:
                    print(f"  No video found for any search term")
                    section_data["video_error"] = f"No video found for: {search_terms}"
            else:
                # Generate image from Gemini
                print(f"  Generating image...")
                image_path = run_path / "images" / f"{i:02d}_{safe_name}.png"
                image_prompt = generate_image_prompt(
                    section["title_clean"],
                    section["content"],
                    settings["imagen"]["image_style"]
                )
                section_data["image_prompt"] = image_prompt

                try:
                    generate_image(
                        google_client,
                        image_prompt,
                        image_path,
                        settings["imagen"]["model"]
                    )
                    section_data["image"] = str(image_path.relative_to(run_path))
                    section_data.pop("image_error", None)
                    print(f"  Image saved: {image_path.name}")
                except Exception as e:
                    print(f"  Image generation failed: {e}")
                    section_data["image_error"] = str(e)

        existing_sections[i] = section_data

        # Save manifest after each section (for restart capability)
        manifest["sections"] = [existing_sections[idx] for idx in sorted(existing_sections.keys())]
        manifest["updated"] = datetime.now().isoformat()
        manifest["last_completed"] = i
        manifest_path = run_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Delay between sections if specified
        if args.delay > 0 and i < end_section:
            print(f"  Waiting {args.delay}s before next section...")
            time.sleep(args.delay)

    # Rebuild manifest sections in order
    manifest["sections"] = [existing_sections[i] for i in sorted(existing_sections.keys())]
    manifest["updated"] = datetime.now().isoformat()

    # Save manifest
    manifest_path = run_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")

    print(f"\nGeneration complete! Output: {run_path}")


if __name__ == "__main__":
    main()

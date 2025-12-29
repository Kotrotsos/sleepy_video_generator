# Sleepy Video Generator

*Generate hour-long documentaries from a single prompt, with zero effort*

## What is this?

An automated pipeline that transforms simple curiosity into hours of calm, slow, thoughtful video. A machine for producing ambient knowledge streams perfect for sleep, study, or quiet contemplation.

## The Philosophy

Our digital world is a relentless firehose of hyper-stimulation. TikTok, Reels, Shorts, an endless torrent designed to fracture attention into 15-second fragments. The same generative AI that powers this frenzy can create its antidote.

This project is an act of technological disobedience: using the tools of the attention economy to build a sanctuary from it.

The aesthetic is intentional:
- **Slow speech** with natural pauses
- **Photorealistic imagery** that doesn't demand attention
- **Ken Burns effects** for gentle, almost imperceptible movement
- **Long crossfades** between segments, never jarring cuts
- **Educational content** delivered like a whisper, not a shout

The result feels like a BBC nature documentary you can fall asleep to. Facts wash over you gently. You learn something, or you drift off. Both outcomes are victories.

## How It Works

1. Write narration in markdown (or have Claude generate it)
2. Run `generate.py` to create audio (via Google Gemini or ElevenLabs TTS) and visuals (via Gemini images or Pixabay videos)
3. Run `compose_video.py` to stitch everything into a finished video with titles, transitions, and Ken Burns effects

Total hands-on time: about 5 minutes.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API keys (copy .env.example to .env and fill in)
cp .env.example .env

# Generate assets (audio + images/videos)
python generate.py

# Compose video with transitions and titles
python compose_video.py runs/run_YYYYMMDD_HHMMSS
```

## Configuration

Edit `settings.json` to configure:
- **tts_provider**: `"google"` (free) or `"elevenlabs"` (premium quality)
- **visual_provider**: `"imagen"` (AI-generated images) or `"pixabay"` (stock videos)
- **narration.source_file**: path to your narration markdown

Edit `video_settings.json` to control the aesthetic:
- Ken Burns zoom rate and direction
- Transition fade duration
- Title card appearance and timing

## API Keys Required

- **GOOGLE_API_KEY**: For Gemini TTS and image generation (free tier available)
- **ELEVENLABS_API_KEY**: Optional, for premium voice quality
- **PIXABAY_API_KEY**: Optional, for stock video integration

## Example Output

For "Slow Space: Meditations on the Cosmos" with 103 sections:
- 88 audio files using Google Gemini TTS
- 94 photorealistic space images
- Final video: ~30 minutes

## Advanced Usage

```bash
# Generate a specific range (useful for batching around API limits)
python generate.py --from 1 --to 50

# Resume generation from where you left off
python generate.py --run runs/run_xxx --from 51

# Regenerate only audio for specific sections
python generate.py --run runs/run_xxx --from 10 --to 15 --audio-only

# Compose only a subset of sections
python compose_video.py runs/run_xxx --from 1 --to 50
```

## License

MIT

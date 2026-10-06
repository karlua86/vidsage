---
name: vidsage-watch
description: Watch a video (YouTube/Vimeo/TikTok URL or local file) by pulling its transcript and a few deduplicated key frames, then answer questions about it. Use when the user shares a video link/file and asks what it says or shows.
---

# vidsage-watch

Runs VidSage's `watch.py` to turn a video into evidence you can read: a timestamped
transcript (YouTube captions first, local Whisper fallback) plus saved JPEG frames.

## Setup
Set `VIDSAGE_DIR` to your VidSage folder (the one containing `watch.py`), e.g.
`C:\Users\<you>\Documents\vidsage`. Needs `ffmpeg` on PATH and `pip install -r requirements.txt`.

## How to use
1. Run (always start with the transcript — it is cheap):
   `python "$VIDSAGE_DIR/watch.py" "<url-or-path>" --transcript-only`
2. Read the transcript. If the answer needs visuals, run again for frames:
   `python "$VIDSAGE_DIR/watch.py" "<url-or-path>" --max-frames 15`
3. Open the listed `.jpg` paths with the Read tool and combine them with the transcript.
4. When the transcript says "as you can see here" at 14:05, grab that exact moment:
   `... --at 14:05,14:40`

## Options
- `--start MM:SS --end MM:SS` — only analyse a window (long videos). Timestamps stay on the original timeline.
- `--mode dense|scene|interval` — dense (default) = sample + drop near-duplicates, best for slides/lectures.
- `--dedup 0.80–0.92` — lower drops more near-identical frames.
- Captions default to English; the report lists other available languages — pass `--lang CODE` (e.g. `ms`, `zh`, `ar`) for those.
- `--hotwords "a, b"` and `--xxl-filter denoise|rnnoise|boost|voice` apply to `--stt xxl` (the filters did not help on clear calls — leave off unless the audio is noisy).
- `--stt xxl` — transcribe on the local GPU with the user's Faster-Whisper-XXL install (free, private; folder from FW_XXL_PATH or auto-detected; model via `--fw-model`, default large-v2).
- `--stt groq|openai` — fast cloud transcription when a video has no captions (uploads the AUDIO only; needs GROQ_API_KEY / OPENAI_API_KEY in the environment or in VidSage's .streamlit/secrets.toml). Default is free local Whisper.
- `--max-frames N`, `--interval SECONDS`, `--lang xx`, `--whisper-model base|small|medium|large`.

Videos over ~10 min: use the transcript first, then pick `--start/--end` windows for frames.

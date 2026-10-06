"""
Cloud speech-to-text (Groq / OpenAI Whisper) — UI-free, shared by app.py and watch.py.

Only the AUDIO track is uploaded (never the video).  Audio is re-encoded to small mono MP3 chunks
so each request stays far below the 25 MB limit that both services enforce on their basic tier.
Timestamps are shifted back so they match the original timeline.
"""
import os
import subprocess
import tempfile
import time

# price_per_hour is only used for the cost hint shown in the UI.
PROVIDERS = {
    "groq":   {"label": "Groq",   "base_url": "https://api.groq.com/openai/v1",
               "model": "whisper-large-v3", "price_per_hour": 0.111},
    "openai": {"label": "OpenAI", "base_url": None,
               "model": "whisper-1", "price_per_hour": 0.36},
}

CHUNK_SECONDS = 1800        # 30 min  →  ~7 MB at 32 kbps (limit 25 MB)
MP3_BITRATE = "32k"
PROMPT_MAX_CHARS = 600      # Whisper prompts are limited to ~224 tokens

# Fallback map when the `whisper` package isn't importable (name → ISO code)
_FALLBACK_LANGS = {
    "english": "en", "chinese": "zh", "malay": "ms", "indonesian": "id", "tamil": "ta",
    "japanese": "ja", "korean": "ko", "thai": "th", "vietnamese": "vi", "hindi": "hi",
    "arabic": "ar", "spanish": "es", "french": "fr", "german": "de", "portuguese": "pt",
    "russian": "ru", "italian": "it", "dutch": "nl", "cantonese": "yue",
}


def lang_code_from_name(name: str | None) -> str | None:
    """'english' → 'en'.  Cloud Whisper returns full names; the app uses ISO codes."""
    if not name:
        return None
    key = name.strip().lower()
    if len(key) <= 3:
        return key
    try:
        from whisper.tokenizer import LANGUAGES            # {code: name}
        for code, lang in LANGUAGES.items():
            if lang == key:
                return code
    except Exception:
        pass
    return _FALLBACK_LANGS.get(key)


def audio_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def _encode_chunk(src: str, dst: str, start: float, length: float) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
         "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-b:a", MP3_BITRATE, dst],
        capture_output=True, check=True)


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def transcribe_cloud(audio_path: str, provider: str, api_key: str,
                     language: str | None = None, prompt: str = "",
                     progress=None) -> tuple:
    """Transcribe with Groq or OpenAI Whisper.
    Returns (segments, language_code) where segments = [{'start','end','text'}, ...].
    progress(done_chunks, total_chunks) is called after each chunk if given."""
    from openai import OpenAI

    if provider not in PROVIDERS:
        raise ValueError(f"Unknown cloud transcription provider: {provider}")
    if not api_key:
        raise ValueError(f"No {PROVIDERS[provider]['label']} API key — add it in the sidebar "
                         f"(Transcription engine) or in .streamlit/secrets.toml.")
    cfg = PROVIDERS[provider]
    base_url = os.environ.get("CLOUD_STT_BASE_URL") or cfg["base_url"]
    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=3, timeout=300)

    total = audio_duration(audio_path)
    starts = [i * CHUNK_SECONDS for i in range(max(1, int(-(-total // CHUNK_SECONDS))))]
    segments, detected = [], None

    with tempfile.TemporaryDirectory() as tmp:
        for idx, start in enumerate(starts):
            length = min(CHUNK_SECONDS, total - start)
            if length < 0.5:
                continue
            chunk = os.path.join(tmp, f"chunk_{idx}.mp3")
            _encode_chunk(audio_path, chunk, start, length)

            kwargs = dict(model=cfg["model"], response_format="verbose_json",
                          timestamp_granularities=["segment"])
            if language:
                kwargs["language"] = language
            if prompt:
                kwargs["prompt"] = prompt[:PROMPT_MAX_CHARS]

            for attempt in range(3):
                try:
                    with open(chunk, "rb") as fh:
                        resp = client.audio.transcriptions.create(file=(os.path.basename(chunk), fh), **kwargs)
                    break
                except Exception as exc:                      # 429 / transient 5xx
                    if attempt < 2 and any(c in str(exc) for c in ("429", "500", "502", "503")):
                        time.sleep(5 * (attempt + 1))
                        continue
                    raise

            detected = detected or _get(resp, "language")
            for seg in (_get(resp, "segments") or []):
                text = str(_get(seg, "text", "")).strip()
                if text:
                    segments.append({"start": float(_get(seg, "start", 0)) + start,
                                     "end": float(_get(seg, "end", 0)) + start,
                                     "text": text})
            if progress:
                progress(idx + 1, len(starts))

    return segments, (lang_code_from_name(detected) or language or "en")

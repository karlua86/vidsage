#!/usr/bin/env python
"""
watch.py — let Claude Code "watch" a video using VidSage's pipeline.

Turns a video (local file or URL) into evidence Claude can read:
  * a timestamped transcript  (captions first, local Whisper as fallback)
  * a handful of deduplicated, text-rich frames saved as JPEG files

Prints a Markdown report to stdout; Claude then opens the listed frame
images with its Read tool.  Timestamps always refer to the ORIGINAL video,
even when --start/--end select only a window.

Examples
  python watch.py https://youtu.be/XXXXXXXXXXX
  python watch.py lecture.mp4 --start 12:30 --end 18:00 --max-frames 12
  python watch.py lecture.mp4 --at 14:05,14:40          # "look here" frames
  python watch.py https://youtu.be/XXXXXXXXXXX --transcript-only
"""
import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

import cv2

from frames_core import (
    fmt_time, frame_is_duplicate, frame_signature, frame_similarity, parse_timecode,
    score_text_density, trim_clip,
)

CAPTION_LANGS: list = []   # other caption languages available (filled by fetch_captions)
YOUTUBE_ID = re.compile(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ── Source handling ──────────────────────────────────────────────────────────
def is_url(src: str) -> bool:
    return bool(re.match(r"https?://", src, re.I))


def download_video(url: str, out_dir: str) -> str:
    import yt_dlp
    opts = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "outtmpl": os.path.join(out_dir, "source.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True, "no_warnings": True, "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=True)
    for name in os.listdir(out_dir):
        if name.startswith("source."):
            return os.path.join(out_dir, name)
    raise FileNotFoundError("yt-dlp finished but produced no file (private/geo-blocked?)")


# ── Transcript: captions first, Whisper fallback ─────────────────────────────
def fetch_captions(url: str, lang: str | None):
    """Return (segments, label) from YouTube captions, or None."""
    m = YOUTUBE_ID.search(url)
    if not m:
        return None
    vid = m.group(1)
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):      # 0.x API
            tlist = YouTubeTranscriptApi.list_transcripts(vid)
        else:                                                      # 1.x API
            tlist = YouTubeTranscriptApi().list(vid)
        cands = []
        # No --lang given: prefer English rather than whichever track YouTube lists first
        # (that once returned Arabic for an English video).  Use --lang for other languages.
        for want in ([lang] if lang else ["en"]):
            for finder in (tlist.find_manually_created_transcript,
                           tlist.find_generated_transcript):
                try:
                    cands.append(finder([want]))
                except Exception:
                    pass
        cands += [t for t in tlist if t not in cands]
        if not cands:
            return None
        tr = cands[0]
        CAPTION_LANGS[:] = sorted({t.language_code for t in tlist
                                   if t.language_code != tr.language_code})
        raw = tr.fetch()
        segs = []
        for e in raw:
            get = e.get if isinstance(e, dict) else (lambda k, d=None, e=e: getattr(e, k, d))
            start = float(get("start", 0))
            segs.append({"start": start,
                         "end": start + float(get("duration", 2)),
                         "text": str(get("text", "")).replace("\n", " ").strip()})
        kind = "auto-generated" if tr.is_generated else "manual"
        return segs, f"YouTube captions ({tr.language}, {kind})"
    except Exception:
        return None


def _cloud_key(provider: str) -> str:
    """API key from the environment, else from VidSage's .streamlit/secrets.toml."""
    name = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}[provider]
    if os.environ.get(name):
        return os.environ[name]
    try:
        import tomllib
        secrets = Path(__file__).with_name(".streamlit") / "secrets.toml"
        return tomllib.loads(secrets.read_text(encoding="utf-8")).get(name, "")
    except Exception:
        return ""


def _xxl_configured() -> str:
    try:
        import tomllib
        secrets = Path(__file__).with_name(".streamlit") / "secrets.toml"
        return tomllib.loads(secrets.read_text(encoding="utf-8")).get("FW_XXL_PATH", "")
    except Exception:
        return ""


FILTER_ALIASES = {"off": "Off", "denoise": "Light denoise", "rnnoise": "Remove non-speech (RNNoise)",
                  "boost": "Boost quiet voices", "voice": "Isolate voice (strong, slow)"}


def whisper_transcribe(clip_path: str, model_size: str, lang: str | None, stt: str = "local",
                       fw_model: str = "large-v2", hotwords: str = "", xxl_filter: str = "off"):
    from subprocess import run
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio.wav")
        run(["ffmpeg", "-y", "-i", clip_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", wav], capture_output=True, check=True)
        if stt == "xxl":
            from xxl_stt import SETUP_HINT, find_xxl, transcribe_xxl
            exe = find_xxl(_xxl_configured())
            if exe is None:
                raise RuntimeError(SETUP_HINT)
            segs, code = transcribe_xxl(wav, exe, fw_model, language=lang, hotwords=hotwords,
                                        noise_filter=FILTER_ALIASES.get(xxl_filter, 'Off'))
            return segs, f"Faster-Whisper-XXL ({fw_model}, local GPU, language: {code})"
        if stt != "local":
            from cloud_stt import PROVIDERS, transcribe_cloud
            segs, code = transcribe_cloud(wav, stt, _cloud_key(stt), language=lang)
            return segs, f"{PROVIDERS[stt]['label']} cloud Whisper ({PROVIDERS[stt]['model']}, language: {code})"
        import whisper
        model = whisper.load_model(model_size)
        res = model.transcribe(wav, verbose=False, task="transcribe", language=lang)
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in res.get("segments", [])]
    return segs, f"local Whisper ({model_size}, language: {res.get('language')})"


# ── Frames ───────────────────────────────────────────────────────────────────
def save_frame(frame, path: str, max_dim: int = 1280) -> None:
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        k = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])


def sample_frames(video: str, mode: str, interval: float, threshold: float,
                  dedup: float, max_frames: int):
    """Return [(seconds_in_video, frame)] — deduplicated, most text-rich kept."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(int(fps * (0.5 if mode == "scene" else interval)), 1)

    cands, hashes, prev = [], [], None
    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        if mode == "scene":
            sig = frame_signature(frame)
            changed = prev is None or (1.0 - frame_similarity(sig, prev)) >= threshold
            prev = sig
            if not changed:
                continue
        ph = frame_signature(frame)
        if mode != "interval" and frame_is_duplicate(ph, hashes, dedup):
            continue
        hashes.append(ph)
        cands.append((idx / fps, frame))
    cap.release()

    if len(cands) > max_frames:
        if mode == "interval":      # evenly spaced
            pick = [cands[round(i * (len(cands) - 1) / (max_frames - 1))]
                    for i in range(max_frames)] if max_frames > 1 else cands[:1]
            cands = pick
        else:                       # most text-rich, back in time order
            cands = sorted(sorted(cands, key=lambda c: -score_text_density(c[1]))[:max_frames],
                           key=lambda c: c[0])
    return cands


def grab_at(video: str, seconds_in_video: float):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(seconds_in_video, 0) * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    for stream in (sys.stdout, sys.stderr):      # Windows consoles default to cp1252
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="video file path or URL (YouTube, Vimeo, TikTok, …)")
    ap.add_argument("--start", help="window start (SS, MM:SS or HH:MM:SS)")
    ap.add_argument("--end", help="window end")
    ap.add_argument("--mode", choices=["dense", "scene", "interval"], default="dense",
                    help="dense=sample+dedup (slides/lectures), scene=scene changes, "
                         "interval=evenly spaced (default: dense)")
    ap.add_argument("--max-frames", type=int, default=20)
    ap.add_argument("--interval", type=float, default=3.0, help="sampling step in seconds")
    ap.add_argument("--threshold", type=float, default=0.4, help="scene-change sensitivity")
    ap.add_argument("--dedup", type=float, default=0.88,
                    help="similarity at/above which frames count as duplicates (0.80–0.92)")
    ap.add_argument("--at", help="comma-separated timestamps to grab exact frames, e.g. 14:05,14:40")
    ap.add_argument("--transcript-only", action="store_true", help="skip frames")
    ap.add_argument("--no-transcript", action="store_true", help="skip transcript")
    ap.add_argument("--stt", choices=["local", "xxl", "groq", "openai"], default="local",
                    help="speech-to-text when there are no captions: local Whisper (free), xxl = your local "
                         "Faster-Whisper-XXL install on the GPU (free, private; folder from FW_XXL_PATH or "
                         ".streamlit/secrets.toml), or cloud "
                         "Groq/OpenAI (fast; uploads the AUDIO only; key from GROQ_API_KEY / "
                         "OPENAI_API_KEY or .streamlit/secrets.toml)")
    ap.add_argument("--fw-model", default="large-v2", help="model for --stt xxl (default large-v2)")
    ap.add_argument("--hotwords", default="", help="names/terms to favour with --stt xxl, comma-separated")
    ap.add_argument("--xxl-filter", choices=list(FILTER_ALIASES), default="off",
                    help="audio clean-up for --stt xxl: denoise | rnnoise | boost | voice (leave off unless the "
                         "recording is noisy — it did not help on clear calls)")
    ap.add_argument("--whisper-model", default="base",
                    choices=["base", "small", "medium", "large"])
    ap.add_argument("--lang", help="language code for captions/Whisper (e.g. en, ms, zh)")
    ap.add_argument("--out", help="output folder for frames (default: temp folder)")
    a = ap.parse_args()

    start, end = parse_timecode(a.start), parse_timecode(a.end)
    at_list = [parse_timecode(t) for t in a.at.split(",")] if a.at else []
    if a.at and None in at_list:
        ap.error("--at has an invalid timestamp")
    out_dir = Path(a.out or tempfile.mkdtemp(prefix="vidsage_watch_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    url = a.source if is_url(a.source) else None
    notes = []

    # 1. Transcript from captions (no download needed)
    segments, tx_label = None, None
    if url and not a.no_transcript:
        got = fetch_captions(url, a.lang)
        if got:
            segments, tx_label = got
            log(f"captions found: {tx_label}")

    need_video = (not a.transcript_only) or at_list or (not a.no_transcript and segments is None)
    work = tempfile.TemporaryDirectory(prefix="vidsage_dl_")
    with work:
        video = a.source
        if need_video:
            if url:
                log("downloading video…")
                video = download_video(url, work.name)
            elif not os.path.isfile(video):
                print(f"error: file not found: {video}", file=sys.stderr)
                return 2

            duration = None
            cap = cv2.VideoCapture(video)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
            cap.release()

            offset = 0.0
            clip = video
            if start is not None or end is not None:
                lo = max(start or 0.0, 0.0)
                hi = min(end, duration) if end is not None else duration
                if hi - lo < 1:
                    print(f"error: empty range {fmt_time(lo)}–{fmt_time(hi)} "
                          f"(video is {fmt_time(duration)})", file=sys.stderr)
                    return 2
                clip = os.path.join(work.name, "clip.mp4")
                trim_clip(video, clip, lo, hi)
                offset = lo
                notes.append(f"Window analysed: {fmt_time(lo)} → {fmt_time(hi)} "
                             f"of {fmt_time(duration)}")

            # Whisper fallback on the trimmed clip
            if segments is None and not a.no_transcript:
                log("no captions — transcribing with Whisper…")
                segments, tx_label = whisper_transcribe(clip, a.whisper_model, a.lang, a.stt, a.fw_model,
                                                        a.hotwords, a.xxl_filter)
                segments = [{**s, "start": s["start"] + offset, "end": s["end"] + offset}
                            for s in segments]

            frames = []
            if not a.transcript_only:
                log("selecting frames…")
                for t, fr in sample_frames(clip, a.mode, a.interval, a.threshold,
                                           a.dedup, a.max_frames):
                    frames.append((t + offset, fr))
            for t in at_list:
                fr = grab_at(video, t)
                if fr is not None:
                    frames.append((t, fr))
                else:
                    notes.append(f"Could not read a frame at {fmt_time(t)}")
            frames.sort(key=lambda x: x[0])

            saved = []
            for t, fr in frames:
                p = out_dir / f"frame_{int(t // 3600):02d}-{int(t % 3600 // 60):02d}-{int(t % 60):02d}.jpg"
                save_frame(fr, str(p))
                saved.append((t, p))
        else:
            saved = []

    # Captions cover the whole video → keep only the requested window
    if segments is not None and (start is not None or end is not None):
        lo = start or 0.0
        hi = end if end is not None else float("inf")
        segments = [s for s in segments if lo <= s["start"] < hi]
        if not notes:
            notes.append(f"Window analysed: {fmt_time(lo)} → "
                         f"{fmt_time(hi) if hi != float('inf') else 'end'}")

    # 2. Report
    print("# Video evidence\n")
    print(f"Source: {a.source}")
    for n in notes:
        print(n)
    if segments is not None:
        print(f"\n## Transcript — {tx_label}\n")
        if CAPTION_LANGS:
            print(f"(other caption languages available — use --lang CODE: "
                  f"{', '.join(CAPTION_LANGS[:15])}{'…' if len(CAPTION_LANGS) > 15 else ''})\n")
        lines = [f"[{fmt_time(s['start'])}] {s['text']}" for s in segments if s["text"]]
        print("\n".join(lines) if lines else "(no speech detected in this window)")
    elif not a.no_transcript:
        print("\n## Transcript\n\n(unavailable)")
    if saved:
        print(f"\n## Frames ({len(saved)}) — open these images with the Read tool\n")
        for t, p in saved:
            print(f"- [{fmt_time(t)}] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

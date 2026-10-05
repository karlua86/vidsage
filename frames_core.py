"""
Shared, UI-free helpers used by both the Streamlit app (app.py) and the
Claude Code command-line tool (watch.py).

Nothing here imports streamlit or whisper, so it is cheap to import.
"""
import subprocess

import cv2
import numpy as np

# Near-duplicate detection for VIDEO frames.
# A tiny mean-hash (8x8 or even 16x16) washes out thin dark text on a white slide,
# so two different slides scored ~0.95 "similar" and got dropped.  Instead we keep a
# 64x36 grayscale signature and measure the fraction of pixels that visibly changed.
# Slide change ≈ 2-3 % of pixels; cursor / compression noise ≈ 0.1 %.
SIG_SIZE = (64, 36)
PIXEL_TOL = 30          # grey-level change that counts as "changed"
CHANGE_GAIN = 20        # 5 % changed pixels → similarity 0.0


def frame_signature(frame) -> np.ndarray:
    """64x36 grayscale thumbnail used to compare video frames."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, SIG_SIZE, interpolation=cv2.INTER_AREA).astype(np.int16)


def frame_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1.0 = identical, falls to 0.0 once ≥5 % of pixels visibly differ."""
    changed = float(np.mean(np.abs(a - b) > PIXEL_TOL))
    return 1.0 - min(1.0, changed * CHANGE_GAIN)


def frame_is_duplicate(sig: np.ndarray, accepted: list, threshold: float = 0.88) -> bool:
    """True if sig is ≥ threshold similar to any already-accepted signature."""
    return any(frame_similarity(sig, s) >= threshold for s in accepted)


# Boolean mean-hash comparison — still used for uploaded-slide dedup in app.py.
def perceptual_hash(frame, size: int = 16) -> np.ndarray:
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (size, size),
                       interpolation=cv2.INTER_AREA)
    flat = small.flatten().astype(float)
    return flat > flat.mean()


def is_duplicate(phash: np.ndarray, accepted: list, threshold: float = 0.92) -> bool:
    """True if phash is ≥ threshold similar to any already-accepted hash."""
    return any(np.mean(phash == h) >= threshold for h in accepted)


def score_text_density(frame) -> float:
    """Higher = frame likely holds more text/data (edge density + sharpness)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (320, 180))
    edge_score = float(cv2.Canny(small, 50, 150).mean())
    lap_score = float(cv2.Laplacian(small, cv2.CV_64F).var())
    return edge_score * 0.65 + min(lap_score / 50.0, 30.0) * 0.35


def parse_timecode(value) -> float | None:
    """'90', '1:30', '01:02:03' or '' → seconds (None if blank/invalid)."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parts = [float(p) for p in text.split(":")]
    except ValueError:
        return None
    if len(parts) > 3:
        return None
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def trim_clip(src: str, dst: str, start: float | None, end: float | None) -> None:
    """Cut [start, end] out of src into dst with ffmpeg (frame-accurate re-encode)."""
    cmd = ["ffmpeg", "-y"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", src]
    if end is not None:
        cmd += ["-t", f"{end - (start or 0):.3f}"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", dst]
    subprocess.run(cmd, capture_output=True, check=True)

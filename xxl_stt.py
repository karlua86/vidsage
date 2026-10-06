"""
Local GPU transcription through a Faster-Whisper-XXL install — UI-free, shared by app.py and watch.py.

Faster-Whisper-XXL (Purfview's standalone Windows build of faster-whisper) is a ready-made program with
its own engine, NVIDIA libraries and models, so VidSage only has to run it and read its JSON result.
Nothing leaves your PC.  VidSage never modifies the XXL folder (it only reads models and runs the .exe).
"""
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

EXE_NAMES = ("faster-whisper-xxl.exe", "faster-whisper-xxl")
SETUP_HINT = ("Faster-Whisper-XXL wasn't found. Download it from "
              "github.com/Purfview/whisper-standalone-win, unzip it, then paste its folder path in the sidebar "
              "(or set FW_XXL_PATH in .streamlit/secrets.toml).")


def _exe_in(folder: Path) -> Path | None:
    for base in (folder, folder / "Faster-Whisper-XXL"):         # zips often nest one level
        for name in EXE_NAMES:
            if (base / name).is_file():
                return base / name
    return None


def find_xxl(configured: str = "", strict: bool = False) -> Path | None:
    """Locate faster-whisper-xxl.exe: the configured path, FASTER_WHISPER_XXL / FW_XXL_PATH env vars,
    then the usual download locations.  strict=True: if a path was typed in, use ONLY that path
    (a wrong path returns None instead of silently picking some other install)."""
    if strict and (configured or "").strip():
        p = Path(configured.strip().strip('"'))
        return p if p.is_file() else (_exe_in(p) if p.is_dir() else None)
    for raw in (configured, os.environ.get("FW_XXL_PATH", ""), os.environ.get("FASTER_WHISPER_XXL", "")):
        raw = (raw or "").strip().strip('"')
        if not raw:
            continue
        p = Path(raw)
        if p.is_file():
            return p
        if p.is_dir() and _exe_in(p):
            return _exe_in(p)
    home = Path.home()
    patterns = [r"D:\Download\Faster-Whisper-XXL*", r"D:\Downloads\Faster-Whisper-XXL*",
                str(home / "Downloads" / "Faster-Whisper-XXL*"), str(home / "Documents" / "Faster-Whisper-XXL*"),
                str(home / "Desktop" / "Faster-Whisper-XXL*"), r"C:\Faster-Whisper-XXL*", r"C:\Tools\Faster-Whisper-XXL*"]
    for pat in patterns:
        for hit in sorted(glob.glob(pat), reverse=True):
            exe = _exe_in(Path(hit))
            if exe:
                return exe
    on_path = shutil.which("faster-whisper-xxl")
    return Path(on_path) if on_path else None


def installed_models(exe: Path) -> list:
    """Models already present in the XXL install's _models folder, best default first."""
    names = []
    models_dir = exe.parent / "_models"
    if models_dir.is_dir():
        for d in models_dir.glob("faster-whisper-*"):
            if (d / "model.bin").is_file():
                names.append(d.name.replace("faster-whisper-", "", 1))
    order = {"large-v2": 0, "large-v3": 1, "large-v3-turbo": 2, "medium": 3, "small": 4, "base": 5, "tiny": 6}
    return sorted(names, key=lambda n: (order.get(n, 9), n))


# Optional audio clean-up applied by XXL before transcribing (off by default).
NOISE_FILTERS = {
    "Off": [],
    "Light denoise": ["--ff_fftdn", "12"],                       # FFT denoise at XXL's "normal strength"
    "Remove non-speech (RNNoise)": ["--ff_rnndn_xiph"],          # neural net that suppresses non-speech sound
    "Boost quiet voices": ["--ff_speechnorm"],                   # fast speech amplification
    "Isolate voice (strong, slow)": ["--ff_vocal_extract", "mdx_kim2"],   # vocal separation model
}


LAST_NOTE: str | None = None     # set when options had to be dropped (shown to the user by the app)


def transcribe_xxl(audio_path: str, exe: Path, model: str = "large-v2", language: str | None = None,
                   prompt: str = "", progress=None, hotwords: str = "", noise_filter: str = "Off") -> tuple:
    """Run Faster-Whisper-XXL on one audio file.  Returns (segments, language_code) where
    segments = [{'start','end','text'}, ...].  progress(fraction 0..1) is called when it reports a percentage.
    hotwords: comma-separated names/terms to favour.  noise_filter: a key of NOISE_FILTERS.
    XXL r245.4 crashed natively in testing when automatic language detection was combined with hotwords or a
    filter on some recordings (forcing the language avoided it).  So on a crash with those options, VidSage
    runs plain to learn the language, re-runs with the language pinned, and only if that fails too returns the
    plain result.  LAST_NOTE explains what happened."""
    global LAST_NOTE
    LAST_NOTE = None
    try:
        return _run_xxl(audio_path, exe, model, language, prompt, progress, hotwords, noise_filter)
    except RuntimeError:
        if not hotwords.strip() and noise_filter == "Off":
            raise
        plain = _run_xxl(audio_path, exe, model, language, prompt, progress, "", "Off")
        used = [x for x in ("hotwords" if hotwords.strip() else "",
                            f"the '{noise_filter}' filter" if noise_filter != "Off" else "") if x]
        names = " and ".join(used)
        if not language and plain[1]:
            try:
                pinned = _run_xxl(audio_path, exe, model, plain[1], prompt, progress, hotwords, noise_filter)
                LAST_NOTE = (f"XXL crashed with {names} and automatic language detection on this recording, "
                             f"so the language was fixed to '{plain[1]}' for this run.")
                return pinned
            except RuntimeError:
                pass
        LAST_NOTE = (f"Faster-Whisper-XXL crashed when using {names} on this recording, "
                     f"so it was transcribed without {'it' if len(used) == 1 else 'them'}.")
        return plain


def _run_xxl(audio_path: str, exe: Path, model: str, language: str | None, prompt: str, progress,
             hotwords: str, noise_filter: str) -> tuple:
    exe = Path(exe)
    if not exe.is_file():
        raise RuntimeError(SETUP_HINT)

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [str(exe), audio_path, "-m", model, "-f", "json", "-o", tmp,
               "--vad_filter", "true", "--print_progress"]
        if language:
            cmd += ["-l", language]
        if prompt:
            cmd += ["--initial_prompt", prompt]
        if hotwords.strip():
            cmd += ["--hotwords", hotwords.strip()]
        cmd += NOISE_FILTERS.get(noise_filter, [])

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                text=True, encoding="utf-8", errors="replace", cwd=str(exe.parent))
        tail = ""
        while True:
            chunk = proc.stdout.read(256)
            if not chunk:
                break
            tail = (tail + chunk)[-1500:]
            if progress:
                pcts = re.findall(r"(\d{1,3})%", chunk)
                if pcts:
                    progress(min(int(pcts[-1]), 100) / 100.0)
        code = proc.wait()

        out_json = Path(tmp) / (Path(audio_path).stem + ".json")
        if code != 0 or not out_json.exists():
            raise RuntimeError(f"Faster-Whisper-XXL failed (exit {code}). {tail[-500:].strip()}")
        data = json.loads(out_json.read_text(encoding="utf-8"))

    segments = []
    for seg in data.get("segments", []):
        text = str(seg.get("text", "")).strip()
        if text:
            segments.append({"start": float(seg.get("start", 0)), "end": float(seg.get("end", 0)), "text": text})
    return segments, (data.get("language") or language or "en")

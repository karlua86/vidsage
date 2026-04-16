# Video Explainer — Setup Guide

## Prerequisites

### 1. Python 3.9+
Check with: `python --version`
Download from: https://python.org

### 2. FFmpeg (required by Whisper)
Install via winget (run in PowerShell as Admin):
```
winget install ffmpeg
```
Then **restart your terminal** so ffmpeg is on PATH.
Verify with: `ffmpeg -version`

---

## Installation

Open a terminal in this folder, then:

```bash
# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `openai-whisper` will also install PyTorch (~2GB). This is normal.

---

## Running the App

```bash
streamlit run app.py
```

The app opens at http://localhost:8501

---

## First Run — Whisper Model Download

The first time you transcribe a video, Whisper downloads the model weights:

| Model  | Size   | Speed  | Accuracy |
|--------|--------|--------|----------|
| base   | 140 MB | Fast   | Basic    |
| small  | 460 MB | Fast   | Good     |
| medium | 1.5 GB | Medium | Great ✓  |
| large  | 3.0 GB | Slow   | Best     |

**Recommended: `medium`** — good accuracy for lectures/meetings without being too slow.

Models are cached in `~/.cache/whisper/` — only downloaded once.

---

## Usage

1. Open the app in your browser
2. Enter your Anthropic API key in the sidebar
3. Select the video type (Lecture, Zoom Meeting, etc.)
4. Adjust frame sampling if needed (default settings work well)
5. Upload your MP4
6. Click **Analyze Video**
7. Get your explanation in the **Explanation** tab

---

## Tips

- **Long videos (1hr+):** Use `medium` Whisper, set frame interval to 60–90s
- **Screen recordings:** Lower frame interval (15–30s) to capture more screen content
- **Slow machine:** Use `small` Whisper model to save time
- **Poor audio quality:** Use `large` Whisper for best transcription accuracy

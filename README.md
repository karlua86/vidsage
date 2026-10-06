# 🎬 VidSage

> **Turn hours of video into instant insights — fully local, fully private.**

VidSage is an AI-powered video analysis tool that extracts transcripts, generates detailed explanations, creates chapter breakdowns, and exports structured notes — all running on your own machine. No cloud uploads. No subscriptions. Your content stays yours.

---

## 🆕 What's new in v1.1 – v1.3

- **⚡ Cloud transcription (v1.4)** — optional Groq or OpenAI Whisper for videos without captions: roughly a minute instead of an hour, about $0.11 (Groq) or $0.36 (OpenAI) per hour of audio. Only the audio is uploaded, never the video; local Whisper stays the free, private default.
- **📚 Batch online videos (v1.3)** — in *Online Video URL* mode, paste several links (one per line) and VidSage analyses them one after another. A broken link doesn't stop the rest; you get a summary table and a one-click download of all explanations.
- **✂️ Time Range** — analyse only part of a long video (Start/End in the sidebar). Timestamps stay on the original timeline.
- **🧠 Smarter frame de-duplication** — different slides on a white background are no longer mistaken for duplicates.
- **💸 DeepSeek Flash engine (v1.2)** — a much cheaper alternative to Claude/OpenAI that also reads video frames.
- **🤖 Use VidSage from Claude Code** — new `watch.py` command-line tool + ready-made skill, so Claude can "watch" a video (YouTube captions first, Whisper fallback). See [below](#use-vidsage-from-claude-code-watchpy).
- **🚀 `start_vidsage.bat`** — double-click launcher for Windows.
- **🐛 Fixes** — *Dedup aggressiveness* labels were backwards; `yt-dlp` and `youtube-transcript-api` were missing from `requirements.txt`.

Full details in [CHANGELOG.md](CHANGELOG.md).

---

## ✨ Features

### 🎙️ Transcription
- Powered by **OpenAI Whisper** (runs locally)
- Choose from `base`, `small`, `medium`, or `large` models
- Supports **40+ languages** with auto-detection
- Outputs plain transcript + **timestamped SRT subtitles**

### 🤖 AI Analysis (your choice of engine)
- **Gemini (Free)** — Google's free tier, great for long videos
- **Claude (Paid)** — Anthropic's Claude Sonnet, best quality
- **OpenAI (Paid)** — GPT-4o vision
- **DeepSeek Flash (Cheap)** — DeepSeek V4.1 Flash, reads images too, roughly a cent or two per video

### 📄 What You Get
- **Explanation** — structured breakdown: Overview, Key Concepts, Detailed Section-by-Section Analysis, Key Takeaways, Summary
- **Chapters** — timestamped chapter list covering the full video duration
- **Timestamped Transcript** — every segment linked to its timestamp
- **PKM Notes** — Obsidian/Notion-ready markdown with tags and backlinks
- **Q&A Chat** — ask questions about the video content

### 📦 Export Formats
- Markdown (`.md`)
- Word Document (`.docx`)
- PDF (`.pdf`)
- SRT Subtitles (`.srt`)
- Plain Text (`.txt`)

### ✂️ Control & Automation
- **Cloud or local transcription** — Groq / OpenAI Whisper (fast) or local Whisper (free, private)
- **Batch online videos** — paste many YouTube/Vimeo/TikTok… links at once
- **Time Range** — analyse only a chosen window of the video
- **YouTube captions first** — skips Whisper when captions exist
- **Claude Code integration** — `watch.py` command-line tool and skill

### 🔒 Privacy First
- Everything runs **on your computer**
- No video files are uploaded anywhere
- API calls send only text and frames — never the full video
- Results saved locally to your chosen folder

---

## 🖥️ Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10/11 (64-bit) |
| Python | 3.9 or higher |
| RAM | 8 GB minimum, 16 GB recommended |
| GPU | Optional — NVIDIA GPU speeds up Whisper significantly |
| FFmpeg | Required (see setup below) |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/karlua86/vidsage.git
cd vidsage
```

### 2. Install FFmpeg
```powershell
winget install ffmpeg
```
Restart your terminal after installation.

### 3. Create a virtual environment & install dependencies
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
> ⚠️ `openai-whisper` will download PyTorch (~2 GB). This is normal and only happens once.

### 4. Run the app
```bash
streamlit run app.py
```
Opens at **http://localhost:8501**

> 💡 **Windows shortcut:** double-click `start_vidsage.bat` instead. It uses the `venv` folder if present, installs the requirements on first run if they're missing, warns if FFmpeg isn't installed, and starts the app.

---

## 🔑 API Keys

VidSage requires an API key for the AI analysis step. Transcription (Whisper) is always free and local.

| Engine | Where to get key | Cost |
|--------|-----------------|------|
| Gemini | [aistudio.google.com](https://aistudio.google.com) | Free tier available |
| Claude | [console.anthropic.com](https://console.anthropic.com) | Pay per use |
| OpenAI | [platform.openai.com](https://platform.openai.com) | Pay per use |
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com) | Pay per use (cheapest) |
| Groq *(optional, transcription only)* | [console.groq.com](https://console.groq.com) | ~$0.11 per hour of audio |

Keys are entered in the app sidebar — never stored in code.

**Optional:** Save your key in `.streamlit/secrets.toml` so you don't have to paste it every time:
```toml
GEMINI_API_KEY = "your-key-here"
ANTHROPIC_API_KEY = "your-key-here"
OPENAI_API_KEY = "your-key-here"
DEEPSEEK_API_KEY = "your-key-here"
GROQ_API_KEY = "your-key-here"      # optional: fast cloud transcription
```

---

## 📋 How It Works

```
Video File
    │
    ▼
┌─────────────────┐
│  Frame Sampling  │  ← Extract key frames (smart scene detection)
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│  Whisper (Local) │  ← Transcribe audio to text + timestamps
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│   AI Analysis    │  ← Combine frames + transcript → structured explanation
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│  Export Files    │  ← MD, DOCX, PDF, SRT, TXT, PKM Notes
└─────────────────┘
```

---

## 🎛️ Supported Video Types

- 🎓 Lectures & Courses
- 📹 Zoom / Teams Meetings
- 📊 Presentations & Webinars
- 🎙️ Podcasts & Interviews
- 🖥️ Screen Recordings
- 🎬 General Videos

---

## ⚙️ Advanced Features

- **Slide upload** — attach PDF/image slides alongside the video for richer analysis
- **Large deck support** — handles 100+ slide decks via chunked processing
- **Re-analyse** — update explanation and chapters with corrections or new slides
- **Multi-language** — transcribe and analyse in any Whisper-supported language
- **Long video support** — tested on 3–4 hour videos with full chapter coverage

---

## 📁 Output Files

After analysis, VidSage saves the following to your results folder:

| File | Description |
|------|-------------|
| `videoname_explanation.md` | Full AI explanation in Markdown |
| `videoname_explanation.docx` | Word document version |
| `videoname_explanation.pdf` | PDF version |
| `videoname_transcript.txt` | Plain transcript |
| `videoname_timestamped.txt` | Transcript with timestamps |
| `videoname_subtitles.srt` | SRT subtitle file |
| `videoname_chapters.txt` | Chapter list with timestamps |
| `videoname_pkm.md` | PKM notes for Obsidian/Notion |

---

## 🛠️ Troubleshooting

**Whisper is slow**
→ Use `small` or `base` model, or install a CUDA-enabled PyTorch for GPU acceleration

**Claude 429 rate limit error**
→ Normal on large decks — VidSage automatically splits into batches with waits

**Gemini quota exhausted**
→ Free tier has daily limits; add billing credits at [console.cloud.google.com](https://console.cloud.google.com)

**YouTube / online video download fails (HTTP 403 or "unable to download")**
→ YouTube changes often and old `yt-dlp` versions stop working. Run `python -m pip install -U yt-dlp` and try again.

**FFmpeg not found**
→ Run `winget install ffmpeg` and restart your terminal

---

## 📜 License

Copyright © 2026 VidSage. Licensed under the [GNU General Public License v3.0](LICENSE).

Free for personal and open-source use. Commercial use requires written permission from the author.

---

## 🙏 Built With

- [Streamlit](https://streamlit.io) — UI framework
- [OpenAI Whisper](https://github.com/openai/whisper) — local transcription
- [Anthropic Claude](https://anthropic.com) — AI analysis
- [Google Gemini](https://aistudio.google.com) — AI analysis
- [OpenCV](https://opencv.org) — video frame extraction
- [PyMuPDF](https://pymupdf.readthedocs.io) — PDF processing
- [python-docx](https://python-docx.readthedocs.io) — Word export

## Use VidSage from Claude Code (`watch.py`)

`watch.py` is a command-line version of the pipeline so Claude Code can "watch" a video:

```bash
python watch.py "https://youtu.be/XXXXXXXXXXX" --transcript-only       # captions first, no download
python watch.py lecture.mp4 --start 12:30 --end 18:00 --max-frames 12  # only a window
python watch.py lecture.mp4 --at 14:05,14:40                           # exact "look here" frames
```

It prints a Markdown report (timestamped transcript + saved frame paths) for Claude to read.
Install the ready-made skill by copying `claude-skill/vidsage-watch/` into `~/.claude/skills/`.

### Time range & smarter de-duplication
- The sidebar has an optional **Time Range** (Start/End) so only part of a long video is analysed.
  Timestamps in the transcript, SRT and frames still match the original video.
- Near-duplicate frames are detected from a 64×36 pixel-change signature instead of an 8×8 mean-hash
  (which treated different white slides as identical). The *Dedup aggressiveness* labels now match
  their behaviour (lower value = more frames removed).

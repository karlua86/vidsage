# Changelog

## v1.6.2 — 2026-10-07

### Fixed
- **Laggy results screen.** Two things made every click on a finished analysis slow, and the cost grew with
  the length of the video:
  - the Word (.docx) export was rebuilt on every redraw (about 0.2 s for a 10-minute video, nearly 1 s for a
    2-hour one) — it is now built once and remembered until the content changes;
  - ticking a frame checkbox forced a second full redraw (`st.rerun()`) — removed; the selected-frames count
    now updates in the same pass, and *Select all / Deselect all* tick the boxes directly.
  Measured redraw / checkbox-click time: 10-min video 0.91 → 0.63 s, 1-hour video 1.66 → 0.67 s,
  2-hour video 2.64 → 0.73 s (test harness; the real app is faster still).
- Removed a Streamlit warning about how the frame checkboxes got their start value.

## v1.6.1 — 2026-10-06

### Added
- **Noise filter** (Faster-Whisper-XXL engine): Off (default) / Light denoise / Remove non-speech (RNNoise) /
  Boost quiet voices / Isolate voice (strong, ~2× slower). On two clear phone-call recordings none of the
  filters improved the transcript and several made it worse (changed wording, dropped the first greeting),
  so the default is Off and the help text says so.
- **Hotwords** (XXL engine): the *Names & terms* box is sent to XXL as hotwords (other engines still get it as
  prompt text). `watch.py --hotwords "..." --xxl-filter denoise|rnnoise|boost|voice`.
- README: new section explaining what Faster-Whisper-XXL is and how to download, unzip and connect it.

### Fixed
- XXL r245.4 crashed (exit `0xC0000409`) when automatic language detection was combined with hotwords or a filter
  on a Malay recording; forcing the language avoided it. VidSage now recovers automatically: it runs plain, then
  re-runs with the detected language fixed, and tells you what happened.

## v1.6.0 — 2026-10-06

### Added
- **Local GPU transcription via Faster-Whisper-XXL.** New *Transcription engine* choice
  "🚀 Local GPU (Faster-Whisper-XXL)". VidSage finds your XXL folder automatically (or paste it; save it
  with `FW_XXL_PATH` in `.streamlit/secrets.toml`), lists the models already inside it, and runs the
  `.exe` on the audio. Free and private; it only reads the folder and never modifies it. A wrong folder gives
  a clear error. Verified on real Malay and Chinese–English call recordings (output identical to a
  standalone XXL run). `watch.py --stt xxl [--fw-model large-v2]` does the same from Claude Code.
- **Names & terms** box (Transcription section): a short comma-separated list that nudges Whisper towards
  spellings such as names or buildings.

### Fixed
- **Multilingual mode no longer sends a hint.** The old instruction-style hint ("This video contains
  multiple languages… Transcribe all languages exactly as spoken…") made Whisper hallucinate in 4 of 6 test
  runs: fake subtitle credits ("字幕由Amara.org社区提供"), "thanks for watching" loops, "please like and subscribe",
  or the hint text repeated. Natural sample phrases were also tried but leaked into the transcript, switched
  Chinese script and dropped English words, so no hint is used; Whisper detects the language itself.

### Notes from testing (large-v2 vs large-v3, on Malay and Chinese–English call recordings)
- large-v2 was the steadier model: large-v3 mis-detected Malay as English (and wrote an English version),
  and produced looping text when Malay was forced. large-v3 spelled some English words better inside Chinese.
  The XXL engine therefore defaults to large-v2.

## v1.5.0 — 2026-10-06

### Added
- **Gemini watches the whole video (opt-in).** With the Gemini engine selected, tick *Let Gemini watch the
  whole video*. Gemini then analyses the actual video (about 1 frame per second plus the audio) instead of
  a handful of still frames, so it can follow motion and sound. Public YouTube links are passed straight to
  Gemini; other videos (and any Time Range clip) are uploaded to Google's Files API and deleted right after
  the request. If video mode fails for any reason it falls back to the normal frame-based analysis.
  Cost is about 100 tokens per second of video (free on the free tier).
  **Privacy:** the whole video leaves your PC, and on the free tier Google may use the data to improve
  its products — keep it off for private videos.
  Verified live: Gemini correctly reported two random codes shown on screen plus the audio tone from an
  uploaded test clip, a YouTube link with no transcript, and the uploaded file was removed afterwards.

## v1.4.0 — 2026-10-06

### Added
- **Cloud transcription (optional).** New *Transcription engine* choice in the sidebar: Local Whisper
  (default, free, private), **Groq Cloud** (whisper-large-v3, ~$0.11 per hour of audio) or **OpenAI Whisper**
  (whisper-1, ~$0.36 per hour). Used only when a video has no YouTube captions. Only the audio track is
  uploaded, re-encoded to small 32 kbps MP3 chunks of 30 minutes (stays far under the 25 MB limit);
  timestamps are stitched back onto the original timeline. A missing key gives a clear error instead of
  silently falling back. Works with the Time Range and Batch features.
- `watch.py --stt groq|openai` for the same option from Claude Code (key from `GROQ_API_KEY` /
  `OPENAI_API_KEY` or `.streamlit/secrets.toml`).
- New shared module `cloud_stt.py`. Add `GROQ_API_KEY` to `.streamlit/secrets.toml` to prefill the key.

## v1.3.1 — 2026-10-06

### Fixed
- **Gemini (Free) engine was broken**: it used `gemini-2.0-flash`, which Google shut down on
  1 June 2026 (HTTP 404). All Gemini calls now try `gemini-3.8-flash`, then `gemini-3.5-flash`, then
  `gemini-3.5-flash-lite`, automatically skipping retired (404) or overloaded (503) models.
  Override the list with the `VIDSAGE_GEMINI_MODELS` environment variable (comma-separated).
  Verified against a live Gemini key, including with a retired model listed first.

## v1.3.0 — 2026-10-06

### Added
- **Batch online videos.** *Online Video URL* mode now takes several links (one per line). They are
  analysed one after another; a failed link is reported without stopping the rest. Results show a
  summary table, per-video explanation/chapters/transcript tabs, and a single download of all
  explanations (.md). One link behaves exactly as before. The sidebar Time Range applies to every video.

### Fixed
- **YouTube captions-first was silently broken** with `youtube-transcript-api` 1.x (it removed
  `list_transcripts`), so every YouTube video fell back to slow Whisper. The app now supports both
  API versions. In testing, a 3-link batch dropped from 391 s to 228 s and transcripts were more complete.
- With no language chosen, captions now prefer English instead of whichever track YouTube lists first
  (dubbed videos could return Arabic).
- The caption language now maps to the right AI output language via its language code (it was always
  falling back to English).

## v1.2.4 — 2026-10-06

### Fixed
- YouTube/online downloads failing with **HTTP 403**: caused by an outdated `yt-dlp`. `requirements.txt` now
  requires `yt-dlp>=2026.8.19`, and the README troubleshooting section explains `pip install -U yt-dlp`.
  Verified downloading through VidSage's own downloader after upgrading.

## v1.2.3 — 2026-10-06

### Fixed
- A malformed `.streamlit/secrets.toml` (for example a key pasted without quotes) no longer
  crashes the whole app on startup; VidSage shows a hint in the sidebar and lets you paste the key.
- Removed a Python `SyntaxWarning` (invalid `\s` escape) in the Word/PDF bullet clean-up code.

### Verified
- DeepSeek Flash measured on a real run (10-min podcast, 2,299-word transcript, 8 frames):
  analysis + chapters + PKM note used 20,702 input / 5,211 output tokens = about **$0.006
  off-peak / $0.013 peak**. Sidebar estimate updated to ~$0.01–0.03 per video (10–60 min).
  Thinking mode confirmed disabled; image `detail` field accepted.

## v1.2.2 — 2026-10-06

### Fixed
- Notes/explanation preview formatting: section headers without a number no longer show a stray
  `##`; Obsidian `[[wiki-links]]` display as plain bold text in the app (the raw note and exports
  keep `[[links]]`); "Example from the video" quotes render as real blockquotes instead of a
  literal `>`, and the line after a quote is no longer swallowed into it.
- The PKM note template now puts the example quote on its own line.

## v1.2.1 — 2026-10-06

### Changed
- **Cost estimates in the sidebar corrected** against current provider prices (Claude Sonnet 4.6
  + Haiku 4.5, GPT-4o + 4o-mini, DeepSeek Flash) and VidSage's real behaviour (frame count,
  transcript size, helper calls for chapters/title/tags/notes), no slides, cleanup off:
  Claude ~$0.10–0.30, GPT-4o ~$0.07–0.20, DeepSeek Flash ~$0.01–0.04 per video (10–60 min).
  Uploading slides or turning on transcript cleanup raises these. They are estimates, not quotes.
- DeepSeek Flash calls now disable its default "thinking" mode, which is slower and bills extra
  output tokens; VidSage's summarising tasks don't need it.
- `start_vidsage.bat` now prefers a Python that already has streamlit (venv, then .venv, then
  system Python) instead of trying to install into an empty venv.

## v1.2.0 — 2026-10-06

### Added
- **DeepSeek Flash (Cheap)** AI engine — DeepSeek V4.1 Flash via its OpenAI-compatible API
  (model `deepseek-flash`, accepts images, 1M-token context). Roughly $0.01–0.02 per video,
  versus ~$0.08 for Claude and ~$0.10 for GPT-4o. Add `DEEPSEEK_API_KEY` to
  `.streamlit/secrets.toml` or paste it in the sidebar. `DEEPSEEK_BASE_URL` can override the endpoint.
  Cleanup, chapters, Q&A and tags use the same model.
- **`start_vidsage.bat`** — generic Windows launcher: uses `venv` if present, installs requirements
  on first run, warns if FFmpeg is missing, starts the app.

## v1.1.0 — 2026-10-06

### Added
- **Time Range** (sidebar): analyse only a Start→End window of a video. Transcript, SRT and
  frame timestamps still match the original video; works with uploads and YouTube captions.
- **`watch.py`** — command-line version of the pipeline so Claude Code can "watch" a video:
  YouTube captions first (no download needed), local Whisper fallback, URL or file input,
  `--start/--end`, `--at 14:05` exact frames, `--mode dense|scene|interval`, `--transcript-only`.
- **`claude-skill/vidsage-watch/SKILL.md`** — drop-in Claude Code skill for `watch.py`.
- **`frames_core.py`** — shared, UI-free helpers used by both the app and `watch.py`.

### Changed
- **Frame de-duplication** now compares a 64×36 pixel-change signature instead of an 8×8
  mean-hash. The old hash scored different white-background slides as ~95 % similar and dropped
  real slide changes. Slide-upload de-duplication is unchanged.
- `watch.py` defaults to English captions (previously the first track YouTube listed, which could
  be a dubbed/translated language) and lists other available caption languages; use `--lang`.

### Fixed
- *Dedup aggressiveness* slider labels were inverted (the "Light" setting removed the most frames).
- `yt-dlp` and `youtube-transcript-api` were used by the app but missing from `requirements.txt`.
- `watch.py` forces UTF-8 output so Windows consoles (cp1252) don't crash on non-Latin text.

## v1.0.0
- Initial release: Whisper transcription, Claude/Gemini/OpenAI analysis, chapters, Q&A,
  PKM notes, Word/PDF/SRT export.

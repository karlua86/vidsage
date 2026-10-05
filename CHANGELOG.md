# Changelog

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

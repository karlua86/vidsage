# Changelog

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

import streamlit as st
import whisper
import cv2
import base64
import tempfile
import os
import subprocess
import re
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image
import io
import threading
import time
from datetime import datetime
from pathlib import Path
from cloud_stt import transcribe_cloud, PROVIDERS as STT_PROVIDERS
import xxl_stt as _xxl_mod
from xxl_stt import NOISE_FILTERS, find_xxl, installed_models as xxl_installed_models, transcribe_xxl
from frames_core import (
    frame_signature as _frame_signature,
    frame_is_duplicate as _frame_is_duplicate,
    is_duplicate as _is_duplicate,
    parse_timecode, trim_clip,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _secret(name: str) -> str:
    """Read a key from .streamlit/secrets.toml; never crash the app if the file is missing or
    malformed (e.g. a key pasted without quotes) — show a hint in the sidebar instead."""
    try:
        return st.secrets.get(name, "")
    except Exception as exc:
        st.sidebar.warning(
            f"Couldn't read .streamlit/secrets.toml ({type(exc).__name__}). Each line must look like "
            f'KEY = "value" (with quotes). Paste your key below instead.')
        return ""


# ── Gemini model fallback ─────────────────────────────────────────────────────
# Google retires Gemini models regularly (gemini-2.0-flash was shut down on 1 June 2026).  Try
# the models below in order, skipping any that are retired (404) or overloaded (503).
# Override with the VIDSAGE_GEMINI_MODELS environment variable (comma-separated).
GEMINI_MODELS = [m.strip() for m in os.environ.get(
    "VIDSAGE_GEMINI_MODELS", "gemini-3.8-flash,gemini-3.5-flash,gemini-3.5-flash-lite"
).split(",") if m.strip()]
_GEMINI_RETIRED: set = set()


def _gemini_generate(client, contents, **kwargs):
    """client.models.generate_content() with automatic model fallback.
    404 (retired) → never try that model again this session; 503 (overloaded) → try the next one;
    anything else (e.g. 429 rate limit) is raised so the callers' retry logic still applies."""
    last_error = None
    for model in GEMINI_MODELS:
        if model in _GEMINI_RETIRED:
            continue
        try:
            return client.models.generate_content(model=model, contents=contents, **kwargs)
        except Exception as exc:
            text = str(exc)
            last_error = exc
            if "404" in text or "NOT_FOUND" in text:
                _GEMINI_RETIRED.add(model)
                continue
            if "503" in text or "UNAVAILABLE" in text:
                continue
            raise
    raise last_error or RuntimeError("No Gemini model is available — set VIDSAGE_GEMINI_MODELS.")


# ── OpenAI-compatible engines (OpenAI + DeepSeek Flash share one code path) ──
DEEPSEEK_ENGINE = "DeepSeek Flash (Cheap)"
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def _is_oai(ai_engine: str) -> bool:
    """True for engines that speak the OpenAI chat-completions protocol."""
    return ai_engine in ("OpenAI (Paid)", DEEPSEEK_ENGINE)


def _oai_client(ai_engine: str, api_key: str):
    from openai import OpenAI
    if ai_engine == DEEPSEEK_ENGINE:
        return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    return OpenAI(api_key=api_key)


def _oai_extra(ai_engine: str) -> dict:
    """Extra request fields. DeepSeek Flash 'thinks' by default (slower, extra billed output
    tokens); VidSage's summarising tasks don't need it."""
    if ai_engine == DEEPSEEK_ENGINE:
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}


def _oai_model(ai_engine: str, size: str = "main") -> str:
    """size: 'main' = full analysis, 'small' = cheap helper calls (cleanup, chapters, tags)."""
    if ai_engine == DEEPSEEK_ENGINE:
        return "deepseek-flash"          # one model: vision-capable, 1M context
    return "gpt-4o-mini" if size == "small" else "gpt-4o"



def extract_audio(video_path: str, audio_path: str):
    """Extract mono 16kHz WAV from video using ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", audio_path, "-y"],
        capture_output=True, check=True
    )


# ── Whisper language catalogue ────────────────────────────────────────────────
# Each entry: display_name → (whisper_lang_code, initial_prompt_hint)
# Regional variants share the parent lang_code but use an initial_prompt
# to guide Whisper toward the correct accent / vocabulary.
# lang_code=None means auto-detect (Whisper chooses per-segment).
WHISPER_LANGUAGES: dict[str, tuple] = {
    "Auto-detect":                              (None, ""),

    # ── Afrikaans ─────────────────────────────────────────────────────
    "Afrikaans":                                ("af", ""),

    # ── Albanian ──────────────────────────────────────────────────────
    "Albanian":                                 ("sq", ""),

    # ── Amharic ───────────────────────────────────────────────────────
    "Amharic":                                  ("am", ""),

    # ── Arabic ────────────────────────────────────────────────────────
    "Arabic":                                   ("ar", ""),
    "Arabic — Modern Standard (فصحى)":          ("ar", "هذا الصوت باللغة العربية الفصحى الحديثة."),
    "Arabic — Egyptian (مصري)":                 ("ar", "الصوت ده بالعربي المصري."),
    "Arabic — Gulf (خليجي)":                    ("ar", "هذا الصوت باللهجة الخليجية."),
    "Arabic — Levantine (شامي)":                ("ar", "هاد الصوت بالعربي الشامي."),
    "Arabic — Moroccan Darija (دارجة)":         ("ar", "هاد الصوت بالدارجة المغربية."),

    # ── Armenian ──────────────────────────────────────────────────────
    "Armenian":                                 ("hy", ""),

    # ── Assamese ──────────────────────────────────────────────────────
    "Assamese":                                 ("as", ""),

    # ── Azerbaijani ───────────────────────────────────────────────────
    "Azerbaijani":                              ("az", ""),

    # ── Bashkir ───────────────────────────────────────────────────────
    "Bashkir":                                  ("ba", ""),

    # ── Basque ────────────────────────────────────────────────────────
    "Basque":                                   ("eu", ""),

    # ── Belarusian ────────────────────────────────────────────────────
    "Belarusian":                               ("be", ""),

    # ── Bengali ───────────────────────────────────────────────────────
    "Bengali":                                  ("bn", ""),
    "Bengali — Bangladesh":                     ("bn", "এই অডিওটি বাংলাদেশী বাংলায়।"),
    "Bengali — West Bengal (India)":            ("bn", "এই অডিওটি পশ্চিমবঙ্গের বাংলায়।"),

    # ── Bosnian ───────────────────────────────────────────────────────
    "Bosnian":                                  ("bs", ""),

    # ── Breton ────────────────────────────────────────────────────────
    "Breton":                                   ("br", ""),

    # ── Bulgarian ─────────────────────────────────────────────────────
    "Bulgarian":                                ("bg", ""),

    # ── Catalan ───────────────────────────────────────────────────────
    "Catalan":                                  ("ca", ""),

    # ── Chinese ───────────────────────────────────────────────────────
    "Chinese — Mandarin Simplified (普通话)":   ("zh", "这段音频是普通话（简体中文）。"),
    "Chinese — Mandarin Traditional (國語)":    ("zh", "這段音頻是國語（繁體中文）。"),
    "Chinese — Shanghainese (上海话)":          ("zh", "这段录音是上海话。"),

    # ── Croatian ──────────────────────────────────────────────────────
    "Croatian":                                 ("hr", ""),

    # ── Czech ─────────────────────────────────────────────────────────
    "Czech":                                    ("cs", ""),

    # ── Danish ────────────────────────────────────────────────────────
    "Danish":                                   ("da", ""),

    # ── Dutch ─────────────────────────────────────────────────────────
    "Dutch":                                    ("nl", ""),
    "Dutch — Netherlands":                      ("nl", "Deze audio is in het Nederlands van Nederland."),
    "Dutch — Belgium (Flemish / Vlaams)":       ("nl", "Deze audio is in het Belgisch-Nederlands (Vlaams)."),

    # ── English ───────────────────────────────────────────────────────
    "English":                                  ("en", ""),
    "English — USA":                            ("en", "This audio is in American English."),
    "English — UK":                             ("en", "This audio is in British English."),
    "English — Australia":                      ("en", "This audio is in Australian English."),
    "English — Canada":                         ("en", "This audio is in Canadian English."),
    "English — India":                          ("en", "This audio is in Indian English."),
    "English — South Africa":                   ("en", "This audio is in South African English."),
    "English — Singapore (Singlish)":           ("en", "This audio is in Singapore English (Singlish)."),
    "English — Malaysia (Manglish)":            ("en", "This audio is in Malaysian English (Manglish)."),
    "English — New Zealand":                    ("en", "This audio is in New Zealand English."),
    "English — Nigeria":                        ("en", "This audio is in Nigerian English."),

    # ── Estonian ──────────────────────────────────────────────────────
    "Estonian":                                 ("et", ""),

    # ── Faroese ───────────────────────────────────────────────────────
    "Faroese":                                  ("fo", ""),

    # ── Finnish ───────────────────────────────────────────────────────
    "Finnish":                                  ("fi", ""),

    # ── French ────────────────────────────────────────────────────────
    "French":                                   ("fr", ""),
    "French — France":                          ("fr", "Cet audio est en français de France."),
    "French — Canada / Québec":                 ("fr", "Cet audio est en français canadien du Québec."),
    "French — Belgium":                         ("fr", "Cet audio est en français de Belgique."),
    "French — Switzerland":                     ("fr", "Cet audio est en français de Suisse."),
    "French — West Africa":                     ("fr", "Cet audio est en français d'Afrique de l'Ouest."),
    "French — North Africa (Maghreb)":          ("fr", "Cet audio est en français du Maghreb."),

    # ── Galician ──────────────────────────────────────────────────────
    "Galician":                                 ("gl", ""),

    # ── Georgian ──────────────────────────────────────────────────────
    "Georgian":                                 ("ka", ""),

    # ── German ────────────────────────────────────────────────────────
    "German":                                   ("de", ""),
    "German — Germany":                         ("de", "Dieses Audio ist auf Hochdeutsch aus Deutschland."),
    "German — Austria (Österreichisch)":        ("de", "Dieses Audio ist auf Österreichischem Deutsch."),
    "German — Switzerland (Schweizerdeutsch)":  ("de", "Dieses Audio ist auf Schweizerdeutsch."),

    # ── Greek ─────────────────────────────────────────────────────────
    "Greek":                                    ("el", ""),

    # ── Gujarati ──────────────────────────────────────────────────────
    "Gujarati":                                 ("gu", ""),

    # ── Haitian Creole ────────────────────────────────────────────────
    "Haitian Creole":                           ("ht", ""),

    # ── Hausa ─────────────────────────────────────────────────────────
    "Hausa":                                    ("ha", ""),

    # ── Hawaiian ──────────────────────────────────────────────────────
    "Hawaiian":                                 ("haw", ""),

    # ── Hebrew ────────────────────────────────────────────────────────
    "Hebrew":                                   ("he", ""),

    # ── Hindi ─────────────────────────────────────────────────────────
    "Hindi":                                    ("hi", ""),

    # ── Hungarian ─────────────────────────────────────────────────────
    "Hungarian":                                ("hu", ""),

    # ── Icelandic ─────────────────────────────────────────────────────
    "Icelandic":                                ("is", ""),

    # ── Indonesian ────────────────────────────────────────────────────
    "Indonesian":                               ("id", ""),

    # ── Italian ───────────────────────────────────────────────────────
    "Italian":                                  ("it", ""),
    "Italian — Italy":                          ("it", "Questo audio è in italiano d'Italia."),
    "Italian — Switzerland":                    ("it", "Questo audio è in italiano della Svizzera."),

    # ── Japanese ──────────────────────────────────────────────────────
    "Japanese":                                 ("ja", ""),

    # ── Javanese ──────────────────────────────────────────────────────
    "Javanese":                                 ("jw", ""),

    # ── Kannada ───────────────────────────────────────────────────────
    "Kannada":                                  ("kn", ""),

    # ── Kazakh ────────────────────────────────────────────────────────
    "Kazakh":                                   ("kk", ""),

    # ── Khmer ─────────────────────────────────────────────────────────
    "Khmer":                                    ("km", ""),

    # ── Korean ────────────────────────────────────────────────────────
    "Korean":                                   ("ko", ""),

    # ── Lao ───────────────────────────────────────────────────────────
    "Lao":                                      ("lo", ""),

    # ── Latin ─────────────────────────────────────────────────────────
    "Latin":                                    ("la", ""),

    # ── Latvian ───────────────────────────────────────────────────────
    "Latvian":                                  ("lv", ""),

    # ── Lingala ───────────────────────────────────────────────────────
    "Lingala":                                  ("ln", ""),

    # ── Lithuanian ────────────────────────────────────────────────────
    "Lithuanian":                               ("lt", ""),

    # ── Luxembourgish ─────────────────────────────────────────────────
    "Luxembourgish":                            ("lb", ""),

    # ── Macedonian ────────────────────────────────────────────────────
    "Macedonian":                               ("mk", ""),

    # ── Malagasy ──────────────────────────────────────────────────────
    "Malagasy":                                 ("mg", ""),

    # ── Malay ─────────────────────────────────────────────────────────
    "Malay":                                    ("ms", ""),
    "Malay — Malaysia (BM)":                    ("ms", "Audio ini dalam Bahasa Malaysia."),
    "Malay — Brunei":                           ("ms", "Audio ini dalam Bahasa Melayu Brunei."),
    "Malay — Singapore":                        ("ms", "Audio ini dalam Bahasa Melayu Singapura."),

    # ── Malayalam ─────────────────────────────────────────────────────
    "Malayalam":                                ("ml", ""),

    # ── Maltese ───────────────────────────────────────────────────────
    "Maltese":                                  ("mt", ""),

    # ── Maori ─────────────────────────────────────────────────────────
    "Maori":                                    ("mi", ""),

    # ── Marathi ───────────────────────────────────────────────────────
    "Marathi":                                  ("mr", ""),

    # ── Mongolian ─────────────────────────────────────────────────────
    "Mongolian":                                ("mn", ""),

    # ── Myanmar / Burmese ─────────────────────────────────────────────
    "Myanmar / Burmese":                        ("my", ""),

    # ── Nepali ────────────────────────────────────────────────────────
    "Nepali":                                   ("ne", ""),

    # ── Norwegian ─────────────────────────────────────────────────────
    "Norwegian (Bokmål)":                       ("no", ""),
    "Norwegian (Nynorsk)":                      ("nn", ""),

    # ── Occitan ───────────────────────────────────────────────────────
    "Occitan":                                  ("oc", ""),

    # ── Pashto ────────────────────────────────────────────────────────
    "Pashto":                                   ("ps", ""),

    # ── Persian / Farsi ───────────────────────────────────────────────
    "Persian / Farsi":                          ("fa", ""),
    "Persian — Iran (Farsi)":                   ("fa", "این صدا به فارسی ایرانی است."),
    "Persian — Afghanistan (Dari)":             ("fa", "این صدا به دری افغانستان است."),

    # ── Polish ────────────────────────────────────────────────────────
    "Polish":                                   ("pl", ""),

    # ── Portuguese ────────────────────────────────────────────────────
    "Portuguese":                               ("pt", ""),
    "Portuguese — Brazil":                      ("pt", "Esse áudio está em português brasileiro."),
    "Portuguese — Portugal":                    ("pt", "Este áudio está em português europeu de Portugal."),
    "Portuguese — Angola":                      ("pt", "Este áudio está em português angolano."),
    "Portuguese — Mozambique":                  ("pt", "Este áudio está em português de Moçambique."),

    # ── Punjabi ───────────────────────────────────────────────────────
    "Punjabi":                                  ("pa", ""),

    # ── Romanian ──────────────────────────────────────────────────────
    "Romanian":                                 ("ro", ""),

    # ── Russian ───────────────────────────────────────────────────────
    "Russian":                                  ("ru", ""),

    # ── Sanskrit ──────────────────────────────────────────────────────
    "Sanskrit":                                 ("sa", ""),

    # ── Serbian ───────────────────────────────────────────────────────
    "Serbian":                                  ("sr", ""),

    # ── Shona ─────────────────────────────────────────────────────────
    "Shona":                                    ("sn", ""),

    # ── Sindhi ────────────────────────────────────────────────────────
    "Sindhi":                                   ("sd", ""),

    # ── Sinhala ───────────────────────────────────────────────────────
    "Sinhala":                                  ("si", ""),

    # ── Slovak ────────────────────────────────────────────────────────
    "Slovak":                                   ("sk", ""),

    # ── Slovenian ─────────────────────────────────────────────────────
    "Slovenian":                                ("sl", ""),

    # ── Somali ────────────────────────────────────────────────────────
    "Somali":                                   ("so", ""),

    # ── Spanish ───────────────────────────────────────────────────────
    "Spanish":                                  ("es", ""),
    "Spanish — Spain (Castellano)":             ("es", "Este audio está en español de España (castellano)."),
    "Spanish — Mexico":                         ("es", "Este audio está en español mexicano."),
    "Spanish — USA / Latino":                   ("es", "Este audio está en español latino de Estados Unidos."),
    "Spanish — Argentina":                      ("es", "Este audio está en español rioplatense de Argentina."),
    "Spanish — Colombia":                       ("es", "Este audio está en español colombiano."),
    "Spanish — Chile":                          ("es", "Este audio está en español chileno."),
    "Spanish — Venezuela":                      ("es", "Este audio está en español venezolano."),
    "Spanish — Peru":                           ("es", "Este audio está en español peruano."),

    # ── Sundanese ─────────────────────────────────────────────────────
    "Sundanese":                                ("su", ""),

    # ── Swahili ───────────────────────────────────────────────────────
    "Swahili":                                  ("sw", ""),

    # ── Swedish ───────────────────────────────────────────────────────
    "Swedish":                                  ("sv", ""),

    # ── Tagalog / Filipino ────────────────────────────────────────────
    "Tagalog / Filipino":                       ("tl", ""),

    # ── Tajik ─────────────────────────────────────────────────────────
    "Tajik":                                    ("tg", ""),

    # ── Tamil ─────────────────────────────────────────────────────────
    "Tamil":                                    ("ta", ""),
    "Tamil — India":                            ("ta", "இந்த ஆடியோ இந்தியத் தமிழில் உள்ளது."),
    "Tamil — Sri Lanka":                        ("ta", "இந்த ஆடியோ இலங்கைத் தமிழில் உள்ளது."),
    "Tamil — Singapore / Malaysia":             ("ta", "இந்த ஆடியோ சிங்கப்பூர்/மலேசியா தமிழில் உள்ளது."),

    # ── Tatar ─────────────────────────────────────────────────────────
    "Tatar":                                    ("tt", ""),

    # ── Telugu ────────────────────────────────────────────────────────
    "Telugu":                                   ("te", ""),

    # ── Thai ──────────────────────────────────────────────────────────
    "Thai":                                     ("th", ""),

    # ── Tibetan ───────────────────────────────────────────────────────
    "Tibetan":                                  ("bo", ""),

    # ── Turkish ───────────────────────────────────────────────────────
    "Turkish":                                  ("tr", ""),

    # ── Turkmen ───────────────────────────────────────────────────────
    "Turkmen":                                  ("tk", ""),

    # ── Ukrainian ─────────────────────────────────────────────────────
    "Ukrainian":                                ("uk", ""),

    # ── Urdu ──────────────────────────────────────────────────────────
    "Urdu":                                     ("ur", ""),
    "Urdu — Pakistan":                          ("ur", "یہ آڈیو پاکستانی اردو میں ہے۔"),
    "Urdu — India":                             ("ur", "یہ آڈیو ہندوستانی اردو میں ہے۔"),

    # ── Uzbek ─────────────────────────────────────────────────────────
    "Uzbek":                                    ("uz", ""),

    # ── Vietnamese ────────────────────────────────────────────────────
    "Vietnamese":                               ("vi", ""),
    "Vietnamese — Northern (Hà Nội)":           ("vi", "Âm thanh này bằng tiếng Việt miền Bắc (Hà Nội)."),
    "Vietnamese — Southern (TP.HCM)":           ("vi", "Âm thanh này bằng tiếng Việt miền Nam (Thành phố Hồ Chí Minh)."),
    "Vietnamese — Central (Huế / Đà Nẵng)":    ("vi", "Âm thanh này bằng tiếng Việt miền Trung."),

    # ── Welsh ─────────────────────────────────────────────────────────
    "Welsh":                                    ("cy", ""),

    # ── Yiddish ───────────────────────────────────────────────────────
    "Yiddish":                                  ("yi", ""),

    # ── Yoruba ────────────────────────────────────────────────────────
    "Yoruba":                                   ("yo", ""),
}

# Flat list of base language names used in the multilingual multi-select.
# Only one entry per language family (no regional variants) — Whisper auto-detects
# the variant when lang_code=None and the initial_prompt lists the languages.
# Whisper treats its "initial prompt" as text that came BEFORE the audio — not as an instruction.
# The old multilingual hint ("Transcribe all languages exactly as spoken…") made it hallucinate
# ("字幕由Amara.org社区提供", "Terima kasih kerana menonton!") or echo the hint back, and even natural
# sample phrases leaked into the transcript and shifted Chinese script / dropped English words in testing.
# So Multilingual mode now sends NO hint: Whisper auto-detects the language by itself.
MULTILINGUAL_BASE_OPTIONS = [
    "Afrikaans", "Albanian", "Amharic", "Arabic", "Armenian", "Azerbaijani",
    "Bashkir", "Basque", "Belarusian", "Bengali", "Bosnian", "Bulgarian",
    "Catalan", "Chinese (Mandarin)", "Croatian", "Czech",
    "Danish", "Dutch", "English", "Estonian",
    "Finnish", "French", "Galician", "Georgian", "German", "Greek", "Gujarati",
    "Haitian Creole", "Hausa", "Hebrew", "Hindi", "Hungarian",
    "Icelandic", "Indonesian", "Italian",
    "Japanese", "Javanese",
    "Kannada", "Kazakh", "Khmer", "Korean",
    "Lao", "Latvian", "Lingala", "Lithuanian",
    "Macedonian", "Malagasy", "Malay", "Malayalam", "Maltese", "Maori", "Marathi",
    "Mongolian", "Myanmar / Burmese",
    "Nepali", "Norwegian",
    "Pashto", "Persian / Farsi", "Polish", "Portuguese", "Punjabi",
    "Romanian", "Russian",
    "Serbian", "Sinhala", "Slovak", "Slovenian", "Somali", "Spanish", "Swahili", "Swedish",
    "Tagalog / Filipino", "Tajik", "Tamil", "Telugu", "Thai", "Turkish",
    "Ukrainian", "Urdu", "Uzbek",
    "Vietnamese", "Welsh", "Yoruba",
]

# Output language options — maps display name → instruction injected into every AI prompt
OUTPUT_LANGUAGES = {
    "English":              "Write your entire response in English.",
    "Bahasa Melayu (BM)":  "Tulis keseluruhan respons anda dalam Bahasa Melayu.",
    "Mandarin (简体)":      "请用简体中文撰写您的全部回复。",
    "Mandarin (繁體)":      "請用繁體中文撰寫您的全部回覆。",
    "Japanese":             "回答はすべて日本語で書いてください。",
    "Korean":               "전체 응답을 한국어로 작성하세요.",
    "Spanish":              "Escribe toda tu respuesta en español.",
    "French":               "Rédigez l'intégralité de votre réponse en français.",
    "Arabic":               "اكتب ردك بالكامل باللغة العربية.",
    "Hindi":                "अपनी पूरी प्रतिक्रिया हिंदी में लिखें।",
}

# Maps Whisper's detected ISO-639-1 language codes → OUTPUT_LANGUAGES keys.
# Used to auto-set the output language after transcription.
WHISPER_LANG_TO_OUTPUT: dict[str, str] = {
    "en":  "English",
    "ms":  "Bahasa Melayu (BM)",
    "zh":  "Mandarin (简体)",
    "yue": "Mandarin (繁體)",   # Cantonese — closest supported option
    "ja":  "Japanese",
    "ko":  "Korean",
    "es":  "Spanish",
    "fr":  "French",
    "ar":  "Arabic",
    "hi":  "Hindi",
}

def transcribe_audio(audio_path: str, model_size: str = "medium",
                     lang_code: str | None = None, initial_prompt: str = ""):
    """Transcribe audio with local Whisper.
    Returns (full_text, timestamped_text, segments, detected_lang_code).
    detected_lang_code is the ISO-639-1 code Whisper identified (e.g. 'zh', 'en', 'ms').
    """
    model = whisper.load_model(model_size, in_memory=False)
    result = model.transcribe(
        audio_path, verbose=False,
        task="transcribe",             # always transcribe (never auto-translate to English)
        condition_on_previous_text=True,
        language=lang_code,            # None = auto-detect language per segment
        initial_prompt=initial_prompt if initial_prompt else None,
    )

    full_text     = result["text"].strip()
    segments      = result.get("segments", [])
    detected_lang = result.get("language") or lang_code or "en"

    # Build a timestamped transcript: [MM:SS] text
    lines = []
    for seg in segments:
        ts  = fmt_time(seg["start"])
        txt = seg["text"].strip()
        if txt:
            lines.append(f"[{ts}] {txt}")
    timestamped_text = "\n".join(lines) if lines else full_text

    return full_text, timestamped_text, segments, detected_lang


def _stt_label(model_or_engine: str) -> str:
    if model_or_engine == "xxl":
        return f"Faster-Whisper-XXL · {globals().get('xxl_model', 'large-v2')} · local GPU"
    return {"groq": "Groq cloud · whisper-large-v3",
            "openai": "OpenAI cloud · whisper-1"}.get(model_or_engine, model_or_engine)


def transcribe_audio_any(audio_path: str, model_size: str = "medium",
                         lang_code: str | None = None, initial_prompt: str = "",
                         engine: str | None = None, groq_api_key: str | None = None,
                         openai_api_key: str | None = None, xxl_exe_path: str | None = None,
                         xxl_model_name: str | None = None):
    """Transcribe with local Whisper (default) or cloud Whisper (Groq / OpenAI).
    Same return shape as transcribe_audio().  The engine and keys default to the sidebar choices."""
    engine = engine if engine is not None else globals().get("stt_engine", "local")
    if engine == "local":
        return transcribe_audio(audio_path, model_size, lang_code, initial_prompt)
    if engine == "xxl":
        exe = xxl_exe_path or globals().get("xxl_exe")
        if not exe:
            raise ValueError("Faster-Whisper-XXL wasn't found — paste its folder in the sidebar (Transcription).")
        segments, detected_lang = transcribe_xxl(
            audio_path, exe, xxl_model_name or globals().get("xxl_model", "large-v2"),
            language=lang_code, prompt=initial_prompt, hotwords=globals().get("names_terms", ""),
            noise_filter=globals().get("xxl_filter", "Off"))
        full_text = " ".join(sg["text"] for sg in segments).strip()
        timestamped_text = "\n".join(f"[{fmt_time(sg['start'])}] {sg['text']}" for sg in segments) or full_text
        return full_text, timestamped_text, segments, detected_lang

    key = (groq_api_key if groq_api_key is not None else globals().get("groq_key", "")) \
        if engine == "groq" else \
        (openai_api_key if openai_api_key is not None else globals().get("openai_stt_key", ""))
    segments, detected_lang = transcribe_cloud(
        audio_path, engine, key, language=lang_code, prompt=initial_prompt)
    full_text = " ".join(sg["text"] for sg in segments).strip()
    timestamped_text = "\n".join(f"[{fmt_time(sg['start'])}] {sg['text']}" for sg in segments) or full_text
    return full_text, timestamped_text, segments, detected_lang


def timestamped_to_srt(timestamped_text: str) -> str:
    """
    Convert the app's timestamped transcript format back to SRT.

    Input lines look like:  [MM:SS] text   or   [H:MM:SS] text
    End time is inferred as the start of the next segment (or start + 5 s for
    the last one).  Milliseconds are zeroed since the source has no sub-second
    precision.
    """
    import re as _re
    _ts_pat = _re.compile(r"^\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.*)")

    def _parse_ts(ts_str: str) -> float:
        parts = ts_str.split(":")
        parts = [float(p) for p in parts]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0] * 3600 + parts[1] * 60 + parts[2]

    def _fmt_srt(sec: float) -> str:
        ms = int((sec % 1) * 1000)
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    segs = []
    for line in timestamped_text.splitlines():
        m = _ts_pat.match(line.strip())
        if m:
            segs.append((_parse_ts(m.group(1)), m.group(2).strip()))

    if not segs:
        return ""

    srt_blocks = []
    for idx, (start, text) in enumerate(segs):
        end = segs[idx + 1][0] if idx + 1 < len(segs) else start + 5.0
        if text:
            srt_blocks.append(
                f"{idx + 1}\n{_fmt_srt(start)} --> {_fmt_srt(end)}\n{text}\n"
            )
    return "\n".join(srt_blocks)


def generate_srt(segments: list) -> str:
    """Convert Whisper segments into a valid SRT subtitle file string."""
    def srt_time(seconds: float) -> str:
        ms = int((seconds % 1) * 1000)
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(segments, start=1):
        start = srt_time(seg["start"])
        end   = srt_time(seg["end"])
        text  = seg["text"].strip()
        if text:
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    return "\n".join(lines)


def frame_to_b64(frame, max_dim: int = 1280) -> str:
    """Convert an OpenCV BGR frame to a base64 JPEG string.

    max_dim controls the longest side in pixels before JPEG encoding.
    1280 (720p-equivalent) is a good balance: text and diagrams remain
    readable by Claude/Gemini while keeping token cost reasonable.
    Claude vision optimal width for detailed content is ~1568 px;
    Gemini supports up to 3072 px.  We stay at 1280 by default.
    """
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_LANCZOS4)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


def score_text_density(frame) -> float:
    """
    Score a BGR frame by how much text/data it likely contains.
    Uses two fast, dependency-free signals:
      1. Canny edge density  — text characters create lots of sharp edges
      2. Laplacian variance  — in-focus text has high local contrast
    Returns a float; higher = more text/data content.
    Runs in <5 ms per frame on CPU.
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize to a small thumbnail for speed
    small = cv2.resize(gray, (320, 180))
    # Signal 1: edge density
    edges = cv2.Canny(small, 50, 150)
    edge_score = float(edges.mean())
    # Signal 2: sharpness/contrast (Laplacian variance)
    lap_score = float(cv2.Laplacian(small, cv2.CV_64F).var())
    # Blend — edge density weighted higher since it maps better to text glyphs
    return edge_score * 0.65 + min(lap_score / 50.0, 30.0) * 0.35


def select_text_dense_frames(candidates: list[tuple], keep: int) -> list[tuple]:
    """
    Given a list of (b64_str, timestamp, raw_frame) tuples, score each by
    text density and return the top `keep` sorted chronologically.
    If fewer candidates than `keep`, all are returned.
    """
    if len(candidates) <= keep:
        return [(b64, ts) for b64, ts, _ in candidates]
    scored = [(score_text_density(frame), b64, ts)
              for b64, ts, frame in candidates]
    scored.sort(key=lambda x: -x[0])          # highest score first
    top = scored[:keep]
    top.sort(key=lambda x: x[2])              # re-sort chronologically
    return [(b64, ts) for _, b64, ts in top]


def infer_video_type(video_name: str) -> str:
    """
    Infer a human-readable video type label from the filename.
    Used in AI prompts to give context without requiring manual selection.
    """
    name = video_name.lower()
    if any(k in name for k in ("zoom", "meeting", "call", "gmeet", "teams")):
        return "Zoom Meeting"
    if any(k in name for k in ("lecture", "class", "lesson", "course", "tutorial")):
        return "Lecture"
    if any(k in name for k in ("training", "workshop", "webinar", "onboard")):
        return "Training Session"
    if any(k in name for k in ("screen", "record", "demo", "walkthrough", "loom")):
        return "Screen Recording"
    return "Video"


def get_video_duration(video_path: str) -> float:
    """Fast metadata-only probe — no frames decoded."""
    cap = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total / fps


def recommended_frames(duration_sec: float, frame_mode: str, ai_engine: str) -> int:
    """
    Calculate the recommended frame count for a given video.

    Rules:
    - Base cadence depends on mode:
        Fixed Interval    → 1 frame per 3 min
        Smart Scene       → 1 frame per 2.5 min
        Speech-Aligned /
        Dense + Dedup     → 1 frame per 2 min  (dedup removes duplicates)
    - Hard floor: 8 frames (even very short videos deserve a few frames).
    - Soft ceiling scales with duration so long videos get adequate coverage:
        Claude / OpenAI   → 40 (≤1 hr) · 55 (≤3 hr) · 70 (>3 hr)
        Gemini            → 80 (≤1 hr) · 110 (≤3 hr) · 140 (>3 hr)
    """
    minutes = duration_sec / 60

    if frame_mode in ("🎙️ Speech-Aligned", "🔍 Dense + Dedup"):
        count = max(8, round(minutes / 2))
    elif frame_mode == "🧠 Smart Scene Detection":
        count = max(8, round(minutes / 2.5))
    else:
        count = max(8, round(minutes / 3))

    if ai_engine == "Claude (Paid)" or _is_oai(ai_engine):
        ceiling = 70 if duration_sec > 10_800 else (55 if duration_sec > 3_600 else 40)
    else:  # Gemini
        ceiling = 140 if duration_sec > 10_800 else (110 if duration_sec > 3_600 else 80)

    return min(count, ceiling)


def extract_keyframes(video_path: str, interval_sec: int = 45, max_frames: int = 15):
    """
    Fixed-interval frame extraction with text-density selection.
    Over-samples at 3× density, then keeps the `max_frames` most text-rich
    candidates so decorative/blank frames are filtered out automatically.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Over-sample at 3× density for text-density selection
    oversample = max(1, interval_sec // 3)
    step = max(int(fps * oversample), 1)
    candidates = []   # (b64, timestamp, raw_frame)

    for frame_idx in range(0, total_frames, step):
        if len(candidates) >= max_frames * 3:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        candidates.append((frame_to_b64(frame), frame_idx / fps, frame))

    cap.release()
    pairs = select_text_dense_frames(candidates, max_frames)
    frames_b64  = [b for b, _ in pairs]
    timestamps  = [t for _, t in pairs]
    return frames_b64, timestamps, duration


def extract_scene_frames(video_path: str, max_frames: int = 15,
                         threshold: float = 0.4, min_gap_sec: float = 3.0):
    """
    Smart scene-detection frame extraction.
    Captures a frame whenever the visual content changes significantly
    (e.g. new slide, screen switch, new content appears).

    threshold  : 0.0–1.0 — sensitivity. Lower = more frames captured.
                 0.3 = very sensitive, 0.5 = moderate, 0.7 = only big changes.
    min_gap_sec: minimum seconds between two captured frames (avoids duplicates
                 during fast transitions).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    min_gap_frames = int(fps * min_gap_sec)

    # Scene candidates — collect ALL scene changes (up to 3× target), then
    # apply text-density selection to keep only the most content-rich frames.
    candidates = []   # (b64, timestamp, raw_frame)
    prev_gray = None
    last_captured = -min_gap_frames

    check_step = max(int(fps * 0.5), 1)
    oversample_limit = max_frames * 3

    for frame_idx in range(0, total_frames, check_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))

        if prev_gray is None:
            candidates.append((frame_to_b64(frame), frame_idx / fps, frame))
            last_captured = frame_idx
            prev_gray = gray
            continue

        diff  = cv2.absdiff(gray, prev_gray)
        score = diff.mean() / 255.0
        gap_ok = (frame_idx - last_captured) >= min_gap_frames

        if score >= threshold and gap_ok:
            candidates.append((frame_to_b64(frame), frame_idx / fps, frame))
            last_captured = frame_idx
            if len(candidates) >= oversample_limit:
                break

        prev_gray = gray

    cap.release()
    pairs = select_text_dense_frames(candidates, max_frames)
    frames_b64 = [b for b, _ in pairs]
    timestamps  = [t for _, t in pairs]
    return frames_b64, timestamps, duration


def extract_speech_aligned_frames(video_path: str, segments: list,
                                   min_words: int = 8, max_frames: int = 15) -> tuple:
    """
    Capture frames guided by Whisper transcript segments.

    Strategy (applied in priority order):
      1. Score every qualifying segment by words-per-second (content density).
      2. Add a bonus frame at the START of any segment that follows a silence
         gap > 1.5 s — these are natural topic / slide boundaries.
      3. Select the top-scoring candidates, seek to 0.5 s into each segment
         (avoiding transition flashes at t=0), and capture.
      4. Skip near-duplicate frames using a pixel-change signature (64×36 grayscale,
         similarity ≥ 92 % → discard).

    Returns (frames_b64, timestamps, duration).
    """
    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps

    # ── Build scored candidates ───────────────────────────────────────────────
    candidates = []   # list of (timestamp_sec, score)

    for i, seg in enumerate(segments):
        words = seg["text"].strip().split()
        if len(words) < min_words:
            continue

        seg_dur = max(seg["end"] - seg["start"], 0.1)
        wps     = len(words) / seg_dur          # words-per-second content density

        # Capture 0.5 s into the segment (skip any transition flash at t=0)
        ts = min(seg["start"] + 0.5, seg["end"] - 0.1)
        candidates.append((ts, wps))

        # Silence-gap bonus: next segment starts after a 1.5 s+ pause
        if i + 1 < len(segments):
            gap = segments[i + 1]["start"] - seg["end"]
            if gap > 1.5:
                gap_ts = min(segments[i + 1]["start"] + 0.3, segments[i + 1]["end"])
                candidates.append((gap_ts, wps * 1.3))   # 30 % score boost

    if not candidates:
        # Fallback: nothing qualified — use fixed interval
        cap.release()
        return extract_keyframes(video_path, interval_sec=45, max_frames=max_frames)

    # ── Select top candidates by score, then re-sort chronologically ─────────
    candidates.sort(key=lambda x: -x[1])
    top = candidates[: max_frames * 3]          # over-fetch before dedup
    top.sort(key=lambda x: x[0])               # chronological for efficient seeking

    # ── Capture + deduplicate → collect all non-duplicate candidates ─────────
    raw_candidates = []   # (b64, ts, raw_frame) — up to max_frames*3
    accepted_hashes = []

    for ts, _ in top:
        if len(raw_candidates) >= max_frames * 3:
            break
        frame_idx = min(int(ts * fps), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        phash = _frame_signature(frame)
        if _frame_is_duplicate(phash, accepted_hashes):
            continue
        raw_candidates.append((frame_to_b64(frame), ts, frame))
        accepted_hashes.append(phash)

    cap.release()

    # ── Apply text-density selection to keep the most content-rich frames ────
    pairs = select_text_dense_frames(raw_candidates, max_frames)
    frames_b64 = [b for b, _ in pairs]
    timestamps  = [t for _, t in pairs]
    return frames_b64, timestamps, duration


def extract_dense_dedup_frames(video_path: str,
                               interval_sec: float = 2.0,
                               max_frames: int = 20,
                               phash_threshold: float = 0.88) -> tuple:
    """
    Dense extraction + perceptual-hash deduplication.

    Strategy:
      1. Extract one frame every `interval_sec` seconds (typically 2–3 s).
      2. Discard near-duplicates using a 64×36 pixel-change signature;
         frames with similarity ≥ phash_threshold are skipped.
      3. From the surviving unique frames, keep the `max_frames`
         most text-dense ones (edge density + Laplacian variance).

    Best for: Zoom meetings, lectures, training sessions, screen recordings.
    Dense sampling catches every slide change; pHash removes the 85–95 % of
    frames where nothing has visually changed (static slide, speaker cam, etc.).
    """
    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps

    step              = max(int(fps * interval_sec), 1)
    unique_candidates = []   # (b64, ts, raw_frame)
    accepted_hashes   = []

    for frame_idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        phash = _frame_signature(frame)
        if _frame_is_duplicate(phash, accepted_hashes, threshold=phash_threshold):
            continue
        ts = frame_idx / fps
        unique_candidates.append((frame_to_b64(frame), ts, frame))
        accepted_hashes.append(phash)

    cap.release()

    if not unique_candidates:
        return extract_keyframes(video_path, interval_sec=45, max_frames=max_frames)

    # Keep the most text-dense unique frames, sorted chronologically
    pairs      = select_text_dense_frames(unique_candidates, max_frames)
    frames_b64 = [b for b, _ in pairs]
    timestamps = [t for _, t in pairs]
    return frames_b64, timestamps, duration


def cleanup_transcript(full_text: str, timestamped_text: str, segments: list,
                       language_label: str, video_type: str,
                       ai_engine: str, claude_key: str, gemini_key: str,
                       openai_key: str = "") -> tuple:
    """
    Use AI to fix transcription errors while preserving meaning and timestamps.
    Returns (cleaned_full_text, cleaned_timestamped_text, cleaned_srt).
    """
    prompt = f"""You are a professional transcript editor. The following transcript was auto-generated
by Whisper speech recognition from a {video_type} video. The audio language is: {language_label}.

Your job is to CORRECT transcription errors only:
- Fix misheared words based on context
- Fix Malaysian/local names, brands, and terms that were mangled
- Fix code-switching errors (Malay, Mandarin, or Cantonese words transcribed as gibberish)
- Fix grammar only where it's clearly a transcription error, not the speaker's natural speech
- DO NOT rephrase, summarise, or change the speaker's meaning
- DO NOT remove filler words like "lah", "mah", "wah", "kan" — these are intentional
- DO NOT translate — keep EVERY segment in the exact same language it was spoken in
- If the speaker switches languages mid-sentence, preserve that code-switching exactly
- Keep ALL timestamp markers exactly as they are (e.g. [01:23])
- Return ONLY the corrected transcript, nothing else

TRANSCRIPT TO CORRECT:
{timestamped_text}"""

    if ai_engine == "Claude (Paid)":
        import anthropic
        client = anthropic.Anthropic(api_key=claude_key)
        response = client.messages.create(
            model="claude-haiku-4-5",  # use fast cheap model for cleanup
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        cleaned_timestamped = response.content[0].text.strip()
    elif _is_oai(ai_engine):
        from openai import OpenAI as _OAI
        _oc = _oai_client(ai_engine, openai_key)
        _or = _oc.chat.completions.create(
            model=_oai_model(ai_engine, "small"), **_oai_extra(ai_engine),
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        cleaned_timestamped = _or.choices[0].message.content.strip()
    else:
        from google import genai as google_genai
        client = google_genai.Client(api_key=gemini_key)
        response = _gemini_generate(client, prompt)
        cleaned_timestamped = response.text.strip()

    # Rebuild full text from cleaned timestamped version (strip timestamps)
    cleaned_full = re.sub(r'\[\d{2}:\d{2}(?::\d{2})?\]\s*', '', cleaned_timestamped).strip()

    return cleaned_full, cleaned_timestamped


def generate_chapters(timestamped_transcript: str, duration: float,
                      video_type: str, ai_engine: str,
                      claude_key: str, gemini_key: str,
                      output_language: str = "English",
                      openai_key: str = "") -> list:
    """
    Use AI to detect logical chapter breaks from the timestamped transcript.
    Returns a list of dicts: [{"time": "00:00", "title": "Introduction"}, ...]
    """
    # Aim for roughly 1 chapter per 5 minutes, min 3, max 15
    target = max(3, min(20, int(duration / 300)))
    lang_instruction = OUTPUT_LANGUAGES.get(output_language, OUTPUT_LANGUAGES["English"])

    # ── Transcript sampling ───────────────────────────────────────────────────
    # For long videos we sample evenly across the full transcript so the AI
    # sees content from every part of the lecture, not just the first few mins.
    _MAX_CHAP_CHARS = 40_000   # ~10 k tokens — fine for Haiku / Flash / gpt-4o-mini
    def _sample_transcript(text: str, max_chars: int) -> tuple[str, bool]:
        if len(text) <= max_chars:
            return text, False
        lines = text.strip().split("\n")
        # Keep every Nth line so samples are spread across the full duration
        keep = max(1, len(lines) * max_chars // max(len(text), 1))
        step = max(1, len(lines) // keep)
        sampled = "\n".join(lines[i] for i in range(0, len(lines), step))
        return sampled[:max_chars], True

    _ts_sample, _was_sampled = _sample_transcript(timestamped_transcript, _MAX_CHAP_CHARS)
    _sample_note = (
        "\n(Note: transcript is sampled evenly across the full duration — "
        "timestamps are accurate but some sections are condensed.)\n"
        if _was_sampled else ""
    )

    prompt = f"""You are analysing a {video_type} video that is {fmt_time(duration)} long.
Below is the timestamped transcript. Identify {target}–{target + 3} logical chapter breaks
where the topic, section, or focus clearly changes.

Rules:
- First chapter MUST start at 00:00
- Use the exact timestamps from the transcript (format MM:SS or HH:MM:SS)
- Chapter titles should be short (3–6 words), descriptive, and professional
- Spread chapters across the FULL {fmt_time(duration)} duration — do not cluster at the start
- Return ONLY a plain list, one chapter per line, in this exact format:
  00:00 - Chapter Title
  05:30 - Another Chapter Title
- No extra text, no numbering, no markdown
- {lang_instruction}

TRANSCRIPT:{_sample_note}
{_ts_sample}"""

    if ai_engine == "Claude (Paid)":
        import anthropic
        client = anthropic.Anthropic(api_key=claude_key)
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
    elif _is_oai(ai_engine):
        from openai import OpenAI as _OAI
        _oc = _oai_client(ai_engine, openai_key)
        _or = _oc.chat.completions.create(
            model=_oai_model(ai_engine, "small"), **_oai_extra(ai_engine),
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _or.choices[0].message.content.strip()
    else:
        from google import genai as google_genai
        client = google_genai.Client(api_key=gemini_key)
        response = _gemini_generate(client, prompt)
        raw = response.text.strip()

    chapters = []
    for line in raw.split("\n"):
        line = line.strip()
        match = re.match(r"^(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–]\s*(.+)$", line)
        if match:
            chapters.append({"time": match.group(1), "title": match.group(2).strip()})

    # Fallback — if parsing failed, return a single chapter
    if not chapters:
        chapters = [{"time": "00:00", "title": video_type}]

    return chapters


def chapters_to_text(chapters: list) -> str:
    """Format chapters as plain text (YouTube-style)."""
    return "\n".join(f"{c['time']} - {c['title']}" for c in chapters)


def answer_question(question: str, explanation: str, timestamped_transcript: str,
                    chapters: list, video_type: str, chat_history: list,
                    ai_engine: str, claude_key: str, gemini_key: str,
                    openai_key: str = "") -> str:
    """Answer a question about the video using the transcript and explanation as context."""

    chapters_text = chapters_to_text(chapters) if chapters else ""

    system = f"""You are a helpful assistant answering questions about a {video_type} video.
You have access to:
1. The full AI-generated explanation of the video
2. The complete timestamped transcript
3. The chapter markers

Answer questions accurately based on this content. When relevant, reference specific
timestamps (e.g. "At 05:30, the speaker mentions..."). Be concise but thorough.
If something is not covered in the video, say so clearly."""

    context = f"""=== VIDEO EXPLANATION ===
{explanation}

=== CHAPTER MARKERS ===
{chapters_text}

=== TIMESTAMPED TRANSCRIPT ===
{timestamped_transcript}"""

    if ai_engine == "Claude (Paid)":
        import anthropic
        client = anthropic.Anthropic(api_key=claude_key)

        messages = []
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": f"{context}\n\n---\nQuestion: {question}"
                         if not chat_history else question})

        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        return response.content[0].text.strip()

    elif _is_oai(ai_engine):
        from openai import OpenAI as _OAI
        _oc = _oai_client(ai_engine, openai_key)

        messages = []
        messages.append({"role": "system", "content": system})
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": f"{context}\n\n---\nQuestion: {question}"
                         if not chat_history else question})

        _or = _oc.chat.completions.create(
            model=_oai_model(ai_engine, "small"), **_oai_extra(ai_engine),
            max_tokens=1024,
            messages=messages,
        )
        return _or.choices[0].message.content.strip()

    else:
        from google import genai as google_genai
        client = google_genai.Client(api_key=gemini_key)

        # Build conversation history for Gemini
        history_text = ""
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n\n"

        full_prompt = f"{system}\n\n{context}\n\n{history_text}User: {question}\nAssistant:"
        response = _gemini_generate(client, full_prompt)
        return response.text.strip()


# ── Markdown rendering helpers (shared by Word + PDF export) ─────────────────

def _md_split_bold(text: str) -> list:
    """Split text at ** markers → [(segment, is_bold), ...]. Skips empty parts."""
    parts = re.split(r'\*\*', text)
    return [(p, i % 2 == 1) for i, p in enumerate(parts) if p]


def _md_is_sep_row(line: str) -> bool:
    """True if this is a markdown table separator row like |---|---|."""
    return bool(re.match(r'^\|[-:| ]+\|$', line.strip()))


def _md_parse_table(lines: list) -> list:
    """Parse markdown table lines → list of row lists, separator rows skipped."""
    rows = []
    for ln in lines:
        if _md_is_sep_row(ln):
            continue
        cells = [c.strip() for c in ln.strip().strip('|').split('|')]
        rows.append(cells)
    return rows


def _md_plain(text: str) -> str:
    """Strip all ** markers from text (for contexts that can't do inline bold)."""
    return re.sub(r'\*\*', '', text)


def _md_normalize(line: str) -> str:
    """
    Normalise a markdown line for Word/PDF rendering:
    - Convert task-list checkboxes (- [ ]) to plain bullets
    - Convert emoji/dingbat bullet markers at line start to '- '
    - Strip all emoji & dingbats that CJK fonts (SimHei etc.) cannot render
      (they would appear as □ tofu boxes in the PDF)
    Safe to call on any line; CJK text and standard punctuation are preserved.
    """
    # 1. Markdown task list: '- [ ]' / '- [x]' → '- '
    line = re.sub(r'^(\s*)-\s+\[[x ]\]\s*', r'\1- ', line)

    # 2. Geometric shapes (□◻◼▪▫ etc.) or ballot boxes at start → bullet
    line = re.sub(r'^(\s*)[\u2610-\u2612\u25a0-\u25ff]\s+', r'\1- ', line)

    # 3. Dingbats (✅❌✓✔☑ etc.) at start of line → bullet
    line = re.sub(r'^(\s*)[\u2700-\u27bf]\s*', r'\1- ', line)

    # 4. Emoji (📌🎯💡 etc.) at start of line → bullet
    line = re.sub(u'^(\\s*)[\U0001F000-\U0001FFFF]\\s*', r'\1- ', line)

    # 5. Strip ALL remaining emoji & dingbats from anywhere in the line
    #    These show as □ in SimHei/SimSun — better to remove them entirely
    line = re.sub(u'[\U0001F000-\U0001FFFF]', '', line)   # emoji block
    line = re.sub(r'[\u2700-\u27bf]', '', line)            # dingbats (✅❌✓ etc.)
    line = re.sub(r'\ufe0f', '', line)                     # variation selector-16

    return line


# ── Word helpers ──────────────────────────────────────────────────────────────

def _word_add_runs(para, text: str):
    """Add runs to a docx paragraph, honouring **bold** markers."""
    from docx.shared import RGBColor
    for segment, bold in _md_split_bold(text):
        run = para.add_run(segment)
        run.bold = bold


def _word_add_table(doc, table_lines: list):
    """Convert markdown table lines into a proper Word table."""
    rows = _md_parse_table(table_lines)
    if not rows:
        return
    max_cols = max(len(r) for r in rows)
    tbl = doc.add_table(rows=len(rows), cols=max_cols)
    tbl.style = 'Table Grid'
    for ri, row_cells in enumerate(rows):
        for ci in range(max_cols):
            cell_text = row_cells[ci] if ci < len(row_cells) else ""
            cell = tbl.rows[ri].cells[ci]
            para = cell.paragraphs[0]
            for segment, bold in _md_split_bold(cell_text):
                run = para.add_run(segment)
                run.bold = bold or (ri == 0)   # header row always bold
    doc.add_paragraph()                         # spacing after table


def _word_render_md(doc, text: str):
    """Render a markdown string into a python-docx Document object."""
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = _md_normalize(lines[i]).strip()

        # ── Markdown table block ──────────────────────────────────────────
        if line.startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            _word_add_table(doc, table_lines)
            continue

        # ── Normal lines ──────────────────────────────────────────────────
        if not line or line in ('---', '***', '___'):
            if line in ('---', '***', '___'):
                pass   # skip horizontal rule markers (no visible rule in Word)
            else:
                doc.add_paragraph()
        elif line.startswith("#### "):
            p = doc.add_heading(level=3)
            _word_add_runs(p, _md_plain(line[5:]))
        elif line.startswith("### "):
            p = doc.add_heading(level=2)
            _word_add_runs(p, _md_plain(line[4:]))
        elif line.startswith("## "):
            p = doc.add_heading(level=1)
            _word_add_runs(p, _md_plain(line[3:]))
        elif line.startswith("- ") or line.startswith("* "):
            p = doc.add_paragraph(style="List Bullet")
            _word_add_runs(p, line[2:])
        elif re.match(r"^\d+\. ", line):
            p = doc.add_paragraph(style="List Number")
            _word_add_runs(p, re.sub(r"^\d+\. ", "", line))
        else:
            p = doc.add_paragraph()
            _word_add_runs(p, line)

        i += 1


# ── PDF helpers ───────────────────────────────────────────────────────────────
# NOTE: fpdf2's write() is unreliable with CJK TTC fonts — it drops characters.
# All PDF rendering therefore uses multi_cell() / cell() exclusively.
# Bold markers (**) are stripped via _md_plain() so they never show literally.

def _pdf_render_table(pdf, table_lines: list, fname: str):
    """Render a markdown table in the PDF with bordered cells."""
    from fpdf import XPos, YPos
    rows = _md_parse_table(table_lines)
    if not rows:
        return
    max_cols = max(len(r) for r in rows)
    if max_cols == 0:
        return

    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    col_w  = page_w / max_cols
    cell_h = 7

    pdf.ln(2)
    pdf.set_draw_color(160, 160, 200)

    for ri, row_cells in enumerate(rows):
        is_hdr = (ri == 0)
        pdf.set_fill_color(*(220, 225, 250) if is_hdr else (252, 252, 255))
        pdf.set_font(fname, "B" if is_hdr else "", 9)
        pdf.set_text_color(20, 20, 60 if is_hdr else 40)

        row_x = pdf.get_x()
        row_y = pdf.get_y()

        # Measure each cell to find the tallest row
        line_height = 5
        col_heights = []
        for ci in range(max_cols):
            raw  = row_cells[ci] if ci < len(row_cells) else ""
            text = _md_plain(raw).strip()
            # Estimate lines needed using fpdf2's get_string_width
            try:
                pdf.set_font(fname, "B" if is_hdr else "", 9)
                words = text.split()
                cur_w, lines_n = 0.0, 1
                for w in words:
                    ww = pdf.get_string_width(w + " ")
                    if cur_w + ww > col_w - 2:
                        lines_n += 1
                        cur_w = ww
                    else:
                        cur_w += ww
            except Exception:
                lines_n = max(1, len(text) // max(1, int(col_w / 3)))
            col_heights.append(max(1, lines_n))

        row_h = max(col_heights) * line_height + 4

        for ci in range(max_cols):
            raw  = row_cells[ci] if ci < len(row_cells) else ""
            text = _md_plain(raw).strip()
            pdf.set_xy(row_x + ci * col_w, row_y)
            pdf.set_font(fname, "B" if is_hdr else "", 9)
            pdf.set_fill_color(*(220, 225, 250) if is_hdr else (252, 252, 255))
            pdf.multi_cell(col_w, line_height, text,
                           border=1, fill=is_hdr, align='L',
                           new_x=XPos.RIGHT, new_y=YPos.TOP)

        pdf.set_xy(row_x, row_y + row_h)

    pdf.ln(4)


def _pdf_render_md(pdf, text: str, fname: str):
    """Render a markdown string into an fpdf2 PDF object.
    Uses multi_cell() exclusively — safe with CJK TTC fonts.
    Bold markers (**) are stripped so they never appear literally.
    """
    from fpdf import XPos, YPos
    NL = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}

    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = _md_normalize(lines[i]).strip()

        # ── Markdown table block ──────────────────────────────────────────
        if line.startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            _pdf_render_table(pdf, table_lines, fname)
            continue

        # ── Normal lines ──────────────────────────────────────────────────
        clean = _md_plain(line)   # strip all ** markers

        if not clean:
            pdf.ln(3)
        elif re.match(r'^-{3,}$', line) or re.match(r'^\*{3,}$', line) or re.match(r'^_{3,}$', line):
            # Horizontal rule (---, ***, ___, or longer variants)
            pdf.set_draw_color(200, 200, 220)
            pdf.line(pdf.l_margin, pdf.get_y(),
                     pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(5)
        elif line.startswith("#### "):
            pdf.set_font(fname, "B", 11)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 6, _md_plain(line[5:]), **NL)
        elif line.startswith("### "):
            pdf.set_font(fname, "B", 12)
            pdf.set_text_color(50, 50, 80)
            pdf.ln(2)
            pdf.multi_cell(0, 7, _md_plain(line[4:]), **NL)
        elif line.startswith("## "):
            pdf.set_font(fname, "B", 14)
            pdf.set_text_color(30, 30, 30)
            pdf.ln(4)
            pdf.multi_cell(0, 8, _md_plain(line[3:]), **NL)
            pdf.ln(2)
        elif line.startswith("# "):
            # Top-level heading
            pdf.set_font(fname, "B", 16)
            pdf.set_text_color(20, 20, 20)
            pdf.ln(4)
            pdf.multi_cell(0, 9, _md_plain(line[2:]), **NL)
            pdf.ln(3)
        elif line.startswith("- ") or line.startswith("* "):
            pdf.set_font(fname, "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(0, 6, "  \u2022  " + _md_plain(line[2:]), **NL)
        elif re.match(r"^\d+\. ", line):
            num    = re.match(r"^(\d+)\.", line).group(1)
            body   = _md_plain(re.sub(r"^\d+\. ", "", line))
            pdf.set_font(fname, "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(0, 6, f"  {num}.  {body}", **NL)
        else:
            pdf.set_font(fname, "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(0, 6, clean, **NL)

        i += 1


# ─────────────────────────────────────────────────────────────────────────────

def export_word(video_name: str, explanation: str,
                full_transcript: str, timestamped_transcript: str,
                chapters: list = None) -> bytes:
    """Generate a formatted Word (.docx) document and return as bytes."""
    from docx import Document
    from docx.shared import RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()

    # Title
    title = doc.add_heading("Video Analysis Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph(f"File: {video_name}")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    date_para = doc.add_paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}")
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_para.runs[0].font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    doc.add_paragraph()

    # Chapters section
    if chapters:
        doc.add_heading("Chapter Markers", level=1)
        for c in chapters:
            p = doc.add_paragraph(style="List Bullet")
            run = p.add_run(f"{c['time']}  \u2014  {c['title']}")
            run.font.color.rgb = RGBColor(0x1F, 0x5C, 0x99)
        doc.add_paragraph()

    # Explanation — full markdown rendering
    _word_render_md(doc, explanation)

    # Transcript section
    doc.add_page_break()
    doc.add_heading("Full Transcript", level=1)
    doc.add_paragraph(full_transcript)

    doc.add_page_break()
    doc.add_heading("Timestamped Transcript", level=1)
    doc.add_paragraph(timestamped_transcript)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def export_pdf(video_name: str, explanation: str,
               full_transcript: str, timestamped_transcript: str) -> bytes:
    """
    Generate a PDF using reportlab with CJK font support.
    Pure Python — no Microsoft Word, no popups, no subprocesses.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable, PageBreak)
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import xml.sax.saxutils as _sax

    # ── Register a CJK-capable font ───────────────────────────────────────────
    _fn = "VidSagePDF"
    if _fn not in pdfmetrics.getRegisteredFontNames():
        _registered = False
        for _fp in [
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simsun.ttc",
            r"C:\Windows\Fonts\arial.ttf",
        ]:
            if os.path.exists(_fp):
                try:
                    pdfmetrics.registerFont(TTFont(_fn, _fp))
                    _registered = True
                    break
                except Exception:
                    continue
        if not _registered:
            _fn = "Helvetica"
    else:
        pass  # already registered from a previous call

    # ── Paragraph styles ──────────────────────────────────────────────────────
    def _ps(name, size=10, color="#1a1a1a", sb=2, sa=4, leading=None, indent=0):
        # Append font name to style name to avoid cross-request conflicts
        return ParagraphStyle(
            f"{name}_{_fn}", fontName=_fn, fontSize=size,
            leading=leading or round(size * 1.45),
            textColor=colors.HexColor(color),
            spaceBefore=sb, spaceAfter=sa,
            leftIndent=indent, wordWrap='CJK',
        )

    S_TITLE   = _ps("title",  22, "#333366", sb=0,  sa=6,  leading=28)
    S_SUB     = _ps("sub",    11, "#888888", sb=0,  sa=3)
    S_H1      = _ps("h1",     18, "#1a1a2e", sb=14, sa=5,  leading=24)
    S_H2      = _ps("h2",     14, "#1a1a4e", sb=10, sa=4,  leading=20)
    S_H3      = _ps("h3",     12, "#2a2a5e", sb=8,  sa=3,  leading=17)
    S_H4      = _ps("h4",     11, "#3a3a6e", sb=6,  sa=2,  leading=15)
    S_BODY    = _ps("body",   10, "#1a1a1a", sb=1,  sa=3)
    S_BULLET  = _ps("bull",   10, "#1a1a1a", sb=1,  sa=2,  indent=8)
    S_TCELL   = _ps("tcell",   9, "#1a1a1a", sb=0,  sa=0)
    S_TCELL_H = _ps("tcellh",  9, "#14143C", sb=0,  sa=0)
    S_TRANS   = _ps("trans",   9, "#333333", sb=0,  sa=2,  leading=13)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _esc(t):
        """Escape for reportlab XML AND strip any character SimHei cannot render.
        Whitelist approach: only keep characters in ranges SimHei is known to cover.
        Anything outside these ranges (emoji, dingbats, geometric shapes, etc.)
        is silently dropped so it never reaches the PDF renderer as a □ box.
        Known missing from SimHei (confirmed by cmap scan):
          U+2022 BULLET (•), U+00A0 NBSP, U+FF65 HW-MIDDLE-DOT, U+30FB KATAKANA-DOT
        """
        # Pre-replace known SimHei gaps with safe equivalents
        t = str(t)
        t = t.replace('\u2022', '\u00b7')   # • → · (middle dot, IS in SimHei)
        t = t.replace('\u00a0', ' ')         # NBSP → regular space
        safe = []
        for ch in t:
            cp = ord(ch)
            if (cp < 0x2500                    # ASCII + Latin + arrows + general punct
                or 0x3000 <= cp <= 0x9FFF      # CJK symbols + ideographs
                or 0x3400 <= cp <= 0x4DBF      # CJK Extension A
                or 0xAC00 <= cp <= 0xD7AF      # Korean
                or 0x3040 <= cp <= 0x30FF      # Japanese kana
                or 0xF900 <= cp <= 0xFAFF      # CJK Compatibility
                or 0xFF00 <= cp <= 0xFFEF):    # Halfwidth / Fullwidth
                safe.append(ch)
            # Everything else (U+2500+ geometric shapes, emoji, dingbats) → drop
        return _sax.escape(''.join(safe))

    def _md_para(text, style):
        parts = re.split(r'\*\*', text)
        html = ""
        for idx, part in enumerate(parts):
            e = _esc(part)
            html += f"<b>{e}</b>" if idx % 2 == 1 else e
        return Paragraph(html, style)

    def _build_table(table_lines):
        rows = _md_parse_table(table_lines)
        if not rows:
            return None
        max_cols = max(len(r) for r in rows)
        col_w = (A4[0] - 40 * mm) / max_cols
        tbl_data = []
        for ri, row in enumerate(rows):
            cells = [_md_para(row[ci] if ci < len(row) else "",
                              S_TCELL_H if ri == 0 else S_TCELL)
                     for ci in range(max_cols)]
            tbl_data.append(cells)
        tbl = Table(tbl_data, colWidths=[col_w] * max_cols, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("FONTNAME",       (0, 0), (-1, -1), _fn),
            ("FONTSIZE",       (0, 0), (-1, -1), 9),
            ("BACKGROUND",     (0, 0), (-1,  0), colors.HexColor("#DCE1FA")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F4FF")]),
            ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#A0A0C8")),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ("LEFTPADDING",    (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
        ]))
        return tbl

    def _render_md(text):
        flowables = []
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            line = _md_normalize(lines[i]).strip()
            if line.startswith("|"):
                tbl_lines = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    tbl_lines.append(lines[i])
                    i += 1
                tbl = _build_table(tbl_lines)
                if tbl:
                    flowables.append(tbl)
                    flowables.append(Spacer(1, 4))
                continue
            if not line:
                flowables.append(Spacer(1, 4))
            elif re.match(r"^[-*_]{3,}$", line):
                flowables.append(HRFlowable(width="100%", thickness=0.5,
                    color=colors.HexColor("#C8C8DC"), spaceBefore=4, spaceAfter=4))
            elif line.startswith("#### "):
                flowables.append(_md_para(line[5:], S_H4))
            elif line.startswith("### "):
                flowables.append(_md_para(line[4:], S_H3))
            elif line.startswith("## "):
                flowables.append(_md_para(line[3:], S_H2))
            elif line.startswith("# "):
                flowables.append(_md_para(line[2:], S_H1))
            elif line.startswith("- ") or line.startswith("* "):
                # U+00B7 (MIDDLE DOT ·) is confirmed in SimHei; U+2022 (•) is NOT
                flowables.append(_md_para("\u00b7  " + line[2:], S_BULLET))
            elif re.match(r"^\d+\. ", line):
                num  = re.match(r"^(\d+)\.", line).group(1)
                body = re.sub(r"^\d+\. ", "", line)
                flowables.append(_md_para(f"{num}.  {body}", S_BULLET))
            else:
                flowables.append(_md_para(line, S_BODY))
            i += 1
        return flowables

    def _hf(canvas_obj, doc):
        canvas_obj.saveState()
        canvas_obj.setFont(_fn, 8)
        canvas_obj.setFillColor(colors.HexColor("#999999"))
        canvas_obj.drawRightString(A4[0] - 20*mm, A4[1] - 12*mm,
            f"Video Analysis Report  \u2014  {video_name[:60]}")
        canvas_obj.drawCentredString(A4[0] / 2, 10*mm, f"Page {doc.page}")
        canvas_obj.restoreState()

    # ── Build document ────────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=22*mm, bottomMargin=20*mm)

    story = [
        Paragraph("Video Analysis Report", S_TITLE),
        Paragraph(_esc(video_name), S_SUB),
        Paragraph(datetime.now().strftime("%d %B %Y, %H:%M"), S_SUB),
        Spacer(1, 6),
        HRFlowable(width="100%", thickness=1.5,
                   color=colors.HexColor("#6366F1"), spaceBefore=0, spaceAfter=10),
    ]
    story.extend(_render_md(explanation))
    story += [
        PageBreak(),
        Paragraph("Full Transcript", S_H2), Spacer(1, 4),
        Paragraph(_esc(full_transcript).replace("\n", "<br/>"), S_TRANS),
        PageBreak(),
        Paragraph("Timestamped Transcript", S_H2), Spacer(1, 4),
        Paragraph(_esc(timestamped_transcript).replace("\n", "<br/>"), S_TRANS),
    ]
    doc.build(story, onFirstPage=_hf, onLaterPages=_hf)
    return buf.getvalue()


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def extract_slides_as_b64(uploaded_file) -> list[str]:
    """
    Convert an uploaded slide/image file into a list of base64-encoded JPEG strings.

    Supported input types:
      - PDF  (.pdf)  — each page rendered at 150 DPI → JPEG
      - PPTX (.pptx) — each slide's embedded images extracted; text overlaid on a
                       white canvas so even text-only slides produce a usable image
      - Images (.png / .jpg / .jpeg / .webp) — single image, returned as-is

    Returns a list of b64 strings (one per slide / page).
    """
    name = uploaded_file.name.lower()
    raw  = uploaded_file.read()
    results: list[str] = []

    # ── PDF ──────────────────────────────────────────────────────────────────
    if name.endswith(".pdf"):
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=raw, filetype="pdf")
            for page in doc:
                mat  = fitz.Matrix(96 / 72, 96 / 72)    # 96 DPI — good readability, ~3× smaller than 150 DPI
                pix  = page.get_pixmap(matrix=mat)
                img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                buf  = io.BytesIO()
                img.save(buf, format="JPEG", quality=75)
                results.append(base64.standard_b64encode(buf.getvalue()).decode())
            doc.close()
        except ImportError:
            st.warning("⚠️ PyMuPDF not installed — PDF slides cannot be processed. "
                       "Run: `pip install pymupdf`")
        except Exception as e:
            st.warning(f"⚠️ Could not read PDF slides: {e}")

    # ── PPTX ─────────────────────────────────────────────────────────────────
    elif name.endswith(".pptx"):
        try:
            from pptx import Presentation
            from pptx.util import Inches
            prs = Presentation(io.BytesIO(raw))

            for slide_idx, slide in enumerate(prs.slides):
                # Canvas matching the presentation's aspect ratio (max 1280 wide)
                slide_w = prs.slide_width.inches
                slide_h = prs.slide_height.inches
                scale   = 1280 / (slide_w * 96)        # 96 px per inch baseline
                canvas_w = int(slide_w * 96 * scale)
                canvas_h = int(slide_h * 96 * scale)
                canvas   = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

                # Extract embedded images from shapes
                has_image = False
                for shape in slide.shapes:
                    if shape.shape_type == 13:           # MSO_SHAPE_TYPE.PICTURE
                        try:
                            img_data = shape.image.blob
                            img      = Image.open(io.BytesIO(img_data)).convert("RGB")
                            # Position on canvas proportionally
                            left = int(shape.left / prs.slide_width  * canvas_w)
                            top  = int(shape.top  / prs.slide_height * canvas_h)
                            w    = int(shape.width  / prs.slide_width  * canvas_w)
                            h    = int(shape.height / prs.slide_height * canvas_h)
                            img  = img.resize((max(w, 1), max(h, 1)), Image.LANCZOS)
                            canvas.paste(img, (left, top))
                            has_image = True
                        except Exception:
                            pass

                # Overlay slide text so text-only slides are still useful
                texts = []
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for para in shape.text_frame.paragraphs:
                            line = para.text.strip()
                            if line:
                                texts.append(line)
                if texts:
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(canvas)
                    try:
                        font = ImageFont.truetype("arial.ttf", size=max(18, canvas_h // 30))
                    except Exception:
                        font = ImageFont.load_default()
                    y = 20
                    for line in texts:
                        draw.text((20, y), line, fill=(20, 20, 20), font=font)
                        y += max(24, canvas_h // 25)
                        if y > canvas_h - 40:
                            break

                buf = io.BytesIO()
                canvas.save(buf, format="JPEG", quality=85)
                results.append(base64.standard_b64encode(buf.getvalue()).decode())

        except ImportError:
            st.warning("⚠️ python-pptx not installed — PPTX slides cannot be processed. "
                       "Run: `pip install python-pptx`")
        except Exception as e:
            st.warning(f"⚠️ Could not read PPTX slides: {e}")

    # ── Images ───────────────────────────────────────────────────────────────
    elif any(name.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp")):
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            # Resize if needed
            max_dim = 1280
            if max(img.width, img.height) > max_dim:
                scale = max_dim / max(img.width, img.height)
                img = img.resize(
                    (int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            results.append(base64.standard_b64encode(buf.getvalue()).decode())
        except Exception as e:
            st.warning(f"⚠️ Could not read image: {e}")

    else:
        st.warning(f"⚠️ Unsupported slide format: `{uploaded_file.name}`. "
                   "Please upload PDF, PPTX, PNG, JPG, or WEBP.")

    return results


def _slide_phash(b64_str: str) -> np.ndarray:
    """8×8 mean-hash of a base64-encoded JPEG slide — reuses the same algorithm as video frames."""
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((8, 8), Image.LANCZOS)
    flat = np.array(img).flatten().astype(float)
    return flat > flat.mean()


# Safety budget per engine (bytes).  At 96 DPI / q75 each slide is ~45–70 KB,
# so these budgets comfortably hold 200–300 unique slides.
# Only slides that would push the payload OVER the budget are trimmed.
_SLIDE_BUDGET = {
    "Claude (Paid)": 16 * 1024 * 1024,   # 16 MB  (API hard limit ~20 MB)
    "OpenAI (Paid)": 16 * 1024 * 1024,   # 16 MB
    DEEPSEEK_ENGINE: 16 * 1024 * 1024,   # 16 MB (limit is 32 MiB/image, 600 images)
    "Gemini (Free)": 20 * 1024 * 1024,   # 20 MB  (Gemini is more lenient)
}


def _recompress_slides(slides: list[str], budget: int) -> tuple[list[str], str]:
    """
    Re-encode all slides at progressively lower JPEG quality AND smaller pixel
    dimensions until the total base64 payload fits within `budget` bytes.

    Passes (quality, max_long_side_px):
      1. q65 / 960 px  — mild quality drop, same size
      2. q55 / 800 px  — noticeable compression + slight resize
      3. q45 / 768 px  — more aggressive
      4. q35 / 640 px  — strong compression + resize
      5. q25 / 512 px  — last resort: still readable by AI vision models

    Returns (recompressed_list, description_string).
    """
    passes = [
        (65, 960),
        (55, 800),
        (45, 768),
        (35, 640),
        (25, 512),
    ]
    for quality, max_dim in passes:
        recompressed = []
        for b64 in slides:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            recompressed.append(base64.standard_b64encode(buf.getvalue()).decode())
        if sum(len(b) for b in recompressed) <= budget:
            return recompressed, f"q{quality}/{max_dim}px"
    return recompressed, f"q25/512px"   # best effort even if still slightly over


def process_slide_files(uploaded_files, ai_engine: str,
                        phash_threshold: float = 0.92) -> tuple[list[str], list[dict], str]:
    """
    Process multiple uploaded slide files into a flat list, with per-file dedup
    and size budget.  Returns:
      - flat list of b64 slides (in file order)
      - sections: [{"name": filename, "start": idx, "count": N}, ...]
        so callers can inject file-label separators at the right positions
      - human-readable info string

    Cross-file dedup: a slide already seen in file A is dropped if it reappears
    in file B (same template slide, logo page, etc.).
    """
    raw_sections   = []   # (name, [b64, ...]) before dedup
    all_accepted   = []   # shared pHash pool across files

    for sf in uploaded_files:
        raw_pages = extract_slides_as_b64(sf)
        unique    = []
        for b64 in raw_pages:
            ph = _slide_phash(b64)
            if not _is_duplicate(ph, all_accepted, threshold=phash_threshold):
                unique.append(b64)
                all_accepted.append(ph)
        raw_sections.append((sf.name, raw_pages, unique))

    total_original = sum(len(r) for _, r, _ in raw_sections)
    all_slides     = []
    for _, _, unique in raw_sections:
        all_slides.extend(unique)
    total_unique = len(all_slides)

    # Apply size budget (recompress → trim as last resort)
    budget      = _SLIDE_BUDGET.get(ai_engine, 16 * 1024 * 1024)
    total_bytes = sum(len(b) for b in all_slides)
    recomp_note = ""
    if total_bytes > budget:
        all_slides, recomp_desc = _recompress_slides(all_slides, budget)
        total_bytes = sum(len(b) for b in all_slides)
        recomp_note = recomp_desc
    if total_bytes > budget:
        avg  = total_bytes / len(all_slides)
        cap  = max(1, int(budget / avg))
        step = len(all_slides) / cap
        all_slides = [all_slides[int(i * step)] for i in range(cap)]

    # Rebuild per-file sections with correct start/count after any trimming
    sections: list[dict] = []
    if len(all_slides) == total_unique:
        # No trimming — exact counts
        idx = 0
        for name, _, unique in raw_sections:
            sections.append({"name": name, "start": idx, "count": len(unique)})
            idx += len(unique)
    else:
        # Trimming occurred — distribute proportionally
        factor = len(all_slides) / total_unique if total_unique else 0
        idx = 0
        for i, (name, _, unique) in enumerate(raw_sections):
            cnt = round(len(unique) * factor)
            if i == len(raw_sections) - 1:
                cnt = len(all_slides) - idx   # last file absorbs rounding remainder
            sections.append({"name": name, "start": idx, "count": max(0, cnt)})
            idx += max(0, cnt)

    # Build info string
    parts = []
    if total_unique < total_original:
        parts.append(f"{total_original - total_unique} duplicate(s) removed")
    if recomp_note:
        parts.append(f"recompressed to {recomp_note}")
    if len(all_slides) < total_unique:
        parts.append(f"trimmed to {len(all_slides)}")

    if len(uploaded_files) > 1:
        file_parts = " + ".join(
            f"**{s['name']}** ({s['count']} slides)" for s in sections if s["count"] > 0
        )
        info = f"{total_original} → **{len(all_slides)}** slides sent to AI: {file_parts}"
        if parts:
            info += f" ({', '.join(parts)})"
    else:
        info = (f"{total_original} → **{len(all_slides)}** slide(s) sent to AI"
                + (f" ({', '.join(parts)})" if parts else ""))

    return all_slides, sections, info


def cap_and_dedup_slides(slides: list[str], ai_engine: str,
                          phash_threshold: float = 0.92) -> tuple[list[str], str]:
    """
    1. pHash-dedup: remove near-identical slides (blank pages, repeated title
       frames, transition slides).
    2. If total payload exceeds the engine's size budget, re-compress all slides
       at progressively lower JPEG quality to fit — keeping every unique slide.
    3. Only trim (uniform subsample) as a last resort if even q35 is still over.

    Returns (filtered_slides, human_readable_info_string).
    """
    if not slides:
        return slides, ""

    original = len(slides)
    budget   = _SLIDE_BUDGET.get(ai_engine, 16 * 1024 * 1024)

    # Step 1: pHash deduplication
    unique: list[str] = []
    accepted: list[np.ndarray] = []
    for b64 in slides:
        ph = _slide_phash(b64)
        if not _is_duplicate(ph, accepted, threshold=phash_threshold):
            unique.append(b64)
            accepted.append(ph)

    # Step 2: Check payload size — recompress before considering any trimming
    total_bytes  = sum(len(b) for b in unique)
    recomp_note  = ""
    if total_bytes > budget:
        unique, recomp_desc = _recompress_slides(unique, budget)
        total_bytes         = sum(len(b) for b in unique)
        recomp_note         = f"recompressed to {recomp_desc}"

    # Step 3: Trim only if recompression still wasn't enough (very rare)
    if total_bytes > budget:
        avg_bytes  = total_bytes / len(unique)
        max_slides = max(1, int(budget / avg_bytes))
        step       = len(unique) / max_slides
        final      = [unique[int(i * step)] for i in range(max_slides)]
    else:
        final = unique

    # Build info string
    parts = []
    if len(unique) < original:
        parts.append(f"{original - len(unique)} duplicate(s) removed")
    if recomp_note:
        parts.append(recomp_note)
    if len(final) < len(unique):
        parts.append(f"trimmed to {len(final)} — payload still too large")
    info = (f"{original} → **{len(final)}** slide(s) sent to AI"
            + (f" ({', '.join(parts)})" if parts else ""))
    return final, info


ANALYSIS_PROMPT = """
---
**VIDEO TYPE:** {video_type}
**TOTAL DURATION:** {duration}
{slides_note}
**TIMESTAMPED TRANSCRIPT:**
(Format: [MM:SS] spoken text — use these timestamps to correlate with the frames above)

{transcript}

---
Using {sources_description} AND the timestamped transcript, produce a comprehensive explanation:

## 1. Overview
What is this video about? (2–3 sentences)

## 2. Key Concepts / Topics
A bullet list of the main ideas, skills, or subjects covered.

## 3. Detailed Breakdown
Walk through the content chronologically. For each section, reference the timestamp
(e.g. "At 05:30…") and describe both what is visible in the frame and what is being said.
Where slide content is visible, extract and reference the exact text, data, or visuals shown.
{verbosity_note}

## 4. Key Takeaways
The most important facts, instructions, insights, or action items a viewer should remember.

## 5. Summary
A concise closing paragraph wrapping up what was covered.

**FORMATTING STYLE (follow exactly):**
- Keep the five section headings exactly as `## 1. …` to `## 5. …` — plain and numbered, with no emoji in these five headings.
- In "Key Concepts / Topics": start every bullet with ONE fitting emoji, then a **bold term**, then a short explanation (for example `- 💰 **Cash flow:** why it matters`).
- In "Detailed Breakdown": give each timestamped part its own heading `### 📍 MM:SS — short title`, then three bold labels: what is **shown on screen**, what is **said**, and your **analysis** (write these labels in the output language). Put direct quotes from the speaker as `>` blockquotes.
- In "Key Takeaways": start each point with ✅ (things to do), ⚠️ (risks or warnings) or 💡 (insights).
- Use bold for key terms. Do not add emojis anywhere else.

Be thorough, use clear headings and bullet points. **Always complete ALL sections — do not stop mid-document.**
{output_language_instruction}
"""


# ── Two-pass prompts (used when slide count ≥ 80) ────────────────────────────
# Pass 1 covers the short sections so their tokens don't compete with the
# Detailed Breakdown.  Pass 2 gets the full 8 192-token budget for section 3.

ANALYSIS_PROMPT_PASS1 = """
---
**VIDEO TYPE:** {video_type}
**TOTAL DURATION:** {duration}
{slides_note}
**TIMESTAMPED TRANSCRIPT:**
(Format: [MM:SS] spoken text — use these timestamps to correlate with the frames above)

{transcript}

---
Using {sources_description} AND the timestamped transcript, write ONLY these two sections:

## 1. Overview
What is this video about? (2–3 sentences)

## 2. Key Concepts / Topics
A bullet list of the main ideas, skills, or subjects covered.

For the key-concept bullets: start every bullet with ONE fitting emoji, then a **bold term**, then a short explanation (for example `- 💰 **Cash flow:** why it matters`).
Keep the two section headings exactly as `## 1. …` and `## 2. …` (plain, numbered, no emoji in the headings).

Be concise but accurate. {output_language_instruction}
"""


ANALYSIS_PROMPT_PASS2 = """
---
**VIDEO TYPE:** {video_type}
**TOTAL DURATION:** {duration}
{slides_note}
**TIMESTAMPED TRANSCRIPT:**
(Format: [MM:SS] spoken text — use these timestamps to correlate with the frames above)

{transcript}

---
Using {sources_description} AND the timestamped transcript, write ONLY these three sections:

## 3. Detailed Breakdown
{verbosity_note}

## 4. Key Takeaways
The most important facts, instructions, insights, or action items a viewer should remember.

## 5. Summary
A concise closing paragraph wrapping up what was covered.

**FORMATTING STYLE (follow exactly):**
- Keep the three section headings exactly as `## 3. …`, `## 4. …`, `## 5. …` — plain and numbered, with no emoji in them.
- In "Detailed Breakdown": give each timestamped part its own heading `### 📍 MM:SS — short title`, then three bold labels: what is **shown on screen**, what is **said**, and your **analysis** (write these labels in the output language). Put direct quotes from the speaker as `>` blockquotes.
- In "Key Takeaways": start each point with ✅ (things to do), ⚠️ (risks or warnings) or 💡 (insights).
- Use bold for key terms. Do not add emojis anywhere else.

**CRITICAL: Complete ALL three sections from start to finish. Cover every topic in the lecture. Do NOT stop mid-document.**
{output_language_instruction}
"""


PKM_NOTE_PROMPT = """
You are a professional knowledge curator writing a detailed study note for a Personal Knowledge
Management (PKM) system such as Obsidian or Heptabase.

Using the transcript and analysis context below, produce a single, self-contained Markdown note
that a learner can save directly into their vault. Follow the structure EXACTLY as shown.

---

**VIDEO:** {video_name}
**TYPE:** {video_type}
**DURATION:** {duration}
**CHAPTERS:** {chapters}

**TRANSCRIPT:**
{transcript}

**EXISTING ANALYSIS (use as reference only — do not copy verbatim):**
{explanation}

---

Produce the PKM note now. Use the exact structure below — do NOT skip any section:

```markdown
---
title: {video_name}
tags: [video-note, {video_type_tag}]
date: {today}
source: video/{video_type}
duration: {duration}
---

## 🗺️ One-Line Summary
< A single sentence that captures the entire video's purpose. >

## 📌 Context & Why It Matters
< 2–3 sentences: what problem this video solves, who it is for, and why this knowledge is valuable. >

## 🧠 Key Concepts

| Concept | Plain-English Definition | Why It Matters |
|---------|--------------------------|----------------|
| Term 1  | ...                      | ...            |
| Term 2  | ...                      | ...            |
(add as many rows as needed — aim for 5–12 concepts)

## 📖 Concept Deep-Dives

For EACH concept in the table above, write a subsection:

### [[Concept Name]]
**What it is:** ...
**How it works:** ...
**Example from the video:**
> "exact quote or paraphrase from transcript" — [MM:SS]
**Common mistake / misconception:** ...

(repeat for every concept)

## 🗓️ Chronological Walkthrough
A structured timeline of the video — use a numbered list, one entry per chapter or major topic shift.
Each entry must include the timestamp, what was shown/said, and the key takeaway from that segment.

1. **[00:00] — Chapter title**
   - What happened: ...
   - Key insight: ...

(continue for all chapters)

## 📊 Comparisons & Relationships
If the video compares methods, tools, options, or steps — add a table here.
If nothing to compare, write a short paragraph on how the concepts relate to each other.

## ✅ Step-by-Step Process (if applicable)
If the video demonstrates a process or workflow, list every step clearly:

1. Step one — detail
2. Step two — detail
(skip this section with a note if the video is not process-based)

## 💡 Key Quotes & Moments
List 3–6 direct quotes or paraphrases from the transcript that are especially insightful or memorable.

> "Quote here" — [MM:SS]
> "Another quote" — [MM:SS]

## 🎯 Takeaways & Action Items
- [ ] Action item 1
- [ ] Action item 2
(concrete, actionable things the viewer should do or remember)

## 🔗 Related Topics
List 5–10 related concepts or topics as wikilinks that the reader might want to explore:
[[Topic A]] · [[Topic B]] · [[Topic C]]

## ❓ Questions This Raises
List 3–5 open questions or areas to explore further after watching this video.

1. ...
2. ...
```

Rules:
- Use rich Markdown: tables, blockquotes, checkboxes, wikilinks, bold, code blocks where relevant.
- Every concept deep-dive MUST include a direct quote from the transcript.
- Every chronological entry MUST include a timestamp.
- Do NOT write generic filler — every sentence must contain specific information from THIS video.
- Output ONLY the markdown content inside the code block above (no preamble, no explanation).
- {output_language_instruction}
"""


def generate_pkm_note(video_name: str, video_type: str, duration: float,
                       explanation: str, timestamped_transcript: str,
                       chapters: list, ai_engine: str,
                       claude_key: str, gemini_key: str,
                       output_language: str = "English",
                       openai_key: str = "") -> str:
    """
    Generate a rich PKM-ready Markdown note (Obsidian / Heptabase compatible).
    Returns the raw markdown string.
    """
    chapters_text = chapters_to_text(chapters) if chapters else "No chapters available."
    video_type_tag = video_type.lower().replace(" ", "-")
    today = datetime.now().strftime("%Y-%m-%d")
    lang_instruction = OUTPUT_LANGUAGES.get(output_language, OUTPUT_LANGUAGES["English"])

    prompt = PKM_NOTE_PROMPT.format(
        video_name=video_name,
        video_type=video_type,
        video_type_tag=video_type_tag,
        duration=fmt_time(duration),
        today=today,
        chapters=chapters_text,
        transcript=timestamped_transcript[:30000],  # ~7 500 tokens — enough for 2–3 hr videos
        explanation=explanation[:3000],
        output_language_instruction=lang_instruction,
    )

    if ai_engine == "Claude (Paid)":
        import anthropic
        client = anthropic.Anthropic(api_key=claude_key)
        response = client.messages.create(
            model="claude-haiku-4-5",   # Haiku: same quality for structured notes, ~75% cheaper
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
    elif _is_oai(ai_engine):
        from openai import OpenAI as _OAI
        _oc = _oai_client(ai_engine, openai_key)
        _or = _oc.chat.completions.create(
            model=_oai_model(ai_engine, "small"), **_oai_extra(ai_engine),
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _or.choices[0].message.content.strip()
    else:
        from google import genai as google_genai
        client = google_genai.Client(api_key=gemini_key)
        for attempt in range(3):
            try:
                response = _gemini_generate(client, prompt)
                raw = response.text.strip()
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(30 * (attempt + 1))
                else:
                    raise

    # Strip wrapping ```markdown ... ``` fence if the model added one
    if raw.startswith("```markdown"):
        raw = raw[len("```markdown"):].lstrip("\n")
    if raw.endswith("```"):
        raw = raw[:-3].rstrip("\n")

    return raw


# ── Tag colour palette ────────────────────────────────────────────────────────
_TAG_PALETTE = [
    "#6366F1", "#EC4899", "#F59E0B", "#10B981", "#3B82F6",
    "#8B5CF6", "#EF4444", "#14B8A6", "#F97316", "#84CC16",
]

def _tag_colour(tag: str) -> str:
    """Return a consistent hex colour for a given tag string."""
    return _TAG_PALETTE[abs(hash(tag)) % len(_TAG_PALETTE)]


def _ai_short_reply(prompt: str, ai_engine: str, claude_key: str = "", gemini_key: str = "",
                    openai_key: str = "", max_tokens: int = 60) -> str:
    """One short text answer from whichever AI engine is selected ('' on any failure).
    Accepts the full engine names used by the app ("Claude (Paid)", "Gemini (Free)", DeepSeek, OpenAI)."""
    try:
        name = str(ai_engine)
        if name.startswith("Claude") and claude_key:
            import anthropic
            client = anthropic.Anthropic(api_key=claude_key)
            resp = client.messages.create(
                model="claude-haiku-4-5", max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        if name.startswith("Gemini") and gemini_key:
            from google import genai as google_genai
            client = google_genai.Client(api_key=gemini_key)
            return _gemini_generate(client, prompt).text.strip()
        if _is_oai(name) and openai_key:
            client = _oai_client(name, openai_key)
            resp = client.chat.completions.create(
                model=_oai_model(name, "small"), max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}], **_oai_extra(name),
            )
            return resp.choices[0].message.content.strip()
    except Exception:
        pass
    return ""


def generate_tags(video_name: str, explanation: str, ai_engine: str, claude_key: str = "",
                  gemini_key: str = "", openai_key: str = "") -> list[str]:
    """
    Ask the AI to suggest 2-4 short topic tags for the video.
    Returns a list of lowercase strings, e.g. ['memory', 'training', 'chinese'].
    Returns [] on any failure.  (Used to silently return [] for every engine because the engine
    name was compared against "Claude"/"Gemini" instead of the real "Claude (Paid)" etc.)
    """
    prompt = (
        "Based on the video title and explanation excerpt, suggest 2 to 4 short topic tags "
        "(1-2 words each) that best categorise this video.\n\n"
        "Rules:\n"
        "- Return ONLY the tags as a comma-separated list — no numbers, no bullets, no explanation\n"
        "- Write the tags in English, lowercase\n"
        "- Be specific and meaningful (avoid generic tags like 'video' or 'content')\n"
        "- Good examples: training, property, memory, marketing, language learning, "
        "productivity, tutorial, sales, investing, real estate\n\n"
        f"VIDEO TITLE: {video_name}\n\n"
        f"EXPLANATION (excerpt):\n{explanation[:1500]}"
    )
    raw = _ai_short_reply(prompt, ai_engine, claude_key, gemini_key, openai_key, max_tokens=60)
    if not raw:
        return []
    tags = [t.strip().lower().strip('"').strip("'").strip(".") for t in re.split(r"[,\n]", raw)]
    return [t for t in tags if t and len(t) <= 30][:4]


def generate_video_title(explanation: str, full_transcript: str, ai_engine: str,
                         claude_key: str = "", gemini_key: str = "", openai_key: str = "") -> str:
    """
    Ask the AI to suggest a concise, descriptive title for the video based on
    its explanation and transcript.  Returns a plain string (no quotes, no
    punctuation at end).  Returns "" on any failure.
    """
    prompt = (
        "Based on the video explanation and transcript excerpt below, suggest a "
        "concise, descriptive title (5–8 words) that captures the specific topic.\n\n"
        "Rules:\n"
        "- Return ONLY the title — no quotes, no trailing punctuation, no explanation\n"
        "- Be specific (avoid generic phrases like 'Video Summary' or 'Tutorial Overview')\n"
        "- Title-case the result\n\n"
        f"EXPLANATION (excerpt):\n{explanation[:1500]}\n\n"
        f"TRANSCRIPT (first 500 words):\n{' '.join(full_transcript.split()[:500])}"
    )
    return _ai_short_reply(prompt, ai_engine, claude_key, gemini_key, openai_key,
                           max_tokens=40).strip('"').strip("'")


def auto_tag_history_entries(entries: list, default_folder: str, ai_engine: str, claude_key: str,
                             gemini_key: str, openai_key: str, progress=None) -> tuple:
    """AI-tag saved videos from their saved explanation. New tags are merged with existing ones (max 6).
    Returns (videos_tagged, videos_failed)."""
    tagged = failed = 0
    for i, e in enumerate(entries):
        folder = e.get("save_folder", default_folder)
        expl_name = next((f for f in e.get("files", []) if f.endswith("_explanation.md")), None)
        text = ""
        if expl_name and os.path.exists(os.path.join(folder, expl_name)):
            with open(os.path.join(folder, expl_name), "r", encoding="utf-8") as fh:
                text = fh.read()
        tags = generate_tags(e["video_name"], text or e["video_name"], ai_engine,
                             claude_key, gemini_key, openai_key) if (text or e.get("video_name")) else []
        if tags:
            try:
                hp = _history_path(folder)
                with open(hp, "r", encoding="utf-8") as fh:
                    recs = json.load(fh)
                for r in recs:
                    if r.get("stamp") == e.get("stamp"):
                        r["tags"] = list(dict.fromkeys(r.get("tags", []) + tags))[:6]
                        break
                with open(hp, "w", encoding="utf-8") as fh:
                    json.dump(recs, fh, ensure_ascii=False, indent=2)
                tagged += 1
            except Exception:
                failed += 1
        else:
            failed += 1
        if progress:
            progress((i + 1) / len(entries), e["video_name"])
    return tagged, failed


def _verbosity_note(n_slides: int, n_frames: int) -> str:
    """Return a prompt instruction that scales detail level to the content volume."""
    total = n_slides + n_frames
    if n_slides >= 80:
        return (
            f"\n⚠️ **Large deck ({n_slides} slides):** Write the Detailed Breakdown by "
            f"**topic/concept** — NOT by slide number. Use meaningful topic headers "
            f"(e.g. '### Cost Approach', '### Market Value Definitions'). "
            f"Do NOT write 'Slide X:' anywhere. Group related content together and "
            f"give 2–4 bullet points per topic. Cover the full lecture from start to finish."
        )
    if n_slides >= 30 or total >= 50:
        return (
            f"\n📌 **{n_slides} slides provided:** Group content by topic, not by slide number. "
            f"Do NOT write 'Slide X:' — use topic-based headers only. 2–3 bullet points per topic."
        )
    return ""


def _claude_retry(client, max_retries: int = 4, **kwargs):
    """Call client.messages.create with automatic rate-limit (429) retry.

    Backs off 65 s, 130 s, 195 s, 260 s between attempts so each retry
    falls in a fresh 60-second TPM window.
    """
    import anthropic as _anth
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return client.messages.create(**kwargs)
        except _anth.RateLimitError as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = 65 * (attempt + 1)
                time.sleep(wait)
            else:
                raise
        except Exception:
            raise
    raise last_exc  # unreachable but satisfies type checkers


def analyze_with_claude(timestamped_transcript, frames, timestamps, duration, api_key, video_type,
                        output_language: str = "English",
                        slide_images: list[str] | None = None,
                        slide_sections: list[dict] | None = None):
    """Send frames + optional slides + timestamped transcript to Claude.

    Large decks (≥ 80 slides) use a chunked two-pass approach to stay within
    Claude's 30 000 input-token-per-minute rate limit:
      • Pass 1  — transcript only → Overview + Key Concepts  (~3 k tokens)
      • Pass 2+ — slides in batches of 25 → Detailed Breakdown chunks
                  (≈ 18 k tokens each, 65 s gap between batches)
    Results are stitched together into a single Markdown document.
    """
    import anthropic
    import math as _math
    client = anthropic.Anthropic(api_key=api_key)
    lang_instruction = OUTPUT_LANGUAGES.get(output_language, OUTPUT_LANGUAGES["English"])

    _n_slides = len(slide_images) if slide_images else 0
    _n_frames = len(frames) if frames else 0
    _verbosity = _verbosity_note(_n_slides, _n_frames)

    slides_note = (
        f"\n**UPLOADED SLIDES:** {_n_slides} slide(s) — treat as primary visual reference.\n"
        if slide_images else ""
    )
    sources_description = (
        "the uploaded slides, the video frames," if slide_images
        else "the video frames above"
    )

    # Build a label-lookup for multi-file decks (used in all paths below)
    _label_at: dict[int, str] = {}
    if slide_sections and len(slide_sections) > 1:
        for sec in slide_sections:
            if sec.get("count", 0) > 0:
                _label_at[sec["start"]] = sec["name"]

    # ── Helper: build an image content block for a slice of slides ───────────
    def _slide_content(start: int, end: int) -> list[dict]:
        blk = [{"type": "text",
                "text": f"## SLIDES {start + 1}–{end} of {_n_slides}"}]
        for i, b64 in enumerate(slide_images[start:end]):
            idx = start + i
            if idx in _label_at:
                blk.append({"type": "text",
                             "text": f"### 📄 Document: {_label_at[idx]}"})
            blk.append({"type": "image",
                        "source": {"type": "base64",
                                   "media_type": "image/jpeg", "data": b64}})
        return blk

    # ── Helper: extract the transcript slice relevant to a slide chunk ────────
    def _ts_slice(chunk_start: int, chunk_end: int,
                  max_chars: int = 20_000) -> str:
        ts_len = len(timestamped_transcript)
        t0 = int((chunk_start / max(_n_slides, 1)) * ts_len)
        t1 = min(ts_len, t0 + max_chars)
        slc = timestamped_transcript[t0:t1]
        return slc + ("\n…[transcript continues]" if t1 < ts_len else "")

    # ══════════════════════════════════════════════════════════════════════════
    # LARGE-DECK PATH  (≥ 80 slides)
    # Each chunk stays under ~25 k input tokens to respect the 30 k TPM limit.
    # ══════════════════════════════════════════════════════════════════════════
    if _n_slides >= 80:
        _CHUNK = 25   # slides per batch → ~16 k tokens of images

        # ── Pass 1: transcript only → sections 1 & 2 ─────────────────────────
        _p1_prompt = ANALYSIS_PROMPT_PASS1.format(
            video_type=video_type, duration=fmt_time(duration),
            transcript=timestamped_transcript[:30_000],   # cap at ~7 500 tokens
            slides_note=slides_note,
            sources_description=sources_description,
            output_language_instruction=lang_instruction,
        )
        r1 = _claude_retry(client,
                           model="claude-sonnet-4-6", max_tokens=2000,
                           messages=[{"role": "user",
                                      "content": [{"type": "text",
                                                   "text": _p1_prompt}]}])
        pass1_text = r1.content[0].text.strip()

        # ── Pass 2+: slides in chunks → section 3 (+ 4 & 5 on final chunk) ───
        _n_chunks = _math.ceil(_n_slides / _CHUNK)
        _breakdown_parts: list[str] = []

        for _ci in range(_n_chunks):
            _s, _e = _ci * _CHUNK, min((_ci + 1) * _CHUNK, _n_slides)
            _is_last = (_ci == _n_chunks - 1)

            # Pause between chunks so each lands in a fresh 60 s TPM window
            if _ci > 0:
                time.sleep(65)

            _chunk_content = _slide_content(_s, _e)

            _chunk_prompt = (
                f"---\n"
                f"**VIDEO TYPE:** {video_type}\n"
                f"**TOTAL DURATION:** {fmt_time(duration)}\n"
                f"**BATCH:** slides {_s + 1}–{_e} of {_n_slides} total\n"
                f"{slides_note}\n"
                f"**RELEVANT TRANSCRIPT:**\n"
                f"{_ts_slice(_s, _e)}\n\n"
                f"---\n"
                f"Write the Detailed Breakdown for ONLY the slides shown in this batch.\n"
                f"{_verbosity}\n"
                + (
                    "Also write **## 4. Key Takeaways** and **## 5. Summary** "
                    "covering the full lecture (use the transcript above).\n"
                    if _is_last else
                    "Write ONLY the Detailed Breakdown for these slides. "
                    "Do NOT write Key Takeaways or Summary yet.\n"
                )
                + f"{lang_instruction}"
            )
            _chunk_content.append({"type": "text", "text": _chunk_prompt})

            r_chunk = _claude_retry(
                client,
                model="claude-sonnet-4-6",
                max_tokens=8192 if _is_last else 4096,
                messages=[{"role": "user", "content": _chunk_content}],
            )
            _breakdown_parts.append(r_chunk.content[0].text.strip())

        return (pass1_text
                + "\n\n## 3. Detailed Breakdown\n\n"
                + "\n\n".join(_breakdown_parts))

    # ══════════════════════════════════════════════════════════════════════════
    # STANDARD PATH  (< 80 slides — single pass, all content in one call)
    # ══════════════════════════════════════════════════════════════════════════
    content: list[dict] = []

    if slide_images:
        content.append({"type": "text",
                        "text": f"## UPLOADED SLIDES ({_n_slides} slide(s))\n"
                                "These are the actual presentation slides used in the video."})
        for i, b64 in enumerate(slide_images):
            if i in _label_at:
                content.append({"type": "text",
                                 "text": f"### 📄 Document: {_label_at[i]}"})
            content.append({"type": "image",
                            "source": {"type": "base64",
                                       "media_type": "image/jpeg", "data": b64}})
        content.append({"type": "text", "text": "---"})

    if frames:
        content.append({"type": "text",
                        "text": f"## VIDEO FRAMES ({_n_frames} frame(s))\n"
                                "Timestamped snapshots captured during the video."})
        for b64, ts in zip(frames, timestamps):
            content.append({"type": "text", "text": f"**Frame @ {fmt_time(ts)}**"})
            content.append({"type": "image",
                            "source": {"type": "base64",
                                       "media_type": "image/jpeg", "data": b64}})

    content.append({"type": "text", "text": ANALYSIS_PROMPT.format(
        video_type=video_type, duration=fmt_time(duration),
        transcript=timestamped_transcript,
        slides_note=slides_note,
        sources_description=sources_description,
        verbosity_note=_verbosity,
        output_language_instruction=lang_instruction)})

    response = _claude_retry(client,
                             model="claude-sonnet-4-6", max_tokens=8192,
                             messages=[{"role": "user", "content": content}])
    return response.content[0].text


def analyze_with_gemini(timestamped_transcript, frames, timestamps, duration, api_key, video_type,
                        output_language: str = "English",
                        slide_images: list[str] | None = None,
                        slide_sections: list[dict] | None = None):
    """Send frames + optional slides + timestamped transcript to Gemini."""
    from google import genai as google_genai
    from google.genai import types
    import time

    client = google_genai.Client(api_key=api_key)
    lang_instruction = OUTPUT_LANGUAGES.get(output_language, OUTPUT_LANGUAGES["English"])

    parts = []

    # ── Uploaded slides first ─────────────────────────────────────────────────
    if slide_images:
        parts.append(types.Part.from_text(
            text=f"## UPLOADED SLIDES ({len(slide_images)} slide(s))\n"
                 "These are the actual presentation slides used in the video. "
                 "Use them as the ground truth for any on-screen content."))
        _label_at = {}
        if slide_sections and len(slide_sections) > 1:
            for sec in slide_sections:
                if sec["count"] > 0:
                    _label_at[sec["start"]] = sec["name"]
        for i, b64 in enumerate(slide_images):
            if i in _label_at:
                parts.append(types.Part.from_text(text=f"### 📄 Document: {_label_at[i]}"))
            parts.append(types.Part.from_bytes(
                data=base64.b64decode(b64), mime_type="image/jpeg"))
        parts.append(types.Part.from_text(text="---"))

    # ── Video frames ──────────────────────────────────────────────────────────
    if frames:
        parts.append(types.Part.from_text(
            text=f"## VIDEO FRAMES ({len(frames)} frame(s))\n"
                 "Timestamped snapshots captured during the video."))
        for b64, ts in zip(frames, timestamps):
            parts.append(types.Part.from_text(text=f"**Frame @ {fmt_time(ts)}**"))
            parts.append(types.Part.from_bytes(
                data=base64.b64decode(b64), mime_type="image/jpeg"))

    # ── Prompt ────────────────────────────────────────────────────────────────
    slides_note = (
        f"\n**UPLOADED SLIDES:** {len(slide_images)} slide(s) provided above "
        f"— treat these as the primary visual reference.\n"
        if slide_images else ""
    )
    sources_description = (
        "the uploaded slides, the video frames," if slide_images
        else "the video frames above"
    )

    parts.append(types.Part.from_text(text=ANALYSIS_PROMPT.format(
        video_type=video_type, duration=fmt_time(duration),
        transcript=timestamped_transcript,
        slides_note=slides_note,
        sources_description=sources_description,
        verbosity_note=_verbosity_note(len(slide_images) if slide_images else 0, len(frames) if frames else 0),
        output_language_instruction=lang_instruction)))

    # Retry up to 3 times if rate limited
    for attempt in range(3):
        try:
            response = _gemini_generate(client, parts)
            return response.text
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 30 * (attempt + 1)
                st.warning(f"⏳ Gemini rate limit hit — waiting {wait}s before retry ({attempt+1}/3)…")
                time.sleep(wait)
            else:
                raise


def analyze_with_gemini_video(video_path: str, timestamped_transcript, duration, api_key, video_type,
                              output_language: str = "English",
                              slide_images: list[str] | None = None,
                              slide_sections: list[dict] | None = None,
                              youtube_url: str | None = None):
    """Let Gemini WATCH the video itself (it samples ~1 frame/s and hears the audio).
    A public YouTube link is passed straight to Gemini; anything else is uploaded to Google's
    Files API for the duration of the request and deleted afterwards."""
    from google import genai as google_genai
    from google.genai import types
    import time

    client = google_genai.Client(api_key=api_key)
    lang_instruction = OUTPUT_LANGUAGES.get(output_language, OUTPUT_LANGUAGES["English"])
    uploaded = None
    try:
        parts = []
        if youtube_url:
            parts.append(types.Part(file_data=types.FileData(file_uri=youtube_url)))
        else:
            uploaded = client.files.upload(file=video_path)
            for _ in range(120):                      # wait up to ~10 min for Google to process it
                state = getattr(getattr(uploaded, "state", None), "name", "ACTIVE")
                if state == "ACTIVE":
                    break
                if state == "FAILED":
                    raise RuntimeError("Google could not process the uploaded video.")
                time.sleep(5)
                uploaded = client.files.get(name=uploaded.name)
            parts.append(types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type))

        if slide_images:
            parts.append(types.Part.from_text(
                text=f"## UPLOADED SLIDES ({len(slide_images)} slide(s))\n"
                     "These are the actual presentation slides used in the video. "
                     "Use them as the ground truth for any on-screen content."))
            _label_at = {}
            if slide_sections and len(slide_sections) > 1:
                for sec in slide_sections:
                    if sec["count"] > 0:
                        _label_at[sec["start"]] = sec["name"]
            for i, b64 in enumerate(slide_images):
                if i in _label_at:
                    parts.append(types.Part.from_text(text=f"### 📄 Document: {_label_at[i]}"))
                parts.append(types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/jpeg"))
            parts.append(types.Part.from_text(text="---"))

        slides_note = (
            f"\n**UPLOADED SLIDES:** {len(slide_images)} slide(s) provided above "
            f"— treat these as the primary visual reference.\n" if slide_images else "")
        sources_description = (
            "the video itself (which you can see AND hear)" + (", the uploaded slides," if slide_images else ""))
        parts.append(types.Part.from_text(text=ANALYSIS_PROMPT.format(
            video_type=video_type, duration=fmt_time(duration),
            transcript=timestamped_transcript,
            slides_note=slides_note,
            sources_description=sources_description,
            verbosity_note=_verbosity_note(len(slide_images) if slide_images else 0, 0),
            output_language_instruction=lang_instruction)))

        for attempt in range(3):
            try:
                return _gemini_generate(client, parts).text
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 30 * (attempt + 1)
                    st.warning(f"⏳ Gemini rate limit hit — waiting {wait}s before retry ({attempt+1}/3)…")
                    time.sleep(wait)
                else:
                    raise
    finally:
        if uploaded is not None:                       # never leave the video on Google's servers
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def analyze_with_openai(timestamped_transcript, frames, timestamps, duration, api_key, video_type,
                        output_language: str = "English",
                        slide_images: list[str] | None = None,
                        slide_sections: list[dict] | None = None,
                        ai_engine: str = "OpenAI (Paid)"):
    """Send frames + optional slides + timestamped transcript to GPT-4o (or DeepSeek Flash)."""
    client = _oai_client(ai_engine, api_key)
    lang_instruction = OUTPUT_LANGUAGES.get(output_language, OUTPUT_LANGUAGES["English"])

    content = []

    # ── Uploaded slides first ────────────────────────────────────────────────
    if slide_images:
        content.append({"type": "text",
                        "text": f"## UPLOADED SLIDES ({len(slide_images)} slide(s))\n"
                                "These are the actual presentation slides used in the video. "
                                "Use them as the ground truth for any on-screen content."})
        _label_at = {}
        if slide_sections and len(slide_sections) > 1:
            for sec in slide_sections:
                if sec["count"] > 0:
                    _label_at[sec["start"]] = sec["name"]
        for i, b64 in enumerate(slide_images):
            if i in _label_at:
                content.append({"type": "text", "text": f"### 📄 Document: {_label_at[i]}"})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}})
        content.append({"type": "text", "text": "---"})

    # ── Video frames ─────────────────────────────────────────────────────────
    if frames:
        content.append({"type": "text",
                        "text": f"## VIDEO FRAMES ({len(frames)} frame(s))\n"
                                "Timestamped snapshots captured during the video."})
        for b64, ts in zip(frames, timestamps):
            content.append({"type": "text", "text": f"**Frame @ {fmt_time(ts)}**"})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}})
        content.append({"type": "text", "text": "---"})

    slides_note = (
        f"\n**UPLOADED SLIDES:** {len(slide_images)} slide(s) provided above "
        f"— treat these as the primary visual reference.\n"
        if slide_images else ""
    )
    sources_description = (
        "the uploaded slides, the video frames," if slide_images
        else "the video frames above"
    )

    content.append({"type": "text", "text": ANALYSIS_PROMPT.format(
        video_type=video_type, duration=fmt_time(duration),
        transcript=timestamped_transcript,
        slides_note=slides_note,
        sources_description=sources_description,
        verbosity_note=_verbosity_note(len(slide_images) if slide_images else 0, len(frames) if frames else 0),
        output_language_instruction=lang_instruction)})

    response = client.chat.completions.create(
        model=_oai_model(ai_engine, "main"),
        **_oai_extra(ai_engine),
        max_tokens=16384,   # GPT-4o supports up to 16 384 output tokens
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content.strip()


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="VidSage — AI Video Explainer", page_icon="🎬", layout="wide")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}

/* ── Hero ── */
.vs-hero {
    text-align: center;
    padding: 2rem 0 1rem;
}
.vs-title {
    font-size: 5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 60%, #A855F7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.vs-tagline {
    color: #64748B;
    font-size: 1.1rem;
    margin: 0.5rem 0 1rem;
}
.vs-pill {
    display: inline-block;
    background: #EEF2FF;
    color: #4338CA;
    border: 1px solid #C7D2FE;
    padding: 4px 13px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 600;
    margin: 3px 2px;
}
.vs-divider { margin: 1.5rem 0 0; border-top: 1px solid #E2E8F0; }

/* ── Empty-state card ── */
.vs-empty {
    background: linear-gradient(135deg, #F8FAFF 0%, #EEF2FF 100%);
    border: 2px dashed #C7D2FE;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    margin: 1rem 0;
}
.vs-empty h3 { color: #4338CA; margin-bottom: 0.4rem; font-size: 1.3rem; }
.vs-empty p  { color: #64748B; font-size: 0.95rem; margin: 0; }

/* ── Metrics row ── */
[data-testid="metric-container"] {
    background: #F8FAFF;
    border: 1px solid #E0E7FF;
    border-radius: 12px;
    padding: 16px !important;
}
[data-testid="metric-container"] label {
    color: #6366F1 !important;
    font-weight: 600 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #F1F5FB;
    border-radius: 10px;
    padding: 4px;
    border-bottom: none;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    padding: 7px 14px;
    font-size: 0.85rem;
    font-weight: 500;
    color: #475569;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #4338CA !important;
    box-shadow: 0 1px 6px rgba(99,102,241,.15);
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #7C3AED 100%);
    border: none;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.01em;
    transition: opacity .15s;
}
.stButton > button[kind="primary"]:hover { opacity: .88; }

/* ── Sidebar (dark, high-contrast: every text colour is ≥ 7:1 against its background) ── */
section[data-testid="stSidebar"] {
    background: #14103A;
    border-right: 1px solid #2B2766;
}
section[data-testid="stSidebar"] :is(h1, h2, h3, h4) { color: #FFFFFF !important; }
section[data-testid="stSidebar"] :is(p, span, label, li, small) { color: #EEF1FF !important; }
section[data-testid="stSidebar"] hr { border-color: #2B2766 !important; }
/* captions / helper text: softer, but still clearly readable */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: #B9C2F7 !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg { fill: #B9C2F7 !important; color: #B9C2F7 !important; }

/* ── Cards (the collapsed option groups) ── */
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #1E1952;
    border: 1px solid #3A3487 !important;
    border-radius: 12px;
    margin: 8px 0 2px;
}
section[data-testid="stSidebar"] [data-testid="stExpander"]:hover { border-color: #6366F1 !important; }
section[data-testid="stSidebar"] [data-testid="stExpander"] summary { padding: 10px 12px; }
section[data-testid="stSidebar"] [data-testid="stExpander"] summary * {
    color: #FFFFFF !important;
    font-weight: 600;
    font-size: 0.92rem;
}
/* the current choice stands out in the accent colour */
section[data-testid="stSidebar"] [data-testid="stExpander"] summary strong { color: #A5B4FC !important; }
section[data-testid="stSidebar"] [data-testid="stExpander"] summary svg { fill: #A5B4FC !important; color: #A5B4FC !important; }
section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding: 4px 12px 12px; }

/* ── Radio options: bigger text, selected one highlighted ── */
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 6px 8px;
    border-radius: 8px;
    margin: 2px 0;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: rgba(99, 102, 241, 0.30);
    outline: 1px solid #818CF8;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label p { font-size: 0.93rem; }

/* ── Inputs ── */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    background: #262065 !important;
    color: #FFFFFF !important;
    border: 1px solid #5B55C9 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder { color: #9AA5E8 !important; opacity: 1 !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] { border-left: none !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #262065 !important;
    border: 1px solid #5B55C9 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] * { color: #FFFFFF !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #B9C2F7 !important; }
section[data-testid="stSidebar"] [data-baseweb="tag"] { background: #4F46E5 !important; }

/* ── Buttons (incl. file-uploader Browse) ── */
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background: #3B36A8 !important;
    border: 1px solid #7C83F5 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] button *,
section[data-testid="stSidebar"] button p { color: #FFFFFF !important; }
/* icon-only / narrow buttons: centre the content (Streamlit left-aligns it by default) */
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
    display: flex !important;
    align-items: center;
    justify-content: center;
    padding: 0 6px !important;
    min-height: 2.5rem;
}
section[data-testid="stSidebar"] .stButton > button > div,
section[data-testid="stSidebar"] .stButton > button span,
section[data-testid="stSidebar"] .stButton > button p {
    margin: 0 !important;
    width: auto !important;
    flex-shrink: 0;
    white-space: nowrap;
    text-align: center;
}
section[data-testid="stSidebar"] .stButton > button:hover { background: #4F46E5 !important; }

/* ── File uploader ── */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #262065 !important;
    border: 1px dashed #7C83F5 !important;
    border-radius: 10px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color: #EEF1FF !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small { color: #B9C2F7 !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { background: #4F46E5 !important; }

/* ── Sliders, toggles, checkboxes ── */
section[data-testid="stSidebar"] [data-testid="stSlider"] *,
section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stTickBarMax"] { color: #EEF1FF !important; }
section[data-testid="stSidebar"] [data-testid="stToggle"] label p,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] label p { font-size: 0.93rem; }

/* ── Alerts inside the sidebar ── */
section[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(99, 102, 241, 0.16) !important;
    border: 1px solid #5B55C9;
}
section[data-testid="stSidebar"] [data-testid="stAlert"] * { color: #EEF1FF !important; }

/* input wrappers (password box with the eye button, number boxes) */
section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-baseweb="base-input"] {
    background: #262065 !important;
    border-color: #5B55C9 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-baseweb="input"] button { background: transparent !important; border: none !important; }
section[data-testid="stSidebar"] [data-baseweb="input"] svg { fill: #B9C2F7 !important; color: #B9C2F7 !important; }
section[data-testid="stSidebar"] [data-testid="stNumberInputContainer"],
section[data-testid="stSidebar"] [data-testid="stNumberInputContainer"] * { background: #262065; }
/* slider value labels (they used the dark-on-dark accent colour) */
section[data-testid="stSidebar"] [data-baseweb="slider"] * { color: #D5DBFF !important; }

/* Remove the vertical line artifact on radio/selectbox */
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div > div {
    border-left: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="vs-hero">
  <p class="vs-title">🎬 VidSage</p>
  <p class="vs-tagline">Drop a video. Get the full story — transcript, chapters, AI explanation &amp; Q&amp;A.</p>
  <div>
    <span class="vs-pill">🎙️ Whisper Transcription</span>
    <span class="vs-pill">🧠 Scene Detection</span>
    <span class="vs-pill">📑 Auto Chapters</span>
    <span class="vs-pill">💬 Q&amp;A Mode</span>
    <span class="vs-pill">📄 Word / PDF Export</span>
    <span class="vs-pill">▶️ Online Video (YouTube, Vimeo &amp; more)</span>
    <span class="vs-pill">🇲🇾 EN / BM / ZH</span>
  </div>
  <div class="vs-divider"></div>
</div>
""", unsafe_allow_html=True)

def _sidebar_section(title: str) -> None:
    """Render a visually distinct section header in the sidebar."""
    st.markdown(
        f"""<div style="
            background: rgba(99,102,241,0.15);
            border-left: 3px solid #6366F1;
            border-radius: 0 8px 8px 0;
            padding: 6px 12px;
            margin: 16px 0 8px;
        "><span style="color:#E0E7FF;font-weight:700;font-size:0.95rem;">{title}</span></div>""",
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
# Every setting lives in a compact "card": the card title shows what is currently selected and you
# open it to change it.  Choices are remembered between sessions (settings.json) — never API keys.

SETTINGS_FILE = Path.home() / "Documents" / "VidSage" / "settings.json"
# Widget keys that are remembered.  Deliberately NOT remembered: API keys, names & terms, time range,
# slides, and the "let Gemini watch the video" switch (it uploads the whole video, so it stays opt-in).
PREF_KEYS = [
    "w_ai_engine", "w_stt", "w_whisper_model", "w_xxl_model", "w_xxl_filter",
    "w_lang_mode", "w_lang_single", "w_lang_multi", "output_language_select",
    "w_frame_mode", "w_auto_frames", "w_max_frames", "w_interval_fixed", "w_scene_sens",
    "w_interval_dense", "w_dedup", "w_minwords", "w_cleanup", "w_autosave",
]


def _load_prefs() -> dict:
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if k in PREF_KEYS}
    except Exception:
        return {}


def _save_prefs() -> None:
    """Write the current choices if they changed (keys of widgets that are hidden right now keep their old value)."""
    try:
        merged = dict(st.session_state.get("_prefs_saved", {}))
        for k in PREF_KEYS:
            if k in st.session_state:
                merged[k] = st.session_state[k]
        if merged != st.session_state.get("_prefs_saved"):
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_FILE.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
            st.session_state["_prefs_saved"] = merged
    except Exception:
        pass                                    # a read-only disk must never break the app


def _ensure(key: str, options, default) -> None:
    """Make sure a widget's value is a valid choice (falls back to the remembered one, then the default)."""
    if st.session_state.get(key) not in options:
        remembered = st.session_state.get("_prefs_saved", {}).get(key)
        st.session_state[key] = remembered if remembered in options else default


def _card(icon: str, title: str, current: str):
    """A collapsed card whose header shows the current choice — open it to change it."""
    return st.expander(f"{icon} {title}  ·  **{current}**", expanded=False)


AI_OPTIONS = ["Gemini (Free)", "Claude (Paid)", "OpenAI (Paid)", DEEPSEEK_ENGINE]
AI_SHORT = {"Gemini (Free)": "Gemini (free)", "Claude (Paid)": "Claude", "OpenAI (Paid)": "OpenAI GPT-4o",
            DEEPSEEK_ENGINE: "DeepSeek Flash"}
STT_OPTIONS = ["💻 Local Whisper", "🚀 Local GPU (Faster-Whisper-XXL)", "⚡ Groq Cloud", "☁️ OpenAI Whisper"]
STT_SHORT = {"💻 Local Whisper": "Local Whisper (CPU)", "🚀 Local GPU (Faster-Whisper-XXL)": "Local GPU",
             "⚡ Groq Cloud": "Groq cloud", "☁️ OpenAI Whisper": "OpenAI cloud"}
FRAME_MODES = ["📅 Fixed Interval", "🧠 Smart Scene Detection", "🎙️ Speech-Aligned", "🔍 Dense + Dedup"]

with st.sidebar:
    st.markdown("### 🎬 VidSage")
    st.caption("AI-powered video explainer")

    # ── remembered settings + first-run defaults (DeepSeek + local GPU) ──
    if "_prefs_loaded" not in st.session_state:
        _saved = _load_prefs()
        for _k, _v in _saved.items():
            st.session_state.setdefault(_k, _v)
        st.session_state["_prefs_saved"] = dict(_saved)
        st.session_state["_prefs_loaded"] = True
    if "w_stt" not in st.session_state:
        st.session_state["w_stt"] = (STT_OPTIONS[1] if find_xxl(_secret("FW_XXL_PATH")) else STT_OPTIONS[0])
    _ensure("w_ai_engine", AI_OPTIONS, DEEPSEEK_ENGINE)
    _ensure("w_stt", STT_OPTIONS, STT_OPTIONS[0])

    gemini_watch_video = False

    # ── AI engine ──────────────────────────────────────────────────────────────
    with _card("🤖", "AI engine", AI_SHORT[st.session_state["w_ai_engine"]]):
        ai_engine = st.radio(
            "AI engine", AI_OPTIONS, key="w_ai_engine", label_visibility="collapsed",
            captions=[
                "Free tier · 1,500 requests/day",
                "Best quality · ~$0.10–0.30 per video",
                "GPT-4o · ~$0.07–0.20 per video",
                "Cheapest · ~$0.01–0.03 per video · reads images",
            ],
        )

        if ai_engine == "Gemini (Free)":
            gemini_key = st.text_input("Gemini API key", value=_secret("GEMINI_API_KEY"), type="password",
                                       help="Free key at aistudio.google.com")
            gemini_watch_video = st.checkbox(
                "🎥 Let Gemini watch the whole video (sees motion, hears audio)",
                value=False,
                help="Uploads the VIDEO to Google (deleted right after; public YouTube links are passed "
                     "directly). Better for demos and anything where motion matters. Costs more tokens "
                     "(~100 per second of video). On the free tier Google may use your data to improve "
                     "its products — don't use it for private videos.")
            if gemini_watch_video:
                st.caption("⚠️ The full video is sent to Google. Falls back to frames if it fails.")
            claude_key = ""
            openai_key = ""
        elif ai_engine == "Claude (Paid)":
            claude_key = st.text_input("Anthropic API key", value=_secret("ANTHROPIC_API_KEY"), type="password",
                                       help="Loaded from .streamlit/secrets.toml")
            gemini_key = ""
            openai_key = ""
        elif ai_engine == DEEPSEEK_ENGINE:
            # Shares the OpenAI-compatible code path, so the key travels in `openai_key`.
            openai_key = st.text_input("DeepSeek API key", value=_secret("DEEPSEEK_API_KEY"), type="password",
                                       help="Get a key at platform.deepseek.com")
            claude_key = ""
            gemini_key = ""
        else:  # OpenAI
            openai_key = st.text_input("OpenAI API key", value=_secret("OPENAI_API_KEY"), type="password",
                                       help="Get a key at platform.openai.com")
            claude_key = ""
            gemini_key = ""

    active_key = (
        gemini_key if ai_engine == "Gemini (Free)"
        else openai_key if _is_oai(ai_engine)
        else claude_key
    )
    st.caption("✅ API key ready" if active_key else "⚠️ API key needed — open **AI engine** above to add it")

    # ── Transcription ──────────────────────────────────────────────────────────
    _stt_now = st.session_state["w_stt"]
    _stt_sub = ""
    if _stt_now == STT_OPTIONS[0]:
        _ensure("w_whisper_model", ["base", "small", "medium", "large"], "medium")
        _stt_sub = f" · {st.session_state['w_whisper_model']}"
    elif _stt_now == STT_OPTIONS[1]:
        _stt_sub = f" · {st.session_state.get('w_xxl_model', 'large-v2')}"
    with _card("🎙️", "Transcription", STT_SHORT[_stt_now] + _stt_sub):
        _stt_choice = st.radio(
            "Transcription engine", STT_OPTIONS, key="w_stt", label_visibility="collapsed",
            captions=[
                "Free & private · runs on your PC's processor (slow)",
                "Free & private · uses your NVIDIA GPU via your Faster-Whisper-XXL folder",
                "Very fast · ~$0.11 per hour of audio · large-v3",
                "Fast · ~$0.36 per hour of audio",
            ],
            help="Only used when the video has no YouTube captions. Cloud options upload the "
                 "AUDIO track (never the video) to the provider.",
        )
        stt_engine = {"💻 Local Whisper": "local", "🚀 Local GPU (Faster-Whisper-XXL)": "xxl",
                      "⚡ Groq Cloud": "groq", "☁️ OpenAI Whisper": "openai"}[_stt_choice]
        groq_key = ""
        openai_stt_key = ""
        xxl_exe = None
        xxl_model = "large-v2"
        xxl_filter = "Off"
        if stt_engine == "xxl":
            _xxl_found = find_xxl(_secret("FW_XXL_PATH"))
            _xxl_text = st.text_input(
                "Faster-Whisper-XXL folder",
                value=str(_xxl_found.parent) if _xxl_found else "",
                help="The folder that contains faster-whisper-xxl.exe. Saved for next time if you add "
                     'FW_XXL_PATH = "..." to .streamlit/secrets.toml.')
            xxl_exe = find_xxl(_xxl_text, strict=True)
            if xxl_exe is None:
                st.error("Couldn't find faster-whisper-xxl.exe in that folder. Download it from "
                         "github.com/Purfview/whisper-standalone-win and paste its folder here.")
            else:
                _xxl_models = xxl_installed_models(xxl_exe) or ["large-v2"]
                _ensure("w_xxl_model", _xxl_models, _xxl_models[0])
                xxl_model = st.selectbox(
                    "Model", _xxl_models, key="w_xxl_model",
                    help="Models already inside your XXL folder. large-v2 was the steadier choice for Malay and "
                         "Chinese–English calls in testing; large-v3 can mis-detect Malay as English.")
                _ensure("w_xxl_filter", list(NOISE_FILTERS), "Off")
                xxl_filter = st.selectbox(
                    "Noise filter (optional)", list(NOISE_FILTERS), key="w_xxl_filter",
                    help="Cleans the audio before transcribing. In testing on clear phone calls these filters did "
                         "NOT improve accuracy and sometimes made it worse (they can drop a call's first words), "
                         "so leave it Off unless the recording is genuinely noisy and compare the results. "
                         "'Isolate voice' stayed closest to the unfiltered text but takes about twice as long.")
        elif stt_engine == "groq":
            groq_key = st.text_input("Groq API key",
                                     value=_secret("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", ""),
                                     type="password", help="Free key at console.groq.com")
        elif stt_engine == "openai":
            openai_stt_key = st.text_input(
                "OpenAI API key (for transcription)",
                value=_secret("OPENAI_API_KEY") or (openai_key if ai_engine == "OpenAI (Paid)" else ""),
                type="password", help="platform.openai.com")

        if stt_engine == "local":
            whisper_model = st.radio(
                "Accuracy vs. speed", ["base", "small", "medium", "large"], key="w_whisper_model",
                captions=["Fastest, less accurate", "Good balance", "Recommended ✓", "Most accurate, slow"],
            )
        else:
            whisper_model = stt_engine      # XXL / cloud: the model is fixed by the choice above

    if stt_engine == "xxl":
        st.caption("🔒 Runs on this PC · nothing uploaded" if xxl_exe else "⚠️ Faster-Whisper-XXL not found — open **Transcription**")
    elif stt_engine == "local":
        st.caption("🔒 Runs on this PC · nothing uploaded")
    elif stt_engine == "groq":
        st.caption("☁️ Audio (not video) goes to Groq" + ("" if groq_key else " · ⚠️ key needed"))
    else:
        st.caption("☁️ Audio (not video) goes to OpenAI" + ("" if openai_stt_key else " · ⚠️ key needed"))

    # ── Language ───────────────────────────────────────────────────────────────
    _ensure("w_lang_mode", ["Single language", "Multilingual"], "Single language")
    _ensure("w_lang_single", list(WHISPER_LANGUAGES.keys()), "Auto-detect")
    _ensure("output_language_select", ["Auto (match video)"] + list(OUTPUT_LANGUAGES.keys()), "Auto (match video)")
    if "w_lang_multi" not in st.session_state:
        st.session_state["w_lang_multi"] = ["English", "Malay"]
    _lang_now = ("Multilingual" if st.session_state["w_lang_mode"] == "Multilingual"
                 else st.session_state["w_lang_single"])
    _out_now = st.session_state["output_language_select"]
    with _card("🌐", "Language", f"{_lang_now} → {'same as video' if _out_now.startswith('Auto') else _out_now}"):
        audio_lang_mode = st.radio(
            "Audio language mode", ["Single language", "Multilingual"], key="w_lang_mode", horizontal=True,
            help="Use **Multilingual** when speakers switch between languages in the same video.",
        )
        if audio_lang_mode == "Single language":
            _lang_display = st.selectbox(
                "Audio language", options=list(WHISPER_LANGUAGES.keys()), key="w_lang_single",
                help="**Auto-detect** is recommended unless Whisper gets the language wrong. "
                     "Forcing the wrong language (e.g. 'English' on a Chinese video) will produce "
                     "garbled output. Type to search for your language.",
            )
            whisper_lang_code, whisper_initial_prompt = WHISPER_LANGUAGES[_lang_display]
            whisper_language_display = _lang_display
        else:
            _multi_langs = st.multiselect(
                "Languages spoken in this video", options=MULTILINGUAL_BASE_OPTIONS, key="w_lang_multi",
                help="Which languages appear in the video (used as a label). Whisper detects the spoken "
                     "language by itself — no instruction is sent to it, because hints made it hallucinate. "
                     "Add key names in the box below if some are being misspelled.",
            )
            if _multi_langs:
                _lang_list = ", ".join(_multi_langs)
                whisper_lang_code = None   # let Whisper auto-detect per segment
                whisper_initial_prompt = ""   # no hint: hints caused hallucinations
                whisper_language_display = f"Multilingual ({_lang_list})"
            else:
                whisper_lang_code = None
                whisper_initial_prompt = ""
                whisper_language_display = "Auto-detect"

        _terms = st.text_input(
            "Names & terms to spell correctly (optional)",
            placeholder="e.g. Rainz, Gudang, Puan Noor",
            help="A short comma-separated list. It nudges Whisper towards these spellings (sent as hotwords "
                 "with Faster-Whisper-XXL, which pushes harder — only list words that really occur). "
                 "(Keep it to words and names — full sentences or instructions can make Whisper hallucinate.)",
            key="w_terms",
        ).strip()
        names_terms = _terms
        if _terms and stt_engine != "xxl":      # Faster-Whisper-XXL gets them as hotwords instead
            whisper_initial_prompt = f"{whisper_initial_prompt} {_terms}".strip()

        _AUTO_LANG = "Auto (match video)"
        output_language = st.selectbox(
            "AI writes in", options=[_AUTO_LANG] + list(OUTPUT_LANGUAGES.keys()), key="output_language_select",
            help="**Auto (match video):** the AI explanation is written in whatever language "
                 "Whisper detects in the audio. Select a specific language to override.",
        )
        if output_language == _AUTO_LANG:
            _last_detected = st.session_state.get("last_detected_output_language", "")
            st.caption(f"🌐 Last detected: **{_last_detected}**" if _last_detected
                       else "🌐 Detected automatically from the audio")
    # Store in session state so _show_results can read it for re-analysis and PKM generation
    st.session_state["output_language"] = output_language

    # ── Frame capture ──────────────────────────────────────────────────────────
    _ensure("w_frame_mode", FRAME_MODES, FRAME_MODES[3])
    st.session_state.setdefault("w_auto_frames", True)
    _fr_now = st.session_state["w_frame_mode"]
    _fr_count = "auto count" if st.session_state["w_auto_frames"] else f"{st.session_state.get('w_max_frames', 15)} frames"
    with _card("🖼️", "Frames", f"{_fr_now.split(' ', 1)[1]} · {_fr_count}"):
        frame_mode = st.radio(
            "Capture mode", FRAME_MODES, key="w_frame_mode", label_visibility="collapsed",
            captions=[
                "One frame every N seconds",
                "Frame when the picture changes",
                "Frames at dense speech moments",
                "Dense sample → drops near-duplicates · best for meetings",
            ],
        )
        auto_frames = st.toggle(
            "Auto-recommend frame count", key="w_auto_frames",
            help=("VidSage calculates the ideal number of frames based on video length, "
                  "capture mode, and AI engine. Turn off to set an exact number manually."),
        )
        if auto_frames:
            max_frames = None   # resolved later once duration is known
            st.caption("📐 About 1 frame per 2–3 min · min 8 · cap 40 (Claude / OpenAI / DeepSeek) or 80 (Gemini)")
            if st.toggle("📊 Show example recommendations", value=False, key="w_show_examples"):
                _preview_rows = [("5 min video", 5 * 60), ("30 min video", 30 * 60),
                                 ("1 hr video", 60 * 60), ("2 hr video", 120 * 60)]
                _ex_data = {"Video length": [], "Fixed": [], "Scene": [], "Speech": [], "Dense": []}
                for label, secs in _preview_rows:
                    _ex_data["Video length"].append(label)
                    _ex_data["Fixed"].append(recommended_frames(secs, "📅 Fixed Interval", ai_engine))
                    _ex_data["Scene"].append(recommended_frames(secs, "🧠 Smart Scene Detection", ai_engine))
                    _ex_data["Speech"].append(recommended_frames(secs, "🎙️ Speech-Aligned", ai_engine))
                    _ex_data["Dense"].append(recommended_frames(secs, "🔍 Dense + Dedup", ai_engine))
                st.dataframe(pd.DataFrame(_ex_data), hide_index=True)
        else:
            st.session_state.setdefault("w_max_frames", 15)
            max_frames = st.number_input(
                "Frames to send to AI", min_value=1, max_value=500, step=5, key="w_max_frames",
                help=("No hard cap — set as many as you need. "
                      "Practical sweet spot: 10–20 for Claude, 20–60 for Gemini. "
                      "Very high counts increase cost and processing time."),
            )

        if frame_mode == "📅 Fixed Interval":
            st.session_state.setdefault("w_interval_fixed", 45)
            frame_interval = st.slider("Capture a frame every N seconds", min_value=15, max_value=120,
                                       step=15, key="w_interval_fixed")
            scene_threshold = 0.4
            min_words_seg = 8

        elif frame_mode == "🧠 Smart Scene Detection":
            frame_interval = 45
            _ensure("w_scene_sens", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 0.4)
            scene_threshold = st.select_slider(
                "Scene sensitivity", options=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7], key="w_scene_sens",
                format_func=lambda x: {0.2: "Very sensitive (most frames)", 0.3: "High", 0.4: "Medium ✓",
                                       0.5: "Low", 0.6: "High threshold", 0.7: "Only big changes"}[x],
                help="Lower = captures more scene changes. Medium works well for lectures and screen recordings.",
            )
            min_words_seg = 8

        elif frame_mode == "🔍 Dense + Dedup":
            st.session_state.setdefault("w_interval_dense", 2)
            frame_interval = st.slider(
                "Sample interval (seconds)", min_value=1, max_value=5, step=1, key="w_interval_dense",
                help=("A frame is read every N seconds before deduplication. "
                      "2s catches virtually all slide changes in meetings and lectures. "
                      "Increase to 3–4s for very long videos (2 hr+) to save processing time."),
            )
            _ensure("w_dedup", [0.80, 0.85, 0.88, 0.90, 0.92], 0.88)
            scene_threshold = st.select_slider(
                "Duplicate removal", options=[0.80, 0.85, 0.88, 0.90, 0.92], key="w_dedup",
                format_func=lambda x: {0.80: "Very aggressive — minimal frames", 0.85: "Aggressive",
                                       0.88: "Balanced ✓", 0.90: "Moderate", 0.92: "Light — keep more frames"}[x],
                help=("Lower = more frames are treated as duplicates and dropped. "
                      "0.88 works well for most Zoom, lecture, and screen recordings."),
            )
            min_words_seg = 8

        else:  # Speech-Aligned
            frame_interval = 45
            scene_threshold = 0.4
            st.session_state.setdefault("w_minwords", 8)
            min_words_seg = st.slider(
                "Min words per speech segment", min_value=3, max_value=25, key="w_minwords",
                help=("Only capture a frame during segments where the speaker says at least "
                      "this many words. Higher = fewer but more content-rich frames. "
                      "Recommended: 8 for lectures, 5 for fast-paced training."),
            )

    # ── Time range ─────────────────────────────────────────────────────────────
    _t0 = (st.session_state.get("range_start_txt") or "").strip()
    _t1 = (st.session_state.get("range_end_txt") or "").strip()
    with _card("✂️", "Time range", f"{_t0 or 'start'} → {_t1 or 'end'}" if (_t0 or _t1) else "whole video"):
        st.caption("Analyse only part of a long video. Leave blank for the whole video. "
                   "Accepts SS, MM:SS or HH:MM:SS.")
        _rc1, _rc2 = st.columns(2)
        with _rc1:
            st.text_input("Start", key="range_start_txt", placeholder="e.g. 12:30")
        with _rc2:
            st.text_input("End", key="range_end_txt", placeholder="e.g. 18:00")

    # ── Slides ─────────────────────────────────────────────────────────────────
    _n_prev_slides = len(st.session_state.get("slide_images") or [])
    with _card("📎", "Slides", f"{_n_prev_slides} slide(s)" if _n_prev_slides else "none"):
        st.caption("Upload the presentation used in the video. The AI reads the slides alongside the "
                   "frames and transcript for a more complete analysis.")
        uploaded_slides = st.file_uploader(
            "Slides file", type=["pdf", "pptx", "png", "jpg", "jpeg", "webp"], accept_multiple_files=True,
            help="PDF or PPTX: all pages/slides are extracted automatically. "
                 "Images: upload one per slide or all at once.",
            label_visibility="collapsed",
        )
        # Process slides immediately so we can show a preview
        _slide_b64_list: list[str] = []
        if uploaded_slides:
            with st.spinner(f"Processing {len(uploaded_slides)} slide file(s)…"):
                _slide_b64_list, _slide_sections, _slide_cap_info = \
                    process_slide_files(uploaded_slides, ai_engine)
            if _slide_b64_list:
                st.success(f"✅ {_slide_cap_info}")
                st.session_state["slide_sections"] = _slide_sections
                if st.toggle(f"👁️ Preview slides ({len(_slide_b64_list)})", value=False, key="w_slide_preview"):
                    _prev_cols = st.columns(3)
                    for _si, _sb64 in enumerate(_slide_b64_list):
                        with _prev_cols[_si % 3]:
                            st.image(base64.b64decode(_sb64), caption=f"Slide {_si + 1}", width='stretch')
    # Store in session state so all modes can access it
    st.session_state["slide_images"] = _slide_b64_list if _slide_b64_list else None

    # ── Quick switches ─────────────────────────────────────────────────────────
    st.session_state.setdefault("w_cleanup", True)
    do_cleanup = st.toggle(
        "🔧 Auto-correct transcript errors", key="w_cleanup",
        help="Uses AI to fix misheard words, local names, and code-switching errors before analysis.",
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    st.session_state.setdefault("w_autosave", True)
    if "save_folder" not in st.session_state:
        st.session_state.save_folder = str(Path.home() / "Documents" / "VidSage" / "results")
    _folder_name = Path(st.session_state.save_folder).name or st.session_state.save_folder
    with _card("💾", "Auto-save", f"on · {_folder_name}" if st.session_state["w_autosave"] else "off"):
        auto_save = st.toggle("Auto-save results after analysis", key="w_autosave")
        col_path, col_btn = st.columns([4, 1])
        with col_path:
            st.session_state.save_folder = st.text_input("Save folder", value=st.session_state.save_folder)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📁", width='stretch', help="Browse for a folder"):
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.wm_attributes("-topmost", True)
                chosen = filedialog.askdirectory(
                    title="Select save folder", initialdir=st.session_state.save_folder)
                root.destroy()
                if chosen:
                    st.session_state.save_folder = chosen
                    st.rerun()
    save_folder = st.session_state.save_folder

    st.caption("ℹ️ First run? Whisper downloads its model once (~1–3 GB).")
    _save_prefs()


def _tidy_md_for_display(text: str) -> str:
    """Make Obsidian-flavoured markdown (PKM notes) render cleanly inside Streamlit.
    Display only — the raw note and exports keep their [[wikilinks]]."""
    # [[Topic]] / [[Topic|alias]] → bold text (Streamlit has no wikilink support)
    text = re.sub(r'\[\[([^\]|\n]+)\|([^\]\n]+)\]\]', r'**\2**', text)
    text = re.sub(r'\[\[([^\]\n]+)\]\]', r'**\1**', text)
    # "**Label:** > quote" → quote on its own line so the blockquote actually renders
    text = re.sub(r'(\*\*[^\n*]+:\*\*)[ \t]*>[ \t]*', r'\1\n\n> ', text)
    # A line right after a "> quote" would be absorbed into the quote (markdown lazy
    # continuation) — end the quote block with a blank line.
    text = re.sub(r'(?m)^(>[^\n]*)\n(?!>|\n|$)', r'\1\n\n', text)
    # Headings that became only-bold from the step above look doubled: "### **X**" → "### X"
    text = re.sub(r'(?m)^(#{1,6})\s+\*\*(.+?)\*\*\s*$', r'\1 \2', text)
    return text


def _section_label(heading: str) -> str:
    """'## 📖 Concept Deep-Dives' → '📖 Concept Deep-Dives'; '## 1. Overview' → 'Overview'."""
    label = re.sub(r'^#{2}\s*', '', heading).strip()
    return re.sub(r'^\d+\.\s*', '', label).strip()


def _render_sectioned_markdown(text: str, first_expanded: bool = True) -> None:
    """
    Split a markdown string on ## headings and render each section
    inside its own st.expander.  The first section is expanded by default;
    all others start collapsed.
    """
    text = _tidy_md_for_display(text)
    # Split on lines that start with exactly "## " (level-2 heading)
    parts = re.split(r'(?m)^(## .+)$', text.strip())
    # parts[0] is anything before the first ## heading (often empty or a title)
    preamble = parts[0].strip()
    if preamble:
        st.markdown(preamble)

    sections = []
    i = 1
    while i < len(parts) - 1:
        heading = parts[i]          # e.g. "## 1. Overview"
        body    = parts[i + 1]      # content until the next heading
        sections.append((_section_label(heading), body.strip()))
        i += 2

    for idx, (label, body) in enumerate(sections):
        expanded = (idx == 0) if first_expanded else False
        with st.expander(label, expanded=expanded):
            st.markdown(body)


def _show_results(video_name, explanation, chapters, full_transcript,
                  timestamped_transcript, srt_content, frames, frame_ts,
                  ai_engine, claude_key, gemini_key,
                  duration=0, save_folder="", analysis_time_sec=0,
                  key_ns: str = ""):
    """Render results tabs + Q&A for any video (upload or YouTube).

    key_ns — optional namespace prefix injected into every widget key.
    Pass the history stamp when calling from the History tab so that
    multiple entries with the same video name never collide.
    """
    base_name  = os.path.splitext(video_name)[0]
    video_type = infer_video_type(video_name)

    # Build a short, alphanumeric-only key stem so widget keys are always
    # valid regardless of how wild the video title is (emojis, arrows, etc.)
    # Strategy: keep only word-chars, truncate to 40 chars, append a short
    # hash of the full name to preserve uniqueness.
    import hashlib as _hl
    _name_hash = _hl.md5(base_name.encode("utf-8", errors="replace")).hexdigest()[:8]
    _safe_stem  = re.sub(r'\W+', '_', base_name)[:40].strip('_')
    _wkey = f"{key_ns}_{_safe_stem}_{_name_hash}" if key_ns else f"{_safe_stem}_{_name_hash}"

    # ── Metrics row ──────────────────────────────────────────────────────────
    if analysis_time_sec:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("⏱️ Duration",  fmt_time(duration) if duration else "—")
        m2.metric("🤖 AI Time",   _fmt_analysis_time(analysis_time_sec))
        m3.metric("📝 Words",     f"{len(full_transcript.split()):,}")
        m4.metric("🖼️ Frames",    len(frames))
        m5.metric("📑 Chapters",  len(chapters))
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⏱️ Duration",  fmt_time(duration) if duration else "—")
        m2.metric("📝 Words",     f"{len(full_transcript.split()):,}")
        m3.metric("🖼️ Frames",    len(frames))
        m4.metric("📑 Chapters",  len(chapters))
    st.markdown("")

    # Prefer a re-analysed explanation if one exists in session state
    _corrected_key       = f"corrected_explanation_{base_name}"
    active_explanation   = st.session_state.get(_corrected_key, explanation)
    _corrected_chaps_key = f"corrected_chapters_{base_name}"
    active_chapters      = st.session_state.get(_corrected_chaps_key, chapters)

    tab_explain, tab_pkm, tab_chapters, tab_transcript, tab_ts, tab_srt, tab_frames = st.tabs(
        ["📄 Explanation", "🧠 PKM Note", "📑 Chapters", "📝 Transcript",
         "🕐 Timestamped Transcript", "💬 Subtitles (SRT)", "🖼️ Sampled Frames"]
    )

    with tab_explain:
        if _corrected_key in st.session_state:
            st.success("✅ Showing **updated** explanation (re-analysed with your corrections).")

        # ── Export buttons at the top so they're always visible ──────────
        _cur_full_trans = st.session_state.get(
            f"edit_transcript_{base_name}", full_transcript)
        _cur_ts_trans   = st.session_state.get(
            f"edit_ts_{base_name}", timestamped_transcript)

        # Word: generated once per content change and remembered (rebuilding it on every click made the
        # results screen lag — about 0.2 s for a 10-min video, nearly 1 s for a 2-hour one).
        _word_key = hashlib.md5(json.dumps(
            [video_name, active_explanation, _cur_full_trans, _cur_ts_trans, chapters],
            default=str, ensure_ascii=False).encode("utf-8")).hexdigest()
        _word_memo = st.session_state.get("_word_memo")
        if _word_memo and _word_memo[0] == _word_key:
            word_bytes, _word_err = _word_memo[1], None
        else:
            try:
                word_bytes = export_word(video_name, active_explanation,
                                         _cur_full_trans, _cur_ts_trans, chapters)
                _word_err = None
                st.session_state["_word_memo"] = (_word_key, word_bytes)
            except Exception as _we:
                word_bytes = None
                _word_err = str(_we)

        # PDF: slow (launches Word), so use lazy generation via session state
        _pdf_cache_key = f"_pdf_cache_{_wkey}"
        _pdf_gen_key   = f"_pdf_gen_{_wkey}"

        col_md, col_word, col_pdf = st.columns(3)
        with col_md:
            st.download_button("⬇️ Markdown (.md)", data=active_explanation,
                               file_name=f"{base_name}.md",
                               mime="text/markdown", width='stretch',
                               key=f"dl_md_{_wkey}")
        with col_word:
            if word_bytes:
                st.download_button("⬇️ Word (.docx)", data=word_bytes,
                                   file_name=f"{base_name}.docx",
                                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                   width='stretch', key=f"dl_docx_{_wkey}")
            else:
                st.error(f"Word export failed: {_word_err}")
        with col_pdf:
            if _pdf_cache_key in st.session_state:
                # Already generated — show download button immediately
                _cached = st.session_state[_pdf_cache_key]
                if isinstance(_cached, bytes):
                    st.download_button("⬇️ PDF", data=_cached,
                                       file_name=f"{base_name}.pdf",
                                       mime="application/pdf", width='stretch',
                                       key=f"dl_pdf_{_wkey}")
                else:
                    st.error(f"PDF failed: {_cached}")
                    if st.button("🔄 Retry PDF", key=f"retry_pdf_{_wkey}", width='stretch'):
                        del st.session_state[_pdf_cache_key]
                        st.rerun()
            else:
                # Not yet generated — show a "Generate PDF" button
                if st.button("📄 Generate PDF", key=_pdf_gen_key, width='stretch',
                             help="Converts via Microsoft Word — takes ~5 seconds"):
                    with st.spinner("Generating PDF via Microsoft Word…"):
                        try:
                            _pdf_bytes = export_pdf(video_name, active_explanation,
                                                    _cur_full_trans, _cur_ts_trans)
                            st.session_state[_pdf_cache_key] = _pdf_bytes
                        except Exception as _pe:
                            st.session_state[_pdf_cache_key] = str(_pe)
                    st.rerun()

        # ── Save updated files to disk (shown only when corrections exist) ─
        if _corrected_key in st.session_state and save_folder and os.path.exists(save_folder):
            st.info(
                "📝 You have a re-analysed explanation in session. "
                "Click below to overwrite the saved `.md`, `.docx`, and `.pdf` on disk."
            )
            if st.button("💾 Save updated files to disk",
                         key=f"save_updated_{_wkey}", type="primary"):
                _saved, _failed = [], []
                try:
                    _expl_files = [f for f in os.listdir(save_folder)
                                   if base_name in f and f.endswith("_explanation.md")]
                    for _m in _expl_files:
                        with open(os.path.join(save_folder, _m),
                                  "w", encoding="utf-8") as _fh:
                            _fh.write(active_explanation)
                        _saved.append(_m)
                except Exception as _e:
                    _failed.append(f".md: {_e}")
                try:
                    _docx_files = [f for f in os.listdir(save_folder)
                                   if base_name in f and f.endswith(".docx")]
                    for _m in _docx_files:
                        with open(os.path.join(save_folder, _m), "wb") as _fh:
                            _fh.write(export_word(video_name, active_explanation,
                                                  _cur_full_trans, _cur_ts_trans,
                                                  chapters))
                        _saved.append(_m)
                except Exception as _e:
                    _failed.append(f".docx: {_e}")
                try:
                    _pdf_files = [f for f in os.listdir(save_folder)
                                  if base_name in f and f.endswith(".pdf")]
                    for _m in _pdf_files:
                        with open(os.path.join(save_folder, _m), "wb") as _fh:
                            _fh.write(export_pdf(video_name, active_explanation,
                                                 _cur_full_trans, _cur_ts_trans))
                        _saved.append(_m)
                except Exception as _e:
                    _failed.append(f".pdf: {_e}")

                if _saved:
                    st.success(f"✅ Updated: {', '.join(_saved)}")
                if _failed:
                    st.warning(f"⚠️ Some files could not be saved: {'; '.join(_failed)}")

        # ── Re-analyse section ────────────────────────────────────────────────
        # Persistent keys for this widget namespace
        _ra_trigger_key = f"_ra_{_wkey}"
        _ra_done_key    = f"_ra_done_{_wkey}"
        _ra_b64_key     = f"_ra_b64_{_wkey}"
        _ra_sec_key     = f"_ra_sec_{_wkey}"

        # Completion banner — shown after st.rerun(), then cleared
        if _ra_done_key in st.session_state:
            st.success(st.session_state.pop(_ra_done_key))
        if f"_ra_warn_{_wkey}" in st.session_state:
            st.warning(st.session_state.pop(f"_ra_warn_{_wkey}"))

        with st.expander("🔄 Re-analyse", expanded=False):
            st.caption(
                "Re-run the AI to regenerate the **explanation** and **chapters** "
                "using the saved transcript. Whisper is **not** re-run."
            )
            _ra_slide_files = st.file_uploader(
                "📎 Attach slides / reference documents *(optional)*",
                type=["pdf", "pptx", "png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key=f"_ra_upload_{_wkey}",
            )
            if _ra_slide_files:
                with st.spinner("Processing slides…"):
                    _new_b64, _new_secs, _new_info = process_slide_files(
                        _ra_slide_files, ai_engine)
                st.session_state[_ra_b64_key] = _new_b64
                st.session_state[_ra_sec_key] = _new_secs
                st.success(f"✅ {_new_info}")

            _use_slides = st.session_state.get(_ra_b64_key) \
                or st.session_state.get("slide_images") or []
            _use_secs   = st.session_state.get(_ra_sec_key) \
                or st.session_state.get("slide_sections") or []

            _info_parts = [f"🖼️ {len(frames)} saved frames"]
            _info_parts.append(
                f"📎 {len(_use_slides)} slide(s) attached"
                if _use_slides else "📎 No slides")
            st.info("  ·  ".join(_info_parts))

            # Warn when Claude + large deck will trigger chunked mode
            if ai_engine == "Claude (Paid)" and len(_use_slides or []) >= 80:
                import math as _m
                _n_chunks = _m.ceil(len(_use_slides) / 25)
                _est_min = int(_n_chunks * 65 / 60) + 2
                st.warning(
                    f"⏱️ **{len(_use_slides)} slides detected** — Claude's rate limit is "
                    f"30 000 tokens/minute. VidSage will automatically split these into "
                    f"**{_n_chunks} batches of 25** with 65 s pauses between each. "
                    f"Estimated time: **~{_est_min} minutes**. "
                    f"Alternatively, switch to **Gemini (Free)** in the sidebar "
                    f"for much higher limits and faster large-deck analysis."
                )

            if st.button(
                "🤖 Re-analyse"
                + (f"  ·  {len(_use_slides)} slide(s)" if _use_slides else ""),
                key=f"btn_ra_{_wkey}", type="primary", width="stretch",
            ):
                st.session_state[_ra_trigger_key] = True

        # ── Execute re-analysis when triggered ───────────────────────────────
        if st.session_state.get(_ra_trigger_key):
            _cur_ts   = st.session_state.get(
                f"edit_ts_{base_name}", timestamped_transcript)
            _use_s    = st.session_state.get(_ra_b64_key) \
                or st.session_state.get("slide_images") or []
            _use_sc   = st.session_state.get(_ra_sec_key) \
                or st.session_state.get("slide_sections") or []
            _out_lang = st.session_state.get("output_language", "English")

            with st.spinner("🤖 Generating explanation…"):
                try:
                    if ai_engine == "Gemini (Free)":
                        _new_expl = analyze_with_gemini(
                            _cur_ts, frames, frame_ts, duration, gemini_key,
                            video_type, _out_lang,
                            slide_images=_use_s or None,
                            slide_sections=_use_sc or None)
                    elif _is_oai(ai_engine):
                        _new_expl = analyze_with_openai(
                            _cur_ts, frames, frame_ts, duration, openai_key,
                            video_type, _out_lang,
                            slide_images=_use_s or None,
                            slide_sections=_use_sc or None, ai_engine=ai_engine)
                    else:
                        _new_expl = analyze_with_claude(
                            _cur_ts, frames, frame_ts, duration, claude_key,
                            video_type, _out_lang,
                            slide_images=_use_s or None,
                            slide_sections=_use_sc or None)
                except Exception as _e:
                    st.error(f"Re-analysis failed: {_e}")
                    st.session_state.pop(_ra_trigger_key, None)
                    st.stop()

            with st.spinner("📑 Regenerating chapters…"):
                try:
                    _new_chaps = generate_chapters(
                        _cur_ts, duration, video_type,
                        ai_engine, claude_key, gemini_key,
                        _out_lang, openai_key=openai_key)
                except Exception:
                    _new_chaps = active_chapters  # fallback

            # Update session state
            st.session_state[_corrected_key]       = _new_expl
            st.session_state[_corrected_chaps_key] = _new_chaps
            st.session_state.pop(f"pkm_note_{base_name}", None)

            # Save all related files to disk
            _resynced_ra, _failed_ra = [], []
            if save_folder and os.path.exists(save_folder):
                _cur_full = st.session_state.get(
                    f"edit_transcript_{base_name}", full_transcript)
                for _fn in os.listdir(save_folder):
                    if base_name not in _fn:
                        continue
                    _fp = os.path.join(save_folder, _fn)
                    try:
                        if _fn.endswith("_explanation.md"):
                            with open(_fp, "w", encoding="utf-8") as _fh:
                                _fh.write(_new_expl)
                            _resynced_ra.append(_fn)
                        elif _fn.endswith("_chapters.txt"):
                            with open(_fp, "w", encoding="utf-8") as _fh:
                                _fh.write(chapters_to_text(_new_chaps))
                            _resynced_ra.append(_fn)
                        elif _fn.endswith(".docx"):
                            with open(_fp, "wb") as _fh:
                                _fh.write(export_word(
                                    video_name, _new_expl,
                                    _cur_full, _cur_ts, _new_chaps))
                            _resynced_ra.append(_fn)
                        elif _fn.endswith(".pdf"):
                            with open(_fp, "wb") as _fh:
                                _fh.write(export_pdf(
                                    video_name, _new_expl, _cur_full, _cur_ts))
                            _resynced_ra.append(_fn)
                    except Exception as _fe:
                        _failed_ra.append(f"{_fn}: {_fe}")

            st.session_state.pop(_ra_trigger_key, None)
            _slide_note = f" · {len(_use_s)} slide(s)" if _use_s else ""
            st.session_state[_ra_done_key] = (
                f"✅ Re-analysis complete{_slide_note}. "
                + (f"Updated: {', '.join(_resynced_ra)}"
                   if _resynced_ra else "Nothing saved (no save folder configured).")
            )
            if _failed_ra:
                st.session_state[f"_ra_warn_{_wkey}"] = (
                    f"⚠️ Some files could not be saved: {'; '.join(_failed_ra)}")
            st.rerun()

        st.markdown("---")
        _render_sectioned_markdown(active_explanation, first_expanded=True)

    with tab_pkm:
        st.caption(
            "A rich, structured study note — paste directly into **Obsidian**, **Heptabase**, "
            "Notion, or any Markdown editor. Generated on-demand to avoid extra API cost."
        )

        pkm_key = f"pkm_note_{base_name}"

        if pkm_key not in st.session_state:
            st.markdown("""
> **What you'll get:** YAML frontmatter · one-line summary · concept table · deep-dives with
> quotes · chronological walkthrough · comparison tables · step-by-step process ·
> key quotes · action items · wikilinks · follow-up questions.
""")
            if st.button("🧠 Generate PKM Note", type="primary", key=f"gen_pkm_{_wkey}"):
                with st.spinner("Writing your knowledge note… (30–60 seconds)"):
                    note = generate_pkm_note(
                        video_name, video_type, duration,
                        active_explanation, timestamped_transcript,
                        chapters, ai_engine, claude_key, gemini_key,
                        st.session_state.get("output_language", "English"),
                        openai_key=openai_key,
                    )
                st.session_state[pkm_key] = note
                # Auto-save to results folder if it exists
                pkm_file = os.path.join(save_folder, f"{base_name}_pkm_note.md")
                try:
                    os.makedirs(save_folder, exist_ok=True)
                    with open(pkm_file, "w", encoding="utf-8") as fh:
                        fh.write(note)
                except Exception:
                    pass  # silent — save folder may not be configured
                st.rerun()
        else:
            note = st.session_state[pkm_key]

            # Download buttons
            pkm_col1, pkm_col2, pkm_col3 = st.columns([2, 2, 4])
            with pkm_col1:
                st.download_button(
                    "⬇️ Download (.md)",
                    data=note,
                    file_name=f"{base_name}_pkm_note.md",
                    mime="text/markdown",
                    width='stretch',
                    key=f"dl_pkm_{_wkey}",
                )
            with pkm_col2:
                if st.button("🔄 Regenerate", key=f"regen_pkm_{_wkey}",
                             width='stretch'):
                    del st.session_state[pkm_key]
                    st.rerun()

            st.markdown("---")

            # Rendered preview — each PKM section is its own expander
            with st.expander("👁️ Preview", expanded=True):
                _render_sectioned_markdown(note, first_expanded=True)

            # Raw markdown for easy copy-paste
            with st.expander("📋 Raw Markdown (copy into Obsidian / Heptabase)",
                             expanded=False):
                st.code(note, language="markdown")

    with tab_chapters:
        st.caption("Auto-generated chapter markers — copy into YouTube, Notion, or your LMS.")
        chapters_text = chapters_to_text(active_chapters)

        with st.expander(f"📑 Chapter List  ({len(active_chapters)} chapters)", expanded=True):
            for c in active_chapters:
                col_t, col_title = st.columns([1, 5])
                with col_t:
                    st.markdown(f"**`{c['time']}`**")
                with col_title:
                    st.markdown(c["title"])

        with st.expander("📋 YouTube-Style Text (copy & paste)", expanded=False):
            st.text_area("Chapters", chapters_text, height=200,
                         key=f"ta_chapters_{_wkey}", label_visibility="collapsed")
            st.download_button("⬇️ Download Chapters (.txt)", data=chapters_text,
                               file_name=f"{base_name}_chapters.txt", mime="text/plain",
                               key=f"dl_chapters_{_wkey}")

    with tab_transcript:
        st.caption(
            "✏️ **Editable** — click inside the box and correct any misheard words. "
            "Then click **Save corrections** to write the changes to disk."
        )
        _tk = f"edit_transcript_{base_name}"
        if _tk not in st.session_state:
            st.session_state[_tk] = full_transcript
        edited_transcript = st.text_area(
            "Full Transcript", st.session_state[_tk],
            height=450, key=f"ta_transcript_{_wkey}"
        )
        _tc1, _tc2, _tc3 = st.columns([2, 2, 4])
        with _tc1:
            if st.button("💾 Save corrections", key=f"save_trans_{_wkey}",
                         width='stretch'):
                st.session_state[_tk] = edited_transcript
                if save_folder and os.path.exists(save_folder):
                    _cur_ts  = st.session_state.get(f"edit_ts_{base_name}", timestamped_transcript)
                    _cur_srt = st.session_state.get(f"edit_srt_{base_name}", srt_content)
                    _updated, _failed = [], []

                    # 1. Transcript .txt
                    for _m in [f for f in os.listdir(save_folder)
                               if base_name in f and f.endswith("_transcript.txt")
                               and "timestamped" not in f]:
                        with open(os.path.join(save_folder, _m), "w", encoding="utf-8") as _fh:
                            _fh.write(edited_transcript)
                        _updated.append(_m)

                    # 2. Word (.docx)
                    for _m in [f for f in os.listdir(save_folder)
                               if base_name in f and f.endswith(".docx")]:
                        try:
                            with open(os.path.join(save_folder, _m), "wb") as _fh:
                                _fh.write(export_word(video_name, active_explanation,
                                                      edited_transcript, _cur_ts, chapters))
                            _updated.append(_m)
                        except Exception as _e:
                            _failed.append(f"{_m}: {_e}")

                    # 3. PDF
                    for _m in [f for f in os.listdir(save_folder)
                               if base_name in f and f.endswith(".pdf")]:
                        try:
                            with open(os.path.join(save_folder, _m), "wb") as _fh:
                                _fh.write(export_pdf(video_name, active_explanation,
                                                     edited_transcript, _cur_ts))
                            _updated.append(_m)
                        except Exception as _e:
                            _failed.append(f"{_m}: {_e}")

                    # 4. Queue re-analysis → will update explanation .md, .docx, .pdf, PKM
                    st.session_state[f"reanalyse_{base_name}"] = True
                    # 5. Invalidate PKM so it regenerates on next view
                    st.session_state.pop(f"pkm_note_{base_name}", None)

                    if _updated:
                        st.success(
                            f"✅ Saved & synced: {', '.join(_updated)}  \n"
                            "🔄 Re-analysing to update explanation, .md, .docx, .pdf and PKM note…"
                        )
                    if _failed:
                        st.warning(f"⚠️ Could not update: {'; '.join(_failed)}")
                else:
                    st.success("✅ Corrections saved!")
        with _tc2:
            st.download_button("⬇️ Download", data=edited_transcript,
                               file_name=f"{base_name}_transcript.txt",
                               mime="text/plain", width='stretch',
                               key=f"dl_trans_{_wkey}")
        with _tc3:
            if edited_transcript != full_transcript:
                st.info("💡 You have unsaved edits. You can also **Re-analyse** in the "
                        "Timestamped tab to update the AI explanation with your corrections.")

    with tab_ts:
        st.caption(
            "✏️ **Editable** — correct timestamps or words directly. "
            "After saving, use **Re-analyse** to regenerate the AI explanation with your fixes."
        )
        _tsk = f"edit_ts_{base_name}"
        if _tsk not in st.session_state:
            st.session_state[_tsk] = timestamped_transcript
        edited_ts = st.text_area(
            "Timestamped Transcript", st.session_state[_tsk],
            height=450, key=f"ta_ts_{_wkey}"
        )

        _tsc1, _tsc2, _tsc3 = st.columns([2, 2, 2])
        with _tsc1:
            if st.button("💾 Save corrections", key=f"save_ts_{_wkey}",
                         width='stretch'):
                st.session_state[_tsk] = edited_ts
                # Also sync the SRT session key to the freshly derived SRT
                _new_srt = timestamped_to_srt(edited_ts)
                if _new_srt:
                    st.session_state[f"edit_srt_{base_name}"] = _new_srt

                if save_folder and os.path.exists(save_folder):
                    _cur_full = st.session_state.get(f"edit_transcript_{base_name}", full_transcript)
                    _updated, _failed = [], []

                    # 1. Timestamped transcript .txt
                    for _m in [f for f in os.listdir(save_folder)
                               if base_name in f and f.endswith("_transcript_timestamped.txt")]:
                        with open(os.path.join(save_folder, _m), "w", encoding="utf-8") as _fh:
                            _fh.write(edited_ts)
                        _updated.append(_m)

                    # 2. Subtitles .srt — regenerated from corrected timestamps
                    if _new_srt:
                        for _m in [f for f in os.listdir(save_folder)
                                   if base_name in f and f.endswith(".srt")]:
                            with open(os.path.join(save_folder, _m), "w", encoding="utf-8") as _fh:
                                _fh.write(_new_srt)
                            _updated.append(_m)

                    # 3. Word (.docx)
                    for _m in [f for f in os.listdir(save_folder)
                               if base_name in f and f.endswith(".docx")]:
                        try:
                            with open(os.path.join(save_folder, _m), "wb") as _fh:
                                _fh.write(export_word(video_name, active_explanation,
                                                      _cur_full, edited_ts, chapters))
                            _updated.append(_m)
                        except Exception as _e:
                            _failed.append(f"{_m}: {_e}")

                    # 4. PDF
                    for _m in [f for f in os.listdir(save_folder)
                               if base_name in f and f.endswith(".pdf")]:
                        try:
                            with open(os.path.join(save_folder, _m), "wb") as _fh:
                                _fh.write(export_pdf(video_name, active_explanation,
                                                     _cur_full, edited_ts))
                            _updated.append(_m)
                        except Exception as _e:
                            _failed.append(f"{_m}: {_e}")

                    # 5. Queue re-analysis → updates explanation .md, .docx, .pdf, PKM
                    st.session_state[f"reanalyse_{base_name}"] = True
                    # 6. Invalidate PKM so it regenerates on next view
                    st.session_state.pop(f"pkm_note_{base_name}", None)

                    if _updated:
                        st.success(
                            f"✅ Saved & synced: {', '.join(_updated)}  \n"
                            "🔄 Re-analysing to update explanation, .md, .docx, .pdf and PKM note…"
                        )
                    if _failed:
                        st.warning(f"⚠️ Could not update: {'; '.join(_failed)}")
                else:
                    st.success("✅ Corrections saved!")
        with _tsc2:
            st.download_button("⬇️ Download", data=edited_ts,
                               file_name=f"{base_name}_transcript_timestamped.txt",
                               mime="text/plain", width='stretch',
                               key=f"dl_ts_{_wkey}")
        with _tsc3:
            _reanalyse_key = f"reanalyse_{base_name}"
            if st.button("🤖 Re-analyse with corrections",
                         key=f"btn_reanalyse_{_wkey}",
                         width='stretch',
                         help="Re-runs the AI explanation using your corrected transcript. "
                              "No Whisper re-run needed — takes ~30 seconds."):
                st.session_state[_reanalyse_key] = True

        if st.session_state.get(f"reanalyse_{base_name}"):
            st.markdown("---")
            st.info(
                "🤖 Re-analysing with your corrected transcript…  \n"
                "Explanation, chapters, and all saved files will be updated."
            )
            _ts_reanalyse_err = None
            try:
                _out_lang_ts = st.session_state.get("output_language", "English")
                with st.spinner("🤖 Generating updated explanation…"):
                    if ai_engine == "Gemini (Free)":
                        new_explanation = analyze_with_gemini(
                            edited_ts, frames, frame_ts,
                            duration, gemini_key, video_type, _out_lang_ts)
                    elif _is_oai(ai_engine):
                        new_explanation = analyze_with_openai(
                            edited_ts, frames, frame_ts,
                            duration, openai_key, video_type, _out_lang_ts, ai_engine=ai_engine)
                    else:
                        new_explanation = analyze_with_claude(
                            edited_ts, frames, frame_ts,
                            duration, claude_key, video_type, _out_lang_ts)

                with st.spinner("📑 Regenerating chapters…"):
                    try:
                        new_chapters_ts = generate_chapters(
                            edited_ts, duration, video_type,
                            ai_engine, claude_key, gemini_key,
                            _out_lang_ts, openai_key=openai_key)
                    except Exception:
                        new_chapters_ts = active_chapters

                # Use the corrected full transcript if the user saved edits
                _cur_full = st.session_state.get(
                    f"edit_transcript_{base_name}", full_transcript)

                # ── Update ALL saved files so everything stays in sync ────────
                _resynced = []
                if save_folder and os.path.exists(save_folder):
                    # 1. Explanation .md
                    for _m in [f for f in os.listdir(save_folder)
                               if base_name in f and f.endswith("_explanation.md")]:
                        with open(os.path.join(save_folder, _m),
                                  "w", encoding="utf-8") as _fh:
                            _fh.write(new_explanation)
                        _resynced.append(_m)

                    # 2. Chapters .txt
                    for _m in [f for f in os.listdir(save_folder)
                               if base_name in f and f.endswith("_chapters.txt")]:
                        with open(os.path.join(save_folder, _m),
                                  "w", encoding="utf-8") as _fh:
                            _fh.write(chapters_to_text(new_chapters_ts))
                        _resynced.append(_m)

                    # 3. Word (.docx)
                    for _m in [f for f in os.listdir(save_folder)
                               if base_name in f and f.endswith(".docx")]:
                        try:
                            with open(os.path.join(save_folder, _m), "wb") as _fh:
                                _fh.write(export_word(
                                    video_name, new_explanation,
                                    _cur_full, edited_ts, new_chapters_ts))
                            _resynced.append(_m)
                        except Exception:
                            pass

                    # 4. PDF
                    for _m in [f for f in os.listdir(save_folder)
                               if base_name in f and f.endswith(".pdf")]:
                        try:
                            with open(os.path.join(save_folder, _m), "wb") as _fh:
                                _fh.write(export_pdf(
                                    video_name, new_explanation,
                                    _cur_full, edited_ts))
                            _resynced.append(_m)
                        except Exception:
                            pass

                    # 5. PKM note — regenerate if one already exists on disk
                    _pkm_files = [f for f in os.listdir(save_folder)
                                  if base_name in f and f.endswith("_pkm_note.md")]
                    if _pkm_files:
                        try:
                            pkm_prompt = PKM_NOTE_PROMPT.format(
                                video_name=video_name,
                                output_language_instruction=OUTPUT_LANGUAGES.get(
                                    _out_lang_ts, OUTPUT_LANGUAGES["English"]),
                                explanation=new_explanation,
                                transcript=edited_ts,
                            )
                            if ai_engine == "Gemini (Free)" and gemini_key:
                                from google import genai as _gai
                                _gc = _gai.Client(api_key=gemini_key)
                                _pr = _gemini_generate(_gc, pkm_prompt)
                                new_pkm = _pr.text.strip()
                            elif _is_oai(ai_engine) and openai_key:
                                from openai import OpenAI as _OAI
                                _oc = _oai_client(ai_engine, openai_key)
                                _or = _oc.chat.completions.create(
                                    model=_oai_model(ai_engine, "small"), **_oai_extra(ai_engine), max_tokens=4096,
                                    messages=[{"role": "user", "content": pkm_prompt}])
                                new_pkm = _or.choices[0].message.content.strip()
                            else:
                                import anthropic as _anth
                                _ac = _anth.Anthropic(api_key=claude_key)
                                _pr = _ac.messages.create(
                                    model="claude-haiku-4-5", max_tokens=4096,
                                    messages=[{"role": "user", "content": pkm_prompt}])
                                new_pkm = _pr.content[0].text.strip()
                            for _m in _pkm_files:
                                with open(os.path.join(save_folder, _m),
                                          "w", encoding="utf-8") as _fh:
                                    _fh.write(new_pkm)
                                _resynced.append(_m)
                            st.session_state[f"pkm_note_{base_name}"] = new_pkm
                        except Exception:
                            pass   # PKM regeneration is non-critical

                # Update session state so all tabs refresh
                st.session_state[_corrected_key]       = new_explanation
                st.session_state[_corrected_chaps_key] = new_chapters_ts
                st.session_state.pop(f"pkm_note_{base_name}", None)
                del st.session_state[f"reanalyse_{base_name}"]
                st.session_state[f"_ra_done_{_wkey}"] = (
                    f"✅ Re-analysis complete — explanation, chapters, and "
                    f"{len(_resynced)} file(s) updated. "
                    "Switch to **📄 Explanation** to see the result."
                )
                st.rerun()
            except Exception as _re_err:
                st.error(f"Re-analysis failed: {_re_err}")
                del st.session_state[f"reanalyse_{base_name}"]

    with tab_srt:
        st.caption(
            "✏️ **Editable** — fix any subtitle text directly. "
            "Load in VLC: **Subtitle → Add Subtitle File**"
        )
        _srtk = f"edit_srt_{base_name}"
        if _srtk not in st.session_state:
            st.session_state[_srtk] = srt_content
        edited_srt = st.text_area(
            "SRT Subtitles", st.session_state[_srtk],
            height=450, key=f"ta_srt_{_wkey}"
        )
        _srt1, _srt2 = st.columns([2, 6])
        with _srt1:
            if st.button("💾 Save corrections", key=f"save_srt_{_wkey}",
                         width='stretch'):
                st.session_state[_srtk] = edited_srt
                if save_folder and os.path.exists(save_folder):
                    _srt_matches = [f for f in os.listdir(save_folder)
                                    if base_name in f and f.endswith(".srt")]
                    for _m in _srt_matches:
                        with open(os.path.join(save_folder, _m), "w", encoding="utf-8") as _fh:
                            _fh.write(edited_srt)
                st.success("✅ Subtitles saved!")
        with _srt2:
            st.download_button("⬇️ Download Subtitle File (.srt)", data=edited_srt,
                               file_name=f"{base_name}_subtitles.srt", mime="text/plain",
                               width='stretch', key=f"dl_srt_{_wkey}")

    with tab_frames:
        if frames:
            # ── Session keys for frame selection state ────────────────────
            _fsel_key    = f"frame_sel_{base_name}"     # list[bool]
            _frerun_key  = f"frame_rerun_{base_name}"   # trigger re-analysis

            if _fsel_key not in st.session_state or \
                    len(st.session_state[_fsel_key]) != len(frames):
                st.session_state[_fsel_key] = [True] * len(frames)

            sel = st.session_state[_fsel_key]
            n_selected = sum(sel)

            st.caption(
                f"**{len(frames)} frame(s)** captured — ranked by text/data density. "
                f"Uncheck any decorative or irrelevant frames, then click "
                f"**Re-analyse** to rerun the AI with only the selected frames."
            )

            # ── Select-all / deselect-all helpers ─────────────────────────
            _fh1, _fh2, _fh3 = st.columns([2, 2, 4])
            with _fh1:
                if st.button("✅ Select all", key=f"fsel_all_{_wkey}",
                             width='stretch'):
                    st.session_state[_fsel_key] = [True] * len(frames)
                    for _i in range(len(frames)):
                        st.session_state[f"fchk_{_wkey}_{_i}"] = True
                    st.rerun()
            with _fh2:
                if st.button("☐ Deselect all", key=f"fsel_none_{_wkey}",
                             width='stretch'):
                    st.session_state[_fsel_key] = [False] * len(frames)
                    for _i in range(len(frames)):
                        st.session_state[f"fchk_{_wkey}_{_i}"] = False
                    st.rerun()

            st.markdown("")

            # ── Frame grid with checkboxes ────────────────────────────────
            cols = st.columns(3)
            for i, (b64, ts) in enumerate(zip(frames, frame_ts)):
                with cols[i % 3]:
                    st.image(
                        base64.b64decode(b64),
                        caption=f"#{i+1}  ·  {fmt_time(ts)}",
                        width='stretch',
                    )
                    _ck = f"fchk_{_wkey}_{i}"
                    if _ck not in st.session_state:              # set the start value via session state only
                        st.session_state[_ck] = sel[i]
                    checked = st.checkbox("Include in analysis", key=_ck)
                    st.session_state[_fsel_key][i] = checked      # no st.rerun(): it doubled the redraw cost

            # The count below the grid reflects the boxes as ticked in THIS run.
            n_selected = sum(st.session_state[_fsel_key])

            # ── Re-analyse with selected frames ───────────────────────────
            st.markdown("---")
            _fc1, _fc2 = st.columns([3, 5])
            with _fc1:
                _disabled = n_selected == 0
                if st.button(
                    f"🤖 Re-analyse with {n_selected} selected frame(s)",
                    key=f"frame_reanalyse_{_wkey}",
                    type="primary",
                    disabled=_disabled,
                    width='stretch',
                    help="Reruns the AI explanation using only the checked frames. "
                         "No Whisper re-run — fast (~30 s).",
                ):
                    st.session_state[_frerun_key] = True
            with _fc2:
                if n_selected == 0:
                    st.warning("Select at least one frame to re-analyse.")
                elif n_selected < len(frames):
                    st.info(
                        f"💡 {len(frames) - n_selected} frame(s) will be excluded "
                        f"from the next analysis run."
                    )

            if st.session_state.get(_frerun_key):
                sel_frames  = [b for b, keep in zip(frames, st.session_state[_fsel_key]) if keep]
                sel_ts      = [t for t, keep in zip(frame_ts, st.session_state[_fsel_key]) if keep]
                _cur_ts_txt = st.session_state.get(f"edit_ts_{base_name}", timestamped_transcript)

                with st.spinner(f"Re-analysing with {len(sel_frames)} frame(s)…"):
                    try:
                        if ai_engine == "Gemini (Free)":
                            new_expl = analyze_with_gemini(
                                _cur_ts_txt, sel_frames, sel_ts,
                                duration, gemini_key, video_type,
                                st.session_state.get("output_language", "English"))
                        elif _is_oai(ai_engine):
                            new_expl = analyze_with_openai(
                                _cur_ts_txt, sel_frames, sel_ts,
                                duration, openai_key, video_type,
                                st.session_state.get("output_language", "English"), ai_engine=ai_engine)
                        else:
                            new_expl = analyze_with_claude(
                                _cur_ts_txt, sel_frames, sel_ts,
                                duration, claude_key, video_type,
                                st.session_state.get("output_language", "English"))

                        st.session_state[f"corrected_explanation_{base_name}"] = new_expl
                        del st.session_state[_frerun_key]

                        # Persist to all saved files
                        _cur_full = st.session_state.get(
                            f"edit_transcript_{base_name}", full_transcript)
                        if save_folder and os.path.exists(save_folder):
                            for _m in [f for f in os.listdir(save_folder)
                                       if base_name in f and f.endswith("_explanation.md")]:
                                with open(os.path.join(save_folder, _m),
                                          "w", encoding="utf-8") as _fh:
                                    _fh.write(new_expl)
                            for _m in [f for f in os.listdir(save_folder)
                                       if base_name in f and f.endswith(".docx")]:
                                try:
                                    with open(os.path.join(save_folder, _m), "wb") as _fh:
                                        _fh.write(export_word(video_name, new_expl,
                                                              _cur_full, _cur_ts_txt, chapters))
                                except Exception:
                                    pass
                            for _m in [f for f in os.listdir(save_folder)
                                       if base_name in f and f.endswith(".pdf")]:
                                try:
                                    with open(os.path.join(save_folder, _m), "wb") as _fh:
                                        _fh.write(export_pdf(video_name, new_expl,
                                                             _cur_full, _cur_ts_txt))
                                except Exception:
                                    pass

                        st.success(
                            "✅ Re-analysis complete. Switch to the **📄 Explanation** "
                            "tab to see the updated explanation."
                        )
                        st.rerun()
                    except Exception as _fe:
                        st.error(f"Re-analysis failed: {_fe}")
                        del st.session_state[_frerun_key]
        else:
            st.info("No frames were captured for this video.")

    # Q&A
    st.markdown("---")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_video" not in st.session_state:
        st.session_state.last_video = ""
    if st.session_state.last_video != video_name:
        st.session_state.chat_history = []
        st.session_state.last_video = video_name

    _chat_count = len(st.session_state.chat_history)
    _qa_label   = (f"💬 Q&A Session  ({_chat_count // 2} question(s))"
                   if _chat_count else "💬 Q&A Session — Ask anything about this video")

    with st.expander(_qa_label, expanded=bool(_chat_count)):
        st.caption("Ask questions about the content, request clarifications, or dig deeper.")

        if not st.session_state.chat_history:
            st.markdown("**Suggested questions:**")
            c1, c2, c3 = st.columns(3)
            suggestions = [
                "What are the key takeaways?",
                "Summarise this in 3 bullet points",
                "What was discussed at the halfway point?",
                "What action items were mentioned?",
                "Explain the main concept in simple terms",
                "What questions were asked during the session?",
            ]
            for i, sug in enumerate(suggestions):
                with [c1, c2, c3][i % 3]:
                    if st.button(sug, width='stretch', key=f"sug_{_wkey}_{i}"):
                        st.session_state.chat_history.append({"role": "user", "content": sug})
                        with st.spinner("Thinking…"):
                            ans = answer_question(sug, explanation, timestamped_transcript,
                                                 chapters, video_type,
                                                 st.session_state.chat_history[:-1],
                                                 ai_engine, claude_key, gemini_key,
                                                 openai_key=openai_key)
                        st.session_state.chat_history.append({"role": "assistant", "content": ans})
                        st.rerun()

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if question := st.chat_input("Ask a question about the video…",
                                     key=f"chat_input_{_wkey}"):
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    ans = answer_question(question, explanation, timestamped_transcript,
                                         chapters, video_type,
                                         st.session_state.chat_history[:-1],
                                         ai_engine, claude_key, gemini_key,
                                         openai_key=openai_key)
                st.markdown(ans)
            st.session_state.chat_history.append({"role": "assistant", "content": ans})

        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat", key=f"clear_{_wkey}"):
                st.session_state.chat_history = []
                st.rerun()


def process_one_video(video_file, video_name: str, status_container,
                      frame_mode, max_frames, frame_interval, scene_threshold, min_words_seg,
                      whisper_model, whisper_lang_code, whisper_initial_prompt,
                      whisper_language_display, do_cleanup,
                      ai_engine, claude_key, gemini_key,
                      auto_save, save_folder,
                      prefetched_transcript: dict | None = None,
                      output_language: str = "English",
                      slide_images: list[str] | None = None,
                      slide_sections: list[dict] | None = None,
                      openai_key: str = "") -> dict:
    """
    Full pipeline for a single video. Writes to status_container for live UI updates.
    Returns a dict with all results.

    Speech-Aligned mode reorders the pipeline: audio → transcribe → frames
    (so Whisper segments can guide frame selection). Other modes keep the
    original order: frames → audio → transcribe.
    """
    _t0 = time.time()
    video_type = infer_video_type(video_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "video.mp4")
        audio_path = os.path.join(tmpdir, "audio.wav")

        with open(video_path, "wb") as f:
            if hasattr(video_file, "getvalue"):
                f.write(video_file.getvalue())
            else:
                f.write(video_file)

        # Optional time range: cut the clip so every later step (audio, frames,
        # Whisper) only sees that window; timestamps are shifted back afterwards.
        _rng_start = parse_timecode(st.session_state.get("range_start_txt"))
        _rng_end   = parse_timecode(st.session_state.get("range_end_txt"))
        range_offset = 0.0
        _rng_window = None
        if _rng_start is not None or _rng_end is not None:
            _full_dur = get_video_duration(video_path)
            _rng_start = max(_rng_start or 0.0, 0.0)
            _rng_end = min(_rng_end, _full_dur) if _rng_end is not None else _full_dur
            if _rng_end - _rng_start < 1:
                raise ValueError(
                    f"Invalid time range: start {fmt_time(_rng_start)} / end "
                    f"{fmt_time(_rng_end)} (video is {fmt_time(_full_dur)} long).")
            status_container.write(
                f"✂️ Analysing only {fmt_time(_rng_start)} → {fmt_time(_rng_end)}…")
            _clip_path = os.path.join(tmpdir, "clip.mp4")
            trim_clip(video_path, _clip_path, _rng_start, _rng_end)
            video_path = _clip_path
            range_offset = _rng_start
            _rng_window = (_rng_start, _rng_end)

        model_times = {"base": "5–10 min", "small": "10–20 min",
                       "medium": "30–60 min", "large": "60–120 min",
                       "groq": "1–3 min", "openai": "1–5 min", "xxl": "about 20 min per hour of audio"}
        est = model_times.get(whisper_model, "a few minutes")

        # Whisper speed multipliers: how many seconds of audio processed per wall-clock second
        _whisper_speed = {"base": 8.0, "small": 4.0, "medium": 2.0, "large": 1.0,
                         "groq": 60.0, "openai": 30.0, "xxl": 3.0}

        def _run_whisper_threaded(label: str):
            """Extract audio, transcribe, return (full_text, timestamped, segments, detected_lang)."""
            status_container.write("🔊 Extracting audio…")
            extract_audio(video_path, audio_path)

            # Measure audio duration for progress estimation
            audio_cap = cv2.VideoCapture(audio_path)
            audio_dur = audio_cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(
                audio_cap.get(cv2.CAP_PROP_FPS), 1)
            audio_cap.release()
            if audio_dur < 1:
                audio_dur = get_video_duration(video_path)  # fallback

            speed      = _whisper_speed.get(whisper_model, 2.0)
            est_sec    = max(audio_dur / speed, 5)

            status_container.write(f"✍️ Transcribing (`{_stt_label(whisper_model)}`) — est. {est}…")
            whisper_bar = status_container.progress(0, text=label)
            rc, ec = {}, {}

            def _worker():
                try:
                    rc["out"] = transcribe_audio_any(audio_path, whisper_model,
                                                    whisper_lang_code, whisper_initial_prompt)
                except Exception as e:
                    ec["err"] = e

            t   = threading.Thread(target=_worker, daemon=True)
            t0  = time.time()
            t.start()
            while t.is_alive():
                elapsed  = time.time() - t0
                # Cap at 95 % while running — jumps to 100 % on completion
                pct      = min(int(elapsed / est_sec * 100), 95)
                elapsed_fmt = _fmt_analysis_time(elapsed)
                est_fmt     = _fmt_analysis_time(est_sec)
                whisper_bar.progress(
                    pct,
                    text=f"Whisper transcribing… {pct}%  ({elapsed_fmt} elapsed / ~{est_fmt} est.)"
                )
                time.sleep(1.0)
            t.join()
            if "err" in ec:
                raise ec["err"]
            total_fmt = _fmt_analysis_time(time.time() - t0)
            whisper_bar.progress(100, text=f"Transcription complete ✅  (took {total_fmt})")
            if stt_engine == "xxl" and _xxl_mod.LAST_NOTE:
                status_container.write(f"   → ⚠️ {_xxl_mod.LAST_NOTE}")
            return rc["out"]

        # ── Resolve auto frame count (needs duration first) ──────────────────
        # For Speech-Aligned, duration comes out of extraction (after transcribe).
        # For other modes, do a fast metadata probe so we can resolve auto count
        # before the actual frame extraction loop.
        if max_frames is None:
            probe_duration = get_video_duration(video_path)
            resolved_frames = recommended_frames(probe_duration, frame_mode, ai_engine)
            status_container.write(
                f"📐 Auto frame count: **{resolved_frames}** frames "
                f"(for {fmt_time(probe_duration)} video)"
            )
        else:
            resolved_frames = max_frames

        def _use_prefetched():
            """Unpack a pre-fetched transcript dict into the standard variables."""
            nonlocal output_language
            full_transcript        = prefetched_transcript["full_text"]
            timestamped_transcript = prefetched_transcript["timestamped_text"]
            srt_content            = prefetched_transcript["srt_content"]
            segments               = prefetched_transcript["segments"]
            lang   = prefetched_transcript.get("language", "unknown")
            source = "auto-generated" if prefetched_transcript.get("is_generated") else "manual"
            # Auto-resolve output language from YouTube transcript language
            if output_language == "Auto (match video)":
                _yt_code = (prefetched_transcript.get("language_code") or "").split("-")[0].lower()
                _yt_mapped = WHISPER_LANG_TO_OUTPUT.get(_yt_code or lang, "English")
                output_language = _yt_mapped
                st.session_state["output_language"] = _yt_mapped
                st.session_state["last_detected_output_language"] = _yt_mapped
            status_container.write(
                f"   → ✅ YouTube transcript used ({lang} · {source}) — "
                f"{len(full_transcript.split()):,} words  *(Whisper skipped)*"
            )
            return full_transcript, timestamped_transcript, srt_content, segments

        # ── SPEECH-ALIGNED: transcript first, then use segments for frame selection
        if frame_mode == "🎙️ Speech-Aligned":
            if prefetched_transcript:
                full_transcript, timestamped_transcript, srt_content, segments = _use_prefetched()
            else:
                full_transcript, timestamped_transcript, segments, _det_lang = \
                    _run_whisper_threaded("Whisper processing…")
                srt_content = generate_srt(segments)
                # Resolve detected language → output language
                _mapped = WHISPER_LANG_TO_OUTPUT.get(_det_lang, "English")
                st.session_state["last_detected_output_language"] = _mapped
                if output_language == "Auto (match video)":
                    output_language = _mapped
                    st.session_state["output_language"] = _mapped
                status_container.write(
                    f"   \u2192 Detected language: **{_det_lang}** "
                    f"\u2192 AI output: **{output_language}**"
                )
                status_container.write(f"   → {len(full_transcript.split()):,} words transcribed")

            # Re-resolve using actual duration from the transcript
            if max_frames is None:
                duration_probe = segments[-1]["end"] if segments else probe_duration
                resolved_frames = recommended_frames(duration_probe, frame_mode, ai_engine)

            status_container.write("🎙️ Selecting speech-aligned frames…")
            frames, frame_ts, duration = extract_speech_aligned_frames(
                video_path, segments, min_words_seg, resolved_frames)
            status_container.write(
                f"   → {len(frames)} frames selected from {fmt_time(duration)} "
                f"(deduplication applied)")

        # ── FIXED INTERVAL, SMART SCENE, or DENSE DEDUP: frames first, then transcribe
        else:
            if frame_mode == "🧠 Smart Scene Detection":
                status_container.write("🧠 Detecting scene changes…")
                frames, frame_ts, duration = extract_scene_frames(
                    video_path, resolved_frames, scene_threshold)
                status_container.write(
                    f"   → {len(frames)} scene changes in {fmt_time(duration)}")
            elif frame_mode == "🔍 Dense + Dedup":
                status_container.write(f"🔍 Dense sampling every {frame_interval}s + pHash dedup…")
                frames, frame_ts, duration = extract_dense_dedup_frames(
                    video_path, frame_interval, resolved_frames, scene_threshold)
                status_container.write(
                    f"   → {len(frames)} unique frames kept from {fmt_time(duration)}")
            else:
                status_container.write("📸 Extracting frames…")
                frames, frame_ts, duration = extract_keyframes(
                    video_path, frame_interval, resolved_frames)
                status_container.write(
                    f"   → {len(frames)} frames from {fmt_time(duration)}")

            if prefetched_transcript:
                full_transcript, timestamped_transcript, srt_content, segments = _use_prefetched()
            else:
                full_transcript, timestamped_transcript, segments, _det_lang = \
                    _run_whisper_threaded("Whisper processing…")
                srt_content = generate_srt(segments)
                # Resolve detected language → output language
                _mapped = WHISPER_LANG_TO_OUTPUT.get(_det_lang, "English")
                st.session_state["last_detected_output_language"] = _mapped
                if output_language == "Auto (match video)":
                    output_language = _mapped
                    st.session_state["output_language"] = _mapped
                status_container.write(
                    f"   \u2192 Detected language: **{_det_lang}** "
                    f"\u2192 AI output: **{output_language}**"
                )
                status_container.write(f"   → {len(full_transcript.split()):,} words transcribed")

        # Put timestamps back on the ORIGINAL video's timeline when a range was used
        if _rng_window:
            if range_offset:
                frame_ts = [t + range_offset for t in frame_ts]
            if prefetched_transcript:
                # Captions cover the whole video → keep only the requested window
                _lo, _hi = _rng_window
                segments = [sg for sg in segments if _lo <= sg["start"] < _hi]
            else:
                segments = [{**sg, "start": sg["start"] + range_offset,
                             "end": sg["end"] + range_offset} for sg in segments]
            full_transcript = " ".join(sg["text"].strip() for sg in segments).strip()
            timestamped_transcript = "\n".join(
                f"[{fmt_time(sg['start'])}] {sg['text'].strip()}"
                for sg in segments if sg["text"].strip()) or full_transcript
            srt_content = generate_srt(segments)

        # 3b. Cleanup
        if do_cleanup:
            status_container.write("🔧 Cleaning up transcript…")
            full_transcript, timestamped_transcript = cleanup_transcript(
                full_transcript, timestamped_transcript, segments,
                whisper_language_display, video_type, ai_engine, claude_key, gemini_key,
                openai_key=openai_key)
            status_container.write("   → Corrected ✅")

        # 4. Chapters
        status_container.write("📑 Generating chapters…")
        chapters = generate_chapters(timestamped_transcript, duration, video_type,
                                     ai_engine, claude_key, gemini_key, output_language,
                                     openai_key=openai_key)
        status_container.write(f"   → {len(chapters)} chapters")

        # 5. AI analysis
        engine_label = (
            "Gemini" if ai_engine == "Gemini (Free)"
            else "DeepSeek Flash" if ai_engine == DEEPSEEK_ENGINE
            else "OpenAI" if _is_oai(ai_engine)
            else "Claude"
        )
        _slide_note = (f" + {len(slide_images)} slide(s)" if slide_images else "")
        status_container.write(f"🤖 Analysing with {engine_label}{_slide_note}…")
        _yt_direct = st.session_state.pop("_gemini_youtube_url", None)
        if ai_engine == "Gemini (Free)":
            explanation = None
            if gemini_watch_video:
                try:
                    status_container.write(
                        "🎥 Gemini is watching the video "
                        + ("(YouTube link passed directly)…" if (_yt_direct and not _rng_window)
                           else "(uploading it to Google first)…"))
                    explanation = analyze_with_gemini_video(
                        video_path, timestamped_transcript, duration, gemini_key, video_type,
                        output_language, slide_images=slide_images, slide_sections=slide_sections,
                        youtube_url=None if _rng_window else _yt_direct)
                except Exception as _gv_err:
                    status_container.write(f"   → ⚠️ Video mode failed ({str(_gv_err)[:120]}); using frames instead.")
            if explanation is None:
                explanation = analyze_with_gemini(
                    timestamped_transcript, frames, frame_ts, duration, gemini_key, video_type,
                    output_language, slide_images=slide_images, slide_sections=slide_sections)
        elif _is_oai(ai_engine):
            explanation = analyze_with_openai(
                timestamped_transcript, frames, frame_ts, duration, openai_key, video_type,
                output_language, slide_images=slide_images, slide_sections=slide_sections, ai_engine=ai_engine)
        else:
            explanation = analyze_with_claude(
                timestamped_transcript, frames, frame_ts, duration, claude_key, video_type,
                output_language, slide_images=slide_images, slide_sections=slide_sections)

        analysis_time_sec = time.time() - _t0

        # 6. Auto-save + history
        base_name = os.path.splitext(video_name)[0]
        stamp     = datetime.now().strftime("%Y%m%d_%H%M%S")

        if auto_save:
            os.makedirs(save_folder, exist_ok=True)

            # ── Text files (always succeed) ───────────────────────────────
            text_files = [
                (f"{base_name}_{stamp}_explanation.md",
                 f"# {base_name}\n*Analysed on {datetime.now().strftime('%d %b %Y %H:%M')}*\n\n{explanation}"),
                (f"{base_name}_{stamp}_transcript.txt",              full_transcript),
                (f"{base_name}_{stamp}_transcript_timestamped.txt",  timestamped_transcript),
                (f"{base_name}_{stamp}_chapters.txt",                chapters_to_text(chapters)),
                (f"{base_name}_{stamp}_subtitles.srt",               srt_content),
            ]
            saved_fnames = []
            for path, content in text_files:
                if not content or not content.strip():
                    continue   # skip empty files — nothing useful to save
                with open(os.path.join(save_folder, path), "w", encoding="utf-8") as f:
                    f.write(content)
                saved_fnames.append(path)

            # ── Save captured frames as JPEGs ─────────────────────────────
            frames_dir = os.path.join(save_folder, f"{base_name}_{stamp}_frames")
            try:
                os.makedirs(frames_dir, exist_ok=True)
                for fi, (b64, ts) in enumerate(zip(frames, frame_ts)):
                    img_bytes = base64.b64decode(b64)
                    img_name  = f"frame_{fi+1:03d}_{int(ts):05d}s.jpg"
                    with open(os.path.join(frames_dir, img_name), "wb") as f:
                        f.write(img_bytes)
            except Exception as _frame_err:
                status_container.write(
                    f"⚠️ Frame images could not be saved to disk: {_frame_err}"
                )

            # ── History written NOW — before binary exports so a PDF/Word  ─
            # ── crash can never prevent the history entry from being saved. ─
            save_history_entry(save_folder, {
                "video_name":         video_name,
                "analyzed_at":        datetime.now().strftime("%d %b %Y, %H:%M"),
                "video_duration_sec": round(duration, 1),
                "analysis_time_sec":  round(analysis_time_sec, 0),
                "word_count":         len(full_transcript.split()),
                "chapter_count":      len(chapters),
                "frame_count":        len(frames),
                "frame_mode":         frame_mode,
                "auto_frames":        (max_frames is None),
                "stamp":              stamp,
                "save_folder":        save_folder,
                "frames_dir":         frames_dir if os.path.isdir(frames_dir) else "",
                "files":              saved_fnames,   # updated below if binaries succeed
            })

            # ── Binary files (Word / PDF) — wrapped so a font/encoding     ─
            # ── error doesn't crash the whole pipeline.                     ─
            binary_fnames = []
            for fname, producer in [
                (f"{base_name}_{stamp}.docx",
                 lambda: export_word(video_name, explanation, full_transcript,
                                     timestamped_transcript, chapters)),
                (f"{base_name}_{stamp}.pdf",
                 lambda: export_pdf(video_name, explanation, full_transcript,
                                    timestamped_transcript)),
            ]:
                try:
                    data = producer()
                    with open(os.path.join(save_folder, fname), "wb") as f:
                        f.write(data)
                    binary_fnames.append(fname)
                except Exception as bin_err:
                    status_container.warning(
                        f"⚠️ Could not save `{fname}`: {bin_err} "
                        f"(other files are saved safely)"
                    )

            # ── Patch the history entry with the binary filenames that     ─
            # ── actually succeeded.                                        ─
            if binary_fnames:
                records = []
                hpath   = _history_path(save_folder)
                with open(hpath, "r", encoding="utf-8") as f:
                    records = json.load(f)
                for rec in records:
                    if rec.get("stamp") == stamp:
                        rec["files"] = saved_fnames + binary_fnames
                        break
                with open(hpath, "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)

        # ── Generate AI title (runs whether or not auto_save is on) ──────
        status_container.write("🏷️ Generating AI title…")
        ai_title = generate_video_title(
            explanation, full_transcript, ai_engine, claude_key, gemini_key, openai_key)

        # Persist AI title into history entry if we auto-saved
        if auto_save and ai_title:
            hpath   = _history_path(save_folder)
            try:
                with open(hpath, "r", encoding="utf-8") as f:
                    records = json.load(f)
                for rec in records:
                    if rec.get("stamp") == stamp:
                        rec["ai_title"] = ai_title
                        break
                with open(hpath, "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # ── Generate AI tags ─────────────────────────────────────────────
        status_container.write("🏷️ Auto-tagging…")
        ai_tags = generate_tags(video_name, explanation, ai_engine, claude_key, gemini_key, openai_key)

        if auto_save and ai_tags:
            hpath = _history_path(save_folder)
            try:
                with open(hpath, "r", encoding="utf-8") as f:
                    records = json.load(f)
                for rec in records:
                    if rec.get("stamp") == stamp:
                        rec["tags"] = ai_tags
                        break
                with open(hpath, "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        return {
            "video_name":          video_name,
            "duration":            duration,
            "analysis_time_sec":   analysis_time_sec,
            "frames":              frames,
            "frame_ts":            frame_ts,
            "full_transcript":     full_transcript,
            "timestamped_transcript": timestamped_transcript,
            "srt_content":         srt_content,
            "chapters":            chapters,
            "explanation":         explanation,
            "ai_title":            ai_title,
            "tags":                ai_tags,
        }


def fetch_youtube_transcript(url: str, lang_code: str | None = None) -> dict | None:
    """
    Attempt to fetch the YouTube caption/transcript without downloading the video.

    Priority order:
      1. Manual captions in the preferred language
      2. Auto-generated captions in the preferred language
      3. Any manual captions (first available)
      4. Any auto-generated captions (first available)

    Returns a dict with keys:
        full_text, timestamped_text, srt_content, segments, language, is_generated
    or None if no transcript is available or the video has captions disabled.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
        import re

        # Extract video ID from any YouTube URL form
        vid_match = re.search(
            r'(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})', url)
        if not vid_match:
            return None
        vid_id = vid_match.group(1)

        # lang_code already resolved by the caller from WHISPER_LANGUAGES

        # youtube-transcript-api 0.x had list_transcripts(); 1.x moved it to an instance .list()
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            transcript_list = YouTubeTranscriptApi.list_transcripts(vid_id)
        else:
            transcript_list = YouTubeTranscriptApi().list(vid_id)

        # Build ordered list of candidates to try.  With no language chosen, prefer English
        # rather than whichever track YouTube lists first (dubbed videos list many languages).
        candidates = []
        for _want in ([lang_code] if lang_code else ["en"]):
            try:
                candidates.append(transcript_list.find_manually_created_transcript([_want]))
            except Exception:
                pass
            try:
                candidates.append(transcript_list.find_generated_transcript([_want]))
            except Exception:
                pass
        # Fallback: grab whatever is available
        for t in transcript_list:
            if t not in candidates:
                candidates.append(t)

        if not candidates:
            return None

        transcript = candidates[0]
        entries    = transcript.fetch()
        if hasattr(entries, "to_raw_data"):      # 1.x returns snippet objects, 0.x plain dicts
            entries = entries.to_raw_data()

        if not entries:
            return None

        # ── full plain text ───────────────────────────────────────────────
        full_text = " ".join(
            e.get("text", "").replace("\n", " ").strip() for e in entries
        )

        # ── timestamped text: [MM:SS] line per entry ──────────────────────
        ts_lines = []
        for e in entries:
            t_str = fmt_time(e.get("start", 0))
            text  = e.get("text", "").replace("\n", " ").strip()
            if text:
                ts_lines.append(f"[{t_str}] {text}")
        timestamped_text = "\n".join(ts_lines)

        # ── SRT ───────────────────────────────────────────────────────────
        def _srt_ts(sec: float) -> str:
            ms = int((sec % 1) * 1000)
            m, s = divmod(int(sec), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        srt_parts = []
        for i, e in enumerate(entries, 1):
            start = e.get("start", 0)
            end   = start + e.get("duration", 2)
            text  = e.get("text", "").replace("\n", " ").strip()
            if text:
                srt_parts.append(f"{i}\n{_srt_ts(start)} --> {_srt_ts(end)}\n{text}\n")
        srt_content = "\n".join(srt_parts)

        # ── Whisper-compatible segments list ──────────────────────────────
        segments = [
            {
                "start": e.get("start", 0),
                "end":   e.get("start", 0) + e.get("duration", 2),
                "text":  e.get("text", "").replace("\n", " "),
            }
            for e in entries
        ]

        return {
            "full_text":        full_text,
            "timestamped_text": timestamped_text,
            "srt_content":      srt_content,
            "segments":         segments,
            "language":         transcript.language,
            "language_code":    getattr(transcript, "language_code", ""),
            "is_generated":     transcript.is_generated,
        }

    except Exception:
        return None


def detect_platform(url: str) -> str:
    """
    Return a friendly platform name from a video URL.
    Falls back to the domain name if not in the known list.
    """
    url_lower = url.lower()
    known = [
        ("youtube.com", "YouTube"), ("youtu.be", "YouTube"),
        ("vimeo.com",   "Vimeo"),
        ("dailymotion.com", "Dailymotion"),
        ("facebook.com", "Facebook"), ("fb.watch", "Facebook"),
        ("instagram.com", "Instagram"),
        ("twitter.com", "Twitter / X"), ("x.com", "Twitter / X"),
        ("tiktok.com", "TikTok"),
        ("twitch.tv", "Twitch"),
        ("bilibili.com", "Bilibili"),
        ("rumble.com", "Rumble"),
        ("odysee.com", "Odysee"),
        ("ted.com", "TED"),
        ("linkedin.com", "LinkedIn"),
    ]
    for fragment, name in known:
        if fragment in url_lower:
            return name
    # Fallback: extract domain
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "Online"


def is_youtube_url(url: str) -> bool:
    """Return True only for YouTube / youtu.be links."""
    return any(h in url.lower() for h in ("youtube.com", "youtu.be"))


def _sanitize_title(raw_title: str, max_len: int = 120) -> str:
    """
    Produce a Windows-safe filename stem from any platform video title.
    - Removes characters illegal on Windows: \\ / : * ? " < > |
    - Collapses runs of whitespace / underscores
    - Strips leading/trailing dots and spaces (Windows rejects those)
    - Truncates to max_len characters
    """
    # Remove Windows-forbidden characters
    safe = re.sub(r'[\\/:*?"<>|]', '', raw_title)
    # Collapse emoji-adjacent junk and multiple spaces/underscores
    safe = re.sub(r'[\s_]+', ' ', safe).strip('. ')
    # Truncate
    if len(safe) > max_len:
        safe = safe[:max_len].rstrip('. ')
    # If nothing is left (title was all special chars), use a timestamp
    return safe or f"video_{int(time.time())}"


def download_online_video(url: str, out_dir: str) -> tuple:
    """
    Download a video from any yt-dlp-supported platform.
    Returns (video_path, sanitized_title, platform_name).

    Supported platforms: YouTube, Vimeo, Dailymotion, Facebook, Instagram,
    Twitter/X, TikTok, Twitch, Bilibili, Rumble, TED, and thousands more
    (anything yt-dlp handles).

    Key design: downloads to a safe fixed filename using the video ID so that
    platform titles containing emojis, special characters, or hundreds of
    characters (common on Facebook/TikTok/Instagram) never cause a Windows
    [Errno 22] / path-too-long error.  The human-readable sanitized title is
    derived separately from info["title"] for display and file naming.
    """
    import yt_dlp

    platform = detect_platform(url)

    # Step 1 — extract metadata only (no download) to get the video ID + title
    meta_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(meta_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    raw_title = info.get("title") or info.get("id") or "video"
    video_id  = info.get("id") or "vid"
    safe_title = _sanitize_title(raw_title)

    # Step 2 — download using the video ID as filename (always Windows-safe)
    safe_filename = re.sub(r'[^A-Za-z0-9_-]', '_', video_id)
    out_template  = os.path.join(out_dir, f"{safe_filename}.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_template,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    # Find the downloaded MP4 (should be safe_filename.mp4)
    for fname in os.listdir(out_dir):
        if fname.endswith(".mp4"):
            video_path = os.path.join(out_dir, fname)
            # Rename to the sanitized human title so downstream code is consistent
            final_path = os.path.join(out_dir, f"{safe_title}.mp4")
            try:
                os.rename(video_path, final_path)
                video_path = final_path
            except Exception:
                pass  # keep original name if rename fails
            return video_path, safe_title, platform

    raise FileNotFoundError(
        "yt-dlp finished but no MP4 was found. "
        "The URL may not be supported, or the video may be private / geo-blocked."
    )


# ── History helpers ───────────────────────────────────────────────────────────

HISTORY_FILE = "vidsage_history.json"

def _history_path(save_folder: str) -> str:
    return os.path.join(save_folder, HISTORY_FILE)


def save_history_entry(save_folder: str, entry: dict):
    """Append one analysis record to vidsage_history.json."""
    path = _history_path(save_folder)
    records = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []
    records.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_history(save_folder: str) -> list:
    """Load all history records, newest first."""
    path = _history_path(save_folder)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return list(reversed(json.load(f)))
    except Exception:
        return []


def delete_history_entry(save_folder: str, stamp: str,
                         delete_files: bool = True) -> tuple[bool, str]:
    """
    Remove a history record by stamp.
    If delete_files=True, also deletes all associated saved files and the
    _frames directory.  Returns (success, message).
    """
    hpath = _history_path(save_folder)
    if not os.path.exists(hpath):
        return False, "History file not found."
    try:
        with open(hpath, "r", encoding="utf-8") as fh:
            records = json.load(fh)

        target = next((r for r in records if r.get("stamp") == stamp), None)
        if not target:
            return False, "Entry not found in history."

        if delete_files:
            import shutil, glob as _glob
            folder = target.get("save_folder", save_folder)
            # Delete all associated files
            for fn in target.get("files", []):
                fpath = os.path.join(folder, fn)
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                except Exception:
                    pass
            # Delete frames directory — try 3 strategies so nothing is missed:
            base_name = os.path.splitext(target.get("video_name", ""))[0]
            _frames_candidates = set()
            # 1. Path stored in the JSON record
            if target.get("frames_dir"):
                _frames_candidates.add(target["frames_dir"])
            # 2. Constructed from base_name + stamp
            _frames_candidates.add(
                os.path.join(folder, f"{base_name}_{stamp}_frames"))
            # 3. Glob scan — catches any naming variant in the save folder
            for _d in _glob.glob(os.path.join(folder, f"{base_name}*_frames")):
                if os.path.isdir(_d):
                    _frames_candidates.add(_d)
            for _fd in _frames_candidates:
                if os.path.isdir(_fd):
                    try:
                        shutil.rmtree(_fd)
                    except Exception:
                        pass

        # Remove record and save
        records = [r for r in records if r.get("stamp") != stamp]
        with open(hpath, "w", encoding="utf-8") as fh:
            json.dump(records, fh, ensure_ascii=False, indent=2)

        return True, f"Deleted {'files and ' if delete_files else ''}history entry."
    except Exception as ex:
        return False, f"Delete failed: {ex}"


def _fmt_analysis_time(seconds: float) -> str:
    """Format analysis duration as e.g. '4m 32s'."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def load_history_entry_for_display(entry: dict, folder: str) -> dict | None:
    """
    Load all saved files for a history entry into a dict that can be passed
    directly to _show_results().  Returns None if the essential files
    (explanation + timestamped transcript) are missing.
    """
    stamp     = entry.get("stamp", "")
    base_name = os.path.splitext(entry["video_name"])[0]

    available = [
        fn for fn in entry.get("files", [])
        if os.path.exists(os.path.join(folder, fn))
    ]

    expl_file  = next((fn for fn in available if fn.endswith("_explanation.md")), None)
    trans_file = next((fn for fn in available
                       if fn.endswith("_transcript.txt") and "timestamped" not in fn), None)
    ts_file    = next((fn for fn in available if fn.endswith("_transcript_timestamped.txt")), None)
    srt_file   = next((fn for fn in available if fn.endswith(".srt")), None)
    chaps_file = next((fn for fn in available if fn.endswith("_chapters.txt")), None)

    if not expl_file or not ts_file:
        return None   # can't show useful results without these two

    # ── Explanation — strip the auto-save header lines ────────────────
    with open(os.path.join(folder, expl_file), "r", encoding="utf-8") as fh:
        raw_expl = fh.read()
    expl_lines = raw_expl.split("\n")
    explanation = "\n".join(
        ln for ln in expl_lines
        if not ln.startswith("*Analysed on") and ln.strip() != f"# {base_name}"
    ).strip()

    # ── Transcripts ───────────────────────────────────────────────────
    full_transcript = ""
    if trans_file:
        with open(os.path.join(folder, trans_file), "r", encoding="utf-8") as fh:
            full_transcript = fh.read()

    with open(os.path.join(folder, ts_file), "r", encoding="utf-8") as fh:
        timestamped_transcript = fh.read()

    srt_content = ""
    if srt_file:
        with open(os.path.join(folder, srt_file), "r", encoding="utf-8") as fh:
            srt_content = fh.read()

    # ── Chapters — parse from saved text file ─────────────────────────
    chapters = []
    if chaps_file:
        with open(os.path.join(folder, chaps_file), "r", encoding="utf-8") as fh:
            for line in fh:
                m = re.match(r"^(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–]\s*(.+)$", line.strip())
                if m:
                    chapters.append({"time": m.group(1), "title": m.group(2).strip()})

    # ── Frames — load JPEGs from the saved frames subfolder ───────────
    frames, frame_ts = [], []
    frames_dir = entry.get("frames_dir", "")
    if not frames_dir or not os.path.isdir(frames_dir):
        candidate = os.path.join(folder, f"{base_name}_{stamp}_frames")
        if os.path.isdir(candidate):
            frames_dir = candidate

    if frames_dir and os.path.isdir(frames_dir):
        for jpg in sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg")):
            try:
                ts_sec = float(jpg.split("_")[-1].replace("s.jpg", ""))
            except ValueError:
                ts_sec = 0.0
            with open(os.path.join(frames_dir, jpg), "rb") as fh:
                frames.append(base64.standard_b64encode(fh.read()).decode())
            frame_ts.append(ts_sec)

    return {
        "explanation":            explanation,
        "chapters":               chapters,
        "full_transcript":        full_transcript,
        "timestamped_transcript": timestamped_transcript,
        "srt_content":            srt_content,
        "frames":                 frames,
        "frame_ts":               frame_ts,
        "duration":               entry.get("video_duration_sec", 0),
    }


def scan_folder_for_history(save_folder: str) -> int:
    """
    Scan the save folder for existing result files and create history entries
    for any stamp that isn't already recorded in vidsage_history.json.
    Returns the number of new entries added.
    """
    if not os.path.exists(save_folder):
        return 0

    # Load existing stamps so we don't duplicate
    hpath = _history_path(save_folder)
    existing = set()
    if os.path.exists(hpath):
        try:
            with open(hpath, "r", encoding="utf-8") as f:
                existing = {r.get("stamp") for r in json.load(f)}
        except Exception:
            existing = set()

    # Group FILES by stamp (format: name_YYYYMMDD_HHMMSS_suffix.ext)
    # Directories (e.g. _frames folders) are intentionally excluded.
    stamp_files: dict = {}
    stamp_pattern = re.compile(r"^(.+?)_(\d{8}_\d{6})(_.*)?(\..+)$")
    for fname in os.listdir(save_folder):
        if not os.path.isfile(os.path.join(save_folder, fname)):
            continue   # skip _frames directories and any other subdirs
        m = stamp_pattern.match(fname)
        if not m:
            continue
        base, stamp = m.group(1), m.group(2)
        if stamp in existing:
            continue
        if stamp not in stamp_files:
            stamp_files[stamp] = {"base": base, "files": []}
        stamp_files[stamp]["files"].append(fname)

    added = 0
    for stamp, info in sorted(stamp_files.items()):
        base  = info["base"]
        files = info["files"]

        # Try to read metadata from saved files
        word_count, chapter_count, duration_sec = 0, 0, 0.0

        transcript_file = next(
            (f for f in files if f.endswith("_transcript.txt")
             and "timestamped" not in f), None)
        if transcript_file:
            try:
                with open(os.path.join(save_folder, transcript_file),
                          "r", encoding="utf-8") as f:
                    word_count = len(f.read().split())
            except Exception:
                pass

        chapters_file = next(
            (f for f in files if f.endswith("_chapters.txt")), None)
        if chapters_file:
            try:
                with open(os.path.join(save_folder, chapters_file),
                          "r", encoding="utf-8") as f:
                    chapter_count = sum(
                        1 for line in f if re.match(r"^\d{1,2}:\d{2}", line.strip()))
            except Exception:
                pass

        # Parse date from stamp
        try:
            dt = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
            analyzed_at = dt.strftime("%d %b %Y, %H:%M")
        except Exception:
            analyzed_at = stamp

        save_history_entry(save_folder, {
            "video_name":         base + ".mp4",
            "analyzed_at":        analyzed_at,
            "video_duration_sec": duration_sec,
            "analysis_time_sec":  0,
            "word_count":         word_count,
            "chapter_count":      chapter_count,
            "frame_count":        0,
            "frame_mode":         "—",
            "auto_frames":        False,
            "stamp":              stamp,
            "save_folder":        save_folder,
            "files":              files,
        })
        added += 1

    return added


def rename_history_entry(save_folder: str, stamp: str,
                          old_base: str, new_base: str) -> tuple[bool, str]:
    """
    Rename all files and the frames directory for a history entry, then update
    the history JSON.  Returns (success, message).

    Every file saved by VidSage follows the pattern:
        {old_base}_{stamp}{suffix}.{ext}
    After renaming it becomes:
        {new_base}_{stamp}{suffix}.{ext}
    The frames directory {old_base}_{stamp}_frames/ is renamed to
        {new_base}_{stamp}_frames/
    """
    hpath = _history_path(save_folder)
    try:
        # ── Load history ──────────────────────────────────────────────────
        with open(hpath, "r", encoding="utf-8") as f:
            records = json.load(f)

        target = next((r for r in records if r.get("stamp") == stamp), None)
        if target is None:
            return False, "History entry not found."

        prefix_old = f"{old_base}_{stamp}"
        prefix_new = f"{new_base}_{stamp}"

        renamed_files = []
        errors = []

        # ── Rename text / binary files ────────────────────────────────────
        for fname in list(target.get("files", [])):
            if not fname.startswith(prefix_old):
                renamed_files.append(fname)   # keep as-is (shouldn't happen)
                continue
            suffix = fname[len(prefix_old):]  # e.g. "_explanation.md"
            new_fname = prefix_new + suffix
            old_path = os.path.join(save_folder, fname)
            new_path = os.path.join(save_folder, new_fname)
            if os.path.exists(old_path):
                try:
                    os.rename(old_path, new_path)
                    renamed_files.append(new_fname)
                except Exception as err:
                    errors.append(f"{fname}: {err}")
                    renamed_files.append(fname)   # keep old name on failure
            else:
                renamed_files.append(new_fname)   # already gone — update name anyway

        # ── Also rename any stray FILES with the old prefix (PKM note etc.) ─
        # Directories (like _frames) are handled separately below.
        try:
            for fn in os.listdir(save_folder):
                if not os.path.isfile(os.path.join(save_folder, fn)):
                    continue   # skip _frames dirs and other subdirectories
                if fn.startswith(prefix_old) and fn not in target.get("files", []):
                    suffix = fn[len(prefix_old):]
                    new_fn = prefix_new + suffix
                    try:
                        os.rename(os.path.join(save_folder, fn),
                                  os.path.join(save_folder, new_fn))
                        if new_fn not in renamed_files:
                            renamed_files.append(new_fn)
                    except Exception:
                        pass
        except Exception:
            pass

        # ── Rename frames directory ───────────────────────────────────────
        old_frames = os.path.join(save_folder, f"{prefix_old}_frames")
        new_frames = os.path.join(save_folder, f"{prefix_new}_frames")
        frames_dir_new = target.get("frames_dir", "")
        if os.path.isdir(old_frames):
            try:
                os.rename(old_frames, new_frames)
                frames_dir_new = new_frames
            except Exception as err:
                errors.append(f"frames dir: {err}")

        # ── Update history record ─────────────────────────────────────────
        target["video_name"]  = new_base + ".mp4"
        target["files"]       = renamed_files
        target["frames_dir"]  = frames_dir_new

        with open(hpath, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        if errors:
            return True, f"Renamed with some issues: {'; '.join(errors)}"
        return True, f"Renamed to '{new_base}' successfully."

    except Exception as e:
        return False, f"Rename failed: {e}"


def _file_label(fname: str) -> tuple:
    """Return (button label, mime type) for a saved file."""
    if fname.endswith("_pkm_note.md"):
        return "🧠 PKM Note (.md)", "text/markdown"
    if fname.endswith("_explanation.md"):
        return "📄 Explanation (.md)", "text/markdown"
    if fname.endswith("_transcript_timestamped.txt"):
        return "🕐 Timestamped", "text/plain"
    if fname.endswith("_transcript.txt"):
        return "📝 Transcript", "text/plain"
    if fname.endswith("_chapters.txt"):
        return "📑 Chapters", "text/plain"
    if fname.endswith("_subtitles.srt") or fname.endswith(".srt"):
        return "💬 Subtitles (.srt)", "text/plain"
    if fname.endswith(".docx"):
        return "📘 Word (.docx)", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if fname.endswith(".pdf"):
        return "📕 PDF", "application/pdf"
    return "⬇️ File", "application/octet-stream"


# ── Main area ─────────────────────────────────────────────────────────────────
app_mode = st.radio(
    "Mode",
    ["🎥  Single Video", "📂  Batch (Multiple Videos)", "▶️  Online Video URL", "📋  History"],
    horizontal=True, label_visibility="collapsed"
)
# Normalise display label back to logic key
_mode_key = app_mode.split("  ", 1)[-1]   # strip the leading emoji

if _mode_key not in ("History", "Online Video URL"):
    uploaded_files = st.file_uploader(
        "Drop your video(s) here", type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=(_mode_key == "Batch (Multiple Videos)"),
        help="Video is processed locally — only frames & transcript are sent to the AI.",
    )
    # Normalise to list
    if uploaded_files is None:
        uploaded_files = []
    elif not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
    uploaded = uploaded_files[0] if len(uploaded_files) == 1 else None

    if uploaded_files and not active_key:
        st.warning("⚠️  Enter your API key in the sidebar to continue.")

    # Empty state — shown when nothing has been uploaded yet
    if not uploaded_files:
        st.markdown("""
<div class="vs-empty">
  <h3>Ready when you are</h3>
  <p>Upload an MP4, MOV, MKV or AVI above — or switch to <strong>Online Video URL</strong> mode to analyse YouTube, Vimeo, Dailymotion and more.<br>
  VidSage will transcribe the audio, detect scene changes, generate chapters and explain everything with AI.</p>
</div>
""", unsafe_allow_html=True)
else:
    # History or Online Video URL mode — no file uploader needed
    uploaded_files = []
    uploaded = None

# ── BATCH MODE ────────────────────────────────────────────────────────────────
if _mode_key == "Batch (Multiple Videos)" and uploaded_files and active_key:
    st.markdown(f"**{len(uploaded_files)} video(s) queued**")
    for f in uploaded_files:
        st.markdown(f"- {f.name}  `{f.size/1_048_576:.1f} MB`")

    if st.button("🚀 Analyse All Videos", type="primary"):
        batch_results = []
        overall = st.progress(0, text="Starting batch…")

        for idx, vid_file in enumerate(uploaded_files):
            st.markdown(f"---\n#### [{idx+1}/{len(uploaded_files)}] {vid_file.name}")
            with st.status(f"Processing {vid_file.name}…", expanded=True) as vstatus:
                try:
                    result = process_one_video(
                        vid_file, vid_file.name, vstatus,
                        frame_mode, max_frames,
                        frame_interval if frame_mode in ("📅 Fixed Interval", "🔍 Dense + Dedup") else 45,
                        scene_threshold if frame_mode in ("🧠 Smart Scene Detection", "🔍 Dense + Dedup") else 0.4,
                        min_words_seg,
                        whisper_model, whisper_lang_code, whisper_initial_prompt,
                        whisper_language_display, do_cleanup,
                        ai_engine, claude_key, gemini_key,
                        auto_save, save_folder,
                        output_language=output_language,
                        slide_images=st.session_state.get("slide_images"),
                        slide_sections=st.session_state.get("slide_sections"),
                        openai_key=openai_key,
                    )
                    batch_results.append({**result, "status": "✅ Done"})
                    vstatus.update(label=f"✅ {vid_file.name} complete", state="complete")
                except Exception as e:
                    batch_results.append({"video_name": vid_file.name,
                                          "status": f"❌ Error: {e}"})
                    vstatus.update(label=f"❌ {vid_file.name} failed", state="error")

            overall.progress((idx + 1) / len(uploaded_files),
                             text=f"Completed {idx+1} of {len(uploaded_files)}")

        # Batch summary table
        st.markdown("---")
        st.subheader("📊 Batch Summary")
        summary_rows = []
        for r in batch_results:
            summary_rows.append({
                "Video":    r["video_name"],
                "Duration": fmt_time(r.get("duration", 0)) if "duration" in r else "—",
                "Words":    f"{len(r.get('full_transcript','').split()):,}" if "full_transcript" in r else "—",
                "Chapters": len(r.get("chapters", [])) if "chapters" in r else "—",
                "Frames":   len(r.get("frames", [])) if "frames" in r else "—",
                "Status":   r["status"],
            })
        st.dataframe(pd.DataFrame(summary_rows))

        if auto_save:
            st.success(f"💾 All results saved to `{save_folder}`")

        # Show individual results expandable
        st.markdown("---")
        st.subheader("📄 Individual Results")
        for r in batch_results:
            if "explanation" not in r:
                continue
            with st.expander(f"📹 {r['video_name']}"):
                t1, t2, t3 = st.tabs(["Explanation", "Chapters", "Transcript"])
                with t1:
                    st.markdown(r["explanation"])
                with t2:
                    for c in r.get("chapters", []):
                        st.markdown(f"**`{c['time']}`** — {c['title']}")
                with t3:
                    st.text_area("Transcript", r["full_transcript"], height=300,
                                 key=f"batch_ts_{r['video_name']}")

def parse_video_urls(text: str) -> tuple:
    """Split pasted text (one link per line; commas/spaces also work) into (urls, ignored).
    Duplicates are dropped, order is kept, and only http(s) links are accepted."""
    urls, ignored, seen = [], [], set()
    for token in re.split(r"[\s,;]+", text or ""):
        token = token.strip()
        if not token:
            continue
        if not re.match(r"https?://", token, re.I):
            ignored.append(token)
        elif token not in seen:
            seen.add(token)
            urls.append(token)
    return urls, ignored


def _analyse_online_url(yt_url: str, yt_status) -> tuple:
    """Captions (YouTube) → download → full analysis for ONE online video.
    Returns (result, video_name, platform_name).  Used by both single and batch URL modes."""
    with tempfile.TemporaryDirectory() as yt_tmpdir:
        _plat = detect_platform(yt_url)

        # ── Step 1: try YouTube captions (YouTube only) ───
        yt_transcript = None
        if is_youtube_url(yt_url):
            yt_status.write("📝 Checking for YouTube captions…")
            yt_transcript = fetch_youtube_transcript(yt_url, whisper_lang_code)
            if yt_transcript:
                lang   = yt_transcript.get("language", "unknown")
                source = ("auto-generated"
                          if yt_transcript.get("is_generated")
                          else "manual")
                yt_status.write(
                    f"   → ✅ Found **{lang}** captions ({source}) — "
                    f"Whisper will be skipped"
                )
            else:
                yt_status.write("   → No captions found — Whisper will transcribe")
        else:
            yt_status.write(
                f"📝 **{_plat}** — captions not available via API; "
                f"Whisper will transcribe the audio"
            )

        # ── Step 2: download video (needed for frames) ────
        yt_status.write(f"⬇️ Downloading from **{_plat}**…")
        video_path, video_title, platform_name = download_online_video(yt_url, yt_tmpdir)
        file_size = os.path.getsize(video_path) / 1_048_576
        yt_status.write(
            f"   → **{video_title}** ({file_size:.1f} MB) from {platform_name} ✅"
        )

        with open(video_path, "rb") as f:
            video_bytes = f.read()

        video_name = f"{video_title}.mp4"

        # ── Step 3: analyse (Whisper skipped if transcript found) ──
        if is_youtube_url(yt_url):
            st.session_state["_gemini_youtube_url"] = yt_url      # lets Gemini watch it directly
        result = process_one_video(
            video_bytes, video_name, yt_status,
            frame_mode, max_frames,
            frame_interval if frame_mode in ("📅 Fixed Interval", "🔍 Dense + Dedup") else 45,
            scene_threshold if frame_mode in ("🧠 Smart Scene Detection", "🔍 Dense + Dedup") else 0.4,
            min_words_seg,
            whisper_model, whisper_lang_code, whisper_initial_prompt,
            whisper_language_display, do_cleanup,
            ai_engine, claude_key, gemini_key,
            auto_save, save_folder,
            prefetched_transcript=yt_transcript,
            output_language=output_language,
            slide_images=st.session_state.get("slide_images"),
            openai_key=openai_key,
        )
        return result, video_name, platform_name


# ── YOUTUBE MODE ──────────────────────────────────────────────────────────────
if _mode_key == "Online Video URL":
    if not active_key:
        st.warning("Enter your API key in the sidebar to continue.")
    else:
        st.caption(
            "Paste a link from **YouTube, Vimeo, Dailymotion, Facebook, Instagram, "
            "Twitter / X, TikTok, Twitch, Bilibili, Rumble, TED** or any other "
            "[yt-dlp supported platform](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md). "
            "Paste one link or several (one per line) for a batch. "
            "The video is downloaded locally — only frames & transcript are sent to the AI."
        )

        yt_urls_text = st.text_area(
            "Video URL(s) — one link per line",
            height=120,
            placeholder=("https://www.youtube.com/watch?v=...\n"
                         "https://vimeo.com/...\n"
                         "https://www.dailymotion.com/video/...   (add as many as you like)"),
            help="Paste one link for a single video, or several links (one per line) to analyse them as a batch.",
        )
        _urls, _ignored = parse_video_urls(yt_urls_text)
        if _ignored:
            st.warning("Ignored (not a http/https link): " + ", ".join(f"`{x[:40]}`" for x in _ignored[:5])
                       + (" …" if len(_ignored) > 5 else ""))
        if _urls and (st.session_state.get("range_start_txt") or st.session_state.get("range_end_txt")):
            st.caption("✂️ The sidebar **Time Range** applies to every video in this list.")

        # ── ONE link: same behaviour as before ───────────────────────────────
        if len(_urls) == 1:
            yt_url = _urls[0]
            st.info(f"🌐 Detected platform: **{detect_platform(yt_url)}**", icon="ℹ️")
            if st.button("🚀 Download & Analyse", type="primary"):
                with st.status("Working…", expanded=True) as yt_status:
                    try:
                        result, video_name, platform_name = _analyse_online_url(yt_url, yt_status)
                        yt_status.update(label="✅ Analysis complete!", state="complete")
                    except Exception as e:
                        yt_status.update(label=f"❌ Failed: {e}", state="error")
                        st.error(f"Error: {e}")
                        st.stop()

                if auto_save:
                    st.success(f"💾 Auto-saved to `{save_folder}`")

                # Store results + metadata
                st.session_state.pop("yt_batch", None)
                st.session_state["yt_result"]   = result
                st.session_state["yt_name"]     = video_name
                st.session_state["yt_platform"] = platform_name

        # ── SEVERAL links: batch ─────────────────────────────────────────────
        elif len(_urls) > 1:
            st.markdown(f"**{len(_urls)} videos queued**")
            st.dataframe(pd.DataFrame({"#": range(1, len(_urls) + 1),
                                       "Platform": [detect_platform(u) for u in _urls],
                                       "Link": _urls}), hide_index=True)
            if st.button(f"🚀 Download & Analyse All ({len(_urls)} videos)", type="primary"):
                st.session_state.pop("yt_result", None)
                _batch = []
                _overall = st.progress(0, text="Starting batch…")
                for _i, _u in enumerate(_urls):
                    st.markdown(f"---\n#### [{_i + 1}/{len(_urls)}] {_u}")
                    with st.status(f"Processing {_u}…", expanded=True) as _vstatus:
                        try:
                            _res, _vname, _plat_name = _analyse_online_url(_u, _vstatus)
                            _batch.append({**_res, "video_name": _vname, "url": _u,
                                           "platform": _plat_name, "status": "✅ Done"})
                            _vstatus.update(label=f"✅ {_vname} complete", state="complete")
                        except Exception as e:      # one bad link must not stop the batch
                            _batch.append({"video_name": _u, "url": _u, "platform": detect_platform(_u),
                                           "status": f"❌ Error: {e}"})
                            _vstatus.update(label=f"❌ Failed: {_u}", state="error")
                    _overall.progress((_i + 1) / len(_urls),
                                      text=f"Completed {_i + 1} of {len(_urls)}")
                st.session_state["yt_batch"] = _batch
                if auto_save:
                    st.success(f"💾 Results saved to `{save_folder}` — open the History tab for the full view and exports.")

        # Show batch results if available
        if st.session_state.get("yt_batch"):
            _b = st.session_state["yt_batch"]
            st.markdown("---")
            st.subheader("📊 Batch Summary")
            st.dataframe(pd.DataFrame([{
                "Video":    r["video_name"],
                "Platform": r.get("platform", ""),
                "Duration": fmt_time(r.get("duration", 0)) if "duration" in r else "—",
                "Words":    f"{len(r.get('full_transcript', '').split()):,}" if "full_transcript" in r else "—",
                "Chapters": str(len(r.get("chapters", []))) if "chapters" in r else "—",
                "Status":   r["status"],
            } for r in _b]), hide_index=True)
            _failed = [r for r in _b if "explanation" not in r]
            if _failed:
                st.error(f"{len(_failed)} video(s) failed — copy their links back into the box and re-run:\n\n"
                         + "\n".join(f"- {r['url']}" for r in _failed))
            _done = [r for r in _b if "explanation" in r]
            if _done:
                _all_md = "\n\n---\n\n".join(
                    f"# {r['video_name']}\nSource: {r['url']}\n\n{r['explanation']}" for r in _done)
                st.download_button("⬇️ Download all explanations (.md)", _all_md,
                                   file_name="vidsage_batch_explanations.md", mime="text/markdown")
            st.subheader("📄 Individual Results")
            for _n, r in enumerate(_done):
                with st.expander(f"📹 {r['video_name']}"):
                    st.caption(f"🌐 {r.get('platform', '')} · {r['url']}")
                    t1, t2, t3 = st.tabs(["Explanation", "Chapters", "Transcript"])
                    with t1:
                        st.markdown(r["explanation"])
                    with t2:
                        for c in r.get("chapters", []):
                            st.markdown(f"**`{c['time']}`** — {c['title']}")
                    with t3:
                        st.text_area("Transcript", r["full_transcript"], height=300,
                                     key=f"ytbatch_ts_{_n}")

        # Show results if available
        if "yt_result" in st.session_state and st.session_state.get("yt_name"):
            r              = st.session_state["yt_result"]
            uploaded_name  = st.session_state["yt_name"]
            _platform      = st.session_state.get("yt_platform", "Online")
            frames                 = r["frames"]
            frame_ts               = r["frame_ts"]
            full_transcript        = r["full_transcript"]
            timestamped_transcript = r["timestamped_transcript"]
            srt_content            = r["srt_content"]
            chapters               = r["chapters"]
            explanation            = r["explanation"]
            duration               = r.get("duration", 0)

            st.markdown("---")
            _ai_t = r.get("ai_title", "")
            st.markdown(f"### 📹 {_ai_t if _ai_t else uploaded_name}")
            st.caption(
                f"🌐 Source: **{_platform}**"
                + (f"  ·  💡 AI title: **{_ai_t}** — rename in History tab" if _ai_t else "")
            )
            _show_results(uploaded_name, explanation, chapters, full_transcript,
                          timestamped_transcript, srt_content, frames, frame_ts,
                          ai_engine, claude_key, gemini_key, duration, save_folder)

# ── SINGLE VIDEO MODE ─────────────────────────────────────────────────────────
if _mode_key == "Single Video" and uploaded and active_key:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.video(uploaded)
    with col2:
        st.metric("File size", f"{uploaded.size / 1_048_576:.1f} MB")
        run = st.button("🚀 Analyze Video", type="primary", width='stretch')

    if run:
        with st.status("Working…", expanded=True) as status:
            result = process_one_video(
                uploaded, uploaded.name, status,
                frame_mode, max_frames,
                frame_interval if frame_mode in ("📅 Fixed Interval", "🔍 Dense + Dedup") else 45,
                scene_threshold if frame_mode in ("🧠 Smart Scene Detection", "🔍 Dense + Dedup") else 0.4,
                min_words_seg,
                whisper_model, whisper_lang_code, whisper_initial_prompt,
                whisper_language_display, do_cleanup,
                ai_engine, claude_key, gemini_key,
                auto_save, save_folder,
                output_language=output_language,
                slide_images=st.session_state.get("slide_images"),
                openai_key=openai_key,
            )
            frames            = result["frames"]
            frame_ts          = result["frame_ts"]
            full_transcript   = result["full_transcript"]
            timestamped_transcript = result["timestamped_transcript"]
            srt_content       = result["srt_content"]
            chapters          = result["chapters"]
            explanation       = result["explanation"]
            duration          = result["duration"]
            status.update(label="✅ Analysis complete!", state="complete")

        if auto_save:
            st.success(f"💾 Auto-saved to `{save_folder}` (.md, .docx, .pdf, .srt, .txt)")

        st.markdown("---")
        _ai_t = result.get("ai_title", "")
        st.markdown(f"### 📹 {_ai_t if _ai_t else uploaded.name}")
        if _ai_t:
            st.caption(
                f"💡 AI-suggested title: **{_ai_t}** "
                f"— you can rename files in the History tab."
            )
        _show_results(uploaded.name, explanation, chapters, full_transcript,
                      timestamped_transcript, srt_content, frames, frame_ts,
                      ai_engine, claude_key, gemini_key, duration, save_folder)

# ── HISTORY MODE ──────────────────────────────────────────────────────────────
if _mode_key == "History":
    st.markdown("## 📋 Analysis History")

    history = load_history(save_folder)

    # ── Scan button — always visible so existing files can be imported ───────
    scan_col, info_col = st.columns([2, 5])
    with scan_col:
        if st.button("🔍 Scan folder for existing results", width='stretch'):
            added = scan_folder_for_history(save_folder)
            if added:
                st.success(f"✅ Found and imported **{added}** past analysis run(s). Refreshing…")
                st.rerun()
            else:
                st.info("No new results found — history is already up to date.")
    with info_col:
        st.caption(
            "Use this if you have existing result files from before the History feature "
            "was added, or after a crash. VidSage will read the saved files and rebuild "
            "the history index automatically."
        )

    st.markdown("---")

    if not history:
        st.markdown("""
<div class="vs-empty">
  <h3>No history yet</h3>
  <p>Once you analyse a video with <strong>Auto-Save</strong> turned on, every run will be recorded here.<br>
  You'll be able to re-read explanations and download all files without re-running the AI.<br><br>
  Already have saved files? Click <strong>Scan folder</strong> above.</p>
</div>
""", unsafe_allow_html=True)
    else:
        # ── Aggregate stats ───────────────────────────────────────────────────
        total_videos   = len(history)
        total_words    = sum(e.get("word_count", 0) for e in history)
        total_vid_sec  = sum(e.get("video_duration_sec", 0) for e in history)
        total_ai_sec   = sum(e.get("analysis_time_sec", 0) for e in history)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("📹 Videos Analysed", total_videos)
        s2.metric("📝 Total Words",      f"{total_words:,}")
        s3.metric("⏱️ Total Video Time", fmt_time(total_vid_sec))
        s4.metric("🤖 Total AI Time",    _fmt_analysis_time(total_ai_sec))

        st.markdown("---")

        # ── Tag filter bar ────────────────────────────────────────────────────
        _all_tags = sorted(set(tag for e in history for tag in e.get("tags", [])))
        if _all_tags:
            # Build colored tag label for multiselect display
            _sel_tags = st.multiselect(
                "🏷️ Filter by tag:",
                options=_all_tags,
                default=[t for t in st.session_state.get("hist_sel_tags", [])
                         if t in _all_tags],
                key="hist_tag_ms",
                placeholder="Select tags to filter…",
            )
            st.session_state["hist_sel_tags"] = _sel_tags
            if _sel_tags:
                history = [e for e in history
                           if any(t in e.get("tags", []) for t in _sel_tags)]
                _tag_labels = " ".join(
                    f'<span style="background:{_tag_colour(t)};color:#fff;padding:2px 10px;'
                    f'border-radius:12px;font-size:12px">{t}</span>'
                    for t in _sel_tags
                )
                st.markdown(
                    f"Showing **{len(history)}** of **{total_videos}** videos &nbsp;·&nbsp; {_tag_labels}",
                    unsafe_allow_html=True,
                )
        st.markdown("")

        # ── Summary table ─────────────────────────────────────────────────────
        st.subheader("📊 Summary Table")
        st.caption(
            "**Double-click a video title** to rename it (renames all related files).  "
            "**Check the ☑ box** to select rows (delete / tag). **Tick 📖 Open** to read a video's full report."
        )

        # Build rows — keep original titles separately so we can detect edits
        # Selection and "open" state survive the table being reset (the key changes when a report is opened/closed)
        _hist_sel_set  = set(st.session_state.get("_hist_sel_stamps", []))
        _hist_open_now = st.session_state.get("_hist_open")
        _editor_ver    = st.session_state.get("_hist_editor_ver", 0)
        rows = []
        for e in history:
            rows.append({
                "☑":          e.get("stamp") in _hist_sel_set,
                "📖 Open":    e.get("stamp") == _hist_open_now,
                "Video":      os.path.splitext(e["video_name"])[0],
                "Tags":       ", ".join(e.get("tags", [])),
                "Analysed":   e["analyzed_at"],
                "Duration":   fmt_time(e.get("video_duration_sec", 0)),
                "AI Time":    _fmt_analysis_time(e.get("analysis_time_sec", 0)),
                "Words":      f"{e.get('word_count', 0):,}",
                "Chapters":   e.get("chapter_count", 0),
                "Frames":     e.get("frame_count", 0),
                "Frame Mode": e.get("frame_mode", "—").split(" ", 1)[-1],
            })

        _orig_titles = [os.path.splitext(e["video_name"])[0] for e in history]

        _edited_df = st.data_editor(
            pd.DataFrame(rows),
            column_config={
                "☑":          st.column_config.CheckboxColumn(
                                  "☑",
                                  help="Check to select this row (for delete / tagging).",
                                  width="small",
                              ),
                "📖 Open":    st.column_config.CheckboxColumn(
                                  "📖 Open",
                                  help="Tick to read this video's full report below the table "
                                       "(one report at a time). Untick to close it.",
                                  width="small",
                              ),
                "Video":      st.column_config.TextColumn(
                                  "📹 Video  (double-click to rename)",
                                  help="Double-click a title to rename it. "
                                       "All saved files will be renamed automatically.",
                                  max_chars=120,
                              ),
                "Tags":       st.column_config.TextColumn(
                                  "🏷️ Tags  (double-click to edit)",
                                  help="Comma-separated. e.g: memory, training",
                                  width="medium",
                              ),
                "Analysed":   st.column_config.TextColumn(disabled=True),
                "Duration":   st.column_config.TextColumn(disabled=True),
                "AI Time":    st.column_config.TextColumn(disabled=True),
                "Words":      st.column_config.TextColumn(disabled=True),
                "Chapters":   st.column_config.NumberColumn(disabled=True),
                "Frames":     st.column_config.NumberColumn(disabled=True),
                "Frame Mode": st.column_config.TextColumn(disabled=True),
            },
            disabled=["Analysed", "Duration", "AI Time", "Words",
                      "Chapters", "Frames", "Frame Mode"],
            hide_index=True,
            key=f"history_table_edit_{_editor_ver}",
            width="stretch",
        )

        # ── Detect inline renames ─────────────────────────────────────────────
        if _edited_df is not None:
            for _ri, (_new_title, _old_title, _entry) in enumerate(
                    zip(_edited_df["Video"], _orig_titles, history)):
                _new_title = (_new_title or "").strip()
                if _new_title and _new_title != _old_title:
                    _bad_chars = [c for c in _new_title if c in r'\/:*?"<>|']
                    if _bad_chars:
                        st.error(
                            f"❌ Title contains invalid characters: "
                            f"{' '.join(set(_bad_chars))}"
                        )
                    else:
                        _entry_folder = _entry.get("save_folder", save_folder)
                        _entry_stamp  = _entry.get("stamp", "")
                        with st.spinner(f'Renaming to "{_new_title}"…'):
                            _ok, _msg = rename_history_entry(
                                _entry_folder, _entry_stamp,
                                _old_title, _new_title,
                            )
                        if _ok:
                            st.success(f"✅ {_msg}")
                            st.rerun()
                        else:
                            st.error(f"❌ {_msg}")

        # ── Detect inline tag edits — SILENT save, no rerun (avoids flash) ─────
        _orig_tags = [", ".join(e.get("tags", [])) for e in history]
        if _edited_df is not None and "Tags" in _edited_df.columns:
            for _ri, (_ntags, _otags, _entry) in enumerate(
                    zip(_edited_df["Tags"], _orig_tags, history)):
                _ntags = (_ntags or "").strip()
                if _ntags != _otags:
                    _parsed = [t.strip().lower() for t in _ntags.split(",") if t.strip()][:6]
                    try:
                        _hp3 = _history_path(_entry.get("save_folder", save_folder))
                        with open(_hp3, "r", encoding="utf-8") as _f3:
                            _r3 = json.load(_f3)
                        for _rec3 in _r3:
                            if _rec3.get("stamp") == _entry.get("stamp"):
                                _rec3["tags"] = _parsed
                                break
                        with open(_hp3, "w", encoding="utf-8") as _f3:
                            json.dump(_r3, _f3, ensure_ascii=False, indent=2)
                        # NO st.rerun() — data_editor already shows updated value
                    except Exception as _te:
                        st.error(f"Tag save failed: {_te}")

        # ── Detect row selection via checkbox ─────────────────────────────────
        _sel_rows = (
            _edited_df.index[_edited_df["☑"] == True].tolist()
            if _edited_df is not None and "☑" in _edited_df.columns
            else []
        )
        _sel_entries = [history[i] for i in _sel_rows if i < len(history)]
        st.session_state["_hist_sel_stamps"] = [e.get("stamp") for e in _sel_entries]

        # ── Detect "Open" ticks: one report at a time ─────────────────────────
        if _edited_df is not None and "📖 Open" in _edited_df.columns:
            _open_rows = [i for i in _edited_df.index[_edited_df["📖 Open"] == True].tolist() if i < len(history)]
            _cur_idx = next((i for i, e in enumerate(history) if e.get("stamp") == _hist_open_now), None)
            _newly = [i for i in _open_rows if i != _cur_idx]
            if _newly:                                   # a different row was ticked → open it
                st.session_state["_hist_open"] = history[_newly[0]].get("stamp", "")
                st.session_state["_hist_editor_ver"] = _editor_ver + 1
                st.rerun()
            elif _cur_idx is not None and _cur_idx not in _open_rows:   # the open row was unticked → close
                st.session_state.pop("_hist_open", None)
                st.session_state["_hist_editor_ver"] = _editor_ver + 1
                st.rerun()

        # ── Row-level actions ─────────────────────────────────────────────────
        _act_col1, _act_col2, _act_col3, _act_col4 = st.columns([2, 2, 2, 2])
        with _act_col1:
            if st.button("📂 Open Save Folder"):
                os.startfile(save_folder)
        _untagged = [e for e in history if not e.get("tags")]
        _auto_targets = None
        with _act_col3:
            if st.button(f"🤖 Auto-tag selected ({len(_sel_entries)})" if _sel_entries else "🤖 Auto-tag selected",
                         disabled=not _sel_entries, key="autotag_sel",
                         help="The AI reads each selected video's saved explanation and adds 2–4 topic tags."):
                _auto_targets = _sel_entries
        with _act_col4:
            if st.button(f"🤖 Auto-tag all untagged ({len(_untagged)})", disabled=not _untagged,
                         key="autotag_all",
                         help="Tag every video that has no tags yet, using the AI engine chosen in the sidebar."):
                _auto_targets = _untagged
        if _auto_targets:
            if not active_key:
                st.error("Add your API key for the selected AI engine (sidebar → AI engine) to auto-tag.")
            else:
                _bar = st.progress(0.0, text="Auto-tagging…")
                _ok, _bad = auto_tag_history_entries(
                    _auto_targets, save_folder, ai_engine, claude_key, gemini_key, openai_key,
                    progress=lambda f, n: _bar.progress(f, text=f"Tagged {n}"))
                _bar.empty()
                st.session_state["_autotag_msg"] = (
                    f"🤖 Auto-tagged {_ok} video(s)" + (f" · {_bad} could not be tagged" if _bad else ""))
                st.rerun()
        if st.session_state.get("_autotag_msg"):
            st.success(st.session_state.pop("_autotag_msg"))
        with _act_col2:
            _del_disabled = len(_sel_entries) == 0
            _del_label = (
                f"🗑️ Delete Selected ({len(_sel_entries)})"
                if _sel_entries else "🗑️ Delete Selected"
            )
            if st.button(_del_label,
                         disabled=_del_disabled,
                         help="Check rows to select them, then click here to delete"):
                st.session_state["_confirm_delete_stamps"] = [
                    e["stamp"] for e in _sel_entries
                ]

        # ── Quick Tag / Remove Tag (selected rows) ───────────────────────────
        if _sel_entries:
            st.markdown("**🏷️ Tag selected videos:**")
            _qt_col1, _qt_col2, _qt_col3 = st.columns([3, 2, 1])
            with _qt_col1:
                # Pick from existing tags
                _qt_existing = st.multiselect(
                    "Existing tags",
                    options=_all_tags,
                    default=[],
                    key="quick_tag_ms",
                    label_visibility="collapsed",
                    placeholder="Pick existing tags…" if _all_tags else "No tags yet — type a new one →",
                )
            with _qt_col2:
                # Type a brand new tag
                _qt_new_raw = st.text_input(
                    "New tag",
                    value="",
                    key="quick_tag_new",
                    label_visibility="collapsed",
                    placeholder="Type a new tag…",
                )
            with _qt_col3:
                _qt_all = list(dict.fromkeys(
                    _qt_existing +
                    [t.strip().lower() for t in _qt_new_raw.split(",") if t.strip()]
                ))
                if st.button("✅ Apply", key="quick_tag_apply",
                             disabled=not _qt_all, width='stretch'):
                    for _qe in _sel_entries:
                        _qfolder = _qe.get("save_folder", save_folder)
                        _qstamp  = _qe.get("stamp", "")
                        try:
                            _qhp = _history_path(_qfolder)
                            with open(_qhp, "r", encoding="utf-8") as _qf:
                                _qrecs = json.load(_qf)
                            for _qr in _qrecs:
                                if _qr.get("stamp") == _qstamp:
                                    _existing = _qr.get("tags", [])
                                    _merged   = list(dict.fromkeys(_existing + _qt_all))[:6]
                                    _qr["tags"] = _merged
                                    break
                            with open(_qhp, "w", encoding="utf-8") as _qf:
                                json.dump(_qrecs, _qf, ensure_ascii=False, indent=2)
                        except Exception:
                            pass
                    st.rerun()

            # Remove tags from selected rows
            _sel_current_tags = sorted(set(
                t for _qe in _sel_entries for t in _qe.get("tags", [])
            ))
            if _sel_current_tags:
                st.markdown("**🗑️ Remove tags from selected:**")
                _qr_col1, _qr_col2 = st.columns([5, 1])
                with _qr_col1:
                    _qt_remove = st.multiselect(
                        "Remove tags",
                        options=_sel_current_tags,
                        default=[],
                        key="quick_tag_remove_ms",
                        label_visibility="collapsed",
                        placeholder="Select tags to remove…",
                    )
                with _qr_col2:
                    if st.button("🗑️ Remove", key="quick_tag_remove_btn",
                                 disabled=not _qt_remove, width='stretch'):
                        for _qe in _sel_entries:
                            _qfolder = _qe.get("save_folder", save_folder)
                            _qstamp  = _qe.get("stamp", "")
                            try:
                                _qhp = _history_path(_qfolder)
                                with open(_qhp, "r", encoding="utf-8") as _qf:
                                    _qrecs = json.load(_qf)
                                for _qr in _qrecs:
                                    if _qr.get("stamp") == _qstamp:
                                        _qr["tags"] = [t for t in _qr.get("tags", [])
                                                       if t not in _qt_remove]
                                        break
                                with open(_qhp, "w", encoding="utf-8") as _qf:
                                    json.dump(_qrecs, _qf, ensure_ascii=False, indent=2)
                            except Exception:
                                pass
                        st.rerun()

        # ── Delete confirmation dialog ────────────────────────────────────────
        if st.session_state.get("_confirm_delete_stamps"):
            _del_stamps   = st.session_state["_confirm_delete_stamps"]
            _del_targets  = [e for e in history if e.get("stamp") in _del_stamps]
            if _del_targets:
                _names_list = "\n".join(
                    f"- **{t['video_name']}** *(analysed {t['analyzed_at']})*"
                    for t in _del_targets
                )
                st.warning(
                    f"⚠️ Delete {len(_del_targets)} video(s)?  \n"
                    f"{_names_list}  \n\n"
                    "This will remove all saved files for these entries."
                )
                _dc1, _dc2, _dc3 = st.columns([2, 2, 4])
                with _dc1:
                    if st.button("✅ Yes, delete files + records",
                                 key="confirm_del_files"):
                        _results = []
                        for _ds in _del_stamps:
                            _ok, _msg = delete_history_entry(
                                save_folder, _ds, delete_files=True)
                            _results.append((_ok, _msg))
                        st.session_state.pop("_confirm_delete_stamps", None)
                        _failed = [m for ok, m in _results if not ok]
                        _passed = [m for ok, m in _results if ok]
                        if _passed:
                            st.success(f"✅ Deleted {len(_passed)} entr{'y' if len(_passed)==1 else 'ies'}.")
                        if _failed:
                            st.error("Some deletions failed: " + "; ".join(_failed))
                        st.rerun()
                with _dc2:
                    if st.button("🗂️ Remove records only (keep files)",
                                 key="confirm_del_record"):
                        for _ds in _del_stamps:
                            delete_history_entry(save_folder, _ds, delete_files=False)
                        st.session_state.pop("_confirm_delete_stamps", None)
                        st.success(f"✅ Removed {len(_del_stamps)} record(s) from history.")
                        st.rerun()
                with _dc3:
                    if st.button("❌ Cancel", key="cancel_del"):
                        st.session_state.pop("_confirm_delete_stamps", None)
                        st.rerun()

        st.markdown("---")

        # ── Per-video expandable cards ────────────────────────────────────────
        _open_stamp = st.session_state.get("_hist_open")
        _open_entries = [e for e in history if e.get("stamp") == _open_stamp]
        if not _open_entries:
            st.caption("📖 Tick **Open** on a row of the table above to read that video's full report here.")
        else:
            st.subheader("📄 Report")

        # Only the opened report is drawn. Drawing every report on every click made this page take ~10 s
        # per click with 40 saved videos.
        for e in _open_entries:
            vid_label = f"📹 {e['video_name']}   ·   {e['analyzed_at']}"
            with st.expander(vid_label, expanded=True):
                if st.button("✖ Close this report", key=f"hist_close_{e.get('stamp', '')}"):
                    st.session_state.pop("_hist_open", None)
                    st.session_state["_hist_editor_ver"] = st.session_state.get("_hist_editor_ver", 0) + 1
                    st.rerun()

                folder    = e.get("save_folder", save_folder)
                stamp     = e.get("stamp", "")
                base_name = os.path.splitext(e["video_name"])[0]

                # ── Persistent re-analysis completion banner ──────────────
                _done_key = f"_hist_reanalysis_done_{stamp}"
                _warn_key = f"_hist_reanalysis_warn_{stamp}"
                if _done_key in st.session_state:
                    st.success(st.session_state.pop(_done_key))
                if _warn_key in st.session_state:
                    st.warning(st.session_state.pop(_warn_key))

                # ── Rename / AI title ─────────────────────────────────────
                _rename_key    = f"rename_mode_{stamp}"
                _ai_title_key  = f"ai_title_draft_{stamp}"
                ai_title_saved = e.get("ai_title", "")

                # Pre-populate draft with saved AI title on first render
                if _ai_title_key not in st.session_state:
                    st.session_state[_ai_title_key] = ai_title_saved or base_name

                ren_col1, ren_col2, ren_col3 = st.columns([5, 1, 1])
                with ren_col1:
                    _disp = (f"💡 AI title: **{ai_title_saved}**"
                             if ai_title_saved and ai_title_saved != base_name
                             else f"📁 `{base_name}`")
                    st.caption(_disp)
                with ren_col2:
                    if st.button("✏️ Rename", key=f"btn_rename_{stamp}",
                                 width='stretch'):
                        st.session_state[_rename_key] = not st.session_state.get(
                            _rename_key, False)
                with ren_col3:
                    # Regenerate AI title on-demand (outside the form)
                    if st.button("✨ AI Title", key=f"btn_ai_title_{stamp}",
                                 width='stretch'):
                        with st.spinner("Generating title…"):
                            # Load explanation from disk to generate title
                            _expl_f = next(
                                (fn for fn in e.get("files", [])
                                 if fn.endswith("_explanation.md")), None)
                            _trans_f = next(
                                (fn for fn in e.get("files", [])
                                 if fn.endswith("_transcript.txt")
                                 and "timestamped" not in fn), None)
                            _expl_txt, _trans_txt = "", ""
                            if _expl_f and os.path.exists(os.path.join(folder, _expl_f)):
                                with open(os.path.join(folder, _expl_f),
                                          "r", encoding="utf-8") as _fh:
                                    _expl_txt = _fh.read()
                            if _trans_f and os.path.exists(os.path.join(folder, _trans_f)):
                                with open(os.path.join(folder, _trans_f),
                                          "r", encoding="utf-8") as _fh:
                                    _trans_txt = _fh.read()
                            new_ai = generate_video_title(
                                _expl_txt, _trans_txt,
                                ai_engine, claude_key, gemini_key, openai_key)
                            if new_ai:
                                st.session_state[_ai_title_key] = new_ai
                                # Persist to history JSON
                                try:
                                    _hp = _history_path(folder)
                                    with open(_hp, "r", encoding="utf-8") as _fh:
                                        _recs = json.load(_fh)
                                    for _r in _recs:
                                        if _r.get("stamp") == stamp:
                                            _r["ai_title"] = new_ai
                                            break
                                    with open(_hp, "w", encoding="utf-8") as _fh:
                                        json.dump(_recs, _fh,
                                                  ensure_ascii=False, indent=2)
                                except Exception:
                                    pass
                                st.session_state[_rename_key] = True  # open form
                                st.rerun()
                            else:
                                st.warning("Could not generate title — check your API key.")

                if st.session_state.get(_rename_key, False):
                    with st.form(key=f"rename_form_{stamp}"):
                        new_title = st.text_input(
                            "New title (no extension — edit the AI suggestion or type your own)",
                            value=st.session_state.get(_ai_title_key, base_name),
                            key=f"ti_rename_{stamp}",
                        )
                        submitted = st.form_submit_button("💾 Apply rename")
                        if submitted:
                            new_title = new_title.strip()
                            _bad = set(r'\/:*?"<>|')
                            _invalid = [c for c in new_title if c in _bad]
                            if not new_title:
                                st.error("Title cannot be empty.")
                            elif _invalid:
                                st.error(
                                    f"Title contains invalid characters: "
                                    f"{' '.join(_invalid)}"
                                )
                            elif new_title == base_name:
                                st.info("Title unchanged.")
                                st.session_state[_rename_key] = False
                            else:
                                ok, msg = rename_history_entry(
                                    folder, stamp, base_name, new_title)
                                if ok:
                                    st.session_state[_rename_key] = False
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)

                # ── Tags display + AI re-tag ──────────────────────────────
                _entry_tags = e.get("tags", [])
                _tag_left, _tag_right = st.columns([6, 1])
                with _tag_left:
                    if _entry_tags:
                        _tag_html = " ".join(
                            f'<span style="background:{_tag_colour(t)};color:#fff;'
                            f'padding:3px 11px;border-radius:14px;font-size:12px;'
                            f'margin-right:4px">{t}</span>'
                            for t in _entry_tags
                        )
                        st.markdown(_tag_html, unsafe_allow_html=True)
                    else:
                        st.caption("No tags — edit in Summary Table above, or click ✨ AI Tag.")
                with _tag_right:
                    if st.button("✨ AI Tag", key=f"btn_retag_{stamp}", width='stretch',
                                 help="Auto-generate tags from the saved explanation"):
                        _expl_f2 = next(
                            (fn for fn in e.get("files", [])
                             if fn.endswith("_explanation.md")), None)
                        _expl_t2 = ""
                        if _expl_f2 and os.path.exists(os.path.join(folder, _expl_f2)):
                            with open(os.path.join(folder, _expl_f2),
                                      "r", encoding="utf-8") as _fh3:
                                _expl_t2 = _fh3.read()
                        with st.spinner("Generating tags…"):
                            _new_ai_tags = generate_tags(
                                e["video_name"], _expl_t2,
                                ai_engine, claude_key, gemini_key, openai_key)
                        if _new_ai_tags:
                            try:
                                _hp2 = _history_path(folder)
                                with open(_hp2, "r", encoding="utf-8") as _fh2:
                                    _recs2 = json.load(_fh2)
                                for _r2 in _recs2:
                                    if _r2.get("stamp") == stamp:
                                        _r2["tags"] = _new_ai_tags
                                        break
                                with open(_hp2, "w", encoding="utf-8") as _fh2:
                                    json.dump(_recs2, _fh2, ensure_ascii=False, indent=2)
                                st.rerun()
                            except Exception as _te:
                                st.error(f"Could not save tags: {_te}")
                        else:
                            st.warning("Could not generate tags — check your API key.")

                st.markdown("")

                # Skip isfile per-file — trust the JSON list; filter out _frames dirs only
                available = [fn for fn in e.get("files", [])
                             if not fn.endswith("_frames")]

                # ── Quick download buttons ────────────────────────────────
                if available:
                    st.markdown("**📥 Download files:**")
                    btn_cols = st.columns(min(len(available), 4))

                    # Cache file contents in session_state — read once per session
                    # avoids reading 3 × 42 = 126 files on every render
                    _hcache_key = f"_hfiles_{stamp}"
                    if _hcache_key not in st.session_state:
                        _h_expl_file  = next((fn for fn in available if fn.endswith("_explanation.md")), None)
                        _h_trans_file = next((fn for fn in available if fn.endswith("_transcript.txt")
                                              and "timestamped" not in fn), None)
                        _h_ts_file    = next((fn for fn in available if fn.endswith("_transcript_timestamped.txt")), None)
                        _h_expl = _h_full = _h_ts = ""
                        try:
                            if _h_expl_file:
                                with open(os.path.join(folder, _h_expl_file), "r", encoding="utf-8") as _hf:
                                    _h_expl = "\n".join(
                                        ln for ln in _hf.read().split("\n")
                                        if not ln.startswith("*Analysed on")
                                        and ln.strip() != f"# {base_name}"
                                    ).strip()
                            if _h_trans_file:
                                with open(os.path.join(folder, _h_trans_file), "r", encoding="utf-8") as _hf:
                                    _h_full = _hf.read()
                            if _h_ts_file:
                                with open(os.path.join(folder, _h_ts_file), "r", encoding="utf-8") as _hf:
                                    _h_ts = _hf.read()
                        except Exception:
                            pass
                        st.session_state[_hcache_key] = (_h_expl, _h_full, _h_ts)
                    _h_expl, _h_full, _h_ts = st.session_state[_hcache_key]

                    for i, fname in enumerate(available):
                        label, mime = _file_label(fname)
                        fpath = os.path.join(folder, fname)

                        # Regenerate Word and PDF fresh so markdown renders correctly
                        if fname.endswith(".docx") and _h_expl:
                            try:
                                _fresh_docx = export_word(e["video_name"], _h_expl, _h_full, _h_ts, [])
                                btn_cols[i % 4].download_button(
                                    label=label, data=_fresh_docx, file_name=fname,
                                    mime=mime, key=f"hist_dl_{stamp}_{i}",
                                    width='stretch',
                                )
                                continue
                            except Exception:
                                pass  # fall through to disk file
                        elif fname.endswith(".pdf") and _h_expl:
                            try:
                                _fresh_pdf = export_pdf(e["video_name"], _h_expl, _h_full, _h_ts)
                                btn_cols[i % 4].download_button(
                                    label=label, data=_fresh_pdf, file_name=fname,
                                    mime=mime, key=f"hist_dl_{stamp}_{i}",
                                    width='stretch',
                                )
                                continue
                            except Exception as _pdf_ex:
                                btn_cols[i % 4].warning(
                                    f"PDF unavailable — download Word and use File → Save As → PDF.\n"
                                    f"({_pdf_ex})"
                                )
                                continue

                        # All other files (or fallback): serve from disk
                        with open(fpath, "rb") as fh:
                            btn_cols[i % 4].download_button(
                                label=label, data=fh.read(), file_name=fname,
                                mime=mime, key=f"hist_dl_{stamp}_{i}",
                                width='stretch',
                            )
                else:
                    st.warning("Saved files not found — the results folder may have moved.")

                # ── Re-export missing Word / PDF ──────────────────────────
                has_docx   = any(fn.endswith(".docx") for fn in available)
                has_pdf    = any(fn.endswith(".pdf")  for fn in available)
                expl_file  = next((fn for fn in available if fn.endswith("_explanation.md")), None)
                trans_file = next((fn for fn in available if fn.endswith("_transcript.txt")
                                   and "timestamped" not in fn), None)
                ts_file    = next((fn for fn in available if fn.endswith("_transcript_timestamped.txt")), None)
                missing = []
                if not has_docx: missing.append("Word (.docx)")
                if not has_pdf:  missing.append("PDF")

                if missing and expl_file and trans_file and ts_file:
                    st.markdown("---")
                    st.caption(
                        f"⚠️ Missing: {', '.join(missing)} — "
                        "can be regenerated instantly from saved files (no AI needed)."
                    )
                    if st.button("🔄 Regenerate missing exports",
                                 key=f"regen_{stamp}"):
                        with st.spinner("Regenerating…"):
                            try:
                                with open(os.path.join(folder, expl_file), "r", encoding="utf-8") as fh:
                                    expl_text = "\n".join(
                                        ln for ln in fh.read().split("\n")
                                        if not ln.startswith("*Analysed on")
                                        and ln.strip() != f"# {base_name}"
                                    ).strip()
                                with open(os.path.join(folder, trans_file), "r", encoding="utf-8") as fh:
                                    full_trans = fh.read()
                                with open(os.path.join(folder, ts_file), "r", encoding="utf-8") as fh:
                                    ts_trans = fh.read()

                                new_files = []
                                if not has_docx:
                                    dname = f"{base_name}_{stamp}.docx"
                                    with open(os.path.join(folder, dname), "wb") as fh:
                                        fh.write(export_word(e["video_name"], expl_text,
                                                             full_trans, ts_trans, []))
                                    new_files.append(dname)
                                if not has_pdf:
                                    pname = f"{base_name}_{stamp}.pdf"
                                    with open(os.path.join(folder, pname), "wb") as fh:
                                        fh.write(export_pdf(e["video_name"], expl_text,
                                                            full_trans, ts_trans))
                                    new_files.append(pname)

                                if new_files:
                                    hpath = _history_path(save_folder)
                                    with open(hpath, "r", encoding="utf-8") as fh:
                                        recs = json.load(fh)
                                    for rec in recs:
                                        if rec.get("stamp") == stamp:
                                            rec["files"] = list(set(rec.get("files", []) + new_files))
                                            break
                                    with open(hpath, "w", encoding="utf-8") as fh:
                                        json.dump(recs, fh, ensure_ascii=False, indent=2)
                                    st.success(f"✅ Generated: {', '.join(new_files)}")
                                    st.rerun()
                            except Exception as regen_err:
                                st.error(f"Export failed: {regen_err}")

                # ── Re-analyse with additional documents ─────────────────
                st.markdown("---")
                with st.expander("🔄 Re-analyse with Additional Documents", expanded=False):
                    st.caption(
                        "Change the output language, attach slides, or both — then re-run the AI "
                        "against the saved transcript and frames. "
                        "**Whisper is not re-run** — this takes ~30 seconds."
                    )

                    # ── Output language override ──────────────────────────────
                    _lang_key = f"hist_lang_{stamp}"
                    _AUTO_LANG = "Auto (match video)"
                    _saved_lang = st.session_state.get("output_language", _AUTO_LANG)
                    if _lang_key not in st.session_state:
                        st.session_state[_lang_key] = _saved_lang

                    _lang_options = [_AUTO_LANG] + list(OUTPUT_LANGUAGES.keys())
                    _hist_lang_raw = st.selectbox(
                        "🌐 Output language for this re-analysis",
                        options=_lang_options,
                        index=_lang_options.index(
                            st.session_state[_lang_key]
                            if st.session_state[_lang_key] in _lang_options
                            else _AUTO_LANG
                        ),
                        key=_lang_key,
                        help="**Auto (match video):** uses the detected audio language. "
                             "Override to force a specific language.",
                    )
                    # Resolve "Auto" using the last detected language
                    _hist_lang = (
                        st.session_state.get("last_detected_output_language", "English")
                        if _hist_lang_raw == _AUTO_LANG
                        else _hist_lang_raw
                    )
                    if _hist_lang_raw == _AUTO_LANG:
                        st.caption(
                            f"🌐 Will use last detected language: **{_hist_lang}**"
                        )

                    st.markdown("---")

                    # ── Slide upload ──────────────────────────────────────────
                    _hist_slides_key    = f"hist_slide_upload_{stamp}"
                    _hist_slide_b64_key = f"hist_slide_b64_{stamp}"

                    st.markdown("**📎 Slides / reference documents** *(optional)*")
                    _hist_slide_files = st.file_uploader(
                        "Upload slides / reference documents",
                        type=["pdf", "pptx", "png", "jpg", "jpeg", "webp"],
                        accept_multiple_files=True,
                        key=_hist_slides_key,
                        label_visibility="collapsed",
                    )

                    # Process uploaded slides into b64
                    if _hist_slide_files:
                        with st.spinner("Processing uploaded slides…"):
                            _hist_b64, _hist_sections, _hist_cap_info = \
                                process_slide_files(_hist_slide_files, ai_engine)
                        st.session_state[_hist_slide_b64_key] = _hist_b64
                        st.session_state[f"hist_slide_sections_{stamp}"] = _hist_sections
                        st.success(f"✅ {_hist_cap_info}")
                        with st.expander(f"👁️ Preview slides ({len(_hist_b64)})",
                                         expanded=False):
                            _prev = st.columns(3)
                            for _si, _sb in enumerate(_hist_b64):
                                with _prev[_si % 3]:
                                    st.image(base64.b64decode(_sb),
                                             caption=f"Slide {_si + 1}",
                                             width='stretch')

                    _hist_b64_ready = st.session_state.get(_hist_slide_b64_key, [])

                    # ── Summary of what will happen ───────────────────────────
                    _summary_parts = [f"🌐 Language: **{_hist_lang}**"]
                    if _hist_b64_ready:
                        _summary_parts.append(f"📎 {len(_hist_b64_ready)} slide(s) attached")
                    else:
                        _summary_parts.append("📎 No slides — saved frames only")
                    st.info("  ·  ".join(_summary_parts))

                    # Warn when Claude + large deck triggers chunked mode
                    if ai_engine == "Claude (Paid)" and len(_hist_b64_ready) >= 80:
                        import math as _mh
                        _hchunks = _mh.ceil(len(_hist_b64_ready) / 25)
                        _hest = int(_hchunks * 65 / 60) + 2
                        st.warning(
                            f"⏱️ **{len(_hist_b64_ready)} slides** — Claude will split into "
                            f"**{_hchunks} batches** with 65 s pauses (~{_hest} min total). "
                            f"Switch to **Gemini (Free)** for faster large-deck analysis."
                        )

                    _reanalyse_btn_label = (
                        f"🤖 Re-analyse  ({_hist_lang}"
                        + (f" · {len(_hist_b64_ready)} slide(s)" if _hist_b64_ready else "")
                        + ")"
                    )

                    if st.button(_reanalyse_btn_label,
                                 key=f"hist_reanalyse_{stamp}",
                                 type="primary"):
                        _rd = load_history_entry_for_display(e, folder)
                        if _rd:
                            _updated_files, _failed_files = [], []
                            _vid_type = infer_video_type(e["video_name"])
                            _full  = _rd["full_transcript"]
                            _ts    = _rd["timestamped_transcript"]
                            _dur   = _rd["duration"]
                            _now   = datetime.now().strftime("%d %b %Y %H:%M")

                            # ── Step 1: regenerate explanation ────────────────
                            with st.spinner(f"🤖 Generating explanation in {_hist_lang}…"):
                                try:
                                    if ai_engine == "Gemini (Free)":
                                        _new_expl = analyze_with_gemini(
                                            _ts, _rd["frames"], _rd["frame_ts"],
                                            _dur, gemini_key, _vid_type,
                                            _hist_lang,
                                            slide_images=_hist_b64_ready or None,
                                            slide_sections=st.session_state.get(f"hist_slide_sections_{stamp}"),
                                        )
                                    elif _is_oai(ai_engine):
                                        _new_expl = analyze_with_openai(
                                            _ts, _rd["frames"], _rd["frame_ts"],
                                            _dur, openai_key, _vid_type,
                                            _hist_lang,
                                            slide_images=_hist_b64_ready or None,
                                            slide_sections=st.session_state.get(f"hist_slide_sections_{stamp}"), ai_engine=ai_engine)
                                    else:
                                        _new_expl = analyze_with_claude(
                                            _ts, _rd["frames"], _rd["frame_ts"],
                                            _dur, claude_key, _vid_type,
                                            _hist_lang,
                                            slide_images=_hist_b64_ready or None,
                                            slide_sections=st.session_state.get(f"hist_slide_sections_{stamp}"),
                                        )
                                except Exception as _e:
                                    st.error(f"AI explanation failed: {_e}")
                                    st.stop()

                            # ── Step 2: regenerate chapters in new language ───
                            with st.spinner(f"📑 Regenerating chapters in {_hist_lang}…"):
                                try:
                                    _new_chaps = generate_chapters(
                                        _ts, _dur, _vid_type,
                                        ai_engine, claude_key, gemini_key,
                                        _hist_lang,
                                        openai_key=openai_key,
                                    )
                                except Exception:
                                    _new_chaps = _rd["chapters"]  # fall back to old

                            # ── Step 3: overwrite ALL related files ───────────
                            _ts_stamp = f"*Re-analysed on {_now} · {_hist_lang}*"
                            if _hist_b64_ready:
                                _ts_stamp += f" · {len(_hist_b64_ready)} slide(s)"

                            # _explanation.md
                            if expl_file:
                                try:
                                    with open(os.path.join(folder, expl_file),
                                              "w", encoding="utf-8") as _fh:
                                        _fh.write(
                                            f"# {base_name}\n{_ts_stamp}\n\n{_new_expl}")
                                    _updated_files.append(expl_file)
                                except Exception as _e:
                                    _failed_files.append(f"{expl_file}: {_e}")

                            # _chapters.txt
                            if chaps_file := next(
                                    (f for f in available if f.endswith("_chapters.txt")), None):
                                try:
                                    with open(os.path.join(folder, chaps_file),
                                              "w", encoding="utf-8") as _fh:
                                        _fh.write(chapters_to_text(_new_chaps))
                                    _updated_files.append(chaps_file)
                                except Exception as _e:
                                    _failed_files.append(f"{chaps_file}: {_e}")

                            # .docx  (explanation + chapters in new language)
                            for _fn in [f for f in available if f.endswith(".docx")]:
                                try:
                                    with open(os.path.join(folder, _fn), "wb") as _fh:
                                        _fh.write(export_word(
                                            e["video_name"], _new_expl,
                                            _full, _ts, _new_chaps))
                                    _updated_files.append(_fn)
                                except Exception as _e:
                                    _failed_files.append(f"{_fn}: {_e}")

                            # .pdf  (explanation in new language)
                            for _fn in [f for f in available if f.endswith(".pdf")]:
                                try:
                                    with open(os.path.join(folder, _fn), "wb") as _fh:
                                        _fh.write(export_pdf(
                                            e["video_name"], _new_expl,
                                            _full, _ts))
                                    _updated_files.append(_fn)
                                except Exception as _e:
                                    _failed_files.append(f"{_fn}: {_e}")

                            # _pkm_note.md — delete stale version so it regenerates
                            # fresh in the new language when user opens PKM tab
                            _pkm_on_disk = next(
                                (f for f in available if f.endswith("_pkm_note.md")), None)
                            if _pkm_on_disk:
                                try:
                                    os.remove(os.path.join(folder, _pkm_on_disk))
                                    _updated_files.append(f"{_pkm_on_disk} (cleared — "
                                                          "regenerate in PKM tab)")
                                except Exception:
                                    pass

                            # ── Step 4: clear session caches ─────────────────
                            for _k in [
                                f"corrected_explanation_{base_name}",
                                f"pkm_note_{base_name}",
                                f"edit_transcript_{base_name}",
                                f"edit_ts_{base_name}",
                                f"edit_srt_{base_name}",
                                _hist_slide_b64_key,
                                f"_hfiles_{stamp}",   # force re-read of updated files
                            ]:
                                st.session_state.pop(_k, None)

                            # ── Step 5: persist completion banner across rerun ─
                            _slide_note = (
                                f" · {len(_hist_b64_ready)} slide(s) used"
                                if _hist_b64_ready else ""
                            )
                            st.session_state[f"_hist_reanalysis_done_{stamp}"] = (
                                f"✅ Re-analysis complete — **{_hist_lang}**{_slide_note}. "
                                f"Updated files: {', '.join(_updated_files) if _updated_files else 'none'}"
                            )
                            if _failed_files:
                                st.session_state[f"_hist_reanalysis_warn_{stamp}"] = (
                                    f"⚠️ Some files could not be saved: {'; '.join(_failed_files)}"
                                )
                            st.rerun()
                        else:
                            st.warning(
                                "Could not load saved transcript — "
                                "essential files may be missing."
                            )

                # ── Full results UI — same tabs as live analysis ───────────
                st.markdown("---")

                # Pre-populate PKM note into the session key _show_results uses
                # so the PKM tab shows the existing note rather than the generate button.
                _pkm_ss_key = f"pkm_note_{base_name}"
                if _pkm_ss_key not in st.session_state:
                    _pkm_candidate = next(
                        (fn for fn in available if fn.endswith("_pkm_note.md")), None
                    )
                    if not _pkm_candidate:
                        # Also scan folder directly in case it was generated but not in files[]
                        try:
                            for fn in os.listdir(folder):
                                if fn.endswith("_pkm_note.md") and stamp in fn:
                                    _pkm_candidate = fn
                                    break
                        except Exception:
                            pass
                    if _pkm_candidate:
                        with open(os.path.join(folder, _pkm_candidate), "r", encoding="utf-8") as fh:
                            st.session_state[_pkm_ss_key] = fh.read()

                hist_data = load_history_entry_for_display(e, folder)
                if hist_data:
                    _show_results(
                        e["video_name"],
                        hist_data["explanation"],
                        hist_data["chapters"],
                        hist_data["full_transcript"],
                        hist_data["timestamped_transcript"],
                        hist_data["srt_content"],
                        hist_data["frames"],
                        hist_data["frame_ts"],
                        ai_engine, claude_key, gemini_key,
                        hist_data["duration"],
                        folder,
                        analysis_time_sec=e.get("analysis_time_sec", 0),
                        key_ns=stamp,
                    )
                else:
                    st.info(
                        "Essential files (explanation or timestamped transcript) are missing. "
                        "The folder may have moved, or the analysis crashed before saving. "
                        "Try **Scan folder** above to rebuild the index."
                    )

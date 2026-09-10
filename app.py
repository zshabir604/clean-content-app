"""
Clean Content — Backend API
============================
Full-audio transcription (faster-whisper) + profanity detection
(better-profanity) + partial-word censoring (beep / silence / remove).

Run locally:
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 7860

Then open http://localhost:7860 in your browser.
"""

import os
import re
import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydub import AudioSegment
from pydub.generators import Sine
from faster_whisper import WhisperModel
from better_profanity import profanity

profanity.load_censor_words()

APP_DIR = Path(__file__).parent
JOBS_DIR = APP_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="Clean Content API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------
# Whisper model — loaded once, kept warm in memory.
# "tiny" = fastest (needed to stay near the 30s target on free CPU
# hosting). Swap to "base" for better accuracy if your host has
# more CPU/RAM and speed isn't critical.
# ---------------------------------------------------------------
_model = None


def get_model():
    global _model
    if _model is None:
        _model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _model


def transcribe_words(path: str):
    model = get_model()
    segments, _info = model.transcribe(path, word_timestamps=True, vad_filter=True)
    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({"word": w.word.strip(), "start": w.start, "end": w.end})
    return words


def find_bad_words(words, extra_words=None):
    if extra_words:
        profanity.add_censor_words(extra_words)
    hits = []
    for w in words:
        clean = re.sub(r"[^a-zA-Z']", "", w["word"]).lower()
        if clean and profanity.contains_profanity(clean):
            hits.append(w)
    return hits


def censor_replacement(duration_ms, mode):
    if mode == "remove":
        return None
    if mode == "silence":
        return AudioSegment.silent(duration=duration_ms)
    return Sine(1000).to_audio_segment(duration=duration_ms).apply_gain(-3)


def apply_partial_censor(audio: AudioSegment, hits, mode):
    """
    Keeps roughly the first letter's worth of sound (~18% of the word's
    duration, minimum 60ms) untouched, then applies beep/silence/remove
    to the rest of the word. This is an approximation — true phoneme-
    level cutting needs forced alignment, which is far heavier to run.
    """
    out = AudioSegment.empty()
    cursor = 0
    events = []
    for h in sorted(hits, key=lambda x: x["start"]):
        start_ms = int(h["start"] * 1000)
        end_ms = int(h["end"] * 1000)
        dur = end_ms - start_ms
        if dur <= 10:
            continue
        keep_ms = min(dur - 10, max(60, int(dur * 0.18)))
        censor_start = start_ms + keep_ms
        censor_end = end_ms
        if censor_start >= censor_end or censor_start < cursor:
            continue
        out += audio[cursor:censor_start]
        repl = censor_replacement(censor_end - censor_start, mode)
        if repl is not None:
            out += repl
        events.append({"word": h["word"], "start": start_ms, "end": censor_end, "mode": mode})
        cursor = censor_end
    out += audio[cursor:]
    return out, events


@app.post("/api/process")
async def process_audio(file: UploadFile = File(...), mode: str = Form("beep")):
    if mode not in ("beep", "silence", "remove"):
        raise HTTPException(400, "Invalid mode. Use beep, silence, or remove.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    in_path = job_dir / f"input{Path(file.filename or 'audio.wav').suffix or '.wav'}"

    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    t0 = time.time()
    try:
        audio = AudioSegment.from_file(in_path)
        wav_path = job_dir / "input.wav"
        audio.export(wav_path, format="wav")

        words = transcribe_words(str(wav_path))
        hits = find_bad_words(words)
        censored, events = apply_partial_censor(audio, hits, mode)

        out_path = job_dir / "output.mp3"
        censored.export(out_path, format="mp3", bitrate="192k")
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}")

    elapsed = round(time.time() - t0, 1)

    return JSONResponse({
        "job_id": job_id,
        "elapsed_seconds": elapsed,
        "transcript": [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in words],
        "censored_words": events,
        "audio_url": f"/api/audio/{job_id}",
    })


@app.get("/api/audio/{job_id}")
def get_audio(job_id: str):
    out_path = JOBS_DIR / job_id / "output.mp3"
    if not out_path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(out_path, media_type="audio/mpeg", filename="cleaned_audio.mp3")


# Serve the frontend (index.html + assets) from /static at the root path
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

"""
Clean Content — Backend API
============================
Comprehensive Censorship Suite:
- Drugs & Narcotics
- Weapons, Firearms & Violence
- Alcohol & Substances
- Profanities, Slurs & Regional Abuses
- Runtime Custom Words & Silent Muting
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

# 1. Base profanity load karein
profanity.load_censor_words()

# 2. Comprehensive High-Risk Keywords Dictionary
PERMANENT_CUSTOM_WORDS = [
    # --- DRUGS & NARCOTICS ---
    "cocaine", "coke", "heroin", "meth", "methamphetamine", "weed", "marijuana",
    "cannabis", "hashish", "chars", "afeem", "opium", "fentanyl", "ecstasy", 
    "mdma", "lsd", "acid", "ketamine", "crack", "morphine", "shrooms", "peyote",
    "oxycodone", "xanax", "adderall", "codeine", "lean", "dope", "pot", "ganja",
    "bhāng", "bhang", "joint", "blunt", "bong", "narcotic", "narcotics",

    # --- WEAPONS, GUNS & VIOLENCE ---
    "gun", "guns", "pistol", "revolver", "rifle", "shotgun", "kalashnikov", 
    "ak47", "m16", "glock", "ammunition", "ammo", "bullet", "bullets", "grenade",
    "bomb", "explosive", "dynamite", "rpg", "missile", "knife", "dagger", "blade",
    "machete", "sword", "firearm", "firearms", "sniper", "carbine", "trigger",
    "shoot", "shooter", "shooting", "kill", "killer", "killing", "murder", 
    "murderer", "massacre", "assassinate", "assassination", "execution", "terrorist",
    "terrorism", "bombing", "suicide", "bloodshed",

    # --- ALCOHOL & INTOXICANTS ---
    "alcohol", "beer", "vodka", "whiskey", "whisky", "rum", "tequila", "gin",
    "brandy", "wine", "champagne", "liquor", "booze", "cocktail", "sharāb", 
    "sharab", "daaru", "daru", "nashai", "intoxicated", "drunk", "hangover",

    # --- URDU / HINDI / REGIONAL ABUSES ---
    "bakwas", "kamina", "kutta", "kanjar", "harami", "chutiya", "chootiya",
    "gandu", "gaandu", "saala", "pagal", "jahil", "lanat", "laanat", "beghairat",
    "ullu", "haramkhor", "bhenchod", "madarchod", "bhosdike", "randi", "tatte",
    "loda", "lauda", "chinal", "kameena", "dalle", "khinzeer", "suar",

    # --- GENERAL INSULTS & EXTREME PROFANITIES ---
    "fuck", "fucker", "fucking", "fucked", "shit", "bullshit", "asshole", "bitch",
    "bastard", "cunt", "dick", "pussy", "whore", "slut", "retard", "nigger", 
    "faggot", "scumbag", "dipshit", "motherfucker"
]

# Set lookup for instant matching
PERMANENT_WORDS_SET = {w.strip().lower() for w in PERMANENT_CUSTOM_WORDS}
profanity.add_censor_words(list(PERMANENT_WORDS_SET))

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


def find_bad_words(words, extra_words_str=None):
    active_banned_set = set(PERMANENT_WORDS_SET)
    
    if extra_words_str:
        user_words = [w.strip().lower() for w in re.split(r'[, \n]+', extra_words_str) if w.strip()]
        if user_words:
            profanity.add_censor_words(user_words)
            active_banned_set.update(user_words)

    hits = []
    for w in words:
        clean = re.sub(r"[^a-zA-Z']", "", w["word"]).lower()
        if clean:
            if clean in active_banned_set or profanity.contains_profanity(clean):
                hits.append(w)
    return hits


def censor_replacement(duration_ms, mode):
    if mode == "remove":
        return None
    if mode == "silence":
        return AudioSegment.silent(duration=duration_ms)
    return Sine(1000).to_audio_segment(duration=duration_ms).apply_gain(-3)


def apply_censor(audio: AudioSegment, hits, mode):
    """
    Completely silences or bleeps the detected word duration.
    A safe padding (+-30ms) ensures no start or tail speech leaks through.
    """
    out = AudioSegment.empty()
    cursor = 0
    events = []
    
    for h in sorted(hits, key=lambda x: x["start"]):
        start_ms = max(0, int(h["start"] * 1000) - 30)
        end_ms = min(len(audio), int(h["end"] * 1000) + 30)

        if start_ms < cursor:
            start_ms = cursor
        if start_ms >= end_ms:
            continue

        out += audio[cursor:start_ms]
        duration_ms = end_ms - start_ms

        repl = censor_replacement(duration_ms, mode)
        if repl is not None:
            out += repl

        events.append({"word": h["word"], "start": start_ms, "end": end_ms, "mode": mode})
        cursor = end_ms

    out += audio[cursor:]
    return out, events


@app.post("/api/process")
async def process_audio(
    file: UploadFile = File(...),
    mode: str = Form("silence"),
    custom_words: str = Form("")
):
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
        hits = find_bad_words(words, extra_words_str=custom_words)
        censored, events = apply_censor(audio, hits, mode)

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


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

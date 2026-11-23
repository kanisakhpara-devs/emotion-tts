"""Local emotion TTS server. Run: .venv\\Scripts\\python server.py  (then open http://127.0.0.1:7860)"""
import gc, json, re, threading, time, uuid
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
OUT, VOICES = Path.home() / "Downloads" / "Emotion Voice Studio", ROOT / "voices"  # clips stay out of the repo
OUT.mkdir(parents=True, exist_ok=True); VOICES.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Kokoro is only used to make the preset-voice reference clips (cloned by Chatterbox).
VOICE_SRC = {"heart": "af_heart", "bella": "af_bella", "nicole": "af_nicole", "adam": "am_adam",
             "michael": "am_michael", "emma": "bf_emma", "george": "bm_george", "lewis": "bm_lewis"}
REF_TEXT = ("The quick brown fox jumps over the lazy dog near the quiet river bank. "
            "Every morning she walks along the shore, listening to the waves and the distant gulls, "
            "and thinking about everything that the new day might bring.")

app = FastAPI()
jobs: dict = {}
_model = {"kind": None, "m": None}
_lock = threading.Lock()  # one generation at a time (6 GB GPU)


def ensure_voices():
    if all((VOICES / f"{v}.wav").exists() for v in VOICE_SRC):
        return
    from kokoro import KPipeline
    pipes = {}
    for name, src in VOICE_SRC.items():
        f = VOICES / f"{name}.wav"
        if f.exists():
            continue
        lang = src[0]
        pipes.setdefault(lang, KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M"))
        audio = np.concatenate([np.asarray(a) for _, _, a in pipes[lang](REF_TEXT, voice=src)])
        sf.write(f, audio, 24000)


def load_model(kind):
    if _model["kind"] == kind:
        return _model["m"]
    _model["m"] = None; gc.collect(); torch.cuda.empty_cache()  # free VRAM before swapping
    if kind == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS as M
    else:
        from chatterbox.tts import ChatterboxTTS as M
    _model.update(kind=kind, m=M.from_pretrained(DEV))
    return _model["m"]


def tags():
    return json.loads((ROOT / "tags.json").read_text("utf-8"))


def parse(text, cfg, default_voice):
    """-> list of (voice, mood, text) segments, plus warnings."""
    moods, nv, alias = cfg["moods"], set(cfg["nonverbal"]), cfg.get("voice_aliases", {})
    voice, mood, segs, warns, buf = default_voice, "neutral", [], [], ""

    def flush():
        nonlocal buf
        if buf.strip():
            segs.append((voice, mood, buf.strip()))
        buf = ""

    pos = 0
    for m in re.finditer(r"\[([^\]]+)\]", text):
        buf += text[pos:m.start()]; pos = m.end()
        name = m.group(1).strip().lower()
        if name in nv:
            buf += f"[{name}]"
        elif name in moods:
            flush(); mood = name
        elif alias.get(name, name) in VOICE_SRC:
            flush(); voice = alias.get(name, name)
        else:
            warns.append(f"Unknown tag [{m.group(1)}] ignored")
    buf += text[pos:]; flush()
    return segs, warns


def chunks(text, limit=250):
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", text):
        if cur and len(cur) + len(s) > limit:
            out.append(cur); cur = s
        else:
            cur = f"{cur} {s}".strip()
    return out + ([cur] if cur else [])


class Req(BaseModel):
    text: str
    voice: str = "heart"
    speed: float = 1.0
    pause: float = 0.35      # seconds between sentences/chunks
    intensity: float = 0.5   # 0..1, 0.5 = tag values as written


def run(job_id, r: Req):
    job = jobs[job_id]
    try:
        with _lock:
            cfg = tags()
            segs, job["warnings"] = parse(r.text, cfg, r.voice if r.voice in VOICE_SRC else "heart")
            items = []  # (index, voice, mood, chunk, kind)
            for voice, mood, t in segs:
                for c in chunks(t):
                    items.append((len(items), voice, mood, c,
                                  "turbo" if re.search(r"\[[a-z ]+\]", c) else "std"))
            if not items:
                raise ValueError("Nothing to say: the text is empty.")
            job["status"] = "Loading voices..."; ensure_voices()
            results, scale = {}, r.intensity / 0.5
            # group by model so each loads once
            for kind in sorted({i[4] for i in items}, key=lambda k: k != _model["kind"]):
                job["status"] = f"Loading {kind} model..."; model = load_model(kind)
                for idx, voice, mood, c, k in items:
                    if k != kind:
                        continue
                    job["status"] = f"Generating {idx + 1}/{len(items)}"; job["progress"] = idx / len(items)
                    mc = cfg["moods"][mood]
                    ref = str(VOICES / f"{voice}.wav")
                    if kind == "turbo":
                        wav = model.generate(c, audio_prompt_path=ref)
                    else:
                        ex = float(np.clip(0.5 + (mc["exaggeration"] - 0.5) * scale, 0.25, 2.0))
                        wav = model.generate(c, audio_prompt_path=ref, exaggeration=ex, cfg_weight=mc["cfg"])
                    a = wav.squeeze().cpu().numpy().astype(np.float32)
                    sr = model.sr
                    sp = r.speed * mc.get("speed", 1.0)
                    if abs(sp - 1) > 0.01:
                        a = librosa.effects.time_stretch(a, rate=sp)
                    results[idx] = (a * mc.get("gain", 1.0), sr)
            sr = results[0][1]
            gap = np.zeros(int(sr * r.pause), np.float32)
            parts = []
            for i in range(len(items)):
                parts += [results[i][0], gap]
            audio = np.clip(np.concatenate(parts[:-1]), -1, 1)
            name = time.strftime("%Y%m%d-%H%M%S")
            sf.write(OUT / f"{name}.wav", audio, sr)
            sf.write(OUT / f"{name}.mp3", audio, sr, format="MP3")
            job.update(done=True, progress=1, status="Done", wav=f"/outputs/{name}.wav",
                       mp3=f"/outputs/{name}.mp3", seconds=round(len(audio) / sr, 1))
    except Exception as e:  # surfaced in the UI
        job.update(done=True, error=str(e), status="Failed")


@app.post("/api/generate")
def generate(r: Req):
    if not r.text.strip():
        raise HTTPException(400, "Enter some text first.")
    jid = uuid.uuid4().hex[:8]
    jobs[jid] = {"done": False, "progress": 0, "status": "Queued", "warnings": []}
    threading.Thread(target=run, args=(jid, r), daemon=True).start()
    return {"id": jid}


@app.get("/api/job/{jid}")
def job(jid: str):
    if jid not in jobs:
        raise HTTPException(404)
    return jobs[jid]


@app.get("/api/tags")
def get_tags():
    c = tags()
    return {"moods": list(c["moods"]), "nonverbal": c["nonverbal"],
            "voices": list(VOICE_SRC), "aliases": list(c.get("voice_aliases", {}))}


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


app.mount("/outputs", StaticFiles(directory=OUT), name="outputs")
app.mount("/voices", StaticFiles(directory=VOICES), name="voices")

if __name__ == "__main__":
    def _warm():
        with _lock:
            ensure_voices()  # so voice previews work early
    threading.Thread(target=_warm, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=7860)

import os
import re
import uuid
import threading
import queue
import itertools
import importlib.util
import dataclasses
from enum import Enum
from types import SimpleNamespace
from typing import Optional, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import subprocess
import tempfile
import numpy as np
import torch
import traceback
import gc

# whisperx / NeMo are imported lazily, so only the selected engine has to be installed.


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_LANGUAGE = "cs"
SAMPLE_RATE = 16000
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "transcripts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ASR engine:
#   whisperx   - WhisperX with WHISPER_MODEL (default large-v3)
#   whisper-cz - WhisperX with the Czech community fine-tune of large-v3
#   canary     - NVIDIA Canary-1B-v2 (NeMo), with its own word timestamps
ASR_ENGINE = os.environ.get("ASR_ENGINE", "whisperx").lower()
if ASR_ENGINE not in ("whisperx", "whisper-cz", "canary"):
    raise ValueError(f"Unknown ASR_ENGINE: {ASR_ENGINE}")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_CZ_HF_MODEL = os.environ.get(
    "WHISPER_CZ_HF_MODEL", "mikr/whisper-large-v3-czech-cv13"
)
# The fine-tune is in Transformers format; faster-whisper needs CTranslate2.
# It's converted once into this directory on first start.
WHISPER_CZ_CT2_DIR = os.environ.get(
    "WHISPER_CZ_CT2_DIR", "models/whisper-large-v3-czech-cv13-ct2"
)
CANARY_MODEL = os.environ.get("CANARY_MODEL", "nvidia/canary-1b-v2")
CANARY_BATCH_SIZE = int(os.environ.get("CANARY_BATCH_SIZE", "8"))
CANARY_LANGUAGES = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it",
    "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
}

# Canary is fed chunks cut at the quietest point between these lengths (seconds)
CHUNK_MIN_SEC = 20.0
CHUNK_MAX_SEC = 30.0
# Cue limits when building subtitles from Canary word timestamps
CUE_MAX_SEC = 7.0
CUE_MAX_CHARS = 84
CUE_MAX_GAP_SEC = 1.0

APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Content-specific prompt, blacklist and replacements (see custom_rules.example.py).
# custom_rules.py is private and ignored by git; without it the example is used.
CUSTOM_RULES_PATH = os.environ.get(
    "CUSTOM_RULES", os.path.join(APP_DIR, "custom_rules.py")
)
EXAMPLE_RULES_PATH = os.path.join(APP_DIR, "custom_rules.example.py")


def _load_rules() -> SimpleNamespace:
    path = CUSTOM_RULES_PATH if os.path.exists(CUSTOM_RULES_PATH) else EXAMPLE_RULES_PATH
    mod = None
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location("custom_rules", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print(f"Loaded custom rules from {path}", flush=True)
    else:
        print("No custom rules file found, using none", flush=True)
    return SimpleNamespace(
        languages=set(getattr(mod, "LANGUAGES", {DEFAULT_LANGUAGE})),
        initial_prompt=getattr(mod, "INITIAL_PROMPT", None),
        blacklist=[p.lower() for p in getattr(mod, "BLACKLIST", [])],
        drop_prefixes=[p.lower() for p in getattr(mod, "DROP_PREFIXES", [])],
        replacements=[
            (re.compile(p) if isinstance(p, str) else p, r)
            for p, r in getattr(mod, "REPLACEMENTS", [])
        ],
    )


RULES = _load_rules()
# Initial prompt for Whisper; the env var overrides the rules file ("" disables it)
INITIAL_PROMPT = os.environ.get("WHISPER_INITIAL_PROMPT", RULES.initial_prompt) or None

# Hallucinated word loops inside one cue: "méně méně méně méně" -> "méně"
_WORD_LOOP = re.compile(r"\b(\w+)(?:[\s,]+\1\b){3,}", re.IGNORECASE)

class Priority(str, Enum):
    high = "high"
    standard = "standard"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    finished = "finished"
    failed = "failed"


class Job(BaseModel):
    id: str
    source_path: str
    priority: Priority
    language: str = DEFAULT_LANGUAGE
    status: JobStatus
    error: Optional[str] = None
    output_path: Optional[str] = None


# PriorityQueue item: (priority_rank, counter, job_id)
job_queue: "queue.PriorityQueue[tuple[int, int, str]]" = queue.PriorityQueue()
jobs: Dict[str, Job] = {}
_jobs_lock = threading.Lock()
_job_counter = itertools.count()


_model = None
_align_models: Dict[str, tuple] = {}  # language -> (align_model, metadata) or (None, None)
_diarize_model = None
_model_lock = threading.Lock()


def _format_timestamp(sec: float) -> str:
    # Convert seconds (float) to WebVTT timestamp "HH:MM:SS.mmm"
    total_ms = int(round(sec * 1000))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    s = total_sec % 60
    total_min = total_sec // 60
    m = total_min % 60
    h = total_min // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _filter_hallucinated_segments(segments: List[dict], language: str) -> List[dict]:
    """
    Apply simple heuristics to drop very likely Whisper hallucinations.
    This is intentionally conservative so we don't remove real speech.
    """
    if not segments:
        return segments

    use_rules = language in RULES.languages

    filtered: List[dict] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        low = text.lower()

        # Drop known hallucination phrases from the custom rules
        if use_rules and (
            any(low.startswith(prefix) for prefix in RULES.drop_prefixes)
            or any(phrase in low for phrase in RULES.blacklist)
        ):
            continue

        # Drop if duration is short but text is unrealistically long
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        dur = max(0.0, end - start)
        if dur < 0.5 and len(text) > 80:
            continue

        # Drop bare "..." / "…" cues (Whisper output on music or silence)
        if not re.sub(r"[\s.…,!?\-–]+", "", text):
            continue

        filtered.append(seg)

    # Collapse repetition loops: the same cue 3+ times in a row -> keep the first
    collapsed: List[dict] = []
    run_key, run_len = None, 0
    for seg in filtered:
        key = re.sub(r"[^\w]+", " ", seg["text"].lower()).strip()
        if key == run_key:
            run_len += 1
        else:
            run_key, run_len = key, 1
        if run_len >= 3:
            # Remove the second copy too once the loop is confirmed
            if run_len == 3:
                collapsed.pop()
            continue
        collapsed.append(seg)

    return collapsed


def _apply_replacements(text: str, language: str) -> str:
    text = _WORD_LOOP.sub(r"\1", text)
    if language in RULES.languages:
        for pattern, repl in RULES.replacements:
            text = pattern.sub(repl, text)
    return text


def _write_vtt(result: dict, out_path: str, language: str = DEFAULT_LANGUAGE) -> None:
    segments = result.get("segments", [])
    segments = _filter_hallucinated_segments(segments, language)
    lines: List[str] = ["WEBVTT", ""]

    for seg in segments:
        start = _format_timestamp(seg.get("start", 0.0))
        end = _format_timestamp(seg.get("end", 0.0))
        text = _apply_replacements(seg.get("text", "").strip(), language)

        # Optional: include speaker label if present
        speaker = seg.get("speaker")
        if speaker:
            text = f"{speaker}: {text}"

        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # blank line between cues

    content = "\n".join(lines).strip() + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


def _convert_whisper_cz() -> str:
    """Convert the Czech fine-tune to CTranslate2 once and return its directory."""
    if os.path.exists(os.path.join(WHISPER_CZ_CT2_DIR, "model.bin")):
        return WHISPER_CZ_CT2_DIR

    from ctranslate2.converters import TransformersConverter

    print(f"Converting {WHISPER_CZ_HF_MODEL} to CTranslate2 (one-time)...", flush=True)
    TransformersConverter(
        WHISPER_CZ_HF_MODEL,
        # faster-whisper reads the tokenizer and the 128 mel bins of large-v3 from these
        copy_files=["tokenizer.json", "preprocessor_config.json"],
    ).convert(WHISPER_CZ_CT2_DIR, quantization="float16")
    return WHISPER_CZ_CT2_DIR


def get_models():
    global _model, _diarize_model
    with _model_lock:
        if _model is None:
            if ASR_ENGINE == "canary":
                from nemo.collections.asr.models import ASRModel

                print(f"Loading {CANARY_MODEL} on", DEVICE, flush=True)
                _model = ASRModel.from_pretrained(model_name=CANARY_MODEL)
                _model = _model.to(DEVICE).eval()
            else:
                import whisperx

                model_name = (
                    _convert_whisper_cz() if ASR_ENGINE == "whisper-cz" else WHISPER_MODEL
                )
                print(f"Loading WhisperX model {model_name} on", DEVICE, flush=True)
                _model = whisperx.load_model(model_name, DEVICE, language=DEFAULT_LANGUAGE)

            try:
                import whisperx

                _diarize_model = whisperx.DiarizationPipeline(
                    device=DEVICE, use_auth_token=None
                )
            except Exception as e:
                print(f"Could not load diarization model: {e}", flush=True)
                _diarize_model = None
        return _model, _diarize_model


def get_align_model(language: str) -> tuple:
    """WhisperX alignment model for a language, loaded on first use and cached."""
    import whisperx

    with _model_lock:
        if language not in _align_models:
            try:
                print(f"Loading alignment model for '{language}'", flush=True)
                _align_models[language] = whisperx.load_align_model(
                    language_code=language, device=DEVICE
                )
            except Exception as e:
                # No default wav2vec2 model for this language: keep segment timestamps
                print(f"No alignment model for '{language}': {e}", flush=True)
                _align_models[language] = (None, None)
        return _align_models[language]


def _set_initial_prompt(model, prompt: Optional[str]) -> None:
    """Swap the WhisperX initial prompt per job (faster-whisper options are immutable)."""
    opts = model.options
    if hasattr(opts, "_replace"):  # NamedTuple in older faster-whisper
        model.options = opts._replace(initial_prompt=prompt)
    else:
        model.options = dataclasses.replace(opts, initial_prompt=prompt)


def _load_audio(path: str) -> np.ndarray:
    """Decode any media file to 16 kHz mono float32 (same as whisperx.load_audio)."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-threads", "0", "-i", path,
         "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def _split_on_silence(audio: np.ndarray) -> List[tuple]:
    """
    Split audio into (start_sample, end_sample) chunks of CHUNK_MIN_SEC..CHUNK_MAX_SEC,
    each cut at the quietest 50 ms frame in that window so words aren't split.
    """
    frame = SAMPLE_RATE // 20
    n_frames = len(audio) // frame
    if n_frames == 0:
        return [(0, len(audio))] if len(audio) else []
    energy = np.sqrt(np.mean(audio[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1))

    min_f = int(CHUNK_MIN_SEC * SAMPLE_RATE) // frame
    max_f = int(CHUNK_MAX_SEC * SAMPLE_RATE) // frame
    chunks, start_f = [], 0
    while n_frames - start_f > max_f:
        window = energy[start_f + min_f : start_f + max_f]
        cut_f = start_f + min_f + int(np.argmin(window))
        chunks.append((start_f * frame, cut_f * frame))
        start_f = cut_f
    chunks.append((start_f * frame, len(audio)))
    return chunks


def _words_to_segments(words: List[dict]) -> List[dict]:
    """Group timed words into subtitle cues (sentence end, length, duration or pause)."""
    segments: List[dict] = []
    cur: List[dict] = []

    def flush():
        if cur:
            segments.append({
                "start": cur[0]["start"],
                "end": cur[-1]["end"],
                "text": " ".join(w["word"] for w in cur),
                "words": list(cur),
            })
            cur.clear()

    for w in words:
        if cur:
            text_len = len(" ".join(x["word"] for x in cur)) + 1 + len(w["word"])
            if (
                w["start"] - cur[-1]["end"] > CUE_MAX_GAP_SEC
                or w["end"] - cur[0]["start"] > CUE_MAX_SEC
                or text_len > CUE_MAX_CHARS
            ):
                flush()
        cur.append(w)
        if w["word"].endswith((".", "?", "!", "…")):
            flush()
    flush()
    return segments


def _transcribe_canary(model, audio: np.ndarray, language: str) -> dict:
    chunks = _split_on_silence(audio)
    words: List[dict] = []

    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, (s, e) in enumerate(chunks):
            p = os.path.join(tmp, f"{i:05d}.wav")
            sf.write(p, audio[s:e], SAMPLE_RATE)
            paths.append(p)

        outputs = model.transcribe(
            paths,
            source_lang=language,
            target_lang=language,
            timestamps=True,
            batch_size=CANARY_BATCH_SIZE,
        )

    for (s, e), out in zip(chunks, outputs):
        offset = s / SAMPLE_RATE
        stamps = (getattr(out, "timestamp", None) or {}).get("word") or []
        if stamps:
            for st in stamps:
                word = (st.get("word") or "").strip()
                if word:
                    words.append({
                        "word": word,
                        "start": offset + float(st["start"]),
                        "end": offset + float(st["end"]),
                    })
        elif (out.text or "").strip():
            # No word timestamps: one cue for the whole chunk
            words.append({"word": out.text.strip(), "start": offset, "end": e / SAMPLE_RATE})

    return {"segments": _words_to_segments(words)}


def transcribe_job(job_id: str):
    model, diarize_model = get_models()

    with _jobs_lock:
        job = jobs[job_id]
    language = job.language

    audio_path = job.source_path
    print(f"[job {job_id}] Starting transcription for: {audio_path}", flush=True)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Source file not found: {audio_path}")

    audio = None
    result = None
    diarize_segments = None

    try:
        audio = _load_audio(audio_path)

        print(
            f"[job {job_id}] Audio loaded, running ASR ({ASR_ENGINE}, language={language})...",
            flush=True,
        )
        align_model = metadata = None
        if ASR_ENGINE == "canary":
            result = _transcribe_canary(model, audio, language)
        else:
            import whisperx

            if ASR_ENGINE == "whisper-cz" and language != "cs":
                print(
                    f"[job {job_id}] Warning: the Czech fine-tune may transcribe "
                    f"'{language}' worse than the original Whisper.",
                    flush=True,
                )
            # A prompt in another language would push the transcription towards it
            _set_initial_prompt(
                model, INITIAL_PROMPT if language in RULES.languages else None
            )
            # ASR language forced to the job's language
            # task must be passed explicitly: when the language differs from the one
            # the model was loaded with, WhisperX rebuilds its tokenizer and would
            # otherwise reuse tokenizer.task, which is a token id (e.g. 50360), not
            # the string "transcribe" -> "'50360' is not a valid task".
            result = model.transcribe(
                audio, batch_size=4, language=language, task="transcribe"
            )
            align_model, metadata = get_align_model(language)
        print(
            f"[job {job_id}] ASR finished, {len(result.get('segments', []))} segments.",
            flush=True,
        )

        if align_model is not None:
            print(f"[job {job_id}] Running alignment...", flush=True)
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                DEVICE,
                return_char_alignments=False,
            )
            print(f"[job {job_id}] Alignment finished.", flush=True)

        if diarize_model is not None:
            try:
                import whisperx

                print(f"[job {job_id}] Running diarization (VAD)...", flush=True)
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                print(f"[job {job_id}] Diarization finished.", flush=True)
            except Exception as e:
                print(f"[job {job_id}] Diarization failed: {e}", flush=True)
                traceback.print_exc()

        # Write VTT file into OUTPUT_DIR, reusing original basename
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        out_path = os.path.join(OUTPUT_DIR, base_name + ".vtt")

        print(f"[job {job_id}] Writing VTT to: {out_path}", flush=True)
        _write_vtt(result, out_path, language)
        print(f"[job {job_id}] Done.", flush=True)

        return out_path
    finally:
        # Explicitly free large tensors / arrays and clear GPU cache
        try:
            del audio
            del result
            del diarize_segments
        except NameError:
            pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()


def worker_loop():
    while True:
        priority_rank, _, job_id = job_queue.get()
        try:
            with _jobs_lock:
                job = jobs.get(job_id)
                if job is None:
                    job_queue.task_done()
                    continue
                job.status = JobStatus.running
                print(f"[job {job_id}] Picked from queue (priority={priority_rank}).", flush=True)

            try:
                output_path = transcribe_job(job_id)
                with _jobs_lock:
                    job = jobs[job_id]
                    job.status = JobStatus.finished
                    job.output_path = output_path
                print(f"[job {job_id}] Finished successfully.", flush=True)
            except Exception as e:
                print(f"[job {job_id}] Error: {e}", flush=True)
                traceback.print_exc()
                with _jobs_lock:
                    job = jobs[job_id]
                    job.status = JobStatus.failed
                    job.error = str(e)
        finally:
            job_queue.task_done()


_worker_thread = threading.Thread(target=worker_loop, daemon=True)
_worker_thread.start()


app = FastAPI(title="WhisperX Transcription Queue")


class CreateJobRequest(BaseModel):
    source_path: str
    priority: Priority = Priority.standard
    # ISO 639-1 code of the spoken language, e.g. "cs", "sk", "en"
    language: Optional[str] = None


class CreateJobResponse(BaseModel):
    id: str
    status: JobStatus
    priority: Priority
    language: str


class JobResponse(Job):
    pass


def _supported_languages() -> Optional[set]:
    if ASR_ENGINE == "canary":
        return CANARY_LANGUAGES
    try:
        from faster_whisper.tokenizer import _LANGUAGE_CODES

        return set(_LANGUAGE_CODES)
    except Exception:
        return None  # can't check; an invalid code fails the job instead


@app.post("/jobs", response_model=CreateJobResponse)
def create_job(req: CreateJobRequest):
    if not os.path.isabs(req.source_path):
        raise HTTPException(
            status_code=400, detail="source_path must be an absolute path"
        )

    language = (req.language or DEFAULT_LANGUAGE).strip().lower()
    supported = _supported_languages()
    if supported is not None and language not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"language '{language}' is not supported by ASR engine '{ASR_ENGINE}'",
        )

    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        source_path=req.source_path,
        priority=req.priority,
        language=language,
        status=JobStatus.queued,
    )

    with _jobs_lock:
        jobs[job_id] = job

    priority_rank = 0 if req.priority == Priority.high else 1
    job_queue.put((priority_rank, next(_job_counter), job_id))

    return CreateJobResponse(
        id=job_id, status=job.status, priority=job.priority, language=job.language
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    with _jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job


@app.get("/jobs", response_model=List[JobResponse])
def list_jobs():
    with _jobs_lock:
        return list(jobs.values())


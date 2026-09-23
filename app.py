import os

# Ask the caching allocator to grow segments instead of fragmenting the 16 GB
# card. Must be set before the first CUDA allocation. Does not unload models.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import uuid
import threading
import queue
import itertools
from enum import Enum
from typing import Optional, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
import whisperx
import traceback
import gc

from qwen_check import correct_transcript_segments


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "large-v3"
LANGUAGE_CODE = "cs"
OUTPUT_DIR = "transcripts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    status: JobStatus
    error: Optional[str] = None
    output_path: Optional[str] = None


# PriorityQueue item: (priority_rank, counter, job_id)
job_queue: "queue.PriorityQueue[tuple[int, int, str]]" = queue.PriorityQueue()
jobs: Dict[str, Job] = {}
_jobs_lock = threading.Lock()
_job_counter = itertools.count()


_model = None
_align_model = None
_metadata = None
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


def _filter_hallucinated_segments(segments: List[dict]) -> List[dict]:
    """
    Apply simple heuristics to drop very likely Whisper hallucinations.
    This is intentionally conservative so we don't remove real speech.
    """
    if not segments:
        return segments

    # Common generic hallucination phrases
    blacklist = [
        "www.hradeckralove.org",
        "www.arkance-systems.cz",
        "titulky vytvořil johnyx",
        "titulky vytvořil johny x",
        "titulky vytvořila",
        "titulky vytvořili",
        "titulky johnyx",
        "titulky: johnyx",
        "titulky: johny x",
        "titulky by",
        "překlad a titulky",
        "překlad:",
        "konec.",
        "konec filmu",
        "konec pořadu",
        "konec vysílání",
    ]

    filtered: List[dict] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        low = text.lower()

        # Special rules:
        # - Drop any line that begins with "Titulky vytvořil"
        if low.startswith("titulky vytvořil"):
            continue

        # Drop if it matches any obvious hallucination phrase
        if any(phrase in low for phrase in blacklist):
            continue

        # Drop if duration is short but text is unrealistically long
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        dur = max(0.0, end - start)
        if dur < 0.5 and len(text) > 80:
            continue

        filtered.append(seg)

    return filtered


def _write_vtt(result: dict, out_path: str) -> None:
    segments = result.get("segments", [])
    segments = _filter_hallucinated_segments(segments)
    lines: List[str] = ["WEBVTT", ""]

    for seg in segments:
        start = _format_timestamp(seg.get("start", 0.0))
        end = _format_timestamp(seg.get("end", 0.0))
        text = seg.get("text", "").strip()

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


def _whisper_batch_size() -> int:
    raw = os.environ.get("WHISPER_BATCH_SIZE", "4").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"WHISPER_BATCH_SIZE must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"WHISPER_BATCH_SIZE must be positive, got {value}")
    return value


def get_whisperx_models():
    global _model, _align_model, _metadata, _diarize_model
    with _model_lock:
        if _model is None:
            print("Loading WhisperX model on", DEVICE, flush=True)
            # float16 keeps large-v3 near 3 GB so a 4-bit Qwen can stay resident
            # with it on a 16 GB RTX PRO 2000 Blackwell. Never unloaded.
            load_kwargs = {}
            if DEVICE == "cuda":
                load_kwargs["compute_type"] = "float16"
            _model = whisperx.load_model(
                MODEL_NAME, DEVICE, language=LANGUAGE_CODE, **load_kwargs
            )

            _align_model, _metadata = whisperx.load_align_model(
                language_code=LANGUAGE_CODE, device=DEVICE
            )

            try:
                _diarize_model = whisperx.DiarizationPipeline(
                    device=DEVICE, use_auth_token=None
                )
            except Exception as e:
                print(f"Could not load diarization model: {e}", flush=True)
                _diarize_model = None
        return _model, _align_model, _metadata, _diarize_model


def transcribe_job(job_id: str):
    model, align_model, metadata, diarize_model = get_whisperx_models()

    with _jobs_lock:
        job = jobs[job_id]

    audio_path = job.source_path
    print(f"[job {job_id}] Starting transcription for: {audio_path}", flush=True)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Source file not found: {audio_path}")

    audio = None
    result = None
    diarize_segments = None

    try:
        audio = whisperx.load_audio(audio_path)

        print(f"[job {job_id}] Audio loaded, running ASR...", flush=True)
        # Force Czech ASR language. Batch size 4 is the 16 GB budget with Qwen
        # kept loaded beside Whisper; lower WHISPER_BATCH_SIZE if a job OOMs.
        result = model.transcribe(
            audio, batch_size=_whisper_batch_size(), language=LANGUAGE_CODE
        )
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
                print(f"[job {job_id}] Running diarization (VAD)...", flush=True)
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                print(f"[job {job_id}] Diarization finished.", flush=True)
            except Exception as e:
                print(f"[job {job_id}] Diarization failed: {e}", flush=True)
                traceback.print_exc()

        # Drop known junk before Qwen, then proofread every remaining line.
        # Qwen is loaded once and does not evict the WhisperX models.
        result["segments"] = _filter_hallucinated_segments(result.get("segments", []))
        print(
            f"[job {job_id}] {len(result['segments'])} segments after hallucination filter. "
            "Running Qwen proofreading.",
            flush=True,
        )
        result["segments"] = correct_transcript_segments(
            result["segments"], log_prefix=f"[job {job_id}] "
        )

        # Write VTT file into OUTPUT_DIR, reusing original basename
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        out_path = os.path.join(OUTPUT_DIR, base_name + ".vtt")

        print(f"[job {job_id}] Writing VTT to: {out_path}", flush=True)
        _write_vtt(result, out_path)
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


class CreateJobResponse(BaseModel):
    id: str
    status: JobStatus
    priority: Priority


class JobResponse(Job):
    pass


@app.post("/jobs", response_model=CreateJobResponse)
def create_job(req: CreateJobRequest):
    if not os.path.isabs(req.source_path):
        raise HTTPException(
            status_code=400, detail="source_path must be an absolute path"
        )

    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        source_path=req.source_path,
        priority=req.priority,
        status=JobStatus.queued,
    )

    with _jobs_lock:
        jobs[job_id] = job

    priority_rank = 0 if req.priority == Priority.high else 1
    job_queue.put((priority_rank, next(_job_counter), job_id))

    return CreateJobResponse(id=job_id, status=job.status, priority=job.priority)


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


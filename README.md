# WhisperX Queue Transcription Service

This project provides a small HTTP service that manages a queue of audio transcription jobs using [WhisperX](https://github.com/m-bain/whisperX).

- **Single worker**: processes one job at a time to avoid GPU overcommit.
- **Priority queue**: supports `high` and `standard` priorities. High‑priority jobs are taken before queued standard jobs (but do not preempt a job already running).
- **WhisperX (Czech)**: uses WhisperX with language forced to Czech (`cs`) and outputs WebVTT (`.vtt`) subtitle files.
- **Hallucination filtering**: removes some common hallucinated subtitle lines (e.g. fixed “Titulky vytvořil…” credit lines, some URLs, and typical outro noise).

---

## Requirements

- Python 3.10+ (recommended)
- A working PyTorch + CUDA or CPU setup compatible with WhisperX

Example install:

```bash
pip install fastapi uvicorn whisperx torch
```

---

## Running the service

From the project root:

```bash
uvicorn app:app --host 0.0.0.0 --port 8093
```

On startup it:

- Lazily loads the WhisperX ASR and alignment models (Czech, `large-v3` by default).
- Starts a background worker thread that pulls jobs from queue.

---

## API

### Create a job

`POST /jobs`

Request body:

```json
{
  "source_path": "/absolute/path/to/media/file.mp4",
  "priority": "standard"
}
```

- `source_path` is an **absolute path** to existing audio/video file.
- `priority` is `"standard"` (default) or `"high"`.

Response:

```json
{
  "id": "job-uuid",
  "status": "queued",
  "priority": "standard"
}
```

---

### Get a single job

`GET /jobs/{job_id}`

Response example:

```json
{
  "id": "job-uuid",
  "source_path": "/absolute/path/to/media/file.mp4",
  "priority": "standard",
  "status": "finished",
  "error": null,
  "output_path": "transcripts/file.vtt"
}
```

`status` can be:

- `queued`
- `running`
- `finished`
- `failed` (with `error` explaining why)

---

### List all jobs

`GET /jobs`

Returns an array of job objects, same shape as `GET /jobs/{job_id}`.

---

## Output files

- For each finished job, a `.vtt` file is written into the `transcripts/` directory.
- The filename is derived from the source file basename:
  - `/data/audio/foo.mp3` → `transcripts/foo.vtt`

The job’s `output_path` field gives you the exact path to the generated VTT file.

---

## Hallucination filtering

Before writing VTT, `_filter_hallucinated_segments` in `app.py`:

- Drops empty segments.
- Drops some obvious hallucinated lines, including:
  - Fixed credit lines starting with `Titulky vytvořil...`
  - Certain URLs, e.g. `www.hradeckralove.org`, `www.arkance-systems.cz`
  - Several common `Konec...` lines.

You can extend or tune this list by editing the blacklist and rules in `_filter_hallucinated_segments`.

---

## Notes

- The service keeps WhisperX models loaded in memory and reuses them across jobs to avoid reload overhead.
- After each job, it attempts to free GPU memory (`torch.cuda.empty_cache()`) and run `gc.collect()` to reduce VRAM buildup.


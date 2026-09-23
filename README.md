# WhisperX Queue Transcription Service

This project provides a small HTTP service that manages a queue of audio transcription jobs using [WhisperX](https://github.com/m-bain/whisperX).

- **Single worker**: processes one job at a time to avoid GPU overcommit.
- **Priority queue**: supports `high` and `standard` priorities. High‑priority jobs are taken before queued standard jobs (but do not preempt a job already running).
- **WhisperX (Czech)**: uses WhisperX with language forced to Czech (`cs`) and outputs WebVTT (`.vtt`) subtitle files.
- **Qwen proofreading**: after each file’s WhisperX pass, a resident Qwen model corrects the lines (Czech i/y and s/z, other ASR spelling, and nonsense caused by non-Czech speech).
- **Hallucination filtering**: removes some common hallucinated subtitle lines (e.g. fixed “Titulky vytvořil…” credit lines, some URLs, and typical outro noise).

---

## Requirements

- Python 3.10+ (recommended)
- A working PyTorch + CUDA setup. The intended GPU is an NVIDIA RTX PRO 2000 Blackwell (16 GB). Install a PyTorch build built for CUDA 12.8 or newer so Blackwell (sm_120) kernels load.
- `bitsandbytes` 0.45.3 or newer, which ships sm_120 kernels for 4-bit Qwen

Example install:

```bash
pip install fastapi uvicorn whisperx torch "transformers>=4.51" "accelerate>=1.0" "bitsandbytes>=0.45.3"
```

---

## Running the service

From the project root:

```bash
uvicorn app:app --host 0.0.0.0 --port 8093
```

On startup it:

- Lazily loads the WhisperX ASR and alignment models (Czech, `large-v3` by default) and leaves them loaded.
- On the first job, also loads Qwen in 4-bit into the VRAM that is still free. Later jobs reuse both models.
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

The written cues are the Qwen-corrected text. Segment times and speaker labels still come from WhisperX.

---

## Qwen proofreading

For every file, after WhisperX transcription, alignment, diarization, and the hallucination filter, `correct_transcript_segments` in `qwen_check.py` sends the lines to Qwen. The checker asks the model to:

- fix Czech spelling that ASR confuses when the sounds match, especially **i/y** (and í/ý) and **s/z** (including final devoicing), plus missing diacritics and small word-boundary mistakes;
- notice when a line is not Czech. Language is forced to `cs`, so speech in another language comes back as nonsense or a phonetic approximation. Qwen should restore that utterance in the language that was actually spoken, and leave coherent Czech in Czech.

Lines are sent a few at a time, with the previous corrected lines as context, so a long file does not grow the Qwen cache. Timestamps are not changed. If one chunk of lines cannot be parsed, those lines keep the Whisper text and the job continues. If every chunk fails, the job fails instead of publishing an unchecked transcript.

### Memory on RTX PRO 2000 Blackwell (16 GB)

Both models stay loaded for the life of the process. Qwen is not given the GPU by unloading Whisper.

| Resident piece | Approx. VRAM |
| --- | --- |
| Whisper large-v3, float16 | ~3 GB |
| Czech alignment model | ~1 GB |
| Diarization, when it loads | ~1–2 GB |
| Whisper activations at batch size 4 | a few GB at peak |
| Qwen3-4B-Instruct-2507, NF4 | ~3–3.5 GB |

That peak is meant to fit in 16 GB. The default model is `Qwen/Qwen3-4B-Instruct-2507` because it is text-only (no vision tower) and does not emit a thinking trace. A full bf16 copy of the same model is about 8 GB and will not fit beside Whisper; the loader refuses that quant when free VRAM is too low, and it does not evict Whisper to make room.

If a job runs out of memory during Whisper, lower `WHISPER_BATCH_SIZE` (default `4`). Do not unload Whisper to free space for a larger Qwen.

### Environment

| Variable | Default | Role |
| --- | --- | --- |
| `QWEN_ENABLED` | `1` | Set `0` to write the WhisperX transcript with no Qwen pass |
| `QWEN_MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | Hugging Face model id |
| `QWEN_QUANT` | `4bit` | `4bit` (default), `8bit`, `bf16`, or `fp16` |
| `QWEN_CHUNK_SEGMENTS` | `6` | Lines per Qwen call |
| `QWEN_CONTEXT_SEGMENTS` | `2` | Previous corrected lines passed as read-only context |
| `QWEN_MAX_NEW_TOKENS` | `768` | Cap on each reply |
| `QWEN_MIN_FREE_GB` | `3.5` for 4-bit | Refuse to load Qwen when less than this is free |
| `QWEN_STRICT` | `0` | Set `1` to fail the job if any chunk cannot be parsed |
| `WHISPER_BATCH_SIZE` | `4` | WhisperX ASR batch size |

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

- The service keeps the WhisperX models and the Qwen checker loaded, and reuses them across jobs. `torch.cuda.empty_cache()` only returns unused caching-allocator blocks; it does not unload either model.
- After each job it also runs `gc.collect()` so temporary tensors from that file can be released.


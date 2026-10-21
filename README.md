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

### Choosing the ASR engine

Set `ASR_ENGINE` before starting the service:

| `ASR_ENGINE` | Model | Extra install |
|---|---|---|
| `whisperx` (default) | WhisperX with `WHISPER_MODEL` (default `large-v3`) | – |
| `whisper-cz` | WhisperX with the Czech community fine-tune [`mikr/whisper-large-v3-czech-cv13`](https://huggingface.co/mikr/whisper-large-v3-czech-cv13) | `pip install transformers ctranslate2` (for the one-time conversion) |
| `canary` | NVIDIA [Canary-1B-v2](https://huggingface.co/nvidia/canary-1b-v2) (CC-BY-4.0), lower Czech WER on FLEURS than Whisper large-v3 | `pip install -U "nemo_toolkit[asr]"`, NVIDIA Ampere or newer GPU |

```bash
ASR_ENGINE=canary uvicorn app:app --host 0.0.0.0 --port 8093
```

- **whisper-cz**: on first start the model is converted to CTranslate2 into `models/whisper-large-v3-czech-cv13-ct2` (`WHISPER_CZ_CT2_DIR`), then loaded from there. WhisperX alignment, the initial prompt and the custom rules work as with `whisperx`.
- **canary**: audio is cut into 20–30 s chunks at the quietest point, transcribed in batches (`CANARY_BATCH_SIZE`, default 8) with Canary's own word timestamps, and grouped into cues (sentence end, max 7 s, 84 characters or a 1 s pause). WhisperX alignment isn't used, and the initial prompt doesn't apply (Canary has no prompt), but the custom rules and filters do. Diarization runs only if WhisperX is installed in the same environment.
- To compare engines on the same files, run each with its own `OUTPUT_DIR` (default `transcripts`).

---

## API

### Create a job

`POST /jobs`

Request body:

```json
{
  "source_path": "/absolute/path/to/media/file.mp4",
  "priority": "standard",
  "language": "cs"
}
```

- `source_path` is an **absolute path** to existing audio/video file.
- `priority` is `"standard"` (default) or `"high"`.
- `language` (optional) is the spoken language as an ISO 639-1 code (`"cs"`, `"sk"`, `"en"`…). Defaults to `"cs"`. Codes the selected engine doesn't support return 400. Custom rules apply only to the languages listed in the rules file (see below); for other languages the WhisperX alignment model for that language is used if one exists.

Response:

```json
{
  "id": "job-uuid",
  "status": "queued",
  "priority": "standard",
  "language": "cs"
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
  "language": "cs",
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

## Hallucination filtering and custom rules

Before writing VTT, `app.py` always:

- drops empty segments and bare `...` / `…` cues,
- drops very short cues with unrealistically long text,
- collapses repetition loops (the same cue 3+ times in a row is kept once, a word repeated 4+ times in one cue is kept once).

Content-specific rules live in a separate Python file, by default `custom_rules.py` next to `app.py` (another path can be set with the `CUSTOM_RULES` env var). It is ignored by git, so your own vocabulary and fixes stay private. Start from the committed example:

```bash
cp custom_rules.example.py custom_rules.py
```

If `custom_rules.py` doesn't exist, `custom_rules.example.py` is used. The file defines:

- `LANGUAGES`: the job languages the rules apply to (e.g. `{"cs"}`). Other jobs get only the built-in filters above.
- `INITIAL_PROMPT`: text passed to Whisper as initial prompt. Whisper copies vocabulary, spelling and capitalisation from it, so write it in your house style with names and terms typical for your programmes. Keep it short: only about 220 tokens fit. `WHISPER_INITIAL_PROMPT` env var overrides it (an empty string disables it). Canary ignores it.
- `BLACKLIST` / `DROP_PREFIXES`: cues containing / starting with these phrases are dropped. The example includes common Czech Whisper hallucinations such as `Titulky vytvořil...`, some URLs and `Konec...` lines.
- `REPLACEMENTS`: `(regex, replacement)` pairs applied to every cue, e.g. `aby jsme` → `abychom`. The replacement can be a function, as in `re.sub`. Only add fixes that are wrong in any context.

---

## Notes

- The service keeps WhisperX models loaded in memory and reuses them across jobs to avoid reload overhead.
- After each job, it attempts to free GPU memory (`torch.cuda.empty_cache()`) and run `gc.collect()` to reduce VRAM buildup.


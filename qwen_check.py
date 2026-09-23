"""Proofread WhisperX subtitles with a resident Qwen model.

Sized for an NVIDIA RTX PRO 2000 Blackwell (16 GB GDDR7). WhisperX large-v3,
the Czech aligner, and diarization stay on the GPU. Qwen is loaded once, in
4-bit by default, into the memory that is still free, and is never swapped in
by evicting Whisper.
"""

import json
import os
import re
import threading
from typing import Callable, List, Optional, Sequence

# Qwen3-4B-Instruct-2507 is text-only and non-thinking. In NF4 it is about
# 3 GB, which leaves room for Whisper large-v3 (float16) plus alignment and
# diarization on a 16 GB card. Override with QWEN_MODEL if you have headroom.
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

SYSTEM_PROMPT = """You proofread subtitle lines produced by Whisper ASR. The recognizer was forced to Czech (language code cs), so every line is an attempt at Czech even when the speech was in another language.

Fix these errors, and no others:
- Czech i/y and í/ý. They often sound the same; pick the spelling Czech orthography requires.
- Czech s/z mixups, including consonants devoiced at the end of a word (s/z, and the same pattern for similar pairs).
- Missing or wrong diacritics (ě š č ř ž ý á í é ů ú ď ť ň) and small ASR grammar or word-boundary mistakes.
- A line that is not real Czech. Forced Czech decoding turns other languages into nonsense or a phonetic approximation. When a line is gibberish as Czech, recover the utterance in the language that was actually spoken (Slovak, English, German, Polish, Ukrainian, and others all occur). Do not translate coherent Czech. Do not rewrite a foreign-language line into Czech.

Rules:
- Return exactly one string per input line, in the same order.
- Do not merge, split, drop, or invent lines. Do not paraphrase a line that is already right; copy it unchanged.
- Keep names, numbers, and filler words.
- No timestamps, speaker labels, or commentary.
- Reply with a JSON array of strings and nothing else.

s/z example: "slyšel jsem spev" -> "slyšel jsem zpěv".
i/y example: "bili jsme doma" -> "byli jsme doma".
Other language example: "helo hau ár jú" -> "Hello, how are you?"."""

_EXAMPLE_USER = """Correct these 4 lines:
[1] šel jsem do skoly a viděl jsem žlute auto
[2] bili jsme tam skoro hodinu a zpivali sme spev
[3] helo hau ár jú
[4] Dnes je středa.
Return a JSON array of exactly 4 strings."""

_EXAMPLE_ASSISTANT = json.dumps(
    [
        "Šel jsem do školy a viděl jsem žluté auto",
        "Byli jsme tam skoro hodinu a zpívali jsme zpěv",
        "Hello, how are you?",
        "Dnes je středa.",
    ],
    ensure_ascii=False,
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_qwen_lock = threading.RLock()
_qwen_model = None
_qwen_tokenizer = None

GenerateFn = Callable[[Sequence[str], Sequence[str]], List[str]]


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}, got {value}")
    return value


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def qwen_enabled() -> bool:
    return _env_flag("QWEN_ENABLED", True)


def _quant_mode() -> str:
    quant = os.environ.get("QWEN_QUANT", "4bit").strip().lower()
    if quant not in {"4bit", "8bit", "bf16", "fp16"}:
        raise RuntimeError(
            "QWEN_QUANT must be one of 4bit, 8bit, bf16, fp16, "
            f"got {quant!r}"
        )
    return quant


def _min_free_bytes(quant: str) -> int:
    # Weights plus a little scratch. Full bf16/fp16 of the 4B model is ~8 GB
    # and will not fit beside Whisper large-v3 on 16 GB.
    defaults = {"4bit": 3.5, "8bit": 5.5, "bf16": 9.0, "fp16": 9.0}
    gib = float(os.environ.get("QWEN_MIN_FREE_GB", str(defaults[quant])))
    return int(gib * (1024 ** 3))


def build_messages(
    context_lines: Sequence[str], lines: Sequence[str], error_hint: Optional[str] = None
) -> list:
    user_lines = []
    if context_lines:
        user_lines.append(
            "Already corrected context (do not include these in the answer):"
        )
        for index, line in enumerate(context_lines, 1):
            user_lines.append(f"[c{index}] {line}")
        user_lines.append("")
    user_lines.append(f"Correct these {len(lines)} lines:")
    for index, line in enumerate(lines, 1):
        user_lines.append(f"[{index}] {line}")
    user_lines.append("")
    user_lines.append(f"Return a JSON array of exactly {len(lines)} strings.")
    if error_hint:
        user_lines.append(f"Your previous answer was rejected: {error_hint}")
        user_lines.append("Return only the JSON array.")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _EXAMPLE_USER},
        {"role": "assistant", "content": _EXAMPLE_ASSISTANT},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def _strip_thinking(text: str) -> str:
    text = _THINK_RE.sub("", text)
    lower = text.lower()
    if "<think>" in lower and "</think>" not in lower:
        # Generation was cut off inside a thinking block, so there is no answer.
        return ""
    return text.strip()


def parse_correction(text: str, expected: int) -> List[str]:
    """Pull a JSON string list of length `expected` out of a model reply."""
    cleaned = _strip_thinking(text or "")
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    array_at = cleaned.find("[")
    object_at = cleaned.find("{")
    if array_at == -1 and object_at == -1:
        raise ValueError("model reply did not contain JSON")
    if array_at != -1 and (object_at == -1 or array_at < object_at):
        end = cleaned.rfind("]")
        if end < array_at:
            raise ValueError("model reply has an unterminated JSON array")
        payload = cleaned[array_at : end + 1]
    else:
        end = cleaned.rfind("}")
        if end < object_at:
            raise ValueError("model reply has an unterminated JSON object")
        payload = cleaned[object_at : end + 1]

    data = json.loads(payload)
    if isinstance(data, dict):
        for key in ("lines", "segments", "corrections", "texts"):
            if key in data:
                data = data[key]
                break
        else:
            raise ValueError("JSON object has no lines/segments/corrections/texts")

    if not isinstance(data, list):
        raise ValueError("JSON correction is not a list")

    lines: List[str] = []
    for item in data:
        if isinstance(item, str):
            lines.append(item)
        elif isinstance(item, dict) and "text" in item:
            lines.append(str(item["text"]))
        else:
            raise ValueError("JSON list items must be strings or {\"text\": ...}")
    if len(lines) != expected:
        raise ValueError(f"expected {expected} lines, got {len(lines)}")
    return lines


def _accept_line(original: str, corrected: str) -> str:
    corrected = (corrected or "").strip()
    original = original or ""
    if not corrected:
        return original
    # A short phonetic line may grow when a foreign language is restored,
    # but a multi-paragraph reply is the model ignoring the line count.
    limit = max(400, len(original) * 4)
    if len(corrected) > limit:
        return original
    return corrected


def apply_corrections(segments: Sequence[dict], texts: Sequence[str]) -> List[dict]:
    if len(segments) != len(texts):
        raise ValueError(
            f"cannot apply {len(texts)} corrections to {len(segments)} segments"
        )
    updated: List[dict] = []
    for segment, text in zip(segments, texts):
        new_segment = dict(segment)
        new_segment["text"] = _accept_line(str(segment.get("text") or ""), text)
        updated.append(new_segment)
    return updated


def _chunked(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def correct_transcript_segments(
    segments: Sequence[dict],
    generate_fn: Optional[GenerateFn] = None,
    log_prefix: str = "",
) -> List[dict]:
    """Return segments with Qwen-corrected text. Timestamps and speakers stay."""
    if not segments:
        return []
    if generate_fn is None and not qwen_enabled():
        print(f"{log_prefix}Qwen proofreading disabled (QWEN_ENABLED=0).", flush=True)
        return [dict(segment) for segment in segments]

    if generate_fn is None:
        generate_fn = generate_corrections

    chunk_size = _env_int("QWEN_CHUNK_SEGMENTS", 6)
    context_size = _env_int("QWEN_CONTEXT_SEGMENTS", 2, minimum=0)
    strict = _env_flag("QWEN_STRICT", False)
    pending = [dict(segment) for segment in segments]
    corrected: List[dict] = []
    failures = 0
    chunks = 0
    prefix = log_prefix

    for start, chunk in _chunked(pending, chunk_size):
        chunks += 1
        original_texts = [str(segment.get("text") or "") for segment in chunk]
        # seq[-0:] is the whole list, so an explicit empty context is required.
        if context_size:
            context = [segment["text"] for segment in corrected[-context_size:]]
        else:
            context = []
        end = start + len(chunk)
        print(
            f"{prefix}Qwen checking lines {start + 1}-{end} of {len(pending)}.",
            flush=True,
        )
        try:
            new_texts = generate_fn(context, original_texts)
            if len(new_texts) != len(chunk):
                raise RuntimeError(
                    f"Qwen returned {len(new_texts)} lines for {len(chunk)} inputs"
                )
            applied = apply_corrections(chunk, new_texts)
        except Exception as exc:
            failures += 1
            print(
                f"{prefix}Qwen check failed for lines {start + 1}-{end}: {exc}",
                flush=True,
            )
            if strict:
                raise
            applied = [dict(segment) for segment in chunk]
        corrected.extend(applied)

    changed = sum(
        1
        for before, after in zip(pending, corrected)
        if (before.get("text") or "").strip() != (after.get("text") or "").strip()
    )
    print(
        f"{prefix}Qwen finished: {changed} of {len(pending)} lines changed"
        f" ({failures} chunk(s) left unchanged).",
        flush=True,
    )
    if chunks and failures == chunks:
        raise RuntimeError(
            "Qwen proofreading failed for every chunk. "
            "The Whisper transcript was not written as the final result."
        )
    return corrected


def _render_prompt(tokenizer, messages) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception as exc:
        # Qwen3-4B-Instruct-2507 has no thinking switch. Other chat templates
        # reject the kwarg; retry without it so the caller still sees a real
        # template error if the messages themselves are the problem.
        if "think" not in str(exc).lower():
            print(
                f"Chat template rejected enable_thinking=False ({exc}). Retrying without it.",
                flush=True,
            )
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _load_qwen():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Qwen proofreading needs torch, transformers, accelerate, and "
            f"bitsandbytes. Import failed: {exc}"
        ) from exc

    model_name = os.environ.get("QWEN_MODEL", DEFAULT_QWEN_MODEL).strip()
    quant = _quant_mode()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and quant in {"4bit", "8bit"}:
        print(
            "Qwen 4-bit/8-bit quantization needs CUDA. Loading full weights on CPU.",
            flush=True,
        )
        quant = "fp32"

    if device == "cuda":
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        needed = _min_free_bytes(quant)
        name = torch.cuda.get_device_name(0)
        print(
            f"Loading {model_name} ({quant}) on {name}: "
            f"{free / 1024 ** 3:.1f} GB free of {total / 1024 ** 3:.1f} GB. "
            "WhisperX stays loaded.",
            flush=True,
        )
        if free < needed:
            raise RuntimeError(
                f"Refusing to load {model_name} ({quant}): "
                f"only {free / 1024 ** 3:.1f} GB VRAM is free, and this quant "
                f"needs at least {needed / 1024 ** 3:.1f} GB. WhisperX is left "
                "loaded. Free some VRAM without unloading Whisper (lower "
                "WHISPER_BATCH_SIZE) or set QWEN_QUANT=4bit."
            )
        major, _minor = torch.cuda.get_device_capability()
        compute_dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        compute_dtype = torch.float32
        print(f"Loading {model_name} ({quant}) on CPU.", flush=True)

    quant_config = None
    torch_dtype = compute_dtype
    if quant == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif quant == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "bf16":
        torch_dtype = torch.bfloat16
    elif quant == "fp16":
        torch_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["device_map"] = {"": 0}
    elif device == "cuda":
        load_kwargs["device_map"] = {"": 0}
    else:
        load_kwargs["device_map"] = {"": "cpu"}

    def _from_pretrained(**extra):
        try:
            return AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs, **extra)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "Qwen could not be allocated in the free VRAM, so it was not "
                "loaded. WhisperX was left on the GPU. Lower WHISPER_BATCH_SIZE "
                "or keep QWEN_QUANT=4bit."
            ) from exc

    try:
        model = _from_pretrained(attn_implementation="sdpa")
    except (TypeError, ValueError) as exc:
        if "attn" not in str(exc).lower() and "attention" not in str(exc).lower():
            raise
        print(
            f"SDPA attention is not available for this install ({exc}). "
            "Loading Qwen with the default attention implementation.",
            flush=True,
        )
        model = _from_pretrained()
    model.eval()
    return model, tokenizer


def get_qwen():
    """Load Qwen once and keep it on device for every later file."""
    global _qwen_model, _qwen_tokenizer
    with _qwen_lock:
        if _qwen_model is None or _qwen_tokenizer is None:
            _qwen_model, _qwen_tokenizer = _load_qwen()
        return _qwen_model, _qwen_tokenizer


def _max_new_tokens(line_count: int) -> int:
    cap = _env_int("QWEN_MAX_NEW_TOKENS", 768)
    return min(cap, max(192, line_count * 80))


def _generate_raw(model, tokenizer, context_lines, lines, error_hint: Optional[str]) -> str:
    import torch

    messages = build_messages(context_lines, lines, error_hint=error_hint)
    prompt = _render_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=_max_new_tokens(len(lines)),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_corrections(context_lines: Sequence[str], lines: Sequence[str]) -> List[str]:
    """Run the resident Qwen model. One retry if the reply is not usable JSON."""
    if not lines:
        return []
    model, tokenizer = get_qwen()
    last_error = None
    with _qwen_lock:
        for attempt in (1, 2):
            raw = _generate_raw(
                model, tokenizer, context_lines, lines, error_hint=last_error
            )
            try:
                return parse_correction(raw, len(lines))
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                print(
                    f"Qwen reply was not usable JSON (attempt {attempt}): {exc}",
                    flush=True,
                )
    raise RuntimeError(
        f"Qwen did not return {len(lines)} corrected lines: {last_error}"
    )

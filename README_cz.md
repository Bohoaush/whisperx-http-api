# WhisperX Queue Transcription Service

Tento projekt spravuje frontu pro přepis audia do vtt pomocí [WhisperX](https://github.com/m-bain/whisperX). Fronta se spravuje přes http API.

- **Single worker**: zpracovává se najednou pouze jeden soubor.
- **Priority**: existuje vysoká - `high` a standardní - `standard` priorita. Požadavky s vyšší prioritou jsou zpracovány před standardními, ale aktuálně běžící požadavek už není přerušen.
- **WhisperX (Czech)**: používá se WhisperX s jazykem nastaveným na `cs`, výstup jsou WebVTT (`.vtt`) titulkové soubory.
- **Filtrování Halucinací**: odtraňují se některé časté Whisper halucinace (např. “Titulky vytvořil…”, některé URLs, atd.).

---

## Požadavky

- Python 3.10+ (doporučeno)
- Fungující PyTorch + CUDA nebo CPU sestava kompatibilní s WhisperX

Instalace:

```bash
pip install fastapi uvicorn whisperx torch
```

---

## Spuštění

From the project root:

```bash
uvicorn app:app --host 0.0.0.0 --port 8093
```

Při spuštění se:

- načtou ASR a alignment modely (defaultně Czech `large-v3`).
- spustí worker který si bere požadavky z fronty.

---

## API

### Zadat požadavek

`POST /jobs`

```json
{
  "source_path": "/absolute/path/to/media/file.mp4",
  "priority": "standard"
}
```

- `source_path` je **absolutní cesta** k existujícímu audio/video souboru.
- `priority` je `"standard"` - stadardní (výchozí) nebo `"high"` - vysoká.

Odpověď:

```json
{
  "id": "job-uuid",
  "status": "queued",
  "priority": "standard"
}
```

---

### Dotaz na stav požadavku

`GET /jobs/{job_id}`

Příklad odpovědi:

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

`status` může být:

- `queued` - naplánováno
- `running` - běží
- `finished` - dokončeno
- `failed` - selhalo (s `error` podrobnostmi o chybě)

---

### Přehled o frontě

`GET /jobs`

Vrací pole se všemi stavy požadavků ve stejném formátu jako `GET /jobs/{job_id}`.

---

## Výstup

- Po každém dokončeném požadavku je vytvořen `.vtt` soubor do podsložky `transcripts/`.
- Jméno souboru se vytvoří podle jména zdrojového souboru:
  - `/data/audio/foo.mp3` → `transcripts/foo.vtt`

`output_path` u požadavku je celá cesta k vytvořenému vtt souboru.

---

## Filtrování halucinací

Před zápisem VTT, `_filter_hallucinated_segments` v `app.py` se vymažou:

- prázdné segmenty.
- definované známé halucinace, např.:
  - Segmenty začínající `Titulky vytvořil...`
  - Některé URL, např. `www.hradeckralove.org`, `www.arkance-systems.cz`
  - Některé řetězce jako `Konec...`.

Filtrování je možné přenastavit editováním pravidel v `_filter_hallucinated_segments`.

---

## Poznámky

- WhisperX modely zůstávají ve VRAM dokud aplikace běží a jsou používány opakovaně.
- Po každém požadavku se čistí cash, aby se nepřeplnila VRAM - (`torch.cuda.empty_cache()`) a `gc.collect()`.

# WhisperX Queue Transcription Service

Tento projekt spravuje frontu pro přepis audia do vtt pomocí [WhisperX](https://github.com/m-bain/whisperX). Fronta se spravuje přes http API.

- **Single worker**: zpracovává se najednou pouze jeden soubor.
- **Priority**: existuje vysoká - `high` a standardní - `standard` priorita. Požadavky s vyšší prioritou jsou zpracovány před standardními, ale aktuálně běžící požadavek už není přerušen.
- **WhisperX (Czech)**: používá se WhisperX s jazykem nastaveným na `cs`, výstup jsou WebVTT (`.vtt`) titulkové soubory.
- **Kontrola Qwen**: po WhisperX průchodu každého souboru Qwen opraví řádky (české i/y a s/z, další chyby ASR a nesmysly, když řeč nebyla česky).
- **Filtrování Halucinací**: odtraňují se některé časté Whisper halucinace (např. “Titulky vytvořil…”, některé URLs, atd.).

---

## Požadavky

- Python 3.10+ (doporučeno)
- Fungující PyTorch + CUDA. Cílová karta je NVIDIA RTX PRO 2000 Blackwell (16 GB). PyTorch musí být build pro CUDA 12.8 nebo novější, aby běžely kernely Blackwell (sm_120).
- `bitsandbytes` 0.45.3 nebo novější (kernely sm_120 pro 4bitový Qwen)

Instalace:

```bash
pip install fastapi uvicorn whisperx torch "transformers>=4.51" "accelerate>=1.0" "bitsandbytes>=0.45.3"
```

---

## Spuštění

From the project root:

```bash
uvicorn app:app --host 0.0.0.0 --port 8093
```

Při spuštění se:

- načtou ASR a alignment modely (defaultně Czech `large-v3`) a zůstanou načtené.
- při prvním požadavku se do volné VRAM načte Qwen ve 4bitové kvantizaci. Další soubory používají oba modely znovu.
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

Text v titulcích je až po kontrole Qwen. Časy segmentů a mluvčí zůstávají z WhisperX.

---

## Kontrola Qwen

U každého souboru se po přepisu, alignmentu, diarizaci a filtru halucinací pošlou řádky do Qwen (`correct_transcript_segments` v `qwen_check.py`). Model má:

- opravit český pravopis, který ASR plete, když se hlásky neliší, hlavně **i/y** (a í/ý) a **s/z** (včetně spodoby znělosti na konci slova), chybějící diakritiku a drobné chyby v hranici slov;
- poznat, že řádek není čeština. Jazyk je vynucený na `cs`, takže řeč v jiném jazyce se vrátí jako nesmysl nebo fonetický přepis. Qwen má větu obnovit v jazyce, kterým se skutečně mluvilo, a souvislou češtinu nechat česky.

Řádky se posílají po malých skupinách a předchozí už opravené řádky jdou do kontextu, aby u dlouhého souboru nerostla cache Qwen. Časy se nemění. Když se jedna skupina nepodaří přečíst, zůstanou u ní texty z Whisperu a požadavek pokračuje. Když selžou všechny skupiny, požadavek skončí chybou a neuloží se nezkontrolovaný přepis.

### Paměť na RTX PRO 2000 Blackwell (16 GB)

Oba modely zůstávají načtené po celou dobu běhu procesu. Qwen se nenačítá tím, že by se Whisper vyhodil z paměti.

| Co zůstává na kartě | Orientačně VRAM |
| --- | --- |
| Whisper large-v3, float16 | ~3 GB |
| Český alignment | ~1 GB |
| Diarizace, pokud se načte | ~1–2 GB |
| Aktivace Whisperu při batch size 4 | jednotky GB ve špičce |
| Qwen3-4B-Instruct-2507, NF4 | ~3–3,5 GB |

Špička má vyjít do 16 GB. Výchozí model je `Qwen/Qwen3-4B-Instruct-2507`, protože je jen textový (bez vision věže) a nepíše thinking stopu. Stejný model v bf16 má asi 8 GB a vedle Whisperu se nevejde. Když na zvolenou kvantizaci není volná VRAM, načtení se odmítne a Whisper se nevyhazuje.

Když požadavek dojde o paměť během Whisperu, snižte `WHISPER_BATCH_SIZE` (výchozí `4`). Whisper se kvůli většímu Qwen neuvolňuje.

### Prostředí

| Proměnná | Výchozí | Význam |
| --- | --- | --- |
| `QWEN_ENABLED` | `1` | `0` vypne Qwen a uloží se přepis z WhisperX |
| `QWEN_MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | Id modelu na Hugging Face |
| `QWEN_QUANT` | `4bit` | `4bit` (výchozí), `8bit`, `bf16` nebo `fp16` |
| `QWEN_CHUNK_SEGMENTS` | `6` | Počet řádků na jedno volání Qwen |
| `QWEN_CONTEXT_SEGMENTS` | `2` | Počet předchozích opravených řádků v kontextu |
| `QWEN_MAX_NEW_TOKENS` | `768` | Strop délky odpovědi |
| `QWEN_MIN_FREE_GB` | `3.5` pro 4bit | Nenačítat Qwen, když je volno méně |
| `QWEN_STRICT` | `0` | `1` shodí požadavek, když některá skupina řádků nejde přečíst |
| `WHISPER_BATCH_SIZE` | `4` | Batch size ASR ve WhisperX |

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

- WhisperX i Qwen zůstávají ve VRAM, dokud aplikace běží, a používají se opakovaně. `torch.cuda.empty_cache()` jen vrátí nevyužité bloky alokátoru, žádný z modelů neodnačte.
- Po každém požadavku se volá i `gc.collect()`, aby se uvolnily dočasné tensory toho souboru.

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

### Výběr ASR enginu

Před spuštěním nastavte `ASR_ENGINE`:

| `ASR_ENGINE` | Model | Instalace navíc |
|---|---|---|
| `whisperx` (výchozí) | WhisperX s `WHISPER_MODEL` (výchozí `large-v3`) | – |
| `whisper-cz` | WhisperX s českým komunitním fine-tunem [`mikr/whisper-large-v3-czech-cv13`](https://huggingface.co/mikr/whisper-large-v3-czech-cv13) | `pip install transformers ctranslate2` (pro jednorázovou konverzi) |
| `canary` | NVIDIA [Canary-1B-v2](https://huggingface.co/nvidia/canary-1b-v2) (CC-BY-4.0), na FLEURS má v češtině nižší WER než Whisper large-v3 | `pip install -U "nemo_toolkit[asr]"`, GPU NVIDIA Ampere nebo novější |

```bash
ASR_ENGINE=canary uvicorn app:app --host 0.0.0.0 --port 8093
```

- **whisper-cz**: při prvním spuštění se model převede do formátu CTranslate2 do `models/whisper-large-v3-czech-cv13-ct2` (`WHISPER_CZ_CT2_DIR`) a dál se načítá odtud. Alignment WhisperX, úvodní prompt i vlastní pravidla fungují stejně jako u `whisperx`.
- **canary**: audio se rozřeže na 20–30 s úseky v nejtišším místě, přepíše se po dávkách (`CANARY_BATCH_SIZE`, výchozí 8) s vlastními časy slov z Canary a seskupí do titulků (konec věty, max. 7 s, 84 znaků nebo pauza 1 s). Alignment WhisperX se nepoužívá a úvodní prompt neplatí (Canary prompt nemá), vlastní pravidla a filtry ano. Diarizace běží, jen pokud je ve stejném prostředí nainstalovaný WhisperX.
- Pro porovnání enginů na stejných souborech spusťte každý s vlastním `OUTPUT_DIR` (výchozí `transcripts`).

---

## API

### Zadat požadavek

`POST /jobs`

```json
{
  "source_path": "/absolute/path/to/media/file.mp4",
  "priority": "standard",
  "language": "cs"
}
```

- `source_path` je **absolutní cesta** k existujícímu audio/video souboru.
- `priority` je `"standard"` - stadardní (výchozí) nebo `"high"` - vysoká.
- `language` (nepovinné) je jazyk mluveného slova jako kód ISO 639-1 (`"cs"`, `"sk"`, `"en"`…). Výchozí je `"cs"`. Kód, který zvolený engine nepodporuje, vrátí 400. Vlastní pravidla se použijí jen pro jazyky uvedené v souboru pravidel (viz níže); pro ostatní jazyky se použije alignment model WhisperX pro daný jazyk, pokud existuje.

Odpověď:

```json
{
  "id": "job-uuid",
  "status": "queued",
  "priority": "standard",
  "language": "cs"
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
  "language": "cs",
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

## Filtrování halucinací a vlastní pravidla

Před zápisem VTT `app.py` vždy:

- vymaže prázdné segmenty a samostatné titulky `...` / `…`,
- vymaže velmi krátké titulky s nereálně dlouhým textem,
- zkrátí smyčky opakování (stejný titulek 3× a víckrát za sebou zůstane jednou, slovo opakované 4× a víckrát v jednom titulku zůstane jednou).

Pravidla pro konkrétní obsah jsou v samostatném Python souboru, výchozí je `custom_rules.py` vedle `app.py` (jinou cestu lze nastavit proměnnou `CUSTOM_RULES`). Git ho ignoruje, takže vlastní slovník a opravy zůstanou soukromé. Začněte z přiloženého příkladu:

```bash
cp custom_rules.example.py custom_rules.py
```

Pokud `custom_rules.py` neexistuje, použije se `custom_rules.example.py`. Soubor definuje:

- `LANGUAGES`: jazyky úloh, na které se pravidla použijí (např. `{"cs"}`). Ostatní úlohy dostanou jen vestavěné filtry výše.
- `INITIAL_PROMPT`: text předaný Whisperu jako úvodní prompt. Whisper z něj přebírá slovní zásobu, pravopis a velká písmena, proto ho pište v domácím stylu se jmény a výrazy typickými pro vaše pořady. Musí být krátký: vejde se jen asi 220 tokenů. Proměnná `WHISPER_INITIAL_PROMPT` ho přepíše (prázdný řetězec ho vypne). Canary ho ignoruje.
- `BLACKLIST` / `DROP_PREFIXES`: titulky, které tyto fráze obsahují / jimi začínají, se vymažou. Příklad obsahuje běžné české halucinace Whisperu jako `Titulky vytvořil...`, některé URL a řetězce `Konec...`.
- `REPLACEMENTS`: dvojice `(regex, náhrada)` použité na každý titulek, např. `aby jsme` → `abychom`. Náhrada může být funkce, jako v `re.sub`. Přidávejte jen opravy, které jsou špatně v jakémkoli kontextu.

---

## Poznámky

- WhisperX modely zůstávají ve VRAM dokud aplikace běží a jsou používány opakovaně.
- Po každém požadavku se čistí cash, aby se nepřeplnila VRAM - (`torch.cuda.empty_cache()`) a `gc.collect()`.

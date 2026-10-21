"""
Example of custom transcription rules.

Copy this file to `custom_rules.py` (ignored by git) and adapt it to your content:
vocabulary prompt, hallucination blacklist and automatic replacements.
If `custom_rules.py` doesn't exist, this example is used. Another path can be set
with the CUSTOM_RULES env var.

All rules apply only to jobs whose language is in LANGUAGES.
"""
import re

# Job languages these rules apply to
LANGUAGES = {"cs"}

# Text passed to Whisper as initial prompt (ignored by Canary). Whisper copies
# vocabulary, spelling and capitalisation from it, so write it in the style you
# want and include names and terms typical for your programmes. Keep it short:
# only ~220 tokens fit and Czech text tokenizes poorly. None disables it; the
# WHISPER_INITIAL_PROMPT env var overrides it.
INITIAL_PROMPT = None
# INITIAL_PROMPT = (
#     "Dobrý den, vítejte u dalšího dílu našeho pořadu. "
#     "Hostem je starostka města Nové Město na Moravě."
# )

# Cues containing any of these phrases (case-insensitive) are dropped.
# Typical Whisper hallucinations on silence or music, learnt from subtitle credits.
BLACKLIST = [
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

# Cues starting with any of these (case-insensitive) are dropped
DROP_PREFIXES = [
    "titulky vytvořil",
]

# (pattern, replacement) pairs applied in order to the text of every cue.
# A pattern is a regex string or a compiled regex; the replacement is a string
# (with \1 group references) or a function taking the match, as in re.sub.
# Only add fixes that are wrong in any context; leave the rest to proofreading.
REPLACEMENTS = [
    # aby jsme -> abychom, by jste -> byste, kdyby jsme -> kdybychom
    (r"\b((?:[Aa]|[Kk]dy)?[Bb])y\s+jsme\b", r"\1ychom"),
    (r"\b((?:[Aa]|[Kk]dy)?[Bb])y\s+jste\b", r"\1yste"),
    # Dubbing credits: "v českém změní" -> "v českém znění"
    (r"\b([Čč]esk(?:é|ém|ého)\s+)změní\b", r"\1znění"),
    # Example of a function replacement keeping the case of the first letter:
    # (re.compile(r"\b[Ss]chromážd"), lambda m: ("S" if m.group(0)[0] == "S" else "s") + "hromážd"),
]

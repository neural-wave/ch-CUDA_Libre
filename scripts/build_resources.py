"""Build the lexicons shipped in ``src/clinanon/resources``.

Run once at development time (needs network); the outputs are committed so that the
anonymizer itself never downloads anything at evaluation time.

Sources (all public / US-government works):
  * First names   - US Census 1990 first-name lists + SSA baby names 1880-2008 (top 1000/yr,
                    via the hadley/data-baby-names mirror because ssa.gov blocks scripted downloads)
  * Surnames      - US Census 2010 surnames (>= 100 occurrences)
  * Drug names    - FDA "Drugs@FDA" products file (brand names + active ingredients)
  * ICD-10-CM     - CDC/NCHS ICD-10-CM 2025 code descriptions (global clinical vocabulary + eponyms)
  * Common words  - words written in lower case mid-sentence in the (properly cased) training notes
  * Places        - GeoNames (CC BY 4.0): cities with >= 15,000 inhabitants, countries, and the states,
                    cantons and regions of the countries most likely in the hidden data

Usage:
    uv run python scripts/build_resources.py
"""

from __future__ import annotations

import csv
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from spacy.lang.en.stop_words import STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "scripts" / "_downloads"
RESOURCES = ROOT / "src" / "clinanon" / "resources"
RAW_DATA = ROOT / "data" / "raw"

SOURCES = {
    "female.first": "https://www2.census.gov/topics/genealogy/1990surnames/dist.female.first",
    "male.first": "https://www2.census.gov/topics/genealogy/1990surnames/dist.male.first",
    "baby-names.csv": "https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv",
    "surnames2010.zip": "https://www2.census.gov/topics/genealogy/2010surnames/names.zip",
    "fda.zip": "https://www.fda.gov/media/89850/download",
    "icd.zip": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2025/"
    "ICD10-CM%20Code%20Descriptions%202025.zip",
    "cities15000.zip": "https://download.geonames.org/export/dump/cities15000.zip",
    "countryInfo.txt": "https://download.geonames.org/export/dump/countryInfo.txt",
    "admin1CodesASCII.txt": "https://download.geonames.org/export/dump/admin1CodesASCII.txt",
}
DICTIONARY = Path("/usr/share/dict/words")  # Webster's 2nd (macOS/BSD); optional

WORD = re.compile(r"[a-z][a-z'-]*[a-z]|[a-z]")


def download_all() -> None:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        target = DOWNLOADS / name
        if target.exists():
            continue
        print(f"downloading {name} ...")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            target.write_bytes(response.read())


def write_tsv(path: Path, rows: dict[str, float], header: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {header}\n")
        for key, value in sorted(rows.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write(f"{key}\t{value:.6g}\n")
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} entries)")


def write_list(path: Path, items: set[str], header: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {header}\n")
        f.writelines(f"{item}\n" for item in sorted(items))
    print(f"wrote {path.relative_to(ROOT)} ({len(items)} entries)")


def build_first_names() -> None:
    """name -> highest observed population share (fraction, 0..1) over all sources."""
    share: dict[str, float] = {}
    for fname in ("female.first", "male.first"):
        for line in (DOWNLOADS / fname).read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                name, pct = parts[0].lower(), float(parts[1]) / 100.0
                share[name] = max(share.get(name, 0.0), pct)
    with (DOWNLOADS / "baby-names.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["name"].lower()
            share[name] = max(share.get(name, 0.0), float(row["percent"]))
    write_tsv(RESOURCES / "first_names.tsv", share, "first name<TAB>max population share (Census 1990, SSA)")


def build_surnames() -> None:
    """surname -> number of people in the 2010 US census."""
    counts: dict[str, float] = {}
    with zipfile.ZipFile(DOWNLOADS / "surnames2010.zip") as z, z.open("Names_2010Census.csv") as raw:
        for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8")):
            name = row["name"].lower()
            if name.isalpha() and name != "all other names":
                counts[name] = float(row["count"])
    write_tsv(RESOURCES / "surnames.tsv", counts, "surname<TAB>count (US Census 2010)")


def build_drugs() -> None:
    """Single-word drug tokens (brand names and active-ingredient words), lower case."""
    tokens: set[str] = set()
    with zipfile.ZipFile(DOWNLOADS / "fda.zip") as z, z.open("Products.txt") as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="latin-1"), delimiter="\t")
        for row in reader:
            for field in (row["DrugName"], row["ActiveIngredient"]):
                for tok in WORD.findall(field.lower()):
                    if len(tok) >= 4 and tok.isalpha():
                        tokens.add(tok)
    write_list(RESOURCES / "drug_tokens.txt", tokens, "drug name tokens (FDA Drugs@FDA brand names + ingredients)")


def build_icd10cm() -> None:
    """Global clinical vocabulary and eponyms from all ICD-10-CM 2025 descriptions."""
    vocab: Counter[str] = Counter()
    eponyms: Counter[str] = Counter()
    with zipfile.ZipFile(DOWNLOADS / "icd.zip") as z, z.open("icd10cm-codes-2025.txt") as raw:
        for line in io.TextIOWrapper(raw, encoding="latin-1"):
            description = line[8:].strip()
            vocab.update(t for t in WORD.findall(description.lower()) if len(t) >= 3)
            # A capitalized word that is not the first word of the description is a proper noun,
            # almost always an eponym ("Hodgkin lymphoma", "Salter-Harris fracture"). Possessives are
            # eponyms wherever they appear, including at the start ("Parkinson's disease").
            for match in re.finditer(r"(?<=[\s(\-])([A-Z][a-z]{2,})(?:'s)?\b", description):
                eponyms[match.group(1).lower()] += 1
            for match in re.finditer(r"\b([A-Z][a-z]{2,})'s\b", description):
                eponyms[match.group(1).lower()] += 1
    write_tsv(RESOURCES / "icd10cm_vocab.tsv", dict(vocab), "token<TAB>count in ICD-10-CM 2025 descriptions")
    write_tsv(RESOURCES / "icd10cm_eponyms.tsv", dict(eponyms), "eponym<TAB>count in ICD-10-CM 2025")


def build_common_words() -> None:
    """Words the (properly cased) notes write in lower case mid-sentence, i.e. ordinary words, not names."""
    counts: Counter[str] = Counter()
    for path in sorted(RAW_DATA.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            note = json.loads(line)["note"]
            # Skip the first token of every sentence/line: it is capitalized for grammatical reasons.
            for match in re.finditer(r"(?<=[a-z,;:)] )([a-z][a-z'-]+)", note):
                counts[match.group(1)] += 1
    common = {w for w, c in counts.items() if c >= 2}
    write_list(RESOURCES / "common_words.txt", common, "lower-case words seen >= 2x mid-sentence in notes")


# Big cities whose name is an everyday word in clinical speech ("in nice shape", "trouble in reading").
_PLACE_BLOCKLIST = {"nice", "reading", "split", "mobile", "surprise", "mission", "independence", "aurora",
                    "providence", "temple", "concord", "orange", "deal", "best", "march", "university",
                    "epsom", "meds"}  # "epsom salts", "a bunch of meds"
_REGION_COUNTRIES = {"US", "CH", "DE", "AT", "IT", "FR", "GB", "ES", "PT", "NL", "BE", "CA", "IE", "LI", "LU"}


def build_places() -> None:
    """Place names (lower case) for the context-gated gazetteer in detectors/rules.py.

    A name that is also an ordinary or clinical word is left out ("Normal", Illinois; "Bath"), unless it is
    a big city (>= 150,000 inhabitants: "Geneva", "Boston"), since Webster's lists many archaic words.
    """
    dictionary = {w.strip() for w in DICTIONARY.read_text().splitlines() if w[:1].islower()} if DICTIONARY.exists() else set()
    common = set((RESOURCES / "common_words.txt").read_text().split()) | set(STOP_WORDS)
    clinical = {w for w, c in _read_tsv(RESOURCES / "icd10cm_vocab.tsv").items() if c >= 2}
    clinical |= set((RESOURCES / "drug_tokens.txt").read_text().split())
    clinical |= set(_read_tsv(RESOURCES / "icd10cm_eponyms.tsv"))  # "Wilson", "Addison" are towns too
    places: dict[str, int] = {}  # name -> population (10**9 for countries and regions)
    alternates: set[str] = set()  # names only known as an alternate name of a big city

    def add(name: str, population: int) -> None:
        name = " ".join(name.lower().replace("’", "'").split())
        if len(name) >= 3 and re.fullmatch(r"[^\W\d_][\w .'\-]*", name):
            places[name] = max(places.get(name, 0), population)
            for saint in ("sankt ", "saint "):  # "Sankt Gallen" is written "St. Gallen"
                if name.startswith(saint):
                    add("st. " + name.removeprefix(saint), population)
                    add("st " + name.removeprefix(saint), population)

    with zipfile.ZipFile(DOWNLOADS / "cities15000.zip") as z, z.open("cities15000.txt") as raw:
        for row in csv.reader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
            population = int(row[14] or 0)
            add(row[1], population)
            add(row[2], population)
            if population >= 100_000:  # English and other Latin-script names of big cities ("Cologne"),
                for alt in row[3].split(","):  # but not airport codes ("BUD", "LED") or ordinary words
                    if alt.isascii() and not alt.isupper() and len(alt) >= 4 and alt.lower() not in places:
                        alternates.add(alt.lower())
                        add(alt, population)
    for line in (DOWNLOADS / "countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            add(line.split("\t")[4], 10**9)
    for line in (DOWNLOADS / "admin1CodesASCII.txt").read_text(encoding="utf-8").splitlines():
        code, name, ascii_name, _ = line.split("\t")
        if code.split(".")[0] in _REGION_COUNTRIES:
            add(name, 10**9)
            add(ascii_name, 10**9)

    def ordinary(token: str) -> bool:
        return token in common or token in clinical or token in dictionary

    kept = set()
    for name, population in places.items():
        tokens = re.findall(r"[^\W\d_]+", name)
        if name in _PLACE_BLOCKLIST or (len(tokens) == 1 and (name in common or name in clinical)):
            continue
        if all(ordinary(t) for t in tokens) and (population < 150_000 or name in alternates):
            continue
        kept.add(name)
    write_list(RESOURCES / "places.txt", kept, "place names, lower case (GeoNames CC BY 4.0, filtered)")


def _read_tsv(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("\t")
            out[key] = float(value)
    return out


if __name__ == "__main__":
    RESOURCES.mkdir(parents=True, exist_ok=True)
    download_all()
    build_first_names()
    build_surnames()
    build_drugs()
    build_icd10cm()
    build_common_words()
    build_places()

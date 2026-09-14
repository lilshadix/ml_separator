"""Deterministic parsers and canonicalisers for the Separation-Archive export.

Every function here is pure and side-effect free so that the whole pipeline can
be re-run and produce byte-identical output.  Each normalisation that actually
changes a value is reported through :class:`NormalizationLog` so that
``normalization_log.json`` can account for it.

Conventions
-----------
* A parser returns ``None`` when it cannot parse, never a guess.
* The raw text is always preserved by the caller alongside the parsed value.
* Synonym merging is opt-in and explicit: a pair of names is only merged when
  the two strings denote the *same substance* beyond reasonable doubt.  Names
  that merely look similar (``octanol`` vs ``1-octanol``, ``CH3Cl`` vs
  ``CHCl3``, ``tetrachloroethane`` vs ``Tetrachloroethylene``) are kept apart
  and routed to the review queue.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import re
import unicodedata

# --------------------------------------------------------------------------
# Blank handling
# --------------------------------------------------------------------------

# Sentinels the archive uses for "no value".  ``- -`` is what the Contact_Time
# column uses; ``-,-`` is what Extractant_inchi uses.
BLANK_TOKENS = frozenset({
    "", "-", "--", "- -", "-,-", "nan", "NaN", "none", "None", "null", "NULL",
    "n/a", "N/A", "na", ".", ",", ", ", "?",
})


def is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().casefold() in {t.casefold() for t in BLANK_TOKENS}


def clean_text(value) -> str | None:
    """Whitespace/unicode normalisation only -- never changes the substance."""
    if is_blank(value):
        return None
    text = unicodedata.normalize("NFKC", str(value))
    # Unify the dash characters that creep in from PDF-scraped names.
    text = text.replace("‐", "-").replace("‑", "-")
    text = text.replace("‒", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def fold(value) -> str | None:
    """Case/space-insensitive matching key for a chemical name."""
    text = clean_text(value)
    if text is None:
        return None
    return re.sub(r"[\s_]+", " ", text.casefold()).strip()


# --------------------------------------------------------------------------
# Normalisation log
# --------------------------------------------------------------------------

@dataclass
class NormalizationLog:
    """Counts every rule application so nothing changes without an entry."""

    events: Counter = field(default_factory=Counter)
    examples: dict[str, list] = field(default_factory=dict)

    def record(self, rule: str, before=None, after=None, limit: int = 5) -> None:
        self.events[rule] += 1
        bucket = self.examples.setdefault(rule, [])
        if len(bucket) < limit and before is not None:
            bucket.append({"before": str(before), "after": None if after is None else str(after)})

    def to_dict(self) -> dict:
        return {
            "rules": dict(sorted(self.events.items())),
            "examples": {k: v for k, v in sorted(self.examples.items())},
        }


# --------------------------------------------------------------------------
# Quantities
# --------------------------------------------------------------------------

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_QTY_RE = re.compile(rf"^\s*({_NUM})\s*([A-Za-zµ%/]*)\s*$")

# Multiplier onto the canonical unit for each recognised unit token.
# Canonical units: concentration -> M (mol/L); time -> min; temperature -> C.
_CONC_UNITS = {"m": 1.0, "mol/l": 1.0, "moll": 1.0, "mm": 1e-3, "µm": 1e-6,
               "um": 1e-6, "nm": 1e-9, "": None}
_TIME_UNITS = {"min": 1.0, "mins": 1.0, "minute": 1.0, "minutes": 1.0,
               "h": 60.0, "hr": 60.0, "hrs": 60.0, "hour": 60.0, "hours": 60.0,
               "s": 1.0 / 60.0, "sec": 1.0 / 60.0, "": None}


@dataclass(frozen=True)
class Quantity:
    value: float
    unit_raw: str | None
    canonical_unit: str
    raw: str


def parse_quantity(text, kind: str, default_unit: str | None = None) -> Quantity | None:
    """Parse ``"0.72 mM"`` / ``"30.0 min"`` / ``"25.0"`` into canonical units.

    ``kind`` is ``concentration`` (canonical M), ``time`` (canonical min) or
    ``temperature`` (canonical C).  Returns ``None`` if the string is blank or
    does not match a single ``<number><unit>`` token -- ranges, inequalities and
    lists are deliberately NOT interpreted here.
    """
    cleaned = clean_text(text)
    if cleaned is None:
        return None
    match = _QTY_RE.match(cleaned)
    if match is None:
        return None
    number = float(match.group(1))
    unit_raw = match.group(2) or None
    unit_key = (unit_raw or "").casefold()

    if kind == "concentration":
        table = _CONC_UNITS
        canonical = "M"
    elif kind == "time":
        table = _TIME_UNITS
        canonical = "min"
    elif kind == "temperature":
        if unit_key in ("", "c", "°c", "degc"):
            return Quantity(number, unit_raw, "C", cleaned)
        if unit_key in ("k",):
            return Quantity(number - 273.15, unit_raw, "C", cleaned)
        if unit_key in ("f",):
            return Quantity((number - 32.0) * 5.0 / 9.0, unit_raw, "C", cleaned)
        return None
    else:  # dimensionless (phase ratio)
        return Quantity(number, unit_raw, "ratio", cleaned) if unit_key == "" else None

    if unit_key not in table:
        return None
    factor = table[unit_key]
    if factor is None:  # unit omitted -> fall back to the column's declared unit
        if default_unit is None:
            return None
        factor = table[default_unit.casefold()]
    return Quantity(number * factor, unit_raw, canonical, cleaned)


# --------------------------------------------------------------------------
# Multi-component splitting
# --------------------------------------------------------------------------

def split_components(text, separators: str = ",") -> list[str]:
    """Split a parallel-list field into its components, preserving order."""
    cleaned = clean_text(text)
    if cleaned is None:
        return []
    parts = re.split(rf"[{re.escape(separators)}]", cleaned)
    return [p.strip() for p in parts if p.strip() and not is_blank(p)]


# --------------------------------------------------------------------------
# Acids
# --------------------------------------------------------------------------
# The archive separates multiple acids with ";" (e.g. "HNO3; oxalic acid").
ACID_CANONICAL = {
    "hno3": "HNO3", "hcl": "HCl", "h2so4": "H2SO4", "hclo4": "HClO4",
    "oxalic acid": "oxalic_acid", "malonic acid": "malonic_acid",
    "lactic acid": "lactic_acid", "tartaric acid": "tartaric_acid",
    "citric acid": "citric_acid",
}

# Acids that dissociate to a coordinating anion; recorded so that a later model
# can distinguish "nitrate medium" from "chloride medium" without re-parsing.
ACID_ANION = {
    "HNO3": "nitrate", "HCl": "chloride", "H2SO4": "sulfate",
    "HClO4": "perchlorate", "oxalic_acid": "oxalate",
    "malonic_acid": "malonate", "lactic_acid": "lactate",
    "tartaric_acid": "tartrate", "citric_acid": "citrate",
}


def normalize_acids(text, log: NormalizationLog | None = None) -> list[str]:
    out = []
    for part in split_components(text, separators=";,"):
        key = fold(part)
        canonical = ACID_CANONICAL.get(key)
        if canonical is None:
            canonical = clean_text(part)
            if log is not None:
                log.record("acid.unmapped", part, canonical)
        elif canonical != clean_text(part) and log is not None:
            log.record("acid.canonicalised", part, canonical)
        if canonical:
            out.append(canonical)
    return out


# --------------------------------------------------------------------------
# Solvents / diluents
# --------------------------------------------------------------------------
# Only merges we are certain about.  Everything else stays distinct.
SOLVENT_SYNONYMS = {
    "chcl3": "chloroform",
    "ccl4": "tetrachloromethane",
    "hydrogenated tetrapropylene": "hydrogenated tetrapropene",
    "isoparl": "isopar l",
    "n-octanol": "1-octanol",
    "nphe": "2-nitrophenyl hexyl ether",
    "2-nphe": "2-nitrophenyl hexyl ether",
    "n-dodecane": "dodecane",
    "dodecane": "dodecane",
    # Abbreviation and spelled-out name for the same diluent, used
    # interchangeably inside a single publication.
    "dipb": "1,4-diisopropylbenzene",
    "tbub": "tert-butylbenzene",
}

# ``TPH`` was previously merged into "total petroleum hydrocarbons". That is
# wrong: in solvent-extraction usage TPH is *tetrapropylene hydrogenated* (a
# branched-C12 kerosene cut), whereas "total petroleum hydrocarbons" is an
# environmental analytical parameter. The archive writes both strings, so they
# are kept as separate keys and referred for adjudication rather than merged
# on a guess.

# Pairs that LOOK mergeable but are chemically distinct or ambiguous.  They are
# kept separate and reported so a chemist can adjudicate.
SOLVENT_AMBIGUOUS = {
    "tph": "TPH = tetrapropylene hydrogenated in extraction usage, but the archive "
           "also writes 'total petroleum hydrocarbons'; not merged",
    "total petroleum hydrocarbons": "an analytical parameter, not a named diluent; "
                                    "kept apart from TPH",
    "ch3cl": "chloromethane is a gas; likely a typo for CHCl3 but not merged",
    "octanol": "positional isomer unspecified (1- vs 2-octanol)",
    "tetrachloroethane": "1,1,2,2- vs 1,1,1,2- isomer unspecified",
    "tetrachloroethylene": "distinct from tetrachloroethane; kept separate",
    "tphkerosene": "single token conflating TPH and kerosene",
    "paraffin": "undefined hydrocarbon cut",
    "hyfrane": "proprietary cut, composition unspecified",
}

# "A with 30 vol% B" / "A with 5% B"
_WITH_RE = re.compile(r"^(?P<a>.+?)\s+with\s+(?P<pct>[\d.]+)\s*(?:vol)?%\s*(?P<b>.+)$", re.I)
# "A 0.7, B 0.3"
_FRAC_RE = re.compile(rf"^(?P<name>.+?)\s+(?P<frac>{_NUM})$")


def _split_solvent_components(text: str) -> list[str]:
    """Split a diluent string on component commas only.

    ``1,4-diisopropylbenzene`` and ``1,1,2,2-tetrachloroethane`` are single
    substances whose names contain IUPAC locant commas.  Splitting blindly on
    "," turned 296 rows into fabricated mixtures whose first "component" was
    the literal string ``"1"``.  A comma separates components only when the
    text on both sides of it contains a letter, which every real mixture in
    this archive satisfies (``kerosene 0.7, 1-octanol 0.3``) and no locant
    does (``1,4-...`` has no letter to the left).
    """
    parts, current = [], ""
    for chunk in text.split(","):
        candidate = f"{current},{chunk}" if current else chunk
        if current and not any(ch.isalpha() for ch in current.split(",")[-1]):
            current = candidate
            continue
        if current and not any(ch.isalpha() for ch in chunk):
            current = candidate
            continue
        if current:
            parts.append(current.strip())
        current = chunk
    if current.strip():
        parts.append(current.strip())
    return [p for p in parts if p]


@dataclass(frozen=True)
class SolventComponent:
    name: str
    fraction: float | None


def parse_solvent(text, log: NormalizationLog | None = None) -> tuple[list[SolventComponent], str]:
    """Split a diluent string into components with volume fractions.

    Returns ``(components, pattern)`` where ``pattern`` is ``single``,
    ``with_percent``, ``fraction_list`` or ``unparsed_multi``.
    """
    cleaned = clean_text(text)
    if cleaned is None:
        return [], "blank"

    match = _WITH_RE.match(cleaned)
    if match:
        pct = float(match.group("pct")) / 100.0
        major = canonical_solvent_name(match.group("a"), log)
        minor = canonical_solvent_name(match.group("b"), log)
        if log is not None:
            log.record("solvent.split_with_percent", cleaned, f"{major}:{1 - pct:.4g}|{minor}:{pct:.4g}")
        return [SolventComponent(major, 1.0 - pct), SolventComponent(minor, pct)], "with_percent"

    parts = _split_solvent_components(cleaned)
    if len(parts) > 1:
        comps, ok = [], True
        for part in parts:
            frac_match = _FRAC_RE.match(part)
            if frac_match is None:
                ok = False
                break
            comps.append(SolventComponent(
                canonical_solvent_name(frac_match.group("name"), log),
                float(frac_match.group("frac")),
            ))
        if ok:
            if log is not None:
                log.record("solvent.split_fraction_list", cleaned,
                           "|".join(f"{c.name}:{c.fraction:g}" for c in comps))
            return comps, "fraction_list"
        return [SolventComponent(canonical_solvent_name(p, log), None) for p in parts], "unparsed_multi"

    return [SolventComponent(canonical_solvent_name(cleaned, log), 1.0)], "single"


def canonical_solvent_name(text, log: NormalizationLog | None = None) -> str:
    cleaned = clean_text(text)
    if cleaned is None:
        return ""
    # Strip parenthetical qualifiers such as "kerosene (solvent 70)".
    stripped = re.sub(r"\s*\([^)]*\)", "", cleaned).strip() or cleaned
    if stripped != cleaned and log is not None:
        log.record("solvent.strip_parenthetical", cleaned, stripped)
    key = fold(stripped)
    target = SOLVENT_SYNONYMS.get(key)
    if target is not None and target != key:
        if log is not None:
            log.record("solvent.synonym_merge", stripped, target)
        return target
    if key in SOLVENT_AMBIGUOUS and log is not None:
        log.record("solvent.ambiguous_kept_separate", stripped, key)
    return key


# --------------------------------------------------------------------------
# DOI
# --------------------------------------------------------------------------
# The archive cites itself on every record it exports.  It is a real reference
# for the *dataset*, but it is not the source of the *measurement*, so it is
# tracked separately from the primary literature DOI.
ARCHIVE_SELF_CITATION_DOI = "10.1021/jacs.5c19738"

# Two DOIs in the archive are malformed and resolve to nothing. Each correction
# was verified against Crossref: the corrected string returns HTTP 200 and a
# topically correct paper, while the recorded string returns 404. Correcting a
# DOI changes no experimental value, and the raw string is preserved in
# ``doi_primary`` -- the repair only ever populates ``doi_primary_corrected``.
DOI_CORRECTIONS: dict[str, dict[str, str]] = {
    "10.1021/acssuschemeng.4c06166x": {
        "corrected": "10.1021/acssuschemeng.4c06166",
        "rule": "stray trailing character",
        "evidence": "recorded DOI -> HTTP 404; corrected -> HTTP 200, "
                    "'Advancing Rare-Earth (4f) and Actinide (5f)...', "
                    "ACS Sustainable Chemistry & Engineering",
    },
    "10.1002/slct.202202610chemistryselect2022,7,e202202610(1of11)\u00a92022wiley-vchgmbh"
    "wileyvchfreitag,09.12.20222237-": {
        "corrected": "10.1002/slct.202202610",
        "rule": "PDF page-footer text concatenated onto the DOI",
        "evidence": "recorded DOI -> HTTP 404; corrected -> HTTP 200, "
                    "'Insights into the Third Phase Formation Behaviour of N,N-...', "
                    "ChemistrySelect",
    },
}


def correct_doi(doi: str | None) -> tuple[str | None, dict | None]:
    """Return ``(corrected_doi, correction_record_or_None)``."""
    if doi is None:
        return None, None
    fix = DOI_CORRECTIONS.get(doi)
    if fix is None:
        return doi, None
    return fix["corrected"], fix

_DOI_RE = re.compile(r"(10\.\d{4,9}/\S+)", re.I)


def normalize_dois(text) -> tuple[list[str], list[str]]:
    """Return ``(dois, other_references)`` with DOIs reduced to bare form."""
    cleaned = clean_text(text)
    if cleaned is None:
        return [], []
    dois, others = [], []
    for part in re.split(r",\s*(?=https?://|doi|DOI|10\.|dx\.)", cleaned):
        part = part.strip().rstrip(".,;")
        if not part:
            continue
        match = _DOI_RE.search(part)
        if match:
            doi = match.group(1).rstrip(".,;").casefold()
            if doi not in dois:
                dois.append(doi)
        elif part not in others:
            others.append(part)
    return dois, others


# --------------------------------------------------------------------------
# Embedded comment fields
# --------------------------------------------------------------------------
# The archive hides several genuinely experimental fields inside
# ``comments_description`` as "Key: value;" pairs.
COMMENT_KEYS = (
    "Data Location", "Additional Comments", "Complexant_Name", "Complexant_SMILES",
    "Complexant_Concentration_M", "Publication_Year", "Title", "Authors",
    "No. of Extractants", "Aqueous Phase Metals", "Metal_Concentration_mM",
    "No. of Metals", "Holdback_Agent_SMILES", "Holdback_Agent_Concentration_M",
    "nitrate concentration(M)", "file",
)
_KEY_ALT = "|".join(re.escape(k) for k in COMMENT_KEYS)
_COMMENT_RE = re.compile(rf"({_KEY_ALT})\s*:\s*(.*?)(?=(?:;\s*(?:{_KEY_ALT})\s*:)|$)", re.S)


def parse_comments(text) -> dict[str, str]:
    cleaned = clean_text(text)
    if cleaned is None:
        return {}
    out: dict[str, str] = {}
    for key, value in _COMMENT_RE.findall(cleaned):
        value = value.strip().rstrip(";").strip()
        if not is_blank(value):
            out[key] = value
    return out

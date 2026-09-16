"""``data/normalize.py`` -- the condition representation (brief section 4.3).

:func:`condition_vector` turns archive rows (``gen19ct.data.load.load_archive``) into a tidy frame
of normalised experimental conditions, one row per input row, index preserved.  It never drops a
row, never imputes, and never reads ``log_D``.

Rules
-----
* **Acid molarity is never converted to pH.**  The archive records acid molarity (``acid_concentration_M``)
  and has no pH column; ``pH`` is emitted as all-NA with ``pH_status`` saying why.  ``log10_acid_M`` is
  the base-10 log of the molarity -- it is *not* ``-pH`` (activity, dissociation and ionic strength are
  unknown), and it is NA for a non-positive molarity.
* Values the archive does not carry at all (pH, saponification degree, organic loading, ionic
  strength) are NA with a ``<name>_status`` reason column -- a clean "not available", never a guess.
* Multi-component systems keep every component.  ``extractant_concentrations_sorted_M`` is a tuple in
  the order of the sorted canonical SMILES, i.e. the order of ``extractant_system_key`` (the archive
  writes ``"A, B"`` and ``"B, A"`` interchangeably, so component order is not stable; the archive's own
  ``extractant_concentrations_M`` array keeps the recorded order).  ``complexant_structure_key`` /
  ``holdback_structure_key`` and their ``*_concentrations_sorted_M`` tuples are built from the
  ``components`` list the same way (sorted by canonical structure, ``name:<name>`` as a fallback when
  the archive gives no structure).  The archive's own name-based ``complexant_signature``
  (``"HEDTA@0.05"``, names as recorded) is passed through verbatim for reference; the condition key
  uses the structure-based fields, because names in this archive are not trustworthy.
* **Acid-molarity semantics check (INFERRED heuristic, flags only).**  ``acid_M_log10_grid`` is True
  when ``-log10(acid M)`` sits on a 0.01 grid (within :data:`LOG10_GRID_TOL`) while the molarity itself
  carries more than :data:`ROUND_SIG_MAX` significant figures (e.g. ``0.000977237 M`` = 10^-3.01,
  ``2.63027e-7 M`` = 10^-6.58).  Such a value is consistent with a quantity entered as pH (or as
  log[H+]) and back-converted, or read off a log-scaled axis -- the archive does not say which.
  ``acid_M_below_1e-3`` marks near-neutral nominal acidity.  Neither flag changes a value; they exist
  so downstream code can refuse to treat these rows as the same local acid response surface.
* ``diluent_family`` is derived from the archive's parsed solvent component names by the rules in
  :data:`COMPONENT_CLASS_EXPLICIT` and :data:`COMPONENT_CLASS_PATTERNS` (first match wins), then
  :func:`diluent_family_from_classes` combines the component classes:

  1. any component classed ``alcohol`` or ``modifier`` (e.g. 1-octanol, isodecanol, Exxal 13, TBP)
     -> ``alcohol_modifier_containing`` (this includes a pure-alcohol diluent);
  2. otherwise every component in one class -> that class (``aliphatic``, ``aromatic``, ``chlorinated``,
     ``nitroaromatic``, ``ionic_liquid``, ``fluorinated``, ``ketone``; ``other`` for the rest);
  3. otherwise (two different non-alcohol classes) -> ``other`` with rule ``mixed_classes``.

  Component precedence: ionic liquid > nitroaromatic > fluorinated > chlorinated > ketone > alcohol >
  modifier > aromatic > aliphatic > other.  So meta-nitrobenzotrifluoride is ``nitroaromatic`` and
  phenyl trifluoromethyl sulfone is ``fluorinated``.  Trade names are matched explicitly and carry a
  basis note; the ones marked ``INFERRED`` rest on general trade-name knowledge that was not checked
  against a source in this repository.
* :func:`condition_key` is the exact normalised condition key (significant-figure rounding, NA kept
  as the literal value ``"NA"``); :func:`experiment_key` adds the publication and the extractant
  system.  Neither includes the metal, the target, or any row identifier, so two rows that differ
  only in the measured metal share a key -- that is what comparable-pair generation needs.  The
  co-present metal list (``aqueous_phase_metals_declared``) is *not* part of the key.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

NA_TOKEN = "NA"
DEFAULT_SIG = 6

#: Unit sanity bounds (brief task D): out-of-range values are *counted*, never altered or dropped.
ACID_M_MIN, ACID_M_MAX = 0.0, 16.0
EXTRACTANT_M_MAX = 5.0
TEMPERATURE_C_MIN, TEMPERATURE_C_MAX = -5.0, 150.0

#: Acid-molarity semantics heuristic (flags only, see module docstring).
LOG10_GRID_STEP = 0.01
LOG10_GRID_TOL = 5e-5
ROUND_SIG_MAX = 3
LOW_ACID_M = 1e-3

DILUENT_FAMILIES: tuple[str, ...] = (
    "aliphatic", "aromatic", "chlorinated", "alcohol_modifier_containing", "nitroaromatic",
    "ionic_liquid", "fluorinated", "ketone", "other",
)
COMPONENT_CLASSES: tuple[str, ...] = (
    "ionic_liquid", "nitroaromatic", "fluorinated", "chlorinated", "ketone", "alcohol", "modifier",
    "aromatic", "aliphatic", "other",
)

#: Exact (lower-cased) component names -> (class, basis).  Checked before the patterns.
COMPONENT_CLASS_EXPLICIT: dict[str, tuple[str, str]] = {
    "tph": ("aliphatic", "abbreviation: hydrogenated tetrapropylene, branched C12 alkane diluent"),
    "tphkerosene": ("aliphatic", "archive token joining TPH and kerosene; both aliphatic"),
    "hydrogenated tetrapropene": ("aliphatic", "systematic name: saturated branched C12 hydrocarbon"),
    "hyfrane": ("aliphatic", "INFERRED: trade name of a TPH-type hydrogenated tetrapropylene diluent"),
    "isopar l": ("aliphatic", "INFERRED: trade name of an isoparaffinic hydrocarbon fluid"),
    "exxsol d60": ("aliphatic", "INFERRED: trade name of a dearomatised aliphatic hydrocarbon fluid"),
    "exxsol d80": ("aliphatic", "INFERRED: trade name of a dearomatised aliphatic hydrocarbon fluid"),
    "exxal 13": ("alcohol", "INFERRED: trade name of isotridecyl alcohol (a phase modifier)"),
    "sulfonated kerosene": ("aliphatic", "INFERRED: kerosene acid-washed to remove aromatics"),
    "avkerosene": ("aliphatic", "archive token for (aviation) kerosene"),
    "total petroleum hydrocarbons": ("aliphatic", "INFERRED: petroleum hydrocarbon cut, aliphatic-dominated"),
    "paraffin": ("aliphatic", "systematic: alkane mixture"),
    "tbp": ("modifier", "tri-n-butyl phosphate recorded as a diluent component"),
    "dmso": ("other", "dimethyl sulfoxide: polar aprotic, none of the named families"),
    "ch3cl": ("chlorinated", "chloromethane as recorded (kept apart from CHCl3 by the archive)"),
}

#: Ordered (class, regex, basis) rules over the lower-cased component name; first match wins.
COMPONENT_CLASS_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("ionic_liquid", r"\[[^\]]*(mim|pyr|py|n\d{3,4}|p\d{3,4})[^\]]*\]|tf2n|ntf2|\bpf6\b|\bbf4\b|ionic liquid",
     "bracketed cation/anion notation or a common IL anion"),
    ("nitroaromatic", r"nitro.*(benz|phen|tolu|xyl)|(benz|phen|tolu|xyl).*nitro",
     "a nitro group on an aromatic ring"),
    ("fluorinated", r"fluor|trifluoromethyl|\bcf3\b", "a fluorinated molecule (non-nitro)"),
    ("chlorinated", r"chlor|\bchcl3\b|\bccl4\b|\bch2cl2\b|\bdce\b", "a chlorinated molecule"),
    ("ketone", r"anone\b|ketone|\bmibk\b|\bacetone\b", "a ketone name"),
    ("alcohol", r"anol\b|alcohol", "an alkanol name"),
    ("modifier", r"tributyl ?phosphate|\bdhoa\b", "a solvating modifier named as a diluent"),
    ("aromatic", r"benz|toluene|xylene|mesitylene|anisole|cumene|naphthalene|phenyl",
     "an aromatic ring without nitro/halogen"),
    ("aliphatic", r"kerosen|dodecane|decane|nonane|octane|heptane|hexane|pentane|alkane|paraffin|"
                  r"isopar|exxsol|shellsol|escaid|petroleum",
     "an alkane or aliphatic hydrocarbon cut"),
)
_COMPILED_PATTERNS = tuple((c, re.compile(p), b) for c, p, b in COMPONENT_CLASS_PATTERNS)

#: Variables emitted as NA because the archive does not record them.
NOT_AVAILABLE: dict[str, str] = {
    "pH": "NOT_RECORDED: the archive has no pH column; acid molarity is never converted to pH (brief 4.3)",
    "saponification_degree": "NOT_RECORDED: the archive has no saponification field",
    "organic_loading": "NOT_RECORDED: the archive has no organic-loading field and it is not reconstructed",
    "ionic_strength_M": "NOT_RECOVERABLE: speciation/activity data needed to compute it are absent",
}

#: Condition-key fields: categorical ones are used verbatim, numeric ones are rounded to ``sig``.
CONDITION_KEY_CATEGORICAL: tuple[str, ...] = (
    "acid_signature", "solvent_key", "modifier_name", "complexant_structure_key", "holdback_structure_key",
)
CONDITION_KEY_NUMERIC: tuple[str, ...] = (
    "acid_concentration_M", "acid_concentration_organic_M", "nitrate_concentration_M",
    "extractant_concentrations_sorted_M", "metal_concentration_M", "phase_ratio_org_aq",
    "modifier_concentration_M", "complexant_concentrations_sorted_M", "holdback_concentrations_sorted_M",
    "temperature_C", "contact_time_min", "shaking_time_min",
)
CONDITION_KEY_FIELDS: tuple[str, ...] = CONDITION_KEY_CATEGORICAL + CONDITION_KEY_NUMERIC

#: Columns :func:`condition_vector` reads from the archive frame.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "components", "acid_primary", "acid_signature", "acid_anion", "acid_concentration_M",
    "acid_concentration_organic_M", "nitrate_concentration_M", "n_organic_extractants",
    "extractant_primary_concentration_M", "metal_concentration_M", "phase_ratio_org_aq",
    "solvent_key", "solvent_primary", "solvent_components", "modifier_name",
    "modifier_concentration_M", "temperature_C", "contact_time_min", "shaking_time_min",
)
#: Read when present (passed through verbatim), never required.
OPTIONAL_COLUMNS: tuple[str, ...] = ("canonical_measurement_id", "complexant_signature")

#: Numeric condition columns with their sanity-check rule, for reports.
UNIT_SANITY_RULES: dict[str, str] = {
    "acid_M_negative": "acid_concentration_M < 0",
    "acid_M_above_16": "acid_concentration_M > 16",
    "extractant_M_above_5": "max(extractant component concentration) > 5",
    "extractant_M_nonpositive": "min(extractant component concentration) <= 0",
    "temperature_below_minus5": "temperature_C < -5",
    "temperature_above_150": "temperature_C > 150",
    "phase_ratio_nonpositive": "phase_ratio_org_aq <= 0",
    "metal_M_negative": "metal_concentration_M < 0",
    "contact_time_nonpositive": "contact_time_min <= 0 or shaking_time_min <= 0",
}


# --------------------------------------------------------------------------- scalar helpers
def is_missing(value: Any) -> bool:
    """True for None, NaN, pd.NA, NaT and the empty / whitespace string."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def format_sig(value: Any, sig: int = DEFAULT_SIG) -> str:
    """Deterministic significant-figure text for a scalar or a sequence of scalars.

    NA -> ``"NA"``; ``-0`` -> ``"0"``; a tuple/list/array -> ``"(a,b)"``; strings pass through stripped.
    """
    if isinstance(value, (tuple, list, np.ndarray)):
        return "(" + ",".join(format_sig(v, sig) for v in value) + ")"
    if is_missing(value):
        return NA_TOKEN
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer, float, np.floating)):
        x = float(value)
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        if x == 0.0:
            return "0"
        return f"{x:.{int(sig)}g}"
    return str(value).strip()


def round_sig(value: Any, sig: int = DEFAULT_SIG) -> float:
    """``value`` rounded to ``sig`` significant figures (NaN stays NaN)."""
    if is_missing(value):
        return float("nan")
    return float(format_sig(float(value), sig))


def _float(value: Any) -> float:
    if is_missing(value):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _log10_positive(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=x.index, dtype=float)
    ok = x > 0
    out[ok] = np.log10(x[ok])
    return out


def significant_figures(value: Any, max_sig: int = 15) -> int | None:
    """Number of significant figures in the shortest round-trip text of ``value`` (None for NA/0/inf)."""
    x = _float(value)
    if np.isnan(x) or x == 0.0 or math.isinf(x):
        return None
    text = f"{abs(x):.{int(max_sig)}g}"
    mantissa = text.split("e")[0].replace(".", "").lstrip("0").rstrip("0")
    return max(1, len(mantissa))


def on_log10_grid(value: Any, step: float = LOG10_GRID_STEP, tol: float = LOG10_GRID_TOL) -> bool:
    """True when ``log10(value)`` is within ``tol`` of a multiple of ``step`` (False for NA/<=0)."""
    x = _float(value)
    if np.isnan(x) or x <= 0 or math.isinf(x):
        return False
    q = math.log10(x) / step
    return abs(q - round(q)) * step < tol


def acid_log10_grid_flag(value: Any) -> bool:
    """The acid-molarity semantics heuristic (module docstring): on the 0.01 log10 grid *and* more
    than :data:`ROUND_SIG_MAX` significant figures, so round molarities (0.1, 0.5, 3) never flag."""
    sf = significant_figures(value)
    return sf is not None and sf > ROUND_SIG_MAX and on_log10_grid(value)


# --------------------------------------------------------------------------- diluent family
def classify_solvent_component(name: Any) -> tuple[str, str]:
    """``(component_class, rule)`` for one parsed solvent component name."""
    if is_missing(name):
        return "other", "missing_name"
    low = str(name).strip().lower()
    if low in COMPONENT_CLASS_EXPLICIT:
        cls, basis = COMPONENT_CLASS_EXPLICIT[low]
        return cls, f"explicit:{low} ({basis})"
    for cls, rx, basis in _COMPILED_PATTERNS:
        if rx.search(low):
            return cls, f"pattern:{cls} ({basis})"
    return "other", "unmatched"


def diluent_family_from_classes(classes: Sequence[str]) -> tuple[str, str]:
    """Combine component classes into ``(diluent_family, rule)`` (module docstring, rules 1-3)."""
    classes = [c for c in classes]
    if not classes:
        return "other", "no_solvent_component"
    if any(c in ("alcohol", "modifier") for c in classes):
        return "alcohol_modifier_containing", "contains_alcohol_or_modifier"
    uniq = sorted(set(classes))
    if len(uniq) == 1:
        return uniq[0], "single_class"
    return "other", "mixed_classes:" + "+".join(uniq)


def diluent_family(components: Any) -> tuple[str, str, str]:
    """``(diluent_family, component_classes, rule)`` for a solvent component list or one name."""
    if isinstance(components, str):
        names = [components]
    elif isinstance(components, (list, tuple, np.ndarray)):
        names = [str(n) for n in components if not is_missing(n)]
    else:
        names = []
    classes = [classify_solvent_component(n)[0] for n in names]
    fam, rule = diluent_family_from_classes(classes)
    return fam, "+".join(classes) if classes else NA_TOKEN, rule


# --------------------------------------------------------------------------- components
def _role_components(components: Any, role: str) -> list[dict]:
    if not isinstance(components, (list, tuple, np.ndarray)):
        return []
    return [c for c in components if isinstance(c, Mapping) and c.get("role") == role]


def _component_identity(comp: Mapping) -> str:
    smi = comp.get("smiles_canonical")
    if not is_missing(smi):
        return str(smi)
    name = comp.get("name")
    return f"name:{str(name).strip()}" if not is_missing(name) else "name:NA"


def _signature(components: Any, role: str) -> tuple[Any, tuple[float, ...]]:
    """(``"|"``-joined sorted identities or NA, concentrations in the same order)."""
    comps = _role_components(components, role)
    if not comps:
        return pd.NA, ()
    pairs = sorted(((_component_identity(c), _float(c.get("concentration_M"))) for c in comps),
                   key=lambda p: (p[0], format_sig(p[1], 17)))
    return "|".join(p[0] for p in pairs), tuple(p[1] for p in pairs)


def _extractant_concentrations(components: Any) -> tuple[float, ...]:
    comps = _role_components(components, "organic_extractant")
    pairs = sorted(((_component_identity(c), _float(c.get("concentration_M"))) for c in comps),
                   key=lambda p: (p[0], format_sig(p[1], 17)))
    return tuple(p[1] for p in pairs)


# --------------------------------------------------------------------------- condition vector
def condition_vector(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy frame of normalised conditions, one row per ``df`` row (same index)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"condition_vector: archive columns missing: {missing}")
    out = pd.DataFrame(index=df.index)
    if "canonical_measurement_id" in df.columns:
        out["canonical_measurement_id"] = df["canonical_measurement_id"]

    # acid identity and molarity (never pH)
    for c in ("acid_primary", "acid_signature", "acid_anion"):
        out[c] = df[c].map(lambda v: pd.NA if is_missing(v) else str(v).strip()).astype("object")
    out["acid_concentration_M"] = pd.to_numeric(df["acid_concentration_M"], errors="coerce").astype(float)
    out["log10_acid_M"] = _log10_positive(out["acid_concentration_M"])
    out["acid_M_log10_grid"] = out["acid_concentration_M"].map(acid_log10_grid_flag).astype(bool)
    out["acid_M_below_1e-3"] = (out["acid_concentration_M"] < LOW_ACID_M).fillna(False).astype(bool)
    out["acid_concentration_organic_M"] = pd.to_numeric(df["acid_concentration_organic_M"],
                                                        errors="coerce").astype(float)
    out["nitrate_concentration_M"] = pd.to_numeric(df["nitrate_concentration_M"], errors="coerce").astype(float)

    # extractants
    out["n_organic_extractants"] = pd.to_numeric(df["n_organic_extractants"], errors="coerce").astype("Int64")
    out["extractant_primary_concentration_M"] = pd.to_numeric(df["extractant_primary_concentration_M"],
                                                              errors="coerce").astype(float)
    out["log10_extractant_primary_M"] = _log10_positive(out["extractant_primary_concentration_M"])
    concs = [_extractant_concentrations(c) for c in df["components"]]
    out["extractant_concentrations_sorted_M"] = pd.Series(concs, index=df.index, dtype="object")
    out["extractant_total_concentration_M"] = [
        float(sum(t)) if t and not any(np.isnan(t)) else np.nan for t in concs]

    # metal, phase ratio
    out["metal_concentration_M"] = pd.to_numeric(df["metal_concentration_M"], errors="coerce").astype(float)
    out["log10_metal_M"] = _log10_positive(out["metal_concentration_M"])
    out["phase_ratio_org_aq"] = pd.to_numeric(df["phase_ratio_org_aq"], errors="coerce").astype(float)

    # diluent
    out["solvent_key"] = df["solvent_key"].map(lambda v: pd.NA if is_missing(v) else str(v)).astype("object")
    out["solvent_primary"] = df["solvent_primary"].map(lambda v: pd.NA if is_missing(v) else str(v)).astype("object")
    fams = [diluent_family(c) for c in df["solvent_components"]]
    out["diluent_family"] = [f[0] for f in fams]
    out["diluent_component_classes"] = [f[1] for f in fams]
    out["diluent_family_rule"] = [f[2] for f in fams]

    # modifier, complexant, holdback
    out["modifier_name"] = df["modifier_name"].map(lambda v: pd.NA if is_missing(v) else str(v).strip()).astype("object")
    out["modifier_concentration_M"] = pd.to_numeric(df["modifier_concentration_M"], errors="coerce").astype(float)
    if "complexant_signature" in df.columns:
        out["complexant_signature"] = df["complexant_signature"].map(
            lambda v: pd.NA if is_missing(v) else str(v).strip()).astype("object")
    else:
        out["complexant_signature"] = pd.Series(pd.NA, index=df.index, dtype="object")
    cx = [_signature(c, "aqueous_complexant") for c in df["components"]]
    out["complexant_structure_key"] = pd.Series([s[0] for s in cx], index=df.index, dtype="object")
    out["complexant_concentrations_sorted_M"] = pd.Series([s[1] for s in cx], index=df.index, dtype="object")
    hb = [_signature(c, "aqueous_holdback") for c in df["components"]]
    out["holdback_structure_key"] = pd.Series([s[0] for s in hb], index=df.index, dtype="object")
    out["holdback_concentrations_sorted_M"] = pd.Series([s[1] for s in hb], index=df.index, dtype="object")

    # physical conditions
    out["temperature_C"] = pd.to_numeric(df["temperature_C"], errors="coerce").astype(float)
    out["contact_time_min"] = pd.to_numeric(df["contact_time_min"], errors="coerce").astype(float)
    out["shaking_time_min"] = pd.to_numeric(df["shaking_time_min"], errors="coerce").astype(float)

    # not available, with reasons
    for name, reason in NOT_AVAILABLE.items():
        out[name] = pd.Series(np.nan, index=df.index, dtype=float)
        out[f"{name}_status"] = reason
    return out


#: Columns only a :func:`condition_vector` frame carries (the archive has none of them).
CONDITION_VECTOR_SENTINELS: tuple[str, ...] = ("pH_status", "diluent_family_rule", "holdback_structure_key")

#: Informational acid-molarity semantics flags (not range violations; values are never altered).
ACID_SEMANTICS_FLAGS: dict[str, str] = {
    "acid_M_log10_grid": f"-log10(acid M) within {LOG10_GRID_TOL} of a {LOG10_GRID_STEP} grid and the molarity has "
                         f"> {ROUND_SIG_MAX} significant figures: consistent with a pH / log[H+] entry back-converted, "
                         "or a log-axis digitisation (INFERRED; the archive does not say which)",
    "acid_M_below_1e-3": f"acid_concentration_M < {LOW_ACID_M:g} (near-neutral nominal acidity)",
}


def is_condition_vector(df: pd.DataFrame) -> bool:
    """True when ``df`` came from :func:`condition_vector` (sentinel columns present)."""
    return all(c in df.columns for c in CONDITION_VECTOR_SENTINELS)


def _as_condition_vector(df: pd.DataFrame) -> pd.DataFrame:
    return df if is_condition_vector(df) else condition_vector(df)


# --------------------------------------------------------------------------- keys
def condition_key(df: pd.DataFrame, sig: int = DEFAULT_SIG, *, exclude: Iterable[str] = (),
                  hashed: bool = False) -> pd.Series:
    """Exact normalised condition key per row.

    ``df`` is an archive frame or a :func:`condition_vector` frame.  The key is the JSON text of
    ``[[field, value], ...]`` over :data:`CONDITION_KEY_FIELDS` minus ``exclude`` -- categorical
    values verbatim, numeric ones via :func:`format_sig`, missing values as ``"NA"`` (so two rows that
    both lack a temperature match, and a row with a temperature never matches one without).
    ``hashed=True`` returns ``"ck_" + sha1(key)[:16]`` instead.
    """
    excl = set(exclude)
    unknown = excl - set(CONDITION_KEY_FIELDS)
    if unknown:
        raise ValueError(f"condition_key: unknown exclude fields {sorted(unknown)}")
    cv = _as_condition_vector(df)
    fields = [f for f in CONDITION_KEY_FIELDS if f not in excl]
    cols = {f: cv[f].tolist() for f in fields}
    keys = []
    for i in range(len(cv)):
        items = [[f, format_sig(cols[f][i], sig)] for f in fields]
        keys.append(json.dumps(items, separators=(",", ":"), ensure_ascii=True))
    s = pd.Series(keys, index=cv.index, dtype="object", name="condition_key")
    return s.map(_hash_key("ck_")) if hashed else s


def experiment_key(df: pd.DataFrame, sig: int = DEFAULT_SIG, *, condition_keys: pd.Series | None = None,
                   exclude: Iterable[str] = (), hashed: bool = False) -> pd.Series:
    """Publication-aware experiment key: ``[g19_publication_id, extractant_system_key, condition_key]``.

    ``df`` must carry ``g19_publication_id`` and ``extractant_system_key`` (the archive frame from
    ``load_archive``).  Pass ``condition_keys`` to reuse keys already computed on the same rows.
    """
    for c in ("g19_publication_id", "extractant_system_key"):
        if c not in df.columns:
            raise KeyError(f"experiment_key: column {c!r} missing (use gen19ct.data.load.load_archive)")
    ck = condition_key(df, sig, exclude=exclude) if condition_keys is None else condition_keys
    if not ck.index.equals(df.index):
        raise ValueError("experiment_key: condition_keys index does not match df")
    keys = [json.dumps([format_sig(p, sig), format_sig(s, sig), k], separators=(",", ":"), ensure_ascii=True)
            for p, s, k in zip(df["g19_publication_id"], df["extractant_system_key"], ck)]
    s = pd.Series(keys, index=df.index, dtype="object", name="experiment_key")
    return s.map(_hash_key("ek_")) if hashed else s


def _hash_key(prefix: str):
    def f(text: str) -> str:
        return prefix + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f


# --------------------------------------------------------------------------- unit sanity
def unit_sanity_flags(cv: pd.DataFrame) -> pd.DataFrame:
    """Boolean flags per row (NA never flags); see :data:`UNIT_SANITY_RULES`."""
    cv = _as_condition_vector(cv)
    acid = cv["acid_concentration_M"]
    ext_max = cv["extractant_concentrations_sorted_M"].map(
        lambda t: np.nanmax(t) if t and not all(np.isnan(t)) else np.nan).astype(float)
    ext_min = cv["extractant_concentrations_sorted_M"].map(
        lambda t: np.nanmin(t) if t and not all(np.isnan(t)) else np.nan).astype(float)
    # fall back to the primary concentration where the components list gave nothing
    ext_max = ext_max.fillna(cv["extractant_primary_concentration_M"])
    ext_min = ext_min.fillna(cv["extractant_primary_concentration_M"])
    t = cv["temperature_C"]
    flags = pd.DataFrame({
        "acid_M_negative": acid < ACID_M_MIN,
        "acid_M_above_16": acid > ACID_M_MAX,
        "extractant_M_above_5": ext_max > EXTRACTANT_M_MAX,
        "extractant_M_nonpositive": ext_min <= 0,
        "temperature_below_minus5": t < TEMPERATURE_C_MIN,
        "temperature_above_150": t > TEMPERATURE_C_MAX,
        "phase_ratio_nonpositive": cv["phase_ratio_org_aq"] <= 0,
        "metal_M_negative": cv["metal_concentration_M"] < 0,
        "contact_time_nonpositive": (cv["contact_time_min"] <= 0) | (cv["shaking_time_min"] <= 0),
    }, index=cv.index)
    return flags.fillna(False).astype(bool)


def unit_sanity_counts(cv: pd.DataFrame) -> dict[str, int]:
    """``{flag: number of rows flagged}`` for :func:`unit_sanity_flags`."""
    return {k: int(v) for k, v in unit_sanity_flags(cv).sum().items()}

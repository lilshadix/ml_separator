"""``data/provenance.py`` -- which brief section 1.4 provenance fields the archive can actually carry.

Brief section 1.4 lists 26 fields every experimental record must retain.  The archive
(``dataset_all_metals/clean/master_clean.parquet``) was not built for that list, so this module
states, field by field, where each one lives and how full it is -- measured, not assumed.

Every field gets exactly one **primary** row and, where the free text carries evidence, **supplement**
rows.  Status vocabulary:

``PRESENT``
    an archive column records the field itself (fill fraction may still be low).
``PROXY``
    only an indirect carrier exists: a nominal value instead of the requested one, a partial set of
    columns, or a regex hit in free text.  A regex hit is evidence that the text *mentions* the field,
    never a parsed value.
``DERIVED_BY_GEN19``
    no archive column, but gen19 computes it reproducibly from archive columns
    (``g19_publication_id``; the nominal loading bound :func:`nominal_max_loading_ratio`).
``ABSENT``
    nothing in the archive carries it.

Fill fractions are reported over all rows, over ``g19_tier == "MODEL"`` rows, and over MODEL rows by
metal group (lanthanide / actinide / other).  Absence of an optional component (modifier, complexant)
is ambiguous in the archive -- "not used" and "not recorded" look the same -- and the notes say so.

Free text.  ``comments_raw`` mixes three formats: a free sentence, the ``"fig3 <note> <x> extraction:
<digitiser>"`` form, and fully structured ``"Data Location: ...; Additional Comments: ...; Title: ...;
Authors: ..."`` / ``"file: ./ST12.json; nitrate concentration(M): ..."`` records.  Regexes are run over
:func:`comment_free_text`, which keeps only the human-written part (the ``Additional Comments`` value of
a structured record), so a paper *title* that mentions "third phase" is not counted as a comment.

Every number in a note is filled from :func:`field_facts` at run time -- no count is hard-coded.
Nothing here reads ``log_D`` except the measured-D field's own fill fraction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

STATUSES = ("PRESENT", "PROXY", "DERIVED_BY_GEN19", "ABSENT")
METAL_GROUPS = ("lanthanide", "actinide", "other")

#: The brief section 1.4 list, in the brief's order.
BRIEF_FIELDS: tuple[str, ...] = (
    "publication/source ID", "DOI", "table/figure/page", "metal", "oxidation state", "extractant",
    "extractant family", "extractant structure identifier", "diluent", "acid", "aqueous composition",
    "extractant concentration", "initial metal concentration", "equilibrium/final pH", "acidity",
    "O/A", "temperature", "contact time", "saponification", "modifiers", "complexants", "loading",
    "measured D", "direct vs reconstructed D", "reported uncertainty", "data-status/quality flag",
)

FREE_TEXT_COLUMN = "comments_free_text"
#: Columns scanned by the free-text evidence patterns (``comments_free_text`` is derived on the fly).
TEXT_COLUMNS: tuple[str, ...] = (FREE_TEXT_COLUMN, "data_location", "ini_comp_raw", "sub_source_file")


# --------------------------------------------------------------------------------------------- #
# Free text
# --------------------------------------------------------------------------------------------- #

_STRUCTURED_RX = re.compile(r"^\s*Data Location:", re.IGNORECASE)
_ADDITIONAL_RX = re.compile(r"Additional Comments:\s*(.*?)\s*;\s*(?:Complexant_Name|Publication_Year|Title|Authors)\s*:",
                            re.IGNORECASE | re.DOTALL)
_SUBSOURCE_RX = re.compile(r"^\s*file:\s*\./ST\w*\.json\s*;", re.IGNORECASE)


def comment_free_text(df: pd.DataFrame) -> pd.Series:
    """The human-written part of ``comments_raw``.

    * structured ``Data Location: ...`` records -> the ``Additional Comments`` value (``None`` when it
      is ``nan``); title, authors, complexant SMILES and year are dropped;
    * ``file: ./ST*.json; nitrate concentration(M): ...`` records -> ``None`` (entirely machine fields);
    * anything else -> the text unchanged.
    """
    if "comments_raw" not in df.columns:
        return pd.Series([None] * len(df), index=df.index, dtype=object, name=FREE_TEXT_COLUMN)
    out: list[str | None] = []
    for t in df["comments_raw"].to_numpy(dtype=object):
        if not isinstance(t, str):
            out.append(None)
        elif _STRUCTURED_RX.match(t):
            m = _ADDITIONAL_RX.search(t)
            v = m.group(1).strip() if m else None
            out.append(None if v is None or v.lower() in ("", "nan", "none") else v)
        elif _SUBSOURCE_RX.match(t):
            out.append(None)
        else:
            out.append(t)
    return pd.Series(out, index=df.index, dtype=object, name=FREE_TEXT_COLUMN)


def with_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` with :data:`FREE_TEXT_COLUMN` added (a shallow copy; the input is never mutated)."""
    if FREE_TEXT_COLUMN in df.columns:
        return df
    return df.assign(**{FREE_TEXT_COLUMN: comment_free_text(df)})


@dataclass(frozen=True)
class TextPattern:
    name: str
    regex: str
    case_sensitive: bool = False
    interpretation: str = ""

    def compiled(self) -> re.Pattern:
        return re.compile(self.regex, 0 if self.case_sensitive else re.IGNORECASE)


#: Regexes run over :data:`TEXT_COLUMNS`.  ``pH`` is case-sensitive on purpose: the case-insensitive
#: form matches ``SO3-Ph-BTP`` (phenyl).
TEXT_PATTERNS: tuple[TextPattern, ...] = (
    TextPattern("pH", r"(?<![A-Za-z])pH(?![A-Za-z])", True, "mentions pH; no pH value column exists"),
    TextPattern("phase_ratio", r"(?<![A-Za-z0-9])(?:O\s*/\s*A|A\s*/\s*O)(?![A-Za-z0-9])|phase ratio|volume ratio|"
                r"equal volumes?|aq\s*/\s*org|org\s*/\s*aq", False, "mentions a phase ratio / equal volumes"),
    TextPattern("saponification", r"sapon|neutrali[sz]", False,
                "mentions saponification / neutralisation of an acidic extractant"),
    TextPattern("equilibrium", r"equilib|aq\s*,\s*eq\b", False,
                "mentions equilibrium (time to equilibrium, or an equilibrium-acidity axis)"),
    TextPattern("uncertainty", r"±|\+/-|\+-\s*\d|uncertaint|error bar|standard deviation|std\.?\s*dev|\bRSD\b|\berror\b",
                False, "mentions a reported uncertainty or error"),
    TextPattern("replicate_tag", r"duplicate|triplicate|replicat|\bduplice\b|\btriplice\b", False,
                "replicate tags (e.g. 'duplice / triplice' series)"),
    TextPattern("digitiser_tag", r"graphreader|webplot|plotdigitizer|digiti[sz]", False,
                "the value was read off a plot by a named digitiser (graphreader1/2/3/M ...)"),
    TextPattern("log_scale_axis", r"log\s*scale", False, "the value was read off a logarithmic axis"),
    TextPattern("percent_extraction", r"%\s*E\b|percent(?:age)? extract|extraction percent|%\s*extract", False,
                "mentions percent extraction (D would be back-calculated)"),
    TextPattern("reconstruction_hint", r"calculat|assum|adjusted|estimat|convert|derived", False,
                "the curator calculated, assumed, adjusted or converted something"),
    TextPattern("figure_or_table_token", r"\bfig(?:ure|urre)?s?\s*\.?\s*S?I?\s*\d|\btables?\s*S?I?\s*\d|\bgraph\s*\d",
                False, "a figure/table token in free text"),
    TextPattern("copied_from_other_source", r"from other paper|same graphs|same figures|taken from ref|"
                r"ref\.?\s*\d+\s+(?:of|in)\s+the\s+(?:paper|manuscript)|republished|off from value shown", False,
                "the curator says the data also appear in / come from another source or figure"),
    TextPattern("tracer_level_metal", r"tracer|trace co?n?c?entration|trace cocentration", False,
                "metal at tracer level; numeric initial concentration often missing"),
    TextPattern("acid_is_total_nitrate", r"total\s+(?:NO3|nitrate)|LiNO3\s+concentration|number is actually", False,
                "the recorded acid molarity is (sometimes) total nitrate, not acid"),
    TextPattern("redox_or_holdback_agent", r"sulph?amate|sulfamate|NH2OH|hydroxylamine|\bHAN\b|hydrazine|KBrO3|"
                r"NaNO2|NH4VO3|K2Cr2O7|Ce\(NO3\)|acetohydroxamic|oxidant", False,
                "an aqueous redox / holding agent recorded only in free text"),
    TextPattern("loading_or_isotherm", r"loading|isotherm", False, "mentions loading or an isotherm"),
    TextPattern("third_phase", r"third\s*phase", False, "mentions third-phase formation"),
    TextPattern("temperature_uncertain", r"unsure about temperature|no temperature given|temperature not given", False,
                "temperature not given / uncertain"),
    TextPattern("ionic_strength", r"ionic strength", False, "mentions ionic strength"),
)
PATTERNS_BY_NAME: dict[str, TextPattern] = {p.name: p for p in TEXT_PATTERNS}


def text_hits(df: pd.DataFrame, pattern: TextPattern, columns: tuple[str, ...] = TEXT_COLUMNS) -> pd.Series:
    """Boolean mask: the pattern matches any of ``columns`` (missing columns are skipped;
    :data:`FREE_TEXT_COLUMN` is derived when absent)."""
    if FREE_TEXT_COLUMN in columns:
        df = with_text_columns(df)
    rx = pattern.compiled()
    hit = np.zeros(len(df), dtype=bool)
    for col in columns:
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=object)
        hit |= np.fromiter((isinstance(v, str) and rx.search(v) is not None for v in vals),
                           dtype=bool, count=len(vals))
    return pd.Series(hit, index=df.index)


def scan_text_metadata(df: pd.DataFrame, model_mask: pd.Series | None = None,
                       columns: tuple[str, ...] = TEXT_COLUMNS + ("comments_raw",), n_examples: int = 3,
                       example_chars: int = 200) -> pd.DataFrame:
    """One row per (pattern, text column): row counts (all / MODEL), distinct texts, examples.

    ``comments_raw`` (the whole string, titles included) is scanned as well as ``comments_free_text`` so
    the effect of dropping the structured fields is visible.  Examples are the most frequent distinct
    matching texts, truncated to ``example_chars``.
    """
    df = with_text_columns(df)
    model_mask = _mask_or_false(df, model_mask)
    rows = []
    for pat in TEXT_PATTERNS:
        for col in columns:
            if col not in df.columns:
                continue
            hit = text_hits(df, pat, (col,))
            texts = df.loc[hit, col].astype(str)
            vc = texts.value_counts()
            ex = _examples(vc, n_examples, example_chars)
            rows.append({
                "section": "text_scan", "evidence_pattern": pat.name, "text_column": col, "regex": pat.regex,
                "case_sensitive": pat.case_sensitive, "n_all": len(df), "n_filled_all": int(hit.sum()),
                "fill_all": _frac(int(hit.sum()), len(df)), "n_model": int(model_mask.sum()),
                "n_filled_model": int((hit & model_mask).sum()),
                "fill_model": _frac(int((hit & model_mask).sum()), int(model_mask.sum())),
                "n_distinct_texts": int(len(vc)), "note": pat.interpretation,
                **{f"example_{i + 1}": ex[i] for i in range(n_examples)},
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- #
# Derived quantities
# --------------------------------------------------------------------------------------------- #

def nominal_max_loading_ratio(df: pd.DataFrame) -> pd.Series:
    """Upper bound on organic loading, metal : primary extractant, from initial values only.

    ``metal_concentration_M / (extractant_primary_concentration_M * phase_ratio_org_aq)`` is the mol
    metal per mol extractant in the organic phase *if every metal ion were extracted*.  It ignores
    stoichiometry, the extent of extraction and any second extractant; NaN whenever one of the three
    inputs is missing or the denominator is not positive.  Nothing is imputed (a missing O/A stays
    missing even though most filled O/A values are 1).
    """
    metal = pd.to_numeric(df["metal_concentration_M"], errors="coerce").to_numpy(dtype=float)
    ext = pd.to_numeric(df["extractant_primary_concentration_M"], errors="coerce").to_numpy(dtype=float)
    oa = pd.to_numeric(df["phase_ratio_org_aq"], errors="coerce").to_numpy(dtype=float)
    denom = ext * oa
    out = np.full(len(df), np.nan)
    ok = np.isfinite(metal) & np.isfinite(denom) & (denom > 0)
    out[ok] = metal[ok] / denom[ok]
    return pd.Series(out, index=df.index, name="g19_nominal_max_loading_ratio")


def d_raw_significant_digits(df: pd.DataFrame) -> pd.Series:
    """Significant digits in the ``D_raw`` string (0 when missing / not numeric)."""
    out = np.zeros(len(df), dtype=int)
    for i, v in enumerate(df["D_raw"].to_numpy(dtype=object)):
        if not isinstance(v, str):
            continue
        mant = re.sub(r"[eE].*$", "", v.strip())
        out[i] = len(re.sub(r"[^0-9]", "", mant).lstrip("0"))
    return pd.Series(out, index=df.index)


def d_raw_float_artefact(df: pd.DataFrame, min_digits: int = 15) -> pd.Series:
    """``D_raw`` written with >= ``min_digits`` significant digits (e.g. ``0.591052825200136``).

    INFERRED interpretation: such strings are machine-serialised floats (a digitiser's output, or D
    back-calculated by a program), not a value typed from a table.  A proxy for "reconstructed", never
    proof; :func:`d_raw_precision_by_location` measures how it splits between figure and table rows.
    """
    return d_raw_significant_digits(df) >= min_digits


def location_kind(df: pd.DataFrame) -> pd.Series:
    """``figure`` / ``table`` / ``main_text`` / ``mixed`` (e.g. 'Figure 10 and Table 1') / ``None``."""
    out: list[str | None] = []
    for v in df["data_location"].to_numpy(dtype=object):
        if not isinstance(v, str) or not v.strip():
            out.append(None)
            continue
        s = v.lower()
        fig = bool(re.search(r"fig|graph", s))
        tab = "table" in s
        out.append("mixed" if fig and tab else "figure" if fig else "table" if tab else
                   "main_text" if "main" in s else "other")
    return pd.Series(out, index=df.index, dtype=object)


def d_raw_precision_by_location(df: pd.DataFrame, model_mask: pd.Series | None = None,
                                min_digits: int = 15) -> pd.DataFrame:
    """Rows with a ``D_raw`` by location kind: how many carry a >= ``min_digits`` machine float."""
    model_mask = _mask_or_false(df, model_mask)
    has = df["D_raw"].notna().to_numpy()
    art = d_raw_float_artefact(df, min_digits).to_numpy()
    kind = location_kind(df).fillna("not_recorded")
    rows = []
    for k in sorted(kind.unique()):
        m = (kind == k).to_numpy() & has
        rows.append({"section": "d_raw_precision_by_location", "evidence_pattern": f"D_raw_sigdigits>={min_digits}",
                     "text_column": "data_location", "location_kind": k,
                     "n_all": int(m.sum()), "n_filled_all": int((m & art).sum()),
                     "fill_all": _frac(int((m & art).sum()), int(m.sum())),
                     "n_model": int((m & model_mask.to_numpy()).sum()),
                     "n_filled_model": int((m & art & model_mask.to_numpy()).sum()),
                     "fill_model": _frac(int((m & art & model_mask.to_numpy()).sum()),
                                         int((m & model_mask.to_numpy()).sum())),
                     "note": "rows with a D_raw string, by data_location kind; filled = machine-float D_raw"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- #
# Facts used in the notes (all computed)
# --------------------------------------------------------------------------------------------- #

def _flag_count(df: pd.DataFrame, flag: str) -> int:
    return int(sum(1 for arr in df["flags"] if arr is not None and flag in list(arr)))


def field_facts(df: pd.DataFrame) -> dict[str, Any]:
    """Counts quoted in the field notes, computed from ``df`` (missing columns give ``"n/a"``)."""
    f: dict[str, Any] = {}

    def safe(name: str, fn: Callable[[], Any]) -> None:
        try:
            f[name] = fn()
        except (KeyError, TypeError, ValueError, AttributeError):
            f[name] = "n/a"

    safe("n_doi_repaired", lambda: int(df["doi_correction_rule"].notna().sum()))
    safe("doi_repair_rules", lambda: "; ".join(f"{k}: {v}" for k, v in
                                               df["doi_correction_rule"].value_counts().items()))
    safe("n_data_location", lambda: int(df["data_location"].notna().sum()))
    safe("data_location_authors", lambda: "; ".join(
        f"{k}: {v}" for k, v in df.loc[df["data_location"].notna(), "entry_author"].value_counts().items()))
    safe("n_entry_authors", lambda: int(df["entry_author"].nunique()))
    safe("n_ox_species_definition", lambda: int(df["metal_oxidation_state_source"].eq("species_definition").sum()))
    safe("n_implausible_ox", lambda: _flag_count(df, "implausible_oxidation_state"))
    def _names_per_structure() -> tuple[int, str]:
        sub = df.dropna(subset=["extractant_primary_smiles", "extractant_primary_name"])
        per = sub.groupby("extractant_primary_smiles")["extractant_primary_name"].nunique().sort_values(
            kind="stable")
        smi = per.index[-1]
        return int(per.iloc[-1]), str(sub.loc[sub["extractant_primary_smiles"] == smi,
                                              "extractant_primary_name"].value_counts().index[0])

    safe("n_names_max_structure", lambda: _names_per_structure()[0])
    safe("name_of_max_structure", lambda: _names_per_structure()[1])
    safe("n_deduced_rows", lambda: int(sum(1 for comps in df["components"] if comps is not None and any(
        c.get("structure_source") == "deduced_by_elimination" for c in comps))))
    safe("n_solvent_keys", lambda: int(df["solvent_key"].nunique()))
    safe("n_multi_component_solvent", lambda: int((pd.to_numeric(df["solvent_n_components"], errors="coerce") > 1).sum()))
    safe("n_acid_signatures", lambda: int(df["acid_signature"].replace("", np.nan).nunique()))
    safe("top_acid", lambda: f"{df['acid_primary'].value_counts().index[0]} "
                             f"({int(df['acid_primary'].value_counts().iloc[0])} rows)")
    safe("n_acid_organic", lambda: int(df["acid_concentration_organic_M"].notna().sum()))
    safe("n_oa", lambda: int(df["phase_ratio_org_aq"].notna().sum()))
    safe("n_oa_one", lambda: int(np.isclose(pd.to_numeric(df["phase_ratio_org_aq"], errors="coerce"), 1.0).sum()))
    safe("n_oa_distinct", lambda: int(df["phase_ratio_org_aq"].nunique()))
    safe("n_modifier", lambda: int(df["modifier_name"].notna().sum()))
    safe("n_complexant", lambda: int(df["complexant_signature"].notna().sum()))
    safe("n_holdback_structure", lambda: int(df["holdback_smiles_canonical"].notna().sum()))
    safe("n_metal_conc", lambda: int(df["metal_concentration_M"].notna().sum()))
    safe("n_log_d", lambda: int(pd.to_numeric(df["log_D"], errors="coerce").notna().sum()))
    safe("n_d_raw", lambda: int(df["D_raw"].notna().sum()))
    safe("readiness_counts", lambda: "; ".join(f"{k}: {v}" for k, v in df["model_readiness"].value_counts().items()))
    safe("n_publications", lambda: int(df["g19_publication_id"].nunique()))
    safe("n_studies", lambda: int(df["g19_study_id"].nunique()))
    safe("publication_status_counts", lambda: "; ".join(
        f"{k}: {v}" for k, v in df["g19_publication_status"].value_counts().items()))
    safe("n_doi_primary_raw", lambda: int(df["doi_primary"].nunique()))
    safe("n_doi_primary_corrected", lambda: int(df["doi_primary_corrected"].nunique()))
    return f


# --------------------------------------------------------------------------------------------- #
# Field registry
# --------------------------------------------------------------------------------------------- #

Mask = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class FieldSpec:
    brief_field: str
    source: str                      # "primary" or "supplement"
    status: str
    archive_columns: tuple[str, ...]
    mask: Mask
    note: str                        # str.format template over field_facts()
    evidence: str = field(default="")  # regex pattern name(s) for text supplements
    example_column: str = field(default="")  # column whose matching values are quoted as examples


def _notna(col: str) -> Mask:
    return lambda d: d[col].notna()


def _text(*names: str, columns: tuple[str, ...] = TEXT_COLUMNS) -> Mask:
    missing = [n for n in names if n not in PATTERNS_BY_NAME]
    if missing:
        raise KeyError(f"unknown text pattern(s) {missing}")
    pats = [PATTERNS_BY_NAME[n] for n in names]

    def f(d: pd.DataFrame) -> pd.Series:
        m = pd.Series(False, index=d.index)
        for p in pats:
            m |= text_hits(d, p, columns)
        return m
    return f


def _never(d: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=d.index)


def _publication_resolved(d: pd.DataFrame) -> pd.Series:
    return d["g19_publication_status"].ne("UNRESOLVED") & d["g19_publication_id"].notna()


def _aqueous_beyond_acid(d: pd.DataFrame) -> pd.Series:
    hold = pd.to_numeric(d["holdback_concentration_M"], errors="coerce").fillna(0) > 0
    mixed_acid = pd.Series([isinstance(s, str) and "|" in s for s in d["acid_signature"]], index=d.index)
    return (d["nitrate_concentration_M"].notna() | d["complexant_signature"].notna()
            | d["holdback_smiles_canonical"].notna() | hold | d["aqueous_phase_metals_declared"].notna() | mixed_acid)


_FIGURE_TOKEN = TextPattern("figure_token", r"\bfig(?:ure|urre)?s?\s*\.?\s*S?I?\s*\d|\bgraph\s*\d")


def _figure_location(d: pd.DataFrame) -> pd.Series:
    kind = location_kind(d)
    fig = kind.isin(["figure", "mixed"])
    text_fig = kind.isna() & text_hits(d, _FIGURE_TOKEN, (FREE_TEXT_COLUMN,))
    return fig | text_fig | _text("digitiser_tag", "log_scale_axis", columns=(FREE_TEXT_COLUMN,))(d)


def _multi_component_solvent(d: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(d["solvent_n_components"], errors="coerce").fillna(0) > 1


def _loading_mask(d: pd.DataFrame) -> pd.Series:
    return nominal_max_loading_ratio(d).notna()


def _contact_or_shaking(d: pd.DataFrame) -> pd.Series:
    return d["contact_time_min"].notna() | d["shaking_time_min"].notna()


def _structure_not_deduced(d: pd.DataFrame) -> pd.Series:
    def ok(comps) -> bool:
        if comps is None or len(comps) == 0:
            return False
        ext = [c for c in comps if c.get("role") == "organic_extractant"]
        return bool(ext) and all(c.get("smiles_canonical") and c.get("structure_source") != "deduced_by_elimination"
                                 for c in ext)
    return d["components"].map(ok).astype(bool)


FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("publication/source ID", "primary", "DERIVED_BY_GEN19",
              ("g19_publication_id", "g19_study_id", "source_record_id", "canonical_measurement_id"),
              _publication_resolved,
              "g19_publication_id ({n_publications} ids; {publication_status_counts}) is the fold group, built by the "
              "gen6 rule over doi_all minus the SAFE self-citation; g19_study_id ({n_studies}) is finer. The raw DOI "
              "is NOT corrected before hashing, so a DOI with a stray trailing character and its corrected form are "
              "different ids (see leakage_doi_multiplicity.csv / leakage_publication_components.csv).",
              example_column="g19_publication_status"),
    FieldSpec("DOI", "primary", "PRESENT", ("doi_primary_corrected", "doi_primary", "doi_all"),
              _notna("doi_primary_corrected"),
              "doi_primary_corrected ({n_doi_primary_corrected} distinct vs {n_doi_primary_raw} raw); {n_doi_repaired} "
              "rows repaired by the archive ({doi_repair_rules}). doi_all also holds the SAFE database self-citation "
              "10.1021/jacs.5c19738, which is not a source.", example_column="doi_primary_corrected"),
    FieldSpec("table/figure/page", "primary", "PRESENT", ("data_location",), _notna("data_location"),
              "data_location ({n_data_location} rows) comes from the structured comments of one entry author "
              "({data_location_authors}). Values are 'Figure n' / 'Table n' / 'Graph n' / 'Main Text'. No page number "
              "anywhere.", example_column="data_location"),
    FieldSpec("table/figure/page", "supplement", "PROXY", ("data_location", "comments_raw"),
              lambda d: d["data_location"].notna() | _text("figure_or_table_token", columns=(FREE_TEXT_COLUMN,))(d),
              "data_location OR a 'fig2' / 'table1' / 'Graph 5' token in the free-text comment (entry format "
              "'<location> <note> <number> extraction:<digitiser>'; examples in text_scan rows)",
              "figure_or_table_token"),
    FieldSpec("metal", "primary", "PRESENT", ("metal_symbol", "metal_raw"), _notna("metal_symbol"),
              "metal_symbol; UO2+2 is expanded to U(VI) by the archive. The per-metal export file (source_file) is NOT "
              "the measured metal (export fan-out; see leakage_metal_alias_risk.csv scope=export_file).",
              example_column="metal_symbol"),
    FieldSpec("oxidation state", "primary", "PRESENT", ("metal_oxidation_state", "metal_oxidation_state_source"),
              _notna("metal_oxidation_state"),
              "metal_oxidation_state (archive field; {n_ox_species_definition} rows from a species definition). Unknown "
              "-> g19_metal_state is None, yet such rows can be MODEL tier: an alias risk for leave-metal-state-out "
              "folds. {n_implausible_ox} rows carry the implausible_oxidation_state flag.",
              example_column="g19_metal_state"),
    FieldSpec("extractant", "primary", "PRESENT", ("extractant_names", "extractant_primary_name", "extractant_name_raw"),
              _notna("extractant_primary_name"),
              "Names are NOT trustworthy: one structure ({name_of_max_structure}) carries {n_names_max_structure} "
              "names (masking agents stored in the name field); the archive 'DEHPA' is an amide. Identity must be "
              "keyed on structure.", example_column="extractant_primary_name"),
    FieldSpec("extractant family", "primary", "ABSENT", (), _never,
              "No family / mechanism / acidic-neutral-basic column in the archive. INFERRED: gen19 derives it from "
              "structure in gen19ct/chemistry/ligands.py (SMARTS rules, descriptors/family_rules.json; another agent's "
              "work, not measured here) and must record it with its own provenance."),
    FieldSpec("extractant structure identifier", "primary", "PRESENT",
              ("extractant_system_key", "extractant_primary_smiles", "extractant_smiles_canonical", "components"),
              _notna("extractant_system_key"),
              "extractant_system_key = order-invariant join of canonical SMILES of every organic extractant.",
              example_column="system_component_class"),
    FieldSpec("extractant structure identifier", "supplement", "PROXY", ("components",), _structure_not_deduced,
              "rows whose every organic extractant has a canonical SMILES NOT tagged structure_source="
              "'deduced_by_elimination' ({n_deduced_rows} rows carry a deduced component: TBP/DHOA/D2EHAA/Br-Cosan per "
              "the archive quality report; Br-Cosan's recorded structure is chemically wrong)"),
    FieldSpec("diluent", "primary", "PRESENT", ("solvent_key", "solvent_components", "solvent_fractions"),
              lambda d: pd.Series([isinstance(s, str) and len(s) > 0 for s in d["solvent_key"]], index=d.index),
              "solvent_key ({n_solvent_keys} mixture-parsed systems); {n_multi_component_solvent} rows have a "
              "multi-component diluent, where a modifier such as 1-octanol is carried as a solvent component rather "
              "than modifier_name.", example_column="solvent_key"),
    FieldSpec("acid", "primary", "PRESENT", ("acid_primary", "acid_signature", "acid_anion"), _notna("acid_primary"),
              "acid_primary / acid_signature ({n_acid_signatures} signatures; top {top_acid}).",
              example_column="acid_signature"),
    FieldSpec("aqueous composition", "primary", "PROXY",
              ("acid_signature", "nitrate_concentration_M", "complexant_signature", "holdback_smiles_canonical",
               "holdback_concentration_M", "aqueous_phase_metals_declared", "ini_comp_raw"),
              _aqueous_beyond_acid,
              "No full aqueous composition (salts, ionic strength, co-metal concentrations). Filled = rows recording "
              "anything beyond the single acid: nitrate salt, complexant, holdback agent (>0 M), declared co-metals or "
              "a mixed-acid signature. An empty value is ambiguous (absent vs not recorded)."),
    FieldSpec("aqueous composition", "supplement", "PROXY", ("comments_raw", "ini_comp_raw"),
              _text("redox_or_holdback_agent", columns=(FREE_TEXT_COLUMN, "ini_comp_raw")),
              "redox / holding agents (ferrous sulfamate, HAN, hydrazine, KBrO3, NaNO2 ...) named only in free text or "
              "in the raw initial-composition string", "redox_or_holdback_agent", FREE_TEXT_COLUMN),
    FieldSpec("extractant concentration", "primary", "PRESENT",
              ("extractant_primary_concentration_M", "extractant_concentrations_M"),
              _notna("extractant_primary_concentration_M"),
              "initial (formal) organic concentration of the primary extractant; per-component values in "
              "extractant_concentrations_M. Some comments say it was not given and could be filled in by the curator "
              "(see the text_scan section)."),
    FieldSpec("initial metal concentration", "primary", "PRESENT", ("metal_concentration_M", "metal_concentration_raw"),
              _notna("metal_concentration_M"),
              "metal_concentration_M ({n_metal_conc} rows; raw strings such as '5 mM'). Tracer-level experiments often "
              "have no numeric value.", example_column="metal_concentration_raw"),
    FieldSpec("initial metal concentration", "supplement", "PROXY", ("comments_raw",),
              _text("tracer_level_metal", columns=(FREE_TEXT_COLUMN,)),
              "qualitative 'tracer level' statements in the free-text comment (no number)", "tracer_level_metal",
              FREE_TEXT_COLUMN),
    FieldSpec("equilibrium/final pH", "primary", "ABSENT", (), _never, "No pH column (initial or equilibrium)."),
    FieldSpec("equilibrium/final pH", "supplement", "PROXY", TEXT_COLUMNS, _text("pH"),
              "case-sensitive 'pH' token in free text (a mention, not a value)", "pH", FREE_TEXT_COLUMN),
    FieldSpec("acidity", "primary", "PRESENT", ("acid_concentration_M", "acid_concentration_organic_M"),
              _notna("acid_concentration_M"),
              "acid_concentration_M is the NOMINAL INITIAL aqueous acid molarity -- not equilibrium acidity and not "
              "pH; in some publications it is total nitrate (supplement). acid_concentration_organic_M (organic-phase "
              "acid, {n_acid_organic} rows) is a separate field."),
    FieldSpec("acidity", "supplement", "PROXY", ("comments_raw",),
              _text("acid_is_total_nitrate", columns=(FREE_TEXT_COLUMN,)),
              "comments saying the recorded acid molarity is total nitrate / a salt concentration",
              "acid_is_total_nitrate", FREE_TEXT_COLUMN),
    FieldSpec("acidity", "supplement", "PROXY", TEXT_COLUMNS, _text("equilibrium"),
              "free-text mention of equilibrium (time to equilibrium, or an 'Haq,eq' axis)", "equilibrium",
              FREE_TEXT_COLUMN),
    FieldSpec("O/A", "primary", "PRESENT", ("phase_ratio_org_aq", "phase_ratio_raw"), _notna("phase_ratio_org_aq"),
              "phase_ratio_org_aq; {n_oa_one} of {n_oa} filled values are 1.0 ({n_oa_distinct} distinct values). "
              "Missing is not 1.", example_column="phase_ratio_org_aq"),
    FieldSpec("O/A", "supplement", "PROXY", TEXT_COLUMNS, _text("phase_ratio"),
              "free-text phase-ratio mention (e.g. 'equal volumes of aq/org phases')", "phase_ratio", FREE_TEXT_COLUMN),
    FieldSpec("temperature", "primary", "PRESENT", ("temperature_C", "temperature_raw"), _notna("temperature_C"),
              "temperature_C; some comments say the temperature was not given (supplement).",
              example_column="temperature_C"),
    FieldSpec("temperature", "supplement", "PROXY", ("comments_raw",),
              _text("temperature_uncertain", columns=(FREE_TEXT_COLUMN,)),
              "comments saying the temperature is unsure / not given", "temperature_uncertain", FREE_TEXT_COLUMN),
    FieldSpec("contact time", "primary", "PRESENT", ("contact_time_min", "shaking_time_min"), _contact_or_shaking,
              "contact_time_min or shaking_time_min (either)."),
    FieldSpec("saponification", "primary", "ABSENT", (), _never,
              "No saponification column. Consistent with the corpus scan that found no PC88A / Cyanex 272 / D2EHPA."),
    FieldSpec("saponification", "supplement", "PROXY", TEXT_COLUMNS, _text("saponification"),
              "free-text saponification / neutralisation mention", "saponification", FREE_TEXT_COLUMN),
    FieldSpec("modifiers", "primary", "PRESENT", ("modifier_name", "modifier_concentration_M"),
              _notna("modifier_name"),
              "modifier_name / modifier_concentration_M ({n_modifier} rows). Empty is ambiguous; many modifiers live "
              "in the diluent string instead (supplement).", example_column="modifier_name"),
    FieldSpec("modifiers", "supplement", "PROXY", ("solvent_n_components", "solvent_components"),
              _multi_component_solvent,
              "multi-component diluent (e.g. 'kerosene:0.7|1-octanol:0.3'), a modifier carried in solvent_key",
              example_column="solvent_key"),
    FieldSpec("complexants", "primary", "PRESENT",
              ("complexant_signature", "complexant_name", "complexant_concentration_M", "holdback_smiles_canonical",
               "holdback_concentration_M"),
              lambda d: d["complexant_signature"].notna() | d["holdback_smiles_canonical"].notna(),
              "aqueous complexant (complexant_signature '<name>@<M>', {n_complexant} rows) or holdback agent structure "
              "({n_holdback_structure} rows). Empty is ambiguous.", example_column="complexant_signature"),
    FieldSpec("loading", "primary", "DERIVED_BY_GEN19",
              ("metal_concentration_M", "extractant_primary_concentration_M", "phase_ratio_org_aq"), _loading_mask,
              "No loading column. gen19ct.data.provenance.nominal_max_loading_ratio = metal_M / (extractant_M x O/A): "
              "an upper bound from initial values (complete extraction, no stoichiometry, primary extractant only); "
              "filled only where all three inputs exist -- nothing imputed."),
    FieldSpec("loading", "supplement", "PROXY", TEXT_COLUMNS, _text("loading_or_isotherm"),
              "free-text mention of loading / extraction isotherms", "loading_or_isotherm", FREE_TEXT_COLUMN),
    FieldSpec("measured D", "primary", "PRESENT", ("D_value", "log_D", "D_raw"),
              lambda d: pd.to_numeric(d["log_D"], errors="coerce").notna(),
              "log_D = log10(D_value) ({n_log_d} rows); null when D is missing or non-positive. The target, never a "
              "feature of itself."),
    FieldSpec("direct vs reconstructed D", "primary", "ABSENT", (), _never,
              "No flag says whether D was reported, digitised from a plot, or back-calculated (from %E, a log axis ...)."),
    FieldSpec("direct vs reconstructed D", "supplement", "PROXY", ("data_location", "comments_raw"), _figure_location,
              "value located in a figure/graph (data_location kind figure/mixed, a 'fig'/'graph' token in the free-text "
              "comment, a digitiser tag or 'log scale'): read off a plot, so digitisation-limited",
              "figure_or_table_token,digitiser_tag,log_scale_axis", "data_location"),
    FieldSpec("direct vs reconstructed D", "supplement", "PROXY", ("comments_raw",),
              _text("reconstruction_hint", "digitiser_tag", "percent_extraction", columns=(FREE_TEXT_COLUMN,)),
              "free-text comment with calculated / assumed / adjusted / converted, a named digitiser, or %E",
              "reconstruction_hint,digitiser_tag,percent_extraction", FREE_TEXT_COLUMN),
    FieldSpec("direct vs reconstructed D", "supplement", "PROXY", ("D_raw",), d_raw_float_artefact,
              "INFERRED: D_raw serialised with >= 15 significant digits (e.g. 0.591052825200136) looks machine-computed "
              "(digitiser output or 10**logD) rather than typed from a table; split by location kind in section "
              "d_raw_precision_by_location", example_column="D_raw"),
    FieldSpec("reported uncertainty", "primary", "ABSENT", (), _never, "No uncertainty / error column."),
    FieldSpec("reported uncertainty", "supplement", "PROXY", TEXT_COLUMNS, _text("uncertainty"),
              "free-text '±' / error / standard deviation mention", "uncertainty", FREE_TEXT_COLUMN),
    FieldSpec("reported uncertainty", "supplement", "PROXY", TEXT_COLUMNS, _text("replicate_tag"),
              "replicate tags in free text (a replicate spread could be estimated; none is reported)", "replicate_tag",
              FREE_TEXT_COLUMN),
    FieldSpec("data-status/quality flag", "primary", "PRESENT",
              ("model_readiness", "duplicate_class", "flags", "in_value_conflict", "has_suspect_flag", "g19_tier"),
              _notna("model_readiness"),
              "model_readiness ({readiness_counts}), duplicate_class, flags, in_value_conflict; g19_tier is derived "
              "from them.", example_column="model_readiness"),
)


def metal_group(df: pd.DataFrame) -> pd.Series:
    """``lanthanide`` / ``actinide`` / ``other`` (any other resolved category) / ``None`` (no metal).

    The archive stores the no-metal category as a null (the corpus report calls it ``none``); both the
    null and the literal string ``"none"`` map to ``None``.
    """
    out: list[str | None] = []
    for c in df["metal_category"].to_numpy(dtype=object):
        if not isinstance(c, str) or c.strip().lower() in ("", "none", "nan"):
            out.append(None)
        elif c in ("lanthanide", "actinide"):
            out.append(c)
        else:
            out.append("other")
    return pd.Series(out, index=df.index, dtype=object)


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def metadata_availability(df: pd.DataFrame, model_mask: pd.Series | None = None,
                          specs: tuple[FieldSpec, ...] = FIELD_SPECS, n_examples: int = 3,
                          facts: dict[str, Any] | None = None) -> pd.DataFrame:
    """One row per :class:`FieldSpec` (``section == "field"``): status, columns, fill fractions (all /
    MODEL / MODEL by metal group), the computed note and up to ``n_examples`` example values.

    ``model_mask`` defaults to ``df["g19_tier"] == "MODEL"`` when that column exists.
    """
    df = with_text_columns(df)
    if model_mask is None:
        model_mask = df["g19_tier"].eq("MODEL") if "g19_tier" in df.columns else pd.Series(False, index=df.index)
    model_mask = _mask_or_false(df, model_mask)
    facts = field_facts(df) if facts is None else facts
    groups = metal_group(df)
    n_all = len(df)
    n_model = int(model_mask.sum())
    group_masks = {g: model_mask & groups.eq(g) for g in METAL_GROUPS}
    rows = []
    seen_primary: set[str] = set()
    for spec in specs:
        if spec.status not in STATUSES:
            raise ValueError(f"{spec.brief_field}: bad status {spec.status}")
        if spec.brief_field not in BRIEF_FIELDS:
            raise ValueError(f"{spec.brief_field}: not a brief section 1.4 field")
        if spec.source == "primary":
            if spec.brief_field in seen_primary:
                raise ValueError(f"{spec.brief_field}: more than one primary row")
            seen_primary.add(spec.brief_field)
        m = pd.Series(np.asarray(pd.Series(spec.mask(df)).fillna(False), dtype=bool), index=df.index)
        k_all, k_model = int(m.sum()), int((m & model_mask).sum())
        row: dict[str, Any] = {
            "section": "field", "brief_field": spec.brief_field,
            "brief_order": BRIEF_FIELDS.index(spec.brief_field) + 1,
            "source": spec.source, "status": spec.status,
            "archive_columns": ";".join(spec.archive_columns), "evidence_pattern": spec.evidence,
            "text_column": spec.example_column,
            "n_all": n_all, "n_filled_all": k_all, "fill_all": _frac(k_all, n_all),
            "n_model": n_model, "n_filled_model": k_model, "fill_model": _frac(k_model, n_model),
        }
        for g, gm in group_masks.items():
            k = int(gm.sum())
            row[f"n_model_{g}"] = k
            row[f"n_filled_model_{g}"] = int((m & gm).sum())
            row[f"fill_model_{g}"] = _frac(int((m & gm).sum()), k)
        row["note"] = spec.note.format_map(_SafeDict(facts))
        ex: list[str] = [""] * n_examples
        if spec.example_column and spec.example_column in df.columns and m.any():
            vals = [_stringify(v) for v in df.loc[m, spec.example_column]]
            vc = pd.Series([v for v in vals if v not in ("nan", "None", "", "<NA>")], dtype=object).value_counts()
            ex = _examples(vc, n_examples, 200)
        for i in range(n_examples):
            row[f"example_{i + 1}"] = ex[i]
        rows.append(row)
    missing = [f for f in BRIEF_FIELDS if f not in seen_primary]
    if missing and specs is FIELD_SPECS:
        raise ValueError(f"brief fields without a primary row: {missing}")
    out = pd.DataFrame(rows)
    return out.sort_values(["brief_order", "source"], kind="stable").reset_index(drop=True)


def availability_table(df: pd.DataFrame, model_mask: pd.Series | None = None) -> pd.DataFrame:
    """The full ``metadata_availability.csv``: field rows, then the text scan, then D_raw precision."""
    df = with_text_columns(df)
    if model_mask is None:
        model_mask = df["g19_tier"].eq("MODEL") if "g19_tier" in df.columns else pd.Series(False, index=df.index)
    parts = [metadata_availability(df, model_mask), scan_text_metadata(df, model_mask),
             d_raw_precision_by_location(df, model_mask)]
    return pd.concat(parts, ignore_index=True, sort=False)


def _examples(vc: pd.Series, n: int, chars: int) -> list[str]:
    ex = [f"[{int(c)}x] {_trunc(t, chars)}" for t, c in vc.head(n).items()]
    return ex + [""] * (n - len(ex))


def _stringify(v: Any) -> str:
    if isinstance(v, (list, tuple, np.ndarray)):
        return "|".join(str(x) for x in v)
    if isinstance(v, float) and np.isnan(v):
        return "nan"
    return str(v)


def _frac(k: int, n: int) -> float:
    return float(k) / n if n else float("nan")


def _mask_or_false(df: pd.DataFrame, mask: pd.Series | None) -> pd.Series:
    if mask is None:
        return pd.Series(False, index=df.index)
    return pd.Series(np.asarray(mask, dtype=bool), index=df.index)


def _trunc(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."

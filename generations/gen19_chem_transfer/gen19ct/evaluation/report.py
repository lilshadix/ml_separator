"""``evaluation/report.py`` -- ``GEN19_REPORT.md``, ``SUMMARY.md``, ``decisions/D02_factorization.md`` and
``tables/claims.json``, GENERATED FROM FILES (brief sections 20, 22, 29, 34; pre-registration sections 9, 10, 15, 16,
17, 19; POST-HOC addendum 1).

Nothing here fits, scores or reads a prediction record.  Every number printed comes from a CSV / JSON / parquet / text
file under the generation directory, cited by path and key, through one :class:`Ledger`: each printed number is
recorded with its raw value, its source path and a key that :func:`resolve_source` can re-read (the build script
re-resolves every number after writing, and ``tests/test_report.py`` does the same on synthetic inputs).  Where an input
file does not exist yet the text says ``not computed (input missing: <path>)`` -- never a placeholder, never an estimate.

Inputs (all relative to the generation root; :data:`INPUTS`):

* pre-seal difficulty and baseline tables (``evaluation/preseal/difficulty.json``, ``tables/preseal_*.csv``);
* the discovery scorer (``tables/discovery_*.csv``, ``evaluation/discovery/contrasts_*.csv``, ``r19_items.csv``,
  ``evaluation/discovery/decisions/{decisions,stop_rule,plan_state,wall_clock}.json``);
* the ladder (``evaluation/ladder/decisions/ladder.json``, ``contrasts_*.csv``, ``evaluation/ladder/M7/metrics.json``);
* H3 (``evaluation/h3/h3_{summary,verdicts,f4}.json``, ``h3_contrasts.csv``); the power check
  (``evaluation/power/power_checks.json``, ``tables/power_kappa.csv``, ``tables/reliability_before_correlation.csv``);
* the figure data of ``scripts/g19_make_figures.py`` (``tables/s1e_error_vs_support.csv``);
* the confirmation run (``evaluation/confirmation/*``: the orchestrator's files; :data:`CONFIRMATION_FILES`) and the
  process step (``evaluation/process/*``, ``tables/process_*.csv``; :data:`PROCESS_FILES`) -- both absent until run;
* the sealed text (``preregistration.md``: the section 17 deviations and the POST-HOC addenda headings), the digests
  (``manifests/prereg_sha256.txt``, ``manifests/confirmation_seeds_sha256.txt``), ``data_audit/dataset_hashes.csv`` and
  the manifests.

Section structure of ``GEN19_REPORT.md``: the ten questions of brief section 34 in order (headline sentence first,
"Under deliberately hidden chemistry, the system can / cannot / has not been shown to ...", then the evidence with
regime columns and both comparators), the S1 / S2 / F1-F6 verdict table (sections 9-10), what was not run and why
(addendum 1, the stop rule, demotions), deviations from the brief (section 17) and the POST-HOC addenda, and
reproducibility.  Verdict labels: every claim carries ``status: discovery / confirmation / not run``; a learned-arm
discovery contrast carries the addendum-1 label ``reduced sensitivity set (addendum 1)`` where it applies; discovery
numbers are labelled optimistically biased (selection half); nothing is called confirmed unless the confirmation files
say so.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gen19ct import paths
from gen19ct.evaluation import calibration as EC
# POST-HOC addendum 4 item 4: the labels the power check may be reported under live with the check (power.REPORTED_LABEL,
# power.NO_POWER_CHECK_LABEL, power.ADDENDUM4_WORDING) so the report and the runner cannot drift apart
from gen19ct.evaluation import power as PW
from gen19ct.evaluation import transfer as ET

SCHEMA = "gen19.report.v1"
NOT_COMPUTED = "not computed (input missing: {path})"
NOT_COMPUTED_KEY = "not computed (input missing: {path} -> {key})"
PREREG = "preregistration.md"
LABEL_DISCOVERY = "discovery, optimistically biased (selection half, seed 104729)"
LABEL_CONFIRMATION = "confirmation (confirmation half, withheld seeds)"
STATUS_DISCOVERY, STATUS_CONFIRMATION, STATUS_NOT_RUN = "discovery", "confirmation", "not run"
ADDENDUM_LABEL = "reduced sensitivity set (addendum 1)"
HEAD_CAN = "Under deliberately hidden chemistry, the system can"
HEAD_CANNOT = "Under deliberately hidden chemistry, the system cannot"
HEAD_UNSHOWN = "Under deliberately hidden chemistry, the system has not been shown to"

#: every input the report can read (logical name -> path relative to the generation root)
INPUTS: dict[str, str] = {
    "difficulty": "evaluation/preseal/difficulty.json",
    "preseal_summary": "tables/preseal_summary.csv",
    "preseal_pair_summary": "tables/preseal_pair_summary.csv",
    "discovery_summary": "tables/discovery_summary.csv",
    "discovery_contrasts": "evaluation/discovery/contrasts_registered.csv",
    "discovery_contrasts_exploratory": "evaluation/discovery/contrasts_exploratory.csv",
    "discovery_r19": "evaluation/discovery/r19_items.csv",
    "decisions": "evaluation/discovery/decisions/decisions.json",
    "stop_rule": "evaluation/discovery/decisions/stop_rule.json",
    "plan_state": "evaluation/discovery/decisions/plan_state.json",
    "wall_clock": "evaluation/discovery/decisions/wall_clock.json",
    "b6_checks": "evaluation/discovery/decisions/b6_checks.json",
    "discovery_coverage": "tables/discovery_coverage_by_domain_status.csv",
    "preseal_coverage": "tables/preseal_coverage_by_domain_status.csv",
    "discovery_pair_summary": "tables/discovery_pair_summary.csv",
    "ladder": "evaluation/ladder/decisions/ladder.json",
    "ladder_wall_clock": "evaluation/ladder/decisions/wall_clock.json",
    "m7_metrics": "evaluation/ladder/M7/metrics.json",
    "h3_summary": "evaluation/h3/h3_summary.json",
    "h3_verdicts": "evaluation/h3/h3_verdicts.json",
    "h3_f4": "evaluation/h3/h3_f4.json",
    "h3_contrasts": "evaluation/h3/h3_contrasts.csv",
    "power_checks": "evaluation/power/power_checks.json",
    "power_kappa": "tables/power_kappa.csv",
    "reliability": "tables/reliability_before_correlation.csv",
    "s1e": "tables/s1e_error_vs_support.csv",
    "figures_index": "evaluation/figures/figures_index.json",
    "confirmation": "evaluation/confirmation/decisions/confirmation.json",
    "confirmation_contrasts": "evaluation/confirmation/contrasts_confirmation.csv",
    "v6_systems": "evaluation/confirmation/v6_systems.csv",
    "v6_pairs": "evaluation/confirmation/v6_pairs.csv",
    "f1_check": "evaluation/confirmation/f1_check.json",
    "process_summary": "evaluation/process/summary.json",
    "process_stability": "evaluation/process/stability.json",
    "process_f5": "evaluation/process/f5.json",
    "process_winners": "tables/process_winners.csv",
    "process_support": "tables/process_support_status.csv",
    "prereg": PREREG,
    "prereg_sha": "manifests/prereg_sha256.txt",
    "seed_commitment": "manifests/confirmation_seeds_sha256.txt",
    "dataset_hashes": "data_audit/dataset_hashes.csv",
    "digest_registry": "manifests/digest_registry.json",
}
CONFIRMATION_FILES: tuple[str, ...] = ("confirmation", "confirmation_contrasts", "v6_systems", "v6_pairs", "f1_check")
PROCESS_FILES: tuple[str, ...] = ("process_summary", "process_stability", "process_f5", "process_winners")
#: brief section 34 item 10: the support category of a recipe from the domain statuses (section 13) of every D it used
#: (``statuses_used`` of ``tables/process_winners.csv``, "|"-separated); first match wins, in this order
SUPPORT_CATEGORY_RULES: tuple[tuple[str, frozenset[str]], ...] = (
    ("unsupported", frozenset({"UNSUPPORTED", "OUTSIDE_TABLE"})),
    ("speculative", frozenset({"FAMILY_EXTRAPOLATION", "CONDITION_EXTRAPOLATION", "CROSS_LIGAND_TRANSFER"})),
    ("transfer-supported", frozenset({"CROSS_METAL_LIGAND_TRANSFER", "CROSS_METAL_TRANSFER"})),
    ("directly supported", frozenset({"IN_DOMAIN", "INTERPOLATION"})),
)
SUPPORT_CATEGORIES: tuple[str, ...] = ("directly supported", "transfer-supported", "speculative", "unsupported",
                                       "unclassified")
#: every token classify_support accepts: the section 13 vocabulary plus the adapter's outside-the-grid pseudo-status
KNOWN_SUPPORT_STATUSES: frozenset[str] = frozenset().union(*(m for _, m in SUPPORT_CATEGORY_RULES))


def classify_support(statuses_used: str) -> str:
    """The support category of one recipe (:data:`SUPPORT_CATEGORY_RULES`): the worst status it used decides.  An
    empty status set is ``unclassified`` (never a support claim); a token outside the known vocabulary raises, so a
    misspelt or new status can never be reported as directly supported (task X finding VL2-02)."""
    used = {s.strip() for s in str(statuses_used).split("|") if s.strip() and s.strip().lower() != "nan"}
    unknown = sorted(used - KNOWN_SUPPORT_STATUSES)
    if unknown:
        raise ValueError(f"classify_support: unknown domain status token(s) {unknown}; known: {sorted(KNOWN_SUPPORT_STATUSES)}")
    if not used:
        return "unclassified"
    for cat, members in SUPPORT_CATEGORY_RULES:
        if used & members:
            return cat
    raise AssertionError("unreachable: every known status belongs to a rule")
#: section 4 averaging unit per design
AVERAGING_UNIT: dict[str, str] = {"V5": "hidden cell", "V5-P": "hidden cell", "V5-PAIR": "hidden cell pair",
                                  "V1": "outer fold (pooled remainder = one unit)", "V2": "metal state",
                                  "V6": "system", "V0": "publication group (diagnostic)"}
CLOSED_FORM_ARMS: frozenset[str] = frozenset({"B0", "B1", "B2", "B3", "B3x", "B3i", "B3l", "B4", "B4x", "B4l", "B7",
                                              "FLAT", "HEAVIER"})
#: the ``variant`` token of the primary setting per design: the pre-seal tables name the fold-file variant (V1__copy,
#: V2__element), the discovery scorer writes ``primary`` for every design
PRESEAL_VARIANT: dict[str, str] = {"V5": "primary", "V1": "copy", "V2": "element", "V5-PAIR": "primary", "V5-P": "base"}


def primary_variant(table: str, design: str) -> str:
    return PRESEAL_VARIANT.get(design, "primary") if table.startswith("preseal") else "primary"
#: the ten questions of brief section 34, in order
QUESTIONS: tuple[tuple[str, str], ...] = (
    ("Q1", "Is there transferable metal-extractant structure in the expanded corpus?"),
    ("Q2", "Can a missing metal x extractant combination be reconstructed?"),
    ("Q3", "Can Pr/Nd selectivity be reconstructed when direct pair data are hidden?"),
    ("Q4", "Does actinide data improve lanthanide prediction?"),
    ("Q5", "Which architecture components actually help?"),
    ("Q6", "How does error grow with chemical distance?"),
    ("Q7", "Is uncertainty calibrated?"),
    ("Q8", "Can the model identify when it does not know?"),
    ("Q9", "Do Gen18 process recommendations remain good after uncertainty propagation?"),
    ("Q10", "Which Pr/Nd process recommendations are directly supported, transfer-supported, speculative or unsupported?"),
)
_ADDENDUM_RE = re.compile(r"^## POST-HOC addendum (\d+) \(([^)]*)\)\s*$")
_DEVIATION_RE = re.compile(r"^(\d+)\. \*\*(.+?)\*\*")


# --------------------------------------------------------------------------------------------- #
# the ledger: every printed number with its source
# --------------------------------------------------------------------------------------------- #

@dataclass
class Number:
    printed: str
    raw: Any
    source: str
    key: str
    kind: str            # value | count | constant | text
    rule: str = ""


def fmt(v: Any, nd: int = 3) -> str:
    """Fixed-precision text of a number (the only formatting the report uses)."""
    if v is None:
        return "NaN"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "NaN"
    if nd == 0:
        return str(int(round(f)))
    return f"{f:.{nd}f}"


class Ledger:
    """Records every number the report prints, every input it touched and every input it found missing."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.numbers: list[Number] = []
        self.inputs: dict[str, bool] = {}
        self.missing_inputs: list[str] = []

    # -- numbers ------------------------------------------------------------------------------ #
    def value(self, raw: Any, source: str, key: str, nd: int = 3) -> str:
        s = fmt(raw, nd)
        self.numbers.append(Number(s, raw, source, key, "value"))
        return s

    def count(self, n: int, source: str, rule: str) -> str:
        s = str(int(n))
        self.numbers.append(Number(s, int(n), source, f"count:{rule}", "count", rule))
        return s

    def constant(self, raw: Any, key: str, nd: int = 3, source: str = PREREG) -> str:
        s = fmt(raw, nd)
        self.numbers.append(Number(s, raw, source, key, "constant", "registered constant; the literal appears in the text"))
        return s

    def text(self, s: str, source: str, key: str) -> str:
        """A number-like token copied verbatim from a text file (a digest prefix, a seed)."""
        self.numbers.append(Number(str(s), str(s), source, key, "text"))
        return str(s)

    def missing(self, source: str, key: str | None = None) -> str:
        if source not in self.missing_inputs:
            self.missing_inputs.append(source)
        return NOT_COMPUTED.format(path=source) if key is None else NOT_COMPUTED_KEY.format(path=source, key=key)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"printed": n.printed, "raw_value": n.raw, "source_path": n.source, "source_key": n.key,
                              "kind": n.kind, "rule": n.rule} for n in self.numbers],
                            columns=["printed", "raw_value", "source_path", "source_key", "kind", "rule"])

    # -- inputs ------------------------------------------------------------------------------- #
    def path(self, name: str) -> Path:
        return self.root / INPUTS[name]

    def exists(self, name: str) -> bool:
        p = self.path(name)
        ok = p.exists()
        self.inputs[INPUTS[name]] = ok
        return ok

    def json(self, name: str) -> dict[str, Any] | None:
        if not self.exists(name):
            return None
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    def csv(self, name: str) -> pd.DataFrame | None:
        if not self.exists(name):
            return None
        return pd.read_csv(self.path(name), dtype=str, keep_default_na=False)

    def rel(self, name: str) -> str:
        return INPUTS[name]


# --------------------------------------------------------------------------------------------- #
# source keys: build and re-resolve
# --------------------------------------------------------------------------------------------- #

def where_key(filters: Mapping[str, Any], column: str) -> str:
    """``where:a=1|b=x;column:point`` -- the CSV key grammar of the ledger."""
    return "where:" + "|".join(f"{k}={v}" for k, v in filters.items()) + f";column:{column}"


def json_get(body: Any, key: str) -> Any:
    """``a.b[2].c`` on a parsed JSON object (``a["k.with.dots"].c`` quotes a key that contains dots); ``KeyError``
    when absent."""
    cur = body
    for part in re.findall(r'\["[^"]*"\]|\[\d+\]|[^.\[\]]+', key):
        if part.startswith('["'):
            name = part[2:-2]
            if not isinstance(cur, Mapping) or name not in cur:
                raise KeyError(key)
            cur = cur[name]
        elif part.startswith("["):
            cur = cur[int(part[1:-1])]
        else:
            if not isinstance(cur, Mapping) or part not in cur:
                raise KeyError(key)
            cur = cur[part]
    return cur


def jkey(name: str) -> str:
    """A JSON key segment safe for :func:`json_get` (quoted when it contains a dot or a bracket)."""
    return f'["{name}"]' if re.search(r"[.\[\]]", str(name)) else str(name)


def jpath(*parts: str) -> str:
    """Join key segments into a :func:`json_get` path (``jpath("s2d", "p0.95_r0.80", "passes")``)."""
    out = ""
    for p in parts:
        seg = jkey(p)
        out += seg if (seg.startswith("[") or not out) else "." + seg
    return out


def _csv_where(df: pd.DataFrame, where: str) -> pd.DataFrame:
    sub = df
    for clause in where.split("|"):
        if not clause:
            continue
        col, _, val = clause.partition("=")
        if col not in sub.columns:
            raise KeyError(col)
        sub = sub[sub[col].astype(str) == val]
    return sub


def resolve_source(root: Path, source: str, key: str) -> Any:
    """Re-read the value a ledger entry cites.  JSON: a dotted path; CSV: ``where:...;column:...`` (exactly one row)
    or ``count:where:...`` / ``count:rows`` (a row count); text: the key is a literal that must occur in the file."""
    p = Path(root) / source
    if not p.exists():
        raise FileNotFoundError(source)
    suf = p.suffix.lower()
    if key.startswith("count:"):
        rest = key[len("count:"):]
        if p.is_dir() and rest.startswith("glob:"):
            return len([q for q in p.glob(rest[len("glob:"):]) if q.is_file()])
        if suf == ".csv":
            df = pd.read_csv(p, dtype=str, keep_default_na=False)
            if rest == "rows":
                return len(df)
            if rest.startswith("where:"):
                return len(_csv_where(df, rest[len("where:"):]))
        if suf == ".json" and rest.startswith("len:"):
            return len(json_get(json.loads(p.read_text(encoding="utf-8")), rest[len("len:"):]))
        if suf == ".md" and rest in TEXT_COUNT_RULES:
            return TEXT_COUNT_RULES[rest](p.read_text(encoding="utf-8"))
        raise KeyError(key)
    if suf == ".json":
        return json_get(json.loads(p.read_text(encoding="utf-8")), key)
    if suf == ".csv":
        m = re.match(r"^where:(.*);column:([^;]+)$", key)
        if not m:
            raise KeyError(key)
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        sub = _csv_where(df, m.group(1))
        if len(sub) != 1:
            raise KeyError(f"{key}: {len(sub)} rows")
        return sub.iloc[0][m.group(2)]
    text = p.read_text(encoding="utf-8").replace("−", "-")          # the sealed text writes minus as U+2212
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)                          # 10,000 -> 10000
    if key in text:
        return key
    raise KeyError(f"{key!r} not in {source}")


def literal_in_file(root: Path, source: str, literal: str) -> bool:
    """Whether ``literal`` occurs in the file's text (minus U+2212 and thousands separators normalised)."""
    p = Path(root) / source
    if not p.exists():
        raise FileNotFoundError(source)
    text = p.read_text(encoding="utf-8").replace("−", "-")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    return str(literal) in text


def verify_numbers(root: Path, numbers: pd.DataFrame) -> pd.DataFrame:
    """Re-resolve every ledger row from its cited source; ``ok`` per row (the build script's self-check).  ``value`` and
    ``count`` rows are re-read through :func:`resolve_source`; ``constant`` and ``text`` rows are literals that must occur
    in the cited file (the sealed text for constants; a digest file or a JSON list for text)."""
    out = []
    for _, r in numbers.iterrows():
        rec = {"printed": r["printed"], "source_path": r["source_path"], "source_key": r["source_key"], "kind": r["kind"]}
        try:
            if r["kind"] in ("text", "constant"):
                ok = literal_in_file(root, str(r["source_path"]), str(r["printed"]))
                if not ok and str(r["source_path"]).endswith(".json"):
                    got = resolve_source(root, str(r["source_path"]), str(r["source_key"]))
                    ok = str(r["printed"]) in str(got) or _same_number(got, r["printed"])
                rec.update(resolved="literal present" if ok else "literal absent", ok=bool(ok))
                out.append(rec)
                continue
            got = resolve_source(root, str(r["source_path"]), str(r["source_key"]))
            if r["kind"] == "count":
                ok = int(got) == int(float(r["printed"]))
            else:
                ok = _same_number(got, r["printed"])
            rec.update(resolved=str(got)[:80], ok=bool(ok))
        except (KeyError, FileNotFoundError, ValueError, TypeError, IndexError) as exc:
            rec.update(resolved=f"{type(exc).__name__}: {exc}"[:120], ok=False)
        out.append(rec)
    return pd.DataFrame(out, columns=["printed", "source_path", "source_key", "kind", "resolved", "ok"])


def _same_number(raw: Any, printed: str) -> bool:
    try:
        f = float(raw)
        p = float(printed)
    except (TypeError, ValueError):
        return str(raw) == str(printed)
    if not math.isfinite(f):
        return printed == "NaN"
    nd = len(printed.split(".")[1]) if "." in printed else 0
    return abs(f - p) <= 0.5 * 10 ** (-nd) + 1e-12


# --------------------------------------------------------------------------------------------- #
# readers of the specific inputs
# --------------------------------------------------------------------------------------------- #

def _pick(df: pd.DataFrame, filters: Mapping[str, Any], prefer: Mapping[str, str] | None = None) -> pd.Series | None:
    """The unique row matching ``filters``; with several, the first whose ``prefer`` column starts with the value."""
    sub = df
    for k, v in filters.items():
        if k not in sub.columns:
            return None
        sub = sub[sub[k].astype(str) == str(v)]
    if sub.empty:
        return None
    if len(sub) > 1 and prefer:
        for k, v in prefer.items():
            if k in sub.columns:
                s2 = sub[sub[k].astype(str).str.startswith(v)]
                if len(s2):
                    sub = s2
    return sub.iloc[0]


def _row_key(row: pd.Series, cols: Sequence[str], column: str) -> str:
    return where_key({c: row[c] for c in cols if c in row.index}, column)


SUMMARY_FILTER_COLS: tuple[str, ...] = ("job", "design", "variant", "half", "arm", "seed", "metric", "aggregation",
                                        "stratum", "scoring_filter", "status", "unit_reading", "seed_set",
                                        "threshold_setting", "unit_variant")


def summary_value(L: Ledger, name: str, *, arm: str, design: str, metric: str, variant: str = "primary",
                  half: str = "selection", seed: str | None = None, aggregation: str = "unit_macro",
                  stratum: str = "all", nd: int = 3) -> tuple[str, str | None]:
    """A metric of a tidy summary table (``tables/preseal_summary.csv`` / ``tables/discovery_summary.csv``): printed
    text plus the resolvable key (``None`` when absent).  ``seed`` None takes any seed row (deterministic arms)."""
    df = L.csv(name)
    if df is None:
        return L.missing(L.rel(name)), None
    filters: dict[str, Any] = {"arm": arm, "design": design, "variant": variant, "half": half, "metric": metric,
                               "aggregation": aggregation, "stratum": stratum}
    if "scoring_filter" in df.columns:
        filters["scoring_filter"] = "none"
    if "status" in df.columns:
        filters["status"] = "registered"
    if seed is not None:
        filters["seed"] = seed
    row = _pick(df, filters, prefer={"unit_reading": "registered"})
    if row is None:
        return L.missing(L.rel(name), where_key(filters, "value")), None
    key = _row_key(row, SUMMARY_FILTER_COLS, "value")
    return L.value(row["value"], L.rel(name), key, nd), key


CONTRAST_KEY_COLS: tuple[str, ...] = ("key", "cluster_unit")


def contrast_row(L: Ledger, name: str, key: str, cluster_unit: str | None = None) -> pd.Series | None:
    df = L.csv(name)
    if df is None or "key" not in df.columns:
        return None
    sub = df[df["key"].astype(str) == key]
    if cluster_unit is not None:
        sub = sub[sub["cluster_unit"].astype(str) == cluster_unit]
    elif "primary_cluster_unit" in sub.columns:
        sub = sub[sub["primary_cluster_unit"].astype(str).str.lower() == "true"]
    return None if sub.empty else sub.iloc[0]


def contrast_cell(L: Ledger, name: str, row: pd.Series | None, column: str, nd: int = 3) -> str:
    """One cited number of a contrast row (``key`` + ``cluster_unit`` identify the row)."""
    if row is None or column not in row.index or row[column] in ("", "nan", "None"):
        return "NaN" if row is not None else L.missing(L.rel(name))
    return L.value(row[column], L.rel(name), _row_key(row, CONTRAST_KEY_COLS, column), nd)


def _s(row: pd.Series | None, col: str, default: str = "") -> str:
    if row is None or col not in row.index:
        return default
    v = str(row[col])
    return default if v in ("", "nan", "None") else v


def _json_value(L: Ledger, name: str, body: Mapping[str, Any] | None, key: str, nd: int = 3) -> str:
    if body is None:
        return L.missing(L.rel(name))
    try:
        v = json_get(body, key)
    except (KeyError, IndexError, TypeError):
        return L.missing(L.rel(name), key)
    if v is None:
        return L.missing(L.rel(name), key)
    if isinstance(v, bool):
        return L.text(str(v), L.rel(name), key)
    if isinstance(v, int):
        return L.value(v, L.rel(name), key, 0)
    return L.value(v, L.rel(name), key, nd)


def _json_str(L: Ledger, name: str, body: Mapping[str, Any] | None, key: str) -> str:
    if body is None:
        return L.missing(L.rel(name))
    try:
        v = json_get(body, key)
    except (KeyError, IndexError, TypeError):
        return L.missing(L.rel(name), key)
    return "None" if v is None else str(v)


# --------------------------------------------------------------------------------------------- #
# the deployed predictor, the claim table
# --------------------------------------------------------------------------------------------- #

def deployed_predictor(decisions: Mapping[str, Any] | None, ladder: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``h3.deployed_configuration`` on the scorer's decisions AND the ladder runner's ``ladder.json`` (imported lazily;
    the same rule as the H3 runner, so Q5's kept steps and the deployed arm of Q2 / Q6 / Q7 / F3 / S1(d) agree; task X
    finding V-01).  Without ``ladder.json`` the rule is pending and every deployed-arm section prints not computed."""
    if decisions is None:
        return {"arm": None, "status": "not computed", "basis": NOT_COMPUTED.format(path=INPUTS["decisions"])}
    from gen19ct.evaluation import h3 as H3
    return H3.deployed_configuration(decisions, ladder)


def parameter_status(arm: str) -> str:
    if arm in CLOSED_FORM_ARMS:
        return "closed form (no fitted parameter)"
    if arm in ("B6", "B6r0"):
        return "ridge / ALS factorisation, inner-tuned per outer fold (addendum 1 simultaneous inner design)"
    return "learned, inner-tuned per outer fold (addendum 1 items 1-2: 3 simultaneous inner folds, seed 104729)"


def claim_status(key: str, contrasts: pd.DataFrame | None, confirmation: Mapping[str, Any] | None) -> str:
    if confirmation is not None:
        for c in confirmation.get("claims") or []:
            if str(c.get("contrast_key") or c.get("contrast")) == key:
                return STATUS_CONFIRMATION
    if contrasts is not None and "key" in contrasts.columns and (contrasts["key"].astype(str) == key).any():
        return STATUS_DISCOVERY
    return STATUS_NOT_RUN


def build_claims(L: Ledger) -> list[dict[str, Any]]:
    """``tables/claims.json``: every registered contrast of section 19 (from the scorer's accounting) with its status,
    comparator, Delta, interval, R19 verdicts and source files; plus the S1 / S2 components as claims."""
    dec = L.json("decisions")
    con = L.csv("discovery_contrasts")
    conf = L.json("confirmation")
    src_con, src_dec = L.rel("discovery_contrasts"), L.rel("decisions")
    claims: list[dict[str, Any]] = []
    rows = (dec or {}).get("registered_family_accounting", {}).get("contrasts") or []
    if not rows:                                                     # the accounting is absent: list the family from code
        from gen19ct.evaluation import discovery as D
        rows = [dict(r) for r in D.REGISTERED_FAMILY_TABLE] + [
            {"family": "confirmation", "contrast": c, "design": "V6", "status": "confirmation_only"} for c in
            D.CONFIRMATION_ONLY_CONTRASTS]
    entries: list[tuple[str, str, str, str, str | None]] = []
    for r in rows:
        fam, contrast, design = r["family"], r["contrast"], r["design"]
        key = f"{contrast}@{design}" + ("#ladder" if fam == "ladder" else "")
        if fam == "ladder":
            key = _ladder_key(con, contrast.split(" ")[0], design) or key
        entries.append((key, fam, contrast, design, r.get("status")))
    if con is not None and "key" in con.columns:                    # evaluated contrasts the accounting does not list
        listed = {e[0] for e in entries}
        for _, r in con.drop_duplicates("key").iterrows():
            if str(r["key"]) not in listed and str(r.get("bh_family", "registered")) == "registered":
                entries.append((str(r["key"]), str(r["family"]), str(r["contrast"]), str(r["design"]), "evaluated"))
    for key, fam, contrast, design, scorer_status in entries:
        row = contrast_row(L, "discovery_contrasts", key) if con is not None else None
        status = claim_status(key, con, conf)
        cand, comp = (_s(row, "candidate"), _s(row, "comparator")) if row is not None else _split_contrast(contrast)
        claim = {"claim": contrast, "family": fam, "design": design, "key": key, "status": status,
                 "scorer_status": scorer_status, "candidate": cand, "comparator": comp,
                 "comparators_required": {"constant_baseline": "B0", "cheapest_sensible_alternative":
                                          _cheapest_alternative(design)},
                 "half": "selection" if status == STATUS_DISCOVERY else ("confirmation" if status == STATUS_CONFIRMATION
                                                                          else None),
                 "seeds": _s(row, "seeds_evaluated") or None, "averaging_unit": AVERAGING_UNIT.get(design),
                 "parameter_status": parameter_status(cand) if cand else None,
                 "delta": _num(_s(row, "point")), "margin": _num(_s(row, "margin")),
                 "interval_percentile_95": [_num(_s(row, "percentile_low")), _num(_s(row, "percentile_high"))],
                 "interval_bca_95": [_num(_s(row, "bca_low")), _num(_s(row, "bca_high"))],
                 "p_two_sided": _num(_s(row, "p_two_sided")), "p_bh": _num(_s(row, "p_bh")),
                 "p_bh_full_family": _num(_s(row, "p_bh_full_family")),
                 "r19_verdict_full": _s(row, "r19_verdict_full") or None,
                 "r19_verdict_freezing_screen": _s(row, "verdict_freezing_screen") or None,
                 "r19_verdict_stop_rule": _s(row, "verdict_stop_rule") or None,
                 "reported_verdict": _s(row, "reported_verdict") or None, "r19_item4": _s(row, "r19_item4") or None,
                 "sensitivity_set": _s(row, "sensitivity_set") or None,
                 "sensitivities_not_run": _s(row, "sensitivities_not_run") or None,
                 "tost_verdict_eps0.05": _s(row, "tost_verdict_eps0.05") or None,
                 "batching_label": _s(row, "batching_label") or None,
                 "label": (_s(row, "label") or (LABEL_DISCOVERY if status == STATUS_DISCOVERY else None)),
                 "source_files": [p for p, ok in ((src_con, row is not None), (src_dec, dec is not None),
                                                  (L.rel("confirmation"), conf is not None)) if ok]}
        if conf is not None:
            for c in conf.get("claims") or []:
                if str(c.get("contrast_key") or c.get("contrast")) == key:
                    claim["confirmation"] = {k: c.get(k) for k in ("confirmed", "r19_verdict", "point", "percentile_low",
                                                                    "percentile_high", "seeds_positive", "n_seeds")}
        claims.append(claim)
    return claims


def _ladder_key(con: pd.DataFrame | None, step: str, design: str) -> str | None:
    if con is None or "key" not in con.columns:
        return None
    ks = [k for k in con["key"].astype(str).unique() if k.startswith(f"{step} vs ") and k.endswith(f"@{design}#ladder")]
    return ks[0] if ks else None


def _split_contrast(contrast: str) -> tuple[str, str]:
    for sep in (" vs ", " - "):
        if sep in contrast:
            a, b = contrast.split(sep, 1)
            return a.strip(), b.strip()
    return contrast, ""


def _cheapest_alternative(design: str) -> str:
    return {"V5": "B3i (radius interpolation inside the system; difficulty.json -> V5.lookup_comparator)",
            "V5-PAIR": "B3i-derived logSF (and the FLAT floor)", "V1": "B3 (= B4)", "V2": "B3i (focus-7 summary)",
            "V6": "B3i and the FLAT floor"}.get(design, "the design comparator of section 5")


def _num(s: str) -> float | None:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# --------------------------------------------------------------------------------------------- #
# report sections
# --------------------------------------------------------------------------------------------- #

def _regime_header() -> list[str]:
    return ["| contrast | design | half | seeds | averaging unit | parameter status | Delta (comparator - candidate) | "
            "margin | percentile 95 % | BCa 95 % | p | p_BH | p_BH (full family, m = 60) | R19 (full) | reported | "
            "sensitivity set | TOST eps 0.05 | status |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]


def _regime_row(L: Ledger, key: str, status: str, note: str = "") -> str:
    name = "discovery_contrasts"
    row = contrast_row(L, name, key)
    design = _s(row, "design") or key.split("@")[-1].split("#")[0]
    if row is None:
        return (f"| {key} | {design} | - | - | {AVERAGING_UNIT.get(design, design)} | - | "
                f"{L.missing(L.rel(name), 'key=' + key)} | | | | | | | | | | | {status} |")
    cand = _s(row, "candidate")
    sens = _s(row, "sensitivity_set")
    return (f"| {_s(row, 'contrast')}{(' (' + note + ')') if note else ''} | {design} | {_s(row, 'half')} | "
            f"{_s(row, 'seeds_evaluated')} | {AVERAGING_UNIT.get(design, design)} | {parameter_status(cand)} | "
            f"{contrast_cell(L, name, row, 'point')} | {contrast_cell(L, name, row, 'margin')} | "
            f"[{contrast_cell(L, name, row, 'percentile_low')}, {contrast_cell(L, name, row, 'percentile_high')}] | "
            f"[{contrast_cell(L, name, row, 'bca_low')}, {contrast_cell(L, name, row, 'bca_high')}] | "
            f"{contrast_cell(L, name, row, 'p_two_sided', 4)} | {contrast_cell(L, name, row, 'p_bh', 4)} | "
            f"{contrast_cell(L, name, row, 'p_bh_full_family', 4)} | {_s(row, 'r19_verdict_full')} "
            f"(item 4: {_s(row, 'r19_item4')}) | {_s(row, 'reported_verdict')}"
            f"{(' [' + _s(row, 'batching_label') + ']') if _s(row, 'batching_label') else ''} | "
            f"{sens}{(' -- not run: ' + _s(row, 'sensitivities_not_run')) if sens == ADDENDUM_LABEL and _s(row, 'sensitivities_not_run') else ''} | "
            f"{_s(row, 'tost_verdict_eps0.05')} | {status} |")


def _mae_table(L: Ledger, arms: Sequence[tuple[str, str]], design: str, variant: str = "primary") -> list[str]:
    """Macro MAE (log D, selection half) of the named arms: closed-form arms from the pre-seal table, learned arms from
    the discovery table (seed 104729); the constant baseline and the cheapest sensible alternative are always listed."""
    lines = [f"| arm | role | design | half | seeds | averaging unit | parameter status | macro MAE (log D) | source |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm, role in arms:
        if arm in CLOSED_FORM_ARMS:
            txt, key = summary_value(L, "preseal_summary", arm=arm, design=design, metric="mae",
                                     variant=primary_variant("preseal_summary", design) if variant == "primary" else variant)
            seeds, src = "deterministic", L.rel("preseal_summary")
        else:
            txt, key = summary_value(L, "discovery_summary", arm=arm, design=design, metric="mae", variant=variant,
                                     seed="104729")
            seeds, src = "104729 (addendum 1 item 3)", L.rel("discovery_summary")
        lines.append(f"| {arm} | {role} | {design} | selection | {seeds} | {AVERAGING_UNIT.get(design, design)} | "
                     f"{parameter_status(arm)} | {txt} | `{src}` |")
    return lines


def section_q1(L: Ledger, ctx: Mapping[str, Any]) -> list[str]:
    dec, con = ctx["decisions"], ctx["contrasts"]
    key = "M2 vs B3i@V5"
    row = contrast_row(L, "discovery_contrasts", key)
    status = claim_status(key, con, ctx["confirmation"])
    verdict = _s(row, "reported_verdict")
    if status == STATUS_CONFIRMATION and _confirmed(ctx["confirmation"], key) is True:
        head = f"{HEAD_CAN} reconstruct hidden metal x extractant cells better than the within-system lookup: the H1 claim (M2 vs B3i, V5) is confirmed on the withheld seeds."
    elif status == STATUS_CONFIRMATION:
        head = f"{HEAD_CANNOT} be said to hold transferable structure: the H1 claim was not confirmed on the withheld seeds."
    elif row is None:
        head = f"{HEAD_UNSHOWN} hold transferable metal-extractant structure: the H1 contrast has not been scored ({L.missing(L.rel('discovery_contrasts'))})."
    elif verdict == "FAIL":
        head = f"{HEAD_CANNOT} be shown to hold transferable structure beyond the within-system lookup: M2 vs B3i on V5 fails R19 in discovery (selection half; a clean null, not hidden)."
    else:
        head = (f"{HEAD_UNSHOWN} hold transferable structure beyond the within-system lookup: M2 vs B3i on V5 is "
                f"**{verdict or 'UNDECIDED'}** in discovery (selection half, seed 104729, optimistically biased; R19 item 4 "
                "NOT_EVALUATED under addendum 1); only the confirmation run can establish it.")
    lines = ["## Q1. Is there transferable metal-extractant structure in the expanded corpus?", "", f"**{head}**", "",
             "The primary hypothesis H1 (pre-registration section 1, section 9 S1(a)): the factorised interaction model M2 "
             "beats the V5 lookup comparator B3i (the cheapest sensible alternative, gen13 Addendum 3) by the margin delta5 "
             "on hidden metal x extractant cells; the constant baseline B0 is S1(b). Delta = macro MAE(comparator) - "
             "macro MAE(candidate), positive favours the candidate.", ""]
    lines += _regime_header()
    for k, note in ((key, "H1 primary"), ("M2 vs B0@V5", "S1(b) constant baseline"), ("B6 vs B3i@V5", "H1b")):
        lines.append(_regime_row(L, k, claim_status(k, con, ctx["confirmation"]), note))
    lines += ["", "Baseline difficulty the margin rests on (pre-seal, selection half, `evaluation/preseal/difficulty.json`):", ""]
    diff = ctx["difficulty"]
    lines += [f"- L5 (lookup comparator B3i macro MAE) = {_json_value(L, 'difficulty', diff, 'V5.L5', 4)}; "
              f"C5 (B0) = {_json_value(L, 'difficulty', diff, 'V5.C5', 4)}; noise floor N0 = "
              f"{_json_value(L, 'difficulty', diff, 'V5.N0', 4)}; delta5 = max(0.05, 0.20 x (L5 - N0)) = "
              f"{_json_value(L, 'difficulty', diff, 'V5.delta5', 4)} "
              f"(rho5 = {L.constant(ET.RHO5, 'section 9: rho5 = 0.20', 2)}; the floor {L.constant(ET.MARGIN_FLOOR, 'section 9: max(0.05, ...)', 2)}).",
              f"- Cells scored in the selection half: {_json_value(L, 'difficulty', diff, 'V5.n_cells_selection', 0)} in "
              f"{_json_value(L, 'difficulty', diff, 'V5.n_systems_selection', 0)} systems (exact leave-one-cell-out, pre-seal).",
              "", "Macro MAE of the arms on V5-primary (selection half; learned arms seed 104729 only, addendum 1 item 3):", ""]
    lines += _mae_table(L, (("B0", "constant baseline (S1(b))"), ("B3i", "cheapest sensible alternative (lookup comparator)"),
                            ("B3x", "lookup, nearest radius"), ("B6", "linear factorisation (H1b)"),
                            ("B6r0", "additive factorisation (S1(b))"), ("B5", "M0 descriptor CatBoost"),
                            ("FLAT_CAT", "flat categorical"), ("M1", "embeddings, no bilinear term"),
                            ("M2", "H1 candidate")), "V5")
    stop = ctx["stop_rule"]
    lines += ["", f"Stop rule (section 7 item 4; `{L.rel('stop_rule')}` -> `stop`): **{_json_str(L, 'stop_rule', stop, 'stop')}** "
              f"(M2 vs B3i stop-rule scope: {_json_str(L, 'stop_rule', stop, 'M2_vs_B3i.verdict')}; B6 vs B3i: "
              f"{_json_str(L, 'stop_rule', stop, 'B6_vs_B3i.verdict')}).", ""]
    if dec is not None:
        pc = (dec.get("power_check") or {}).get("required_before_reporting_a_null") or []
        if pc:
            lines += [f"Power check required before a null is reported (section 8): {', '.join(c['contrast'] for c in pc)} "
                      f"(`{L.rel('decisions')}` -> `power_check.required_before_reporting_a_null`); results: "
                      + _power_summary(L, ctx), ""]
    return lines


def _confirmed(conf: Mapping[str, Any] | None, key: str) -> bool | None:
    if conf is None:
        return None
    for c in conf.get("claims") or []:
        if str(c.get("contrast_key") or c.get("contrast")) == key:
            return c.get("confirmed")
    return None


def _power_summary(L: Ledger, ctx: Mapping[str, Any]) -> str:
    pw = L.json("power_checks")
    if pw is None:
        return L.missing(L.rel("power_checks"))
    parts = []
    for i, c in enumerate(pw.get("checks") or []):
        km = c.get("kappa_min")
        parts.append(f"{c.get('contrast')}: kappa_min = "
                     f"{L.value(km, L.rel('power_checks'), f'checks[{i}].kappa_min', 2) if km is not None else 'none in the grid'}"
                     f" ({c.get('verdict')})")
    return "; ".join(parts) if parts else "no contrast checked"


def section_q2(L: Ledger, ctx: Mapping[str, Any]) -> list[str]:
    dec, con = ctx["decisions"], ctx["contrasts"]
    s1 = (dec or {}).get("S1_components") or {}
    a, b0, b6 = s1.get("S1a_M2_vs_B3i"), s1.get("S1b_M2_vs_B0"), s1.get("S1b_M2_vs_B6r0")
    rep = [x.get("reported_verdict") for x in (a, b0, b6) if x]
    forced = bool(s1.get("S1_forced_undecided"))
    dep = ctx["deployed"]
    if dec is None:
        head = f"{HEAD_UNSHOWN} reconstruct a missing metal x extractant cell: the scorer has not run ({L.missing(L.rel('decisions'))})."
    elif forced:
        head = f"{HEAD_UNSHOWN} reconstruct a missing cell: S1 is reported UNDECIDED because the re-coloured batched-vs-exact check failed (section 7 item 6; label 'batched (check failed)')."
    elif rep and all(v == "PASS" for v in rep):
        head = f"{HEAD_CAN} reconstruct a missing metal x extractant cell in discovery (S1(a) and S1(b) pass on the selection half), pending confirmation on the withheld seeds."
    elif rep and any(v == "FAIL" for v in rep):
        head = f"{HEAD_CANNOT} reconstruct a missing metal x extractant cell better than the registered comparators by the registered margins: an S1(a)/S1(b) component fails in discovery."
    else:
        head = f"{HEAD_UNSHOWN} reconstruct a missing metal x extractant cell: S1(a)/S1(b) are {', '.join(rep) or 'not evaluated'} in discovery (learned-arm contrasts are at best UNDECIDED under addendum 1 until confirmation)."
    lines = ["## Q2. Can a missing metal x extractant combination be reconstructed?", "", f"**{head}**", "",
             f"Deployed predictor (h3.deployed_configuration on `{L.rel('decisions')}`): **{dep.get('arm') or 'undecided'}** -- "
             f"{dep.get('basis')}.", "",
             "S1 components on V5-primary (section 9; discovery values are the selection half, seed 104729, optimistically "
             "biased; the confirmation half with the withheld seeds decides):", "",
             "| component | contrast | Delta | margin | R19 full | reported | scopes | status |", "|---|---|---|---|---|---|---|---|"]
    for label, comp, key in (("S1(a)", a, "S1a_M2_vs_B3i"), ("S1(b) constant", b0, "S1b_M2_vs_B0"),
                             ("S1(b) additive", b6, "S1b_M2_vs_B6r0")):
        if comp is None:
            lines.append(f"| {label} | - | {L.missing(L.rel('decisions'), 'S1_components.' + key)} | | | | | {STATUS_NOT_RUN} |")
            continue
        ck = f"S1_components.{key}"
        scopes = ", ".join(f"{k}={v}" for k, v in (comp.get("scopes") or {}).items())
        lines.append(f"| {label} | {key.split('_', 1)[1].replace('_', ' ')} | {_json_value(L, 'decisions', dec, ck + '.point')} | "
                     f"{_json_value(L, 'decisions', dec, ck + '.margin')} | {comp.get('r19_full')} | {comp.get('reported_verdict')} | "
                     f"{scopes} | {STATUS_DISCOVERY} |")
    lines += ["", "S1(c)-(e) are reported under Q3 (direction), Q6 (support distance) and Q7 (calibration).", "",
              f"Strata of the deployed predictor ({dep.get('arm') or 'undecided'})'s V5 error (macro MAE per stratum, "
              "`tables/discovery_summary.csv`):", ""]
    # the summary table writes the deployed arm's rows under the name its RECORDS use (M0 -> B5,
    # discovery.ARM_ALIASES), so the lookup takes arm_alias and the sentence keeps arm (task X finding V-P02)
    arm = dep.get("arm_alias") or dep.get("arm")
    if arm and arm not in CLOSED_FORM_ARMS:
        ds = L.csv("discovery_summary")
        if ds is not None:
            sub = ds[(ds["arm"] == arm) & (ds["design"] == "V5") & (ds["variant"] == "primary") & (ds["metric"] == "mae")
                     & (ds["aggregation"] == "unit_macro") & (ds["stratum"] != "all") & (ds["scoring_filter"] == "none")]
            if sub.empty:
                lines.append(f"- {L.missing(L.rel('discovery_summary'), f'arm={arm} strata rows')}")
            for _, r in sub.iterrows():
                lines.append(f"- {r['stratum']}: {L.value(r['value'], L.rel('discovery_summary'), _row_key(r, SUMMARY_FILTER_COLS, 'value'))} "
                             f"(n_units {L.value(r['n_units'], L.rel('discovery_summary'), _row_key(r, SUMMARY_FILTER_COLS, 'n_units'), 0)})")
        else:
            lines.append(f"- {L.missing(L.rel('discovery_summary'))}")
    else:
        lines.append(f"- the deployed predictor is {arm or 'undecided'}: no learned-arm strata to report.")
    lines += ["", "Ladder H4 (architecture question, section 6) is reported under Q5.", ""]
    return lines


def section_q3(L: Ledger, ctx: Mapping[str, Any]) -> list[str]:
    dec, conf = ctx["decisions"], ctx["confirmation"]
    diff = ctx["difficulty"]
    s2 = (conf or {}).get("S2") or {}
    if conf is None:
        head = (f"{HEAD_UNSHOWN} reconstruct Pr/Nd selectivity with the pair hidden: V6 is run once, at confirmation, and the "
                f"confirmation decision file does not exist ({L.missing(L.rel('confirmation'))}).")
    else:
        passes = [s2.get(k, {}).get("pass") if isinstance(s2.get(k), Mapping) else None for k in ("a", "b", "c")]
        overall = s2.get("passed")
        if ((conf.get("V6") or {}).get("run")) is False:
            head = f"{HEAD_UNSHOWN} reconstruct Pr/Nd selectivity: the confirmation decision file says V6 has not run."
        elif all(p is True for p in passes) or (overall is True and all(p is None for p in passes)):
            head = f"{HEAD_CAN} reconstruct Pr/Nd selectivity in the 13 V6 systems: S2 passes in the single confirmation run."
        elif any(p is False for p in passes) or overall is False:
            head = f"{HEAD_CANNOT} reconstruct Pr/Nd selectivity to the registered standard: S2 fails in the confirmation run."
        else:
            head = f"{HEAD_UNSHOWN} reconstruct Pr/Nd selectivity: S2 is undecided in the confirmation decision file."
    lines = ["## Q3. Can Pr/Nd selectivity be reconstructed when direct pair data are hidden?", "", f"**{head}**", "",
             "Two registered tests: S1(c) on V5-PAIR (selectivity direction and logSF magnitude on hidden cell pairs; the "
             "selection-half counterweight is discovery, the confirmation half decides) and S2 on V6 (the Pr/Nd double-cell "
             "hold-out, run ONCE at confirmation; section 3.4). The FLAT floor (predict logSF = 0) and the lookup-derived "
             "logSF are the two comparators.", "",
             "**S1(c), V5-PAIR** (section 9 as redefined 2026-09-15):", ""]
    s1c = ((dec or {}).get("S1_components") or {}).get("S1c_selection")
    if s1c is None:
        lines.append(f"- {L.missing(L.rel('decisions'), 'S1_components.S1c_selection')} (the M2 V5-PAIR run or the scorer's pair block has not run)")
    else:
        base = "S1_components.S1c_selection"
        lines += [f"- selection-half counterweight (seed 104729): min_Y Delta_Y = {_json_value(L, 'decisions', dec, base + '.min_delta')} "
                  f"(yardstick {_json_str(L, 'decisions', dec, base + '.min_delta_yardstick')}; registered floor "
                  f"{L.constant(ET.S1C_SELECTION_MIN_DELTA, 'section 9 S1(c): min_Y Delta_Y >= -0.02', 2)}); verdict with the confirmation "
                  f"half missing: {_json_str(L, 'decisions', dec, base + '.verdict_with_confirmation_missing')}; reported: "
                  f"{_json_str(L, 'decisions', dec, base + '.reported_verdict')}; pairs {_json_value(L, 'decisions', dec, base + '.n_pairs', 0)}"]
        for y, blk in sorted((s1c.get("direction") or {}).items()):
            if isinstance(blk, Mapping):
                lines.append(f"  - direction Delta_{y} (M2 - {y}) = {_json_value(L, 'decisions', dec, base + f'.direction.{y}.delta')} "
                             f"on {_json_value(L, 'decisions', dec, base + f'.direction.{y}.n_pairs', 0)} pairs, "
                             f"interval excludes 0: {_json_str(L, 'decisions', dec, base + f'.direction.{y}.interval_excludes_zero')}")
        for y, blk in sorted((s1c.get("logsf_mae") or {}).items()):
            if isinstance(blk, Mapping):
                lines.append(f"  - logSF MAE gain over {y} ({y} - M2) = {_json_value(L, 'decisions', dec, base + f'.logsf_mae.{y}.gain')} "
                             f"(eta5 = {L.constant(ET.ETA5, 'section 9 S1(c): eta5 = 0.02', 2)})")
    lines += ["", "V5-PAIR difficulty (pre-seal, selection half): FLAT floor SF5_FLAT = "
              f"{_json_value(L, 'difficulty', diff, 'V5_PAIR.SF5_FLAT', 4)} log units; lookup-derived (B3i) logSF MAE = "
              f"{_json_value(L, 'difficulty', diff, 'V5_PAIR.chosen_lookup_derived_logsf_mae', 4)}; DIR5 (descriptive, enters no rule) = "
              f"{_json_value(L, 'difficulty', diff, 'V5_PAIR.DIR5', 4)}.", ""]
    ps = L.csv("discovery_pair_summary")
    if ps is not None:
        lines += ["M2 on the seed-104729 batched V5-PAIR folds (selection half; `tables/discovery_pair_summary.csv`, unit macro over cell pairs):", ""]
        for metric in ("logsf_mae", "direction_accuracy_abs_ge_0.3"):
            row = _pick(ps, {"arm": "M2", "metric": metric, "aggregation": "unit_macro", "stratum": "all"})
            if row is None:
                lines.append(f"- {metric}: {L.missing(L.rel('discovery_pair_summary'), f'arm=M2 metric={metric}')}")
            else:
                m = re.search(r"n_pairs=(\d+)", str(row.get("note", "")))
                n_txt = (f"n_pairs = {L.text(m.group(1), L.rel('discovery_pair_summary'), _row_key(row, SUMMARY_FILTER_COLS, 'note'))}"
                         if m else "")
                lines.append(f"- {metric}: {L.value(row['value'], L.rel('discovery_pair_summary'), _row_key(row, SUMMARY_FILTER_COLS, 'value'))} "
                             f"({n_txt}; unit macro over cell pairs) -- FLAT direction accuracy is 1/2 by definition")
    else:
        lines.append(f"- M2 V5-PAIR metrics: {L.missing(L.rel('discovery_pair_summary'))}")
    lines += ["", "**S2, V6** (confirmation only; sections 9 S2(a)-(d), 3.4; the confirmation decision file "
              f"`{L.rel('confirmation')}`, schema gen19.confirmation.v1: `S1.passed`, `S1.deployed_predictor`, `S2.passed`, `V6.run`, "
              "`seeds.verified`; the per-component blocks `S2.a`-`S2.d` and the `claims` list are read when the confirmation "
              "orchestrator writes them):", ""]
    if conf is None:
        lines += [f"- {L.missing(L.rel('confirmation'))}", f"- {L.missing(L.rel('v6_systems'))}", f"- {L.missing(L.rel('v6_pairs'))}"]
    else:
        lines += [f"- S1 passed at confirmation (deployed predictor {_json_str(L, 'confirmation', conf, 'S1.deployed_predictor')}): "
                  f"**{_json_str(L, 'confirmation', conf, 'S1.passed')}**; S2 passed: **{_json_str(L, 'confirmation', conf, 'S2.passed')}**; "
                  f"V6 run: {_json_str(L, 'confirmation', conf, 'V6.run')}; withheld seeds verified: "
                  f"{_json_str(L, 'confirmation', conf, 'seeds.verified')}"]
        for k, text in (("a", "sign of logSF in >= 11 of 13 systems and paired direction +0.05 over every yardstick"),
                        ("b", "logSF MAE <= FLAT - 0.02 and <= lookup-derived; hidden Pr/Nd log D MAE <= L5"),
                        ("c", "pooled logSF interval coverage: 80 % in [0.70, 0.90], 95 % >= 0.88"),
                        ("d", "top recipe identity kept in >= 0.80 of the D-draw re-rankings (only if Phase H ran)")):
            blk = s2.get(k)
            if blk is None:
                lines.append(f"- S2({k}) {text}: {L.missing(L.rel('confirmation'), f'S2.{k}')}")
            else:
                lines.append(f"- S2({k}) {text}: **{_json_str(L, 'confirmation', conf, f'S2.{k}.pass')}** -- "
                             + "; ".join(f"{kk} = {_json_value(L, 'confirmation', conf, f'S2.{k}.{kk}') if isinstance(vv, (int, float)) and not isinstance(vv, bool) else _json_str(L, 'confirmation', conf, f'S2.{k}.{kk}')}"
                                         for kk, vv in blk.items() if kk != "pass"))
        v6 = L.csv("v6_systems")
        if v6 is not None:
            n_ok = int((v6["sign_agrees"].astype(str).str.lower() == "true").sum()) if "sign_agrees" in v6.columns else None
            lines += ["", f"Per-system sign agreement (`{L.rel('v6_systems')}`): "
                      + (f"{L.count(n_ok, L.rel('v6_systems'), 'where:sign_agrees=True')} of "
                         f"{L.count(len(v6), L.rel('v6_systems'), 'rows')} systems" if n_ok is not None else "column sign_agrees absent")]
    lines.append("")
    return lines


def section_q4(L: Ledger, ctx: Mapping[str, Any]) -> list[str]:
    ver, f4, summ = L.json("h3_verdicts"), L.json("h3_f4"), L.json("h3_summary")
    sdep = (summ or {}).get("deployed") or {}
    dep_arm = sdep.get("arm") or ctx["deployed"].get("arm")
    # h3_verdicts.json is keyed by the arm the H3 RECORDS use (M0 -> B5, discovery.ARM_ALIASES); the printed sentences
    # keep dep_arm (task X finding V-P02)
    dep_key = sdep.get("arm_alias") or ctx["deployed"].get("arm_alias") or dep_arm
    dv = (ver or {}).get(dep_key or "", {}).get("verdict") if ver else None
    if ver is None:
        head = f"{HEAD_UNSHOWN} gain from actinide data: the section 11 ablation has not run ({L.missing(L.rel('h3_verdicts'))})."
    elif dv == "helps":
        head = f"{HEAD_CAN} use actinide rows: WITH beats WITHOUT and ACT_PERMUTED on hidden Ln(III) cells and is non-inferior on V1 / V2 (H3 'helps', discovery scope)."
    elif dv == "hurts":
        head = f"{HEAD_CANNOT} use actinide rows without loss: WITHOUT beats WITH (H3 'hurts'; F4 is checked below)."
    elif dv == "equivalent":
        head = f"{HEAD_CANNOT} be shown to gain or lose from actinide rows: the WITH - WITHOUT difference lies inside +-0.05 log D (TOST 'equivalent')."
    else:
        head = f"{HEAD_UNSHOWN} gain from actinide data: H3 is UNDECIDED for the deployed arm (kappa_min decides whether the null is informative)."
    lines = ["## Q4. Does actinide data improve lanthanide prediction?", "", f"**{head}**", "",
             "Section 11 / brief section 30: same architecture, same folds, same seeds, same Ln(III) test set; WITH vs "
             "WITHOUT (every actinide training row removed) and vs ACT_PERMUTED (actinide log D permuted within system x "
             "publication group). Verdicts are read on the freezing-screen scope in discovery (R19 item 4 NOT_EVALUATED, "
             "addendum 1 item 3).", ""]
    if ver is None:
        lines += [f"- {L.missing(L.rel('h3_verdicts'))}", f"- {L.missing(L.rel('h3_f4'))}", f"- {L.missing(L.rel('h3_contrasts'))}", ""]
        return lines
    lines += ["| model arm | verdict | WITH beats WITHOUT (V5) | WITH beats PERMUTED (V5) | V1/V2 non-inferior | TOST V5 | kappa_min | status |",
              "|---|---|---|---|---|---|---|---|"]
    for arm, v in sorted(ver.items()):
        ni = v.get("v1_v2_non_inferior") or {}
        km = v.get("kappa_min")
        mark = f" (deployed = {dep_arm})" if arm == dep_key else ""
        lines.append(f"| {arm}{mark} | **{v.get('verdict')}** | {v.get('v5_with_beats_without')} | "
                     f"{v.get('v5_with_beats_permuted')} | {json.dumps(ni)} | {(v.get('tost_v5_with_minus_without') or {}).get('verdict')} | "
                     f"{L.value(km, L.rel('h3_verdicts'), f'{arm}.kappa_min', 2) if km is not None else v.get('kappa_status')} | {STATUS_DISCOVERY} |")
    hc = L.csv("h3_contrasts")
    if hc is not None and "primary_cluster_unit" in hc.columns:
        prim = hc[hc["primary_cluster_unit"].astype(str).str.lower() == "true"]
        lines += ["", "| model arm | contrast | design | Delta MAE | pct 95 % | p | scope verdict | full R19 |", "|---|---|---|---|---|---|---|---|"]
        for _, r in prim.iterrows():
            kc = {c: r[c] for c in ("model_arm", "contrast", "design", "cluster_unit") if c in r.index}
            lines.append(f"| {r.get('model_arm', '')} | {r['contrast']} | {r['design']} | "
                         f"{L.value(r['point'], L.rel('h3_contrasts'), where_key(kc, 'point'))} | "
                         f"[{L.value(r['percentile_low'], L.rel('h3_contrasts'), where_key(kc, 'percentile_low'))}, "
                         f"{L.value(r['percentile_high'], L.rel('h3_contrasts'), where_key(kc, 'percentile_high'))}] | "
                         f"{L.value(r['p_two_sided'], L.rel('h3_contrasts'), where_key(kc, 'p_two_sided'), 4)} | "
                         f"{r.get('verdict_freezing_screen', '')} | {r.get('r19_verdict_full', '')} |")
    else:
        lines.append(f"- contrast table: {L.missing(L.rel('h3_contrasts'))}")
    lines += ["", f"**F4 (negative actinide transfer):** {'HOLDS' if (f4 or {}).get('failure') else ('does not hold' if (f4 or {}).get('status') == 'computed' else L.missing(L.rel('h3_f4')))}"
              + (f" (deployed arm {f4.get('deployed_arm')}; designs computed {f4.get('designs_computed')})" if f4 else ""), ""]
    return lines


def section_q5(L: Ledger, ctx: Mapping[str, Any]) -> list[str]:
    dec, con = ctx["decisions"], ctx["contrasts"]
    lad = (dec or {}).get("ladder") or {}
    ladder = L.json("ladder")
    kept = {s: (lad.get(s) or {}).get("kept") for s in ("M1", "M2")}
    if ladder is not None:
        for s, rec in (ladder.get("steps") or {}).items():
            kept[s] = rec.get("kept")
    kept_steps = [s for s, k in kept.items() if k is True]
    if dec is None:
        head = f"{HEAD_UNSHOWN} benefit from any architecture component: the ladder has not been scored ({L.missing(L.rel('decisions'))})."
    elif kept_steps:
        head = f"{HEAD_CAN} be improved by these components on strict folds (selection half, optimistically biased): {', '.join(kept_steps)}; every other step adds no strict-fold benefit or was not run."
    elif any(k is None for k in kept.values()):
        head = f"{HEAD_UNSHOWN} benefit from any architecture component: ladder decisions are pending."
    else:
        head = f"{HEAD_CANNOT} be improved by any tested component over the descriptor base M0 on strict folds: no ladder step is kept (F6 is checked below)."
    lines = ["## Q5. Which architecture components actually help?", "", f"**{head}**", "",
             "Ladder rule (section 6): a step is kept only if it passes R19 against its retained predecessor on the V5-primary "
             "selection half and is non-inferior (TOST, eps 0.05) on V1 and V2. Keep/remove decisions use outer selection-half "
             "scores and are optimistically biased by construction; they are never confirmed claims.", "",
             "| step | adds | predecessor | kept | status | source |", "|---|---|---|---|---|---|"]
    adds = {"M0": "B5 descriptor CatBoost (base)", "M1": "metal / ligand embeddings + condition encoder",
            "M2": "bilinear interaction e_m^T W e_l (H1 candidate)", "M3": "mechanism experts", "M4": "publication-group hierarchy",
            "M5": "pairwise loss", "M6": "physics penalties", "M7": "5-member heteroscedastic ensemble"}
    for s in ("M0", "M1", "M2"):
        rec = lad.get(s) or {}
        lines.append(f"| {s} | {adds[s]} | {rec.get('predecessor', '-') if s != 'M0' else '-'} | {rec.get('kept') if rec else L.missing(L.rel('decisions'), f'ladder.{s}')} | "
                     f"{STATUS_DISCOVERY if rec else STATUS_NOT_RUN} | `{L.rel('decisions')}` -> `ladder.{s}.kept` |")
    for s in ("M3", "M4", "M5", "M6", "M7"):
        if ladder is None:
            lines.append(f"| {s} | {adds[s]} | - | {L.missing(L.rel('ladder'))} | {STATUS_NOT_RUN} | `{L.rel('ladder')}` |")
        else:
            rec = (ladder.get("steps") or {}).get(s)
            if rec is None:
                lines.append(f"| {s} | {adds[s]} | - | not run | {STATUS_NOT_RUN} ({'stop rule' if ladder.get('stop_rule') else 'not reached'}) | `{L.rel('ladder')}` -> `steps.{s}` |")
            else:
                lines.append(f"| {s} | {adds[s]} | {rec.get('predecessor', '-')} | {rec.get('kept')} | {rec.get('status')} "
                             f"({STATUS_DISCOVERY}) | `{L.rel('ladder')}` -> `steps.{s}.kept` |")
    lines += ["", "**H4 -- flat vs descriptor vs factorised vs factorised + mechanism** (section 6 / brief section 31; V5-primary, identical batches, seeds and cells):", ""]
    lines += _regime_header()
    for k in ("M2 vs FLAT_CAT@V5", "M2 vs M0@V5", "M3 vs M2@V5", "B6 vs B6r0@V5"):
        lines.append(_regime_row(L, k, claim_status(k, con, ctx["confirmation"]), "H4"))
    lines += ["", "**Ladder contrasts** (each step vs its retained predecessor; V5 R19 ladder scope, V1 / V2 TOST):", ""]
    lines += _regime_header()
    if con is not None and "key" in con.columns:
        for k in sorted(k for k in con["key"].astype(str).unique() if k.endswith("#ladder")):
            lines.append(_regime_row(L, k, STATUS_DISCOVERY, "ladder"))
    else:
        lines.append(f"| ladder contrasts | - | - | - | - | - | {L.missing(L.rel('discovery_contrasts'))} | | | | | | | | | | | {STATUS_NOT_RUN} |")
    lines += ["", "**H5 -- chemistry priors on / off** (M3 vs M2, M4, M6a, M6b; V1, V2, V5):", ""]
    if ladder is None or not (ladder.get("steps") or {}).get("H5"):
        lines.append(f"- {L.missing(L.rel('ladder'), 'steps.H5')} (H5 runs after M7 in `scripts/g19_run_ladder.py`)")
    else:
        h5 = ladder["steps"]["H5"]
        lines.append(f"- H5 status: {h5.get('status')}; contrast rows: {_json_value(L, 'ladder', ladder, 'steps.H5.n_contrast_rows', 0) if h5.get('n_contrast_rows') is not None else 'none'} "
                     f"(`evaluation/ladder/decisions/contrasts_H5.csv`)")
    lines += ["", f"**F6 (complexity adds no strict-fold benefit):** {_f6_text(L, ctx, lad, ladder)}", ""]
    return lines


def _f6_text(L: Ledger, ctx: Mapping[str, Any], lad: Mapping[str, Any], ladder: Mapping[str, Any] | None) -> str:
    s1a = (((ctx["decisions"] or {}).get("S1_components") or {}).get("S1a_M2_vs_B3i") or {}).get("reported_verdict")
    if ctx["decisions"] is None:
        return L.missing(L.rel("decisions"))
    kept_any = any((lad.get(s) or {}).get("kept") is True for s in ("M1", "M2"))
    if ladder is not None:
        kept_any = kept_any or any(r.get("kept") is True for r in (ladder.get("steps") or {}).values() if isinstance(r, Mapping))
    pending = any((lad.get(s) or {}).get("kept") is None for s in ("M1", "M2"))
    if s1a == "PASS":
        return "does not hold in discovery: S1(a) passes (selection half; confirmation decides)."
    if pending or s1a is None:
        return "not decidable yet: a ladder decision or S1(a) is pending."
    if not kept_any:
        return ("HOLDS in discovery: S1(a) does not pass and no step M1-M6 is kept; the deployed predictor is the best-passing "
                f"baseline ({ctx['deployed'].get('arm')}) and the architecture claim failed (section 10 F6).")
    return "does not hold: at least one ladder step is kept on the selection half (optimistically biased)."


def section_q6(L: Ledger, ctx: Mapping[str, Any]) -> list[str]:
    s1e = L.csv("s1e")
    rel = L.csv("reliability")
    dep = ctx["deployed"].get("arm")
    row = None
    if s1e is not None and dep:
        row = _pick(s1e, {"arm": dep, "design": "V5", "x": "support_score"})
    comps_ok = None
    if rel is not None:
        sub = rel[rel["quantity"].astype(str) == "support_score_components"]
        if len(sub):
            comps_ok = bool((sub["gate"].astype(str).str.upper() == "PASS").all())
    if row is None:
        head = f"{HEAD_UNSHOWN} err more with chemical distance: the support-distance statistic has not been computed ({L.missing(L.rel('s1e'))})."
    else:
        rho = _num(row.get("rho"))
        lo = _num(row.get("percentile_low"))
        hi = _num(row.get("percentile_high"))
        ok = rho is not None and rho <= ET.S1E_MAX_SPEARMAN and hi is not None and hi < 0 and comps_ok
        if comps_ok is False:
            head = f"{HEAD_UNSHOWN} err sensibly with chemical distance: a support-score component fails the section 8 reliability floor, so the correlation is UNDECIDED (unreliable)."
        elif ok:
            head = f"{HEAD_CAN} be trusted to err more on cells farther from their training support: Spearman(cell MAE, support_score) <= -0.10 with the system-cluster interval excluding 0 (S1(e), discovery)."
        else:
            head = f"{HEAD_CANNOT} be shown to err more with chemical distance: S1(e) is not met (rho above -0.10 or the interval includes 0)."
    lines = ["## Q6. How does error grow with chemical distance?", "", f"**{head}**", "",
             "S1(e) (section 9): Spearman rho between the cell MAE of the deployed predictor and the cell's registered "
             f"support_score (section 13 v1, target-free) is <= {L.constant(ET.S1E_MAX_SPEARMAN, 'section 9 S1(e): rho <= -0.10', 2)} with the "
             "system-cluster bootstrap interval excluding 0, and the support-score components pass the reliability floor "
             f"{L.constant(ET.RELIABILITY_FLOOR, 'section 8: floor 0.3', 1)} (reliability before correlation).", ""]
    if s1e is None:
        lines.append(f"- {L.missing(L.rel('s1e'))} (written by `scripts/g19_make_figures.py` from the discovery records and `_support` files)")
    else:
        lines += ["| arm | design | x | Spearman rho | percentile 95 % (system cluster) | n cells | status |", "|---|---|---|---|---|---|---|"]
        for _, r in s1e.iterrows():
            kc = {c: r[c] for c in ("arm", "design", "x") if c in r.index}
            lines.append(f"| {r['arm']} | {r['design']} | {r['x']} | {L.value(r['rho'], L.rel('s1e'), where_key(kc, 'rho'))} | "
                         f"[{L.value(r['percentile_low'], L.rel('s1e'), where_key(kc, 'percentile_low'))}, "
                         f"{L.value(r['percentile_high'], L.rel('s1e'), where_key(kc, 'percentile_high'))}] | "
                         f"{L.value(r['n_cells'], L.rel('s1e'), where_key(kc, 'n_cells'), 0)} | {STATUS_DISCOVERY} |")
    lines += ["", "Reliability of the support-score components (section 8; `tables/reliability_before_correlation.csv`):", ""]
    if rel is None:
        lines.append(f"- {L.missing(L.rel('reliability'))}")
    else:
        sub = rel[rel["quantity"].astype(str) == "support_score_components"]
        if sub.empty:
            lines.append(f"- {L.missing(L.rel('reliability'), 'quantity=support_score_components')}")
        for _, r in sub.iterrows():
            kc = {"quantity": r["quantity"], "unit": r["unit"]}
            lines.append(f"- {r['unit']}: reliability {L.value(r['reliability'], L.rel('reliability'), where_key(kc, 'reliability'))} "
                         f"({r['method']}) -> gate {r['gate']}")
    diff = ctx["difficulty"]
    f08 = (diff or {}).get("F08_descriptive_spearman")
    if f08:
        lines += ["", f"Pre-seal descriptive reading (comparator {f08.get('comparator')}, `evaluation/preseal/difficulty.json` -> `F08_descriptive_spearman`; no reliability assessed, so not interpreted):"]
        for col, v in (f08.get("values") or {}).items():
            if isinstance(v, Mapping) and "rho" in v:
                lines.append(f"- {col}: rho {_json_value(L, 'difficulty', diff, f'F08_descriptive_spearman.values.{col}.rho', 2)} over "
                             f"{_json_value(L, 'difficulty', diff, f'F08_descriptive_spearman.values.{col}.n_cells', 0)} cells")
    lines += ["", "Figure F08 (`figures/F08_error_vs_support.png`) shows the cell errors against support_score per domain-status category.", ""]
    return lines


def _arm_tables(arm: str) -> tuple[str, str, str | None]:
    """``(summary table, coverage-by-domain-status table, seed filter)`` of an arm: closed-form arms live in the pre-seal
    tables (any seed row), learned arms in the discovery tables on seed 104729."""
    if arm in CLOSED_FORM_ARMS:
        return "preseal_summary", "preseal_coverage", None
    return "discovery_summary", "discovery_coverage", "104729"


def _coverage_by_category(L: Ledger, arm: str) -> tuple[pd.DataFrame | None, str]:
    """The arm's V5-primary ``coverage_80`` unit-macro rows per domain-status category, and the table name."""
    _, cov_name, _ = _arm_tables(arm)
    cov = L.csv(cov_name)
    if cov is None or "category" not in cov.columns:
        return None, cov_name
    sub = cov[(cov["arm"] == arm) & (cov["design"] == "V5") & (cov["metric"] == "coverage_80") & (cov["aggregation"] == "unit_macro")]
    if "variant" in sub.columns:
        sub = sub[sub["variant"] == "primary"]
    if "half" in sub.columns:
        sub = sub[sub["half"] == "selection"]
    return sub, cov_name


def _deployed_coverage(L: Ledger, arm: str) -> dict[float, float | None]:
    """Unit-macro 50 / 80 / 95 % coverage of an arm on V5-primary (selection half) as numbers, from its summary table."""
    sum_name, _, seed = _arm_tables(arm)
    ds = L.csv(sum_name)
    if ds is None:
        return {}
    out: dict[float, float | None] = {}
    for lvl in EC.LEVELS:
        pct = int(round(lvl * 100))
        filters: dict[str, Any] = {"arm": arm, "design": "V5", "variant": "primary", "half": "selection", "metric": f"coverage_{pct}",
                                   "aggregation": "unit_macro", "stratum": "all"}
        if "scoring_filter" in ds.columns:
            filters["scoring_filter"] = "none"
        if "status" in ds.columns:
            filters["status"] = "registered"
        if seed is not None:
            filters["seed"] = seed
        row = _pick(ds, filters, prefer={"unit_reading": "registered"})
        out[lvl] = _num(row["value"]) if row is not None else None
    return out


def _coverage_rows(L: Ledger, name: str, arm: str, design: str, seed: str | None) -> dict[float, str]:
    out = {}
    for lvl in EC.LEVELS:
        pct = int(round(lvl * 100))
        txt, _ = summary_value(L, name, arm=arm, design=design, metric=f"coverage_{pct}", seed=seed,
                               variant=primary_variant(name, design))
        out[lvl] = txt
    return out


def section_q7(L: Ledger, ctx: Mapping[str, Any]) -> list[str]:
    dep = ctx["deployed"].get("arm")
    sum_name, _, dep_seed = _arm_tables(dep or "M2")
    ds = L.csv(sum_name)
    m7 = L.json("m7_metrics")
    cov_vals = _deployed_coverage(L, dep) if dep else {}
    s1d = None
    if cov_vals and all(v is not None for v in cov_vals.values()):
        cats = {}
        sub, _ = _coverage_by_category(L, dep)
        if sub is not None:
            for _, r in sub.iterrows():
                cats[str(r["category"])] = (_num(r["value"]) or float("nan"), int(float(r["category_units"] or 0)))
        s1d = EC.s1d_check({lvl: float(v) for lvl, v in cov_vals.items()}, cats)
        f3 = EC.f3_check(float(cov_vals[0.80]), float(cov_vals[0.95]))
    else:
        f3 = None
    if dep is None or ds is None:
        head = f"{HEAD_UNSHOWN} carry calibrated uncertainty: the deployed predictor's interval metrics are not available ({L.missing(L.rel(sum_name))})."
    elif s1d is None:
        head = f"{HEAD_UNSHOWN} carry calibrated uncertainty: no interval rows for the deployed predictor {dep} on V5-primary."
    elif s1d["pass"]:
        head = f"{HEAD_CAN} carry calibrated split-conformal intervals on V5-primary: S1(d) bands met at 50 / 80 / 95 % (discovery, selection half)."
    else:
        head = f"{HEAD_CANNOT} be called calibrated on V5-primary: an S1(d) band is missed (discovery, selection half)."
    b50, b80, b95 = (EC.S1D_BANDS[l] for l in (0.50, 0.80, 0.95))
    c80, f80 = EC.S1D_CATEGORY_BAND_80, EC.F3_BAND_80
    band = lambda b, key: f"[{L.constant(b[0], key, 2)}, {L.constant(b[1], key, 2)}]"  # noqa: E731
    lines = ["## Q7. Is uncertainty calibrated?", "", f"**{head}**", "",
             f"S1(d) (section 9): macro coverage over V5-primary cells in {band(b50, 'section 9 S1(d): 50 % band')} at 50 %, "
             f"{band(b80, 'section 9 S1(d): 80 % band')} at 80 %, {band(b95, 'section 9 S1(d): 95 % band')} at 95 %, and in every "
             f"domain-status category with >= {L.constant(EC.S1D_CATEGORY_MIN_CELLS, 'section 9 S1(d): >= 20 scored cells', 0)} scored "
             f"cells 80 % coverage in {band(c80, 'section 9 S1(d): category band')}. F3 (section 10): 80 % coverage outside "
             f"{band(f80, 'section 10 F3: 80 % band')} or 95 % coverage < {L.constant(EC.F3_MIN_95, 'section 10 F3: 95 % floor', 2)} on "
             "V5-primary or V1. Intervals are cross-fitted split-conformal (section 12; the only uncertainty method built for arms "
             "before M7).", "",
             "| arm | design | half | seeds | averaging unit | coverage 50 | coverage 80 | coverage 95 | width 80 | status |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    arms = [a for a in ((dep,) if dep else ()) if a] + ["B3i", "B0"]
    for arm in dict.fromkeys(arms):
        for design in ("V5", "V1", "V2"):
            name = "preseal_summary" if arm in CLOSED_FORM_ARMS else "discovery_summary"
            seed = None if arm in CLOSED_FORM_ARMS else "104729"
            c = _coverage_rows(L, name, arm, design, seed)
            w, _ = summary_value(L, name, arm=arm, design=design, metric="width_80", seed=seed,
                                 variant=primary_variant(name, design))
            lines.append(f"| {arm} | {design} | selection | {seed or '104729 (conformal inner folds)'} | {AVERAGING_UNIT[design]} | "
                         f"{c[0.5]} | {c[0.8]} | {c[0.95]} | {w} | {STATUS_DISCOVERY if arm not in CLOSED_FORM_ARMS else 'pre-seal comparator'} |")
    lines += ["", f"S1(d) on V5-primary for {dep or 'the deployed predictor'}: **{'PASS' if s1d and s1d['pass'] else ('FAIL' if s1d else 'not computed')}**"
              + ("; items: " + ", ".join(f"{k} {'pass' if v['pass'] else 'fail'}" for k, v in s1d["items"].items()) if s1d else ""),
              f"F3 (V5-primary) for {dep or 'the deployed predictor'}: **{'HOLDS' if f3 and f3['failure'] else ('does not hold' if f3 else 'not computed')}**", ""]
    sub, cov_name = _coverage_by_category(L, dep) if dep else (None, "discovery_coverage")
    if sub is not None and len(sub):
        lines += [f"80 % coverage by domain-status category for {dep} on V5-primary (`{L.rel(cov_name)}`):", "",
                  "| category | cells | coverage 80 |", "|---|---|---|"]
        for _, r in sub.iterrows():
            kc = {c: r[c] for c in ("arm", "design", "variant", "half", "metric", "aggregation", "category", "seed") if c in r.index}
            lines.append(f"| {r['category']} | {L.value(r['category_units'], L.rel(cov_name), where_key(kc, 'category_units'), 0)} | "
                         f"{L.value(r['value'], L.rel(cov_name), where_key(kc, 'value'))} |")
    else:
        lines.append(f"- coverage by domain status: {L.missing(L.rel(cov_name), f'arm={dep}') if sub is not None else L.missing(L.rel(cov_name))}")
    lines += ["", "M7 ensemble (section 12; `evaluation/ladder/M7/metrics.json`):", ""]
    if m7 is None:
        lines.append(f"- {L.missing(L.rel('m7_metrics'))}")
    else:
        v = m7.get("verdict") or {}
        lines.append(f"- verdict kept = {_json_str(L, 'm7_metrics', m7, 'verdict.kept')}; calibration S1(d) pass = "
                     f"{_json_str(L, 'm7_metrics', m7, 'verdict.calibration_s1d_pass')}; MAE non-inferior on every design = "
                     f"{_json_str(L, 'm7_metrics', m7, 'verdict.mae_non_inferior_all_designs')}")
        for d, b in (m7.get("designs") or {}).items():
            im = b.get("interval_metrics") if isinstance(b, Mapping) else None
            if isinstance(im, Mapping):
                lines.append(f"  - {d}: coverage 50 / 80 / 95 = {_json_value(L, 'm7_metrics', m7, f'designs.{d}.interval_metrics.coverage_50')} / "
                             f"{_json_value(L, 'm7_metrics', m7, f'designs.{d}.interval_metrics.coverage_80')} / "
                             f"{_json_value(L, 'm7_metrics', m7, f'designs.{d}.interval_metrics.coverage_95')}; CRPS "
                             f"{_json_value(L, 'm7_metrics', m7, f'designs.{d}.interval_metrics.crps')}")
    lines.append("")
    return lines


def section_q8(L: Ledger, ctx: Mapping[str, Any]) -> list[str]:
    m7 = L.json("m7_metrics")
    dec = ctx["decisions"]
    kn = None
    if m7 is not None:
        kn = ((m7.get("designs") or {}).get("V5") or {}).get("knows_when_it_does_not_know")
    if kn is None:
        head = (f"{HEAD_UNSHOWN} know when it does not know: the test needs a predictive SD (Spearman(|error|, SD) with the "
                "system-cluster interval excluding 0, and coverage in UNSUPPORTED / FAMILY_EXTRAPOLATION within 0.10 of "
                "IN_DOMAIN), which only the M7 ensemble provides"
                + (f" ({L.missing(L.rel('m7_metrics'))})." if m7 is None else " (no V5 entry in the M7 metrics)."))
    elif kn.get("established"):
        head = f"{HEAD_CAN} identify when it does not know: both section 12 conditions hold for the M7 ensemble on V5-primary (discovery)."
    else:
        head = f"{HEAD_CANNOT} be shown to know when it does not know: a section 12 condition fails for M7 on V5-primary."
    lines = ["## Q8. Can the model identify when it does not know?", "", f"**{head}**", ""]
    if dec is not None:
        nr = ((dec.get("not_run") or {}).get("uncertainty_metrics") or {})
        if nr:
            lines += ["For every arm before M7 these metrics are NOT_RUN (`" + L.rel("decisions") + "` -> `not_run.uncertainty_metrics`):", ""]
            lines += [f"- {k}: {v}" for k, v in nr.items()]
            lines.append("")
    if kn is not None:
        lines += [f"- M7 on V5: established = {_json_str(L, 'm7_metrics', m7, 'designs.V5.knows_when_it_does_not_know.established')}; "
                  f"Spearman condition = {_json_str(L, 'm7_metrics', m7, 'designs.V5.knows_when_it_does_not_know.spearman_pass')}; "
                  f"coverage-gap condition = {_json_str(L, 'm7_metrics', m7, 'designs.V5.knows_when_it_does_not_know.coverage_gap_pass')}"]
        sp = ((m7.get("designs") or {}).get("V5") or {}).get("spearman_abs_error_sd")
        if isinstance(sp, Mapping):
            lines.append(f"- Spearman(|error|, SD) = {_json_value(L, 'm7_metrics', m7, 'designs.V5.spearman_abs_error_sd.point')} "
                         f"[{_json_value(L, 'm7_metrics', m7, 'designs.V5.spearman_abs_error_sd.low_95')}, "
                         f"{_json_value(L, 'm7_metrics', m7, 'designs.V5.spearman_abs_error_sd.high_95')}]")
    lines += ["", "Domain-status labels (section 13) are attached to every prediction regardless: the support table under Q7 "
              "shows whether coverage degrades in the extrapolation categories.", ""]
    return lines


def section_q9(L: Ledger, ctx: Mapping[str, Any]) -> list[str]:
    ps = L.json("process_summary")
    stab = L.json("process_stability")
    f5 = L.json("process_f5")
    s1_pass = _s1_pass(ctx)
    label = ((ps or {}).get("label") or {}).get("label") if isinstance((ps or {}).get("label"), Mapping) else (ps or {}).get("label")
    s2d = (ps or {}).get("s2d") or {}
    all_stable = bool(s2d) and all(bool(v.get("passes")) for v in s2d.values() if isinstance(v, Mapping))
    if ps is None:
        reason = ("section 14 gate: Phase H runs only if S1 passes for the deployed predictor"
                  + (f"; {s1_pass}" if s1_pass is not None else ""))
        head = f"{HEAD_UNSHOWN} keep the Gen18 process recommendations good after uncertainty propagation: the process step has not run ({L.missing(L.rel('process_summary'))}; {reason})."
    elif label and str(label).startswith("transfer-unsupported"):
        head = (f"{HEAD_CANNOT} support a headline process recommendation: every process result is labelled transfer-unsupported "
                "(S1, S2 or V6 not established, or an exploratory run; section 14).")
    elif all_stable:
        head = f"{HEAD_CAN} keep the top recipe under uncertainty propagation in every specification cell: its identity survives >= 0.80 of the D-draw re-rankings (S2(d))."
    else:
        head = f"{HEAD_CANNOT} keep a stable top recipe under uncertainty propagation in every specification cell: S2(d) stability below 0.80 in at least one cell, or not computed."
    lines = ["## Q9. Do Gen18 process recommendations remain good after uncertainty propagation?", "", f"**{head}**", "",
             "Section 14: 64 joint Monte-Carlo draws per operating point through the unmodified gen18 cascade; lexicographic "
             "ranking with the support gate first (UNSUPPORTED D makes a recipe ineligible); S2(d) stability over 20 bootstrap "
             "re-rankings (pass >= 0.80); F5 audit with the gate lifted. When S1 fails, gen18's B1 nearest-condition lookup "
             "stays the process chain's default D source (`scripts/g19_run_process.py`, `decisions/D06_process_integration.md`).", ""]
    if ps is None:
        lines += [f"- {L.missing(L.rel('process_summary'))}", f"- {L.missing(L.rel('process_stability'))}",
                  f"- {L.missing(L.rel('process_f5'))}", f"- {L.missing(L.rel('process_winners'))}", ""]
        return lines
    lab = ps.get("label") if isinstance(ps.get("label"), Mapping) else {"label": label}
    lines += [f"- label: **{lab.get('label')}** (headline allowed: {lab.get('headline_allowed')}; reasons: {lab.get('reasons')})",
              f"- gate: S1 passed = {_json_str(L, 'process_summary', ps, 'gate.confirmation.s1_passed')}; exploratory = "
              f"{_json_str(L, 'process_summary', ps, 'gate.exploratory')}",
              f"- S2(d) per specification cell (`{L.rel('process_summary')}` -> `s2d`; threshold {L.constant(0.80, 'section 9 S2(d): >= 0.80', 2)}):"]
    for cell in sorted(s2d):
        lines.append(f"  - {cell}: fraction kept {_json_value(L, 'process_summary', ps, jpath('s2d', cell, 'fraction_kept'), 2)}, passes "
                     f"{_json_str(L, 'process_summary', ps, jpath('s2d', cell, 'passes'))}")
    if not s2d:
        lines.append(f"  - {L.missing(L.rel('process_summary'), 's2d')}")
    if f5 is not None:
        lines.append(f"- F5: **{'HOLDS' if f5.get('F5') else 'does not hold'}** (F5(i) any cell: {_json_str(L, 'process_f5', f5, 'F5_i_any_cell')}; "
                     f"F5(ii) any cell: {_json_str(L, 'process_f5', f5, 'F5_ii_any_cell')}; `{L.rel('process_f5')}`)")
    else:
        lines.append(f"- F5: {L.missing(L.rel('process_f5'))}")
    win = L.csv("process_winners")
    if win is not None and len(win):
        cols = [c for c in ("cell", "candidate", "support_rank", "p_both_feasible", "p_both", "p_feasible", "n_stages_total",
                            "s2d_fraction_kept", "s2d_passes", "statuses_used") if c in win.columns]
        lines += ["", f"Top recipe per specification cell (`{L.rel('process_winners')}`):", "",
                  "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in win.iterrows():
            cells = []
            for c in cols:
                v = _num(r[c])
                if c in ("cell", "statuses_used", "s2d_passes") or v is None:
                    cells.append(str(r[c]))
                else:
                    nd = 0 if c in ("candidate", "support_rank", "n_stages_total") else (2 if c == "s2d_fraction_kept" else 3)
                    cells.append(L.value(v, L.rel("process_winners"), where_key({"cell": r["cell"]}, c), nd))
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append(f"- winners: {L.missing(L.rel('process_winners'))}")
    lines.append("")
    return lines


def _s1_pass(ctx: Mapping[str, Any]) -> str | None:
    conf = ctx.get("confirmation")
    if conf is not None:
        return f"S1 at confirmation: passed = {((conf.get('S1') or {}).get('passed'))}"
    s1 = ((ctx["decisions"] or {}).get("S1_components") or {})
    a = (s1.get("S1a_M2_vs_B3i") or {}).get("reported_verdict")
    return None if a is None else f"S1(a) {a} in discovery; no confirmation decision yet"


def support_status_table(win: pd.DataFrame | None) -> pd.DataFrame:
    """``tables/process_support_status.csv`` (written by the report): every winning recipe of
    ``tables/process_winners.csv`` with its support category (:func:`classify_support`) -- brief section 34 item 10."""
    cols = ["cell", "candidate", "label", "support_rank", "statuses_used", "support_category"]
    if win is None or not len(win) or "statuses_used" not in win.columns:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({"cell": win["cell"].astype(str) if "cell" in win.columns else range(len(win)),
                        "candidate": win["candidate"] if "candidate" in win.columns else "",
                        "label": win["label"] if "label" in win.columns else "",
                        "support_rank": win["support_rank"] if "support_rank" in win.columns else "",
                        "statuses_used": win["statuses_used"].astype(str)})
    out["support_category"] = out["statuses_used"].map(classify_support)
    return out[cols]


def section_q10(L: Ledger, ctx: Mapping[str, Any]) -> list[str]:
    sup: pd.DataFrame | None = ctx.get("process_support")
    src = L.rel("process_support")
    if sup is None or not len(sup):
        head = (f"{HEAD_UNSHOWN} classify Pr/Nd process recommendations by support: the process step has not run "
                f"({L.missing(L.rel('process_winners'))}).")
    else:
        counts = {c: int((sup["support_category"] == c).sum()) for c in SUPPORT_CATEGORIES}
        L.inputs[src] = True
        head = (f"{HEAD_CAN} classify its Pr/Nd recommendations (one top recipe per specification cell): "
                + ", ".join(f"{L.count(n, src, f'where:support_category={c}')} {c}" for c, n in counts.items())
                + f"; every recommendation carries the label '{sup['label'].iloc[0]}'.")
    lines = ["## Q10. Which Pr/Nd process recommendations are directly supported, transfer-supported, speculative or unsupported?",
             "", f"**{head}**", "",
             "Section 17 item 1 / section 14: only TODGA (nitrate) has direct Pr AND Nd support in the corpus; PC88A, Cyanex 272 and "
             "D2EHPA are not Gen19 predictions (UNSUPPORTED_NO_DIRECT_NO_FAMILY / SAME_FAMILY_ONLY) and gen18's literature case for "
             "them is quoted only as literature. The category of a recipe is decided by the worst section 13 domain status among the "
             "D values it used (`statuses_used` of `tables/process_winners.csv`): "
             + "; ".join(f"{cat} <- {sorted(m)}" for cat, m in SUPPORT_CATEGORY_RULES)
             + f" (a reading of brief section 34 item 10, recorded in `{src}`; not a registered definition).", ""]
    if sup is None or not len(sup):
        lines += [f"- {L.missing(L.rel('process_winners'))}", ""]
        return lines
    cols = list(sup.columns)
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in sup.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines.append("")
    return lines


# --------------------------------------------------------------------------------------------- #
# verdict table, not-run, deviations, reproducibility
# --------------------------------------------------------------------------------------------- #

def verdict_table(L: Ledger, ctx: Mapping[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    dec, conf = ctx["decisions"], ctx["confirmation"]
    s1 = (dec or {}).get("S1_components") or {}
    rows: list[dict[str, Any]] = []

    def add(name: str, rule: str, verdict: str, status: str, source: str) -> None:
        rows.append({"criterion": name, "rule": rule, "verdict": verdict, "status": status, "source": source})

    def comp(key: str) -> str:
        c = s1.get(key)
        return c.get("reported_verdict", "UNDECIDED") if c else L.missing(L.rel("decisions"), f"S1_components.{key}")
    st_s1 = STATUS_DISCOVERY if dec is not None else STATUS_NOT_RUN
    add("S1(a)", "M2 vs B3i passes R19 with delta5 (V5)", comp("S1a_M2_vs_B3i"), st_s1, L.rel("decisions"))
    add("S1(b)", "M2 vs B0 and vs B6r0 pass R19 with 0.05 (V5)",
        " / ".join(comp(k) for k in ("S1b_M2_vs_B0", "S1b_M2_vs_B6r0")), st_s1, L.rel("decisions"))
    s1c = s1.get("S1c_selection")
    add("S1(c)", "paired direction and logSF contrasts on V5-PAIR (confirmation half decides; selection counterweight)",
        (s1c or {}).get("reported_verdict", "UNDECIDED") if s1c else L.missing(L.rel("decisions"), "S1_components.S1c_selection"),
        st_s1 if s1c else STATUS_NOT_RUN, L.rel("decisions"))
    q7 = ctx.get("q7_checks") or {}
    add("S1(d)", "coverage bands 50 / 80 / 95 and per category (V5)", q7.get("s1d", "not computed"),
        STATUS_DISCOVERY if q7.get("s1d") else STATUS_NOT_RUN, L.rel("discovery_summary"))
    add("S1(e)", "Spearman(cell MAE, support_score) <= -0.10, interval excludes 0, components reliable",
        ctx.get("s1e_verdict") or "not computed", STATUS_DISCOVERY if ctx.get("s1e_verdict") else STATUS_NOT_RUN, L.rel("s1e"))
    def tri(v: Any) -> str:
        return "PASS" if v is True else ("FAIL" if v is False else "UNDECIDED")
    s1c_conf = ((conf or {}).get("S1") or {}).get("passed") if conf else None
    add("S1 (confirmation)", "S1 passed for the deployed predictor on the confirmation half, 5 of 5 withheld seeds",
        tri(s1c_conf) if conf else L.missing(L.rel("confirmation"), "S1.passed"), STATUS_CONFIRMATION if conf else STATUS_NOT_RUN,
        L.rel("confirmation"))
    s2 = (conf or {}).get("S2") or {}
    add("S2 (overall)", "S2 passed in the single V6 confirmation run", tri(s2.get("passed")) if conf else L.missing(L.rel("confirmation"), "S2.passed"),
        STATUS_CONFIRMATION if conf else STATUS_NOT_RUN, L.rel("confirmation"))
    for k, rule in (("a", "sign in >= 11 of 13 systems; paired direction +0.05"), ("b", "logSF MAE <= FLAT - 0.02 and <= lookup; log D MAE <= L5"),
                    ("c", "pooled coverage 80 % in [0.70, 0.90], 95 % >= 0.88"), ("d", "top recipe stable in >= 0.80 of re-rankings")):
        blk = s2.get(k) if isinstance(s2.get(k), Mapping) else None
        v = tri(blk.get("pass")) if blk else L.missing(L.rel("confirmation"), f"S2.{k}")
        add(f"S2({k})", rule, v, STATUS_CONFIRMATION if blk else STATUS_NOT_RUN, L.rel("confirmation"))
    f1 = L.json("f1_check")
    add("F1", "gains only on random split (V0 beats B3, S1(a) fails, V1 interval includes 0)",
        ("HOLDS" if f1.get("failure") else "does not hold") if f1 else L.missing(L.rel("f1_check")),
        STATUS_CONFIRMATION if f1 else STATUS_NOT_RUN, L.rel("f1_check"))
    add("F2", "deployed predictor's gain over B1 or B2 includes 0 on V5 / V1 / V2", ctx.get("f2_verdict") or "not computed",
        STATUS_DISCOVERY if ctx.get("f2_verdict") else STATUS_NOT_RUN, L.rel("discovery_contrasts_exploratory"))
    add("F3", "80 % coverage outside [0.60, 0.95] or 95 % < 0.85 (V5 / V1)", q7.get("f3", "not computed"),
        STATUS_DISCOVERY if q7.get("f3") else STATUS_NOT_RUN, L.rel("discovery_summary"))
    f4 = L.json("h3_f4")
    add("F4", "WITHOUT beats the deployed WITH configuration on the Ln test set",
        ("HOLDS" if f4.get("failure") else ("does not hold" if f4.get("status") == "computed" else "not computed")) if f4 else L.missing(L.rel("h3_f4")),
        STATUS_DISCOVERY if f4 else STATUS_NOT_RUN, L.rel("h3_f4"))
    f5 = L.json("process_f5")
    add("F5", "process optimum depends on unsupported predictions (F5(i) code check; F5(ii) barred vs allowed)",
        ("HOLDS" if f5.get("F5") else ("does not hold" if f5.get("F5") is False else "UNDECIDED")) if f5 else L.missing(L.rel("process_f5"), "F5"),
        (STATUS_CONFIRMATION if (f5.get("label") == "registered") else STATUS_DISCOVERY + " (transfer-unsupported)") if f5 else STATUS_NOT_RUN,
        L.rel("process_f5"))
    add("F6", "S1(a) fails and no ladder step M1-M6 is kept", ctx.get("f6_verdict") or "not computed",
        STATUS_DISCOVERY if dec is not None else STATUS_NOT_RUN, L.rel("decisions"))
    lines = ["## Verdict table: S1, S2 and the failure conditions F1-F6 (sections 9-10)", "",
             "Every discovery verdict is the selection half on seed 104729 and optimistically biased; a learned-arm R19 verdict is "
             "at best UNDECIDED in discovery (R19 item 4 NOT_EVALUATED, addendum 1 item 3). `confirmation` rows come only from "
             "the orchestrator's confirmation files.", "",
             "| criterion | rule | verdict | status | source |", "|---|---|---|---|---|"]
    lines += [f"| {r['criterion']} | {r['rule']} | {r['verdict']} | {r['status']} | `{r['source']}` |" for r in rows]
    lines.append("")
    return lines, rows


def _f2_verdict(L: Ledger, ctx: Mapping[str, Any]) -> str | None:
    """F2 from the exploratory contrasts (deployed arm vs B1 and vs B2 on V5 / V1 / V2) when they exist."""
    dep = ctx["deployed"].get("arm")
    ex = L.csv("discovery_contrasts_exploratory")
    if dep is None or ex is None or "key" not in ex.columns:
        return None
    found, holds = [], False
    for comp in ("B1", "B2"):
        for design in ("V5", "V1", "V2"):
            row = contrast_row(L, "discovery_contrasts_exploratory", f"{dep} vs {comp}@{design}")
            if row is None:
                continue
            lo = _num(_s(row, "percentile_low"))
            found.append(f"{comp}@{design}: low {contrast_cell(L, 'discovery_contrasts_exploratory', row, 'percentile_low')}")
            holds |= lo is None or lo <= 0
    if not found:
        return None
    return ("HOLDS" if holds else "does not hold") + " (" + "; ".join(found) + ")"


def not_run_section(L: Ledger, ctx: Mapping[str, Any]) -> list[str]:
    dec = ctx["decisions"]
    lines = ["## What was not run and why", ""]
    if dec is None:
        lines += [f"- {L.missing(L.rel('decisions'))}", ""]
    else:
        nr = dec.get("not_run") or {}
        a1 = nr.get("addendum_1") or {}
        seeds_txt = ", ".join(L.text(str(s), L.rel("decisions"), "not_run.addendum_1.discovery_seeds")
                              for s in (a1.get("discovery_seeds") or [])) or "none listed"
        lines += ["**POST-HOC addendum 1 (compute-driven; the sealed section 7 plan priced at 1,535.7 CPU-hours against 60 h):**", "",
                  f"- discovery seeds not run for learned arms: {seeds_txt} (R19 item 4: {a1.get('r19_item_4')})",
                  "- refit sensitivities not run for learned arms (R19 item 6 on the reduced set, labelled "
                  f"'{ADDENDUM_LABEL}'): " + "; ".join(f"{d}: {v}" for d, v in (a1.get("sensitivities") or {}).items()),
                  f"- heavy-arm V5-P runs: {a1.get('v5p_heavy_arm_runs')}",
                  f"- comparator-interval jobs: {a1.get('comparator_interval_jobs')}",
                  f"- M3+: {nr.get('M3+')}", f"- V0: {nr.get('V0')}", f"- confirmation: {nr.get('confirmation')}", "",
                  "**Registered contrasts of section 19 not evaluated by the scorer** (`registered_family_accounting`):", ""]
        for c in nr.get("registered_family") or []:
            lines.append(f"- {c['family']}: {c['contrast']} @ {c['design']} -- {c['status']}")
        acc = dec.get("registered_family_accounting") or {}
        if acc:
            lines += ["", f"Family accounting: m_full = {_json_value(L, 'decisions', dec, 'registered_family_accounting.m_full', 0)} "
                      f"(m_discovery = {_json_value(L, 'decisions', dec, 'registered_family_accounting.m_discovery', 0)}), evaluated = "
                      f"{_json_value(L, 'decisions', dec, 'registered_family_accounting.m_evaluated', 0)}; BH-adjusted p is shown beside the raw p in "
                      "every contrast table and never decides (section 8)."]
        stop = ctx["stop_rule"]
        lines += ["", f"**Stop rule** (section 7 item 4): stop = {_json_str(L, 'stop_rule', stop, 'stop')}; consequences: "
                  f"{(stop or {}).get('consequences') if stop else L.missing(L.rel('stop_rule'))}"]
    ladder = L.json("ladder")
    if ladder is not None:
        lines += ["", f"**Ladder demotions** (section 7 item 5 budget): {ladder.get('demoted') or 'none'}; notes: {ladder.get('notes') or 'none'}"]
    wc = L.json("wall_clock")
    if wc is not None:
        used = json_get(wc, "total_hours") if "total_hours" in wc else None
        cap = json_get(wc, "budget.budget_hours") if (wc.get("budget") or {}).get("budget_hours") is not None else None
        over = (None if used is None or cap is None else round(max(0.0, float(used) - float(cap)), 4))
        lines += ["", f"**Discovery wall clock**: {_json_value(L, 'wall_clock', wc, 'total_hours', 2)} h of the "
                  f"{_json_value(L, 'wall_clock', wc, 'budget.budget_hours', 0)} h budget (`{L.rel('wall_clock')}`); exhausted = "
                  f"{_json_str(L, 'wall_clock', wc, 'budget.exhausted')}"
                  + ("" if not over else f" -- **an overrun of {over:.4f} h**, acted on by the discovery ledger's own "
                                         "demotion of M7-M3 at the time and added to no later budget (task X finding V-L4)")]
    lwc = L.json("ladder_wall_clock")
    if lwc is not None:
        key = "total_hours" if "total_hours" in lwc else ("ladder_hours" if "ladder_hours" in lwc else None)
        if key is not None:
            inv = lwc.get("invocations") or []
            proc = sum(float(i.get("process_seconds") or 0.0) for i in inv)
            lines.append(f"- ladder wall clock: {_json_value(L, 'ladder_wall_clock', lwc, key, 4)} h charged to the "
                         f"{_json_value(L, 'ladder_wall_clock', lwc, 'budget_hours', 0)} h ladder budget over "
                         f"{L.count(len(inv), L.rel('ladder_wall_clock'), 'invocations')} invocation(s) "
                         f"(`{L.rel('ladder_wall_clock')}`)"
                         + (f"; {proc:.1f} s of process wall clock, so the runner ran and skipped every step"
                            if proc else "; no per-invocation process wall clock recorded"))
    ps = ctx["plan_state"]
    if ps is not None:
        lines += ["", f"**B6 checks** (sections 3.1 / 3.2, `{L.rel('plan_state')}`): V5 batched-vs-exact = {ps.get('v5_batched_check')} "
                  f"(heavy V5 scheme {ps.get('heavy_v5_scheme')}, label '{ps.get('heavy_v5_label')}'); V1 ten-fold = {ps.get('v1_tenfold_check')} "
                  f"(heavy V1 scheme {ps.get('heavy_v1_scheme')}); freezing candidates: {len(ps.get('freezing_candidates') or [])}"]
    lines.append("")
    return lines


def parse_deviations(prereg_text: str) -> list[str]:
    """The numbered bold items of section 17 of the sealed text."""
    out, inside = [], False
    for ln in prereg_text.splitlines():
        if ln.startswith("## 17."):
            inside = True
            continue
        if inside and ln.startswith("## "):
            break
        if inside:
            m = _DEVIATION_RE.match(ln.strip())
            if m:
                out.append(f"{m.group(1)}. {m.group(2)}")
    return out


def parse_addenda(prereg_text: str) -> list[str]:
    return [f"addendum {m.group(1)} ({m.group(2)})" for ln in prereg_text.splitlines() for m in [_ADDENDUM_RE.match(ln)] if m]


#: the documented count rules of the sealed text (``count:<rule>`` keys on ``preregistration.md``)
TEXT_COUNT_RULES: dict[str, Any] = {"deviations": lambda t: len(parse_deviations(t)), "addenda": lambda t: len(parse_addenda(t))}


def readings_needing_addenda(L: Ledger) -> list[str]:
    """Every reading string in the decision JSON files that names a POST-HOC addendum."""
    out = []
    for name in ("decisions", "h3_summary", "power_checks", "ladder", "m7_metrics"):
        body = L.json(name)
        if body is None:
            continue
        for key, text in _walk_strings(body):
            if "post-hoc addendum" in text.lower() and "needs" in text.lower():
                out.append(f"`{L.rel(name)}` -> `{key}`: {text[:160]}{'...' if len(text) > 160 else ''}")
    return out


def _walk_strings(obj: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield from _walk_strings(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


def deviations_section(L: Ledger) -> list[str]:
    lines = ["## Deviations from the brief (pre-registration section 17) and the POST-HOC addenda", ""]
    if not L.exists("prereg"):
        lines += [f"- {L.missing(L.rel('prereg'))}", ""]
        return lines
    text = L.path("prereg").read_text(encoding="utf-8")
    devs = parse_deviations(text)
    lines += [f"Section 17 lists {L.count(len(devs), L.rel('prereg'), 'deviations')} deviations forced by the data "
              "(`preregistration.md` section 17; rule `count:deviations` = the numbered bold items under `## 17.`):", ""]
    lines += [f"- {d}" for d in devs]
    adds = parse_addenda(text)
    lines += ["", f"POST-HOC addenda below the sealed footer: {L.count(len(adds), L.rel('prereg'), 'addenda')} "
              "(rule `count:addenda` = the headings `## POST-HOC addendum N (date ...)`)", ""]
    lines += [f"- {a}" for a in adds] or ["- none"]
    need = readings_needing_addenda(L)
    lines += ["", f"Readings recorded by the runners that need a further POST-HOC addendum before a result is quoted as registered "
              f"({len(need)} found in the decision files):", ""]
    lines += [f"- {n}" for n in need] or ["- none found in the available decision files"]
    lines.append("")
    return lines


def digest_registry_lines(L: Ledger, registry: Mapping[str, Any] | None) -> list[str]:
    """Addendum 2 item 5, "the report prints every registry entry": one line per stage (below-footer digest, code digest,
    addenda count, git head, note) and one per logged code change, every token through the ledger; absent registry ->
    one line saying the constants govern (nothing invented)."""
    rel = L.rel("digest_registry")
    if registry is None:
        return [f"- digest registry (`{rel}`): absent -- the gate constants of `gen19ct/evaluation/discovery.py` govern "
                "every stage (addendum 2 item 5 replaces them by the registry once it is created)"]
    stages = registry.get("stages") or {}
    lines = [f"- digest registry (`{rel}`, addendum 2 item 5): {L.count(len(stages), rel, 'len:stages')} stage entr"
             f"{'y' if len(stages) == 1 else 'ies'}, {L.count(len(registry.get('code_changes') or []), rel, 'len:code_changes')} "
             "logged code change(s); a record is verified against the entry of its stage, never against the live text or code"]
    for stage, e in sorted(stages.items()):
        bf, cd = str(e.get("below_footer_sha256")), str(e.get("code_digest"))
        lines.append(f"  - stage `{stage}`: below-footer SHA-256 `{L.text(bf, rel, jpath('stages', stage, 'below_footer_sha256'))}`, "
                     f"code digest `{L.text(cd, rel, jpath('stages', stage, 'code_digest'))}`, "
                     f"{L.value(int(e.get('addenda_count', 0)), rel, jpath('stages', stage, 'addenda_count'), 0)} addenda, "
                     f"git head `{e.get('git_head') or 'unknown'}`, registered {e.get('registered_utc') or 'unknown'}: "
                     f"{e.get('note') or ''}")
    for i, ch in enumerate(registry.get("code_changes") or []):
        files = ", ".join(f"`{f.get('path')}`" for f in (ch.get("files") or []))
        lines.append(f"  - code change {i + 1} (commit `{ch.get('commit') or 'uncommitted'}`, {ch.get('logged_utc') or 'undated'}): "
                     f"{files} -- {ch.get('reason') or ''}; {ch.get('effect_on_records') or 'validates or invalidates no record'}")
    for s in registry.get("superseded") or []:
        lines.append(f"  - superseded entry of stage `{s.get('stage')}` ({s.get('superseded_utc') or 'undated'}): "
                     f"{s.get('superseded_by_note') or ''}")
    return lines


def reproducibility_section(L: Ledger, extra: Mapping[str, Any]) -> list[str]:
    lines = ["## Reproducibility", ""]
    lines.append(f"- git HEAD at report build: `{extra.get('git_head') or 'unknown (git not available)'}`")
    if L.exists("prereg_sha"):
        sha = L.path("prereg_sha").read_text(encoding="utf-8").strip().split()[0]
        lines.append(f"- sealed pre-registration SHA-256 (`{L.rel('prereg_sha')}`): `{L.text(sha, L.rel('prereg_sha'), sha)}`")
    else:
        lines.append(f"- sealed pre-registration digest: {L.missing(L.rel('prereg_sha'))}")
    registry = L.json("digest_registry") if (L.root / INPUTS["digest_registry"]).exists() else None
    if extra.get("addenda_sha256"):
        # the live addenda digest is printed only against a file that holds the literal, else the disagreement is stated
        # without the token (task X finding VL2-01).  Addendum 2 item 5: when manifests/digest_registry.json exists, that
        # file is the reference (the stage entries whose below-footer digest equals the live text are named); otherwise
        # the code constant discovery.REGISTERED_ADDENDA_SHA256 is, as before
        adds_lit = str(extra["addenda_sha256"])
        if registry is not None:
            reg_src = L.rel("digest_registry")
            stages = [s for s, e in sorted((registry.get("stages") or {}).items()) if e.get("below_footer_sha256") == adds_lit]
            if stages and literal_in_file(L.root, reg_src, adds_lit):
                adds_txt = (f"`{L.text(adds_lit, reg_src, adds_lit)}` (= the below-footer digest of registry stage(s) "
                            f"{', '.join(f'`{s}`' for s in stages)} in `{reg_src}`)")
            else:
                adds_txt = (f"matches NO stage entry of `{reg_src}` (not printed: the live text differs from every registered "
                            "below-footer digest; a post-discovery stage must be registered under it before its runner starts)")
        else:
            adds_src = "gen19ct/evaluation/discovery.py"
            if literal_in_file(L.root, adds_src, adds_lit):
                adds_txt = f"`{L.text(adds_lit, adds_src, adds_lit)}` (= `REGISTERED_ADDENDA_SHA256` in `{adds_src}`)"
            else:
                adds_txt = f"DIFFERS from `REGISTERED_ADDENDA_SHA256` in `{adds_src}` (not printed: no file holds it)"
        n_add = len(parse_addenda(L.path("prereg").read_text(encoding="utf-8"))) if L.exists("prereg") else None
        n_txt = (f"{L.count(n_add, L.rel('prereg'), 'addenda')} addendum heading(s) in `{L.rel('prereg')}`" if n_add is not None
                 else L.missing(L.rel("prereg")))
        rec_lit = str(extra.get("recomputed"))
        if L.exists("prereg_sha") and literal_in_file(L.root, L.rel("prereg_sha"), rec_lit):
            rec_txt = f"`{L.text(rec_lit, L.rel('prereg_sha'), rec_lit)}` (equals the sealed digest file)"
        else:
            rec_txt = f"DIFFERS from `{L.rel('prereg_sha')}` (not printed)"
        lines.append(f"- SHA-256 of the POST-HOC addenda text (below the footer, LF-normalised): {adds_txt}; {n_txt}; "
                     f"footer digest recomputed from the text: {rec_txt}")
    lines += digest_registry_lines(L, registry)
    if L.exists("seed_commitment"):
        c = L.path("seed_commitment").read_text(encoding="utf-8").strip().split()[0]
        lines.append(f"- confirmation-seed commitment (`{L.rel('seed_commitment')}`): `{L.text(c, L.rel('seed_commitment'), c)}`; the seeds are "
                     "revealed only in `decisions/CONFIRMATION.md`")
    dh = L.csv("dataset_hashes")
    if dh is not None and "is_headline_dataset_hash" in dh.columns:
        sub = dh[dh["is_headline_dataset_hash"].astype(str) == "True"]
        if len(sub) == 1:
            sha = sub.iloc[0]["sha256"]
            lines.append(f"- dataset hash (`{L.rel('dataset_hashes')}`, headline row `{sub.iloc[0]['path']}`): "
                         f"`{L.text(sha, L.rel('dataset_hashes'), sha)}`"
                         + ("" if sha == paths.ARCHIVE_MASTER_SHA256 else " -- DIFFERS from `paths.ARCHIVE_MASTER_SHA256`"))
        else:
            lines.append(f"- dataset hash: {len(sub)} headline rows in `{L.rel('dataset_hashes')}`")
    else:
        lines.append(f"- dataset hash: {L.missing(L.rel('dataset_hashes'))}")
    mans = sorted(q for q in (L.root / "manifests").glob("*.json") if q.is_file()) if (L.root / "manifests").exists() else []
    n_man = L.count(len(mans), "manifests", "glob:*.json") if (L.root / "manifests").exists() else L.missing("manifests")
    lines += ["", f"Manifests (`manifests/<script>.json`, deterministic; `manifests/run_info/` volatile): {n_man} files", ""]
    for m in mans:
        try:
            b = json.loads(m.read_text(encoding="utf-8"))
            outs = (f"outputs {L.count(len(b['outputs']), f'manifests/{m.name}', 'len:outputs')}"
                    if isinstance(b.get("outputs"), list) else "outputs: not recorded")
            lines.append(f"- `manifests/{m.name}`: script {b.get('script')}, git HEAD `{b.get('git_head')}`, {outs}")
        except (OSError, ValueError):
            lines.append(f"- `manifests/{m.name}`: unreadable")
    lines += ["", "Inputs this report read (exists / missing) are listed in `evaluation/report/report_inputs.json`; every printed "
              "number with its source path and key is in `tables/report_numbers.csv`, re-resolved after writing "
              "(`verification` block of `manifests/g19_build_report.json`).", ""]
    return lines


# --------------------------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------------------------- #

def _context(L: Ledger) -> dict[str, Any]:
    dec = L.json("decisions")
    ctx: dict[str, Any] = {"decisions": dec, "contrasts": L.csv("discovery_contrasts"), "difficulty": L.json("difficulty"),
                           "stop_rule": L.json("stop_rule"), "plan_state": L.json("plan_state"),
                           "confirmation": L.json("confirmation"), "deployed": deployed_predictor(dec, L.json("ladder")),
                           "process_support": support_status_table(L.csv("process_winners"))}
    return ctx


def _q7_checks(L: Ledger, ctx: Mapping[str, Any]) -> dict[str, str]:
    """S1(d) / F3 verdict strings for the verdict table (the same computation as :func:`section_q7`, no new number)."""
    dep = ctx["deployed"].get("arm")
    if not dep:
        return {}
    vals = _deployed_coverage(L, dep)
    if not vals or any(v is None for v in vals.values()):
        return {}
    cats = {}
    sub, _ = _coverage_by_category(L, dep)
    if sub is not None:
        for _, r in sub.iterrows():
            cats[str(r["category"])] = (_num(r["value"]) or float("nan"), int(float(r["category_units"] or 0)))
    s1d = EC.s1d_check(vals, cats)
    f3 = EC.f3_check(vals[0.80], vals[0.95])
    return {"s1d": "PASS" if s1d["pass"] else "FAIL", "f3": "HOLDS" if f3["failure"] else "does not hold"}


def _s1e_verdict(L: Ledger, ctx: Mapping[str, Any]) -> str | None:
    s1e = L.csv("s1e")
    dep = ctx["deployed"].get("arm")
    if s1e is None or not dep:
        return None
    row = _pick(s1e, {"arm": dep, "design": "V5", "x": "support_score"})
    if row is None:
        return None
    rel = L.csv("reliability")
    comps = None
    if rel is not None:
        sub = rel[rel["quantity"].astype(str) == "support_score_components"]
        comps = bool((sub["gate"].astype(str).str.upper() == "PASS").all()) if len(sub) else None
    rho, hi = _num(row.get("rho")), _num(row.get("percentile_high"))
    if comps is False:
        return "UNDECIDED (unreliable)"
    if rho is None or hi is None or comps is None:
        return "UNDECIDED"
    return "PASS" if (rho <= ET.S1E_MAX_SPEARMAN and hi < 0) else "FAIL"


def build_report(root: Path, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Every report artefact as in-memory objects: ``report`` (GEN19_REPORT.md), ``summary`` (SUMMARY.md), ``d02``
    (decisions/D02_factorization.md), ``claims`` (tables/claims.json), ``numbers`` (tables/report_numbers.csv),
    ``inputs`` (evaluation/report/report_inputs.json) and the ``ledger``."""
    L = Ledger(root)
    extra = dict(extra or {})
    ctx = _context(L)
    ctx["q7_checks"] = _q7_checks(L, ctx)
    ctx["s1e_verdict"] = _s1e_verdict(L, ctx)
    ctx["f2_verdict"] = _f2_verdict(L, ctx)
    lad = (ctx["decisions"] or {}).get("ladder") or {}
    ctx["f6_verdict"] = _f6_text(L, ctx, lad, L.json("ladder")) if ctx["decisions"] is not None else None
    sections = (section_q1, section_q2, section_q3, section_q4, section_q5, section_q6, section_q7, section_q8,
                section_q9, section_q10)
    body: list[str] = []
    headlines: list[tuple[str, str, str]] = []
    for (qid, question), fn in zip(QUESTIONS, sections):
        lines = fn(L, ctx)
        body += lines
        head = next((ln for ln in lines if ln.startswith("**Under deliberately hidden chemistry")), "")
        headlines.append((qid, question, head.strip("*")))
    vt_lines, vt_rows = verdict_table(L, ctx)
    body += vt_lines
    body += not_run_section(L, ctx)
    body += deviations_section(L)
    body += reproducibility_section(L, extra)
    header = ["# GEN19 report -- the chemistry-transfer engine under deliberately hidden chemistry", "",
              f"*Generated by `scripts/g19_build_report.py` from the files under `generations/gen19_chem_transfer/` "
              f"(git HEAD `{extra.get('git_head') or 'unknown'}`). Every number cites its source file and key "
              "(`tables/report_numbers.csv`); where an input does not exist the text says `not computed (input missing: ...)`. "
              "Discovery numbers are the selection half on seed 104729 and optimistically biased; only the confirmation files "
              "can confirm a claim. A learned-arm contrast carries the label `reduced sensitivity set (addendum 1)` where R19 "
              "item 6 was decided on the reduced set.*", "",
              "The Gen19 standard (brief section 34): not \"model X achieved R^2 = Y\" but whether, under deliberately hidden "
              "chemistry, the system can reconstruct missing extraction behaviour to a degree sufficient for process "
              "decision-making. The ten questions below are answered in one sentence first, then the evidence.", ""]
    report = "\n".join(header + body) + "\n"
    summary = summary_text(headlines, vt_rows, extra)
    d02 = d02_text(L, ctx, extra)
    claims = build_claims(L)
    numbers = L.frame()
    inputs = {"schema": SCHEMA, "root": str(root), "inputs": {k: {"path": v, "exists": (Path(root) / v).exists()}
                                                                for k, v in INPUTS.items()},
              "missing_inputs": sorted(L.missing_inputs), "n_numbers": int(len(numbers)),
              "deployed_predictor": ctx["deployed"]}
    return {"report": report, "summary": summary, "d02": d02, "claims": {"schema": SCHEMA, "claims": claims,
                                                                          "verdicts": vt_rows, "headlines": [
                                                                              {"question": q, "text": qq, "headline": h}
                                                                              for q, qq, h in headlines]},
            "numbers": numbers, "inputs": inputs, "ledger": L, "context": ctx}


def summary_text(headlines: Sequence[tuple[str, str, str]], verdicts: Sequence[Mapping[str, Any]],
                 extra: Mapping[str, Any]) -> str:
    lines = ["# GEN19 summary -- one page", "",
             f"*Generated by `scripts/g19_build_report.py` (git HEAD `{extra.get('git_head') or 'unknown'}`); the evidence is in "
             "`GEN19_REPORT.md`, every number in `tables/report_numbers.csv`. Discovery = selection half, seed 104729, "
             "optimistically biased; only `confirmation` rows are confirmed.*", "", "## The ten questions (brief section 34)", ""]
    for qid, q, h in headlines:
        lines.append(f"{qid[1:]}. **{q}** {h or 'not computed'}")
    lines += ["", "## Verdicts (sections 9-10)", "", "| criterion | verdict | status |", "|---|---|---|"]
    lines += [f"| {r['criterion']} | {r['verdict']} | {r['status']} |" for r in verdicts]
    lines.append("")
    return "\n".join(lines)


def d02_text(L: Ledger, ctx: Mapping[str, Any], extra: Mapping[str, Any]) -> str:
    """``decisions/D02_factorization.md`` (brief section 29 format): H1, H1b and H4 from the scorer's contrast files, the
    ladder M1 / M2 from decisions.json and M3-M7 from the ladder runner."""
    con = ctx["contrasts"]
    dec = ctx["decisions"]
    lad = (dec or {}).get("ladder") or {}
    ladder = L.json("ladder")
    lines = ["# D02 -- factorisation: H1, H1b, H4 and the ladder", "",
             f"*Generated by `scripts/g19_build_report.py` at git HEAD `{extra.get('git_head') or 'unknown'}` from "
             f"`{L.rel('discovery_contrasts')}`, `{L.rel('decisions')}` and `{L.rel('ladder')}`. Selection half, seed 104729, "
             "optimistically biased; R19 item 4 NOT_EVALUATED in discovery (addendum 1 item 3); a learned-arm contrast's item 6 is "
             f"the '{ADDENDUM_LABEL}'. Nothing here is confirmed.*", "",
             "## question", "",
             "Does factorising metals and extractants (M2: bilinear e_m^T W e_l) reconstruct hidden metal x extractant cells "
             "better than the within-system lookup B3i (H1, section 9 S1(a)), does the linear factorisation B6 (H1b), and does the "
             "factorised model beat flat categorical (FLAT_CAT, B6r0) and descriptor (M0) modelling under V5 (H4, brief section 31)?",
             "", "## evidence", "", "| file | contents |", "|---|---|",
             f"| `{L.rel('discovery_contrasts')}` | R19 rows per registered contrast and cluster unit, BH-adjusted p beside raw p |",
             f"| `{L.rel('discovery_r19')}` | R19 items 1-6 per contrast |",
             f"| `{L.rel('decisions')}` | stop rule, ladder M1 / M2, S1 components, freezing screen, addendum-1 block |",
             f"| `{L.rel('ladder')}` | ladder M3-M7 decisions (when run) |", "", "## metrics", ""]
    lines += _regime_header()
    for k, note in (("M2 vs B3i@V5", "H1"), ("B6 vs B3i@V5", "H1b"), ("M2 vs FLAT_CAT@V5", "H4"), ("M2 vs M0@V5", "H4"),
                    ("M3 vs M2@V5", "H4"), ("B6 vs B6r0@V5", "H4")):
        lines.append(_regime_row(L, k, claim_status(k, con, ctx["confirmation"]), note))
    if con is not None and "key" in con.columns:
        for k in sorted(k for k in con["key"].astype(str).unique() if k.endswith("#ladder")):
            lines.append(_regime_row(L, k, STATUS_DISCOVERY, "ladder"))
    lines += ["", "## null / supported / ambiguous", ""]

    # section 8 / addendum 3 item 3: "the power check runs BEFORE any null is reported", and "a null with kappa_min >
    # 0.25 log D is reported as UNDECIDED (underpowered)".  A FAILED contrast is therefore printed as a null ONLY when
    # its own power record screened it as an informative null; with an underpowered record, or with no registered power
    # check at all, it prints UNDECIDED and says which (task X finding V-P01).
    pw_checks = ((L.json("power_checks") or {}).get("checks") or [])
    pw_index = {str(c.get("contrast")): (i, c) for i, c in enumerate(pw_checks)}

    def power_reading(key: str) -> str:
        # POST-HOC addendum 4 item 4: the three outcomes the check may report, and no other.  A POWERED_NOT_A_NULL /
        # NOT_A_NULL_UNDERPOWERED record is a contrast that is NOT a null -- it must never print as "UNDECIDED
        # (underpowered)" (the earlier fall-through), because the registered wording "null" never applies to it.
        if key not in pw_index:
            return (f"**{PW.NO_POWER_CHECK_LABEL}** -- section 8 requires the signal-injection check before "
                    f"a failed contrast is reported as a null ({L.missing(L.rel('power_checks'), 'checks[].contrast=' + key)})")
        i, c = pw_index[key]
        km, pv = c.get("kappa_min"), str(c.get("verdict"))
        shown = (L.value(km, L.rel("power_checks"), f"checks[{i}].kappa_min", 2) if km is not None
                 else "none in the registered grid")
        if pv == "INFORMATIVE_NULL":
            return f"null (informative: kappa_min {shown}, `{L.rel('power_checks')}` -> `checks[{i}].verdict`)"
        if pv in ("POWERED_NOT_A_NULL", "NOT_A_NULL_UNDERPOWERED"):
            return (f"**{PW.REPORTED_LABEL[pv]}** -- not a null: {c.get('uninjected_point_favours') or 'the point estimate has a direction'}; "
                    f"kappa_min {shown} (`{L.rel('power_checks')}` -> `checks[{i}].verdict` {pv})")
        return (f"**{PW.REPORTED_LABEL.get(pv, 'UNDECIDED (underpowered)')}** -- kappa_min {shown} "
                f"(`{L.rel('power_checks')}` -> `checks[{i}].verdict` {pv})")

    def verdict_of(key: str) -> str:
        row = contrast_row(L, "discovery_contrasts", key)
        if row is None:
            return f"ambiguous / not run ({L.missing(L.rel('discovery_contrasts'), 'key=' + key)})"
        v = _s(row, "reported_verdict")
        fs = _s(row, "verdict_freezing_screen")
        if v == "PASS":
            return "supported"
        if v == "FAIL":
            return power_reading(key)
        return f"ambiguous ({v or 'UNDECIDED'}; freezing screen {fs or 'n/a'})"
    for k, label in (("M2 vs B3i@V5", "H1"), ("B6 vs B3i@V5", "H1b"), ("M2 vs FLAT_CAT@V5", "H4 factorised vs flat"),
                     ("M2 vs M0@V5", "H4 factorised vs descriptor"), ("M3 vs M2@V5", "H4 + mechanism"),
                     ("B6 vs B6r0@V5", "H4 linear analogue")):
        lines.append(f"- {label} ({k}): {verdict_of(k)}")
    for s in ("M1", "M2"):
        rec = lad.get(s)
        lines.append(f"- ladder {s}: {'kept' if rec and rec.get('kept') is True else ('removed' if rec and rec.get('kept') is False else 'pending / not run')}")
    if ladder is not None:
        for s, rec in (ladder.get("steps") or {}).items():
            if s in ("M3", "M4", "M5", "M6", "M7"):
                lines.append(f"- ladder {s}: {rec.get('status')} (kept {rec.get('kept')})")
    lines += ["", "## decision", "",
              f"- deployed predictor: **{ctx['deployed'].get('arm') or 'undecided'}** -- {ctx['deployed'].get('basis')}",
              f"- stop rule: {_json_str(L, 'stop_rule', ctx['stop_rule'], 'stop')}", "",
              "## next action", "",
              "- confirmation of the frozen claims (<= 5, `decisions/CONFIRMATION_PLAN.md`) on the withheld seeds and the confirmation "
              "half only (section 15); V6 once;",
              "- the section 8 power check for every failed H1 / H1b contrast before it is reported as a null (`scripts/g19_run_power.py`);",
              "- a POST-HOC addendum for every runner reading listed in the report's deviations section.", ""]
    return "\n".join(lines)


def write_outputs(root: Path, res: Mapping[str, Any]) -> list[Path]:
    """Write every artefact of :func:`build_report` under ``root``."""
    from gen19ct.manifest import write_csv, write_json, write_text

    root = Path(root)
    outs = []
    sup = (res.get("context") or {}).get("process_support")
    if isinstance(sup, pd.DataFrame) and len(sup):                    # cited by Q10: written before the numbers are verified
        outs.append(write_csv(sup, root / INPUTS["process_support"]))
    outs += [write_text(root / "GEN19_REPORT.md", res["report"]), write_text(root / "SUMMARY.md", res["summary"]),
             write_text(root / "decisions" / "D02_factorization.md", res["d02"]),
             write_json(root / "tables" / "claims.json", _json_safe(res["claims"])),
             write_csv(res["numbers"], root / "tables" / "report_numbers.csv"),
             write_json(root / "evaluation" / "report" / "report_inputs.json", _json_safe(res["inputs"]))]
    return outs


def _json_safe(obj: Any) -> Any:
    def default(o: Any) -> Any:
        if hasattr(o, "item"):
            return o.item()
        if isinstance(o, Path):
            return o.as_posix()
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return str(o)
    return json.loads(json.dumps(obj, default=default), parse_constant=lambda c: None)

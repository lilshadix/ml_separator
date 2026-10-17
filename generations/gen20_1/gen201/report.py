"""gen20.1 S10 (VERDICTS + REPORT): TAB-01..TAB-14, RULE-T / RULE-C / RULE-H4 verdicts, NC1, BH over the
exploratory family and GEN20_1_REPORT.md, from saved tables only (IMPLEMENTATION_PLAN.md §5, §6, §9 S10).

Inputs: ``fitting.load_tables(label, design, variant)`` (the long tables, verified against their DONE markers by
the models side), the selection records next to them, and the label-free row manifests
(``manifests/row_manifests/*.parquet``; COH-REPCAP's label-derived manifest from S5).  Populations are selected
at scoring time by joining the long tables to the manifest keys (ci, im) / (ci, ia, ib) per split seed; every
manifest key must be present in every seed (coverage = row manifests) and every scored arm finite.

Nothing here chooses anything on an outer score: arms, comparators, populations, metrics and thresholds are the
sealed ones, classified from unrounded floats by ``gen201.rules``.  A ``smoke*`` label is refused (LT-29).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import bootstrap
from . import inference as I
from . import metrics as M
from . import rules as R

DESIGNS: tuple[str, ...] = ("BP", "B", "BR", "BQ", "BPG", "A")
VARIANTS: tuple[str, ...] = ("primary", "nodmd")
N_UNITS = 25
ROW_MANIFESTS = bootstrap.MANIFESTS / "row_manifests"
REPORT_MD = bootstrap.G201 / "GEN20_1_REPORT.md"
S5_DESCRIBE_JSON = "describe_g201.json"      # scripts/g201_describe.py OUT_JSON (MET-13 cards, MET-19 noise, REPCAP)

LOGD_TABLE_ARMS: tuple[str, ...] = ("A", "BD_MEAN", "BD_METAL", "BD_COND", "BD_NN", "BD_LEVEL", "BD_LEVEL_CURVE",
                                    "BL_D_INNER", "A_PRIME")
LOGSF_TABLE_ARMS: tuple[str, ...] = ("B", "C", "B_selSF", "C_selSF", "C_ACFG", "C_PAIR", "FLAT", "PAIRMEAN",
                                     "G14_ZERO", "MEAN_CURVE_REF", "NN_TANIMOTO_REF", "BL_SF_INNER", "BD_METAL_SF",
                                     "BD_COND_SF", "BD_NN_SF", "BD_LEVEL_CURVE_SF")
LOGD_WITHIN_ONLY: tuple[str, ...] = ("C",)               # the centred prediction on log D rows (MET-08/MET-20 only)
NOT_SELECTABLE: dict[str, tuple[str, ...]] = {"logd": ("A_K1", "A_K2", "A_K3", "A_K4"),
                                              "logsf": ("B_K1", "B_K2", "B_K3", "B_K4", "C_K1", "C_K2", "C_K3", "C_K4")}
SIMPLE_BASELINE = {"logd": "BD_METAL", "logsf": "FLAT"}
SELECTED_ARMS: tuple[str, ...] = ("A", "C", "BD_COND", "B_selSF", "C_selSF")
SECONDARY_POPULATIONS: tuple[str, ...] = ("COH-D-C3M", "COH-D-ALL", "COH-SF-ANY", "COH-SF-C1", "COH-SF-NDSM",
                                          "COH-SC009-D", "COH-SC009-SF", "COH-NOSC009-D", "COH-NOSC009-SF",
                                          "COH-D-C3@NODMD", "COH-SF-C3@NODMD", "COH-REPCAP-D", "COH-REPCAP-SF")
IDENTITIES = {
    "I1": "R²_pooled = s_b·R²_level + s_w·R²_within_cell on any log D row set (centred residuals sum to 0 in each cell).",
    "I2": ("R²_within_cell(log D) equals the FLAT-referenced skill 1 − Σw·e²/Σw·y² of the derived log SF over all "
           "unordered within-cell pairs, pair weight 1/n_c: a shape score over all pairs, not adjacent pairs, and not "
           "a deployable log D accuracy."),
    "I3": "(log SF) 1 − R²_pooled = (1 − R²_0)(1 − R²_FLAT), R²_FLAT = −ȳ²/var0(y), y_rms0² = y_sd0² + ȳ².",
    "I4": "on a seed-invariant row set, seed-mean R² = 1 − mean_s(SSE_s)/SST.",
    "I5": "e_SF = r̃_a − r̃_b (r = y − ŷ_A, r̃ centred within the cell): A's cell-level residual cancels.",
}
TAB14_CONTEXT: tuple[dict, ...] = (
    {"value": "0.30991136191292135", "arm": "RIDGE2|k3 (CTX_RIDGE2_K3)", "setting": "metal model, k = 3, DOPT_Q, BP, C3, regime Q",
     "modification": "measured mode (4 read metals): context only, never a comparator or threshold reference",
     "sd_n": "y_sd0 0.22196; 1,802 pairs/seed", "source": "generations/gen20_kshot/results/discovery/levels_BP_metal_C3_DOPT_Q.csv:73"},
    {"value": "0.3209589052649169", "arm": "RIDGE2|k3", "setting": "gen20 confirmation seeds", "modification": "measured mode",
     "sd_n": "", "source": "generations/gen20_kshot/results/confirmation/levels_BP_metal_C3_DOPT_Q.csv:73"},
    {"value": "-0.13471207313225375", "arm": "FLAT@ZERO", "setting": "k = 0, COH-SF-C3, BP", "modification": "anchor",
     "sd_n": "RMSE 0.23644 (= y_rms0)", "source": "levels csv:6"},
    {"value": "0.05287231149042522", "arm": "G14@ZERO", "setting": "k = 0, COH-SF-C3, BP", "modification": "anchor",
     "sd_n": "RMSE 0.21601", "source": "levels csv:18"},
    {"value": "-0.11561984257982724", "arm": "MEAN_CURVE@ZERO", "setting": "k = 0, COH-SF-C3, BP", "modification": "anchor",
     "sd_n": "RMSE 0.23444", "source": "levels csv:30"},
    {"value": "-0.3217936240551599", "arm": "NNC@ZERO", "setting": "k = 0, COH-SF-C3, BP", "modification": "cited only (NNC never instantiated)",
     "sd_n": "RMSE 0.25514", "source": "levels csv:50"},
    {"value": "-0.3860763898872736", "arm": "NN_TANIMOTO@ZERO", "setting": "k = 0, COH-SF-C3, BP", "modification": "anchor",
     "sd_n": "RMSE 0.26128", "source": "levels csv:62"},
    {"value": "0.4293", "arm": "RIDGE2", "setting": "k = 3, C3", "modification": "Nd-Sm added (1,984 pairs, 236 cells)",
     "sd_n": "target sd 0.2523", "source": "GEN20_REPORT.md:184-185"},
    {"value": "0.42 / 0.38 / 0.36", "arm": "LINE Pr/Nd / LINE Sm/Eu / G14@BLUP20 Gd/Tb", "setting": "k = 3",
     "modification": "single pair type", "sd_n": "127 / 158 / 165 rows", "source": "GEN20_REPORT.md:238-241"},
    {"value": "0.66 / 0.63", "arm": "G14@BLUP20 / FLAT@BLUP20", "setting": "C10, k = 10",
     "modification": "pair measurement model with lab-known interpolation", "sd_n": "114 cells", "source": "GEN20_REPORT.md:297-299"},
    {"value": "0.471", "arm": "G14@BLUP20", "setting": "k = 5", "modification": "pair measurement model, C5",
     "sd_n": "", "source": "results/discovery/levels_BP_pair_C5_DOPT_Q.csv:16"},
    {"value": "0.416", "arm": "G14@BLUP20", "setting": "k = 6, C6, UNIFORM", "modification": "design B (no publication mask)",
     "sd_n": "", "source": "results/discovery/levels_B_metal_C6_UNIFORM.csv:19"},
    {"value": "+0.426", "arm": "RIDGE2 ΔR²(k = 3 − k = 0)", "setting": "", "modification": "a change in R², not an R²",
     "sd_n": "", "source": "GEN20_REPORT.md:154"},
    {"value": "0.401", "arm": "RIDGE2|k3 CI upper bound", "setting": "", "modification": "an interval bound, not attainment",
     "sd_n": "", "source": "levels csv:73"},
)


class ReportRefused(RuntimeError):
    """The report refuses this request (a smoke label, LT-29; an unsealed pre-registration)."""


def refuse_smoke(label: str) -> None:
    """LT-29: smoke runs are never reported."""
    if str(label).strip().lower().startswith("smoke"):
        raise ReportRefused(f"label {label!r}: smoke output is never reported (LT-29, IMPLEMENTATION_PLAN.md §12 RK18)")


# --------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------
def default_loader(label: str, design: str, variant: str):
    from . import fitting  # builder M's module; imported only when real tables are read
    return fitting.load_tables(label, design, variant)


def _read_done(p: Path) -> dict | None:
    """A DONE marker's record, or None when it does not parse to a JSON object (torn by a crash, being written)."""
    try:
        rec = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def default_status(label: str, design: str, variant: str, *, n_units: int = N_UNITS, root: Path | None = None) -> dict:
    """complete / incomplete / not run, from the unit DONE markers (the report uses complete designs only).  Only
    markers whose JSON parses count (the glob never matches the ``.tmp`` files of an interrupted write); any
    unreadable marker makes the design "incomplete"."""
    d = Path(root if root is not None else bootstrap.PREDICTIONS) / label / design / variant
    n = bad = 0
    if d.is_dir():
        for p in sorted(d.glob("s*_f*_DONE.json")):
            if _read_done(p) is None:
                bad += 1
            else:
                n += 1
    if bad:
        status = "incomplete"
    else:
        status = "complete" if n >= n_units else ("incomplete" if n else "not run")
    return {"status": status, "n_done": n, "n_units": n_units, "n_unreadable": bad}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest_str(cd) -> str | None:
    """A DONE marker's / RUN_ONCE record's code digest as its sha256 string (a dict carries it under 'sha256')."""
    if isinstance(cd, dict):
        cd = cd.get("sha256")
    return str(cd) if cd else None


def unit_digests(label: str, design: str, variant: str, *, root: Path | None = None) -> set[str]:
    """The code digests recorded by every ``s*_f*_DONE.json`` of (label, design, variant); a marker without one
    contributes "missing" and a marker that does not parse "unreadable" (the report then refuses: every scored unit
    must name the code that produced it)."""
    d = Path(root if root is not None else bootstrap.PREDICTIONS) / label / design / variant
    out: set[str] = set()
    if not d.is_dir():
        return out
    for p in sorted(d.glob("s*_f*_DONE.json")):
        rec = _read_done(p)
        out.add("unreadable" if rec is None else (_digest_str(rec.get("code_digest")) or "missing"))
    return out


def load_selections(label: str, design: str, variant: str) -> dict:
    """{(seed, fold): selection record} read from predictions/<label>/<design>/<variant>/s*_f*_selection.json.
    Only completed units count (a selection file without its DONE marker belongs to an interrupted unit that
    ``fitting.load_tables`` does not read either); each file must match its DONE marker's ``selection_sha256``."""
    d = bootstrap.PREDICTIONS / label / design / variant
    out = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("s*_f*_selection.json")):
        stem = p.name.split("_selection")[0]
        done = d / f"{stem}_DONE.json"
        rec = _read_done(done) if done.exists() else None
        if rec is None:                                    # no marker, or an unreadable one: an incomplete unit
            continue
        want = rec.get("selection_sha256")
        if want is not None and _file_sha256(p) != want:
            raise M.CoverageError(f"{p} differs from its DONE marker (selection_sha256)")
        s, f = stem.split("_")
        out[(int(s[1:]), int(f[1:]))] = json.loads(p.read_text(encoding="utf-8"))
    return out


#: S5 (scripts/g201_describe.py) writes the COH-REPCAP drop list ``COH-REPCAP.parquet`` and one filtered copy
#: ``COH-REPCAP@<parent>.parquet`` of every test-side S3 manifest; the scored COH-REPCAP populations (EX-10, TAB-09
#: (g)) are the primary parents' copies only.  The drop list and every other parent's copy are not populations.
REPCAP_POPULATIONS: dict[str, tuple[str, str]] = {"COH-REPCAP@COH-D-C3": ("COH-REPCAP-D", "logd"),
                                                  "COH-REPCAP@COH-SF-C3": ("COH-REPCAP-SF", "logsf")}


def load_row_manifests(path: Path = ROW_MANIFESTS) -> dict[str, pd.DataFrame]:
    """Test-side population keys (seed- and design-invariant): {name: unique key rows with identity columns}.
    COH-REPCAP (label-derived, S5): ``COH-REPCAP@COH-D-C3`` / ``COH-REPCAP@COH-SF-C3`` are read as COH-REPCAP-D /
    COH-REPCAP-SF (``REPCAP_POPULATIONS``); the drop list and the other parents' REPCAP copies are skipped."""
    from .data import SCORED_POPULATIONS
    path = Path(path)
    out = {}
    scored = set(SCORED_POPULATIONS) | {f"{n}@NODMD" for n in SCORED_POPULATIONS}
    for p in sorted(path.glob("COH-*.parquet")):
        name = p.stem
        if name.startswith("COH-REPCAP"):
            if name not in REPCAP_POPULATIONS:
                continue
            name, want = REPCAP_POPULATIONS[name]
        elif name not in scored:
            continue
        else:
            want = None
        t = pd.read_parquet(p)
        if "ci" not in t.columns or not ({"im"} <= set(t.columns) or {"ia", "ib"} <= set(t.columns)):
            continue
        target = "logsf" if "ia" in t.columns else "logd"
        if want is not None and target != want:
            raise M.CoverageError(f"{p.name}: a {target} manifest where {name} needs {want} keys")
        keys = list(M.keys_of(target))
        ident = [c for c in ("cell_id", "extractant", "chemotype", "publication_id", "metal", "a", "b", "pair")
                 if c in t.columns]
        u = t[keys + ident].drop_duplicates(subset=keys).sort_values(keys).reset_index(drop=True)
        out[name] = u
    return out


def select_population(table: pd.DataFrame, keys: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """The rows of ``table`` whose key is in the population; every key must be present once in every seed and
    its identity columns must agree with the manifest (else CoverageError: the run fails)."""
    target = M.target_of(table)
    k = list(M.keys_of(target))
    if M.target_of(keys) != target:
        raise M.CoverageError(f"{name}: manifest target differs from the table's")
    sel = table.merge(keys[k], on=k, how="inner")
    for s, g in sel.groupby("split_seed"):
        if len(g) != len(keys) or g[k].duplicated().any():
            raise M.CoverageError(f"{name}: seed {s} has {len(g)} rows for {len(keys)} manifest keys")
    if sel["split_seed"].nunique() != table["split_seed"].nunique():
        raise M.CoverageError(f"{name}: a split seed has no row of the population")
    chk = sel[k + [c for c in ("chemotype", "extractant", "publication_id") if c in sel.columns]].drop_duplicates(k)
    ref = keys[k + [c for c in ("chemotype", "extractant", "publication_id") if c in keys.columns]]
    mm = chk.merge(ref, on=k, suffixes=("", "_m"))
    for c in ("chemotype", "extractant", "publication_id"):
        if c in chk.columns and f"{c}_m" in mm.columns and (mm[c].astype(str) != mm[f"{c}_m"].astype(str)).any():
            raise M.CoverageError(f"{name}: identity column {c} differs from the row manifest")
    return sel.reset_index(drop=True)


def _chosen(entry) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for k in ("chosen", "choice", "selected", "chosen_K", "chosen_config", "config", "K", "arm"):
            v = entry.get(k)
            if isinstance(v, str):
                return v
    return None


def _inner_scores(entry) -> dict | None:
    if not isinstance(entry, dict):
        return None
    for k in ("inner_scores", "scores", "inner_score", "criterion"):
        v = entry.get(k)
        if isinstance(v, dict):
            return {str(a): float(b) for a, b in v.items() if b is not None}
        if isinstance(v, list) and len(v) == 4:
            return {f"K{i + 1}": (float(b) if b is not None else float("nan")) for i, b in enumerate(v)}
    return None


def add_inner_choice(table: pd.DataFrame, selections: dict, name: str, members) -> pd.DataFrame:
    """Compose BL_D_INNER / BL_SF_INNER per (seed, fold) from the selection record's choice among ``members``
    (an EX-11 reference, never a selected arm); left absent when a record or member column is missing."""
    if not selections or not all(m in table.columns for m in members):
        return table
    col = np.full(len(table), np.nan)
    ss, ff = table["split_seed"].to_numpy(), table["fold"].to_numpy()
    for (s, f), rec in selections.items():
        ch = _chosen(rec.get(name)) if isinstance(rec, dict) else None
        if ch not in members:
            return table
        m = (ss == s) & (ff == f)
        col[m] = table.loc[m, ch].to_numpy(dtype=float)
    if not np.isfinite(col).all():
        return table
    out = table.copy()
    out[name] = col
    return out


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _clean(x):
    """JSON-safe: NaN/inf -> None, numpy -> python, recursively."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    return x


def _f(x, nd: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "NA" if math.isnan(v) else ("+inf" if v > 0 else "-inf")
    return f"{v:.{nd}f}"


def _variant_of(pop: str) -> str:
    return "nodmd" if pop.endswith("@NODMD") else "primary"


# --------------------------------------------------------------------------------------
# the scorer
# --------------------------------------------------------------------------------------
class Scorer:
    """Scores one label from saved tables.  ``loader(label, design, variant) -> (logd, logsf)``; ``status(label,
    design, variant) -> {'status': 'complete'|...}``; ``manifests`` {population: keys}; ``selections(label, design,
    variant) -> {(seed, fold): record}``; ``digests(label, design, variant) -> set of code digests`` (default
    ``unit_digests`` when the tables come from ``fitting.load_tables``; with an injected loader and no
    ``digests`` the check is skipped and ``code_digest`` is None)."""

    def __init__(self, label: str, *, loader=None, status=None, manifests=None, selections=None, digests=None,
                 designs=DESIGNS, variants=VARIANTS, replicates: int = I.REPLICATES, log=print):
        refuse_smoke(label)
        self.label = label
        self.loader = loader or default_loader
        self._status_fn = status or default_status
        self.manifests = load_row_manifests() if manifests is None else manifests
        self._sel_fn = selections or load_selections
        self._digest_fn = digests if digests is not None else (unit_digests if loader is None else None)
        self.designs, self.variants = tuple(designs), tuple(variants)
        self.replicates = int(replicates)
        self.log = log
        self._tables: dict = {}
        self._frames: dict = {}
        self._status = {d: {v: self._status_fn(label, d, v) for v in self.variants} for d in self.designs}
        self.code_digest, self.code_digest_by_design = self._check_digests()

    def _check_digests(self) -> tuple[str | None, dict]:
        """``fitting.load_tables`` enforces one code digest within a (label, design, variant) only; every complete
        (design, variant) the report combines (design points, TAB-08, COH-NODMD) must share one digest, else the
        run fails (CoverageError)."""
        if self._digest_fn is None:
            return None, {}
        per = {f"{d}/{v}": sorted(self._digest_fn(self.label, d, v))
               for d in self.designs for v in self.variants if self.complete(d, v)}
        found = set().union(*map(set, per.values())) if per else set()
        if not per:
            return None, per
        if len(found) != 1 or "missing" in found or "unreadable" in found or None in found:
            raise M.CoverageError(f"label {self.label!r}: the complete designs were produced by different or "
                                  f"unrecorded code versions: {per}")
        return next(iter(found)), per

    # -- data -------------------------------------------------------------------------
    def complete(self, design: str, variant: str = "primary") -> bool:
        return self._status.get(design, {}).get(variant, {}).get("status") == "complete"

    def tables(self, design: str, variant: str):
        key = (design, variant)
        if key not in self._tables:
            if not self.complete(design, variant):
                self._tables[key] = None
            else:
                logd, logsf = self.loader(self.label, design, variant)
                sel = self._sel_fn(self.label, design, variant)
                logd = add_inner_choice(logd, sel, "BL_D_INNER", R.H1_COMPARATORS)
                logsf = add_inner_choice(logsf, sel, "BL_SF_INNER", ("FLAT", "PAIRMEAN"))
                self._tables[key] = (logd, logsf)
        return self._tables[key]

    def pop_table(self, pop: str, design: str = "BP") -> pd.DataFrame | None:
        if pop not in self.manifests:
            return None
        tb = self.tables(design, _variant_of(pop))
        if tb is None:
            return None
        keys = self.manifests[pop]
        t = tb[1] if M.target_of(keys) == "logsf" else tb[0]
        return select_population(t, keys, f"{pop}/{design}")

    def frame(self, pop: str, design: str = "BP") -> I.Frame | None:
        key = (pop, design)
        if key not in self._frames:
            t = self.pop_table(pop, design)
            self._frames[key] = None if t is None else I.Frame(t, name=f"{pop}/{design}/{_variant_of(pop)}")
        return self._frames[key]

    def summary(self, pop: str, arm: str, design: str = "BP") -> dict | None:
        F = self.frame(pop, design)
        if F is None or not F.has(arm):
            return None
        return I._direct(F, arm)

    # -- registered claims ---------------------------------------------------------------
    def design_points(self, pop: str, X: str, Y: str) -> dict:
        out = {}
        for d in R.OTHER_DESIGNS:
            if d not in self.designs or not self.complete(d, _variant_of(pop)):
                out[d] = None
                continue
            F = self.frame(pop, d)
            out[d] = (I.point(F, "MET01", X, Y, contrast=True)["point"]
                      if F is not None and F.has(X) and F.has(Y) else float("nan"))
        return out

    def claim(self, hid: str, pop: str, *, h_from: dict | None = None) -> dict:
        cand, target, comps = R.CLAIMS[hid]
        F = self.frame(pop)
        if F is None or not F.has(cand):
            v = R.verdict_h([], candidate_run=False)
            v.update({"hid": hid, "population": pop, "evidence": []})
            return v
        evs, comp_res = [], []
        for Y in comps:
            if not F.has(Y):
                raise M.CoverageError(f"{pop}: comparator {Y} of {hid} missing from the saved tables")
            ev = I.contrast_evidence(F, cand, Y, replicates=self.replicates)
            ev["design_points"] = (h_from or {}).get((hid, Y)) if h_from is not None else self.design_points(pop, cand, Y)
            evs.append(ev)
            comp_res.append(R.rule_c(ev))
        v = R.verdict_h(comp_res)
        v.update({"hid": hid, "population": pop, "evidence": evs})
        return v

    def claim_h4(self, pop: str, *, h_from: dict | None = None) -> dict:
        F = self.frame(pop)
        if F is None or not (F.has("C") and F.has("B")):
            v = R.rule_h4(None, None, c_run=False)
            v.update({"hid": "H4", "population": pop, "evidence": []})
            return v
        e1 = I.contrast_evidence(F, "C", "B", replicates=self.replicates)
        e2 = I.contrast_evidence(F, "B", "C", replicates=self.replicates)
        for ev, (X, Y) in ((e1, ("C", "B")), (e2, ("B", "C"))):
            ev["design_points"] = (h_from or {}).get(("H4", Y)) if h_from is not None else self.design_points(pop, X, Y)
        v = R.rule_h4(e1, e2)
        v.update({"hid": "H4", "population": pop, "evidence": [e1, e2]})
        return v

    def registered(self, *, nodmd: bool = False, h_from: dict | None = None) -> dict:
        sfx = "@NODMD" if nodmd else ""
        out = {"H1": self.claim("H1", "COH-D-C3" + sfx, h_from=h_from),
               "H2": self.claim("H2", "COH-SF-C3" + sfx, h_from=h_from),
               "H3": self.claim("H3", "COH-SF-C3" + sfx, h_from=h_from),
               "H4": self.claim_h4("COH-SF-C3" + sfx, h_from=h_from)}
        return out

    def nc1(self) -> dict:
        F = self.frame("COH-D-C3")
        if F is None or not (F.has("A") and F.has("A_PRIME")):
            return {"flag": False, "status": "A_PRIME not run", "p": [float("nan"), float("nan")]}
        e1 = I.contrast_evidence(F, "A", "A_PRIME", replicates=self.replicates, full=False)
        e2 = I.contrast_evidence(F, "A_PRIME", "A", replicates=self.replicates, full=False)
        res = R.nc1(e1, e2)
        res["status"] = "run"
        res["p"] = [e1.get("p_wcr"), e2.get("p_wcr")]
        res["evidence"] = [e1, e2]
        return res

    def prnd(self, X: str) -> dict:
        t = self.pop_table("COH-SF-C3")
        if t is None or X not in t.columns:
            return {"status": "not verified", "reason": "not run"}
        blk = t[t["pair"].astype(str) == "Pr/Nd"]
        if not len(blk):
            return {"status": "not verified", "reason": "no Pr/Nd rows"}
        F = I.Frame(blk, name="COH-SF-C3[Pr/Nd]")
        ub, pts = {}, {}
        for Y in R.SF_COMPARATORS:
            inf = I.infer(F, "MET01", X, Y, contrast=True, units=("chemotype",), replicates=self.replicates)
            ub[Y] = -inf["chem_pct9875_low"] if np.isfinite(inf["chem_pct9875_low"]) else (
                float("inf") if inf["chem_pct9875_low"] == -np.inf else float("nan"))
            pts[Y] = -inf["point"]
        return {"status": R.prnd_status(ub, pts), "upper_bound_X_minus_Y": ub, "point_X_minus_Y": pts}

    # -- thresholds --------------------------------------------------------------------
    def threshold(self, tid: str, arm: str, pop: str, stat: str) -> dict:
        F = self.frame(pop)
        if F is None or not F.has(arm):
            return {"id": tid, "arm": arm, "population": pop, "stat": stat,
                    "classes": {str(t): {"class": "NA", "reason": "not run"} for t in R.TAUS}}
        ev = I.threshold_evidence(F, arm, stat=stat, replicates=self.replicates)
        out = {"id": tid, "arm": arm, "population": pop, "stat": stat, "evidence": ev,
               "classes": {str(t): R.rule_t(ev, t) for t in R.TAUS}}
        target = F.target
        sfx_pop = {"logd": ("COH-SC009-D", "COH-NOSC009-D", "COH-D-C3@NODMD"),
                   "logsf": ("COH-SC009-SF", "COH-NOSC009-SF", "COH-SF-C3@NODMD")}[target]
        ann = {"chem_pct9875_low": ev.get("chem_pct9875_low"), "pub_pct9875_low": ev.get("pub_pct9875_low"),
               "loco_min": ev.get("loco_min"), "se_boot": ev.get("chem_se")}
        ann["tau_plus_1p96se"] = {str(t): t + 1.96 * ev["chem_se"] if np.isfinite(ev.get("chem_se", np.nan)) else None
                                  for t in R.TAUS}
        Fs = self.frame(sfx_pop[0])
        ann["sc009_point"] = I.point(Fs, stat, arm)["point"] if Fs is not None and Fs.has(arm) else None
        Fn = self.frame(sfx_pop[1])
        if Fn is not None and Fn.has(arm):
            inf = I.infer(Fn, stat, arm, replicates=self.replicates)
            ann.update({"nosc009_point": inf["point"], "nosc009_chem_pct95_low": inf.get("chem_pct95_low"),
                        "nosc009_chem_pct95_high": inf.get("chem_pct95_high"),
                        "nosc009_pub_pct95_low": inf.get("pub_pct95_low"),
                        "nosc009_pub_pct95_high": inf.get("pub_pct95_high")})
        Fd = self.frame(sfx_pop[2])
        ann["nodmd_point"] = I.point(Fd, stat, arm)["point"] if Fd is not None and Fd.has(arm) else None
        s = I._direct(F, arm)
        ann["context"] = self.context(pop, arm)
        if tid in ("T1", "T1w"):
            ann["I1"] = {"s_b": s["s_b"], "s_w": s["s_w"], "R2_level": s["R2_level"],
                         "R2_within_cell": s["R2_within_cell"], "R2_pooled": s["R2_pooled"]}
        if target == "logsf":
            flat = self.summary(pop, "FLAT")
            r2flat = flat["R2_pooled"] if flat else float("nan")
            ann["R2_0"] = s["R2_0"]
            ann["R2_FLAT"] = r2flat
            ann["I3_R2_0_equivalent"] = {str(t): (1.0 - (1.0 - t) / (1.0 - r2flat)) if np.isfinite(r2flat) else None
                                         for t in R.TAUS}
        out["annotations"] = ann
        return out

    def context(self, pop: str, arm: str) -> dict:
        """The §5 context printed beside every R² or RULE-T class: n, cells, y_sd0, y_var0, the arm's MET-01/02/05
        and the simple baseline's R², MAE and RMSE on the same rows."""
        F = self.frame(pop)
        if F is None:
            return {}
        c = M.card(F.rows.assign(split_seed=F.seeds[0]))
        s = I._direct(F, arm) if F.has(arm) else None
        simple = SIMPLE_BASELINE[F.target]
        b = I._direct(F, simple) if F.has(simple) else None
        return {"n_rows": c["n_rows"], "n_cells": c["n_cells"], "y_sd0": c["y_sd0"], "y_var0": c["y_var0"],
                "MET01": s["MET01_MAE_macro_ext"] if s else None, "MET02": s["MET02_MAE_pooled"] if s else None,
                "MET05": s["MET05_RMSE_pooled"] if s else None, "simple_baseline": simple,
                "simple_R2": b["R2_pooled"] if b else None, "simple_R2_reason": b["R2_pooled_reason"] if b else None,
                "simple_MAE": b["MET02_MAE_pooled"] if b else None,
                "simple_MAE_macro_ext": b["MET01_MAE_macro_ext"] if b else None,
                "simple_RMSE": b["MET05_RMSE_pooled"] if b else None}

    def thresholds(self) -> dict:
        out = {"T1": self.threshold("T1", "A", "COH-D-C3", "MET06"),
               "T1w": self.threshold("T1w", "A", "COH-D-C3", "MET08"),
               "T2": self.threshold("T2", "B", "COH-SF-C3", "MET06"),
               "T3": self.threshold("T3", "C", "COH-SF-C3", "MET06")}
        # the same class for every baseline, reference and exploratory arm on the same rows
        others = {}
        for pop, arms, stats in (("COH-D-C3", LOGD_TABLE_ARMS, ("MET06", "MET08")),
                                 ("COH-SF-C3", LOGSF_TABLE_ARMS, ("MET06",))):
            F = self.frame(pop)
            if F is None:
                continue
            for arm in arms:
                if not F.has(arm):
                    continue
                for st in stats:
                    others[f"{pop}|{arm}|{st}"] = self.threshold(f"{arm}:{st}", arm, pop, st)
        # T1 annotations that need other arms' classes
        t1 = out["T1"]
        for tau in map(str, R.TAUS):
            if "evidence" not in t1:
                continue
            reached_by = [y for y in R.H1_COMPARATORS
                          if others.get(f"COH-D-C3|{y}|MET06", {}).get("classes", {}).get(tau, {}).get("class") == "REACHED"]
            lvl = others.get("COH-D-C3|BD_LEVEL|MET06", {}).get("classes", {}).get(tau, {}).get("class", "NA")
            sA = self.summary("COH-D-C3", "A")
            sC = self.summary("COH-D-C3", "BD_LEVEL_CURVE")
            la = R.level_attributable(t1["classes"][tau]["class"], lvl, sA["R2_within_cell"] if sA else None,
                                      sC["R2_within_cell"] if sC else None)
            t1["classes"][tau]["reached_also_by"] = [R.REACHED_ALSO.format(baseline=y) for y in reached_by]
            t1["classes"][tau]["level_attributable"] = la
        out["_others"] = others
        return out

    # -- exploratory family ------------------------------------------------------------
    def exploratory(self, nc1: dict | None = None) -> list[dict]:
        rows = []
        for spec in R.exploratory_family():
            row = dict(spec)
            if spec["kind"] == "spearman":
                Fd, Fs = self.frame("COH-D-C3"), self.frame("COH-SF-C3")
                a_c, a_r = spec["candidate"].split(">")
                b_c, b_r = spec["comparator"].split(">")
                if Fd is None or Fs is None or not all(F.has(x) for F, x in ((Fd, a_c), (Fd, a_r), (Fs, b_c), (Fs, b_r))):
                    row.update({"status": "not run", "p": float("nan")})
                else:
                    res = I.spearman_gains(Fd, a_c, a_r, Fs, b_c, b_r)
                    row.update(res)
                    row.update({"status": "run" if res["readable"] else "not readable (split-half reliability < 0.3)",
                                "p": res["p_for_bh"], "delta": res["rho"]})
            else:
                F = self.frame(spec["population"])
                if F is None or not (F.has(spec["candidate"]) and F.has(spec["comparator"])):
                    row.update({"status": "not run", "p": float("nan")})
                else:
                    ev = I.contrast_evidence(F, spec["candidate"], spec["comparator"], stat=spec["metric"],
                                             replicates=self.replicates, full=False)
                    row.update({"status": "run", "p": ev["p_wcr"], "delta": ev["delta"],
                                "chem_pct9875_low": ev.get("chem_pct9875_low"),
                                "chem_pct9875_high": ev.get("chem_pct9875_high"),
                                "pub_pct9875_low": ev.get("pub_pct9875_low"),
                                "pub_pct9875_high": ev.get("pub_pct9875_high"),
                                "chem_pct95_low": ev.get("chem_pct95_low"), "chem_pct95_high": ev.get("chem_pct95_high"),
                                "p_two_sided": ev.get("chem_p_two_sided"), "seeds_positive": ev.get("seeds_positive"),
                                "n_rows": ev.get("n_rows")})
            rows.append(row)
        b = R.bh([r.get("p") for r in rows])
        for r, pu, q, rej in zip(rows, b["p_used"], b["q"], b["reject"]):
            r.update({"p_used_in_bh": float(pu), "q_bh": float(q), "reject_bh_q05": bool(rej)})
        return rows

    # -- tables ------------------------------------------------------------------------
    def arm_rows(self, pop: str, arms, *, design: str = "BP", intervals: bool = True) -> list[dict]:
        F = self.frame(pop, design)
        if F is None:
            return []
        card = M.card(F.rows.assign(split_seed=F.seeds[0]))
        target = F.target
        named = R.H1_COMPARATORS if target == "logd" else R.SF_COMPARATORS
        rows = []
        extra = []
        if target == "logsf" and F.has("HEAVIER_ALWAYS"):
            s = I._direct(F, "HEAVIER_ALWAYS")
            row = {"population": pop, "design": design, "arm": "HEAVIER_ALWAYS", "status": "run (MET-14 only)"}
            row.update({k: v for k, v in s.items() if k.startswith("MET14_") and not k.endswith("_per_seed")})
            if F.has("PAIRMEAN"):
                sp = I._direct(F, "PAIRMEAN")
                row.update({f"PAIRMEAN_{k}": v for k, v in sp.items() if k.startswith("MET14_dir_acc")
                            and not k.endswith("_per_seed")})
            extra.append(row)
        if target == "logd":
            for arm in LOGD_WITHIN_ONLY:
                if F.has(arm):
                    s = I._direct(F, arm)
                    extra.append({"population": pop, "design": design, "arm": arm,
                                  "status": "run (centred prediction: within-cell metrics only)",
                                  "R2_within_cell": s["R2_within_cell"], "R2_within_cell_reason": s["R2_within_cell_reason"],
                                  "R2_balanced_within": s["R2_balanced_within"],
                                  "MET20_MAE_macro_ext_within": s["MET20_MAE_macro_ext_within"]})
        for arm in arms:
            if not F.has(arm):
                rows.append({"population": pop, "design": design, "arm": arm, "status": "not run"})
                continue
            s = I._direct(F, arm)
            row = {"population": pop, "design": design, "variant": _variant_of(pop), "arm": arm, "status": "run",
                   "mode": "Z", "regime": "k = 0", "averaging": "5-seed mean"}
            row.update({f"card_{k}": v for k, v in card.items()})
            keys = ["MET01_MAE_macro_ext", "MET02_MAE_pooled", "MET03_MAE_macro_chem", "MET04_MAE_bal", "MET04_RMSE_bal",
                    "MET04_Bias_bal", "MET05_RMSE_pooled", "R2_pooled", "R2_balanced", "R2_balanced_within",
                    "R2_within_cell", "R2_level", "s_b", "s_w", "MET12_EV_abs", "MET15_within_1p5x", "MET15_within_2x",
                    "MET15_within_1p5x_bal", "MET15_within_2x_bal", "MET16_bias", "MET16_sd_ratio",
                    "MET16_within_dispersion", "MET16_ols_slope", "MET20_MAE_macro_ext_within", "R2_within_system"]
            if target == "logsf":
                keys += ["R2_0"] + [k for k in s if k.startswith("MET14_") and not k.endswith("_per_seed")]
            for k in keys:
                row[k] = s.get(k)
                if f"{k}_reason" in s:
                    row[f"{k}_reason"] = s[f"{k}_reason"]
                    row[f"{k}_n_seeds_defined"] = s[f"{k}_n_seeds_defined"]
            row["MET01_per_seed"] = ";".join(_f(v, 6) for v in s["MET01_MAE_macro_ext_per_seed"])
            row["R2_pooled_per_seed"] = ";".join(_f(v, 6) for v in s["R2_pooled_per_seed"])
            for y in named:
                if F.has(y) and y != arm:
                    sk = I.point(F, "MET11", arm, y)
                    row[f"Skill_MSE_vs_{y}"] = sk["point"]
            if intervals:
                inf = I.infer(F, "MET01", arm, replicates=self.replicates)
                for k in ("chem_pct95_low", "chem_pct95_high", "chem_pct9875_low", "chem_pct9875_high",
                          "pub_pct95_low", "pub_pct95_high", "pub_pct9875_low", "pub_pct9875_high",
                          "chem_status", "pub_status", "chem_preliminary", "pub_preliminary",
                          "chem_n_clusters", "pub_n_clusters"):
                    row[f"MET01_{k}"] = inf.get(k)
            rows.append(row)
        return rows + extra

    def design_table(self) -> list[dict]:
        rows = []
        for d in self.designs:
            for pop, arms, stats in (("COH-D-C3", ("A",) + R.H1_COMPARATORS, ("MET01", "MET06", "MET08")),
                                     ("COH-SF-C3", ("B", "C") + R.SF_COMPARATORS, ("MET01", "MET06", "MET10"))):
                F = self.frame(pop, d)
                if F is None:
                    rows.append({"design": d, "population": pop, "status": self._status[d]["primary"]["status"]})
                    continue
                ctx_simple = SIMPLE_BASELINE[F.target]
                b = I._direct(F, ctx_simple) if F.has(ctx_simple) else None
                card = M.card(F.rows.assign(split_seed=F.seeds[0]))
                for arm in arms:
                    if not F.has(arm):
                        continue
                    row = {"design": d, "population": pop, "arm": arm, "status": "complete", "n_rows": card["n_rows"],
                           "n_cells": card["n_cells"], "y_sd0": card["y_sd0"], "y_var0": card["y_var0"]}
                    s = I._direct(F, arm)
                    for st in stats:
                        row[st] = I.point(F, st, arm)["point"]
                        key = I.DIRECT_KEY[st]
                        if f"{key}_reason" in s:            # every R² carries R2_reason and n_seeds_defined (§5)
                            row[f"{st}_reason"] = s[f"{key}_reason"]
                            row[f"{st}_n_seeds_defined"] = s[f"{key}_n_seeds_defined"]
                    row.update({"MET02": s["MET02_MAE_pooled"], "MET05": s["MET05_RMSE_pooled"],
                                "simple_baseline": ctx_simple,
                                "simple_R2": b["R2_pooled"] if b else None,
                                "simple_R2_reason": b["R2_pooled_reason"] if b else None,
                                "simple_R2_n_seeds_defined": b["R2_pooled_n_seeds_defined"] if b else None,
                                "simple_MAE": b["MET02_MAE_pooled"] if b else None,
                                "simple_RMSE": b["MET05_RMSE_pooled"] if b else None})
                    rows.append(row)
                # sign of each claim's point Δ
                for hid in ("H1",) if F.target == "logd" else ("H2", "H3", "H4"):
                    if hid == "H4":
                        pairs = [("C", "B")]
                    else:
                        cand, _, comps = R.CLAIMS[hid]
                        pairs = [(cand, y) for y in comps]
                    for X, Y in pairs:
                        if F.has(X) and F.has(Y):
                            dlt = I.point(F, "MET01", X, Y, contrast=True)["point"]
                            rows.append({"design": d, "population": pop, "claim": hid, "X": X, "Y": Y,
                                         "delta_MET01": dlt, "sign": int(np.sign(dlt))})
        # publication-leak price: R2_pooled(A) under B minus under BP
        FB, FBP = self.frame("COH-D-C3", "B"), self.frame("COH-D-C3", "BP")
        if FB is not None and FBP is not None and FB.has("A") and FBP.has("A"):
            rows.append({"design": "B-BP", "population": "COH-D-C3", "arm": "A",
                         "publication_leak_price_R2_pooled": I.point(FB, "MET06", "A")["point"]
                         - I.point(FBP, "MET06", "A")["point"]})
        return rows

    def secondary_rows(self) -> list[dict]:
        rows = []
        for pop in SECONDARY_POPULATIONS:
            if pop not in self.manifests:
                rows.append({"population": pop, "status": "manifest absent"})
                continue
            F = self.frame(pop)
            if F is None:
                rows.append({"population": pop, "status": "not run"})
                continue
            arms = LOGD_TABLE_ARMS if F.target == "logd" else LOGSF_TABLE_ARMS
            # TAB-09: the TAB-03 / TAB-04 columns, identical for all arms, MET-01 intervals included (COH-SC009:
            # chemotype interval NA "one cluster", publication interval over its publications; PRE_REGISTRATION §6)
            rows += self.arm_rows(pop, arms, intervals=True)
        return rows

    def type_rows(self) -> tuple[list[dict], list[dict]]:
        rows, prnd = [], []
        for pop, arms in (("COH-D-C3", ("A", "BD_METAL", "BD_COND", "BD_NN", "BD_MEAN")),
                          ("COH-SF-C3", ("B", "C", "FLAT", "PAIRMEAN", "G14_ZERO"))):
            t = self.pop_table(pop)
            if t is None:
                continue
            simple = "BD_METAL" if M.target_of(t) == "logd" else None
            for arm in arms:
                if not M.has_arm(t, arm):
                    continue
                pt = M.per_type(t, arm, simple)
                pt.insert(0, "arm", arm)
                pt.insert(0, "population", pop)
                rows += pt.to_dict("records")
                if M.target_of(t) == "logsf":
                    prnd.append({"population": pop, "arm": arm, **M.met24(t, arm)})
        return rows, prnd

    def selection_rows(self) -> tuple[list[dict], list[dict]]:
        """TAB-11 (per design, variant, seed, fold: sizes, inner sizes, skipped inner folds, selected K and inner
        scores) and MET-22 (selection frequency over the 25 BP primary selections, skipped inner folds)."""
        rows, freq = [], {}
        bp_units, bp_skipped_units, bp_skipped = 0, 0, []
        for d in self.designs:
            for v in self.variants:
                if not self.complete(d, v):
                    continue
                for (s, f), rec in sorted(self._sel_fn(self.label, d, v).items()):
                    row = {"design": d, "variant": v, "split_seed": s, "fold": f}
                    if isinstance(rec, dict):
                        for arm in SELECTED_ARMS + ("BL_D_INNER", "BL_SF_INNER"):
                            e = rec.get(arm)
                            ch = _chosen(e)
                            row[f"{arm}_chosen"] = ch
                            sc = _inner_scores(e)
                            if sc:
                                vals = sorted(x for x in sc.values() if np.isfinite(x))
                                row[f"{arm}_inner_scores"] = json.dumps(sc)
                                row[f"{arm}_gap_best_second"] = vals[1] - vals[0] if len(vals) > 1 else None
                            if isinstance(e, dict) and arm in SELECTED_ARMS:
                                row[f"{arm}_n_inner_val_cells"] = e.get("n_inner_val_cells")
                                row[f"{arm}_n_inner_rows"] = e.get("n_inner_rows")
                                if isinstance(e.get("skipped_inner_folds"), list):
                                    row[f"{arm}_skipped_inner_folds"] = ";".join(map(str, e["skipped_inner_folds"]))
                            if d == "BP" and v == "primary" and ch is not None and arm in SELECTED_ARMS:
                                freq.setdefault(arm, {}).setdefault(ch, 0)
                                freq[arm][ch] += 1
                        for k, val in rec.items():
                            if k not in SELECTED_ARMS + ("BL_D_INNER", "BL_SF_INNER") and isinstance(val, (int, float, str, bool)):
                                row[f"rec_{k}"] = val
                        dmd = rec.get("dmd_removed")
                        if isinstance(dmd, dict):
                            for k, val in dmd.items():
                                row[f"rec_dmd_removed_{k}"] = val
                        # inner sizes and skipped inner folds (plan TAB-11, MET-22; PRE_REGISTRATION §2, O12)
                        for inn in rec.get("inner") or []:
                            if not isinstance(inn, dict):
                                continue
                            j = inn.get("j")
                            for k in ("n_train", "n_train_unmasked", "n_val", "skipped"):
                                row[f"inner{j}_{k}"] = inn.get(k)
                        skipped = rec.get("skipped_inner_folds")
                        skipped = [int(j) for j in skipped] if isinstance(skipped, list) else []
                        row["skipped_inner_folds"] = ";".join(map(str, skipped))
                        row["n_skipped_inner_folds"] = len(skipped)
                        if d == "BP" and v == "primary":
                            bp_units += 1
                            if skipped:
                                bp_skipped_units += 1
                                bp_skipped += [f"({s}, {f}, {j})" for j in skipped]
                        fb = rec.get("fallbacks")
                        if isinstance(fb, dict):
                            for k, val in fb.items():
                                if isinstance(val, (int, float)) and not isinstance(val, bool):
                                    row[f"fallback_{k}"] = val
                    rows.append(row)
        met22 = [{"arm": a, "config": k, "selected_in_BP": n} for a, dd in sorted(freq.items()) for k, n in sorted(dd.items())]
        met22.append({"arm": "ALL", "config": "skipped inner folds", "BP_units": bp_units,
                      "BP_units_with_skipped_inner_fold": bp_skipped_units,
                      "skipped_seed_fold_j": "; ".join(bp_skipped)})
        return rows, met22

    @staticmethod
    def fallback_summary(sel_rows: list[dict]) -> list[dict]:
        """Baseline fallback counts per (design, variant) over all units (printed; > 50 % of rows labelled)."""
        df = pd.DataFrame(sel_rows)
        out = []
        if df.empty:
            return out
        denom = {"BD_METAL_rows": "rec_n_logd_rows", "BD_NN_rows_any": "rec_n_logd_rows",
                 "BD_NN_rows_all": "rec_n_logd_rows", "PAIRMEAN_rows": "rec_n_logsf_rows"}
        for (d, v), g in df.groupby(["design", "variant"]):
            for k, den in denom.items():
                col = f"fallback_{k}"
                if col in g and den in g:
                    num, tot = float(g[col].sum()), float(g[den].sum())
                    frac = num / tot if tot else float("nan")
                    out.append({"design": d, "variant": v, "fallback": k, "rows": num, "of_rows": tot, "fraction": frac,
                                "label": "more than 50 % fallbacks" if frac > 0.5 else ""})
        return out

    def not_selectable_rows(self) -> list[dict]:
        rows = []
        for pop, arms in (("COH-D-C3", NOT_SELECTABLE["logd"]), ("COH-SF-C3", NOT_SELECTABLE["logsf"])):
            F = self.frame(pop)
            if F is None:
                continue
            ctx_arm = SIMPLE_BASELINE[F.target]
            b = I._direct(F, ctx_arm) if F.has(ctx_arm) else None
            card = M.card(F.rows.assign(split_seed=F.seeds[0]))
            for arm in arms:
                if not F.has(arm):
                    continue
                s = I._direct(F, arm)
                st2, key2 = ("MET08", "R2_within_cell") if F.target == "logd" else ("MET10", "R2_0")
                rows.append({"population": pop, "arm": arm, "note": "not selectable (TAB-13)",
                             "MET01": s["MET01_MAE_macro_ext"], "MET02": s["MET02_MAE_pooled"],
                             "MET05": s["MET05_RMSE_pooled"], "MET06": s["R2_pooled"],
                             "MET06_reason": s["R2_pooled_reason"],
                             "MET06_n_seeds_defined": s["R2_pooled_n_seeds_defined"],
                             st2: s[key2], f"{st2}_reason": s[f"{key2}_reason"],
                             f"{st2}_n_seeds_defined": s[f"{key2}_n_seeds_defined"],
                             "n_rows": card["n_rows"], "n_cells": card["n_cells"], "y_sd0": card["y_sd0"],
                             "y_var0": card["y_var0"], "simple_baseline": ctx_arm,
                             "simple_R2": b["R2_pooled"] if b else None,
                             "simple_R2_reason": b["R2_pooled_reason"] if b else None,
                             "simple_R2_n_seeds_defined": b["R2_pooled_n_seeds_defined"] if b else None,
                             "simple_MAE": b["MET02_MAE_pooled"] if b else None,
                             "simple_RMSE": b["MET05_RMSE_pooled"] if b else None})
        return rows

    def target_rows(self) -> list[dict]:
        """TAB-02 (part): MET-13 on every scored population (design BP), per-pair-type y_sd0 for log SF; the S5
        noise records (MET-19: BP summary over the 25 training folds, per-fold records, log SF status) and the
        COH-REPCAP counts are copied from manifests/describe_g201.json (scripts/g201_describe.py) when it exists."""
        rows = []
        for pop in sorted(self.manifests):
            F = self.frame(pop)
            if F is None:
                continue
            c = M.card(F.rows.assign(split_seed=F.seeds[0]))
            rows.append({"population": pop, "variant": _variant_of(pop), **c})
            if F.target == "logsf":
                for typ, blk in F.rows.groupby("pair"):
                    rows.append({"population": pop, "pair_type": typ, "n_rows": int(len(blk)),
                                 "y_sd0": float(blk["y"].std(ddof=0))})
        s5 = bootstrap.MANIFESTS / S5_DESCRIBE_JSON
        if s5.exists():
            rec = json.loads(s5.read_text(encoding="utf-8"))
            for k, v in (rec.get("noise_logd_summary_BP") or {}).items():
                rows.append({"population": "MET-19 log D noise over the BP training folds (S5)", "statistic": k,
                             **(v if isinstance(v, dict) else {"value": v})})
            for r in (rec.get("noise_logd") or {}).get("BP", []):
                rows.append({"population": "MET-19 log D noise per BP training fold (S5)", **r})
            nl = rec.get("noise_logsf") or {}
            if nl:
                rows.append({"population": "MET-19 log SF noise (S5)", "status": nl.get("status"),
                             "reason": nl.get("reason"), "bound": nl.get("bound")})
            rc = rec.get("repcap") or {}
            if rc:
                rows.append({"population": "COH-REPCAP drop list (S5, label-derived)",
                             "n_entries_dropped": rc.get("n_entries_dropped"), "n_cells_touched": rc.get("n_cells_touched")})
        return rows

    def fold_rows(self) -> list[dict]:
        """TAB-01 (part): per-fold sizes of the six designs and the fold digests (manifests/fold_plan_g201.json)."""
        p = bootstrap.MANIFESTS / "fold_plan_g201.json"
        if not p.exists():
            return []
        plan = json.loads(p.read_text(encoding="utf-8"))
        rows = [dict(r) for r in plan.get("per_fold", [])]
        rows.append({"design": "ALL", "gate_e_sha256": plan.get("gate_e", {}).get("sha256"),
                     "gen16_sha256": plan.get("gen16_digest", {}).get("sha256"),
                     "gen16_sha256_train_test": plan.get("gen16_digest", {}).get("sha256_train_test")})
        return rows

    def population_rows(self) -> list[dict]:
        rows = []
        for name, keys in sorted(self.manifests.items()):
            vc = keys["chemotype"].astype(str).value_counts() if "chemotype" in keys else pd.Series(dtype=int)
            rows.append({"population": name, "target": M.target_of(keys), "n_rows": int(len(keys)),
                         "n_cells": int(keys["ci"].nunique()),
                         "n_extractants": int(keys["extractant"].nunique()) if "extractant" in keys else None,
                         "n_chemotypes": int(keys["chemotype"].nunique()) if "chemotype" in keys else None,
                         "n_publications": int(keys["publication_id"].nunique()) if "publication_id" in keys else None,
                         "kish_chemotypes": M.kish(keys["chemotype"].to_numpy()) if "chemotype" in keys else None,
                         "largest_chemotype_share": float(vc.iloc[0] / vc.sum()) if len(vc) else None})
        return rows

    def cross_target(self, reg: dict) -> list[dict]:
        rows = []
        for pop, arms in (("COH-D-C3", ("A",) + R.H1_COMPARATORS), ("COH-D-C3M", ("A",) + R.H1_COMPARATORS),
                          ("COH-SF-C3", ("B", "C") + R.SF_COMPARATORS)):
            F = self.frame(pop)
            if F is None:
                continue
            card = M.card(F.rows.assign(split_seed=F.seeds[0]))
            named = R.H1_COMPARATORS if F.target == "logd" else R.SF_COMPARATORS
            for arm in arms:
                if not F.has(arm):
                    continue
                s = I._direct(F, arm)
                row = {"population": pop, "arm": arm, **{f"card_{k}": v for k, v in card.items()},
                       "R2_pooled": s["R2_pooled"], "R2_reason": s["R2_pooled_reason"],
                       "R2_n_seeds_defined": s["R2_pooled_n_seeds_defined"],
                       "MET01": s["MET01_MAE_macro_ext"], "MET02": s["MET02_MAE_pooled"],
                       "MET03": s["MET03_MAE_macro_chem"], "MET05": s["MET05_RMSE_pooled"],
                       "MET12_EV_abs": s["MET12_EV_abs"]}
                for y in named:
                    if F.has(y) and y != arm:
                        row[f"Skill_MSE_vs_{y}"] = I.point(F, "MET11", arm, y)["point"]
                        sy = I._direct(F, y)
                        row.update({f"MET12_vs_{y}_{k}": v for k, v in M.met12(s, sy).items()})
                rows.append(row)
        # MET-17 with CI (A's residuals, COH-D-C3 log D rows, COH-SF-C3 pairs)
        td, ts = self.pop_table("COH-D-C3"), self.pop_table("COH-SF-C3")
        if td is not None and ts is not None and M.has_arm(td, "A"):
            m17 = I.met17_infer(td, ts, "A", replicates=self.replicates)
            rows.append({"population": "COH-D-C3 + COH-SF-C3", "arm": "A", "MET17": json.dumps(_clean(m17))})
        return rows

    def reliability_panel(self, reg: dict) -> list[dict]:
        """MET-23 (descriptive, both targets side by side, never a verdict on 'more reliable')."""
        rows = []
        for hid in ("H1", "H2", "H3"):
            v = reg[hid]
            for ev in v.get("evidence", []):
                X, Y, pop = ev["X"], ev["Y"], ev["population"].split("/")[0]
                F = self.frame(pop)
                sX = I._direct(F, X)
                met01_by_design = [I.point(self.frame(pop, d), "MET01", X)["point"] for d in self.designs
                                   if self.frame(pop, d) is not None and self.frame(pop, d).has(X)]
                sc = {}
                for tag, sp in (("sc009", "COH-SC009-D" if F.target == "logd" else "COH-SC009-SF"),
                                ("nosc009", "COH-NOSC009-D" if F.target == "logd" else "COH-NOSC009-SF")):
                    Fs = self.frame(sp)
                    sc[tag] = (I.point(Fs, "MET01", X, Y, contrast=True)["point"]
                               if Fs is not None and Fs.has(X) and Fs.has(Y) else float("nan"))
                ta, tb = I.per_type_met01(F, X), I.per_type_met01(F, Y)
                signs = [np.sign(tb[k] - ta[k]) for k in ta]
                overall = np.sign(ev["delta"])
                rows.append({"claim": hid, "target": F.target, "X": X, "Y": Y,
                             "seed_sd_MET01_X": float(np.std(sX["MET01_MAE_macro_ext_per_seed"], ddof=1)),
                             "ci95_width_over_MET01_Y": ((ev["chem_pct95_high"] - ev["chem_pct95_low"]) / ev["met01_Y"])
                             if np.isfinite(ev.get("chem_pct95_high", np.nan)) else None,
                             "range_MET01_X_over_designs": (max(met01_by_design) - min(met01_by_design))
                             if met01_by_design else None, "n_designs": len(met01_by_design),
                             "loco_positive": ev.get("loco_n_positive"), "loco_n": ev.get("chem_n_clusters"),
                             "sc009_delta": sc["sc009"], "nosc009_delta": sc["nosc009"],
                             "sc009_nosc009_sign_agree": bool(np.sign(sc["sc009"]) == np.sign(sc["nosc009"]))
                             if np.isfinite(sc["sc009"]) and np.isfinite(sc["nosc009"]) else None,
                             "type_sign_consistency": float(np.mean([s == overall for s in signs])) if signs else None})
        return rows

    # -- everything --------------------------------------------------------------------
    def score(self, *, h_from: dict | None = None, full: bool = True) -> dict:
        t0 = time.time()
        st = {"label": self.label, "replicates": self.replicates, "design_status": self._status,
              "designs_complete": [d for d in self.designs if self.complete(d, "primary")],
              "code_digest": self.code_digest, "code_digest_by_design": self.code_digest_by_design}
        if not self.complete("BP", "primary"):
            st["registered"] = {h: {"verdict": "NA", "label": "NA", "reason": "design BP not complete"}
                                for h in ("H1", "H2", "H3", "H4")}
            st["status"] = "design BP not complete: no verdict"
            return st
        self.log("registered claims (design BP, primary)")
        reg = self.registered(h_from=h_from)
        st["registered"] = reg
        self.log("COH-NODMD verdicts")
        if h_from is None:
            st["registered_nodmd"] = self.registered(nodmd=True) if self.complete("BP", "nodmd") else None
        else:
            st["registered_nodmd"] = (self.registered(nodmd=True, h_from=h_from.get("_nodmd"))
                                      if self.complete("BP", "nodmd") else None)
        self.log("NC1")
        st["nc1"] = self.nc1()
        st["prnd"] = {X: self.prnd(X) for X in ("B", "C")}
        self.log("thresholds")
        st["thresholds"] = self.thresholds()
        labels = {h: reg[h]["label"] for h in ("H1", "H2", "H3", "H4")}
        nod = st["registered_nodmd"]
        for h in ("H1", "H2", "H3", "H4"):
            pr = st["prnd"].get({"H2": "B", "H3": "C"}.get(h, ""), {}).get("status") if h in ("H2", "H3") else None
            reg[h]["annotations"] = R.annotate(reg[h], hid=h, nc1_flag=st["nc1"].get("flag", False),
                                               nodmd_label=nod[h]["label"] if nod else None, prnd=pr)
        st["labels"] = labels
        if full:
            self.log("exploratory family")
            st["exploratory"] = self.exploratory(st["nc1"])
            self.log("tables")
            st["tables"] = self.tables_all(reg)
        st["sentences"] = sentences(st)
        st["seconds"] = time.time() - t0
        return st

    def tables_all(self, reg: dict) -> dict:
        types, prnd = self.type_rows()
        sel, met22 = self.selection_rows()
        return {"tab01_populations": self.population_rows(), "tab01_folds": self.fold_rows(),
                "tab02_targets": self.target_rows(),
                "tab03_logd": self.arm_rows("COH-D-C3", LOGD_TABLE_ARMS),
                "tab04_logsf": self.arm_rows("COH-SF-C3", LOGSF_TABLE_ARMS),
                "tab07_cross_target": self.cross_target(reg),
                "tab07_reliability_MET23": self.reliability_panel(reg),
                "tab08_designs": self.design_table(),
                "tab09_secondary": self.secondary_rows(),
                "tab10_types": types, "tab10_prnd_MET24": prnd,
                "tab11_folds": sel, "tab11_MET22": met22, "tab03_fallbacks": self.fallback_summary(sel),
                "tab13_not_selectable": self.not_selectable_rows(),
                "tab14_context": list(TAB14_CONTEXT)}


# --------------------------------------------------------------------------------------
# verdict tables, sentences, files, markdown
# --------------------------------------------------------------------------------------
def verdict_rows(reg: dict) -> list[dict]:
    """TAB-05: one row per component (H1 4, H2 3, H3 3, H4 2 directions)."""
    rows = []
    for hid in ("H1", "H2", "H3", "H4"):
        v = reg.get(hid, {})
        comps = v.get("components", []) or []
        evs = v.get("evidence", []) or []
        if not evs:
            rows.append({"claim": hid, "verdict": v.get("label", "NA"), "reason": v.get("reason")})
            continue
        for ev, rc in zip(evs, comps):
            row = {"claim": hid, "verdict": v.get("label"), "population": ev.get("population"), "X": ev["X"],
                   "Y": ev["Y"], "delta": ev.get("delta"), "delta_Y": rc.get("delta_Y"),
                   "met01_X": ev.get("met01_X"), "met01_Y": ev.get("met01_Y")}
            for k in ("chem_pct9875_low", "chem_pct9875_high", "chem_bca9875_low", "chem_pct95_low", "chem_pct95_high",
                      "chem_p_two_sided", "chem_na_draws", "chem_degenerate", "p_wcr", "p_wcu", "wcr_mean",
                      "wcr_weights", "pub_pct9875_low", "pub_pct9875_high", "pub_p_two_sided", "seeds_positive",
                      "loco_min", "loco_min_cluster", "loco_all_positive", "delta_mae_bal", "delta_r2_bal",
                      "delta_r2_bal_chem_low", "delta_r2_bal_pub_low", "r2_bal_X", "skill_flat_X", "mde80_95",
                      "mde80_dec", "mde80_cr", "ann_delta_mae_pooled", "ann_delta_mae_macro_chem", "ann_skill_mse_XY",
                      "ann_skill_mse_XY_chem_low9875", "ann_delta_r2_pooled", "chem_n_clusters", "pub_n_clusters",
                      "chem_preliminary", "pub_preliminary"):
                row[k] = ev.get(k)
            row["delta_seeds"] = ";".join(_f(x, 6) for x in ev.get("delta_seeds", []))
            for dsg, val in (ev.get("design_points") or {}).items():
                row[f"sign_{dsg}"] = "not run" if val is None else (int(np.sign(val)) if R._fin(val) else "NA")
                row[f"delta_{dsg}"] = val
            for k, val in rc["conditions"].items():
                row[f"cond_{k}"] = val
            row["baseline_better"] = rc.get("baseline_better")
            row["annotations"] = " | ".join(v.get("annotations", []))
            rows.append(row)
    return rows


def threshold_rows(th: dict) -> list[dict]:
    rows = []
    items = [(k, v) for k, v in th.items() if not k.startswith("_")] + list(th.get("_others", {}).items())
    for key, t in items:
        ev = t.get("evidence", {})
        ann = t.get("annotations", {})
        for tau, c in t["classes"].items():
            row = {"id": t["id"], "arm": t["arm"], "population": t["population"], "stat": t["stat"], "tau": float(tau),
                   "class": c.get("class"), "also_ruled_out": c.get("also_ruled_out"), "point": ev.get("point"),
                   "reason": ev.get("reason"), "n_seeds_defined": ev.get("n_seeds_defined"),
                   "per_seed": ";".join(_f(x, 6) for x in ev.get("per_seed", [])),
                   "chem_pct2_5": ev.get("chem_pct95_low"), "chem_pct97_5": ev.get("chem_pct95_high"),
                   "chem_bca95_low": ev.get("chem_bca95_low"), "pub_pct2_5": ev.get("pub_pct95_low"),
                   "pub_pct97_5": ev.get("pub_pct95_high"), "chem_pct9875_low": ev.get("chem_pct9875_low"),
                   "pub_pct9875_low": ev.get("pub_pct9875_low"), "r2_bal": ev.get("r2_bal"),
                   "loco_min": ev.get("loco_min"), "sc009_point": ann.get("sc009_point"),
                   "nosc009_point": ann.get("nosc009_point"), "nosc009_chem_pct95_low": ann.get("nosc009_chem_pct95_low"),
                   "nosc009_chem_pct95_high": ann.get("nosc009_chem_pct95_high"), "nodmd_point": ann.get("nodmd_point"),
                   "tau_plus_1p96se": (ann.get("tau_plus_1p96se") or {}).get(tau),
                   "feature_set_label": R.FEATURE_LABEL}
            for k, v in (c.get("conditions") or {}).items():
                row[f"cond_{k}"] = v
            for k, v in (ann.get("context") or {}).items():
                row[f"ctx_{k}"] = v
            if "I1" in ann:
                row.update({f"I1_{k}": v for k, v in ann["I1"].items()})
            if "R2_0" in ann:
                row.update({"R2_0": ann["R2_0"], "R2_FLAT": ann["R2_FLAT"],
                            "I3_R2_0_at_tau": (ann.get("I3_R2_0_equivalent") or {}).get(tau)})
            if "level_attributable" in c:
                la = c["level_attributable"]
                row.update({"level_attributable": la["text"], "level_class_clause": str(la["class_clause"]),
                            "reached_also_by": " | ".join(c.get("reached_also_by", []))})
            rows.append(row)
    return rows


T_IDS: tuple[str, ...] = ("T1", "T1w", "T2", "T3")


def _ctx_text(ctx: dict | None) -> str:
    """§5 context on the same rows: n, y_sd0, y_var0, MET-01/02/05 and the simple baseline's R², MAE, RMSE."""
    if not ctx:
        return "context NA"
    why = ctx.get("simple_R2_reason")
    return (f"n {ctx.get('n_rows')} rows / {ctx.get('n_cells')} cells, y_sd0 {_f(ctx.get('y_sd0'))}, "
            f"y_var0 {_f(ctx.get('y_var0'))}; MET-01 {_f(ctx.get('MET01'))}, MET-02 {_f(ctx.get('MET02'))}, "
            f"MET-05 {_f(ctx.get('MET05'))}; simple baseline {ctx.get('simple_baseline')}: "
            f"R² {_f(ctx.get('simple_R2'))}{f' ({why})' if why else ''}, MAE {_f(ctx.get('simple_MAE'))}, "
            f"RMSE {_f(ctx.get('simple_RMSE'))}")


def t_annotations(t: dict, tau: str) -> list[str]:
    """The §5 context and the §7 mandatory annotations of one RULE-T row (``Scorer.threshold`` output) at τ, as
    printed beside its class: context, R2_reason / n_seeds_defined, the 98.75 % chemotype and publication lower
    bounds, LOCO minimum, COH-SC009 point, COH-NOSC009 point and 95 % intervals, COH-NODMD point,
    τ + 1.96·se_boot; I1 for T1 / T1w; R2_0, R²_FLAT and the I3 equivalent for log SF."""
    ev = t.get("evidence") or {}
    ann = t.get("annotations") or {}
    if not ann:
        return [f"{t.get('arm')} not run"]
    out = [f"context: {_ctx_text(ann.get('context'))}",
           f"R2_reason {ev.get('reason') or 'none'}, n_seeds_defined {ev.get('n_seeds_defined', 'NA')}",
           f"98.75 % lower bounds: chemotype {_f(ann.get('chem_pct9875_low'))}, "
           f"publication {_f(ann.get('pub_pct9875_low'))}",
           f"LOCO min {_f(ann.get('loco_min'))}",
           f"COH-SC009 point {_f(ann.get('sc009_point'))}"]
    if "nosc009_point" in ann:
        out.append(f"COH-NOSC009 point {_f(ann.get('nosc009_point'))} (chemotype 95 % "
                   f"[{_f(ann.get('nosc009_chem_pct95_low'))}, {_f(ann.get('nosc009_chem_pct95_high'))}], publication "
                   f"95 % [{_f(ann.get('nosc009_pub_pct95_low'))}, {_f(ann.get('nosc009_pub_pct95_high'))}])")
    else:
        out.append("COH-NOSC009 not run")
    out.append(f"COH-NODMD point {_f(ann.get('nodmd_point'))}")
    out.append(f"τ + 1.96·se_boot = {_f((ann.get('tau_plus_1p96se') or {}).get(tau))}")
    i1 = ann.get("I1")
    if i1:
        out.append(f"I1: R²_pooled {_f(i1.get('R2_pooled'))} = s_b {_f(i1.get('s_b'))}·R²_level "
                   f"{_f(i1.get('R2_level'))} + s_w {_f(i1.get('s_w'))}·R²_within_cell {_f(i1.get('R2_within_cell'))}")
    if "R2_0" in ann:
        out.append(f"R2_0 {_f(ann.get('R2_0'))}; R²_FLAT {_f(ann.get('R2_FLAT'))}; I3: R²_pooled {_f(tau, 2)} ⇔ "
                   f"R2_0 {_f((ann.get('I3_R2_0_equivalent') or {}).get(tau))}")
    return out


def _tau_key(th: dict, tau) -> str:
    """The classes key of ``tau`` (str of a float in R.TAUS) in a (possibly JSON-round-tripped) threshold state."""
    for t in th.values():
        if isinstance(t, dict):
            for k in (t.get("classes") or {}):
                if float(k) == float(tau):
                    return k
    return str(tau)


def threshold_context(th: dict, tau) -> str:
    """T1, T1w, T2, T3 at τ with each one's point, class, §5 context and §7 annotations (I1 with T1 in the same
    sentence): printed with every threshold sentence (PAT-6, 'pattern not anticipated', NO-REACHED)."""
    key = _tau_key(th, tau)
    parts = []
    for tid in T_IDS:
        t = th.get(tid)
        if not t:
            continue
        ev = t.get("evidence") or {}
        cls = ((t.get("classes") or {}).get(key) or {}).get("class", "NA")
        parts.append(f"{tid} ({t.get('arm')}, {t.get('population')}, {t.get('stat')}): point {_f(ev.get('point'))}, "
                     f"class {cls}; " + "; ".join(t_annotations(t, key)))
    return " | ".join(parts)


def sentences(st: dict) -> dict:
    labels = st.get("labels") or {h: st["registered"][h].get("label", "NA") for h in ("H1", "H2", "H3", "H4")}
    out = {"easier": R.easier_sentence(labels), "pat": R.pat_sentences(labels, h4_label=labels.get("H4")),
           "h4": R.h4_sentence(st["registered"]["H4"]), "thresholds": [], "more_reliable": R.MORE_RELIABLE}
    th = st.get("thresholds") or {}
    for tau in map(str, R.TAUS):
        cls = {k: th[k]["classes"][tau]["class"] for k in ("T1", "T1w", "T2", "T3") if k in th}
        la = (th.get("T1", {}).get("classes", {}).get(tau, {}).get("level_attributable") or {}).get("text",
                                                                                                     R.NOT_EVALUABLE)
        s = R.threshold_sentence(float(tau), cls, la)
        s = s if s else {"id": "NO-REACHED", "tau": float(tau),
                         "text": f"At τ = {float(tau):g}: " + "; ".join(f"{k}: {v}" for k, v in cls.items())}
        s["context"] = threshold_context(th, tau)          # §5 / §7: printed with the sentence, never inside it
        out["thresholds"].append(s)
    return out


def write_results(st: dict, out_dir: Path) -> list[Path]:
    """results/<label>/*.csv and *.json (state, verdicts, thresholds, exploratory, every table)."""
    out_dir = bootstrap.assert_under_g201(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def csv(name, rows):
        p = out_dir / f"{name}.csv"
        pd.DataFrame(rows).to_csv(p, index=False, lineterminator="\n")
        paths.append(p)

    if st.get("registered"):
        csv("tab05_verdicts", verdict_rows(st["registered"]))
        if st.get("registered_nodmd"):
            csv("tab09f_verdicts_nodmd", verdict_rows(st["registered_nodmd"]))
    if st.get("thresholds"):
        csv("tab06_thresholds", threshold_rows(st["thresholds"]))
    if st.get("exploratory"):
        csv("tab12_exploratory", st["exploratory"])
    for name, rows in (st.get("tables") or {}).items():
        csv(name, rows)
    clean = _clean(st)
    for name, obj in (("report_state", clean), ("verdicts", {k: clean.get(k) for k in ("label", "labels", "registered",
                                                                                       "registered_nodmd", "nc1", "prnd")}),
                      ("sentences", clean.get("sentences"))):
        p = out_dir / f"{name}.json"
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(obj, fh, indent=1, ensure_ascii=False)
        paths.append(p)
    return paths


def h_points_from_json(state: dict) -> dict:
    """Same as ``h_points`` from a saved report_state.json (null = design not run is kept as None; a stored NA
    point is null too, which the confirmation treats as not run only if the discovery status says so)."""
    status = state.get("design_status", {})
    out = {}
    for key, sub, variant in (("registered", None, "primary"), ("registered_nodmd", "_nodmd", "nodmd")):
        reg = state.get(key) or {}
        tgt = out if sub is None else out.setdefault(sub, {})
        for hid in ("H1", "H2", "H3", "H4"):
            for ev in (reg.get(hid) or {}).get("evidence", []) or []:
                pts = {}
                for d, v in (ev.get("design_points") or {}).items():
                    complete = status.get(d, {}).get(variant, {}).get("status") == "complete"
                    pts[d] = (float(v) if v is not None else float("nan")) if complete else None
                tgt[(hid, ev["Y"])] = pts
    return out


def _tab06_lines(th: dict, conf_th: dict | None) -> list[str]:
    """TAB-06 markdown rows of T1, T1w, T2, T3: every class with its §5 context and §7 annotations on the same row;
    ``conf_th`` (the confirmation's thresholds) adds the §11 confirmation status of each class."""
    L = ["| T | arm | τ | class | point | chem 2.5th | BCa 95 % low | pub 2.5th | seeds ≥ τ | R2_balanced | "
         "context (§5) and annotations (§7) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tid in T_IDS:
        t = th.get(tid)
        if not t:
            continue
        ev = t.get("evidence") or {}
        for tau, c in t["classes"].items():
            ann = [R.FEATURE_LABEL]
            if c.get("also_ruled_out"):
                ann.append(f"both 97.5th percentiles below τ (chemotype {_f(ev.get('chem_pct95_high'))}, publication "
                           f"{_f(ev.get('pub_pct95_high'))}; POINT_ONLY printed, O7)")
            if "level_attributable" in c:
                ann.append(c["level_attributable"]["text"])
                ann += c.get("reached_also_by", [])
            if conf_th is not None:
                ct = ((conf_th.get(tid) or {}).get("classes") or {}).get(_tau_key(conf_th, tau), {}).get("class")
                ann.append(R.confirm_t(c.get("class"), ct)["text"])
            ann += t_annotations(t, tau)
            L.append(f"| {tid} | {t['arm']} | {tau} | {c.get('class')} | {_f(ev.get('point'))} | "
                     f"{_f(ev.get('chem_pct95_low'))} | {_f(ev.get('chem_bca95_low'))} | {_f(ev.get('pub_pct95_low'))} | "
                     f"{c.get('seeds_at_or_above', 'NA')} | {_f(ev.get('r2_bal'))} | {'; '.join(ann)} |")
    return L


def _digest_line(st: dict, confirmation: dict | None) -> str:
    """The prediction code digest (``fitting.code_digest``; one digest over every complete design and variant)."""
    d = st.get("code_digest")
    txt = (f"Prediction code digest (`fitting.code_digest`, one digest over every complete design and variant): "
           f"`{d}`." if d else "Prediction code digest: not recorded (tables not read from DONE markers).")
    if confirmation is not None:
        c = confirmation.get("code_digest")
        same = "identical" if (c and d and c == d) else ("not checked" if not (c and d) else "DIFFERENT")
        txt += f" Confirmation: `{c or 'not recorded'}` ({same})."
    return txt


def render(st: dict, *, confirmation: dict | None = None) -> str:
    """GEN20_1_REPORT.md from a (JSON-clean) state: the applicable pre-written sentences first, verbatim."""
    st = _clean(st)
    L = []
    L.append(f"# gen20.1 report: absolute log D versus strict-adjacent log SF (label `{st['label']}`)")
    L.append("")
    note = ("*Produced by `scripts/g201_report.py` (S10) from saved tables only; every class and verdict is assigned "
            "by the sealed rules (`PRE_REGISTRATION.md`, SEALED-SHA256 footer) with no override. Design BP, mode Z "
            "(k = 0), 5-seed means; " + R.FEATURE_LABEL + ".*")
    if st.get("status"):
        L.append(f"**{st['status']}**")
        L.append("")
        L.append(note)
        return "\n".join(L) + "\n"
    s = st["sentences"]
    conf_status = {}
    if confirmation is not None:
        for h in ("H1", "H2", "H3", "H4"):
            conf_status[h] = R.confirm_h(h, st["registered"][h], (confirmation.get("registered") or {}).get(h))
    L.append("## Pre-written sentences (PRE_REGISTRATION §1, §7)")
    L.append("")
    L.append(f"- {s['easier']['text']}")
    for p in s["pat"]:
        L.append(f"- **{p['id']}.** {p['text']}")
    for t in s["thresholds"]:
        ctx = threshold_context(st.get("thresholds") or {}, t["tau"]) if "tau" in t else t.get("context")
        L.append(f"- **{t['id']}.** {t['text']}" + (f" *Context (§5, §7; on the same rows): {ctx}.*" if ctx else ""))
    ex_beside = [r for r in (st.get("exploratory") or []) if r.get("ex") in ("EX-01", "EX-02", "EX-03", "EX-04")]
    beside = "; ".join(f"{r['ex']} {r['candidate']} vs {r['comparator']}: Δ {_f(r.get('delta'))}, q {_f(r.get('q_bh'))}"
                       for r in ex_beside)
    L.append(f"- {s['h4']['text']}" + (f" Beside it: {beside}." if beside else ""))
    L.append(f"- {s['more_reliable']}")
    L.append("")
    L.append(note)
    L.append("")
    L.append(_digest_line(st, confirmation))
    L.append("")
    L.append("## Registered verdicts (TAB-05; K = 4, deciding intervals 98.75 %)")
    L.append("")
    L.append("| claim | verdict | annotations |")
    L.append("|---|---|---|")
    for h in ("H1", "H2", "H3", "H4"):
        v = st["registered"][h]
        ann = list(v.get("annotations") or [])
        if h in conf_status:
            ann = [a for a in ann if a != R.CONFIRMATION_PENDING] + [conf_status[h]["status"]]
        L.append(f"| {h} | {v.get('label', 'NA')} | {'; '.join(ann)} |")
    L.append("")
    L.append("| claim | X | Y | Δ | δ_Y | chem 98.75 % low | BCa low | p_wcr | pub 98.75 % low | seeds > 0 | LOCO min | "
             "(a)-(k) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in verdict_rows(st["registered"]):
        if "X" not in r:
            continue
        cond = "".join(k if r.get(f"cond_{k}") else ("·" if r.get(f"cond_{k}") is None else "✗")
                       for k in R.CONDITIONS_ALL)
        L.append(f"| {r['claim']} | {r['X']} | {r['Y']} | {_f(r['delta'])} | {_f(r['delta_Y'])} | "
                 f"{_f(r['chem_pct9875_low'])} | {_f(r['chem_bca9875_low'])} | {_f(r['p_wcr'])} | "
                 f"{_f(r['pub_pct9875_low'])} | {r['seeds_positive']} | {_f(r['loco_min'])} | {cond} |")
    L.append("")
    nc = st.get("nc1") or {}
    nc_txt = "; ".join(f"{k}: Δ {_f(v.get('delta'))}, chem 98.75 % [{_f(v.get('chem_pct9875_low'))}, "
                       f"{_f(v.get('chem_pct9875_high'))}], p_wcr {_f(v.get('p_wcr'))}, (b)-(g) met = {v.get('meets_b_to_g')}"
                       for k, v in nc.items() if isinstance(v, dict) and "meets_b_to_g" in v)
    L.append(f"**NC1** (A vs A_PRIME, RULE-C (b)-(g), both directions): flag = {nc.get('flag')} "
             f"({nc.get('status', '')}){'; ' + nc_txt if nc_txt else ''}.")
    L.append("")
    L.append("## Threshold attainment (TAB-06; RULE-T at τ = 0.4 and 0.5)")
    L.append("")
    L += _tab06_lines(st.get("thresholds") or {},
                      None if confirmation is None else (confirmation.get("thresholds") or {}))
    L.append("")
    if confirmation is not None:
        L.append("### Threshold classes on the confirmation seeds (§11; design BP)")
        L.append("")
        L += _tab06_lines(confirmation.get("thresholds") or {}, None)
        L.append("")
    ex = st.get("exploratory") or []
    if ex:
        n_rej = sum(1 for r in ex if r.get("reject_bh_q05"))
        L.append(f"## Exploratory family (TAB-12; BH q = 0.05 over m = {len(ex)}; never promoted)")
        L.append("")
        L.append(f"{n_rej} of {len(ex)} tests have q ≤ 0.05; tests whose arm was not run enter with p = 1.")
        L.append("")
    L.append("## Identities (fixed text)")
    L.append("")
    for k, v in IDENTITIES.items():
        L.append(f"- **{k}** {v}")
    L.append("")
    L.append("Separation claims rest on the log SF table only; R² is never ranked, differenced or normalised across "
             "targets (TAB-07 prints the cards side by side).")
    L.append("")
    if confirmation is None:
        L.append(f"Confirmation (§11): {R.CONFIRMATION_PENDING}.")
    else:
        L.append(f"Confirmation (§11): design BP on the confirmation seeds; {R.CONFIRMATION_LABEL}; (h) from discovery.")
    L.append("")
    L.append(f"Tables: `results/{st['label']}/` (tab01..tab14 CSV, report_state.json).")
    return "\n".join(L) + "\n"


def build(label: str = "discovery", *, out_dir: Path | None = None, report_md: Path | None = None,
          confirmation: dict | None = None, scorer_kwargs: dict | None = None, prereg: Path | None = None,
          log=print) -> dict:
    """S10: refuse smoke labels, require the seal, score, write results/<label>/ and GEN20_1_REPORT.md."""
    from .seal import require_sealed
    refuse_smoke(label)
    require_sealed(prereg or bootstrap.PREREG)
    sc = Scorer(label, log=log, **(scorer_kwargs or {}))
    if confirmation is not None:
        cd, dd = confirmation.get("code_digest"), sc.code_digest
        if cd and dd and cd != dd:
            raise ReportRefused(f"the confirmation predictions were produced by code {cd}, the {label} predictions by "
                                f"{dd}: not the identical procedure (PRE_REGISTRATION §11)")
    st = sc.score()
    out_dir = Path(out_dir) if out_dir is not None else bootstrap.RESULTS / label
    paths = write_results(st, out_dir)
    md = render(st, confirmation=confirmation)
    rp = bootstrap.assert_under_g201(report_md or REPORT_MD)
    with open(rp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(md)
    paths.append(rp)
    return {"state": st, "paths": paths, "markdown": md}

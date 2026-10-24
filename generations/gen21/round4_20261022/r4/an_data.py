"""Block C data layer: verified An(III) records as TRAINING-ONLY auxiliary data (brief section 6 'Actinide transfer';
PROTOCOL section 12).  Label-free selection and exclusion rules, fixed before any block-C outer score.

Eligible records (gen19 archive via gen19ct.data.load.load_archive, which adds g19 publication ids):
  metal category actinide, metal in {Am, Cm, Pu, Cf}, oxidation state 3 (An(VI) / unknown states excluded);
  acid_anion == 'nitrate' (nitric acid, the cohort's majority regime); no aqueous complexant and no active (or
  unknown-concentration) aqueous hold-back agent (SANEX-type aqueous complexation reverses the selectivity mechanism);
  no ionic-liquid diluent (solvent key contains '['); finite log D; parseable extractant; gen19 tier MODEL (drops
  duplicate-member and not-modelable rows); no gen19 value conflict; no suspect flag; exactly one organic extractant
  and no acidic co-extractant recorded as a modifier (synergistic mixtures dropped: the encoder sees one extractant;
  alcohol phase modifiers are kept).
Chemotype assignment of non-cohort extractants (the project's rule, levels.TANIMOTO_CLUSTER_THRESHOLD = 0.7, DIRECT
one-hop links as in gen11.pools.chemotype_assignment; no An-An transitive closure): an An extractant is LINKED to every
cohort chemotype that has a member at Morgan (radius 2, 2048 bits) Tanimoto >= 0.7.
Exclusions (union of domains; applied before any fit, tuning or weight choice):  for a set of held-out Ln cells, every
An record is dropped that shares (a) the gen19 publication, (b) the gen19 cross-publication copy group, (c) the exact
extractant (canonical SMILES), or (d) a linked chemotype with any held-out cell.  Inner blocks apply the same rule again
against the outer-training cells outside the block's inner-training set.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import paths

AN3 = ("Am", "Cm", "Pu", "Cf")
ACIDIC_MODIFIERS = r"HDEHP|HEH\[?EHP|D2EHPA|EHEHPA|Cyanex|HDBP"   # acidic co-extractants recorded as 'modifier' (synergism)
TANIMOTO = 0.7
FP_RADIUS, FP_BITS = 2, 2048


def _canon(smi: str) -> str | None:
    from rdkit import Chem
    m = Chem.MolFromSmiles(str(smi))
    return Chem.MolToSmiles(m) if m is not None else None


@lru_cache(maxsize=None)
def _fp(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    return gen.GetFingerprint(Chem.MolFromSmiles(smiles))


@lru_cache(maxsize=1)
def an3_records() -> pd.DataFrame:
    import sys
    sys.path.insert(0, str(paths.ROOT / "generations" / "gen19_chem_transfer"))
    from gen19ct.data.load import load_archive
    from gen20k.splits import copy_group_map
    a = load_archive()
    a = a[(a.metal_category == "actinide") & a.metal_symbol.isin(AN3)
          & a.metal_oxidation_state.astype(str).isin(["3", "3.0"])]
    has_cx = a.complexant_names.map(lambda v: v is not None and not (isinstance(v, float)) and len(v) > 0)
    hb_smi = a.holdback_smiles_canonical.astype("string")
    hb_conc = pd.to_numeric(a.holdback_concentration_M, errors="coerce")
    has_hb = (hb_smi.notna() & (hb_smi.str.len() > 0) & ~(hb_conc == 0)).fillna(False).astype(bool)   # active / unknown
    acid_mod = a.modifier_name.astype("string").str.contains(ACIDIC_MODIFIERS, case=False, regex=True).fillna(False)
    a = a[(a.acid_anion.astype(str) == "nitrate") & ~has_cx & ~has_hb
          & ~a.solvent_key.astype(str).str.contains(r"\[", regex=True)
          & a.log_D.notna() & (a.g19_tier == "MODEL") & ~a.in_value_conflict.astype(bool)
          & (a.n_organic_extractants == 1) & ~acid_mod.astype(bool) & ~a.has_suspect_flag.fillna(False).astype(bool)]
    a = a.assign(extractant=a.extractant_primary_smiles.astype(str).map(_canon))
    a = a[a.extractant.notna()].copy()
    cg = copy_group_map()
    a["publication"] = a.g19_publication_id.astype(str)
    a["copy_group"] = a.publication.map(lambda p: cg.get(p, "own:" + p))
    out = a[["source_record_id", "metal_symbol", "log_D", "extractant", "publication", "copy_group", "acid_concentration_M",
             "extractant_primary_concentration_M", "solvent_key", "doi_primary_corrected"]].reset_index(drop=True)
    out["rid"] = np.arange(len(out))
    return out


@lru_cache(maxsize=1)
def cohort_chemotypes() -> dict:
    """canonical cohort extractant -> chemotype, from the frozen bench frame."""
    from gen20k import data as D
    f = D.load_bench().frame
    return {_canon(e): str(c) for e, c in zip(f.extractant, f.chemotype)}


@lru_cache(maxsize=1)
def an_links() -> dict:
    """An extractant (canonical) -> frozenset of cohort chemotypes it links to (Tanimoto >= 0.7 to any member)."""
    from rdkit import DataStructs
    coh = cohort_chemotypes()
    coh_ext = sorted(coh)
    coh_fps = [_fp(e) for e in coh_ext]
    links = {}
    for e in sorted(set(an3_records().extractant)):
        sims = DataStructs.BulkTanimotoSimilarity(_fp(e), coh_fps)
        links[e] = frozenset(coh[coh_ext[i]] for i, s in enumerate(sims) if s >= TANIMOTO)
    return links


def excluded_by(cells: list, an: pd.DataFrame | None = None) -> np.ndarray:
    """Boolean mask over An records: True where the record shares publication, copy group, exact extractant or a
    linked chemotype with any of the given (held-out) Ln cell records (dicts with publication / extractant /
    chemotype)."""
    from gen20k.splits import copy_group_map
    an = an3_records() if an is None else an
    cg = copy_group_map()
    pubs = {str(c["publication"]) for c in cells}
    cgs = {cg.get(p, "own:" + p) for p in pubs}
    exts = {_canon(str(c["extractant"])) for c in cells}
    chems = {str(c["chemotype"]) for c in cells}
    links = an_links()
    return (an.publication.isin(pubs) | an.copy_group.isin(cgs) | an.extractant.isin(exts)
            | an.extractant.map(lambda e: bool(links[e] & chems))).to_numpy()


def fold_an(bench, rec) -> pd.DataFrame:
    """An records usable in an outer fold: exclusions against every held-out cell of the fold (all test cells, not only
    C3) and against the design-excluded cells (neither train nor test), which carry held-out publications."""
    f = bench.frame
    train = set(map(int, rec["outer"]["train_index"]))
    held = [{"publication": f.publication_id.iat[i], "extractant": f.extractant.iat[i], "chemotype": f.chemotype.iat[i]}
            for i in range(len(f)) if i not in train]
    an = an3_records()
    return an[~excluded_by(held, an)].reset_index(drop=True)


def inner_an(bench, rec, blk, an_fold: pd.DataFrame) -> pd.DataFrame:
    """An records usable inside one inner block: the outer-fold set minus records linked to the block's
    inner-validation cells (and to outer-training cells outside the block's inner-training set)."""
    f = bench.frame
    itr = set(map(int, blk["train_index"]))
    held = [{"publication": f.publication_id.iat[i], "extractant": f.extractant.iat[i], "chemotype": f.chemotype.iat[i]}
            for i in map(int, rec["outer"]["train_index"]) if i not in itr]
    return an_fold[~excluded_by(held, an_fold)].reset_index(drop=True)


def counts(an: pd.DataFrame) -> dict:
    return {"rows": int(len(an)), "publications": int(an.publication.nunique()),
            "copy_groups": int(an.copy_group.nunique()), "extractants": int(an.extractant.nunique()),
            "metals": {k: int(v) for k, v in an.metal_symbol.value_counts().items()}}

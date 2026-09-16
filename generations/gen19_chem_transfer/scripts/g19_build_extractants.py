"""Gen19 extractant representation, chemical families and mechanisms (brief sections 4.2, 5, 27, 28 A.5/A.11).

Writes
  descriptors/family_rules.json            ordered SMARTS family rules (read back by the classifier)
  descriptors/extractant_components.csv    one row per canonical component structure (+ name-only rows)
  descriptors/extractant_systems.csv       one row per archive extractant_system_key (incl. the null key)
  data_audit/ligand_alias_collisions.csv   name <-> structure collisions, spelling variants, SMILES mismatches
  data_audit/family_coverage.csv           per family / mechanism coverage of MODEL rows; levels system_family,
                                           component_family and mechanism carry the name corrections (known
                                           name override, name-implied structure), *_structural levels do not
  data_audit/named_extractant_presence.csv row-level search for named industrial extractants
  manifests/g19_build_extractants.json     (+ run_info/)

No model is trained or fitted.  Run from the repository root:
  PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \
      .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_build_extractants.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.chemistry import ligands as L  # noqa: E402
from gen19ct.chemistry import mechanisms as M  # noqa: E402
from gen19ct.data.load import load_archive  # noqa: E402
from gen19ct.manifest import Run, write_csv, write_json  # noqa: E402

SCRIPT = "g19_build_extractants"
OUT_RULES = paths.DESCRIPTORS_DIR / "family_rules.json"
OUT_COMPONENTS = paths.DESCRIPTORS_DIR / "extractant_components.csv"
OUT_SYSTEMS = paths.DESCRIPTORS_DIR / "extractant_systems.csv"
OUT_COLLISIONS = paths.DATA_AUDIT_DIR / "ligand_alias_collisions.csv"
OUT_COVERAGE = paths.DATA_AUDIT_DIR / "family_coverage.csv"
OUT_PRESENCE = paths.DATA_AUDIT_DIR / "named_extractant_presence.csv"
RAW_DIR = paths.ARCHIVE_DIR / "raw"

METAL_CATEGORIES = ("lanthanide", "actinide", "rare_earth_non_lanthanide", "transition_metal",
                    "alkaline_earth", "post_transition_metal")
TIERS = ("MODEL", "TARGET_ONLY", "NO_TARGET")
NULL_KEY_LABEL = "<NO_EXTRACTANT_RECORDED>"

PRESENCE_NAME_TERMS = ("PC88A", "PC-88A", "P507", "EHEHPA", "HEH[EHP]", "Ionquest 801", "Cyanex 272",
                       "Cyanex272", "D2EHPA", "DEHPA", "HDEHP", "P204", "TODGA", "TBP", "TOPO", "CMPO",
                       "Aliquat", "HTTA")
PRESENCE_SMILES_REFS = ("D2EHPA", "PC88A", "Cyanex 272")
MASTER_TEXT_COLUMNS = ("extractant_name_raw", "extractant_smiles_raw", "modifier_name_raw", "complexant_name_raw",
                       "ini_comp_raw", "comments_raw")
TITLE_TEXT_COLUMNS = ("reference_title", "publication_title")
#: full chemical names of the brief section 19 acids (searched in text and title columns; the short aliases of
#: REFERENCE_EXTRACTANTS are also searched in the titles).  Separators between name parts may be spaces,
#: hyphens, commas or nothing; brackets are optional.
_SEP = r"[\s\-,]*"
FULL_NAME_PATTERNS: dict[str, tuple[str, ...]] = {
    "D2EHPA": (rf"(?:bis|di){_SEP}\(?{_SEP}2{_SEP}ethyl{_SEP}hexyl{_SEP}\)?{_SEP}(?:hydrogen{_SEP})?phosphoric{_SEP}acid",
               rf"(?:bis|di){_SEP}\(?{_SEP}2{_SEP}ethyl{_SEP}hexyl{_SEP}\)?{_SEP}(?:hydrogen{_SEP})?phosphate"),
    "PC88A": (rf"2{_SEP}ethyl{_SEP}hexyl{_SEP}phosphonic{_SEP}acid",
              rf"\(?{_SEP}2{_SEP}ethyl{_SEP}hexyl{_SEP}\)?{_SEP}phosphonate"),
    "Cyanex 272": (rf"(?:bis|di){_SEP}\(?{_SEP}2{_SEP}4{_SEP}4{_SEP}trimethyl{_SEP}pentyl{_SEP}\)?{_SEP}phosphinic{_SEP}acid",),
}


# ----------------------------------------------------------------------------------------------
def _s(x: object) -> str | None:
    return None if L._is_missing(x) else str(x)


def _join(items, sep="; ") -> str:
    return sep.join(str(i) for i in items)


def _counter_str(c: Counter, sep="; ") -> str:
    return sep.join(f"{k}:{v}" for k, v in sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def system_id(key: str | None) -> str:
    return "sys_null" if key is None else "sys_" + hashlib.sha1(key.encode()).hexdigest()[:10]


def term_regex(term: str) -> re.Pattern:
    """Case-insensitive, alphanumeric-bounded literal; a space in the term also matches a hyphen or
    nothing (``Cyanex 272`` ~ ``Cyanex-272`` ~ ``Cyanex272``); a hyphen in the term stays literal."""
    body = r"[\s\-]*".join(re.escape(tok) for tok in term.split())
    tail = "" if term == "Aliquat" else r"(?![A-Za-z0-9])"
    return re.compile(r"(?<![A-Za-z0-9])" + body + tail, re.IGNORECASE)


# ----------------------------------------------------------------------------------------------
def build(args: argparse.Namespace) -> None:
    with Run(SCRIPT, args=vars(args), seed=None) as run:
        write_json(OUT_RULES, L.FAMILY_RULES_SPEC)
        clf = L.FamilyClassifier.from_json(OUT_RULES)
        assert L.rules_digest(clf.rules) == L.rules_digest(L.FAMILY_RULES_SPEC)

        df = load_archive(copy=False)
        n = len(df)
        raw_files = sorted(RAW_DIR.glob("*.csv"))
        run.inputs(paths.ARCHIVE_MASTER, raw_files, Path(L.__file__), Path(M.__file__))

        tier = df["g19_tier"].to_numpy()
        metal = df["g19_metal"].to_numpy()
        category = df["metal_category"].to_numpy()
        pub = df["g19_publication_id"].to_numpy()
        study = df["g19_study_id"].to_numpy()
        comps_col = [L.iter_component_dicts(c) for c in df["components"]]
        keys = [_s(k) for k in df["extractant_system_key"]]
        exp_ids = df["source_record_id"].astype(str).to_numpy()
        z_of = {m: z for m, z in zip(df["g19_metal"], df["atomic_number"]) if isinstance(m, str) and pd.notna(z)}

        def metal_list(ms) -> str:
            return _join(sorted(set(ms), key=lambda m: (z_of.get(m, 999), m)), ";")

        # ---------------- alias index ----------------
        idx = L.AliasIndex(clf.rules)
        masking = np.zeros(n, dtype=bool)
        for i, comps in enumerate(comps_col):
            masking[i] = L.AliasIndex.is_masking_row(comps)
            idx.add(i, tier[i] == "MODEL", comps, masking_row=bool(masking[i]))
        idx.finalize()

        # ---------------- per-structure / per-name aggregation ----------------
        st = defaultdict(lambda: {"roles": Counter(), "sources": Counter(), "raw": Counter(), "rows": set(),
                                  "model": set(), "pubs": set(), "masking": set()})
        name_only = defaultdict(lambda: {"roles": Counter(), "rows": set(), "model": set(), "pubs": set(),
                                         "masking": set()})
        empty_slot = {"roles": Counter(), "rows": set(), "model": set(), "pubs": set(), "masking": set(),
                      "conc": Counter()}
        raw_pairs = Counter()
        name_roles = defaultdict(Counter)
        for i, comps in enumerate(comps_col):
            for c in comps:
                smi, nm, role = _s(c.get("smiles_canonical")), _s(c.get("name")), c.get("role")
                if nm is not None:
                    name_roles[nm][role] += 1
                if smi is not None:
                    r = st[smi]
                    r["roles"][role] += 1
                    r["sources"][str(c.get("structure_source"))] += 1
                    if _s(c.get("smiles_raw")):
                        r["raw"][str(c["smiles_raw"])] += 1
                        raw_pairs[(str(c["smiles_raw"]), smi)] += 1
                elif nm is not None:
                    r = name_only[nm]
                    r["roles"][role] += 1
                else:
                    r = empty_slot
                    r["roles"][role] += 1
                    r["conc"][c.get("concentration_M")] += 1
                r["rows"].add(i)
                r["pubs"].add(pub[i])
                if tier[i] == "MODEL":
                    r["model"].add(i)
                if masking[i]:
                    r["masking"].add(i)

        parent_groups = defaultdict(set)
        for smi in st:
            parent_groups[L.parent_key(smi)].add(smi)

        # ---------------- components table ----------------
        comp_rows = []
        desc_cache: dict[str, dict] = {}
        for smi in sorted(st, key=lambda s: (-len(st[s]["rows"]), s)):
            r = st[smi]
            d = L.describe_structure(smi, clf)
            desc_cache[smi] = d
            mech = M.component_mechanism(d["family"], d["core_family"])
            aliases = idx.aliases(smi)
            rejected = []
            fails = set()
            for nm, _, _ in aliases:
                reason = idx.rejection_reason(nm, smi)
                if reason:
                    rejected.append(f"{nm}:{reason}")
                fails.update(idx.check_fail.get((nm, smi), []))
            overrides = [nm for nm, _, _ in aliases if nm in L.KNOWN_NAME_OVERRIDES
                         and L.KNOWN_NAME_OVERRIDES[nm]["requires_failed_check"] in idx.check_fail.get((nm, smi), [])]
            flags = []
            if d.get("family_status") == "AMBIGUOUS":
                flags.append("FAMILY_AMBIGUOUS")
            if len(aliases) > 1:
                flags.append("MULTIPLE_NAMES")
            if rejected:
                flags.append("HAS_REJECTED_NAMES")
            if len(parent_groups[d.get("parent_connectivity_key")]) > 1:
                flags.append("SHARES_PARENT_WITH_OTHER_CANONICAL_SMILES")
            if overrides:
                flags.append("KNOWN_NAME_OVERRIDE_APPLIES")
            if idx.canonical_name_status(smi).startswith("FALLBACK"):
                flags.append("CANONICAL_NAME_FALLBACK")
            rec = {
                "component_id": L.component_id(smi), "record_type": "STRUCTURE", "smiles_canonical": smi,
                "resolved_smiles": smi, "structure_basis": "ARCHIVE_STRUCTURE",
                "canonical_name": idx.canonical_name(smi), "canonical_name_status": idx.canonical_name_status(smi),
                "n_names": len(aliases),
                "aliases": _join(f"{a}|{nr}|{nm_}" for a, nr, nm_ in aliases),
                "rejected_names": _join(rejected), "name_structure_checks_failed": _join(sorted(fails), "|"),
                "roles_seen": _counter_str(r["roles"]), "primary_role": r["roles"].most_common(1)[0][0],
                "structure_sources": _counter_str(r["sources"]), "n_raw_smiles_variants": len(r["raw"]),
                "n_entries": sum(r["roles"].values()), "n_rows": len(r["rows"]), "n_model_rows": len(r["model"]),
                "n_publications": len(r["pubs"]), "n_masking_rows": len(r["masking"]),
            }
            rec.update({k: v for k, v in d.items() if k != "morgan_r2_2048_onbits"})
            rec["n_structures_same_parent"] = len(parent_groups[d.get("parent_connectivity_key")])
            rec.update(mech)
            rec.update({"family_structural": d["family"], "core_family_structural": d["core_family"],
                        "mechanism_structural": mech["mechanism"], "acidity_class_structural": d["acidity_class"],
                        "family_rules_status": d["family_status"]})
            # every name on the structure triggers a known override -> the component IS the named chemistry
            if overrides and all(nm in overrides for nm, _, _ in aliases):
                ov = L.KNOWN_NAME_OVERRIDES[overrides[0]]
                ov_mech, ov_basis = M.family_mechanism(ov["family_by_name"])
                rec.update({"family": ov["family_by_name"], "core_family": ov["family_by_name"], "mechanism": ov_mech,
                            "mechanism_basis": "KNOWN_NAME_OVERRIDE:" + ov_basis, "acidity_class": ov["acidity_class"]})
                flags.append("KNOWN_NAME_OVERRIDE_APPLIED_TO_COMPONENT")
            canon = idx.canonical_name(smi)
            conflict = bool(canon is not None and idx.check_fail.get((canon, smi)))
            rec["name_structure_conflict"] = conflict
            d["_name_structure_conflict"] = conflict
            if conflict:
                rec["family_status"] = "NAME_STRUCTURE_CONFLICT"
                flags.append("NAME_STRUCTURE_CONFLICT")
            rec["known_name_override"] = _join(f"{o}->{L.KNOWN_NAME_OVERRIDES[o]['family_by_name']}" for o in overrides)
            rec["pka_value"] = None
            rec["pka_status"] = "NOT_CURATED: no literature pKa entered (never fabricated)"
            rec["flags"] = _join(flags, "|")
            rec["morgan_r2_2048_onbits"] = d.get("morgan_r2_2048_onbits")
            comp_rows.append(rec)

        for nm in sorted(name_only, key=lambda s: (-len(name_only[s]["rows"]), s)):
            r = name_only[nm]
            res = L.resolve_occurrence(idx, nm, None, clf)
            look = L.NAME_ONLY_LOOKUP.get(nm, {})
            rec = {"component_id": L.component_id(None, nm), "record_type": "NAME_ONLY", "smiles_canonical": None,
                   "resolved_smiles": res["smiles"], "structure_basis": res["basis"] + (
                       f" [{look['basis']}]" if look.get("basis") else ""),
                   "canonical_name": nm, "canonical_name_status": "NAME_ONLY", "n_names": 1,
                   "aliases": f"{nm}|{len(r['rows'])}|{len(r['model'])}", "rejected_names": "",
                   "name_structure_checks_failed": "",
                   "roles_seen": _counter_str(r["roles"]), "primary_role": r["roles"].most_common(1)[0][0],
                   "structure_sources": "None", "n_raw_smiles_variants": 0,
                   "n_entries": sum(r["roles"].values()), "n_rows": len(r["rows"]), "n_model_rows": len(r["model"]),
                   "n_publications": len(r["pubs"]), "n_masking_rows": len(r["masking"])}
            if res["smiles"]:
                d = L.describe_structure(res["smiles"], clf)
                rec.update({k: v for k, v in d.items() if k != "morgan_r2_2048_onbits"})
                rec["n_structures_same_parent"] = len(parent_groups.get(d.get("parent_connectivity_key"), ()))
            else:
                rec.update({"parse_ok": False, "family": res["family"], "core_family": res["core_family"],
                            "family_status": "NAME_ONLY_NO_STRUCTURE", "acidity_class": "UNKNOWN",
                            "family_rules_version": clf.version})
            rec.update(M.component_mechanism(rec["family"], rec.get("core_family")))
            rec.update({"family_structural": rec["family"], "core_family_structural": rec.get("core_family"),
                        "mechanism_structural": rec["mechanism"], "acidity_class_structural": rec.get("acidity_class"),
                        "family_rules_status": rec.get("family_status"), "name_structure_conflict": False})
            rec["known_name_override"] = ""
            rec["pka_value"] = None
            rec["pka_status"] = "NOT_CURATED: no literature pKa entered (never fabricated)"
            rec["flags"] = "NO_ARCHIVE_STRUCTURE|" + ("STRUCTURE_INFERRED_FROM_NAME" if res["smiles"] else
                                                       "NO_STRUCTURE")
            comp_rows.append(rec)

        comp_rows.append({
            "component_id": L.component_id(None, None), "record_type": "EMPTY_SLOT", "canonical_name": None,
            "canonical_name_status": "NO_NAME", "structure_basis": "NONE",
            "roles_seen": _counter_str(empty_slot["roles"]),
            "primary_role": empty_slot["roles"].most_common(1)[0][0] if empty_slot["roles"] else None,
            "n_entries": sum(empty_slot["roles"].values()), "n_rows": len(empty_slot["rows"]),
            "n_model_rows": len(empty_slot["model"]), "n_publications": len(empty_slot["pubs"]),
            "n_masking_rows": len(empty_slot["masking"]), "parse_ok": False, "family": "none",
            "core_family": "none", "family_status": "EMPTY_SLOT", "acidity_class": "UNKNOWN",
            "mechanism": "UNKNOWN", "mechanism_basis": "EMPTY_SLOT", "is_hydrophilic_agent": False,
            "is_electrolyte_salt": False,
            "flags": "EMPTY_SLOT|concentration_M=" + _counter_str(empty_slot["conc"], ","),
        })
        comp_df = pd.DataFrame(comp_rows)
        front = ["component_id", "record_type", "canonical_name", "canonical_name_status", "family", "core_family",
                 "family_status", "family_rules_status", "name_structure_conflict", "family_candidates", "mechanism",
                 "mechanism_basis", "acidity_class", "family_structural", "core_family_structural", "mechanism_structural",
                 "acidity_class_structural",
                 "is_hydrophilic_agent", "is_electrolyte_salt", "smiles_canonical", "resolved_smiles", "structure_basis"]
        comp_df = comp_df[front + [c for c in comp_df.columns if c not in front and c != "morgan_r2_2048_onbits"]
                          + ["morgan_r2_2048_onbits"]]
        write_csv(comp_df, OUT_COMPONENTS)

        # ---------------- row-level resolution ----------------
        def entry(res: dict, role: str) -> M.Entry:
            fam, core = res["family"], res["core_family"]
            if res["basis"] == "KNOWN_NAME_OVERRIDE":
                mech, basis = M.family_mechanism(fam)
                cm = {"mechanism": mech, "mechanism_basis": "KNOWN_NAME_OVERRIDE:" + basis, "is_hydrophilic_agent": False}
            else:
                cm = M.component_mechanism(fam, core)
            return M.Entry(key=res["key"], mechanism=cm["mechanism"], family=core or fam,
                           is_hydrophilic_agent=bool(cm["is_hydrophilic_agent"]), role=role, basis=cm["mechanism_basis"])

        row_mech = [None] * n                 # corrected, acidic modifiers promoted (per-row label)
        row_mech_nomod = [None] * n           # corrected, organic extractants only (system-table vote)
        row_family_corr = [None] * n          # corrected family of the organic extractants
        row_comp_fams = [None] * n            # corrected core families of every extractant-slot component
        # corrected (family, core_family) of each organic-extractant structure recorded in the row, by SMILES
        row_comp_res: list[dict[str, list[tuple[str, str]]]] = [{} for _ in range(n)]
        row_coext: list[tuple[str, ...]] = [()] * n   # promoted acidic co-extractant modifiers
        row_mech_struct_key = [None] * n
        row_family = [None] * n
        row_flags = [()] * n
        row_has_mod = np.zeros(n, dtype=bool)
        row_has_aq = np.zeros(n, dtype=bool)
        row_has_hydrophilic = np.zeros(n, dtype=bool)
        row_conflicts: list[list[str]] = [[] for _ in range(n)]
        row_effective_basis: list[Counter] = [Counter() for _ in range(n)]
        for i, comps in enumerate(comps_col):
            struct_ext, corr_ext, corr_mod, coext = [], [], [], []
            row_structs = [s for s in (_s(c.get("smiles_canonical")) for c in comps) if s is not None]
            for c in comps:
                role, nm, smi = c.get("role"), _s(c.get("name")), _s(c.get("smiles_canonical"))
                if role == "organic_extractant":
                    rs = L.resolve_occurrence(idx, nm, smi, clf, name_corrected=False)
                    rc = L.resolve_occurrence(idx, nm, smi, clf, name_corrected=True,
                                              row_structures=[s for s in row_structs if s != smi])
                    if rs is not None:
                        struct_ext.append(entry(rs, role))
                    if rc is not None:
                        corr_ext.append(entry(rc, role))
                        row_effective_basis[i][rc["basis"]] += 1
                        if smi is not None:
                            row_comp_res[i].setdefault(smi, []).append((str(rc["family"]), str(rc["core_family"])))
                    if nm is not None and smi is not None:
                        reason = idx.rejection_reason(nm, smi)
                        if reason:
                            row_conflicts[i].append(f"{nm}@{L.component_id(smi)}({idx.canonical_name(smi)}):{reason}")
                elif role == "phase_modifier":
                    if nm is not None or smi is not None:
                        row_has_mod[i] = True
                        rc = L.resolve_occurrence(idx, nm, smi, clf, name_corrected=True)
                        if rc is not None:
                            corr_mod.append(entry(rc, role))
                            if corr_mod[-1].mechanism in M.ACIDIC_SET:
                                coext.append(nm or idx.canonical_name(smi) or smi)
                elif role in L.AQUEOUS_ROLES:
                    if nm is not None or smi is not None:
                        row_has_aq[i] = True
                        if smi is not None and desc_cache.get(smi, {}).get("family") == "hydrophilic_aqueous_agent":
                            row_has_hydrophilic[i] = True
            key_level = M.system_mechanism(struct_ext, ())
            row_mech_struct_key[i] = key_level.mechanism
            row_family[i] = key_level.system_family
            corr = M.system_mechanism(corr_ext, corr_mod)
            row_mech[i] = corr.mechanism
            row_flags[i] = corr.flags
            nomod = M.system_mechanism(corr_ext, ())
            row_mech_nomod[i] = nomod.mechanism
            row_family_corr[i] = nomod.system_family
            row_comp_fams[i] = "|".join(sorted({e.family for e in corr_ext}))
            row_coext[i] = tuple(sorted(set(coext)))
            if "HYDROPHILIC_AGENT_IN_EXTRACTANT_SLOT" in corr.flags:
                row_has_aq[i] = True
                row_has_hydrophilic[i] = True

        # ---------------- systems table ----------------
        by_key = defaultdict(list)
        for i, k in enumerate(keys):
            by_key[k].append(i)
        prim_smiles = df["extractant_primary_smiles"].to_numpy()
        names_col = [tuple(str(x) for x in (v if v is not None else ())) for v in df["extractant_names"]]
        cls_col = df["system_component_class"].to_numpy()
        sys_rows = []
        res_comp: dict[tuple[str, str], tuple[str, str]] = {}   # (system key, structure) -> name-corrected (family, core)
        for k in sorted(by_key, key=lambda kk: (-len(by_key[kk]), kk or "")):
            rows = by_key[k]
            model_rows = [i for i in rows if tier[i] == "MODEL"]
            smiles_list = k.split("|") if k else []
            uniq_smiles = sorted(set(smiles_list))
            fams = [desc_cache[s]["family"] for s in uniq_smiles]
            cores = [desc_cache[s]["core_family"] for s in uniq_smiles]
            # the same components with the builder's name corrections (known name override, name-implied
            # structure), voted over the system's rows; a structure never corrected keeps its structural family
            fams_res, cores_res = [], []
            for s, f0, c0 in zip(uniq_smiles, fams, cores):
                fv, cv = Counter(), Counter()
                for i in rows:
                    for fr_, cr_ in row_comp_res[i].get(s, ()):
                        fv[fr_] += 1
                        cv[cr_] += 1
                fr_s = M.vote(fv)[0] if fv else f0
                cr_s = M.vote(cv)[0] if cv else c0
                fams_res.append(fr_s)
                cores_res.append(cr_s)
                res_comp[(k, s)] = (fr_s, cr_s)
            prim = Counter(_s(prim_smiles[i]) for i in rows).most_common(1)[0][0]
            vote_rows = model_rows or rows
            mech, votes = M.vote(Counter(row_mech_nomod[i] for i in vote_rows))
            _, votes_incl = M.vote(Counter(row_mech[i] for i in vote_rows))
            fam_corr, fam_corr_votes = M.vote(Counter(row_family_corr[i] for i in rows))
            comp_fams_resolved, _ = M.vote(Counter(row_comp_fams[i] for i in rows))
            coext_rows = [i for i in rows if row_coext[i]]
            coext_model = [i for i in coext_rows if tier[i] == "MODEL"]
            mech_coext = M.vote(Counter(row_mech[i] for i in (coext_model or coext_rows)))[0] if coext_rows else None
            conflict_ids = [L.component_id(s) for s in uniq_smiles if desc_cache[s].get("_name_structure_conflict")]
            fam_key = Counter(row_family[i] for i in rows)
            mech_struct = Counter(row_mech_struct_key[i] for i in rows)
            assert len(fam_key) == 1 and len(mech_struct) == 1, (k, fam_key, mech_struct)
            flag_c = Counter(f for i in rows for f in row_flags[i])
            conflicts = Counter(cf for i in rows for cf in row_conflicts[i])
            basis_c = Counter()
            for i in rows:
                basis_c.update(row_effective_basis[i])
            cat_c = Counter(category[i] for i in model_rows)
            rec = {
                "system_id": system_id(k), "extractant_system_key": k if k is not None else NULL_KEY_LABEL,
                "is_null_key": k is None, "n_organic_extractants": len(smiles_list),
                "component_ids": _join((L.component_id(s) for s in smiles_list), "|"),
                "component_canonical_names": _join((idx.canonical_name(s) for s in smiles_list), "|"),
                "component_families": _join(fams_res, "|"), "component_core_families": _join(cores_res, "|"),
                "component_families_structural": _join(fams, "|"), "component_core_families_structural": _join(cores, "|"),
                "primary_extractant_smiles": prim,
                "primary_extractant_canonical_name": idx.canonical_name(prim) if prim else None,
                "primary_extractant_family": desc_cache[prim]["family"] if prim else None,
                "system_family": fam_corr, "system_family_row_votes": fam_corr_votes,
                "system_family_structural": next(iter(fam_key)),
                "component_core_families_resolved": comp_fams_resolved,
                "mechanism": mech, "mechanism_row_votes": votes,
                "mechanism_row_votes_incl_acidic_coextractant": votes_incl,
                "mechanism_structural": next(iter(mech_struct)),
                "n_rows_acidic_coextractant_modifier": len(coext_rows),
                "n_model_rows_acidic_coextractant_modifier": len(coext_model),
                "acidic_coextractant_modifiers": _counter_str(Counter(x for i in coext_rows for x in row_coext[i])),
                "mechanism_with_acidic_coextractant": mech_coext,
                "has_name_structure_conflict": bool(conflict_ids),
                "name_structure_conflict_component_ids": _join(conflict_ids, "|"),
                "mechanism_flags": _counter_str(flag_c),
                "effective_component_basis": _counter_str(basis_c),
                "has_phase_modifier": bool(row_has_mod[rows].any()),
                "n_rows_with_phase_modifier": int(row_has_mod[rows].sum()),
                "has_aqueous_agent": bool(row_has_aq[rows].any()),
                "n_rows_with_aqueous_agent": int(row_has_aq[rows].sum()),
                "n_rows_with_hydrophilic_complexant": int(row_has_hydrophilic[rows].sum()),
                "n_rows": len(rows),
            }
            for t in TIERS:
                rec[f"n_rows_{t}"] = sum(1 for i in rows if tier[i] == t)
            for cat in METAL_CATEGORIES:
                rec[f"n_model_rows_{cat}"] = cat_c.get(cat, 0)
            mm = [metal[i] for i in model_rows]
            rec.update({
                "n_metals_model": len(set(mm)), "metals_model": metal_list(mm),
                "n_ln_model": len({metal[i] for i in model_rows if category[i] == "lanthanide"}),
                "n_an_model": len({metal[i] for i in model_rows if category[i] == "actinide"}),
                "n_metals_all": len({metal[i] for i in rows if isinstance(metal[i], str)}),
                "n_publications_all": len({pub[i] for i in rows}),
                "n_publications_model": len({pub[i] for i in model_rows}),
                "n_studies_model": len({study[i] for i in model_rows}),
                "n_masking_rows": int(masking[rows].sum()),
                "system_component_classes": _counter_str(Counter(cls_col[i] for i in rows)),
                "names_seen": _counter_str(Counter(" + ".join(names_col[i]) or "<none>" for i in rows)),
                "conflicting_component_names_seen": _counter_str(conflicts),
                "n_rows_with_name_conflict": sum(1 for i in rows if row_conflicts[i]),
            })
            sys_rows.append(rec)
        sys_df = pd.DataFrame(sys_rows)
        write_csv(sys_df, OUT_SYSTEMS)

        # ---------------- alias collisions ----------------
        col_rows = []

        def name_aq_evidence(nm: str, smi: str | None) -> list[str]:
            ev = []
            if smi is not None and idx.is_masking_collision(nm, smi):
                ev.append("MASKING_AGENT_NAME_ON_THIS_STRUCTURE")
            if any(role in L.AQUEOUS_ROLES for role in name_roles[nm]):
                ev.append("NAME_USED_IN_AQUEOUS_ROLE")
            own = idx.owner(nm)
            if own is not None and desc_cache.get(own, {}).get("family") in ("hydrophilic_aqueous_agent",
                                                                             "electrolyte_salt"):
                ev.append("NAME_OWNER_IS_HYDROPHILIC_AGENT_OR_SALT")
            return ev

        def occ_record(ctype, group, nm, smi, n_members):
            o = idx.occ[(nm, smi)]
            ev = name_aq_evidence(nm, smi)
            own = idx.owner(nm)
            return {"collision_type": ctype, "collision_group": group, "n_group_members": n_members,
                    "name": nm, "name_key": L.normalize_name_key(nm), "component_id": L.component_id(smi),
                    "smiles_canonical": smi, "structure_canonical_name": idx.canonical_name(smi),
                    "structure_family": desc_cache[smi]["family"],
                    "n_rows": len(o.n_rows), "n_model_rows": len(o.n_model_rows), "n_masking_rows": len(o.masking_rows),
                    "roles": _counter_str(o.roles), "accepted_as_alias": idx.accepted(nm, smi),
                    "rejection_reason": idx.rejection_reason(nm, smi),
                    "name_owner_component_id": L.component_id(own) if own else None,
                    "name_owner_canonical_name": idx.canonical_name(own) if own else None,
                    "colliding_name_is_aqueous_agent": bool(ev), "aqueous_agent_evidence": _join(ev, "|")}

        for nm in sorted(idx.by_name):
            occs = [o for o in idx.by_name[nm] if o.smiles is not None]
            if len({o.smiles for o in occs}) > 1:
                for o in sorted(occs, key=lambda o: (-len(o.n_rows), o.smiles)):
                    col_rows.append(occ_record("NAME_TO_MULTIPLE_STRUCTURES", nm, nm, o.smiles, len(occs)))
        for smi in sorted(idx.by_smiles):
            occs = [o for o in idx.by_smiles[smi] if o.name is not None]
            if len(occs) > 1:
                grp = idx.canonical_name(smi) or smi
                for o in sorted(occs, key=lambda o: (-len(o.n_rows), o.name)):
                    col_rows.append(occ_record("STRUCTURE_TO_MULTIPLE_NAMES", f"{L.component_id(smi)}:{grp}",
                                               o.name, smi, len(occs)))
        key_groups = defaultdict(set)
        for nm in list(idx.by_name) + list(name_only):
            key_groups[L.normalize_name_key(nm)].add(nm)
        for kname, spellings in sorted(key_groups.items()):
            if len(spellings) < 2:
                continue
            owners = {sp: idx.owner(sp) for sp in spellings}
            same = len({v for v in owners.values()}) == 1 and None not in owners.values()
            for sp in sorted(spellings):
                occs = idx.by_name.get(sp, [])
                col_rows.append({
                    "collision_type": "NAME_SPELLING_VARIANTS", "collision_group": kname,
                    "n_group_members": len(spellings), "name": sp, "name_key": kname,
                    "component_id": L.component_id(owners[sp]) if owners[sp] else None,
                    "smiles_canonical": owners[sp], "structure_canonical_name": idx.canonical_name(owners[sp]) if owners[sp] else None,
                    "structure_family": desc_cache[owners[sp]]["family"] if owners[sp] else None,
                    "n_rows": len(set().union(*[o.n_rows for o in occs])) if occs else len(name_only[sp]["rows"]),
                    "n_model_rows": len(set().union(*[o.n_model_rows for o in occs])) if occs else len(name_only[sp]["model"]),
                    "accepted_as_alias": same,
                    "rejection_reason": None if same else "SPELLING_VARIANTS_POINT_TO_DIFFERENT_OR_NO_STRUCTURES",
                    "colliding_name_is_aqueous_agent": bool(name_aq_evidence(sp, None)),
                    "aqueous_agent_evidence": _join(name_aq_evidence(sp, None), "|"),
                    "detail": "merge allowed (same owner structure)" if same else "merge blocked"})
        for (raw, can), cnt in sorted(raw_pairs.items()):
            rc = L.canonical_smiles(raw)
            if rc == can:
                continue
            if rc is None:
                kind = "RAW_UNPARSEABLE"
            else:
                mr, mc = L.mol_from_smiles(rc), L.mol_from_smiles(can)
                if L.parent_key(rc) != L.parent_key(can):
                    kind = "DIFFERENT_CONNECTIVITY"
                elif len(L.Chem.GetMolFrags(mr)) != len(L.Chem.GetMolFrags(mc)):
                    kind = "SALT_OR_FRAGMENT_DIFFERS"
                elif sum(a.GetFormalCharge() for a in mr.GetAtoms()) != sum(a.GetFormalCharge() for a in mc.GetAtoms()):
                    kind = "CHARGE_DIFFERS"
                else:
                    kind = "STEREO_OR_TAUTOMER_DIFFERS"
            col_rows.append({"collision_type": "RAW_VS_CANONICAL_SMILES", "collision_group": L.component_id(can),
                             "n_group_members": 1, "component_id": L.component_id(can), "smiles_canonical": can,
                             "structure_canonical_name": idx.canonical_name(can),
                             "structure_family": desc_cache[can]["family"], "n_rows": cnt,
                             "detail": f"{kind}; smiles_raw={raw}; rdkit_canonical(raw)={rc}"})
        for pk, members in sorted(parent_groups.items(), key=lambda kv: str(kv[0])):
            if len(members) < 2:
                continue
            for smi in sorted(members):
                col_rows.append({"collision_type": "SAME_PARENT_DIFFERENT_CANONICAL_SMILES", "collision_group": pk,
                                 "n_group_members": len(members), "component_id": L.component_id(smi),
                                 "smiles_canonical": smi, "structure_canonical_name": idx.canonical_name(smi),
                                 "structure_family": desc_cache[smi]["family"], "n_rows": len(st[smi]["rows"]),
                                 "n_model_rows": len(st[smi]["model"]),
                                 "detail": "same neutralised, stereo-free largest-fragment InChIKey block; kept apart by the archive"})
        n_raw_pairs_disagree = sum(1 for r in col_rows if r["collision_type"] == "RAW_VS_CANONICAL_SMILES")
        type_counts = Counter(r["collision_type"] for r in col_rows)
        summary_detail = {
            "NAME_TO_MULTIPLE_STRUCTURES": f"names checked={len(idx.by_name)}",
            "STRUCTURE_TO_MULTIPLE_NAMES": f"structures checked={len(idx.by_smiles)}",
            "NAME_SPELLING_VARIANTS": f"normalised name keys checked={len(key_groups)}",
            "RAW_VS_CANONICAL_SMILES": (f"distinct (smiles_raw, smiles_canonical) pairs checked={len(raw_pairs)}; "
                                        f"RDKit canonical(raw) != archive canonical in {n_raw_pairs_disagree}"),
            "SAME_PARENT_DIFFERENT_CANONICAL_SMILES": f"parent connectivity keys checked={len(parent_groups)}",
        }
        for ctype, det in reversed(list(summary_detail.items())):
            col_rows.insert(0, {"collision_type": "CHECK_SUMMARY", "collision_group": ctype,
                                "n_group_members": type_counts.get(ctype, 0),
                                "detail": f"{det}; member rows written={type_counts.get(ctype, 0)}"})
        col_df = pd.DataFrame(col_rows)
        col_cols = ["collision_type", "collision_group", "n_group_members", "name", "name_key", "component_id",
                    "smiles_canonical", "structure_canonical_name", "structure_family", "n_rows", "n_model_rows",
                    "n_masking_rows", "roles", "accepted_as_alias", "rejection_reason", "name_owner_component_id",
                    "name_owner_canonical_name", "colliding_name_is_aqueous_agent", "aqueous_agent_evidence", "detail"]
        col_df = col_df.reindex(columns=col_cols)
        write_csv(col_df, OUT_COLLISIONS)

        # ---------------- family / mechanism coverage ----------------
        cov_rows = []

        def coverage(level: str, label: str, rows: list[int], structures: set[str]) -> dict:
            model_rows = [i for i in rows if tier[i] == "MODEL"]
            sys_metals = defaultdict(set)
            for i in model_rows:
                sys_metals[keys[i]].add(metal[i])
            wide = sorted((kk for kk, ms in sys_metals.items() if len(ms) >= 5), key=lambda kk: -len(sys_metals[kk]))
            cat_c = Counter(category[i] for i in model_rows)
            rec = {"level": level, "label": label,
                   "n_systems": len({keys[i] for i in rows}), "n_systems_model": len(sys_metals),
                   "n_component_structures": len(structures), "n_rows_all": len(rows), "n_rows_model": len(model_rows),
                   "n_metals_model": len({metal[i] for i in model_rows}),
                   "n_ln_model": len({metal[i] for i in model_rows if category[i] == "lanthanide"}),
                   "n_an_model": len({metal[i] for i in model_rows if category[i] == "actinide"}),
                   "n_publications_model": len({pub[i] for i in model_rows}),
                   "n_studies_model": len({study[i] for i in model_rows})}
            for cat in METAL_CATEGORIES:
                rec[f"n_model_rows_{cat}"] = cat_c.get(cat, 0)
            rec["n_systems_ge5_metals_model"] = len(wide)
            rec["systems_ge5_metals_model"] = _join((f"{idx.canonical_name(kk.split('|')[0]) if kk else NULL_KEY_LABEL}"
                                                     + (f"+{len(kk.split('|')) - 1}" if kk and '|' in kk else "")
                                                     + f"({len(sys_metals[kk])})" for kk in wide), "; ")
            rec["metals_model"] = metal_list(metal[i] for i in model_rows)
            return rec

        # levels system_family / component_family carry the builder's name corrections (per row, organic
        # extractants: row_family_corr); the *_structural levels are the structure-only labels (row_family)
        fam_rows, fam_rows_struct = defaultdict(list), defaultdict(list)
        comp_fam_rows, comp_fam_rows_struct = defaultdict(list), defaultdict(list)
        mech_rows = defaultdict(list)
        for i in range(n):
            fam_rows[row_family_corr[i]].append(i)
            fam_rows_struct[row_family[i]].append(i)
            mech_rows[row_mech[i]].append(i)
            if keys[i] is not None:
                for f in set(row_family_corr[i].split("+")):
                    comp_fam_rows[f].append(i)
                for f in set(row_family[i].split("+")):
                    comp_fam_rows_struct[f].append(i)
        struct_of_key = {k: set(k.split("|")) if k else set() for k in by_key}

        def structs(rows, family=None, corrected=True):
            out = set()
            for kk in {keys[i] for i in rows}:
                for s in struct_of_key[kk]:
                    fam_s, core_s = res_comp[(kk, s)] if corrected else (desc_cache[s]["family"], desc_cache[s]["core_family"])
                    if family is None or core_s == family or fam_s == family:
                        out.add(s)
            return out

        def by_size(d):
            return sorted(d, key=lambda x: (-len(d[x]), str(x)))

        for f in by_size(fam_rows):
            cov_rows.append(coverage("system_family", f, fam_rows[f], structs(fam_rows[f])))
        for f in by_size(comp_fam_rows):
            cov_rows.append(coverage("component_family", f, comp_fam_rows[f], structs(comp_fam_rows[f], f)))
        for mch in by_size(mech_rows):
            cov_rows.append(coverage("mechanism", mch, mech_rows[mch], structs(mech_rows[mch])))
        for f in by_size(fam_rows_struct):
            cov_rows.append(coverage("system_family_structural", f, fam_rows_struct[f],
                                     structs(fam_rows_struct[f], corrected=False)))
        for f in by_size(comp_fam_rows_struct):
            cov_rows.append(coverage("component_family_structural", f, comp_fam_rows_struct[f],
                                     structs(comp_fam_rows_struct[f], f, corrected=False)))
        cov_df = pd.DataFrame(cov_rows)
        write_csv(cov_df, OUT_COVERAGE)

        # ---------------- named extractant presence ----------------
        pres_df = presence_check(df, comps_col, tier, metal, exp_ids, raw_files, idx, clf, desc_cache, metal_list)
        write_csv(pres_df, OUT_PRESENCE)

        run.outputs(OUT_RULES, OUT_COMPONENTS, OUT_SYSTEMS, OUT_COLLISIONS, OUT_COVERAGE, OUT_PRESENCE)
        run.extra["summary"] = {
            "n_structures": len(st), "n_name_only": len(name_only), "n_systems_incl_null": len(by_key),
            "family_status_counts": dict(Counter(desc_cache[s]["family_status"] for s in st)),
            "family_rules_status_counts": dict(Counter(desc_cache[s]["family_status"] for s in st)),
            "component_family_status_counts": dict(Counter(comp_df.loc[comp_df["record_type"] == "STRUCTURE", "family_status"])),
            "name_structure_conflict_structures": sorted(
                f"{r.component_id}:{r.canonical_name}" for r in comp_df[comp_df["name_structure_conflict"] == True].itertuples()),  # noqa: E712
            "n_model_rows_acidic_coextractant_modifier": int(sys_df["n_model_rows_acidic_coextractant_modifier"].sum()),
            "structure_family_counts": dict(Counter(desc_cache[s]["family"] for s in st)),
            "n_collision_rows": dict(Counter(col_df["collision_type"])),
        }
        print(f"structures={len(st)} name_only={len(name_only)} systems={len(by_key)} "
              f"collision_rows={len(col_df)} coverage_rows={len(cov_df)} presence_rows={len(pres_df)}")


# ----------------------------------------------------------------------------------------------
def presence_check(df, comps_col, tier, metal, exp_ids, raw_files, idx, clf, desc_cache, metal_list) -> pd.DataFrame:
    n = len(df)
    row_of_exp = {e: i for i, e in enumerate(exp_ids)}
    regs = {t: term_regex(t) for t in PRESENCE_NAME_TERMS}
    hits: dict[str, dict] = {t: {"rows": set(), "where": Counter(), "matched": Counter(), "raw_lines": 0,
                                 "raw_files": set(), "unmapped_raw": 0} for t in PRESENCE_NAME_TERMS}

    # master columns
    text_cols = {c: df[c].to_numpy() for c in MASTER_TEXT_COLUMNS}
    for i in range(n):
        comp_text = " \x1f ".join(f"{_s(c.get('name')) or ''} {_s(c.get('smiles_raw')) or ''} "
                                  f"{_s(c.get('smiles_canonical')) or ''}" for c in comps_col[i])
        cells = {f"master:{c}": _s(v[i]) or "" for c, v in text_cols.items()}
        cells["master:components"] = comp_text
        joined = " \x1e ".join(cells.values())
        for t, rg in regs.items():
            if rg.search(joined):
                h = hits[t]
                h["rows"].add(i)
                for where, txt in cells.items():
                    for m in rg.finditer(txt):
                        h["where"][where] += 1
                        h["matched"][_context(txt, m)] += 1

    # raw CSVs, every column
    raw_tokens: dict[str, set] = defaultdict(set)   # smiles token -> exp_ids
    prefilter = re.compile("|".join(f"(?:{rg.pattern})" for rg in regs.values()), re.IGNORECASE)
    for f in raw_files:
        with open(f, encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            ci_exp = header.index("exp_id")
            ci_smi = header.index("Extractant_SMILES")
            ci_com = header.index("comments_description")
            for rowv in reader:
                if not rowv:
                    continue
                joined = " \x1e ".join(rowv)
                e = rowv[ci_exp]
                for tok in str(rowv[ci_smi]).split(","):
                    if tok.strip() and tok.strip() not in ("-", "nan"):
                        raw_tokens[tok.strip()].add(e)
                for m in re.finditer(r"SMILES:\s*([^;\s]+)", rowv[ci_com]):
                    if m.group(1) not in ("nan", "-"):
                        raw_tokens[m.group(1)].add(e)
                if not prefilter.search(joined):
                    continue
                for t, rg in regs.items():
                    if rg.search(joined):
                        h = hits[t]
                        h["raw_lines"] += 1
                        h["raw_files"].add(f.name)
                        if e in row_of_exp:
                            h["rows"].add(row_of_exp[e])
                        else:
                            h["unmapped_raw"] += 1
                        for ci, cell in enumerate(rowv):
                            if rg.search(cell):
                                h["where"][f"raw:{header[ci]}"] += 1

    out = []
    for t in PRESENCE_NAME_TERMS:
        h = hits[t]
        rows = sorted(h["rows"])
        comp_structs = Counter()
        row_structs = Counter()
        comp_roles = Counter()
        rg = regs[t]
        for i in rows:
            matched_comp = False
            for c in comps_col[i]:
                nm = _s(c.get("name"))
                if nm is not None and rg.search(nm):
                    matched_comp = True
                    comp_roles[str(c.get("role"))] += 1
                    smi = _s(c.get("smiles_canonical"))
                    if smi:
                        comp_structs[f"{nm}=>{smi} [{desc_cache[smi]['family']}; canonical_name={idx.canonical_name(smi)}]"] += 1
                    else:
                        comp_structs[f"{nm}=>NO_SMILES_IN_ARCHIVE (role {c.get('role')})"] += 1
            if not matched_comp:
                for c in comps_col[i]:
                    smi = _s(c.get("smiles_canonical"))
                    if smi and c.get("role") == "organic_extractant":
                        row_structs[f"{idx.canonical_name(smi)} [{desc_cache[smi]['family']}]"] += 1
        fams = {s.split("[")[1].split(";")[0] for s in comp_structs if "=>NO_SMILES" not in s}
        ref = next((r for r, v in L.REFERENCE_EXTRACTANTS.items() if t in v["aliases"]
                    or t.replace(" ", "").lower() in [a.replace(" ", "").lower() for a in v["aliases"]]), None)
        expected = L.REFERENCE_EXTRACTANTS[ref]["expected_family"] if ref else None
        ref_smi = L.canonical_smiles(L.REFERENCE_EXTRACTANTS[ref]["smiles"]) if ref else None
        ref_rows = [i for i in range(n) if ref_smi and any(_s(c.get("smiles_canonical")) == ref_smi
                                                            for c in comps_col[i])]
        if not rows:
            verdict = "ABSENT"
        elif comp_structs and expected and expected in fams:
            verdict = "PRESENT_AS_EXPECTED_STRUCTURE_FAMILY"
        elif comp_structs and fams and expected and expected not in fams:
            verdict = "NAME_TRAP_DIFFERENT_STRUCTURE_FAMILY"
        elif comp_structs and not fams:
            verdict = "NAME_ONLY_NO_STRUCTURE"
        elif comp_structs:
            verdict = "PRESENT_AS_STRUCTURE"
        else:
            verdict = "TEXT_MENTION_ONLY"
        out.append({
            "term": t, "term_kind": "NAME_TEXT", "reference_extractant": ref, "expected_family": expected,
            "verdict": verdict, "where_found": _counter_str(h["where"]), "n_rows": len(rows),
            "n_model_rows": sum(1 for i in rows if tier[i] == "MODEL"),
            "metals_all": metal_list(metal[i] for i in rows if isinstance(metal[i], str)),
            "metals_model": metal_list(metal[i] for i in rows if tier[i] == "MODEL"),
            "n_publications": len({df["g19_publication_id"].iat[i] for i in rows}),
            "reference_smiles": ref_smi,
            "n_rows_with_reference_structure": len(ref_rows),
            "n_model_rows_with_reference_structure": sum(1 for i in ref_rows if tier[i] == "MODEL"),
            "metals_model_reference_structure": metal_list(metal[i] for i in ref_rows if tier[i] == "MODEL"),
            "n_raw_lines": h["raw_lines"], "raw_files": _join(sorted(h["raw_files"]), "|"),
            "n_raw_lines_unmapped_to_master": h["unmapped_raw"],
            "structures_on_matching_components": _counter_str(comp_structs),
            "roles_on_matching_components": _counter_str(comp_roles),
            "row_extractant_structures_when_text_only": _counter_str(row_structs),
            "matched_text_examples": _join([k for k, _ in h["matched"].most_common(6)], " || "),
        })

    # full chemical names and publication / reference titles (brief section 19 acids): a paper can name the
    # solvent only in its title while the row records a different structure in the extractant slot
    title_cols = {c: df[c].to_numpy() for c in TITLE_TEXT_COLUMNS}
    text_cols_all = {**{c: df[c].to_numpy() for c in MASTER_TEXT_COLUMNS}, **title_cols}
    for ref, patterns in FULL_NAME_PATTERNS.items():
        aliases = [re.escape(a) for a in L.REFERENCE_EXTRACTANTS[ref]["aliases"] if a != "DEHPA"]
        name_rx = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(aliases) + r")(?![A-Za-z0-9])", re.IGNORECASE)
        full_rx = re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)
        rows, where, examples, structs, structs_model = [], Counter(), Counter(), Counter(), Counter()
        for i in range(n):
            hit = False
            for col, vals in text_cols_all.items():
                txt = _s(vals[i]) or ""
                if not txt:
                    continue
                m_full = full_rx.search(txt)
                m_name = name_rx.search(txt) if col in title_cols else None
                if m_full or m_name:
                    hit = True
                    where[f"master:{col}"] += 1
                    examples[_context(txt, m_full or m_name, 60)] += 1
            if hit:
                rows.append(i)
                names = [idx.canonical_name(_s(c.get("smiles_canonical"))) or _s(c.get("name")) or "?"
                         for c in comps_col[i] if c.get("role") == "organic_extractant"]
                structs[" + ".join(sorted(str(x) for x in names)) or "<none>"] += 1
                if tier[i] == "MODEL":
                    structs_model[" + ".join(sorted(str(x) for x in names)) or "<none>"] += 1
        out.append({
            "term": f"{ref}:FULL_NAME_OR_TITLE_TEXT", "term_kind": "FULL_NAME_OR_TITLE_TEXT", "reference_extractant": ref,
            "expected_family": L.REFERENCE_EXTRACTANTS[ref]["expected_family"],
            "verdict": "TITLE_OR_FULL_NAME_MENTION_ROWS" if rows else "ABSENT",
            "where_found": _counter_str(where), "n_rows": len(rows),
            "n_model_rows": sum(1 for i in rows if tier[i] == "MODEL"),
            "metals_all": metal_list(metal[i] for i in rows if isinstance(metal[i], str)),
            "metals_model": metal_list(metal[i] for i in rows if tier[i] == "MODEL"),
            "n_publications": len({df["g19_publication_id"].iat[i] for i in rows}),
            "reference_smiles": L.canonical_smiles(L.REFERENCE_EXTRACTANTS[ref]["smiles"]),
            "n_raw_lines": None, "raw_files": "", "n_raw_lines_unmapped_to_master": None,
            "structures_on_matching_components": "",
            "row_extractant_structures_when_text_only": _counter_str(structs),
            "row_extractant_structures_model_rows": _counter_str(structs_model),
            "matched_text_examples": _join([k for k, _ in examples.most_common(4)], " || "),
        })

    # SMILES / family search
    master_tokens: dict[str, set] = defaultdict(set)
    for i, comps in enumerate(comps_col):
        for c in comps:
            for fld in ("smiles_raw", "smiles_canonical"):
                if _s(c.get(fld)):
                    master_tokens[str(c[fld])].add(i)
        raw = _s(df["extractant_smiles_raw"].iat[i])
        if raw:
            for tok in raw.split(","):
                if tok.strip():
                    master_tokens[tok.strip()].add(i)
    canon = {}
    for tok in set(master_tokens) | set(raw_tokens):
        cs = L.canonical_smiles(tok)
        canon[tok] = (cs, L.parent_key(cs) if cs else None, clf.classify(cs).family if cs else None)
    for ref in PRESENCE_SMILES_REFS:
        rs = L.canonical_smiles(L.REFERENCE_EXTRACTANTS[ref]["smiles"])
        rpk = L.parent_key(rs)
        rfam = clf.classify(rs).family
        for kind, test in (("SMILES_EXACT_CANONICAL", lambda c: c[0] == rs),
                           ("SMILES_CONNECTIVITY_KEY", lambda c: c[1] == rpk),
                           ("SUBSTRUCTURE_FAMILY", lambda c: c[2] == rfam)):
            rows, where, found = set(), Counter(), Counter()
            for tok, idxs in master_tokens.items():
                if canon[tok][0] and test(canon[tok]):
                    rows |= idxs
                    where["master:components/extractant_smiles_raw"] += len(idxs)
                    found[canon[tok][0]] += len(idxs)
            n_raw = 0
            unmapped = 0
            for tok, es in raw_tokens.items():
                if canon[tok][0] and test(canon[tok]):
                    n_raw += len(es)
                    where["raw:Extractant_SMILES/comments_description"] += len(es)
                    for e in es:
                        if e in row_of_exp:
                            rows.add(row_of_exp[e])
                        else:
                            unmapped += 1
            rows = sorted(rows)
            structs = _counter_str(Counter({f"{s} [{clf.classify(s).family}; canonical_name={idx.canonical_name(s)}]": c
                                            for s, c in found.items()}))
            out.append({
                "term": f"{ref}:{kind}", "term_kind": kind, "reference_extractant": ref, "expected_family": rfam,
                "verdict": ("ABSENT" if not rows else "PRESENT_AS_STRUCTURE"),
                "where_found": _counter_str(where), "n_rows": len(rows),
                "n_model_rows": sum(1 for i in rows if tier[i] == "MODEL"),
                "metals_all": metal_list(metal[i] for i in rows if isinstance(metal[i], str)),
                "metals_model": metal_list(metal[i] for i in rows if tier[i] == "MODEL"),
                "n_publications": len({df["g19_publication_id"].iat[i] for i in rows}),
                "reference_smiles": rs,
                "n_rows_with_reference_structure": len(rows) if kind == "SMILES_EXACT_CANONICAL" else None,
                "n_model_rows_with_reference_structure": (sum(1 for i in rows if tier[i] == "MODEL")
                                                          if kind == "SMILES_EXACT_CANONICAL" else None),
                "metals_model_reference_structure": None,
                "n_raw_lines": n_raw, "raw_files": "", "n_raw_lines_unmapped_to_master": unmapped,
                "structures_on_matching_components": structs,
                "row_extractant_structures_when_text_only": "",
                "matched_text_examples": f"reference SMILES {rs}; connectivity key {rpk}",
            })
    return pd.DataFrame(out)


def _context(txt: str, m: re.Match, width: int = 30) -> str:
    a, b = max(0, m.start() - width), min(len(txt), m.end() + width)
    return txt[a:b].replace("\x1f", "|").replace("\n", " ").strip()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = ap.parse_args(argv)
    build(args)


if __name__ == "__main__":
    main()

"""Label-blind supply ledger, PROVISIONAL laboratory key, frozen mechanism-class map and the supply-audit kill
(CLAUDE_CODE_PROMPT.md sections 4.3 items 1 and 7, 4.6, 5.5, 7.5; I2 confirm-1 ... confirm-4, lab-4; PC P4).

**Metadata only.**  Inputs are bibliographic / screening tables: probes/P4_candidates.csv, P4_corpus_registry.csv,
P4_corpus_surnames.csv (design folder) and round-4 results/provenance/sources_by_doi.csv.  ``read_meta`` asserts that
no D, log D or SF column is loaded.  Nothing here touches the bench values, a prediction or a score.

Laboratory key (PROVISIONAL until the user's U6 review; section 5.5):
* Author parsing of the registry's ``authors`` field (several representations joined by '; '): titles dropped;
  initials recognised ("E.A.", "SA", "ZY"); an all-upper token is the family name when it is last after a full
  given name or has >= 4 letters ("Yuji SASAKI"); **Family-Given order** ("Sun Guoxin"): a two-word entry is read
  family-first when its first word is a Chinese family name and its second is not, or (both possible) when the reversed
  person ("Guoxin Sun") is attested in another reference, or by the majority of the decided entries in the same
  representation; otherwise Given-Family.
* **Common-surname links dropped**: the four named in I2 lab-4 (liu, zhang, zhou, xu) plus P4's own common-surname
  list (P4_check_candidates.py ``COMMON``, which P4 applied in the probe) never count toward a >= 2-shared-surname
  link (DROP_L2).  Dropping only the four named surnames is a sensitivity: on the 135-reference registry it merges
  distinct Chinese groups through li / wang / yang / sun.
* Links between corpus references: L2 = >= 2 shared distinctive family names; **single-PI link** LPI = the same
  person (family + first initial; only the four named surnames excluded) in a PI position (first or last author of
  the reference's first ordered representation) of both references.  conservative lab = connected component of L2
  or LPI; liberal lab = the senior (last-listed) person of the reference, no transitive merging.
* Literature candidates: liberal lab = P4's hand ``lab_id`` (one PI group); conservative = P4's merge rule (CIAC
  groups, UFV / Aveiro, KU Leuven, ISSP-Turanov / INEOS-Odinets).  ``lab_in_corpus_frozen`` = P4's hand flag with the
  **institution-link (P) rule applied consistently** (PC P4: CSM-Shafer, Osaka-Shinohara and Kanazawa-Imura are P;
  P4-025 is Y by the stated two-co-author rule).  An automatic author-overlap flag with the repaired surname rule is
  reported beside it as evidence for U6.  Second-order links (Binnemans via P4-026, Kostikova via P4-001) are a
  sensitivity, not errors (PC P4).
* Eligibility as frozen by PC P4: P4-063 moves E1 -> E2; title-only "no abstract" E2 rows move to U (P4's tier rule
  as written).

Mechanism-class map (frozen, applied blind to values; ``MECH_RULES``): first matching rule on the chemotype label (then
the extractant text), after removing an "in <diluent>" clause (ionic-liquid diluent flag) and "aqueous <complexant>"
tokens (aqueous-complexant flag).  ``MEDIUM_RULES`` classify the aqueous medium for the Gd claims (nitrate and inert
salts eligible; every complexant, including acetate, decided now as excluded).  ``t_classes`` parses the sibling
design into T0 (pH, [HA] or phase ratio for a cation exchanger), T1 (aqueous medium), T2 (organic phase), other.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from . import paths

P4DIR = paths.DESIGN / "probes"
CAND = P4DIR / "P4_candidates.csv"
REG = P4DIR / "P4_corpus_registry.csv"
SURN = P4DIR / "P4_corpus_surnames.csv"
SRC_DOI = paths.R4RES / "provenance" / "sources_by_doi.csv"

FORBIDDEN_COL = re.compile(r"(^d$|^d_|_d$|^logd|^log_d|logd__|d_raw|d_value|^sf$|^sf_|_sf$|logsf|separation_factor"
                           r"|^y$|^b1$|^b0$)", re.I)
DROP = frozenset({"liu", "zhang", "zhou", "xu"})                  # I2 lab-4 named common-surname links
P4_COMMON = frozenset({  # P4_check_candidates.py COMMON (sensitivity)
    "li", "wang", "zhang", "liu", "chen", "yang", "zhao", "wu", "zhou", "xu", "sun", "ma", "zhu", "hu", "guo", "he",
    "gao", "lin", "luo", "zheng", "liang", "xie", "tang", "han", "cao", "deng", "feng", "huang", "ye", "lu", "cui",
    "jiang", "pan", "wei", "shi", "xiao", "tian", "dong", "yu", "yuan", "peng", "fang", "su", "ding", "jin", "qi",
    "suzuki", "sasaki", "tanaka", "yamamoto", "kobayashi", "ito", "kim", "lee", "park", "nguyen", "kumar", "singh",
    "reddy", "ali", "petrov", "ivanov", "smith", "wilson", "mohamed", "sharma", "xue", "bi", "yao", "ban", "kaneko"})
DROP_L2 = DROP | P4_COMMON           # surnames that never count toward a >= 2-shared-surname link (primary)
CJK_FAMILY = frozenset((
    "li wang zhang liu chen yang huang zhao wu zhou xu sun ma zhu hu guo he gao lin luo zheng liang xie song tang han "
    "feng deng cao peng zeng xiao tian dong yuan pan cai jiang yu du ye cheng su wei lu ding ren shen yao cui zhong tan "
    "fan jin shi liao jia xia fu fang bai zou meng xiong qin qiu yin xue yan duan lei hou long tao gu mao hao gong shao "
    "wan qian dai mo kong xiang chang wen kang qi zhan bi nie ning ou gan ji lan ai jiao rong mi ling niu bao huo hong "
    "che xin geng ge ke miao yi bian shang yue jing zuo zhai man xing quan an pu zhai").split())
TITLES = frozenset({"dr", "prof", "mr", "ms", "mrs", "phd", "jr", "sr", "professor"})
CONS_MERGE = (("CIAC", lambda s: s in {"CIAC-Li", "CIAC-Liao", "CIAC-Deng"}),
              ("UFV/Aveiro-Schaeffer", lambda s: s.startswith("UFV-") or s.startswith("Aveiro-")),
              ("KULeuven-Binnemans", lambda s: s.startswith("KULeuven")),
              ("ISSP-Turanov", lambda s: "Turanov" in s))
INSTITUTION_FIX = {  # PC P4: the institution-link (P) rule applied consistently; P4-025 by the stated Y rule
    "CSM-Shafer": ("P", "Argonne co-authors on P4-077 / P4-073 (Servis at ANL); CSM-Jensen already P [PC P4]"),
    "Osaka-Shinohara": ("P", "JAEA heavy-element co-authors (Nagame, Asai, Toyoshima); not confirmed from metadata [PC P4]"),
    "Kanazawa-Imura": ("P", "co-author Okamura at JAEA (P4's own note) [PC P4]"),
}
CAND_FIX = {"P4-025": ("lab_in_corpus", "Y", "two co-authors (Sasaki, Matsumiya) on archive 10.1246/cl.200431 [PC P4]"),
            "P4-063": ("eligibility", "E2", "adjacent pair not visible in the accessible text [PC P4]")}
SECOND_ORDER = {"KULeuven-Binnemans": "Binnemans co-authored P4-026 with the corpus Juelich lab",
                "Mendeleev/Frumkin-Stepanov": "Kostikova co-authored P4-001 with the corpus MSU phen-diamide lab"}
TOMANO3_MERGE = {"quaternary ammonium nitrate + TBP": "quaternary-onium nitrate IL"}   # [A336][NO3] = TOMANO3 [PC P4]

# ------------------------------------------------------------------------------------------------------------
# frozen mechanism-class map
# ------------------------------------------------------------------------------------------------------------
MECH_RULES = [  # (rule_id, pattern, class, in C-GD stratum, E-CT report class, note)
    ("X01", r"^[\s+/]*$|\breview\b", "NOT_LLE", "N", "excluded",
     "aqueous-only chelator or review: not liquid-liquid extraction data"),
    ("X02", r"eutectic", "OTHER_EUTECTIC", "N", "other",
     "hydrophobic eutectic solvents (TOPO + carboxylic acid): no single mechanism; outside the C-GD stratum"),
    ("X03", r"bifunctional|\[a336\]\[|anion[- ]bifunctional|dga-anion|organophosphorus-anion", "IL_BIFUNCTIONAL", "N",
     "ionic liquid", "bifunctional ionic liquids ([A336][P507], DGA-anion ILs): ion-association of the metal salt, no "
                     "proton release; ionic-liquid class, outside C-GD"),
    ("X04", r"quaternary|aliquat|tomano3|methyltrioctylammonium|\[a336\]|\[p66614\]|onium", "IL_QUATERNARY", "N",
     "ionic liquid", "quaternary-ammonium / phosphonium nitrate or thiocyanate (anion exchange); [A336][NO3] and "
                     "TOMANO3 are one compound and count once"),
    ("X05", r"beta-diketone|β-diketone|\bhtta\b|\btta\b|thenoyl|acylpyrazolone|pyrazolone|hpmbp|8-hq\b|"
            r"hydroxyquinolin|tropolone", "CX_CHELATING", "Y", "cation exchanger",
     "chelating cation exchangers; HTTA + TOPO and other chelator + neutral-donor synergists stay here (proton "
     "release, adduct formation); synergist flag = sensitivity"),
    ("X06", r"carboxylic acid|versatic|naphthenic|diglycolamic acid|phosphoryl carboxylic|hydroxyacetic acid",
     "CX_CARBOXYLIC", "Y", "cation exchanger", "carboxylic-acid cation exchangers (incl. diglycolamic acid, crown-ether "
                                                 "carboxylic acids, phosphoryl carboxylic acids)"),
    ("X07", r"acidic\+neutral organophosph|acidic organophosph|\bhdehp\b|\bd2ehpa\b|\bdehpa\b|\bp507\b|pc88a|heh\[ehp\]|"
            r"cyanex ?272|cyanex ?301|dithiophosphinic|phosphinic acid|phosphonic acid|alkylphosphoric|"
            r"hydrogen phosphate", "CX_ACIDIC_OP", "Y", "cation exchanger",
     "acidic organophosphorus (HDEHP/D2EHPA, HEH[EHP]/P507, Cyanex 272/301); acidic + neutral OP synergists stay here"),
    ("X08", r"cmpo|carbamoylphosph|carbamoylmethylphosph|phosphoryl ?acetamide|phosphorylurea", "NEUTRAL_CMPO", "N",
     "neutral solvating", "CMPO-type phosphoryl acetamides and phosphorylureas: neutral solvating, outside C-GD"),
    ("X09", r"phosphine oxide|\btopo\b|\btrpo\b|cyanex ?92[35]|\btbp\b|neutral organophosph|phosphonate ester|"
            r"phosphate ester", "NEUTRAL_OP", "N", "neutral solvating",
     "neutral organophosphorus compounds (TBP, TOPO, Cyanex 923/925): outside C-GD"),
    ("X10", r"\bbtp\b|btbp|btphen|phenanthrolin|\bphen\b|phen-|blphen|\bpda\b|dipicolinamide|pyridine ?dicarboxamide|"
            r"dicarboxamide|bipyridyl|bipyridine|triazin|terpyridin|n-donor", "SOFT_N", "N", "soft N-donor",
     "soft N-donors (BTP, BTBP, PDA and phenanthroline amides): separate descriptive family, selectivity direction "
     "varies; N-donor + acid synergists stay here"),
    ("X11", r"diglycolamide|\bdga|dga\b|malonamide|dooda|diamide|amide", "NEUTRAL_AMIDE", "N", "neutral solvating",
     "amide O-donors (DGA, malonamides, DOODA)"),
    ("X12", r"crown|calixarene|macrocycl", "NEUTRAL_MACROCYCLE", "N", "neutral solvating",
     "macrocycles (crown ethers, calixarenes; ion pairs with lipophilic anions)"),
    ("X13", r"ionic liquid|\bil\b", "IL_OTHER", "N", "ionic liquid", "other ionic-liquid extractants"),
    ("X99", r".*", "OTHER", "N", "other", "fallback"),
]
MEDIUM_RULES = [  # (rule_id, pattern on the aqueous-medium text, medium class, Gd-claim eligible, note)
    ("M01", r"lactat|citrat|dtpa|hedta|cdta|edta|\bnta\b|oxalat|tartrat|malonat|glycolat|push-pull|aqueous blphen|"
            r"aqueous phen|buffer", "AQ_COMPLEXANT", "N", "aqueous complexant: no Gd term (M29 rule) [IR M29]"),
    ("M02", r"acetat|acetic", "AQ_COMPLEXANT_ACETATE", "N",
     "acetate decided in advance as a complexant (excluded from the Gd claims); inclusion is a sensitivity [I2 confirm-2]"),
    ("M03", r"thiocyanat|sulfat|h2so4|phosphoric|picrate|trichloroacetic", "NON_INERT", "N",
     "non-nitrate, non-inert medium: outside the Gd claims"),
    ("M04", r"hno3|nitrat|nh4no3|nano3|\bean\b", "NITRATE", "Y", "nitrate medium"),
    ("M05", r"perchlorat|naclo4|chlorid|\bhcl\b|cacl2", "INERT_SALT", "Y", "inert-salt medium"),
    ("M06", r"\bph\b|acidity|neutral|^u$|^$", "UNKNOWN", "U", "medium not stated in the metadata"),
    ("M99", r".*", "UNKNOWN", "U", "fallback"),
]
T_RULES = [  # sibling-design factor -> perturbation class (pair-level T0 needs a cation exchanger)
    (r"\bph\b", "T0|T1"),                    # pH: T0 for a cation exchanger, an aqueous-medium change (T1) otherwise
    (r"phase ratio", "T0|OTHER"),
    (r"(hdehp|d2ehpa|extractant|ligand|\bl\b|tbp|topo|il)\s*(conc|concentration)|\[ha\]|hdehp concentration",
     "T0|T2"),                               # [HA]: T0 for a cation exchanger, organic phase (T2) for solvating
    (r"diluent|toluene|dodecane|kerosene|chcl3|heptane|octanol|modifier|\bils?\b|ionic liquid|nitrobenzene|undiluted|"
     r"crown|synergist|\balone\b|\bwith\b|without|additive|\btopo\b|\btbp\b", "T2"),
    (r"hno3|acid|nitrate|nh4no3|salt|media|medium|ionic strength|thiocyanate|chloride|acetate|water content|\bean\b|"
     r"simulant|feed|composition", "T1"),
    (r"\bt\b|temperature", "OTHER"),
    (r"ligand|extractant|isomer|structure|variants", "EXTRACTANT_CHANGE"),
]


def strip_acc(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def read_meta(path, **kw) -> pd.DataFrame:
    """Read a metadata table and assert that no D / log D / SF (or prediction) column is present."""
    t = pd.read_csv(path, **kw)
    bad = [c for c in t.columns if FORBIDDEN_COL.search(str(c))]
    if bad:
        raise AssertionError(f"{path}: value columns {bad} must not be loaded by a supply audit")
    return t


def mech_rules_table() -> pd.DataFrame:
    rows = [{"rule_type": "extractant", "order": k, "rule_id": r[0], "pattern": r[1], "mechanism_class": r[2],
             "cgd_stratum": r[3], "gd_medium_eligible": "", "ect_report_class": r[4], "note": r[5]}
            for k, r in enumerate(MECH_RULES, 1)]
    rows += [{"rule_type": "medium", "order": k, "rule_id": r[0], "pattern": r[1], "mechanism_class": r[2],
              "cgd_stratum": "", "gd_medium_eligible": r[3], "ect_report_class": "", "note": r[4]}
             for k, r in enumerate(MEDIUM_RULES, 1)]
    rows += [{"rule_type": "sibling_factor", "order": k, "rule_id": f"T{k:02d}", "pattern": r[0], "mechanism_class": r[1],
              "cgd_stratum": "", "gd_medium_eligible": "", "ect_report_class": "",
              "note": "first matching factor rule; 'a|b' = a for a cation exchanger, b otherwise; T0 = pH / [HA] / "
                      "phase ratio for a cation exchanger, T1 aqueous medium, T2 organic phase [I2 confirm-4]"}
             for k, r in enumerate(T_RULES, 1)]
    flags = [("F01", r"\bin\b(.*)", "IL_DILUENT", "an 'in <diluent>' clause naming an ionic liquid sets the IL-diluent "
                                                  "flag (C-GD sensitivity); the class is taken from the extractant "
                                                  "before 'in'"),
             ("F02", r"aqueous\s+\S+", "AQ_COMPLEXANT", "an 'aqueous <ligand>' token is a hold-back complexant: removed "
                                                        "before classification, flags the unit out of the Gd claims"),
             ("F03", r"synergist|\+", "SYNERGIST", "mixture / synergist flag (sensitivity)")]
    rows += [{"rule_type": "flag", "order": k, "rule_id": r[0], "pattern": r[1], "mechanism_class": r[2], "cgd_stratum": "",
              "gd_medium_eligible": "", "ect_report_class": "", "note": r[3]} for k, r in enumerate(flags, 1)]
    return pd.DataFrame(rows)


def _classify_text(text: str, src: str) -> dict:
    t = strip_acc(str(text)).lower().strip()
    il_dil = False
    m = re.search(r"\bin\b(.*)$", t)
    if m and re.search(r"\bil\b|ionic liquid|c4mim|tf2n|quaternary il|aliquat", m.group(1)):
        il_dil = True
        t = t[:m.start()].strip()
    aq = bool(re.search(r"aqueous\s+\S+", t))
    t2 = re.sub(r"aqueous\s+\S+", "", t).strip()
    syn = bool(re.search(r"synergist|\+", t2))
    for rid, pat, cls, cgd, ect, _ in MECH_RULES:
        if re.search(pat, t2):
            return {"mech_rule": rid, "mechanism_class": cls, "cgd_stratum": cgd, "ect_report_class": ect,
                    "il_diluent_flag": il_dil, "aq_complexant_flag": aq, "synergist_flag": syn, "mech_source": src}
    raise AssertionError("MECH_RULES has a catch-all row")


def classify_extractant(label: str, extractant: str = "") -> dict:
    """Frozen map: first matching MECH_RULES row on the chemotype label; the extractant text is used only when the
    label falls through to OTHER.  Flags: IL diluent, aqueous complexant, synergist."""
    a = _classify_text(label, "label")
    if a["mechanism_class"] != "OTHER" or not str(extractant).strip():
        return a
    b = _classify_text(extractant, "extractant")
    return b if b["mechanism_class"] != "OTHER" else a


def classify_medium(text: str) -> dict:
    t = strip_acc(str(text)).lower().strip()
    for rid, pat, cls, elig, _ in MEDIUM_RULES:
        if re.search(pat, t):
            return {"medium_rule": rid, "medium_class": cls, "gd_medium_eligible": elig}
    return {}


def t_classes(desc: str, cx: bool) -> list:
    """Perturbation classes of a sibling design string (factors split on ';', ',', ' x ', ' and ', ' vs ')."""
    out = []
    for fac in re.split(r";|,| x |\band\b|\bvs\b|/", strip_acc(str(desc)).lower()):
        fac = fac.strip()
        if not fac:
            continue
        for pat, cls in T_RULES:
            if re.search(pat, fac):
                if "|" in cls:
                    a, b = cls.split("|")
                    cls = a if cx else b
                out.append(cls)
                break
    return sorted(set(out))


# ------------------------------------------------------------------------------------------------------------
# author parsing and the corpus laboratory key
# ------------------------------------------------------------------------------------------------------------
_TOK = re.compile(r"[^\W\d_](?:[^\W\d_]|[-'’‐‑.])*")


def _norm(tok: str) -> str:
    t = strip_acc(tok).lower().replace("‐", "-").replace("‑", "-").replace("’", "").replace("'", "")
    return t.strip(".-")


def _letters(tok: str) -> str:
    return re.sub(r"[^A-Za-z]", "", strip_acc(tok))


def _is_initials(tok: str) -> bool:
    """'E.A.', 'St.', 'P', 'SA', 'ZY' are initials; 'Wu', 'Li', 'Sasaki' are not."""
    letters = _letters(tok)
    if len(letters) <= 1:
        return True
    if "." in tok and len(letters) <= 3:
        return True
    return letters.isupper() and len(letters) <= 3


def _entries(authors: str) -> list:
    """Split the registry author string into representations (lists of author entries, order kept)."""
    reps, run = [], []
    for piece in str(authors).split(";"):
        piece = piece.strip()
        if not piece:
            continue
        parts = [p.strip() for p in re.split(r",|&|\band\b", piece) if p.strip()]
        if len(parts) >= 2:
            if run:
                reps.append(run)
                run = []
            reps.append(parts)
        else:
            run.append(piece)
    if run:
        reps.append(run)
    return reps


def _tokens(entry: str) -> list:
    toks = [t for t in _TOK.findall(entry) if _norm(t) and _norm(t) not in TITLES]
    return toks


def parse_reference(authors: str, gf_index=None, ref_key: str = "") -> dict:
    """Persons (family, first initial), family names, PI positions and the senior person of one registry entry."""
    reps = _entries(authors)
    persons, fams, decided = [], set(), []
    rep_persons = []
    for rep in reps:
        rp, pending = [], []
        for e in rep:
            toks = _tokens(e)
            if not toks:
                continue
            # an all-upper family name: 'Yuji SASAKI', 'Yan LI' (last, after a full given name) or >= 4 letters
            ups = [k for k, t in enumerate(toks) if "." not in t and _letters(t).isupper() and len(_letters(t)) >= 2]
            fam = given = None
            if ups:
                k = ups[-1]
                u = toks[k]
                before = [t for t in toks[:k] if not _is_initials(t) and not _letters(t).isupper()]
                if (k == len(toks) - 1 and before) or len(_letters(u)) >= 4:
                    fam, given = u, (toks[:k] + toks[k + 1:] or [""])[0]
            if fam is None:
                full = [t for t in toks if not _is_initials(t)]
                if not full:
                    continue
                if len(full) == 1:
                    fam = full[0]
                    given = ([t for t in toks if t is not fam] or [""])[0]
                elif len(full) == 2 and len(toks) == 2:
                    a, b = _norm(full[0]).replace("-", ""), _norm(full[1]).replace("-", "")
                    if a in CJK_FAMILY and b not in CJK_FAMILY:
                        fam, given = full[0], full[1]
                        decided.append("FG")
                    elif b in CJK_FAMILY and a not in CJK_FAMILY:
                        fam, given = full[1], full[0]
                        decided.append("GF")
                    elif a in CJK_FAMILY and b in CJK_FAMILY:
                        gi = gf_index or {}
                        rev = gi.get((a, b[:1]), set()) - {ref_key}
                        dirn = gi.get((b, a[:1]), set()) - {ref_key}
                        if rev and not dirn:
                            fam, given = full[0], full[1]
                            decided.append("FG")
                        elif dirn and not rev:
                            fam, given = full[1], full[0]
                            decided.append("GF")
                        else:
                            pending.append((len(rp), full))
                            rp.append(None)
                            continue
                    else:
                        fam, given = full[-1], full[0]
                else:
                    fam, given = full[-1], full[0]
            rp.append((_norm(fam), _norm(given)[:1] if given else ""))
        if pending:
            fg = decided.count("FG") > decided.count("GF")
            for k, full in pending:
                f, g = (full[0], full[1]) if fg else (full[1], full[0])
                rp[k] = (_norm(f), _norm(g)[:1])
        rp = [p for p in rp if p and p[0]]
        rep_persons.append(rp)
        persons += rp
    for f, _ in persons:
        fams.add(f)
    first = rep_persons[0] if rep_persons else []
    trunc = len(str(authors)) >= 399 and len(rep_persons) == 1
    pi = [] if not first else ([first[0], first[-1]] if not trunc else [first[0]])
    senior = first[-1] if first and not trunc else None
    return {"persons": sorted(set(persons)), "families": sorted(fams), "pi": sorted(set(pi)), "senior": senior,
            "truncated": trunc, "n_reps": len(rep_persons), "fg_decisions": decided}


def gf_index_from(reg: pd.DataFrame) -> dict:
    """(family, given initial) -> registry keys, from unambiguous Given-Family entries (>= 2 full words or a
    surname-final full word after initials), used as cross-reference evidence for Family-Given order."""
    idx = {}
    for k, a in zip(reg["key"].astype(str), reg["authors"].fillna("").astype(str)):
        for rep in _entries(a):
            for e in rep:
                full = [t for t in _tokens(e) if not _is_initials(t)]
                if len(full) >= 2:
                    f, g = _norm(full[-1]).replace("-", ""), _norm(full[0]).replace("-", "")
                    idx.setdefault((f, g[:1]), set()).add(k)
    return idx


def _same_person(p, q) -> bool:
    return p[0] == q[0] and (not p[1] or not q[1] or p[1] == q[1])


def corpus_links(parsed: dict, drop=None, drop_pi=DROP) -> pd.DataFrame:
    """L2 (>= 2 shared distinctive family names; ``drop`` excluded, default DROP_L2) and LPI (the same PI person,
    family + first initial, in a PI position of both; ``drop_pi`` excluded, default the four named surnames)."""
    drop = DROP_L2 if drop is None else frozenset(drop)
    keys = sorted(parsed)
    rows = []
    for i, a in enumerate(keys):
        A = parsed[a]
        fa = set(A["families"]) - drop
        for b in keys[i + 1:]:
            B = parsed[b]
            sh = sorted(fa & (set(B["families"]) - drop))
            pis = sorted({p[0] for p in A["pi"] for q in B["pi"] if _same_person(p, q) and p[0] not in drop_pi})
            if len(sh) >= 2 or pis:
                rows.append({"key_a": a, "key_b": b, "shared_families": ";".join(sh), "n_shared": len(sh),
                             "L2": len(sh) >= 2, "LPI": bool(pis), "pi_shared": ";".join(pis),
                             "single_pi_only": bool(pis) and len(sh) < 2})
    return pd.DataFrame(rows, columns=["key_a", "key_b", "shared_families", "n_shared", "L2", "LPI", "pi_shared",
                                       "single_pi_only"])


def components(nodes, links) -> dict:
    par = {n: n for n in nodes}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in links:
        par[find(a)] = find(b)
    roots = {n: find(n) for n in nodes}
    name = {}
    for n in sorted(nodes):
        name.setdefault(roots[n], f"LAB{len(name) + 1:03d}")
    return {n: name[roots[n]] for n in nodes}


def corpus_key(reg: pd.DataFrame, drop=None, drop_pi=DROP) -> tuple:
    """(per-reference table with conservative / liberal lab ids, link table)."""
    gfi = gf_index_from(reg)
    parsed = {k: parse_reference(a, gfi, k) for k, a in zip(reg["key"].astype(str), reg["authors"].fillna("").astype(str))}
    L = corpus_links(parsed, DROP_L2 if drop is None else drop, drop_pi)
    cons = components(list(parsed), list(zip(L.key_a, L.key_b)))
    lib = {}
    for k, P in parsed.items():
        s = P["senior"]
        lib[k] = f"PI:{s[0]}_{s[1]}" if s else f"PI:unknown:{k}"
    rows = []
    for r in reg.itertuples(index=False):
        k = str(r.key)
        P = parsed[k]
        rows.append({"unit_type": "corpus", "key": k, "year": r.year, "in_v1_cohort": bool(r.in_v1_cohort),
                     "c3_cells": r.c3_cells, "v1_cells": r.v1_cells, "n_ln_metals": r.n_ln_metals,
                     "senior": f"{P['senior'][0]} {P['senior'][1]}." if P["senior"] else "", "truncated": P["truncated"],
                     "families": ";".join(P["families"]), "pi_positions": ";".join(f"{f} {g}." for f, g in P["pi"]),
                     "family_given_decisions": ",".join(P["fg_decisions"]),
                     "lab_conservative": cons[k], "lab_liberal": lib[k]})
    return pd.DataFrame(rows), L, parsed


# ------------------------------------------------------------------------------------------------------------
# literature candidates
# ------------------------------------------------------------------------------------------------------------
def lab_cons(lab: str) -> str:
    for name, rule in CONS_MERGE:
        if rule(lab):
            return name
    return lab


def candidates_frozen(cand: pd.DataFrame, parsed: dict, drop=None) -> pd.DataFrame:
    """P4 candidates with the frozen (PC P4) eligibility and laboratory flags, the automatic author-overlap
    evidence (``drop`` excluded, default DROP_L2), the mechanism / medium classes and the E-CT perturbation classes."""
    drop = DROP_L2 if drop is None else frozenset(drop)
    c = cand.copy()
    c["eligibility_P4"] = c["eligibility"]
    c["lab_in_corpus_P4"] = c["lab_in_corpus"]
    elig, fixes = [], []
    for r in c.itertuples(index=False):
        e, why = r.eligibility, []
        if r.cand_id in CAND_FIX and CAND_FIX[r.cand_id][0] == "eligibility":
            e = CAND_FIX[r.cand_id][1]
            why.append(CAND_FIX[r.cand_id][2])
        if e == "E2" and r.screen_level == "TITLE":
            e = "U"
            why.append("title-only 'no abstract' row: U by P4's tier rule as written [PC P4]")
        elig.append(e)
        fixes.append(why)
    c["eligibility_frozen"] = elig
    lic, lwhy = [], []
    for r, why in zip(c.itertuples(index=False), fixes):
        v = r.lab_in_corpus
        if r.lab_id in INSTITUTION_FIX:
            v = INSTITUTION_FIX[r.lab_id][0]
            why = why + [INSTITUTION_FIX[r.lab_id][1]]
        if r.cand_id in CAND_FIX and CAND_FIX[r.cand_id][0] == "lab_in_corpus":
            v = CAND_FIX[r.cand_id][1]
            why = why + [CAND_FIX[r.cand_id][2]]
        lic.append(v)
        lwhy.append(" | ".join(why))
    c["lab_in_corpus_frozen"] = lic
    c["frozen_fix_notes"] = lwhy
    c["second_order_link"] = c["lab_id"].map(SECOND_ORDER).fillna("")
    c["lab_liberal"] = c["lab_id"]
    c["lab_conservative"] = c["lab_id"].map(lab_cons)
    # automatic author-overlap evidence with the repaired surname rule
    corp_fams = {k: set(P["families"]) - drop for k, P in parsed.items()}
    allf = set().union(*corp_fams.values())
    auto, ev, bref = [], [], []
    for r in c.itertuples(index=False):
        sn = {_norm(s) for s in str(r.author_surnames).split(";") if s.strip() and s.strip() != "..."} - drop
        last = _norm(r.last_author)
        best = max(((len(sn & f), k) for k, f in sorted(corp_fams.items())), default=(0, ""))
        if not best[0] and last in allf:
            best = (1, sorted(k for k, f in corp_fams.items() if last in f)[0])
        if (last and last not in drop and last in allf) or best[0] >= 2:
            a = "Y"
        elif best[0] == 1 or (sn & allf):
            a = "P"
        else:
            a = "N"
        auto.append(a)
        bref.append(best[1] if best[0] else "")
        ev.append(f"senior '{last}' in corpus: {last in allf}; best shared {best[0]} ({best[1]}); "
                  f"overlap {sorted(sn & allf)}")
    c["lab_in_corpus_auto"] = auto
    c["auto_best_ref"] = bref
    c["auto_evidence"] = ev
    mech = [classify_extractant(r.chemotype_label, r.extractant) for r in c.itertuples(index=False)]
    for k in ("mech_rule", "mechanism_class", "cgd_stratum", "ect_report_class", "il_diluent_flag",
              "aq_complexant_flag", "synergist_flag"):
        c[k] = [m.get(k) for m in mech]
    med = [classify_medium(x) for x in c["acid_medium"]]
    for k in ("medium_rule", "medium_class", "gd_medium_eligible"):
        c[k] = [m.get(k) for m in med]
    cx = c["mechanism_class"].str.startswith("CX_")
    c["t_classes"] = [";".join(t_classes(d, bool(x))) for d, x in zip(c["sibling_desc"], cx)]
    c["non_t0_possible"] = [bool(set(t.split(";")) & {"T1", "T2"}) if t else False for t in c["t_classes"]]
    c["t0_only"] = [bool(t) and set(t.split(";")) <= {"T0"} for t in c["t_classes"]]
    c["chemotype_label_merged"] = c["chemotype_label"].replace(TOMANO3_MERGE)
    return c


# ------------------------------------------------------------------------------------------------------------
# counts, ledger and the supply-audit kill
# ------------------------------------------------------------------------------------------------------------
def new_labs(c: pd.DataFrame, tiers, elig_col: str, lic_col: str, merge: str = "lab_conservative") -> list:
    s = c[c[elig_col].isin(tiers) & (c[lic_col] == "N")]
    return sorted(s[merge].unique())


def expected_yield(c: pd.DataFrame, elig_col: str, lic_col: str, p_e2: float, p_u: float,
                   merge: str = "lab_conservative") -> float:
    """P4's formula: E1 labs + p_E2 x (E1+E2 - E1) + p_U x (E1+E2+U - E1+E2), labs counted at their best tier."""
    e1 = len(new_labs(c, {"E1"}, elig_col, lic_col, merge))
    e2 = len(new_labs(c, {"E1", "E2"}, elig_col, lic_col, merge))
    eu = len(new_labs(c, {"E1", "E2", "U"}, elig_col, lic_col, merge))
    return e1 + p_e2 * (e2 - e1) + p_u * (eu - e2)


def claim_supply(c: pd.DataFrame, lic_col: str = "lab_in_corpus_frozen", elig_col: str = "eligibility_frozen") -> dict:
    """Lab-by-lab supply per claim from the frozen key (G_ECT, G_HS, G_Gd); verified / expected / all-convert."""
    new = c[c[lic_col] == "N"].copy()
    tier = new[elig_col]
    out = {}

    def labs(mask, unit="lab"):
        s = new[mask]
        if unit == "lab":
            return sorted(s["lab_conservative"].unique())
        return sorted(set(map(tuple, s[["lab_conservative", "chemotype_label_merged"]].to_numpy())))
    full_v = new.full_LaLu_series == "Y"
    full_u = new.full_LaLu_series.isin(["Y", "U"])
    sib = new.sibling_conditions == "Y"
    nt0 = new.non_t0_possible
    ect_v = labs((tier == "E1") & sib & full_v & nt0)
    ect_e2 = labs(tier.isin(["E1", "E2"]) & sib & full_u & nt0)
    ect_u = labs(tier.isin(["E1", "E2", "U"]) & sib.where(tier != "U", True) & full_u & nt0)
    out["G_ECT"] = {"unit": "new laboratory with >= 3 full-series sibling conditions of one extractant, a T1/T2 factor",
                    "verified": ect_v, "e1_e2_all_convert": ect_e2, "e1_e2_u_all_convert": ect_u}
    gd_med = new.gd_medium_eligible.isin(["Y", "U"]) & ~new.aq_complexant_flag.astype(bool)
    hs = new.cgd_stratum == "Y"
    hs_v = labs((tier == "E1") & hs & (new.m22_rule_8Ln_EuGdTb == "Y") & gd_med)
    hs_e2 = labs(tier.isin(["E1", "E2"]) & hs & new.m22_rule_8Ln_EuGdTb.isin(["Y", "U"]) & gd_med)
    hs_u = labs(tier.isin(["E1", "E2", "U"]) & hs & new.m22_rule_8Ln_EuGdTb.isin(["Y", "U"]) & gd_med)
    fam = {lab: sorted(set(new[(new.lab_conservative == lab) & hs].mechanism_class)) for lab in hs_e2}
    out["G_HS"] = {"unit": "new cation-exchange laboratory (C-GD stratum) reporting Eu, Gd, Tb, non-complexant medium",
                   "verified": hs_v, "e1_e2_all_convert": hs_e2, "e1_e2_u_all_convert": hs_u, "classes_by_lab": fam}
    gd_v = labs((tier == "E1") & (new.m22_rule_8Ln_EuGdTb == "Y") & gd_med, unit="labxchem")
    gd_e2 = labs(tier.isin(["E1", "E2"]) & new.m22_rule_8Ln_EuGdTb.isin(["Y", "U"]) & gd_med, unit="labxchem")
    gd_u = labs(tier.isin(["E1", "E2", "U"]) & new.m22_rule_8Ln_EuGdTb.isin(["Y", "U"]) & gd_med, unit="labxchem")
    out["G_Gd"] = {"unit": "new laboratory x chemotype unit reporting Eu, Gd, Tb, non-complexant medium (any class)",
                   "verified": [list(x) for x in gd_v], "e1_e2_all_convert": [list(x) for x in gd_e2],
                   "e1_e2_u_all_convert": [list(x) for x in gd_u]}
    return out


def ect_chem_units(cand: pd.DataFrame, labs) -> dict:
    """C-ECT decision units under D3 fallback (ii) (CONFIRMATION_PACKAGE 4.2): P4's hand chemotype labels (TOMANO3 =
    [A336][NO3] merged) of the G_ECT-eligible candidate rows of the given laboratories; a laboratory with eligible rows
    in more than one chemotype links them into one unit (connected components of the laboratory-chemotype graph), so
    no laboratory contributes to two units.  Hand labels are PROVISIONAL (the frozen gen6 map fixes them at the U5
    audit, DATA_CONTRACT 8.3).  Moved here from scripts/s0_package_supplements.py (s0-statistics-3), unchanged."""
    c = cand[(cand.unit_type == "candidate")] if "unit_type" in cand.columns else cand
    c = c[c.lab_in_corpus_frozen == "N"].copy()
    tier = c.eligibility_frozen
    full_u = c.full_LaLu_series.isin(["Y", "U"])
    sib = c.sibling_conditions == "Y"
    nt0 = c.non_t0_possible.astype(str) == "True"
    m = tier.isin(["E1", "E2", "U"]) & sib.where(tier != "U", True) & full_u & nt0
    s_ = c[m & c.lab_conservative.isin(list(labs))]
    pairs = sorted(set(zip(s_.lab_conservative, s_.chemotype_label_merged)))
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for lab, ch in pairs:
        parent[find(("L", lab))] = find(("C", ch))
    comps = {}
    for lab, ch in pairs:
        comps.setdefault(find(("C", ch)), set()).add((lab, ch))
    units = [sorted({ch for _, ch in v}) for v in comps.values()]
    return {"labs": sorted(set(s_.lab_conservative)), "lab_chemotype_pairs": [list(q) for q in pairs],
            "distinct_chemotypes": len({ch for _, ch in pairs}), "chemotype_units_linked": len(units),
            "units": sorted(units)}


CLAIMS = [  # claim, supply key, minimum G, brief's realistic supply range (section 4.6 / 7.5), decision unit, role
    {"claim": "C-ECT", "supply": "G_ECT_UNITS", "min_G": 7, "realistic": [2, 3],
     "unit": "chemotype units (D3 ii; linked laboratory-chemotype components)", "role": "test",
     "source": "brief 4.6, 7.5; I2 confirm-4 (minimum 7; < 7 -> descriptive case series); D3 fallback (ii); the brief's "
               "realistic range counts laboratories"},
    {"claim": "C-GD", "supply": "G_HS", "min_G": 6, "realistic": [2, 5], "unit": "laboratories (>= 3 families)",
     "role": "descriptive (D3 i)", "source": "brief 4.6, 7.5; I2 confirm-1, confirm-2 (6 laboratories from >= 3 families)"},
    {"claim": "Gd-dip transport (merged SAFE + literature)", "supply": "G_Gd+SAFE", "min_G": 10, "realistic": [2, 6],
     "unit": "laboratory x chemotype units", "role": "descriptive (D3 i)",
     "source": "brief 7.2, 7.5; IR M33 (>= 10 units to run); literature 2-3 verified + SAFE 0-3"},
    {"claim": "SAFE route (M33 gate)", "supply": "SAFE", "min_G": 10, "realistic": [0, 3], "role": "descriptive (D3 i)",
     "unit": "experimental multi-Ln units outside the corpus", "source": "brief 4.6, 7.2 (SAFE 0-3 against 10)"},
]
SAFE_RANGE = (0, 3)          # U2 unanswered: 0 verified, 3 expected at most [IR M33]


AUDIT_RULE = ("decision supply (D4) = units verified in hand from metadata under the frozen key (E1 frozen, the claim's "
              "metadata verified), the institution-link rule applied consistently, plus SAFE's maximum (3) where SAFE "
              "units count while U2 is unanswered; C-ECT counts linked chemotype units (D3 fallback ii; P4 hand labels, "
              "PROVISIONAL); below_min = decision supply < minimum G.  The kill fires when every claim with role 'test' "
              "is below its minimum (C-GD and Gd-dip transport are descriptive intervals under D3 fallback i; their "
              "'>= minimum' is a descriptive-run trigger, never a test).  Forecasts (P4's and PC P4's conversion rates), "
              "the all-convert scenario and the brief's realistic bracket (bracket_high = max(in hand, realistic high)) "
              "are reported and never decide; the audit reruns after each answered request (section 4.6).")


def supply_audit(cs: dict, forecasts: dict, cand: pd.DataFrame | None = None) -> dict:
    """Section 4.3 item 7 under AUDIT_RULE (D4).  ``forecasts``: {name: expected_claims(...)}; ``cand``: the
    candidate frame (lab_key rows) for C-ECT's linked chemotype units (required for that claim)."""
    res = []
    for cl in CLAIMS:
        k = cl["supply"]
        extra = {}
        if k == "SAFE":
            ver, inhand, fc, allc = SAFE_RANGE[0], SAFE_RANGE[1], {n: float(SAFE_RANGE[1]) for n in forecasts}, SAFE_RANGE[1]
        elif k == "G_Gd+SAFE":
            ver = len(cs["G_Gd"]["verified"])
            inhand = ver + SAFE_RANGE[1]
            fc = {n: e["G_Gd"] + SAFE_RANGE[1] for n, e in forecasts.items()}
            allc = len(cs["G_Gd"]["e1_e2_u_all_convert"]) + SAFE_RANGE[1]
        elif k == "G_ECT_UNITS":
            if cand is None:
                raise ValueError("supply_audit: C-ECT's chemotype units need the candidate frame")
            uv = ect_chem_units(cand, cs["G_ECT"]["verified"])
            ua = ect_chem_units(cand, cs["G_ECT"]["e1_e2_u_all_convert"])
            ver = inhand = uv["chemotype_units_linked"]
            fc = {n: e["G_ECT"] for n, e in forecasts.items()}
            allc = ua["chemotype_units_linked"]
            extra = {"supply_in_hand_laboratories": len(cs["G_ECT"]["verified"]),
                     "supply_all_convert_laboratories": len(cs["G_ECT"]["e1_e2_u_all_convert"]),
                     "all_convert_distinct_chemotypes": ua["distinct_chemotypes"], "verified_units": uv["units"],
                     "forecast_unit": "laboratories (G_ECT); the chemotype-unit reach is in supply_reach.json"}
        else:
            ver = inhand = len(cs[k]["verified"])
            fc = {n: e[k] for n, e in forecasts.items()}
            allc = len(cs[k]["e1_e2_u_all_convert"])
        dec = inhand                                          # D4 (s0-fidelity-5, s0-statistics-3)
        res.append({**cl, "supply_verified_literature": ver, "supply_in_hand": inhand, "decision_supply": dec,
                    "bracket_high": max(inhand, cl["realistic"][1]), "below_min": bool(dec < cl["min_G"]),
                    "forecasts": {n: {"expected": round(float(v), 2), "reaches_min": bool(v >= cl["min_G"])}
                                  for n, v in fc.items()},
                    "supply_all_convert": allc, "all_convert_reaches_min": bool(allc >= cl["min_G"]), **extra})
    tests = [r for r in res if r["role"] == "test"]
    kill = all(r["below_min"] for r in tests)
    fkill = {n: all(not r["forecasts"][n]["reaches_min"] for r in tests) for n in forecasts}
    gd = next(r for r in res if r["supply"] == "G_Gd+SAFE")
    return {"claims": res, "kill_fires": kill, "kill_under_forecast": fkill,
            "kill_under_bracket_high": all(r["bracket_high"] < r["min_G"] for r in tests),
            "descriptive_run_trigger_gd_dip": bool(gd["decision_supply"] >= gd["min_G"]),
            "outcome": "SUPPLY-AUDIT KILL: every test claim below its minimum G -> no confirmation runs; round 5 is "
                       "reported as descriptive (section 7.6)" if kill else "at least one test claim reaches its minimum G",
            "rule": AUDIT_RULE}


def expected_claims(cs: dict, p_e2: float = 0.5, p_u: float = 0.25) -> dict:
    """P4-style expected supply per claim: verified + p_E2 x (E1+E2 extra) + p_U x (U extra)."""
    out = {}
    for k in ("G_ECT", "G_HS", "G_Gd"):
        v, e2, u = (len(cs[k][x]) for x in ("verified", "e1_e2_all_convert", "e1_e2_u_all_convert"))
        out[k] = v + p_e2 * (e2 - v) + p_u * (u - e2)
    return out


def ledger_rows(c: pd.DataFrame) -> list:
    """SUPPLY_LEDGER rows: brief 7.5 quantities with P4's value, PC P4's corrections and this recount."""
    E1, E12, E12U = {"E1"}, {"E1", "E2"}, {"E1", "E2", "U"}
    c2 = c.copy()
    c2["lic_shafer"] = np.where(c2.lab_id == "CSM-Shafer", "P", c2.lab_in_corpus_P4)
    c2["elig_tier"] = [("U" if (e == "E2" and s == "TITLE") else e) for e, s in zip(c2.eligibility_P4, c2.screen_level)]
    c2["elig_063"] = np.where(c2.cand_id == "P4-063", "E2", c2.eligibility_P4)

    def n(tiers, elig, lic, merge="lab_conservative"):
        return len(new_labs(c2, tiers, elig, lic, merge))
    rows = []

    def add(q, tier, rep, corr, rec, src, note=""):
        rows.append({"section": "supply", "quantity": q, "tier": tier, "reported_P4": rep, "corrected_PC_P4": corr,
                     "recomputed_r5": rec, "source": src, "note": note})
    add("verified new independent laboratories", "E1", 7, "6 strict (P4-063 to E2); 5 institution rule consistent",
        f"{n(E1, 'eligibility_P4', 'lab_in_corpus_P4')} as P4; {n(E1, 'elig_063', 'lab_in_corpus_P4')} strict; "
        f"{n(E1, 'eligibility_frozen', 'lab_in_corpus_frozen')} frozen", "P4 0, 4; PC P4; brief 7.5")
    add("new laboratories (conservative merge)", "E1+E2", 16, "14 tier rule as written; 13 CSM-Shafer P; 11 Osaka and "
        "Kanazawa P", f"{n(E12, 'eligibility_P4', 'lab_in_corpus_P4')} as P4; {n(E12, 'elig_tier', 'lab_in_corpus_P4')} "
        f"tier rule; {n(E12, 'elig_tier', 'lic_shafer')} +CSM-Shafer P; "
        f"{n(E12, 'eligibility_frozen', 'lab_in_corpus_frozen')} frozen (all corrections)", "P4 4; PC P4; brief 7.5")
    add("new laboratories (liberal: each PI)", "E1 / E1+E2 / E1+E2+U", "7 / 19 / 28", "",
        f"{n(E1, 'eligibility_P4', 'lab_in_corpus_P4', 'lab_liberal')} / {n(E12, 'eligibility_P4', 'lab_in_corpus_P4', 'lab_liberal')}"
        f" / {n(E12U, 'eligibility_P4', 'lab_in_corpus_P4', 'lab_liberal')} as P4; "
        f"{n(E1, 'eligibility_frozen', 'lab_in_corpus_frozen', 'lab_liberal')} / "
        f"{n(E12, 'eligibility_frozen', 'lab_in_corpus_frozen', 'lab_liberal')} / "
        f"{n(E12U, 'eligibility_frozen', 'lab_in_corpus_frozen', 'lab_liberal')} frozen", "P4 4")
    add("new laboratories (conservative merge)", "E1+E2+U", 24, "", f"{n(E12U, 'eligibility_P4', 'lab_in_corpus_P4')} as P4; "
        f"{n(E12U, 'eligibility_frozen', 'lab_in_corpus_frozen')} frozen", "P4 4")
    p4f = expected_yield(c, "eligibility_P4", "lab_in_corpus_P4", 0.5, 0.25)
    add("expected new laboratories after verification", "formula", 13.5, "9.3-11.5 with corrections; 7.0 with every "
        "correction", f"{p4f:.2f} P4 formula; {expected_yield(c, 'eligibility_P4', 'lab_in_corpus_P4', 0.5, 0.0):.2f} "
        f"no p_U; {expected_yield(c2, 'elig_063', 'lab_in_corpus_P4', 2 / 6, 0.25):.2f} / "
        f"{expected_yield(c2, 'elig_063', 'lab_in_corpus_P4', 2 / 6, 0.0):.2f} P4-063 reclassified (p_E2 2/6, with / "
        f"without p_U); {expected_yield(c, 'eligibility_frozen', 'lab_in_corpus_frozen', 2 / 6, 0.25):.2f} / "
        f"{expected_yield(c, 'eligibility_frozen', 'lab_in_corpus_frozen', 2 / 6, 0.0):.2f} every correction",
        "P4 0; PC P4", "E1 + p_E2 x E2-only + p_U x U-only labs")
    s = c[c.eligibility_P4.isin(E12) & (c.lab_in_corpus_P4 == "N")]
    newchem = s[s.chemotype_in_corpus == "N"].lab_conservative.nunique()
    add("new laboratories on chemotypes absent from the corpus", "E1+E2", "15 of 16", "",
        f"{newchem} of {s.lab_conservative.nunique()} as P4", "P4 0, 4; brief 7.5")
    m22 = c[(c.eligibility_P4 == "E1") & (c.lab_in_corpus_P4 == "N") & (c.m22_rule_8Ln_EuGdTb == "Y")]
    m22f = c[(c.eligibility_frozen == "E1") & (c.lab_in_corpus_frozen == "N") & (c.m22_rule_8Ln_EuGdTb == "Y")]
    add("new laboratories with >= 8 Ln incl. Eu, Gd, Tb (verified)", "E1", 3, "2 if Osaka is P",
        f"{m22.lab_conservative.nunique()} as P4 ({', '.join(sorted(m22.lab_conservative.unique()))}); "
        f"{m22f.lab_conservative.nunique()} frozen ({', '.join(sorted(m22f.lab_conservative.unique()))})", "P4 4; PC P4")
    sb = c[(c.eligibility_P4 == "E1") & (c.lab_in_corpus_P4 == "N") & (c.sibling_conditions == "Y")
           & c.full_LaLu_series.isin(["Y", "U"])]
    sbf = c[(c.eligibility_frozen == "E1") & (c.lab_in_corpus_frozen == "N") & (c.sibling_conditions == "Y")
            & c.full_LaLu_series.isin(["Y", "U"])]
    sbv = sbf[sbf.full_LaLu_series == "Y"]
    add("new laboratories with >= 3 full-series sibling conditions", "E1", "about 4 (KU Leuven, Chiba, Turanov, "
        "Atanassova)", "3 after Turanov to E2; full La-Lu verified only KU Leuven and Chiba (Chiba pH-controlled, T0)",
        f"{sb.lab_conservative.nunique()} as P4 ({', '.join(sorted(sb.lab_conservative.unique()))}); "
        f"{sbf.lab_conservative.nunique()} frozen ({', '.join(sorted(sbf.lab_conservative.unique()))}); full La-Lu "
        f"verified {sbv.lab_conservative.nunique()} ({', '.join(sorted(sbv.lab_conservative.unique()))})",
        "P4 4, 5; PC P4; brief 7.5", "full series Y or U; the mechanical sibling parse finds a diluent (T2) factor "
                                     "for Chiba; the brief treats Chiba as T0 (U6 / U5 review)")
    cpg = []
    for tiers in (E1, E12, E12U):
        t = c[c.eligibility_P4.isin(tiers) & (c.chemotype_in_corpus == "N")]
        cpg.append((t.chemotype_label.nunique(), t.chemotype_label_merged.nunique()))
    add("cpg-unit supply upper bounds (new chemotypes, any lab)", "E1 / E1+E2 / E1+E2+U", "10 / 16 / 21",
        "9 / 15 / 20 with TOMANO3 = [A336][NO3]", " / ".join(f"{a}" for a, _ in cpg) + " raw; "
        + " / ".join(f"{b}" for _, b in cpg) + " merged", "P4 4; PC P4; brief 7.5")
    return rows

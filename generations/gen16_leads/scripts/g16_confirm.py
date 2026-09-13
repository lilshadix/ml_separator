"""The single confirmation run (PRE_REGISTRATION.md section 4).

Runs every frozen claim ONCE on the five withheld confirmation seeds, under all five designs,
and writes the discovery-versus-confirmation table.  Nothing here is re-run after it is read.

Protocol enforced by this script:

* the seeds are read from a JSON file OUTSIDE the repository (``--seeds-file``), and the SHA-256
  of ``json.dumps(sorted(seeds))`` must equal the commitment recorded in PRE_REGISTRATION.md;
  the seeds are also recomputed from the registered rule and must agree;
* the claim list is ``gen16_leads/gen16/claims.py`` (frozen at the end of discovery, at most
  five entries); each claim exposes ``run(seeds) -> (board, contrasts)`` where ``contrasts`` has
  the ``paired_contrasts`` columns and one row per design for the claim's registered comparison;
* results go to ``gen16_leads/results/confirmation/`` and a marker file
  ``CONFIRMATION_RUN_ONCE.json`` refuses a second run unless ``--force`` is given, in which case
  the second run is written beside the first with an explicit ``rerun`` suffix (never over it).

Run:  .venv/Scripts/python.exe generations/gen16_leads/scripts/g16_confirm.py --seeds-file <path> [--claims id,id] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent
for p in (ROOT / "generations" / "gen13_separation", ROOT / "generations" / "gen14_direction", ROOT / "generations" / "gen15_curve", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

COMMITMENT = "5a30455bc67d364aa3e65f6d3d28bd4b6fcf3301fe979ca4a1be4a9547de76a9"
DISCOVERY = (104729, 130363, 155921, 196613, 262147)
OUT = HERE / "results" / "confirmation"
MARK = OUT / "CONFIRMATION_RUN_ONCE.json"


def seeds_from_rule() -> list[int]:
    seeds: list[int] = []
    i = 1
    while len(seeds) < 5:
        h = hashlib.sha256(f"gen16-confirmation-seed-{i}".encode()).hexdigest()
        s = int(h[:8], 16) % 900_000 + 100_000
        if s not in DISCOVERY and s not in seeds:
            seeds.append(s)
        i += 1
    return sorted(seeds)


def verify_seeds(path: Path) -> list[int]:
    seeds = sorted(int(s) for s in json.loads(path.read_text())["seeds"])
    digest = hashlib.sha256(json.dumps(seeds).encode()).hexdigest()
    if digest != COMMITMENT:
        raise SystemExit(f"seed file does not match the pre-registered commitment: {digest}")
    if seeds != seeds_from_rule():
        raise SystemExit("seed file does not match the registered rule")
    if set(seeds) & set(DISCOVERY):
        raise SystemExit("a confirmation seed collides with a discovery seed")
    return seeds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-file", required=True)
    ap.add_argument("--claims", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    seeds = verify_seeds(Path(a.seeds_file))
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if MARK.exists():
        if not a.force:
            raise SystemExit("the confirmation run has already been executed once; refusing to rerun")
        suffix = "_rerun_" + datetime.now().strftime("%Y%m%dT%H%M%S")
    from gen16.claims import CLAIMS  # noqa: E402  (frozen at the end of discovery)
    wanted = [c for c in a.claims.split(",") if c] or list(CLAIMS)
    if len(wanted) > 5:
        raise SystemExit("at most five claims go to confirmation")
    rows = []
    t0 = time.time()
    print(f"[confirm] seeds {seeds} verified against commitment; {len(wanted)} claims", flush=True)
    for cid in wanted:
        claim = CLAIMS[cid]
        t1 = time.time()
        board, con = claim.run(seeds)
        board.insert(0, "claim", cid)
        con.insert(0, "claim", cid)
        board.to_csv(OUT / f"{cid}_board{suffix}.csv", index=False)
        con.to_csv(OUT / f"{cid}_contrasts{suffix}.csv", index=False)
        disc = pd.read_csv(HERE / claim.discovery_contrasts)
        disc = disc[disc["comparison"] == claim.comparison]
        for _, r in con[con["comparison"] == claim.comparison].iterrows():
            d = disc[disc["design"] == r["design"]]
            rows.append({
                "claim": cid, "design": r["design"], "comparison": claim.comparison,
                "discovery_point": float(d["point"].iloc[0]) if len(d) else float("nan"),
                "discovery_ci_low": float(d["ci95_low"].iloc[0]) if len(d) else float("nan"),
                "discovery_ci_high": float(d["ci95_high"].iloc[0]) if len(d) else float("nan"),
                "discovery_p": float(d["p_two_sided"].iloc[0]) if len(d) else float("nan"),
                "confirmation_point": float(r["point"]), "confirmation_ci_low": float(r["ci95_low"]),
                "confirmation_ci_high": float(r["ci95_high"]), "confirmation_p": float(r["p_two_sided"]),
                "confirmation_seeds_positive": int(r["seeds_positive"]),
                "confirmation_loco_stable": bool(r["loco_sign_stable"]),
                "confirmation_passes_P1": bool(r["passes_P1"]),
            })
        print(f"[confirm] {cid} done in {time.time() - t1:.0f}s", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / f"discovery_vs_confirmation{suffix}.csv", index=False)
    verdict = (table.groupby("claim")["confirmation_passes_P1"].all()
               .rename("confirmed_all_five_designs").reset_index())
    verdict.to_csv(OUT / f"verdicts{suffix}.csv", index=False)
    MARK.write_text(json.dumps({"ran_at": datetime.now().isoformat(timespec="seconds"),
                                "seeds": seeds, "claims": wanted, "suffix": suffix,
                                "seconds": time.time() - t0}, indent=1))
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))
    print(verdict.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

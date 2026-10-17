"""gen20.1 stage S10 (VERDICTS + REPORT): TAB-01..TAB-14, RULE-T / RULE-C / RULE-H4 verdicts, NC1, BH over the
exploratory family and GEN20_1_REPORT.md, from saved tables only (IMPLEMENTATION_PLAN.md §9 S10).

* refuses a ``smoke*`` label before anything else (LT-29), then requires the sealed PRE_REGISTRATION.md;
* installs the forbidden-writer guard at import and snapshots the repository outside generations/gen20_1 (plus
  the three gen20.1 documents) before scoring, failing if anything there changed;
* reads only ``fitting.load_tables`` output, the selection records and the label-free row manifests (plus
  COH-REPCAP's S5 manifest); writes results/<label>/*.csv|json and generations/gen20_1/GEN20_1_REPORT.md; when
  results/confirmation/report_state.json exists (S11 done), the confirmation status is rendered beside every
  verdict;
* every complete (design, variant) must carry one prediction code digest (DONE markers); it is stored as
  ``code_digest`` in report_state.json and printed; a confirmation state with another digest is refused.

Usage (repo root):  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B generations/gen20_1/scripts/g201_report.py
                    [--label discovery]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
G201 = Path(__file__).resolve().parents[1]
if str(G201) not in sys.path:
    sys.path.insert(0, str(G201))

from gen201 import bootstrap  # noqa: E402
from gen201 import guard  # noqa: E402

guard.install()

from gen201 import report as RP  # noqa: E402
from gen201.run import Run  # noqa: E402
from gen201.seal import require_sealed  # noqa: E402

CONFIRMATION_STATE = bootstrap.RESULTS / "confirmation" / "report_state.json"


def load_confirmation(label: str, path: Path = CONFIRMATION_STATE) -> dict | None:
    if label != "discovery" or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", default="discovery")
    args = ap.parse_args(argv)
    try:
        RP.refuse_smoke(args.label)                      # LT-29: before the seal check and before any read
    except RP.ReportRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    require_sealed(bootstrap.PREREG)
    before = guard.snapshot()
    with Run("g201_report", label=args.label, args=vars(args)) as run:
        res = RP.build(args.label, confirmation=load_confirmation(args.label), log=run.log)
        st = res["state"]
        run.record("labels", st.get("labels"))
        run.record("designs_complete", st.get("designs_complete"))
        run.record("code_digest", {"sha256": st.get("code_digest"), "by_design": st.get("code_digest_by_design")})
        run.record("outputs", [str(p) for p in res["paths"]])
        run.record("guard", guard.installed())
        changed = guard.diff(before, guard.snapshot())
        run.record("frozen_trees_changed", changed)
        if any(changed.values()):
            raise AssertionError(f"frozen trees changed during S10: { {k: v[:10] for k, v in changed.items() if v} }")
    print(res["markdown"].split("\n## Registered")[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())

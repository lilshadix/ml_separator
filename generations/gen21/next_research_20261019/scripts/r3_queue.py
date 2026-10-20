#!/usr/bin/env python
"""Run a queue of (config, design) jobs with at most N parallel processes (no oversubscription:
each job uses <= 4 threads).  Logs: results/<ID>_<design>.log.  Resumable: r3_run.py skips finished folds.

    .venv/Scripts/python.exe generations/gen21/next_research_20261019/scripts/r3_queue.py --jobs R01,R02,R03 --workers 4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

R3 = Path(__file__).resolve().parents[1]
ROOT = R3.parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="comma list of config IDs (both designs each)")
    ap.add_argument("--designs", default="BP,BPG")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    queue = [(c, d) for c in a.jobs.split(",") for d in a.designs.split(",")]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    running: list = []
    done = []
    while queue or running:
        while queue and len(running) < a.workers:
            c, d = queue.pop(0)
            log = open(R3 / "results" / f"{c}_{d}.log", "a", encoding="utf-8")
            log.write(f"\n===== start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            log.flush()
            p = subprocess.Popen([sys.executable, "-W", "ignore", str(R3 / "scripts" / "r3_run.py"), "--config", c, "--design", d],
                                 cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((c, d, p, log))
            print(f"started {c} {d}", flush=True)
        for item in list(running):
            c, d, p, log = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                done.append((c, d, p.returncode))
                print(f"finished {c} {d} exit {p.returncode}", flush=True)
        time.sleep(5)
    bad = [x for x in done if x[2] != 0]
    print("ALL DONE", "failures:", bad, flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

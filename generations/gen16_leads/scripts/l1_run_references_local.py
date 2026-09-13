"""Run the L1 Stage 2 reference-species xtb jobs LOCALLY, in parallel.

The hand-over assumed a cluster because `xtb` was not on this machine.  It is now, so this script
replaces `submit_xtb_references.sh` + `collect_reference_energies.sh` with one local run.  The
chemistry is byte-identical to the submit script: same binary, same GFN2 gas-phase level, same
charge convention, same `--opt` / `--sp` mode per species.

  361 tasks, ~6.7 CPU-hours serial, ~1 h wall at 6 workers.

Each task runs in its own directory `out/<species>/` because xtb writes scratch files into the
working directory.  Every worker gets `OMP_NUM_THREADS=1` so N workers use N cores.

**Resumable.**  A species whose `out/<species>/xtb.out` already holds a TOTAL ENERGY and a normal
termination is skipped, so an interrupted run continues where it stopped.  `--force` re-runs
everything.

Output `reference_energies.csv` has exactly the schema `l1_stage2_cycle.py` expects:
`species,energy_Eh,energy_eV,converged,xtb_version`.

Run:
    export PATH="/c/Users/Bandai/opt/xtb-6.7.1/bin:$PATH"      # or pass --xtb
    .venv/Scripts/python.exe gen16_leads/scripts/l1_run_references_local.py --workers 6
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REFDIR = HERE / "results" / "L1" / "reference_species"
OUT = REFDIR / "out"
HARTREE_EV = 27.211386245988
E_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")
V_RE = re.compile(r"xtb version\s+(\S+)")


def read_jobs() -> list[tuple[str, str, int, int, str]]:
    jobs = []
    for line in (REFDIR / "jobs.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        species, file, chrg, uhf, mode = line.split("\t")
        jobs.append((species, file, int(chrg), int(uhf), mode.strip()))
    return jobs


def parse(out_file: Path) -> tuple[float | None, str, str]:
    """(energy_Eh, converged, version) from an xtb.out, or (None, ...) if unusable."""
    if not out_file.exists():
        return None, "MISSING", ""
    txt = out_file.read_text(encoding="utf-8", errors="replace")
    e = E_RE.findall(txt)
    v = V_RE.search(txt)
    ok = ("normal termination" in txt) and bool(e)
    return (float(e[-1]) if e else None,
            "ok" if ok else "CHECK",
            v.group(1) if v else "")


def run_one(job, xtb: str, force: bool) -> dict:
    species, file, chrg, uhf, mode = job
    wd = OUT / species
    out_file = wd / "xtb.out"
    if not force:
        eh, conv, ver = parse(out_file)
        if eh is not None and conv == "ok":
            return {"species": species, "energy_Eh": eh, "converged": conv,
                    "xtb_version": ver, "seconds": 0.0, "cached": True}
    if wd.exists():
        shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True, exist_ok=True)
    shutil.copy(REFDIR / file, wd / "in.xyz")
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OMP_STACKSIZE"] = "1G"
    cmd = [xtb, "in.xyz", "--gfn", "2", "--chrg", str(chrg), "--uhf", str(uhf),
           "--opt" if mode == "opt" else "--sp"]
    t0 = time.time()
    try:
        with out_file.open("w", encoding="utf-8") as fh:
            subprocess.run(cmd, cwd=wd, stdout=fh, stderr=subprocess.STDOUT,
                           env=env, timeout=3600, check=False)
    except subprocess.TimeoutExpired:
        return {"species": species, "energy_Eh": None, "converged": "TIMEOUT",
                "xtb_version": "", "seconds": time.time() - t0, "cached": False}
    eh, conv, ver = parse(out_file)
    return {"species": species, "energy_Eh": eh, "converged": conv, "xtb_version": ver,
            "seconds": time.time() - t0, "cached": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--xtb", default=shutil.which("xtb") or "xtb")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N jobs (smoke test)")
    a = ap.parse_args()

    if not shutil.which(a.xtb) and not Path(a.xtb).exists():
        print(f"xtb not found ({a.xtb}). Put it on PATH or pass --xtb <path>.")
        return 2
    ver = subprocess.run([a.xtb, "--version"], capture_output=True, text=True).stdout
    print("using:", a.xtb, "|", (V_RE.search(ver).group(1) if V_RE.search(ver) else "?"), flush=True)

    jobs = read_jobs()
    if a.limit:
        jobs = jobs[:a.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs, {a.workers} workers", flush=True)

    rows, done, t0 = [], 0, time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, j, a.xtb, a.force): j[0] for j in jobs}
        for fut in cf.as_completed(futs):
            r = fut.result()
            rows.append(r)
            done += 1
            if r["converged"] != "ok" or done % 25 == 0 or done == len(jobs):
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                eta = (len(jobs) - done) / rate / 60 if rate > 0 else 0
                flag = "" if r["converged"] == "ok" else f"  <-- {r['species']} {r['converged']}"
                print(f"  {done:4d}/{len(jobs)}  {el/60:5.1f} min elapsed, ~{eta:4.1f} min left{flag}",
                      flush=True)

    rows.sort(key=lambda r: r["species"])
    bad = [r for r in rows if r["converged"] != "ok" or r["energy_Eh"] is None]
    csv = REFDIR / "reference_energies.csv"
    with csv.open("w", encoding="utf-8", newline="") as fh:
        fh.write("species,energy_Eh,energy_eV,converged,xtb_version\n")
        for r in rows:
            eh = r["energy_Eh"]
            ev = "" if eh is None else f"{eh * HARTREE_EV:.9f}"
            fh.write(f"{r['species']},{'' if eh is None else repr(eh)},{ev},"
                     f"{r['converged']},{r['xtb_version']}\n")
    cached = sum(1 for r in rows if r["cached"])
    print(f"\nwrote {csv.relative_to(HERE)}: {len(rows)} rows, {len(bad)} not ok, "
          f"{cached} reused from a previous run, {(time.time() - t0)/60:.1f} min wall")
    if bad:
        print("NOT OK:", ", ".join(f"{r['species']}({r['converged']})" for r in bad[:20]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

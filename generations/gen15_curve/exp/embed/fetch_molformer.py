"""Fetch MoLFormer-XL into a local directory with a *resumable* byte-range download.

``huggingface_hub``'s downloader kept starting a fresh ``.incomplete`` temp file after every stalled
connection, so the 187 MB ``model.safetensors`` never got past ~17 MB on this machine.  This does the
boring thing instead: HTTP Range requests into one file, resuming from whatever is already on disk,
retrying until the size matches.  Nothing here is specific to MoLFormer beyond the file list.

Usage:  python fetch_molformer.py [minutes_budget]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests

REPO = "ibm-research/MoLFormer-XL-both-10pct"
DEST = Path(os.environ.get("TEMP", "/tmp")) / "molformer_xl"
FILES = ["config.json", "configuration_molformer.py", "modeling_molformer.py",
         "tokenizer.json", "tokenizer_config.json", "model.safetensors"]
BASE = f"https://huggingface.co/{REPO}/resolve/main/"


def expected_size(name: str) -> int | None:
    """Declared size of a repo file.  DNS on this machine drops intermittently, so retry."""
    for _ in range(10):
        try:
            r = requests.head(BASE + name, allow_redirects=True, timeout=30)
            n = r.headers.get("x-linked-size") or r.headers.get("Content-Length")
            if n:
                return int(n)
        except Exception as exc:                                       # noqa: BLE001
            print(f"  HEAD {name}: {type(exc).__name__}, retrying", flush=True)
        time.sleep(3)
    return None


def fetch(name: str, budget_s: float) -> tuple[bool, int, int | None]:
    out = DEST / name
    out.parent.mkdir(parents=True, exist_ok=True)
    want = expected_size(name)
    t0 = time.time()
    while time.time() - t0 < budget_s:
        have = out.stat().st_size if out.exists() else 0
        if want is not None and have >= want:
            return True, have, want
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(BASE + name, headers=headers, stream=True, timeout=(30, 60)) as r:
                if r.status_code not in (200, 206):
                    print(f"  {name}: HTTP {r.status_code}", flush=True)
                    time.sleep(3)
                    continue
                mode = "ab" if (have and r.status_code == 206) else "wb"
                with open(out, mode) as fh:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            fh.write(chunk)
        except Exception as exc:                                       # noqa: BLE001
            print(f"  {name}: {type(exc).__name__} at {out.stat().st_size if out.exists() else 0}"
                  f"/{want} bytes, resuming", flush=True)
            time.sleep(2)
    have = out.stat().st_size if out.exists() else 0
    return (want is not None and have >= want), have, want


def main() -> None:
    budget = float(sys.argv[1]) * 60 if len(sys.argv) > 1 else 25 * 60
    per = budget / len(FILES)
    ok_all = True
    for f in FILES:
        share = budget - per * (len(FILES) - 1) if f == "model.safetensors" else per
        ok, have, want = fetch(f, max(share, 60))
        print(f"[fetch] {f}: {'OK' if ok else 'INCOMPLETE'} {have}/{want}", flush=True)
        ok_all &= ok
    print(f"[fetch] destination {DEST}")
    print("[fetch] ALL OK" if ok_all else "[fetch] INCOMPLETE")


if __name__ == "__main__":
    main()

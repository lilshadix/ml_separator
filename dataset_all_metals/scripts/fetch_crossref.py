"""Fetch authoritative publication metadata for every DOI in the archive.

Run this once; it writes ``cache/crossref_metadata.json``, which the pipeline
then reads as a fixed input.  Keeping the network call out of the pipeline is
what lets ``run_all.py --verify-determinism`` stay meaningful: reruns read the
cache and never touch the network.

**Scope, deliberately narrow.** This fills *bibliographic* fields only — title,
year, journal, authors, publisher. It does NOT attempt to recover experimental
conditions (metal concentration, contact time, the measured D, or the identity
of the metal) from the papers. Those values live in figures and tables, differ
per row, and would have to be matched to individual digitised points by
judgement. Getting one of them wrong would silently corrupt the target or the
conditions of a training row, which is far worse than leaving it missing, so
those gaps stay in the manual-review queue instead.

Usage::

    ../.venv/bin/python scripts/fetch_crossref.py
    ../.venv/bin/python scripts/fetch_crossref.py --refresh   # ignore the cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import certifi
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sae_paths import ROOT, INTERMEDIATE_DIR  # noqa: E402

CACHE = ROOT / "cache" / "crossref_metadata.json"
# Crossref asks for a contact address in the User-Agent so they can reach the
# operator of a badly-behaved client; it also earns the faster "polite" pool.
USER_AGENT = "sae-dataset-audit/1.0 (mailto:bogdan_mironov@icloud.com)"
REQUEST_DELAY_S = 0.2


# This interpreter has no system CA bundle wired into ssl, so an unconfigured
# urlopen fails with CERTIFICATE_VERIFY_FAILED.  certifi ships one; use it
# explicitly rather than disabling verification.
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def fetch_one(doi: str) -> dict:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30, context=_SSL_CONTEXT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"doi": doi, "status": f"http_{exc.code}"}
    except Exception as exc:                       # network / parse failure
        return {"doi": doi, "status": "error", "error": f"{exc.__class__.__name__}: {exc}"}

    message = payload.get("message", {})
    authors = []
    for person in message.get("author", []) or []:
        name = " ".join(x for x in (person.get("given"), person.get("family")) if x)
        if name:
            authors.append(name)
    issued = (message.get("issued", {}).get("date-parts") or [[None]])[0]
    titles = message.get("title") or []
    containers = message.get("container-title") or []
    return {
        "doi": doi,
        "status": "ok",
        "crossref_title": titles[0] if titles else None,
        "crossref_year": issued[0] if issued else None,
        "crossref_journal": containers[0] if containers else None,
        "crossref_publisher": message.get("publisher"),
        "crossref_type": message.get("type"),
        "crossref_authors": "; ".join(authors) or None,
        "crossref_url": message.get("URL"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="ignore the existing cache")
    args = parser.parse_args()

    records = pd.read_parquet(INTERMEDIATE_DIR / "records_normalized.parquet",
                              columns=["doi_all"])
    import sae_normalize as NORM
    raw_dois = {d for row in records["doi_all"] if row is not None for d in row}
    # Look up the repaired form as well, so a malformed DOI still yields metadata.
    dois = sorted(raw_dois | {NORM.correct_doi(d)[0] for d in raw_dois})
    print(f"{len(dois)} distinct DOIs in the archive")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = {} if args.refresh or not CACHE.exists() else json.loads(CACHE.read_text())

    pending = [d for d in dois if d not in cache]
    print(f"{len(cache)} cached, {len(pending)} to fetch")
    for index, doi in enumerate(pending, 1):
        cache[doi] = fetch_one(doi)
        if index % 20 == 0 or index == len(pending):
            print(f"  {index}/{len(pending)}", flush=True)
        time.sleep(REQUEST_DELAY_S)

    CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    statuses = pd.Series([v.get("status") for v in cache.values()]).value_counts()
    print("\nresolution status:")
    print(statuses.to_string())
    resolved = sum(1 for v in cache.values() if v.get("status") == "ok")
    print(f"\n{resolved}/{len(cache)} DOIs resolved -> {CACHE}")


if __name__ == "__main__":
    main()

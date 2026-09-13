"""Protocol invariants of gen16: the seal, the seed withholding, the confirmation commitment.

Run:  .venv/Scripts/python.exe -m pytest gen16_leads/tests/test_protocol.py -q
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PY = sys.executable

DISCOVERY = (104729, 130363, 155921, 196613, 262147)


def _prereg_text() -> str:
    return (HERE / "PRE_REGISTRATION.md").read_text(encoding="utf-8")


def test_pre_registration_seal_verifies():
    r = subprocess.run([PY, str(HERE / "scripts" / "g16_prereg_hash.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.startswith("OK")


def test_seed_audit_is_clean():
    r = subprocess.run([PY, str(HERE / "scripts" / "g16_audit_seeds.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_confirmation_rule_reproduces_commitment():
    """The registered rule, applied verbatim, hashes to the commitment in PRE_REGISTRATION.md.

    This does not print the seeds; it only checks that the commitment is the hash of what the
    public rule produces, so the confirmation run cannot silently use a different seed set.
    """
    m = re.search(r"commitment is `sha256\(json\.dumps\(seeds\)\)` =\s*`([0-9a-f]{64})`", _prereg_text())
    assert m, "commitment line missing from PRE_REGISTRATION.md"
    seeds: list[int] = []
    i = 1
    while len(seeds) < 5:
        h = hashlib.sha256(f"gen16-confirmation-seed-{i}".encode()).hexdigest()
        s = int(h[:8], 16) % 900_000 + 100_000
        if s not in DISCOVERY and s not in seeds:
            seeds.append(s)
        i += 1
    digest = hashlib.sha256(json.dumps(sorted(seeds)).encode()).hexdigest()
    assert digest == m.group(1)
    assert not set(seeds) & set(DISCOVERY)
    assert len(set(seeds)) == 5


def test_discovery_seeds_are_the_frozen_defaults():
    sys.path.insert(0, str(HERE.parent / "gen13_separation"))
    from gen13sep.splits import SPLIT_SEEDS  # noqa: E402
    assert tuple(SPLIT_SEEDS) == DISCOVERY


def test_claims_registry_has_at_most_five():
    sys.path.insert(0, str(HERE))
    from gen16.claims import CLAIMS  # noqa: E402
    assert len(CLAIMS) <= 5

"""``g19_seal_prereg.py`` -- seal, check and seed-commit the Gen19 pre-registration (brief section 21).

Modes (run from the repository root with ``.venv/Scripts/python.exe``):

``--check``
    Read-only.  Verifies ``preregistration.md``: exactly one footer line, the SHA-256 in it equals the
    digest recomputed from everything above it, ``manifests/prereg_sha256.txt`` holds the same digest,
    the confirmation-seed commitment is quoted above the footer, and everything below the footer is a
    sequence of dated POST-HOC addenda.  When nothing is sealed yet it prints what still blocks sealing
    (the readiness report) and exits 1.  Exit 0 only when sealed and intact.  Fit scripts call it.
``--seal``
    Copies ``preregistration_draft.md`` to ``preregistration.md`` with the footer appended and writes
    the digest to ``manifests/prereg_sha256.txt``.  Refuses (exit 2) unless ``DATA_AUDIT.md`` and
    ``FEASIBILITY.md`` exist, the placeholder marker :data:`PLACEHOLDER` and the draft banner
    :data:`DRAFT_BANNER` are gone, no markdown checklist box is left unticked (``- [ ]``, the pre-seal
    checklist), every section brief section 21 requires has a heading, the draft carries no footer line of
    its own, and the digest written by ``--commit-seeds`` is quoted in the draft.  Refuses to overwrite a sealed ``preregistration.md`` whose text differs from the draft.
``--commit-seeds --seed-store PATH``
    Draws :data:`N_CONFIRMATION` confirmation seeds with :mod:`secrets` (not a public rule: unlike gen16
    they cannot be recomputed), writes them with a random salt to ``PATH`` -- which must lie OUTSIDE the
    repository -- and writes only ``sha256(canonical JSON)`` to ``manifests/confirmation_seeds_sha256.txt``.
    Prints the digest, never the seeds.  Refuses when a commitment or the store already exists (a
    second draw after seeing the first would be seed shopping).
``--verify-seeds --seed-store PATH``
    Checks a revealed store against the committed digest; prints the verdict, never the seeds.

Footer mechanism (reused from ``generations/gen18_process/scripts/g18_seal_prereg.py``): the last line of
the sealed text above the addenda is ``Sealed SHA-256 of everything above this line: `<hex>```; the
digest is ``sha256((above + "\\n").encode("utf-8"))`` where ``above`` is the text before the footer
line with CRLF normalised to LF (and a UTF-8 BOM dropped), so Windows and Linux checkouts agree.

Portability fix over gen18 (brief section 24): gen18 wrote ``prereg_sha256.txt`` with
``Path.write_text``, which on Windows translates ``\\n`` to CRLF, so the file's binary hash differed
between platforms.  Every file this script writes is opened with ``newline="\\n"``.

POST-HOC rule: after sealing, the text above the footer is never edited.  A change to the registered
analysis is appended BELOW the footer as a section headed
``## POST-HOC addendum <n> (<YYYY-MM-DD>, <author>; results seen: <yes|no>)`` -- numbered from 1,
dates non-decreasing -- stating what changed, why, and whether any outcome had been seen.  The digest
does not cover the addenda, so adding one keeps ``--check`` green; editing anything above the footer
turns it red.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen19ct import paths  # noqa: E402

FOOTER_PREFIX = "Sealed SHA-256 of everything above this line:"
#: the marker every unresolved value in the draft carries; sealing refuses while any remains
PLACEHOLDER = "TO BE FINALISED"
#: the banner of the draft; sealing refuses while it is present (it must be replaced by a status line)
DRAFT_BANNER = "DRAFT / UNSEALED"
#: an unticked markdown checklist box (the pre-seal checklist); sealing refuses while any remains
_UNCHECKED_BOX = re.compile(r"^\s*[-*+]\s+\[ \]")
#: separator written between the draft body and the footer line (covered by the digest)
SEAL_SEPARATOR = "\n\n---"
#: brief section 21 contents (plus seeds and deviations): each must appear in some markdown heading
REQUIRED_SECTIONS: dict[str, str] = {
    "primary hypothesis": r"primary hypothes",
    "primary metrics": r"metric",
    "primary holdouts": r"hold-?out",
    "success thresholds": r"success",
    "baselines": r"baseline",
    "allowed model families": r"model famil",
    "tuning procedure": r"tuning",
    "uncertainty evaluation": r"uncertainty",
    "process evaluation": r"process",
    "failure conditions": r"failure",
    "seeds": r"seed",
    "deviations from the brief": r"deviation",
}
DISCOVERY_SEEDS: tuple[int, ...] = (104729, 130363, 155921, 196613, 262147)
N_CONFIRMATION = 5
SEED_LOW, SEED_HIGH = 100_000, 999_999
SEED_SCHEMA = "gen19.confirmation_seeds.v1"

_BOM = chr(0xFEFF)
_HEX_IN_FOOTER = re.compile(r"`([0-9a-f]{64})`")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ADDENDUM = re.compile(r"^## POST-HOC addendum (\d+) \((\d{4}-\d{2}-\d{2})\b[^)]*\)\s*$")


@dataclass(frozen=True)
class PreregPaths:
    """Every file the sealing touches, rooted at the gen19 directory (tests pass a ``tmp_path``)."""

    root: Path
    repo_root: Path

    @property
    def draft(self) -> Path:
        return self.root / "preregistration_draft.md"

    @property
    def sealed(self) -> Path:
        return self.root / "preregistration.md"

    @property
    def data_audit(self) -> Path:
        return self.root / "DATA_AUDIT.md"

    @property
    def feasibility(self) -> Path:
        return self.root / "FEASIBILITY.md"

    @property
    def sha_file(self) -> Path:
        return self.root / "manifests" / "prereg_sha256.txt"

    @property
    def seed_commitment(self) -> Path:
        return self.root / "manifests" / "confirmation_seeds_sha256.txt"


def default_paths() -> PreregPaths:
    return PreregPaths(root=paths.G19_ROOT, repo_root=paths.REPO_ROOT)


# ------------------------------------------------------------------------------------------------ #
# Text handling
# ------------------------------------------------------------------------------------------------ #

def normalise(text: str) -> str:
    """Drop a UTF-8 BOM and normalise CRLF to LF (the only transformations before hashing)."""
    if text.startswith(_BOM):
        text = text[1:]
    return text.replace("\r\n", "\n")


def read_normalised(path: Path) -> str:
    return normalise(Path(path).read_bytes().decode("utf-8"))


def write_lf(path: Path, text: str) -> None:
    """Write UTF-8 with LF line endings on every platform (the gen18 portability fix)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def digest_of_above(above: str) -> str:
    return hashlib.sha256((normalise(above) + "\n").encode("utf-8")).hexdigest()


def footer_line_indices(text: str) -> list[int]:
    return [i for i, ln in enumerate(normalise(text).split("\n")) if ln.startswith(FOOTER_PREFIX)]


def split_footer(text: str) -> tuple[str, str, str]:
    """``(above, footer_line, below)``.  Raises ``ValueError`` unless exactly one footer line exists."""
    lines = normalise(text).split("\n")
    idx = footer_line_indices(text)
    if not idx:
        raise ValueError(f"no footer line starting with {FOOTER_PREFIX!r}")
    if len(idx) > 1:
        raise ValueError(f"{len(idx)} footer lines (lines {[i + 1 for i in idx]}); exactly one is allowed")
    i = idx[0]
    return "\n".join(lines[:i]), lines[i], "\n".join(lines[i + 1:])


def footer_digest(footer: str) -> str | None:
    m = _HEX_IN_FOOTER.search(footer)
    return m.group(1) if m else None


def build_sealed_text(draft_text: str) -> tuple[str, str]:
    """``(sealed_text, digest)`` for a draft: body + separator + footer line + LF."""
    above = normalise(draft_text).rstrip("\n") + SEAL_SEPARATOR
    digest = digest_of_above(above)
    return f"{above}\n{FOOTER_PREFIX} `{digest}`\n", digest


def placeholder_lines(text: str) -> list[int]:
    """1-based line numbers carrying :data:`PLACEHOLDER`."""
    return [i + 1 for i, ln in enumerate(normalise(text).split("\n")) if PLACEHOLDER in ln]


def unchecked_box_lines(text: str) -> list[int]:
    """1-based line numbers of unticked markdown checklist boxes (``- [ ]``)."""
    return [i + 1 for i, ln in enumerate(normalise(text).split("\n")) if _UNCHECKED_BOX.match(ln)]


def headings(text: str) -> list[str]:
    return [ln.lstrip("#").strip() for ln in normalise(text).split("\n") if re.match(r"^#{1,6}\s", ln)]


def missing_sections(text: str) -> list[str]:
    heads = [h.lower() for h in headings(text)]
    return [name for name, rx in REQUIRED_SECTIONS.items() if not any(re.search(rx, h) for h in heads)]


def below_footer_problems(below: str) -> list[str]:
    """Everything below the footer must be blank or dated, consecutively numbered POST-HOC addenda."""
    problems: list[str] = []
    expected, last_date, seen_heading = 1, None, False
    for k, ln in enumerate(normalise(below).split("\n"), start=1):
        if ln.startswith("## "):
            m = _ADDENDUM.match(ln)
            if not m:
                problems.append(f"below-footer line {k}: heading {ln[:60]!r} is not "
                                "'## POST-HOC addendum <n> (<YYYY-MM-DD>, ...)'")
                continue
            n = int(m.group(1))
            try:
                d = _dt.date.fromisoformat(m.group(2))
            except ValueError:
                problems.append(f"below-footer line {k}: invalid date {m.group(2)!r}")
                continue
            if n != expected:
                problems.append(f"below-footer line {k}: addendum {n} where {expected} was expected")
            if last_date is not None and d < last_date:
                problems.append(f"below-footer line {k}: addendum dated {d} precedes {last_date}")
            expected, last_date, seen_heading = n + 1, d, True
        elif not seen_heading and ln.strip():
            problems.append(f"below-footer line {k}: text outside a POST-HOC addendum")
    return problems


def read_digest_file(path: Path) -> str | None:
    if not Path(path).exists():
        return None
    tokens = read_normalised(path).split()
    return tokens[0] if tokens and _HEX64.match(tokens[0]) else None


# ------------------------------------------------------------------------------------------------ #
# Seal / check
# ------------------------------------------------------------------------------------------------ #

def readiness(p: PreregPaths) -> list[str]:
    """Everything that blocks ``--seal`` (empty list = ready)."""
    problems: list[str] = []
    for label, f in (("DATA_AUDIT.md", p.data_audit), ("FEASIBILITY.md", p.feasibility)):
        if not f.exists():
            problems.append(f"{label} does not exist ({f})")
    if not p.draft.exists():
        problems.append(f"preregistration_draft.md does not exist ({p.draft})")
        return problems
    text = read_normalised(p.draft)
    ph = placeholder_lines(text)
    if ph:
        shown = ", ".join(str(n) for n in ph[:8]) + (" ..." if len(ph) > 8 else "")
        problems.append(f"{len(ph)} line(s) still carry the placeholder marker {PLACEHOLDER!r} (lines {shown})")
    if DRAFT_BANNER in text:
        problems.append(f"the draft banner {DRAFT_BANNER!r} is still present")
    boxes = unchecked_box_lines(text)
    if boxes:
        problems.append(f"{len(boxes)} checklist box(es) still unticked (lines {', '.join(str(n) for n in boxes)})")
    if footer_line_indices(text):
        problems.append("the draft already contains a footer line; the footer is written by --seal only")
    miss = missing_sections(text)
    if miss:
        problems.append("no heading for required section(s): " + ", ".join(miss))
    commitment = read_digest_file(p.seed_commitment)
    if commitment is None:
        problems.append(f"no confirmation-seed commitment ({p.seed_commitment}); run --commit-seeds first")
    elif commitment not in text:
        problems.append(f"the confirmation-seed commitment {commitment[:12]}... is not quoted in the draft")
    return problems


def seal(p: PreregPaths) -> tuple[int, str]:
    problems = readiness(p)
    if problems:
        return 2, "refused:\n  - " + "\n  - ".join(problems)
    draft_text = read_normalised(p.draft)
    sealed_text, digest = build_sealed_text(draft_text)
    if p.sealed.exists():
        existing = read_normalised(p.sealed)
        try:
            above, footer, _ = split_footer(existing)
        except ValueError as exc:
            return 2, f"refused: {p.sealed.name} exists but is not a sealed file ({exc}); remove it deliberately"
        recorded = footer_digest(footer)
        if recorded == digest and digest_of_above(above) == digest:
            write_lf(p.sha_file, digest + "\n")
            return 0, f"already sealed: {digest}"
        return 2, (f"refused: {p.sealed.name} is already sealed with a different text. The draft is frozen "
                   "once sealed; a change to the registered analysis is a dated POST-HOC addendum appended "
                   f"below the footer of {p.sealed.name}.")
    write_lf(p.sealed, sealed_text)
    write_lf(p.sha_file, digest + "\n")
    return 0, f"sealed: {digest} (footer of {p.sealed.name} and {p.sha_file.name})"


def check(p: PreregPaths) -> tuple[bool, list[str]]:
    """``(ok, messages)``.  Read-only."""
    if not p.sealed.exists():
        msgs = [f"unsealed: {p.sealed.name} does not exist"]
        blockers = readiness(p)
        msgs += ["sealing is blocked by:"] + [f"  - {b}" for b in blockers] if blockers \
            else ["the draft is ready to seal (run --seal)"]
        return False, msgs
    text = read_normalised(p.sealed)
    try:
        above, footer, below = split_footer(text)
    except ValueError as exc:
        return False, [f"invalid: {exc}"]
    recorded = footer_digest(footer)
    if recorded is None:
        return False, ["unsealed: the footer line carries no digest"]
    actual = digest_of_above(above)
    problems: list[str] = []
    if recorded != actual:
        problems.append(f"mismatch: footer {recorded[:12]}... != recomputed {actual[:12]}... "
                        "(text above the footer was edited after sealing)")
    on_disk = read_digest_file(p.sha_file)
    if on_disk is None:
        problems.append(f"mismatch: {p.sha_file.name} missing or unreadable")
    elif on_disk != actual:
        problems.append(f"mismatch: {p.sha_file.name} {on_disk[:12]}... != recomputed {actual[:12]}...")
    commitment = read_digest_file(p.seed_commitment)
    if commitment is None:
        problems.append(f"missing: {p.seed_commitment.name}")
    elif commitment not in above:
        problems.append("mismatch: the confirmation-seed commitment is not quoted above the footer")
    problems += below_footer_problems(below)
    if problems:
        return False, problems
    msgs = [f"sealed: {actual}"]
    n_add = sum(1 for ln in below.split("\n") if _ADDENDUM.match(ln))
    msgs.append(f"POST-HOC addenda below the footer: {n_add}")
    if p.draft.exists():
        body = normalise(read_normalised(p.draft)).rstrip("\n") + SEAL_SEPARATOR
        if body != above:
            msgs.append("warning: preregistration_draft.md differs from the sealed text (the sealed file governs)")
    return True, msgs


# ------------------------------------------------------------------------------------------------ #
# Confirmation seeds
# ------------------------------------------------------------------------------------------------ #

def _canonical_payload(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def seed_digest(payload: dict) -> str:
    return hashlib.sha256(_canonical_payload(payload).encode("utf-8")).hexdigest()


def draw_confirmation_seeds(n: int = N_CONFIRMATION, randbelow=secrets.randbelow) -> list[int]:
    out: set[int] = set()
    while len(out) < n:
        s = SEED_LOW + randbelow(SEED_HIGH - SEED_LOW + 1)
        if s not in DISCOVERY_SEEDS:
            out.add(s)
    return sorted(out)


def _outside(path: Path, repo_root: Path) -> bool:
    return not Path(path).resolve().is_relative_to(Path(repo_root).resolve())


def commit_seeds(p: PreregPaths, store: Path, n: int = N_CONFIRMATION, randbelow=secrets.randbelow,
                 token_hex=secrets.token_hex) -> tuple[int, str]:
    store = Path(store)
    if not _outside(store, p.repo_root):
        return 2, f"refused: the seed store must lie outside the repository ({p.repo_root})"
    if p.seed_commitment.exists():
        return 2, (f"refused: {p.seed_commitment.name} already exists; seeds are committed once "
                   "(a second draw after seeing the first would be seed shopping)")
    if store.exists():
        return 2, f"refused: {store} already exists; it is never overwritten"
    payload = {"n": n, "salt": token_hex(32), "schema": SEED_SCHEMA,
               "seeds": draw_confirmation_seeds(n, randbelow)}
    digest = seed_digest(payload)
    write_lf(store, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_lf(p.seed_commitment, digest + "\n")
    return 0, (f"committed: {digest}\n  written to {p.seed_commitment} (the digest only); the seeds are in the "
               "store outside the repository and are not printed. Quote the digest in the draft's seeds section.")


def load_committed_seeds(p: PreregPaths, store: Path) -> list[int]:
    """The confirmation seeds, verified against the commitment; raises ``ValueError`` on any mismatch."""
    commitment = read_digest_file(p.seed_commitment)
    if commitment is None:
        raise ValueError(f"no commitment at {p.seed_commitment}")
    payload = json.loads(read_normalised(Path(store)))
    if payload.get("schema") != SEED_SCHEMA:
        raise ValueError("seed store has an unexpected schema")
    seeds = payload.get("seeds")
    if (not isinstance(seeds, list) or len(seeds) != payload.get("n") or seeds != sorted(set(seeds))
            or any(not isinstance(s, int) or not SEED_LOW <= s <= SEED_HIGH or s in DISCOVERY_SEEDS for s in seeds)):
        raise ValueError("seed store is malformed")
    if seed_digest(payload) != commitment:
        raise ValueError("seed store does not hash to the committed digest")
    return list(seeds)


def verify_seed_store(p: PreregPaths, store: Path) -> tuple[bool, str]:
    try:
        seeds = load_committed_seeds(p, store)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return False, f"MISMATCH: {exc}"
    return True, f"verified: {len(seeds)} seeds hash to the committed digest (seeds not printed)"


# ------------------------------------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------------------------------------ #

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify the sealed file (read-only); exit 1 if not intact")
    mode.add_argument("--seal", action="store_true", help="seal preregistration_draft.md into preregistration.md")
    mode.add_argument("--commit-seeds", action="store_true", help="draw and commit the confirmation seeds")
    mode.add_argument("--verify-seeds", action="store_true", help="verify a revealed seed store")
    ap.add_argument("--seed-store", type=Path, help="seed store path (outside the repository)")
    ap.add_argument("--root", type=Path, default=None, help=argparse.SUPPRESS)  # tests only
    args = ap.parse_args(argv)

    p = default_paths() if args.root is None else PreregPaths(root=args.root, repo_root=args.root)
    real = args.root is None

    if args.check:
        ok, msgs = check(p)
        print("\n".join(msgs))
        return 0 if ok else 1
    if args.verify_seeds:
        if args.seed_store is None:
            ap.error("--verify-seeds needs --seed-store")
        ok, msg = verify_seed_store(p, args.seed_store)
        print(msg)
        return 0 if ok else 1
    if args.commit_seeds:
        if args.seed_store is None:
            ap.error("--commit-seeds needs --seed-store")
        if not real:
            code, msg = commit_seeds(p, args.seed_store)
            print(msg)
            return code
        from gen19ct.manifest import Run
        if p.seed_commitment.exists() or not _outside(args.seed_store, p.repo_root) or args.seed_store.exists():
            code, msg = commit_seeds(p, args.seed_store)   # refuses without writing anything
            print(msg)
            return code
        with Run("g19_commit_seeds", args={"mode": "commit-seeds", "n": N_CONFIRMATION}) as run:
            code, msg = commit_seeds(p, args.seed_store)
            run.outputs(p.seed_commitment)
        print(msg)
        return code
    # --seal
    problems = readiness(p)
    if problems or not real:
        code, msg = seal(p)
        print(msg)
        return code
    from gen19ct.manifest import Run
    with Run("g19_seal_prereg", args={"mode": "seal"}) as run:
        code, msg = seal(p)
        run.inputs(p.draft, p.data_audit, p.feasibility, p.seed_commitment)
        run.outputs(p.sealed, p.sha_file)
    print(msg)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

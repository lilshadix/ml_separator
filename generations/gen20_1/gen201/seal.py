"""Seal mechanics for gen20.1 (re-implemented with the file path as a parameter; gen20k.manifest's
sealer is hard-wired to gen20's PRE_REGISTRATION.md and is forbidden, IMPLEMENTATION_PLAN.md §8).

The digest is the SHA-256 of the **LF-normalised** UTF-8 text (this clone has core.autocrlf = true,
so a raw-byte hash would differ between a CRLF checkout and the committed LF blob).  A sealed file
ends with exactly one line ``SEALED-SHA256: <hex>`` where ``<hex>`` is the digest of every line above
it, joined with "\\n" and terminated by one "\\n" (gen20's convention, gen20k/manifest.py:48-79, but over
LF-normalised text).  Sealing is refused while any draft marker remains (O19).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

SEAL_LINE = "SEALED-SHA256:"
DRAFT_MARKERS: tuple[str, ...] = ("DRAFT / UNSEALED", "TO BE FINALISED", "- [ ]")


def lf_text(path) -> str:
    """The file's UTF-8 text with CRLF (and stray CR) normalised to LF."""
    raw = Path(path).read_bytes().decode("utf-8")
    return raw.replace("\r\n", "\n").replace("\r", "\n")


def lf_digest(path) -> str:
    """SHA-256 of the LF-normalised UTF-8 text of the whole file."""
    return hashlib.sha256(lf_text(path).encode("utf-8")).hexdigest()


def _split(text: str) -> tuple[list[str], bool]:
    lines = text.rstrip("\n").split("\n")
    return lines, bool(lines) and lines[-1].startswith(SEAL_LINE)


def body_digest(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def markers_remaining(text: str) -> dict[str, int]:
    return {m: text.count(m) for m in DRAFT_MARKERS if text.count(m)}


def preregistration_state(path) -> dict:
    """``sealed`` (a footer is present), ``valid`` (its hex equals the body digest), ``digest`` (the
    body digest, i.e. of everything above the footer, or of the whole LF text when unsealed),
    ``recorded`` (the footer's hex), ``markers_remaining`` (draft markers in the body)."""
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False, "sealed": False, "valid": False, "digest": None,
                "recorded": None, "markers_remaining": {}}
    text = lf_text(p)
    lines, sealed = _split(text)
    body = "\n".join(lines[:-1]) + "\n" if sealed else text
    recorded = lines[-1].split(":", 1)[1].strip() if sealed else None
    digest = body_digest(body)
    return {"path": str(p), "exists": True, "sealed": sealed, "valid": bool(sealed and recorded == digest),
            "digest": digest, "recorded": recorded, "lf_digest_whole_file": lf_digest(p),
            "markers_remaining": markers_remaining(body)}


def seal(path) -> str:
    """Append ``SEALED-SHA256: <hex>``; refuse while a draft marker remains or a footer exists.

    The file keeps its line-ending style (CRLF if the original had CRLF, else LF); the digest is over
    the LF-normalised body either way.  Returns the hex digest.
    """
    p = Path(path)
    raw = p.read_bytes().decode("utf-8")
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines, sealed = _split(text)
    if sealed:
        raise RuntimeError(f"{p.name} already carries a seal footer; changes after the seal are dated addenda")
    left = markers_remaining(text)
    if left:
        raise RuntimeError(f"refusing to seal {p.name}: draft markers remain {left}")
    body = "\n".join(lines) + "\n"
    digest = body_digest(body)
    out = body + f"{SEAL_LINE} {digest}\n"
    if "\r\n" in raw:
        out = out.replace("\n", "\r\n")
    p.write_bytes(out.encode("utf-8"))
    st = preregistration_state(p)
    if not st["valid"] or st["digest"] != digest:  # pragma: no cover - self-check
        raise RuntimeError("seal self-check failed")
    return digest


def require_sealed(path) -> dict:
    """Raise unless the last line is ``SEALED-SHA256: <hex>``, the hex equals the SHA-256 of the
    LF-normalised UTF-8 text of every line above it, and no draft marker remains in that body (a footer
    appended by hand or by another tool over an unfinished pre-registration is refused, as ``seal`` would)."""
    st = preregistration_state(path)
    if not st["exists"]:
        raise RuntimeError(f"{path} does not exist")
    if not st["sealed"]:
        raise RuntimeError(f"{Path(path).name} is not sealed; outcome-bearing runs are refused")
    if not st["valid"]:
        raise RuntimeError(f"{Path(path).name}: seal footer {st['recorded']} does not match the body digest {st['digest']}")
    if st["markers_remaining"]:
        raise RuntimeError(f"{Path(path).name} carries a valid footer but draft markers remain in its body "
                           f"{st['markers_remaining']}; outcome-bearing runs are refused")
    return st

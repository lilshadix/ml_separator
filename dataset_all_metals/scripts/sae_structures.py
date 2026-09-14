"""RDKit canonicalisation with an explicit, inspectable failure channel."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sae_normalize import clean_text, is_blank  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RDKIT_VERSION = __import__("rdkit").__version__


@dataclass(frozen=True)
class CanonicalStructure:
    raw: str | None
    canonical: str | None
    inchikey: str | None
    status: str          # ok | blank | parse_failure
    error: str | None


_cache: dict[str, CanonicalStructure] = {}


def canonicalize(smiles) -> CanonicalStructure:
    """Canonical SMILES + InChIKey, or a recorded failure.

    A parse failure is never silently dropped: the raw string is preserved and
    ``status`` says why, so ``rdkit_failures.csv`` can list every one.
    """
    text = clean_text(smiles)
    if text is None or is_blank(text):
        return CanonicalStructure(None, None, None, "blank", None)
    hit = _cache.get(text)
    if hit is not None:
        return hit

    try:
        mol = Chem.MolFromSmiles(text)
    except Exception as exc:                      # pragma: no cover - defensive
        result = CanonicalStructure(text, None, None, "parse_failure", repr(exc))
        _cache[text] = result
        return result

    if mol is None:
        result = CanonicalStructure(text, None, None, "parse_failure", "MolFromSmiles returned None")
    else:
        try:
            key = Chem.MolToInchiKey(mol) or None
        except Exception:
            key = None
        result = CanonicalStructure(text, Chem.MolToSmiles(mol), key, "ok", None)
    _cache[text] = result
    return result

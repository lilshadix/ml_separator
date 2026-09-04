#!/usr/bin/env python
"""Check every exported figure: PNG resolution, PDF vector-ness, and PNG/PDF agreement.

The user-visible requirement is that the vector PDF must not introduce clipping or a
layout change relative to the raster PNG.  Both are written from one figure object with
one bounding box, so the check that matters is that their aspect ratios and page sizes
agree; a mismatch means one of them was saved with different bbox settings.

Run:  .venv/bin/python figure_refinement/common/verify_exports.py [root]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from PIL import Image

MIN_DPI = 300
ASPECT_TOL = 0.01


def pdf_page_size(path: Path) -> tuple[float, float] | None:
    """First MediaBox of a PDF, in points, without a PDF library."""
    blob = path.read_bytes()
    match = re.search(rb"/MediaBox\s*\[\s*([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)",
                      blob)
    if not match:
        return None
    x0, y0, x1, y1 = (float(v) for v in match.groups())
    return abs(x1 - x0), abs(y1 - y0)


def has_raster_payload(path: Path) -> bool:
    """True if the PDF embeds a bitmap image — i.e. it is not fully vector."""
    blob = path.read_bytes()
    return b"/Subtype /Image" in blob or b"/Subtype/Image" in blob


def main(root: str = "figure_refinement") -> int:
    rows = []
    # Only the deliverable itself is checked.  A folder may also hold original.png — the
    # copy of the source render kept for the before/after record — which has no PDF
    # companion by design and is not a deliverable.
    for png in sorted(Path(root).rglob("figure.png")):
        if "/common/" in str(png):
            continue
        pdf = png.with_suffix(".pdf")
        if not pdf.exists():
            rows.append({"figure": str(png.parent), "status": "FAIL",
                         "problem": "no PDF export"})
            continue
        with Image.open(png) as im:
            px_w, px_h = im.size
            dpi = im.info.get("dpi", (0, 0))[0] or 0
        size = pdf_page_size(pdf)
        problems = []
        if dpi and dpi < MIN_DPI:
            problems.append(f"PNG is {dpi:.0f} dpi (< {MIN_DPI})")
        if size is None:
            problems.append("PDF MediaBox unreadable")
        else:
            png_aspect = px_w / px_h
            pdf_aspect = size[0] / size[1]
            if abs(png_aspect - pdf_aspect) / pdf_aspect > ASPECT_TOL:
                problems.append(f"aspect mismatch PNG {png_aspect:.4f} "
                                f"vs PDF {pdf_aspect:.4f}")
        raster = has_raster_payload(pdf)
        lint = png.parent / "figure.lint.json"
        lint_ok = None
        if lint.exists():
            payload = json.loads(lint.read_text())
            lint_ok = payload.get("ok")
            if not lint_ok:
                problems.append(f"{len(payload.get('violations', []))} layout violation(s)")
        rows.append({
            "raster_layer": raster,
            "figure": str(png.parent), "png_px": f"{px_w}x{px_h}", "dpi": dpi,
            "pdf_pt": f"{size[0]:.1f}x{size[1]:.1f}" if size else "-",
            "pdf_in": f"{size[0]/72:.2f}x{size[1]/72:.2f}" if size else "-",
            "lint_ok": lint_ok,
            "status": "PASS" if not problems else "FAIL",
            "problem": "; ".join(problems)})

    width = max((len(r["figure"]) for r in rows), default=10)
    for r in rows:
        note = r["problem"]
        if r.get("raster_layer"):
            # imshow / pcolormesh / rasterized scatter embed a bitmap by design; that is
            # correct for a heat map and a defect only for line art, so it is reported
            # rather than failed.
            note = (note + "; " if note else "") + "contains a raster layer (check: heat map?)"
        print(f"{r['status']:4s} {r['figure']:<{width}}  {r.get('png_px','-'):>11s}  "
              f"{r.get('dpi',0):>4.0f} dpi  pdf {r.get('pdf_in','-'):>11s} in  "
              f"lint={r.get('lint_ok')}  {note}")
    bad = [r for r in rows if r["status"] == "FAIL"]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} exports PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))

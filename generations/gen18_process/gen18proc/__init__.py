"""gen18proc — the process-design chain of generation 18.

Chain: extraction-system composition -> D of each metal at the given loading -> countercurrent
cascade (extraction + scrub + strip + organic recycle) -> purity, recovery, throughput, reagent
consumption -> local regime optimisation.  The contract is ``generations/gen18_process/DESIGN.md``
(dated interface addenda live in ``generations/gen18_process/addenda/``).

Module layout (DESIGN.md section 2 plus addendum WB0):

* ``paths``       roots, frozen inputs (re-exported from ``gen13sep.paths``), output directories
* ``types``       every enum, protocol and frozen dataclass of the design (WB0)
* ``systems``     re-exports ``types`` and adds the database loader / validator (WB1)
* ``literature``, ``ingest``                     (WB1)
* ``dmodel``, ``domain``, ``equilibrium``, ``testsystems``   (WB2)
* ``cascade``, ``metrics``                       (WB3)
* ``evalproto``, ``optimize``, ``screen``, ``report``       (WB4)

Units: concentrations mol/L in the phase named (feed metals may enter as mM in JSON and are
converted at the boundary), flows L/h, O/A volumetric, temperature degC, log means log10.
Seeds default to 18 everywhere.  Nothing here imports the heavy modules on package import.
"""
from __future__ import annotations

__version__ = "gen18.1"
"""Schema version string of the extraction-systems database (DESIGN.md section 3.2)."""

DEFAULT_SEED = 18
"""Seed used by every stochastic step unless overridden (DESIGN.md section 1.5)."""

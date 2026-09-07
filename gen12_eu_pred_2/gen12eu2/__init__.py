"""Gen12.2 — coordination-aware prediction of the intrinsic Eu extraction level.

This package reads Gen12 (``gen12_eu_pred/``) strictly read-only: the frozen
cohort builder, the frozen design-B fold plan, the metrics, the bootstrap and the
few-shot draw are *imported* from ``gen12eu`` so that every Gen12.2 comparison
is paired against the identical held-out extractants and query rows.  Nothing
here writes under ``gen12_eu_pred/``.
"""

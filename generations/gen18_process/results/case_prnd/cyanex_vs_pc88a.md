# PC88A versus Cyanex 272 at identical feed, spec grid and price table (DESIGN.md section 13.3)

*regime: PLACEHOLDER_PARAMETERS (assumed with ranges, --allow-placeholders); feed_status assumed; spec_status assumed; feed prnd_feed.json, spec prnd_spec.json, 64 draws, seed 18, 250 LHS rows per draw, stage bounds 1-40*

**Conditional on the placeholder ranges.** Every parameter of both systems is `ASSUMED_PLACEHOLDER` (PC88A SF from EP2388344A1 / Banda 2014, Cyanex 272 SF an assumed sweep {1.1, 1.2, 1.3, 1.4} with no source found); the ranking cannot be decided until the literature values are transcribed by a person (open item U4) or measured. Draw i of PC88A is paired with draw i of Cyanex 272 (same LHS design, independent parameter draws).

| subset | purity_min | recovery_min | draws | pc88a_reachable | cyanex272_reachable | both_reachable | only_pc88a_reachable | only_cyanex272_reachable | pc88a_fewer_stages | pc88a_less_acid | fraction_pc88a_fewer_stages | fraction_pc88a_less_acid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 0.95 | 0.8 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| all | 0.95 | 0.85 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| all | 0.95 | 0.9 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| all | 0.97 | 0.8 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| all | 0.97 | 0.85 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| all | 0.97 | 0.9 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| all | 0.99 | 0.8 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| all | 0.99 | 0.85 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| all | 0.99 | 0.9 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| in_domain_only | 0.95 | 0.8 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| in_domain_only | 0.95 | 0.85 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| in_domain_only | 0.95 | 0.9 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| in_domain_only | 0.97 | 0.8 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| in_domain_only | 0.97 | 0.85 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| in_domain_only | 0.97 | 0.9 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| in_domain_only | 0.99 | 0.8 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| in_domain_only | 0.99 | 0.85 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| in_domain_only | 0.99 | 0.9 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |


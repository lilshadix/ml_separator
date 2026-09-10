#!/bin/bash
# Collect one CSV row per reference species from the xtb output directories.
# Usage (from results/L1/reference_species/):   bash collect_reference_energies.sh > reference_energies.csv
set -euo pipefail
echo "species,energy_Eh,energy_eV,converged,xtb_version"
while IFS=$'\t' read -r SPECIES FILE CHRG UHF MODE; do
    OUT="out/${SPECIES}/xtb.out"
    if [ ! -f "$OUT" ]; then
        echo "${SPECIES},,,MISSING,"
        continue
    fi
    EH=$(grep -E "TOTAL ENERGY" "$OUT" | tail -1 | awk '{print $4}')
    VER=$(grep -m1 -oE "version [0-9.]+" "$OUT" | head -1 | awk '{print $2}')
    CONV=$(grep -qE "GEOMETRY OPTIMIZATION CONVERGED|normal termination" "$OUT" && echo ok || echo CHECK)
    EV=$(awk -v e="$EH" 'BEGIN{ if (e=="") print ""; else printf "%.9f", e*27.211386245988 }')
    echo "${SPECIES},${EH},${EV},${CONV},${VER}"
done < jobs.tsv

import sys
sys.path.insert(0, "D:/ml_separator_gh/generations/gen13_separation/analysis/stage3/verify_direction")
from v_runner import *
import numpy as np
r = BASIS[0]
print("radius basis row: min %.3f max %.3f span %.3f norm %.3f" % (r.min(), r.max(), np.ptp(r), np.linalg.norm(r)))
for a in (0.05, 0.1, 0.2, 0.3, 0.5):
    print(f"  |amp|={a}: implied La->Lu log D contrast = {a*np.ptp(r):.3f} log units")
print("\nreplicate noise floor from the cohort:")
print("  median replicate_sd:", float(FRAME.replicate_sd_median.median(skipna=True)))
print("  implied pair sd (sqrt2 x):", float(FRAME.replicate_sd_median.median(skipna=True))*2**.5)

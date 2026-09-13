import sys, time
sys.path.insert(0, "D:/ml_separator_gh/generations/gen13_separation/analysis/stage3/verify_direction")
from v_runner import *
t0 = time.time()
rich = FRAME.n_metals.to_numpy() >= 5
y = (COEF[:, 0] < 0).astype(int)
tab = run(TOPO_COLS, y, rich, design="BP", est="et")
u = units(tab); ub = const_units(tab, 1)
print("topo39 ET BP macro =", u.hit.mean(), " baseline =", ub.hit.mean())
print("gain:", boot(u, ub))
print("acc CI:", boot(u))
tab2 = run(LEAN_COLS, y, rich, design="BP", est="et")
u2 = units(tab2)
print("lean209 ET BP macro =", u2.hit.mean())
print("topo - lean:", boot(u, u2))
print("seconds", time.time()-t0)

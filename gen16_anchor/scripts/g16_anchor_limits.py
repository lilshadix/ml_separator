"""BP at the three anchor limits only: gamma = 0 (within/fixed effects), 1 (pooled OLS),
1000 (between/IV).  Same scoring path as the rest of the ladder."""
import sys
sys.path.insert(0, "D:/ml_separator_gh/gen16_anchor/scripts")
import g16_anchor_fast as G
G.GAMMAS = (0.0, 1.0, 1000.0)
G.main(["BP"], ["COND_MA", "TOPO39"])

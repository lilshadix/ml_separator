import sys
from pathlib import Path

G20 = Path(__file__).resolve().parents[1]
if str(G20) not in sys.path:
    sys.path.insert(0, str(G20))

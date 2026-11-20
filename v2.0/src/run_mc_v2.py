"""Run the v2.0 Monte Carlo (chunks of 10 draws on 2 processes) and cache the result."""
import os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from endo_ai_v2 import monte_carlo

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(OUT, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
t0 = time.time()
draws, o = monte_carlo(N=N)
with open(os.path.join(OUT, "mc_cache.pkl"), "wb") as fh:
    pickle.dump({"draws": draws, "o": o}, fh)
print(f"Monte Carlo with {N} draws done in {time.time() - t0:.0f} s")

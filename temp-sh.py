import numpy as np
from pathlib import Path

f = next(Path("data/emission_1_reaper_6_400s/emission_1_c0000/shot_000").glob("part_*.npz"))
with np.load(f) as d:
    for k in d.files:
        print(k, d[k].shape, d[k].dtype)

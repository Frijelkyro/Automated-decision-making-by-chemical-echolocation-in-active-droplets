#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np


# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------

shot = int(sys.argv[1])
stats_dir = Path(sys.argv[2])

param_file = Path("./data/param.txt.bak")
output_file = stats_dir / f"shot_{shot:04d}.npz"


# ------------------------------------------------------------
# Read param.txt.bak
# ------------------------------------------------------------

data = {}

with param_file.open("r") as f:
    exec(f.read(), {}, data)


# ------------------------------------------------------------
# Extract arrays
# ------------------------------------------------------------

birth_times = np.asarray(data["birth_times"], dtype=float)
exit_trigger_time = np.asarray(data["exit_trigger_time"], dtype=float)


# ------------------------------------------------------------
# Store shot
# ------------------------------------------------------------

stats_dir.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    output_file,
    birth_times=birth_times,
    exit_trigger_time=exit_trigger_time,
)

print(f"Saved {output_file}")
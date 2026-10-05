#!/usr/bin/env python3

import sys
from pathlib import Path
import numpy as np

shot = int(sys.argv[1])
stats_dir = Path(sys.argv[2])

state_file = Path("./data/param_state.npz")
output_file = stats_dir / f"shot_{shot:04d}.npz"

if not state_file.exists():
    raise FileNotFoundError(f"State file not found: {state_file}")

with np.load(state_file) as state:
    birth_times = np.asarray(state["birth_times"], dtype=float)
    exit_trigger_time = np.asarray(
        state["exit_trigger_time"],
        dtype=float,
    )

stats_dir.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    output_file,
    birth_times=birth_times,
    exit_trigger_time=exit_trigger_time,
)

print(f"Saved {output_file}")
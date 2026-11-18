import argparse
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Print a shot's junction decision log.")
parser.add_argument("directory", type=Path, help="Shot output/data directory")
parser.add_argument("--head", type=int, default=20, help="Rows to print per NPZ file")
args = parser.parse_args()

paths = sorted(
    args.directory.glob("decisions_*.npz"),
    key=lambda path: int(path.stem.rsplit("_", 1)[-1])
    if path.stem.rsplit("_", 1)[-1].isdigit()
    else path.name,
)

for path in paths:
    with np.load(path, allow_pickle=False) as data:
        print(path.name)
        junctions = data["junction_ids"] if "junction_ids" in data else data["junction_id"]
        for step, time, particle, label, decision in zip(
            data["global_step"][:args.head],
            data["sim_time"][:args.head],
            data["particle_id"][:args.head],
            junctions[:args.head],
            data["correct"][:args.head],
        ):
            result = "correct" if decision else "wrong"
            print(
                f"step={int(step)} time={float(time):.6g} "
                f"particle={int(particle)} junction={str(label)} decision={result}"
            )

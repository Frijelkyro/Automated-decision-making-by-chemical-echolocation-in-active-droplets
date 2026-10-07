#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ER, REAPER_TIMER = 0.5, 6.0
STATS_DIR = Path(f"./output/exit_time_statistics/{ER}_emission_rate/{REAPER_TIMER}_2")
files = sorted(STATS_DIR.glob("shot_*.npz"))

fig, ax = plt.subplots(figsize=(10, 6))

i = 0
for f in files:
    if i < 4:
        i += 1
        continue
    with np.load(f) as d:
        b, e = d["birth_times"], d["exit_trigger_time"]

    valid = np.isfinite(b) & np.isfinite(e)
    shot = f.stem.split("_")[-1]
    ax.scatter(
        b[valid],
        e[valid] - b[valid],
        s=12,
        label=f"Shot {int(shot)} ({valid.sum()} particles)",
    )

ax.set_xlabel("Birth time [s]")
ax.set_ylabel("Exit time − birth time [s]")
ax.set_title(f"Particle exit times by shot")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(STATS_DIR / "exit_time_scatter_by_shot.png", dpi=300)
plt.show()

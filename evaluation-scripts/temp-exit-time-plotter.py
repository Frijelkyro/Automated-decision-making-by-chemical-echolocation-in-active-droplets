#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ER, REAPER_TIMER, BIN_WIDTH = 0.5, 6.0, 2.0
STATS_DIR = Path(f"./output_npz/exit_time_statistics/{ER}_emission_rate/{REAPER_TIMER}")
files = sorted(STATS_DIR.glob("shot_*.npz"))
print(len(files))
if not files:
    raise FileNotFoundError(f"No shot files found in {STATS_DIR}")

birth, duration = [], []

i = 0
for f in files:
    with np.load(f) as d:
        b, e = d["birth_times"].astype(float), d["exit_trigger_time"].astype(float)
    valid = np.isfinite(b) & np.isfinite(e)
    birth.extend(b[valid])
    duration.extend((e[valid] - b[valid]))
    # print(f.relative_to(f"./output/exit_time_statistics/{ER}_emission_rate/{REAPER_TIMER}_2"))
    if i == 3:
        print(b)
        print(e)
        print(np.array(birth, dtype=float))
        # print(list(duration))
    i += 1

birth, duration = np.asarray(birth), np.asarray(duration)

bins = np.arange(0, birth.max() + BIN_WIDTH, BIN_WIDTH)
idx = np.digitize(birth, bins) - 1

x, mean, std = [], [], []
for i in range(len(bins) - 1):
    v = duration[idx == i]
    if v.size:
        x.append((bins[i] + bins[i + 1]) / 2)
        mean.append(v.mean())
        std.append(v.std(ddof=1) if v.size > 1 else 0)

plt.errorbar(x, mean, yerr=std, fmt="o-", capsize=3)
plt.xlabel("Birth time [s]")
plt.ylabel("Exit time − birth time [s]")
plt.title(f"Exit time vs birth time ({BIN_WIDTH:g} s bins)")
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(STATS_DIR / "exit_time_vs_birth_time.png", dpi=300)
plt.show()


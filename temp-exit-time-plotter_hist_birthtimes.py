from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ER, REAPER_TIMER = 0.5, 6.0
STATS_DIR = Path(f"./output_npz/exit_time_statistics/{ER}_emission_rate/{REAPER_TIMER}")
files = sorted(STATS_DIR.glob("shot_*.npz"))[4:]
groups = [(0, 40), (41, 81), (82, 122), (123, 163)]
data = [[] for _ in groups]

for f in files:
    with np.load(f) as d:
        b, e = d["birth_times"], d["exit_trigger_time"]
    valid = np.isfinite(b) & np.isfinite(e)
    for i, (lo, hi) in enumerate(groups):
        m = valid & (b >= lo) & (b <= hi)
        data[i].extend((e[m] - b[m]).tolist())

data = [np.asarray(x) for x in data]
valid_data = [x for x in data if x.size]
if not valid_data:
    raise RuntimeError("No valid particle data found.")

bins = np.linspace(0, max(x.max() for x in valid_data), 40)

fig, ax = plt.subplots(figsize=(10, 6))

for (lo, hi), x, color in zip(groups, data, ["C0", "C1", "C2", "C3"]):
    if x.size:
        weights = np.ones(x.size) * 100 / x.size
        ax.hist(
            x, bins=bins, weights=weights,
            histtype="stepfilled", alpha=.25,
            color=color, edgecolor=color, linewidth=1.5,
            label=f"{lo}–{hi} s (n={len(x)})",
        )
        ax.hist(
            x, bins=bins, weights=weights,
            histtype="step", linewidth=2, color=color,
        )

ax.set(
    xlabel="Exit time − birth time [s]",
    ylabel="Particles [%]",
    title="Exit-time distributions by birth time",
)
ax.set_ylim(bottom=0)
ax.legend()
ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(STATS_DIR / "exit_duration_by_birth_group.png", dpi=300)
plt.show()

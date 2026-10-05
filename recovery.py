import glob, re, os, numpy as np

data = "data"
dt = 0.25e-3
grim_reaper_delay = 12.1

files = sorted(glob.glob(f"{data}/part_*.txt"),
               key=lambda f: int(re.search(r"part_(\d+)", f).group(1)))

removed = {}
for fn in files:
    with open(fn) as f:
        lines = f.readlines()
    timestep = int(lines[1])
    a = np.genfromtxt(lines[3:], dtype=float)
    for row in np.atleast_2d(a):
        pid = int(row[0])
        if pid not in removed and np.isnan(row[1:3]).any():
            removed[pid] = timestep * dt - grim_reaper_delay

with open(f"{data}/exit_times.txt", "w") as f:
    f.write("ExitTime Beta JobID ParticleID\n")
    for pid in range(350):
        f.write(f"{removed.get(pid, -1)} -8 1 {pid}\n")

#!/usr/bin/env python3
"""Exit-time histograms (all selected runs in one plot) and exit time vs spawn time (one plot per run_id).

Every particle of every shot is used as an individual data point (no averaging over shots).
"""
import argparse, itertools, tomllib
from fnmatch import fnmatch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config", type=Path, default=ROOT / "experiments" / "simulation_series_parameters.toml")
ap.add_argument("--runs", nargs="*", default=["*"], help="run_id globs, e.g. 'sweep_c0001' 'baseline*'")
ap.add_argument("--bins", type=int, default=20, help="number of histogram bins")
ap.add_argument("--roll-width", type=float, default=10.0, help="rolling window width in spawn time")
ap.add_argument("--roll-step", type=float, default=2.0, help="distance between window centers")
ap.add_argument("--min-count", type=int, default=5, help="min. particles in a window to draw its mean")
ap.add_argument("--color-by-shot", action="store_true", help="color scatter points by shot instead of gray")
ap.add_argument("--no-hist", action="store_true")
ap.add_argument("--no-spawn", action="store_true")
ap.add_argument("--out", type=Path, help="histogram png (default: <series>/exit_time/exit_time_hist.png)")
ap.add_argument("--show", action="store_true")
args = ap.parse_args()

cfg = tomllib.load((args.config if args.config.is_absolute() else ROOT / args.config).open("rb"))
s = cfg["series"]
absr = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p
base, dbase = absr(s["output_root"]) / s["name"], absr(s["data_root"]) / s["name"]

# Rebuild run_id -> parameters exactly like make_jobs.py
runs = {}
for reg in cfg["regimes"]:
    for i, vals in enumerate(itertools.product(*reg["grid"].values())):
        runs[f"{reg['name']}_c{i:04d}"] = cfg.get("defaults", {}) | dict(zip(reg["grid"], vals))

data = {}
for rid, p in runs.items():
    files = sorted((base / rid).glob("shot_*/exit_times.txt"))
    if not files or not any(fnmatch(rid, g) for g in args.runs):
        continue
    rate = p.get("emission_rate", 0.5)

    birth, exit_t, shot_id = [], [], []
    n_never = 0
    for k, f in enumerate(files):
        t = np.loadtxt(f, skiprows=1, usecols=0, ndmin=1)  # exit time - birth time, -1 = never exited
        ck = dbase / rid / f.parent.name / "ckpt_current.npz"
        if ck.exists():  # actual spawn times; fallback: nominal i / emission_rate
            with np.load(ck) as state:
                b = state["birth_times"][: len(t)]
        else:
            b = np.arange(len(t)) / rate
        t = t[: len(b)]
        ok = t >= 0
        n_never += int((~ok).sum())
        birth.append(b[ok])
        exit_t.append(t[ok])
        shot_id.append(np.full(ok.sum(), k))

    data[rid] = dict(
        rate=rate,
        n_shots=len(files),
        n_never=n_never,
        birth=np.concatenate(birth),
        exit=np.concatenate(exit_t),
        shot=np.concatenate(shot_id),
    )
if not data:
    ap.error(f"no exit_times.txt found under {base} matching {args.runs}")
outdir = base / "exit_time"
outdir.mkdir(parents=True, exist_ok=True)

# ---------------- histogram: all individual particles pooled over shots ----------------
if not args.no_hist:
    edges = np.linspace(0, max(d["exit"].max() for d in data.values()), args.bins + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for rid, d in sorted(data.items(), key=lambda kv: kv[1]["rate"]):
        ax.hist(d["exit"], bins=edges, density=True, alpha=0.5,
                label=f"{rid}: emission rate {d['rate']:g} ({d['n_shots']} shots, {len(d['exit'])} particles)")
    ax.set(xlabel="Exit time - birth time [sim. time]", ylabel="Density", title=s["name"])
    ax.legend(); ax.grid(alpha=0.3)
    out = args.out or outdir / "exit_time_hist.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")

# ---------------- exit time vs spawn time: scatter of all particles + rolling mean ----------------
if not args.no_spawn:
    for rid, d in data.items():
        x, y = d["birth"], d["exit"]
        centers = np.arange(x.min(), x.max() + args.roll_step, args.roll_step)

        mu, sem, cs = [], [], []
        for c in centers:
            m = np.abs(x - c) <= args.roll_width / 2
            if m.sum() >= args.min_count:
                cs.append(c)
                mu.append(y[m].mean())
                sem.append(y[m].std(ddof=1) / np.sqrt(m.sum()))
        cs, mu, sem = map(np.array, (cs, mu, sem))

        fig, ax = plt.subplots(figsize=(9, 5))
        if args.color_by_shot:
            sc = ax.scatter(x, y, s=8, alpha=0.4, c=d["shot"], cmap="tab10", label="particles (individual)")
            fig.colorbar(sc, ax=ax, label="shot")
        else:
            ax.scatter(x, y, s=8, alpha=0.25, color="gray", label=f"particles (all {d['n_shots']} shots)")
        if len(cs):
            ax.plot(cs, mu, "o-", ms=3, color="C3",
                    label=f"rolling mean over particles (width {args.roll_width:g}, step {args.roll_step:g})")
            ax.fill_between(cs, mu - sem, mu + sem, color="C3", alpha=0.25, label="± standard error")
        ax.set(xlabel="Spawn time [sim. time]", ylabel="Exit time - birth time [sim. time]",
               title=f"{rid}: emission rate {d['rate']:g} ({d['n_shots']} shots, "
                     f"{d['n_never']} particles never exited)")
        ax.legend(); ax.grid(alpha=0.3)
        out = outdir / f"{rid}_exit_vs_spawn.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")
if args.show:
    plt.show()

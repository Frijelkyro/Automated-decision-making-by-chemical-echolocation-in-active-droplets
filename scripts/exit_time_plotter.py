#!/usr/bin/env python3
"""Exit-time histograms (all selected param_configs in one plot) and exit time vs spawn time (one plot per run_id)."""
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
    t = np.array([np.loadtxt(f, skiprows=1, usecols=0, ndmin=1) for f in files])  # shots x particles
    # actual spawn times from the final checkpoint; fallback: nominal i / emission_rate
    b = np.array([np.load(ck)["birth_times"][: t.shape[1]] if (ck := dbase / rid / f.parent.name / "ckpt_current.npz").exists()
                  else np.arange(t.shape[1]) / rate for f in files])
    tm = np.ma.masked_less(t, 0)  # -1 = never exited
    mean_t = tm.mean(axis=0)
    mean_b = np.ma.array(b, mask=np.ma.getmaskarray(tm)).mean(axis=0)
    keep = ~np.ma.getmaskarray(mean_t)
    data[rid] = (rate, mean_t.compressed(), len(files), mean_b[keep].data, mean_t[keep].data)
if not data:
    ap.error(f"no exit_times.txt found under {base} matching {args.runs}")
outdir = base / "exit_time"
outdir.mkdir(parents=True, exist_ok=True)

if not args.no_hist:
    edges = np.linspace(0, max(d[1].max() for d in data.values()), args.bins + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for rid, (rate, m, n, *_) in sorted(data.items(), key=lambda kv: kv[1][0]):
        ax.hist(m, bins=edges, density=True, alpha=0.5, label=f"emission rate {rate:g} ({n} shots)")
    ax.set(xlabel="Mean exit time - birth time [sim. time]", ylabel="Density", title=s["name"])
    ax.legend(); ax.grid(alpha=0.3)
    out = args.out or outdir / "exit_time_hist.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")

if not args.no_spawn:
    for rid, (rate, _, n, x, y) in data.items():
        centers = np.arange(x.min(), x.max() + args.roll_step, args.roll_step)
        sel = [np.abs(x - c) <= args.roll_width / 2 for c in centers]
        ok = np.array([m.any() for m in sel])
        mu = np.array([y[m].mean() for m, k in zip(sel, ok) if k])
        sd = np.array([y[m].std() for m, k in zip(sel, ok) if k])
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(x, y, s=6, alpha=0.25, color="gray", label="particles (mean over shots)")
        ax.plot(centers[ok], mu, "o-", ms=3, label=f"rolling mean (width {args.roll_width:g}, step {args.roll_step:g})")
        ax.fill_between(centers[ok], mu - sd, mu + sd, alpha=0.25)
        ax.set(xlabel="Spawn time [sim. time]", ylabel="Exit time - birth time [sim. time]",
               title=f"{rid}: emission rate {rate:g} ({n} shots)")
        ax.legend(); ax.grid(alpha=0.3)
        out = outdir / f"{rid}_exit_vs_spawn.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")
if args.show:
    plt.show()

"""Focused second-round sweep with train_hf.py (HF Trainer).

Round 1 (pico-train) showed that the best runs used the highest learning rates (4.28e-5 and 6.69e-5).
This round samples learning rates around/above those, log-uniformly, and keeps gradient accumulation
small. Everything is set with environment variables, so no code edits are needed:

  LR_MIN, LR_MAX      learning-rate range, log-uniform        (default 2e-5 .. 2e-4)
  ACC_MIN, ACC_MAX    gradient-accumulation range, log-uniform (default 1 .. 8)
  N_RUNS              number of runs                          (default 8)
  SEED                random seed for sampling                 (default 1)
  RUN_PREFIX          run names are <prefix>-00, <prefix>-01.. (default r2)
  DRY_RUN=1           print the planned runs and stop
Schedules alternate linear / cosine so the split is always even.
"""

import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np

USER = os.environ["HF_USER"]
DATASET = os.environ.get(
    "DATASET", f"{USER}/babylm-strict-small-512"
)  # single copy; Trainer does the epochs
MB = int(os.environ.get("MB", 16))
D_MODEL = int(os.environ.get("D_MODEL", 256))
LAYERS = int(os.environ.get("LAYERS", 6))
PRECISION = os.environ.get("PRECISION", "fp16")
OUT_DIR = os.environ.get("OUT_DIR", "runs_hf")
LR_MIN, LR_MAX = (
    float(os.environ.get("LR_MIN", 2e-5)),
    float(os.environ.get("LR_MAX", 2e-4)),
)
ACC_MIN, ACC_MAX = (
    float(os.environ.get("ACC_MIN", 1)),
    float(os.environ.get("ACC_MAX", 8)),
)
N_RUNS = int(os.environ.get("N_RUNS", 8))
SEED = int(os.environ.get("SEED", 1))
PREFIX = os.environ.get("RUN_PREFIX", "r2")
sel = os.environ.get("TRIALS")  # e.g. "0-3" to split across Jobs

rng = np.random.default_rng(SEED)
trials = []
for i in range(N_RUNS):
    lr = float(np.exp(rng.uniform(np.log(LR_MIN), np.log(LR_MAX))))
    accum = int(round(np.exp(rng.uniform(np.log(ACC_MIN), np.log(ACC_MAX)))))
    sched = ["linear", "cosine"][i % 2]
    trials.append((i, lr, accum, sched))
if sel:
    lo, hi = map(int, sel.split("-"))
    trials = [t for t in trials if lo <= t[0] <= hi]
for i, lr, accum, sched in trials:
    print(
        f"{PREFIX}-{i:02d}: lr={lr:.2e} accum={accum} effective_batch={MB * accum} {sched}",
        flush=True,
    )
json.dump(trials, open(f"sweep_trials_{PREFIX}.json", "w"), indent=2)
if os.environ.get("DRY_RUN"):
    sys.exit(0)

for i, lr, accum, sched in trials:
    name = f"{PREFIX}-{i:02d}"
    if os.path.isdir(f"{OUT_DIR}/{name}/final"):
        print(f"{name}: already done, skipping", flush=True)
        continue
    shutil.rmtree(f"{OUT_DIR}/{name}", ignore_errors=True)
    cmd = [
        sys.executable,
        "train_hf.py",
        "--run_name",
        name,
        "--dataset",
        DATASET,
        "--lr",
        f"{lr:.6e}",
        "--accum",
        str(accum),
        "--sched",
        sched,
        "--micro_batch",
        str(MB),
        "--d_model",
        str(D_MODEL),
        "--layers",
        str(LAYERS),
        "--precision",
        PRECISION,
        "--out_dir",
        OUT_DIR,
    ]
    print(f"{name}: starting", flush=True)
    with open(f"sweep_{name}.log", "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"{name}: exit code {rc}", flush=True)

# merge per-run CSVs into one file for plotting (same columns as the pico-train sweep CSV)
out_csv = f"sweep_val_ppl_{PREFIX}.csv"
with open(out_csv, "w", newline="") as out:
    w, header_written = csv.writer(out), False
    for i, *_ in trials:
        p = f"{OUT_DIR}/{PREFIX}-{i:02d}/val_ppl.csv"
        if not os.path.exists(p):
            continue
        rows = list(csv.reader(open(p)))
        if not header_written:
            w.writerow(rows[0])
            header_written = True
        w.writerows(rows[1:])
print(f"wrote {out_csv}", flush=True)

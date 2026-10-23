import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import yaml

USER = "seashello"
ENTITY = os.environ.get("WANDB_ENTITY")  # unset = no W&B
PROJECT = os.environ.get("WANDB_PROJECT", "babylm-sweep")
MB = int(os.environ.get("MB", 16))  # micro-batch: largest that fits your GPU
N_LAYERS = int(os.environ.get("N_LAYERS", 12))
N_RUNS, EPOCHS = 10, 10

stats = json.load(open("data_stats.json"))
N_TRAIN = stats["n_train_chunks"]
WORDS_PER_SAMPLE = stats["words_per_epoch"] / N_TRAIN
# BabyLM Strict-Small: every 1M words to 10M, then every 10M words to 100M
MARKS = list(range(1_000_000, 10_000_001, 1_000_000)) + list(
    range(20_000_000, 100_000_001, 10_000_000)
)


def save_steps(batch, max_steps):
    wps = batch * WORDS_PER_SAMPLE  # words consumed per optimizer step
    steps = {max(1, round(m / wps)) for m in MARKS}
    steps = {s for s in steps if s <= max_steps} | {max_steps}
    return sorted(steps)


def make_cfg(i, lr, accum, sched):
    batch = MB * accum  # effective batch
    max_steps = (EPOCHS * N_TRAIN) // batch
    monitoring = {"save_to_wandb": bool(ENTITY), "logging": {"log_every_n_steps": 50}}
    if ENTITY:
        monitoring["wandb"] = {"project": PROJECT, "entity": ENTITY}
    return {
        "data": {
            "dataset": {"name": f"{USER}/babylm-strict-small-512-x10"},
            "dataloader": {"batch_size": batch},
        },
        "model": {
            "d_model": 96,
            "n_layers": N_LAYERS,
            "activation_hidden_dim": 384,
            "max_seq_len": 512,
            "batch_size": batch,
        },
        "checkpointing": {
            "run_name": f"sweep-{i:02d}",
            "save_to_hf": False,
            "save_steps": save_steps(batch, max_steps),
            "training": {"auto_resume": False},
            "learning_dynamics": {"layer_suffixes": [], "eval_data": None},
        },
        "evaluation": {"metrics": []},
        "monitoring": monitoring,
        "training": {
            "max_steps": max_steps,
            "optimization": {
                "lr": float(lr),
                "lr_scheduler": sched,
                "lr_warmup_steps": max(1, int(0.05 * max_steps)),
                "gradient_accumulation_steps": int(accum),
            },
        },
    }


def finished(name, max_steps):
    p = f"runs/{name}/val_ppl.csv"
    if not os.path.exists(p):
        return False
    rows = list(csv.reader(open(p)))[1:]
    return bool(rows) and int(rows[-1][0]) == max_steps


rng = np.random.default_rng(0)
trials = []
for i in range(N_RUNS):
    lr = float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-4))))  # log-uniform
    accum = int(
        round(np.exp(rng.uniform(np.log(1), np.log(16))))
    )  # log-uniform integer
    sched = str(rng.choice(["linear_with_warmup", "cosine_with_warmup"]))
    trials.append((i, lr, accum, sched))
json.dump(trials, open("sweep_trials.json", "w"), indent=2)
for t in trials:
    print(t, flush=True)

os.makedirs("configs/sweep", exist_ok=True)
env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
for i, lr, accum, sched in trials:
    cfg = make_cfg(i, lr, accum, sched)
    name, ms = cfg["checkpointing"]["run_name"], cfg["training"]["max_steps"]
    if finished(name, ms):
        print(f"{name}: already done, skipping", flush=True)
        continue
    shutil.rmtree(f"runs/{name}", ignore_errors=True)
    path = f"configs/sweep/{name}.yaml"
    yaml.safe_dump(cfg, open(path, "w"))
    print(f"{name}: lr={lr:.2e} accum={accum} {sched} steps={ms}", flush=True)
    with open(f"sweep_{name}.log", "w") as f:
        rc = subprocess.run(
            [sys.executable, "scripts/train.py", "--config_path", path],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        ).returncode
    print(f"{name}: exit code {rc}", flush=True)

with open("sweep_val_ppl.csv", "w", newline="") as out:
    w = csv.writer(out)
    w.writerow(
        [
            "run",
            "step",
            "samples_seen",
            "words_seen",
            "val_loss",
            "val_ppl",
            "lr",
            "grad_accum",
            "schedule",
        ]
    )
    for i, lr, accum, sched in trials:
        p = f"runs/sweep-{i:02d}/val_ppl.csv"
        if not os.path.exists(p):
            continue
        for r in list(csv.reader(open(p)))[1:]:
            w.writerow(
                [
                    f"sweep-{i:02d}",
                    r[0],
                    r[1],
                    round(int(r[1]) * WORDS_PER_SAMPLE),
                    r[2],
                    r[3],
                    f"{lr:.3e}",
                    accum,
                    sched.split("_")[0],
                ]
            )
print("wrote sweep_val_ppl.csv", flush=True)

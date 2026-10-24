"""Run the full experimental grid as a queue of train_eval.py jobs.

  E2 leave-one-out   : train on 3 sites, test on the 4th (main generalisation result)
  E1 single-source   : train on 1 site, test on all 4 (transfer matrix)
  E3 ablations       : PPO-DeepSets on the leave-one-out setting, one design choice removed

Jobs whose result CSV already exists are skipped, so the script can be resumed.
Usage: python scripts/run_experiments.py --workers 5 --steps 2000000 [--only E2]
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import SITES  # noqa: E402

ARCHS = ["deepsets", "mlp", "attention"]
SEEDS = [0, 1, 2, 3, 4]
ABLATIONS = {
    "no_shaping": {"shaping": False},
    "no_center": {"center_price": False},
    "per_ev_action": {"action": "per_ev", "action_repeat": 1},
    "no_noise_train": {"dep_noise_train": 0.0},
}


def jobs(steps, only):
    out = []
    if only in (None, "E2"):
        for seed in SEEDS:
            for arch in ARCHS:
                if arch == "attention" and seed > 2:   # attention: 3 seeds (~3x slower per run)
                    continue
                for t in SITES:
                    src = ",".join(s for s in SITES if s != t)
                    out.append(dict(arch=arch, sources=src, seed=seed, tag=f"{arch}_loo_{t}_s{seed}", out="runs_loo"))
    if only in (None, "E1"):
        # transfer matrix (compute budget): 2 seeds, DeepSets and MLP only
        for seed in SEEDS[:2]:
            for arch in ["deepsets", "mlp"]:
                for s in SITES:
                    out.append(dict(arch=arch, sources=s, seed=seed, tag=f"{arch}_{s}_s{seed}", out="runs"))
    if only in (None, "E3"):
        best = json.loads((ROOT / "results" / "hpo" / "best_deepsets.json").read_text())["params"]
        for name, over in ABLATIONS.items():
            for seed in SEEDS[:1]:   # ablations: 1 seed x 4 held-out sites (compute budget)
                for t in SITES:
                    src = ",".join(s for s in SITES if s != t)
                    out.append(dict(arch="deepsets", sources=src, seed=seed, tag=f"abl_{name}_loo_{t}_s{seed}",
                                    out="runs_abl", params=json.dumps({**best, **over})))
    for j in out:
        j["steps"] = steps
    return out


def running_tags():
    """Tags of train_eval.py processes already running (e.g. left by a previous orchestrator)."""
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                          "ForEach-Object { $_.CommandLine }"], capture_output=True, text=True).stdout
    return {line.split("--tag ")[1].split()[0] for line in out.splitlines() if "train_eval.py" in line and "--tag " in line}


RUNNING = set()


def run(j):
    csv = ROOT / "results" / j["out"] / f"{j['tag']}.csv"
    if csv.exists():
        return j["tag"], "skip"
    if j["tag"] in RUNNING:   # adopt the orphan job: wait for it instead of starting a duplicate
        while j["tag"] in running_tags() and not csv.exists():
            time.sleep(60)
        if csv.exists():
            print(j["tag"], "ok (adopted)", flush=True)
            return j["tag"], 0
    cmd = [sys.executable, str(ROOT / "scripts" / "train_eval.py"), "--arch", j["arch"], "--sources", j["sources"],
           "--seed", str(j["seed"]), "--steps", str(j["steps"]), "--tag", j["tag"], "--out", j["out"]]
    if "params" in j:
        cmd += ["--params", j["params"]]
    log = ROOT / "results" / "logs"
    log.mkdir(parents=True, exist_ok=True)
    with open(log / f"{j['tag']}.log", "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(j["tag"], "ok" if r.returncode == 0 else f"FAILED {r.returncode}", flush=True)
    return j["tag"], r.returncode


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    js = jobs(a.steps, a.only)
    RUNNING.update(running_tags())
    # adopted jobs first so they hold worker slots and nothing is oversubscribed
    js.sort(key=lambda j: j["tag"] not in RUNNING)
    print(len(js), "jobs", "| already running:", sorted(RUNNING), flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, js))

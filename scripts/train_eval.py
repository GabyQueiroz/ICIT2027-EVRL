"""Train one PPO agent on one or more source sites and evaluate it on every site.

Evaluations (all on held-out TEST weeks, never used for training or tuning):
  * nominal         : test weeks with 2023-2025 prices
  * stress          : same weeks with 2022 (energy-crisis) prices
  * dep_noise_1h    : nominal, but users announce departure with N(0, 1 h) error

Usage:
  python scripts/train_eval.py --arch deepsets --sources dundee --seed 0
  python scripts/train_eval.py --arch deepsets --sources paloalto,perth,trondheim --seed 0 --tag loo_dundee
Hyperparameters come from results/hpo/best_<arch>.json unless --params is given.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import PPO, PPOConfig  # noqa: E402
from data import SITES  # noqa: E402
from scenario import build  # noqa: E402

EVALS = [("nominal", "test", 0.0), ("stress", "stress", 0.0), ("dep_noise_1h", "test", 1.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--params", default=None, help="JSON overrides (else best HPO params)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()

    if a.params:
        params = json.loads(a.params)
    else:
        params = json.loads((ROOT / "results" / "hpo" / f"best_{a.arch}.json").read_text())["params"]
    sources = a.sources.split(",")
    tag = a.tag or f"{a.arch}_{'+'.join(sources)}_s{a.seed}"
    out = ROOT / "results" / a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"{tag}.csv").exists():
        print("exists", tag)
        return

    # final experiments use price augmentation and departure-announcement noise randomisation
    # (std ~ U[0, 1.5] h per batch); neither was used during tuning (documented in the paper)
    cfg = PPOConfig(**{"price_aug": True, "dep_noise_train": 1.5, **params, "arch": a.arch, "seed": a.seed, "total_steps": a.steps,
                       "threads": 2})
    agent = PPO(cfg, [build(s) for s in sources])
    t0 = time.time()
    hist = agent.learn(log_every=0)
    train_time = time.time() - t0
    (out / "models").mkdir(exist_ok=True)
    agent.save(out / "models" / f"{tag}.pt")
    pd.DataFrame(hist).to_csv(out / f"{tag}_curve.csv", index=False)

    rows = []
    for target in SITES:
        sc = build(target)
        for name, split, noise in EVALS:
            m = agent.evaluate(sc, split, seed=a.seed, dep_noise_h=noise)
            for i in range(len(m["J"])):
                rows.append(dict(tag=tag, arch=a.arch, sources="+".join(sources), seed=a.seed, target=target,
                                 eval=name, ep=i, in_domain=target in sources, train_time=train_time,
                                 **{k: float(v[i]) for k, v in m.items()}))
    pd.DataFrame(rows).to_csv(out / f"{tag}.csv", index=False)
    print("done", tag, f"{train_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Quick diagnostic run: train PPO with config overrides, print validation curve.

Usage: python scripts/probe.py '{"arch": "mlp", "lr": 1e-3}' [site] [n_val]
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import PPO, PPOConfig  # noqa: E402
from scenario import build  # noqa: E402

over = json.loads(sys.argv[1])
site = sys.argv[2] if len(sys.argv) > 2 else "dundee"
n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 20
sc = build(site)
val = sc.val[:n_val]
cfg = PPOConfig(**{"total_steps": 150_000, **over})
ag = PPO(cfg, sc)
curve = []


def cb(a, u, n):
    if u % max(1, n // 6) == 0 or u == n - 1:
        m = a.evaluate(sc, "val", episodes=val)
        curve.append((u, round(float(m["J_per_kwh"].mean()), 4), round(float(m["served"].mean()), 4),
                      round(float(m["cost_per_kwh"].mean()), 4)))


t = time.time()
hist = ag.learn(log_every=0, callback=cb)
print(json.dumps(over), "| curve(u, J, served, cost):", curve, f"| kl={np.mean([h['kl'] for h in hist]):.4f}",
      f"| {time.time() - t:.0f}s", flush=True)

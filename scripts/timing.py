"""Wall-clock time per control decision on one CPU thread: PPO policies vs MPC (results/tables/timing.csv).

Run when no other experiment is using the CPU.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent import PPO  # noqa: E402
from baselines import MPC, llf  # noqa: E402
from data import SITES  # noqa: E402
from scenario import build  # noqa: E402

torch.set_num_threads(1)
rows = []
for site in SITES:
    sc = build(site)
    eps = sc.test[:3]
    for arch in ["mlp", "deepsets", "attention"]:
        f = ROOT / "results" / "runs_loo" / "models" / f"{arch}_loo_{site}_s0.pt"
        if not f.exists():
            continue
        ag = PPO.load(f, [build(s) for s in SITES if s != site])
        torch.set_num_threads(1)
        for e in eps:
            sim = ag.make_sim(sc, "test", episodes=[e])
            o = sim.reset()
            ts, done = [], False
            while not done:
                t0 = time.perf_counter()
                a = ag.act(*o)
                r = sim.aggregate_rates(a[:, 0])
                ts.append(time.perf_counter() - t0)
                for _ in range(ag.cfg.action_repeat):
                    _, done = sim.step_rates(sim.aggregate_rates(a[:, 0]))
                o = sim.obs()
            rows.append(dict(site=site, method=f"PPO-{arch}", ms_per_decision=1e3 * np.mean(ts)))
    for name, pol, kw in [("MPC+S", MPC, dict(projection="llf", guard=True)), ("LLF", None, {})]:
        for e in eps:
            sim = sc.sim("test", episodes=[e], **kw)
            p = pol() if pol else llf
            sim.reset()
            ts, done = [], False
            while not done:
                t0 = time.perf_counter()
                r = p(sim)
                ts.append(time.perf_counter() - t0)
                _, done = sim.step_rates(r)
            rows.append(dict(site=site, method=name, ms_per_decision=1e3 * np.mean(ts),
                             ms_max=1e3 * np.max(ts)))
df = pd.DataFrame(rows).groupby(["method", "site"]).mean().round(3)
df.to_csv(ROOT / "results" / "tables" / "timing.csv")
print(df.to_string())

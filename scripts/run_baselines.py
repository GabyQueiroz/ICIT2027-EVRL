"""Evaluate all non-learning controllers and the clairvoyant LP optimum.

Writes results/baselines.csv with one row per (site, split, method, episode).
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import MPC, PriceGreedy, asap, edf, llf, offline_optimum, run_policy  # noqa: E402
from data import SITES  # noqa: E402
from scenario import LAM, build  # noqa: E402

# (eval name, split, departure-announcement noise std in hours) -- same as train_eval.py
EVALS = [("val", "val", 0.0), ("nominal", "test", 0.0), ("stress", "stress", 0.0), ("dep_noise_1h", "test", 1.0)]


def rows_for(site, ev):
    name_ev, split, noise = ev
    sc = build(site)
    eps = getattr(sc, split)
    out = []
    safe = dict(projection="llf", guard=True)   # the safety layer used by the RL agent
    zero = lambda s: 0.0 * s.max_rate()  # noqa: E731  -> charge only what the guard forces
    for name, pol, kw in [("ASAP", asap, {}), ("EDF", edf, {}), ("LLF", llf, {}),
                          ("PriceGreedy", PriceGreedy(), {}), ("MPC", MPC(), {}),
                          ("ALAP+S", zero, safe), ("PriceGreedy+S", PriceGreedy(), safe), ("MPC+S", MPC(), safe)]:
        t0 = time.perf_counter()
        m = run_policy(sc.sim(split, dep_noise_h=noise, **kw), pol)
        sec = (time.perf_counter() - t0) / len(eps)
        for i in range(len(eps)):
            out.append(dict(site=site, eval=name_ev, method=name, ep=i, sec_per_episode=sec,
                            **{k: float(v[i]) for k, v in m.items()}))
    if noise > 0:   # the clairvoyant optimum does not depend on announcements
        print("done", site, name_ev, flush=True)
        return out
    for i, ep in enumerate(eps):
        J = offline_optimum(ep, LAM)
        out.append(dict(site=site, eval=name_ev, method="Optimal", ep=i, J=J, demand=float(ep.energy.sum()),
                        J_per_kwh=J / float(ep.energy.sum())))
    print("done", site, name_ev, flush=True)
    return out


if __name__ == "__main__":
    jobs = [(s, ev) for s in SITES for ev in EVALS]
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(rows_for, *zip(*jobs)))
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(ROOT / "results" / "baselines.csv", index=False)
    print(df.groupby(["site", "eval", "method"])[["J_per_kwh", "served", "cost_per_kwh"]].mean().round(4).to_string())

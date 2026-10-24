"""Non-learning controllers and the offline (perfect-information) optimum.

All controllers act through the same BatchSim safety layer.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from env import DT, HORIZON, BatchSim, Episode


def run_policy(sim: BatchSim, policy) -> dict:
    """policy(sim) -> per-slot rates in kW (B, M)."""
    sim.reset()
    done = False
    while not done:
        _, done = sim.step_rates(policy(sim))
    return sim.metrics()


# ---- heuristics ---------------------------------------------------------------------
def asap(sim):
    return sim.max_rate()


def _priority_fill(sim, key):
    """Give the hub capacity to EVs in increasing order of `key`."""
    r = sim.max_rate()
    out = np.zeros_like(r)
    order = np.argsort(np.where(sim.present, key, np.inf), axis=1)
    for b in range(sim.B):
        budget = sim.cap[b, 0]
        for k in order[b]:
            if not sim.present[b, k] or budget <= 0:
                break
            out[b, k] = min(r[b, k], budget)
            budget -= out[b, k]
    return out


def edf(sim):
    return _priority_fill(sim, sim.dep_obs)


def llf(sim):
    need = sim.rem / (np.maximum(sim.pmax, 1e-6) * DT)
    return _priority_fill(sim, sim.dep_obs - sim.t - need)


class PriceGreedy:
    """Each EV books the cheapest known steps of its window at arrival
    (day-ahead prices), ignoring the hub limit; least-laxity EVs override."""

    def __init__(self):
        self.plan = None

    def __call__(self, sim):
        B, M = sim.present.shape
        if self.plan is None or sim.t == 0:
            self.plan = np.zeros((B, M, HORIZON + 1))
            self.seen = np.zeros((B, M), bool)
        t = sim.t
        new = sim.present & ~self.seen
        for b, k in zip(*np.nonzero(new)):
            end = int(min(np.ceil(sim.dep_obs[b, k]), HORIZON))
            win = np.arange(t, max(end, t + 1))
            order = win[np.argsort(sim.price[b, win], kind="stable")]
            need, row = sim.rem[b, k], np.zeros(HORIZON + 1)
            for s in order:
                if need <= 1e-9:
                    break
                q = min(sim.pmax[b, k] * DT, need)
                row[s] = q / DT
                need -= q
            self.plan[b, k] = row
        self.seen = sim.present.copy()
        rates = self.plan[:, :, t].copy()
        need = sim.rem / (np.maximum(sim.pmax, 1e-6) * DT)
        urgent = sim.present & (sim.dep_obs - t - need <= 1)
        rates[urgent] = sim.pmax[urgent]
        return rates


# ---- LP machinery -------------------------------------------------------------------
def _lp(price, t0, arr, dep, energy, pmax, p_cap, lam):
    """min sum p_t r_it dt + lam u_i  s.t. energy balance, rate and hub limits.

    arr/dep are absolute steps; the plan starts at t0.  Returns (r dict, objective).
    """
    n = len(energy)
    if n == 0:
        return [], 0.0
    idx, cols, c, ub = [], [], [], []
    rows_t = {}
    for i in range(n):
        a, d = max(arr[i], t0), dep[i]
        steps = np.arange(a, max(d, a))
        idx.append((len(c), len(steps), a))
        for s in steps:
            rows_t.setdefault(s, []).append(len(c))
            c.append(price[s] * DT)
            ub.append(pmax[i])
    nr = len(c)
    c = np.r_[c, np.full(n, lam)]
    bounds = list(zip(np.zeros(nr), ub)) + [(0, None)] * n
    # equality: sum r dt + u = E
    ei, ej, ev = [], [], []
    for i, (o, L, _) in enumerate(idx):
        ei += [i] * (L + 1)
        ej += list(range(o, o + L)) + [nr + i]
        ev += [DT] * L + [1.0]
    A_eq = sp.csr_matrix((ev, (ei, ej)), shape=(n, nr + n))
    # hub limit per step
    ui, uj = [], []
    for k, (s, cols_s) in enumerate(sorted(rows_t.items())):
        ui += [k] * len(cols_s)
        uj += cols_s
    A_ub = sp.csr_matrix((np.ones(len(ui)), (ui, uj)), shape=(len(rows_t), nr + n))
    b_ub = np.full(len(rows_t), p_cap)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=energy, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    x = res.x
    plans = [(a, x[o:o + L]) for (o, L, a) in idx]
    return plans, res.fun


def offline_optimum(ep: Episode, lam: float) -> float:
    """Clairvoyant optimum of J for one episode (ignores the slot limit M)."""
    _, J = _lp(ep.price, 0, ep.arr, ep.dep, ep.energy, ep.pmax, ep.p_cap, lam)
    return J


class MPC:
    """Receding-horizon LP over the EVs currently plugged in (no arrival forecast).

    Re-plans whenever an EV arrives or `every` steps have passed.
    """

    def __init__(self, every: int = 4):
        self.every = every

    def __call__(self, sim):
        B, M = sim.present.shape
        t = sim.t
        if t == 0:
            self.plan = np.zeros((B, M, HORIZON + 1))
            self.last = -10 ** 9 * np.ones(B, int)
            self.seen = np.zeros((B, M), bool)
        new = (sim.present & ~self.seen).any(1)
        for b in range(B):
            if not (new[b] or t - self.last[b] >= self.every):
                continue
            ks = np.flatnonzero(sim.present[b])
            self.plan[b] = 0
            if ks.size:
                dep = np.minimum(np.ceil(sim.dep_obs[b, ks]).astype(int), HORIZON)
                plans, _ = _lp(sim.price[b], t, np.full(ks.size, t), np.maximum(dep, t + 1),
                               sim.rem[b, ks], sim.pmax[b, ks], sim.cap[b, 0], sim.cfg.lam)
                for k, (a, r) in zip(ks, plans):
                    self.plan[b, k, a:a + len(r)] = r
            self.last[b] = t
        self.seen = sim.present.copy()
        rates = self.plan[:, :, t].copy()
        return rates

"""Batched EV-charging hub simulator driven by real session data.

One episode = one calendar week of real arrivals (Mon 00:00 -> Mon 00:00) plus a
24 h tail so every EV can leave.  Time step = 15 min.  A hub has M charger slots
and an aggregate power limit P_cap (transformer / feeder).  The simulator is
vectorised over B episodes so that RL rollouts and baselines run fast on CPU.

Objective (identical for every method, lower is better):
    J = sum_t  p_t * E_t   +   lambda * U
where p_t is the price normalised by the episode mean price, E_t the hub energy
in step t (kWh) and U the energy not delivered at departure (kWh).

The safety layer makes every action feasible: per-EV rates are clipped to the
EV's max rate and remaining energy, then scaled down proportionally if the hub
exceeds P_cap.  The constraint therefore can never be violated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DT = 0.25                      # h per step
STEPS_DAY = 96
WEEK = 7 * STEPS_DAY
HORIZON = WEEK + STEPS_DAY     # arrivals during one week, 24 h tail
LOOKAHEAD = STEPS_DAY          # day-ahead prices known 24 h ahead
N_PRICE_FEATS = 24             # hourly means of the next 24 h


@dataclass
class Episode:
    arr: np.ndarray            # int step of arrival
    dep: np.ndarray            # int step of departure (exclusive)
    energy: np.ndarray         # kWh
    pmax: np.ndarray           # kW
    price: np.ndarray          # (HORIZON + LOOKAHEAD,) normalised price
    start: pd.Timestamp = None
    site: str = ""
    p_cap: float = 0.0         # hub limit of this week (kW), set by the scenario


@dataclass
class SimConfig:
    M: int = 48                # charger slots
    p_cap: float = 50.0        # kW, set per site
    lam: float = 3.0           # penalty per undelivered kWh (x mean price)
    dep_noise_h: float = 0.0   # std of the error in the departure time the user announces
    projection: str = "proportional"   # hub-limit projection: proportional | llf
    guard: bool = False        # raise each rate to the minimum that keeps the EV feasible
    seed: int = 0


def build_episodes(sessions: pd.DataFrame, price15: pd.Series, price_weeks: list,
                   site: str, min_sessions: int = 10) -> list[Episode]:
    """Cut a session table into weekly episodes and pair each with a price week.

    price_weeks: list of Monday timestamps (UTC) into price15.  Week i of the
    session data gets price week i mod len(price_weeks): a fixed, documented
    pairing, identical for every method.
    """
    s = sessions
    weeks = s.arrival.dt.to_period("W-SUN").dt.start_time
    eps = []
    for k, (wk, g) in enumerate(s.groupby(weeks)):
        if len(g) < min_sessions:
            continue
        t0 = wk
        arr = np.ceil((g.arrival - t0).dt.total_seconds() / 900).astype(int).values
        dep = np.floor((g.departure - t0).dt.total_seconds() / 900).astype(int).values
        dep = np.maximum(dep, arr + 1)
        keep = arr < WEEK
        arr, dep = arr[keep], np.minimum(dep[keep], HORIZON)
        pmax = g.pmax.values[keep]
        energy = np.minimum(g.energy.values[keep], 0.98 * pmax * (dep - arr) * DT)
        pw = price_weeks[len(eps) % len(price_weeks)]
        p = price15.loc[pw:].values[: HORIZON + LOOKAHEAD]
        p = p / p[:HORIZON].mean()
        eps.append(Episode(arr, dep, energy, pmax, p.astype(np.float32), t0, site))
    return eps


@dataclass
class BatchSim:
    cfg: SimConfig
    episodes: list
    rng: np.random.Generator = field(init=False)

    # ---- per-slot feature layout -------------------------------------------------
    SLOT_FEATS = 6   # present, rem/pmax-hours, time-left, laxity, pmax, rem/energy0
    GLOB_FEATS = 6 + 2 + N_PRICE_FEATS + 1 + 3

    def __post_init__(self):
        self.rng = np.random.default_rng(self.cfg.seed)
        self.B = len(self.episodes)
        self.reset()

    # ------------------------------------------------------------------------------
    def reset(self):
        B, M = self.B, self.cfg.M
        self.t = 0
        self.present = np.zeros((B, M), bool)
        self.rem = np.zeros((B, M))
        self.e0 = np.ones((B, M))
        self.dep = np.zeros((B, M), int)          # true departure
        self.dep_obs = np.zeros((B, M), float)    # announced departure
        self.pmax = np.zeros((B, M))
        self.ptr = np.zeros(B, int)               # next session index per episode
        self.order = [np.argsort(e.arr, kind="stable") for e in self.episodes]
        self.price = np.stack([e.price for e in self.episodes])
        caps = np.array([e.p_cap if e.p_cap > 0 else self.cfg.p_cap for e in self.episodes])
        self.cap = caps[:, None]
        # metrics
        self.cost = np.zeros(B)
        self.unmet = np.zeros(B)
        self.delivered = np.zeros(B)
        self.demand = np.array([e.energy.sum() for e in self.episodes])
        self.dropped = np.zeros(B)
        self.peak = np.zeros(B)
        self._arrive()
        return self.obs()

    def _arrive(self):
        t = self.t
        for b, ep in enumerate(self.episodes):
            o = self.order[b]
            while self.ptr[b] < len(o) and ep.arr[o[self.ptr[b]]] == t:
                i = o[self.ptr[b]]
                self.ptr[b] += 1
                free = np.flatnonzero(~self.present[b])
                if free.size == 0:
                    self.dropped[b] += ep.energy[i]
                    continue
                k = free[0]
                self.present[b, k] = True
                self.rem[b, k] = self.e0[b, k] = ep.energy[i]
                self.dep[b, k] = ep.dep[i]
                noise = self.rng.normal(0, self.cfg.dep_noise_h / DT) if self.cfg.dep_noise_h else 0
                self.dep_obs[b, k] = max(t + 1, ep.dep[i] + noise)
                self.pmax[b, k] = ep.pmax[i]

    # ------------------------------------------------------------------------------
    def max_rate(self):
        """Per-slot feasible rate ignoring the hub limit (kW)."""
        return np.where(self.present, np.minimum(self.pmax, self.rem / DT), 0.0)

    def steps_left(self):
        return np.maximum(self.dep_obs - self.t, 1.0)

    def laxity(self):
        """Announced steps left minus steps needed at max rate (inf for empty slots)."""
        need = self.rem / (np.maximum(self.pmax, 1e-6) * DT)
        return np.where(self.present, self.steps_left() - need, np.inf)

    def min_rate(self):
        """Smallest rate now that still lets the EV finish at max rate afterwards."""
        r = (self.rem - self.pmax * (self.steps_left() - 1) * DT) / DT
        return np.clip(np.where(self.present, r, 0.0), 0, None)

    def infeasible_energy(self, hub=True):
        """Lower bound on the energy of the plugged-in EVs that can no longer be
        delivered (kWh, per episode), assuming no further arrivals.

        ev  : sum_i [rem_i - pmax_i * time_left_i]^+          (each EV alone)
        hub : max_k [sum_{dep_i <= dep_k} rem_i - P_cap * (dep_k - t)]^+
              (EDF / processor-demand condition for the shared hub limit)
        The bound is max(ev, hub); both are necessary conditions of feasibility.
        """
        left = np.maximum(self.dep_obs - self.t, 0.0)
        ev = np.where(self.present, np.clip(self.rem - self.pmax * left * DT, 0, None), 0.0).sum(1)
        if not hub:
            return ev
        deficit = self._edf_deficit()
        return np.maximum(ev, np.clip(deficit, 0, None).max(1))

    def _edf_deficit(self):
        """For each deadline k: energy due by k minus hub energy available until k."""
        left = np.maximum(self.dep_obs - self.t, 0.0)
        d = np.where(self.present, self.dep_obs, np.inf)
        order = np.argsort(d, 1)
        rem = np.take_along_axis(np.where(self.present, self.rem, 0.0), order, 1)
        dl = np.take_along_axis(np.where(self.present, left, 0.0), order, 1)
        pres = np.take_along_axis(self.present, order, 1)
        return np.where(pres, np.cumsum(rem, 1) - self.cap * dl * DT, -np.inf)

    def _fill(self, req, cap, order):
        """Grant requests in `order` until the per-row capacity is exhausted."""
        rq = np.take_along_axis(req, order, 1)
        before = np.cumsum(rq, 1) - rq
        g = np.minimum(rq, np.clip(cap - before, 0, None))
        out = np.empty_like(req)
        np.put_along_axis(out, order, g, 1)
        return out

    def apply(self, rates):
        """Safety layer: clip to EV limits, optional feasibility guard, hub limit."""
        mr = self.max_rate()
        r = np.minimum(np.clip(rates, 0, None), mr)
        if self.cfg.guard:
            r = np.maximum(r, np.minimum(self.min_rate(), mr))
        if self.cfg.projection == "proportional":
            tot = r.sum(1, keepdims=True)
            scale = np.where(tot > self.cap, self.cap / np.maximum(tot, 1e-9), 1.0)
            return r * scale
        # llf: urgent part first (least laxity first), then the rest of the requests
        order = np.argsort(self.laxity(), 1, kind="stable")
        urgent = np.minimum(self.min_rate(), r) if self.cfg.guard else np.zeros_like(r)
        g1 = self._fill(urgent, self.cap, order)
        left = self.cap - g1.sum(1, keepdims=True)
        g2 = self._fill(r - g1, left, order)
        return g1 + g2

    def step_rates(self, rates):
        """Advance one step with per-slot rates in kW. Returns per-episode reward."""
        r = self.apply(rates)
        e = r.sum(1) * DT
        p = self.price[:, self.t]
        step_cost = p * e
        self.rem = np.maximum(self.rem - r * DT, 0.0)
        self.cost += step_cost
        self.delivered += e
        self.peak = np.maximum(self.peak, r.sum(1) / self.cap[:, 0])
        self.t += 1
        # departures
        leaving = self.present & (self.dep <= self.t)
        un = np.where(leaving, self.rem, 0.0).sum(1)
        self.unmet += un
        self.present &= ~leaving
        self.rem[leaving] = 0.0
        self._arrive()
        done = self.t >= HORIZON
        if done:   # anything still plugged in is unmet (cannot happen: dep <= HORIZON)
            self.unmet += self.rem.sum(1)
        return -(step_cost + self.cfg.lam * un), done

    def step(self, action):
        """action in [0,1]^(B,M): fraction of the per-slot feasible rate."""
        return self.step_rates(np.clip(action, 0, 1) * self.max_rate())

    def aggregate_rates(self, u):
        """Disaggregate a hub power set-point u*P_cap (u in [0,1], shape (B,)).

        Urgent energy (feasibility guard) is always served first; the rest of the
        set-point is filled in least-laxity-first order.  Every EV pays the same
        price at a given step, so the order only matters for meeting deadlines.
        """
        mr = self.max_rate()
        order = np.argsort(self.laxity(), 1, kind="stable")
        urgent = np.minimum(self.min_rate(), mr)
        g1 = self._fill(urgent, self.cap, order)
        target = np.clip(u, 0, 1)[:, None] * self.cap
        left = np.clip(target - g1.sum(1, keepdims=True), 0, None)
        g2 = self._fill(mr - g1, left, order)
        return g1 + g2

    def step_aggregate(self, u):
        return self.step_rates(self.aggregate_rates(u))

    # ------------------------------------------------------------------------------
    def obs(self):
        """Returns (slots (B,M,F), mask (B,M), glob (B,G)), all scale-free."""
        t = self.t
        tl = np.maximum(self.dep_obs - t, 1.0)                       # steps left (announced)
        need = self.rem / (np.maximum(self.pmax, 1e-6) * DT)         # steps needed at pmax
        lax = (tl - need) / STEPS_DAY
        slots = np.stack([
            self.present.astype(float),
            need / STEPS_DAY,
            tl / STEPS_DAY,
            lax,
            self.pmax / 22.0,
            self.rem / np.maximum(self.e0, 1e-6),
        ], -1) * self.present[..., None]
        hod = (t % STEPS_DAY) / STEPS_DAY * 2 * np.pi
        dow = ((t // STEPS_DAY) % 7) / 7 * 2 * np.pi
        fut = self.price[:, t:t + LOOKAHEAD]                          # next 24 h
        mu, sd = fut.mean(1, keepdims=True), fut.std(1, keepdims=True) + 1e-3
        hourly = ((fut.reshape(self.B, N_PRICE_FEATS, -1).mean(-1)) - mu) / sd
        now = (self.price[:, t:t + 1] - mu) / sd
        rank = (fut < self.price[:, t:t + 1]).mean(1, keepdims=True)
        maxr = self.max_rate().sum(1, keepdims=True)
        glob = np.concatenate([
            np.sin([[hod]] * self.B), np.cos([[hod]] * self.B),
            np.sin([[dow]] * self.B), np.cos([[dow]] * self.B),
            self.present.mean(1, keepdims=True),
            maxr / self.cap,
            now, rank, hourly,
            np.full((self.B, 1), t / HORIZON),
            # hub-level congestion summaries
            self.min_rate().sum(1, keepdims=True) / self.cap,
            np.clip(self._edf_deficit().max(1, initial=-1e9, keepdims=True)
                    / (self.cap * DT * STEPS_DAY), -2, 2) * self.present.any(1, keepdims=True),
            np.where(self.present, self.rem, 0).sum(1, keepdims=True) / (self.cap * DT * STEPS_DAY),
        ], 1)
        return slots.astype(np.float32), self.present.copy(), glob.astype(np.float32)

    def metrics(self):
        J = self.cost + self.cfg.lam * (self.unmet + self.dropped)
        return dict(J=J, cost=self.cost, unmet=self.unmet + self.dropped,
                    delivered=self.delivered, demand=self.demand,
                    J_per_kwh=J / self.demand, cost_per_kwh=self.cost / np.maximum(self.delivered, 1e-9),
                    served=self.delivered / self.demand, peak=self.peak)

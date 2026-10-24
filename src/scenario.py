"""Train/validation/test episodes per site, with fixed price pools and hub limits.

Splits are chronological on the session data (60/20/20 of the weeks) so that the
test weeks are always in the future of the training weeks.  Prices are real GB
day-ahead-style half-hourly market index prices (Elexon APX MID), used as a
common exogenous signal for all sites so that the cross-site comparison isolates
the shift in user behaviour.  Price pools are disjoint in time:
    train 2018-2019 | val 2020 | test 2023-2025 | stress 2022 (energy crisis)
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data import PROC, RAW, SITES, load
from env import HORIZON, BatchSim, SimConfig, build_episodes

KAPPA = 0.8       # hub limit = KAPPA * P95 of the uncontrolled (ASAP) hub load
LAM = 3.0         # penalty per undelivered kWh, in units of the mean price
M_SLOTS = 48

PRICE_POOLS = {"train": ("2018-01-01", "2019-12-31"), "val": ("2020-01-01", "2020-12-31"),
               "test": ("2023-01-01", "2025-08-15"), "stress": ("2022-01-01", "2022-12-31")}


def load_price15() -> pd.Series:
    df = pd.read_csv(RAW / "gb_mid_prices.csv", parse_dates=["StartTime"])
    s = df.set_index("StartTime")["Price"].astype(float)
    s[(df.set_index("StartTime")["Volume"] <= 0).values] = np.nan   # empty auctions
    s.index = s.index.tz_convert(None)
    idx = pd.date_range(s.index.min().floor("D"), s.index.max().ceil("D"), freq="15min")
    return s.reindex(idx).interpolate(limit_direction="both")


def price_weeks(price15: pd.Series, pool: str) -> list:
    a, b = PRICE_POOLS[pool]
    mondays = pd.date_range(a, b, freq="W-MON")
    need = pd.Timedelta(minutes=15 * (HORIZON + 96))
    return [m for m in mondays if m + need <= price15.index[-1]]


_BANK = {}


def price_bank(pool: str = "train") -> np.ndarray:
    """All weeks of a price pool as normalised (n_weeks, HORIZON + LOOKAHEAD) arrays.

    Used for training-time augmentation: an episode's real arrivals are paired with
    a randomly drawn price week of the same (training) pool.
    """
    if pool not in _BANK:
        p15 = load_price15()
        rows = []
        for w in price_weeks(p15, pool):
            x = p15.loc[w:].values[: HORIZON + 96]
            rows.append(x / x[:HORIZON].mean())
        _BANK[pool] = np.stack(rows).astype(np.float32)
    return _BANK[pool]


@dataclass
class Scenario:
    site: str
    p_cap: float
    train: list
    val: list
    test: list
    stress: list

    def sim(self, split: str, seed=0, dep_noise_h=0.0, episodes=None, p_cap=None,
            projection="proportional", guard=False):
        cfg = SimConfig(M=M_SLOTS, p_cap=p_cap or self.p_cap, lam=LAM, dep_noise_h=dep_noise_h, seed=seed,
                        projection=projection, guard=guard)
        return BatchSim(cfg, episodes if episodes is not None else getattr(self, split))


def _set_caps(eps) -> None:
    """Hub limit of each week = KAPPA * P95 of its uncontrolled (ASAP) load.

    Sized per week because the sites grow strongly over the years (Palo Alto
    x7 between 2012 and 2019); a limit fitted on early weeks would make later
    weeks infeasible.  Floor: one EV at its max rate.
    """
    for e in eps:
        e.p_cap = 1e9
    sim = BatchSim(SimConfig(M=M_SLOTS, p_cap=1e9, lam=LAM), eps)
    loads, done = [], False
    while not done:
        loads.append(sim.max_rate().sum(1))
        _, done = sim.step_rates(sim.max_rate())
    loads = np.stack(loads, 1)
    for e, l in zip(eps, loads):
        e.p_cap = float(max(KAPPA * np.percentile(l[l > 0], 95), e.pmax.max()))


def build(site: str, price15=None) -> Scenario:
    path = PROC / f"scenario_{site}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    price15 = load_price15() if price15 is None else price15
    s = load(site)
    weeks = s.arrival.dt.to_period("W-SUN").dt.start_time
    uw = np.sort(weeks.unique())
    n = len(uw)
    cut1, cut2 = uw[int(0.6 * n)], uw[int(0.8 * n)]
    parts = {"train": s[weeks < cut1], "val": s[(weeks >= cut1) & (weeks < cut2)], "test": s[weeks >= cut2]}
    eps = {k: build_episodes(v, price15, price_weeks(price15, k), site) for k, v in parts.items()}
    eps["stress"] = build_episodes(parts["test"], price15, price_weeks(price15, "stress"), site)
    for v in eps.values():
        _set_caps(v)
    p_cap = float(np.median([e.p_cap for e in eps["train"]]))
    sc = Scenario(site, p_cap, eps["train"], eps["val"], eps["test"], eps["stress"])
    with open(path, "wb") as f:
        pickle.dump(sc, f)
    return sc


if __name__ == "__main__":
    from scenario import build, load_price15  # pickle Scenario under its module name
    p = load_price15()
    for site in SITES:
        sc = build(site, p)
        print(site, f"median P_cap(train)={sc.p_cap:.1f} kW, test={np.median([e.p_cap for e in sc.test]):.1f} kW", {k: len(getattr(sc, k)) for k in ["train", "val", "test", "stress"]},
              "sessions/week(train)=", np.mean([len(e.arr) for e in sc.train]).round(1))

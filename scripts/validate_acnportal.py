"""Cross-check our batched simulator against Caltech's ACN-Sim (acnportal).

The same weeks of real sessions are replayed in both simulators with the
earliest-deadline-first policy under the same aggregate limit.  acnportal works in
amps; we use a 1000 V single-phase network so that 1 A == 1 kW and ideal batteries.
ACN-Sim's EDF does not know each vehicle's max rate (it allocates up to the EVSE
limit), so the check uses one common max rate P for all EVs and EVSEs.
Writes results/validation_acnportal.csv (per-week delivered energy in both).
"""
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acnportal import acnsim  # noqa: E402
from acnportal.acnsim.network import ChargingNetwork  # noqa: E402
from acnportal.algorithms import SortedSchedulingAlgo, earliest_deadline_first  # noqa: E402

from baselines import edf, run_policy  # noqa: E402
from data import SITES  # noqa: E402
from env import DT, HORIZON  # noqa: E402
from scenario import build  # noqa: E402

V = 1000.0


def acn_week(ep, p_cap, M):
    net = ChargingNetwork()
    ids = [f"S{k}" for k in range(M)]
    P = float(ep.pmax.max())
    for sid in ids:
        net.register_evse(acnsim.models.EVSE(sid, max_rate=P), V, 0)
    net.add_constraint(acnsim.network.Current(ids), p_cap, name="hub")
    # slot assignment identical to BatchSim: lowest free slot at arrival
    free_at = np.zeros(M, int)
    evs = []
    for i in np.argsort(ep.arr, kind="stable"):
        k = np.flatnonzero(free_at <= ep.arr[i])
        if k.size == 0:
            continue
        k = k[0]
        free_at[k] = ep.dep[i]
        bat = acnsim.models.Battery(capacity=1e6, init_charge=0, max_power=ep.pmax[i])
        evs.append(acnsim.models.EV(int(ep.arr[i]), int(ep.dep[i]), float(ep.energy[i]), ids[k], f"e{i}", bat))
    events = acnsim.events.EventQueue([acnsim.events.PluginEvent(ev.arrival, ev) for ev in evs])
    sch = SortedSchedulingAlgo(earliest_deadline_first)
    sim = acnsim.Simulator(net, sch, events, pd.Timestamp("2020-01-01", tz="UTC"), period=15, verbose=False)
    sim.run()
    # acnsim reports energy in kWh using period/60 * V / 1000
    load = acnsim.aggregate_current(sim)[:HORIZON]
    delivered = sum(ev.energy_delivered for ev in sim.ev_history.values())
    return delivered, load * DT


if __name__ == "__main__":
    rows = []
    for site in SITES:
        sc = build(site)
        eps = []
        for e in sc.val[:5]:
            P = 7.0
            dur = (e.dep - e.arr) * DT
            eps.append(replace(e, pmax=np.full_like(e.pmax, P), energy=np.minimum(e.energy, 0.98 * P * dur)))
        ours = run_policy(sc.sim("val", episodes=eps), edf)
        for i, ep in enumerate(eps):
            d_acn, _ = acn_week(ep, ep.p_cap, 48)
            rows.append(dict(site=site, ep=i, demand=float(ep.energy.sum()),
                             delivered_ours=float(ours["delivered"][i]), delivered_acn=float(d_acn)))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df["rel_diff"] = (df.delivered_ours - df.delivered_acn) / df.demand
    df.to_csv(ROOT / "results" / "validation_acnportal.csv", index=False)
    print(df.groupby("site").rel_diff.describe())

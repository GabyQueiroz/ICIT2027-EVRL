"""Figures for the paper that are produced directly from the data and trained models.

  fig_data.pdf        arrival-hour distribution and dwell/flexibility per site
  fig_regret.pdf      regret per held-out site and test condition (baselines vs PPO)
  fig_week.pdf        one test week: price, hub load of LLF, MPC+S and PPO, hub limit
Usage: python scripts/figures.py [data|regret|week|all]
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from data import SITES, load  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
NAMES = {"paloalto": "Palo Alto", "dundee": "Dundee", "perth": "Perth", "trondheim": "Trondheim"}
COL = {"paloalto": "#1b9e77", "dundee": "#d95f02", "perth": "#7570b3", "trondheim": "#e7298a"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "figure.dpi": 200, "axes.spines.top": False,
                     "axes.spines.right": False})


def fig_data():
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.7))
    for s in SITES:
        d = load(s)
        h = d.arrival.dt.hour + d.arrival.dt.minute / 60
        hist, edges = np.histogram(h, bins=48, range=(0, 24), density=True)
        ax[0].plot(edges[:-1] + 0.25, hist, color=COL[s], lw=1, label=NAMES[s])
    data_d = [((load(s).departure - load(s).arrival).dt.total_seconds() / 3600).clip(upper=24) for s in SITES]
    bp = ax[1].boxplot(data_d, showfliers=False, patch_artist=True, widths=0.6)
    for p, s in zip(bp["boxes"], SITES):
        p.set_facecolor(COL[s]); p.set_alpha(0.6)
    for med in bp["medians"]:
        med.set_color("k")
    ax[1].set_xticks(range(1, 5), ["PA", "DU", "PE", "TR"])
    ax[1].set_ylabel("dwell time (h)")
    ax[0].set_xlabel("arrival hour")
    ax[0].set_ylabel("density")
    ax[0].set_xticks([0, 6, 12, 18, 24])
    ax[0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_data.pdf")
    plt.close(fig)


def fig_regret():
    import analyze
    b, r = analyze.load_all()
    evs = [("nominal", "nominal"), ("dep_noise_1h", "departure error 1 h")]
    methods = ["LLF", "PriceGreedy+S", "MPC+S"]
    archs = [("mlp", "PPO-MLP"), ("deepsets", "PPO-DS"), ("attention", "PPO-Att")]
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.5), sharex=True)
    for ax, (ev, title) in zip(axes, evs):
        pos, data, labels, colors = 0, [], [], []
        for k, s in enumerate(SITES):
            for m in methods:
                data.append(b[(b.site == s) & (b["eval"] == ev) & (b.method == m)].regret.values)
                colors.append("0.75")
            for a, _ in archs:
                x = r[(r.exp == "runs_loo") & (r.arch == a) & (r.target == s) & (~r.in_domain) & (r["eval"] == ev)]
                data.append(x.groupby("ep").regret.mean().values)
                colors.append({"mlp": "#80b1d3", "deepsets": "#fb8072", "attention": "#b3de69"}[a])
        positions = [i + (i // 6) * 1.0 for i in range(len(data))]
        bp = ax.boxplot(data, positions=positions, showfliers=False, patch_artist=True, widths=0.7)
        for p, c in zip(bp["boxes"], colors):
            p.set_facecolor(c)
        for med in bp["medians"]:
            med.set_color("k")
        ax.set_ylabel("regret (%)")
        ax.set_title(title, fontsize=8)
        centers = [np.mean(positions[6 * k:6 * k + 6]) for k in range(4)]
        ax.set_xticks(centers, [NAMES[s] for s in SITES])
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in ["0.75", "#80b1d3", "#fb8072", "#b3de69"]]
    fig.legend(handles, ["LLF / PG+S / MPC+S", "PPO-MLP", "PPO-DeepSets", "PPO-Attention"],
               frameon=False, fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               handlelength=1, columnspacing=0.8)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "fig_regret.pdf")
    plt.close(fig)
    return
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_regret.pdf")
    plt.close(fig)


def fig_week(site="dundee", ep=5, seed=0):
    """One held-out test week: hub load of three controllers and the price."""
    from agent import PPO
    from baselines import MPC, llf
    from env import DT, HORIZON
    from scenario import build
    sc = build(site)
    e = [sc.test[ep]]

    def trace(sim, step):
        loads = []
        sim.reset()
        done = False
        while not done:
            r = step(sim)
            loads.append(sim.apply(r).sum(1)[0])
            _, done = sim.step_rates(r)
        return np.array(loads), sim.metrics()

    l_llf, m_llf = trace(sc.sim("test", episodes=e), llf)
    l_mpc, m_mpc = trace(sc.sim("test", episodes=e, projection="llf", guard=True), MPC())
    others = [s for s in SITES if s != site]
    ag = PPO.load(ROOT / "results" / "runs_loo" / "models" / f"deepsets_loo_{site}_s{seed}.pt", [build(s) for s in others])
    sim = ag.make_sim(sc, "test", episodes=e)
    o = sim.reset()
    loads, done = [], False
    while not done:
        a = ag.act(*o)
        for _ in range(ag.cfg.action_repeat):
            r = sim.aggregate_rates(a[:, 0])
            loads.append(sim.apply(r).sum(1)[0])
            _, done = sim.step_rates(r)
        o = sim.obs()
    l_ppo, m_ppo = np.array(loads), sim.metrics()
    t = np.arange(HORIZON) * DT / 24
    days = slice(0, 4 * 96)
    fig, ax = plt.subplots(2, 1, figsize=(3.5, 2.1), sharex=True, gridspec_kw=dict(height_ratios=[1, 2]))
    ax[0].plot(t[days], e[0].price[:HORIZON][days], "k", lw=0.8)
    ax[0].set_ylabel("price\n(norm.)")
    for l_, lab, c in [(l_llf, "LLF", "0.5"), (l_mpc, "MPC+S", "#1f78b4"), (l_ppo, "PPO-DeepSets", "#e31a1c")]:
        ax[1].plot(t[days], l_[days], lw=0.8, color=c, label=lab)
    ax[1].axhline(e[0].p_cap, color="k", ls="--", lw=0.6, label=r"$\bar P$")
    ax[1].set_ylabel("hub load (kW)")
    ax[1].set_xlabel("day of the test week")
    ax[1].legend(frameon=False, fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_week.pdf")
    plt.close(fig)
    out = {k: dict(J=float(m["J"][0]), cost_per_kwh=float(m["cost_per_kwh"][0]), served=float(m["served"][0]))
           for k, m in [("LLF", m_llf), ("MPC+S", m_mpc), ("PPO-DeepSets", m_ppo)]}
    pd.DataFrame(out).T.to_csv(ROOT / "results" / "tables" / "week_example.csv")
    return out


def fig_transfer():
    """Single-source transfer matrices (E1): mean regret (%) of the agent trained on row, tested on column."""
    import analyze
    _, r = analyze.load_all()
    x = r[(r.exp == "runs") & (r["eval"] == "nominal")]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
    mats = {}
    for ax, (a, title) in zip(axes, [("deepsets", "PPO-DeepSets"), ("mlp", "PPO-MLP")]):
        mat = x[x.arch == a].groupby(["sources", "target"]).regret.mean().unstack().reindex(index=SITES, columns=SITES)
        mats[a] = mat
        ax.imshow(mat.values, cmap="viridis_r", vmin=0, vmax=np.nanmax([m.values for m in mats.values()]) + 1)
        for i in range(4):
            for j in range(4):
                v = mat.values[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                        color="w" if v > np.nanmedian(mat.values) else "k", weight="bold" if i == j else None)
        ax.set_xticks(range(4), ["PA", "DU", "PE", "TR"])
        ax.set_yticks(range(4), ["PA", "DU", "PE", "TR"])
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("test site")
    axes[0].set_ylabel("training site")
    fig.tight_layout(pad=0.8)
    fig.savefig(FIG / "fig_transfer.pdf")
    plt.close(fig)
    for a, m in mats.items():
        m.round(2).to_csv(ROOT / "results" / "tables" / f"transfer_{a}.csv")
    return mats


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("transfer", "all"):
        print(fig_transfer())
    if what in ("data", "all"):
        fig_data()
    if what in ("regret", "all"):
        fig_regret()
    if what in ("week", "all"):
        print(fig_week())

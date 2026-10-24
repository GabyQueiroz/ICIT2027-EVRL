"""Tables, figures and statistical tests for the paper.

Metric: regret (%) = 100 * (J - J*) / J*, where J* is the clairvoyant LP optimum of
the same week (lower is better, 0 = optimal).  Seeds are averaged per episode
before paired tests; tests are Wilcoxon signed-rank over test weeks with Holm
correction across targets.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import SITES, load  # noqa: E402

RES, FIG, TAB = ROOT / "results", ROOT / "figures", ROOT / "results" / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
NAMES = {"paloalto": "Palo Alto", "dundee": "Dundee", "perth": "Perth", "trondheim": "Trondheim"}
ARCH = {"mlp": "PPO-MLP", "deepsets": "PPO-DeepSets", "attention": "PPO-Attention"}
BASE_SHOW = ["ASAP", "LLF", "PriceGreedy+S", "MPC+S"]
plt.rcParams.update({"font.size": 8, "font.family": "serif", "figure.dpi": 200})


def load_all():
    b = pd.read_csv(RES / "baselines.csv")
    opt = b[b.method == "Optimal"][["site", "eval", "ep", "J"]].rename(columns={"J": "J_opt"})
    opt_noise = opt[opt["eval"] == "nominal"].assign(eval="dep_noise_1h")
    opt = pd.concat([opt, opt_noise])
    b = b[b.method != "Optimal"].merge(opt, on=["site", "eval", "ep"])
    b["regret"] = 100 * (b.J - b.J_opt) / b.J_opt
    runs = []
    for d in ["runs", "runs_loo", "runs_abl"]:
        for f in (RES / d).glob("*.csv") if (RES / d).exists() else []:
            if not f.name.endswith("_curve.csv"):
                runs.append(pd.read_csv(f).assign(exp=d))
    r = pd.concat(runs) if runs else pd.DataFrame()
    if len(r):
        r = r.merge(opt.rename(columns={"site": "target"}), on=["target", "eval", "ep"])
        r["regret"] = 100 * (r.J - r.J_opt) / r.J_opt
    return b, r


def table_datasets():
    rows = []
    for s in SITES:
        d = load(s)
        dwell = (d.departure - d.arrival).dt.total_seconds() / 3600
        rows.append([NAMES[s], f"{len(d):,}", d.station.nunique(), f"{d.arrival.min():%Y}--{d.arrival.max():%Y}",
                     f"{d.energy.mean():.1f}", f"{dwell.mean():.1f}",
                     f"{100 * (1 - d.energy / (d.pmax * dwell)).mean():.0f}"])
    df = pd.DataFrame(rows, columns=["Site", "Sessions", "Chargers", "Period", "kWh/session", "Dwell (h)", r"Flex. (\%)"])
    df.to_latex(TAB / "datasets.tex", index=False, escape=False)
    return df


def fmt(m, s):
    return f"{m:.1f}" if np.isnan(s) else f"{m:.1f}$\\pm${s:.1f}"


def main_table(b, r, ev="nominal"):
    """Regret per target site: baselines, in-domain RL, leave-one-out RL."""
    out = {}
    for m in BASE_SHOW:
        x = b[(b.method == m) & (b["eval"] == ev)].groupby("site").regret.mean()
        out[m] = {s: fmt(x[s], np.nan) for s in SITES}
    for exp, label, cond in [("runs", "in-domain", True), ("runs_loo", "leave-one-out", False)]:
        sub = r[(r.exp == exp) & (r["eval"] == ev) & (r.in_domain == cond)] if len(r) else r
        for a in ARCH:
            x = sub[sub.arch == a] if len(sub) else sub
            if not len(x):
                continue
            per_seed = x.groupby(["target", "seed"]).regret.mean().groupby("target")
            mu, sd = per_seed.mean(), per_seed.std()
            out[f"{ARCH[a]} ({label})"] = {s: fmt(mu.get(s, np.nan), sd.get(s, np.nan)) for s in SITES}
    df = pd.DataFrame(out).T[SITES].rename(columns=NAMES)
    df.to_latex(TAB / f"main_{ev}.tex", escape=False)
    return df


def served_cost(b, r):
    """Delivered fraction (%) and energy cost per delivered kWh (normalised price), averaged over sites."""
    rows = {}
    for ev in ["nominal", "dep_noise_1h"]:
        for m in BASE_SHOW:
            x = b[(b.method == m) & (b["eval"] == ev)]
            rows[(m, ev)] = dict(served=100 * x.served.mean(), cost=x.cost_per_kwh.mean(), regret=x.regret.mean())
        if len(r):
            for a in ARCH:
                x = r[(r.exp == "runs_loo") & (r.arch == a) & (~r.in_domain) & (r["eval"] == ev)]
                if len(x):
                    rows[(ARCH[a], ev)] = dict(served=100 * x.served.mean(), cost=x.cost_per_kwh.mean(),
                                               regret=x.regret.mean())
    df = pd.DataFrame(rows).T
    df.to_csv(TAB / "served_cost.csv")
    return df.round(3)


def transfer_matrix(r, arch):
    x = r[(r.exp == "runs") & (r.arch == arch) & (r["eval"] == "nominal")]
    if not len(x):
        return None
    mat = x.groupby(["sources", "target"]).regret.mean().unstack()[SITES].reindex(SITES)
    fig, ax = plt.subplots(figsize=(2.6, 2.3))
    im = ax.imshow(mat.values, cmap="viridis_r")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mat.values[i, j]:.1f}", ha="center", va="center",
                    color="w" if mat.values[i, j] > np.nanmean(mat.values) else "k", fontsize=7)
    ax.set_xticks(range(4), [NAMES[s] for s in SITES], rotation=30)
    ax.set_yticks(range(4), [NAMES[s] for s in SITES])
    ax.set_xlabel("target (test weeks)")
    ax.set_ylabel("source (training weeks)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="regret (%)")
    fig.tight_layout()
    fig.savefig(FIG / f"transfer_{arch}.pdf")
    plt.close(fig)
    return mat


def generalization_gap(r):
    x = r[(r.exp == "runs") & (r["eval"] == "nominal")]
    g = x.groupby(["arch", "in_domain", "seed", "sources", "target"]).regret.mean().reset_index()
    t = g.groupby(["arch", "in_domain"]).regret.agg(["mean", "std"]).unstack()
    t.to_latex(TAB / "gap.tex", float_format="%.2f")
    return t


def robustness(b, r):
    rows = {}
    for m in BASE_SHOW:
        rows[m] = {ev: b[(b.method == m) & (b["eval"] == ev)].regret.mean() for ev in ["nominal", "stress", "dep_noise_1h"]}
    for a in ARCH:
        x = r[(r.exp == "runs_loo") & (r.arch == a) & (~r.in_domain)]
        if len(x):
            rows[f"{ARCH[a]} (LOO)"] = {ev: x[x["eval"] == ev].regret.mean() for ev in ["nominal", "stress", "dep_noise_1h"]}
    df = pd.DataFrame(rows).T
    df.to_latex(TAB / "robustness.tex", float_format="%.2f")
    return df


def tests(b, r):
    """Paired Wilcoxon over test weeks (seed-averaged), Holm-corrected per comparison."""
    res = []
    for ev in ["nominal", "dep_noise_1h"]:
        res += _tests_ev(b, r, ev)
    df = pd.DataFrame(res)
    if len(df):   # Holm across the four targets of each comparison and condition
        df["p_holm"] = df.groupby(["eval", "comparison"]).p.transform(
            lambda p: np.minimum(1, np.maximum.accumulate(np.sort(p.values) * (len(p) - np.arange(len(p))))
                                 [np.argsort(np.argsort(p.values))]))
        df.to_csv(TAB / "wilcoxon.csv", index=False)
    return df


def _tests_ev(b, r, ev):
    res = []
    x = r[(r.exp == "runs_loo") & (~r.in_domain) & (r["eval"] == ev)]
    for t in SITES:
        per = x[x.target == t].groupby(["arch", "ep"]).regret.mean().unstack(0)
        base = b[(b.site == t) & (b["eval"] == ev)].pivot_table(index="ep", columns="method", values="regret")
        pairs = [("deepsets", "mlp"), ("attention", "mlp"), ("deepsets", "attention")]
        for a1, a2 in pairs:
            if a1 in per and a2 in per:
                d = (per[a1] - per[a2]).dropna()
                res.append(dict(eval=ev, target=t, comparison=f"{ARCH[a1]} vs {ARCH[a2]}", median_diff=d.median(),
                                p=wilcoxon(d).pvalue if (d != 0).any() else 1.0, n=len(d)))
        for a in per:
            for m in ["MPC+S", "LLF"]:
                d = (per[a] - base[m]).dropna()
                res.append(dict(eval=ev, target=t, comparison=f"{ARCH[a]} vs {m}", median_diff=d.median(),
                                p=wilcoxon(d).pvalue if (d != 0).any() else 1.0, n=len(d)))
    return res


def curves():
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    for a, c in zip(ARCH, ["C0", "C1", "C2"]):
        fs = list((RES / "runs").glob(f"{a}_dundee_s*_curve.csv")) if (RES / "runs").exists() else []
        if not fs:
            continue
        cs = [pd.read_csv(f) for f in fs]
        n = min(len(c_) for c_ in cs)
        R = np.stack([c_.reward.values[:n] for c_ in cs])
        k = max(1, n // 50)
        R = np.stack([np.convolve(row, np.ones(k) / k, mode="valid") for row in R])
        st = cs[0].steps.values[: R.shape[1]] / 1e6
        ax.plot(st, R.mean(0), c, label=ARCH[a])
        ax.fill_between(st, R.mean(0) - R.std(0), R.mean(0) + R.std(0), color=c, alpha=0.2)
    ax.set_xlabel("simulator steps (M)")
    ax.set_ylabel("normalised reward")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "learning_curves.pdf")
    plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(table_datasets().to_string())
    b, r = load_all()
    print("\nMAIN (nominal test, regret %)\n", main_table(b, r).to_string())
    print("\nSTRESS prices\n", main_table(b, r, "stress").to_string())
    print("\nDEPARTURE ERROR 1 h\n", main_table(b, r, "dep_noise_1h").to_string())
    print("\nSERVED / COST\n", served_cost(b, r).to_string())
    if len(r):
        for a in ARCH:
            m = transfer_matrix(r, a)
            if m is not None:
                print(f"\ntransfer {a}\n", m.round(2).to_string())
        print("\nGAP\n", generalization_gap(r).to_string())
        print("\nROBUSTNESS\n", robustness(b, r).to_string())
        print("\nTESTS\n", tests(b, r).to_string())
        curves()

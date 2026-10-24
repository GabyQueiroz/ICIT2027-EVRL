"""Hyperparameter search for PPO with Optuna (TPE sampler + median pruner).

Tuning protocol (fixed before any test-set evaluation):
  * development site: Dundee; train on its training weeks, score on its
    validation weeks (never on test weeks, never on other sites);
  * objective: mean J per kWh on the validation weeks (lower is better);
  * 1M simulator steps per trial, intermediate value at 50% for pruning;
  * one study per architecture, same search space for all architectures
    (attention adds the number of heads).

Usage: python scripts/hpo.py --arch deepsets --trials 12 --worker 0
(several workers can run the same study in parallel: journal storage on disk)
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import PPO, PPOConfig  # noqa: E402
from scenario import build  # noqa: E402

DEV_SITE = "dundee"
STEPS = 1_000_000
N_VAL = 20


def objective(trial, arch):
    cfg = PPOConfig(
        arch=arch,
        hidden=trial.suggest_categorical("hidden", [64, 128, 256]),
        layers=trial.suggest_int("layers", 1, 3),
        activation=trial.suggest_categorical("activation", ["tanh", "relu"]),
        lr=trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        ent_coef=trial.suggest_float("ent_coef", 1e-5, 1e-2, log=True),
        gamma=trial.suggest_categorical("gamma", [0.95, 0.97, 0.99]),
        gae_lambda=trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98]),
        clip=trial.suggest_categorical("clip", [0.1, 0.2, 0.3]),
        epochs=trial.suggest_categorical("epochs", [3, 5, 10]),
        rollout_len=trial.suggest_categorical("rollout_len", [48, 96, 192]),
        action_repeat=trial.suggest_categorical("action_repeat", [2, 4]),
        heads=trial.suggest_categorical("heads", [2, 4]) if arch == "attention" else 4,
        n_envs=trial.suggest_categorical("n_envs", [16, 32]),
        total_steps=STEPS, seed=trial.number, threads=2,
    )
    sc = build(DEV_SITE)
    val = sc.val[:N_VAL]
    agent = PPO(cfg, sc)

    def cb(ag, u, n):
        if u == n // 2:
            v = float(np.mean(ag.evaluate(sc, "val", episodes=val)["J_per_kwh"]))
            trial.report(v, 1)
            if trial.should_prune():
                return False
        return True

    agent.learn(log_every=0, callback=cb)
    if trial.should_prune():
        raise optuna.TrialPruned()
    m = agent.evaluate(sc, "val", episodes=val)
    trial.set_user_attr("served", float(np.mean(m["served"])))
    trial.set_user_attr("cost_per_kwh", float(np.mean(m["cost_per_kwh"])))
    return float(np.mean(m["J_per_kwh"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--worker", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / "results" / "hpo"
    out.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            str(out / f"{a.arch}.log"),
            lock_obj=optuna.storages.journal.JournalFileOpenLock(str(out / f"{a.arch}.log"))))  # no symlinks on Windows)
    study = optuna.create_study(
        study_name=f"ppo_{a.arch}", storage=storage, direction="minimize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=1000 + a.worker, n_startup_trials=6),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=0))
    study.optimize(lambda t: objective(t, a.arch), n_trials=a.trials)
    best = study.best_trial
    (out / f"best_{a.arch}.json").write_text(json.dumps(dict(value=best.value, params=best.params), indent=2))
    print("best", a.arch, best.value, best.params)

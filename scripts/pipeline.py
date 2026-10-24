"""Run everything after the hyperparameter search, in order, and keep results/STATUS.txt current.

  1. wait for the three HPO studies (results/hpo/best_<arch>.json)
  2. baselines with all evaluation settings (results/baselines.csv)
  3. experiments E2 (leave-one-out), E1 (single source), E3 (ablations)
  4. analysis: tables and figures
Progress: python scripts/status.py
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "results" / "STATUS.txt"
STEPS = 1_500_000
WORKERS = 6


def status(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M}  {msg}"
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


# Compute budget (laptop CPU, conference deadline): the attention backbone was ~5x slower per
# trial than DeepSets, so it is not tuned separately; it reuses the tuned DeepSets
# configuration (same per-EV encoder and PPO settings) with 4 attention heads.
HPO_TRIALS = {"deepsets": 10, "mlp": 20}


def _study(a):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    f = str(ROOT / "results" / "hpo" / f"{a}.log")
    return optuna.load_study(study_name=f"ppo_{a}", storage=optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(f, lock_obj=optuna.storages.journal.JournalFileOpenLock(f))))


def hpo_finished():
    """All workers done: every planned trial is complete or pruned."""
    try:
        return all(sum(t.state.name in ("COMPLETE", "PRUNED") for t in _study(a).trials) >= n
                   for a, n in HPO_TRIALS.items())
    except Exception:  # noqa: BLE001 - study file being written
        return False


def write_best():
    import json
    for a in HPO_TRIALS:
        b = _study(a).best_trial
        (ROOT / "results" / "hpo" / f"best_{a}.json").write_text(
            json.dumps(dict(value=b.value, trial=b.number, params=b.params), indent=2))
        status(f"   melhor {a}: J_val={b.value:.4f} (trial {b.number})")
    ds = json.loads((ROOT / "results" / "hpo" / "best_deepsets.json").read_text())
    (ROOT / "results" / "hpo" / "best_attention.json").write_text(
        json.dumps(dict(value=None, note="DeepSets configuration reused", params={**ds["params"], "heads": 4}), indent=2))
    status("   attention: usa a configuracao do DeepSets (4 cabecas)")


def run(args, log):
    with open(ROOT / "results" / log, "w") as f:
        r = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        status(f"ERRO em {' '.join(args)} (ver results/{log})")
        sys.exit(1)


if __name__ == "__main__":
    status("ETAPA 1/4: aguardando a busca de hiperparametros (Optuna)")
    while not hpo_finished():
        time.sleep(60)
    # stop any worker still running a surplus trial so it does not compete for CPU
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                    "Where-Object { $_.CommandLine -like '*hpo.py*' } | "
                    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"])
    write_best()
    status("ETAPA 2/4: baselines (ASAP, EDF, LLF, PriceGreedy, MPC, otimo offline)")
    import pandas as pd
    b = ROOT / "results" / "baselines.csv"
    if not (b.exists() and "dep_noise_1h" in set(pd.read_csv(b)["eval"])):
        run(["scripts/run_baselines.py"], "baselines.log")
    for exp, name in [("E2", "leave-one-out"), ("E1", "fonte unica"), ("E3", "ablacoes")]:
        status(f"ETAPA 3/4: experimentos {exp} ({name})")
        run(["scripts/run_experiments.py", "--workers", str(WORKERS), "--steps", str(STEPS), "--only", exp],
            f"experiments_{exp}.log")
    status("ETAPA 4/4: analise (tabelas e figuras)")
    run(["scripts/analyze.py"], "analysis.txt")
    status("CONCLUIDO: tudo rodou. Resultados em results/analysis.txt, results/tables e figures/")

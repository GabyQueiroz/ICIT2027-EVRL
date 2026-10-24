"""Where is the pipeline?  python scripts/status.py"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def hpo():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for a, total in [("deepsets", 10), ("mlp", 20)]:
        f = ROOT / "results" / "hpo" / f"{a}.log"
        if not f.exists():
            continue
        st = optuna.load_study(study_name=f"ppo_{a}", storage=optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(str(f), lock_obj=optuna.storages.journal.JournalFileOpenLock(str(f)))))
        done = [t for t in st.trials if t.state.name in ("COMPLETE", "PRUNED")]
        best = min((t.value for t in st.trials if t.state.name == "COMPLETE"), default=None)
        fin = (ROOT / "results" / "hpo" / f"best_{a}.json").exists()
        print(f"  HPO {a:9s}: {len(done):2d}/{total} trials" + (f", melhor J={best:.4f}" if best else "")
              + ("  [terminado]" if fin else ""))


def experiments():
    from run_experiments import jobs
    try:
        js = jobs(1, None)
    except FileNotFoundError:   # ablations need best_deepsets.json
        js = jobs(1, "E2") + jobs(1, "E1")
    done = [j for j in js if (ROOT / "results" / j["out"] / f"{j['tag']}.csv").exists()]
    logs = ROOT / "results" / "logs"
    running = []
    if logs.exists():
        now = time.time()
        for j in js:
            lf = logs / f"{j['tag']}.log"
            if lf.exists() and j not in done and now - lf.stat().st_mtime < 3 * 3600:
                running.append(j["tag"])
    print(f"  Experimentos: {len(done)}/{len(js)} treinos concluidos, {len(running)} rodando agora")
    for t in running:
        print("     rodando:", t)
    if done:
        times = sorted((ROOT / "results" / j["out"] / f"{j['tag']}.csv").stat().st_mtime for j in done)
        if len(times) >= 2:
            rate = (times[-1] - times[0]) / (len(times) - 1)
            eta = rate * (len(js) - len(done)) / 3600
            print(f"  Estimativa: ~{eta:.1f} h para terminar os treinos")


if __name__ == "__main__":
    st = ROOT / "results" / "STATUS.txt"
    print("=" * 60)
    if st.exists():
        lines = st.read_text(encoding="utf-8").strip().splitlines()
        print("Historico:")
        for line in lines:
            print("  " + line)
        print("\nETAPA ATUAL:", lines[-1][18:])
    else:
        print("Pipeline ainda nao iniciado.")
    print()
    hpo()
    experiments()
    print("=" * 60)
    if st.exists() and "CONCLUIDO" in st.read_text(encoding="utf-8"):
        print(">>> TUDO PRONTO <<<")

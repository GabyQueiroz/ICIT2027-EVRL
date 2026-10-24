"""Print the state of the three Optuna studies."""
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
for a in ["deepsets", "mlp", "attention"]:
    try:
        st = optuna.load_study(study_name=f"ppo_{a}", storage=optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(f"results/hpo/{a}.log", lock_obj=optuna.storages.journal.JournalFileOpenLock(f"results/hpo/{a}.log"))))
    except Exception as e:  # noqa: BLE001
        print(a, "no study", e); continue
    print(a, [(t.number, t.state.name[:4], round(t.value, 4) if t.value is not None else None) for t in st.trials])

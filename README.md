# Set-based PPO for cross-site EV smart charging (ICIT 2027)

Code for the paper *"Cross-Site Transfer of Set-Based Deep Reinforcement Learning for EV Charging-Hub
Control on Four Public Session Datasets"* (ICIT 2027 submission; LaTeX in `paper/icit2027_evrl.tex`).

Extra scripts: `scripts/attrition.py` (sessions removed per rule), `scripts/figures.py` (paper figures),
`scripts/timing.py` (decision time), `scripts/pipeline.py` + `scripts/status.py` (full run with status).

Everything here runs on a laptop CPU, uses only public data, and every number in
the paper is produced by the scripts below from fixed seeds.

## Data (all public, downloaded by hand or by script)

| Site | Source | Licence | Kept |
|---|---|---|---|
| Palo Alto, US (public, city) | data.paloalto.gov dataset 194693 | City open data | Level-1/2 sessions 2016–2020 |
| Dundee, UK (public, city) | Dundee City Council, ArcGIS Hub (2021–2025 files) | OGL | AC sessions |
| Perth & Kinross, UK (public, rural/town) | Perth & Kinross Council, ArcGIS Hub (2016–2019) | OGL | 7/22 kW AC sessions |
| Trondheim, NO (residential garages) | Sørensen et al., Mendeley Data `jbks2rcwyj` | CC BY 4.0 | all sessions |
| GB electricity prices | Elexon Insights API, APX market index (half-hourly) | Elexon open data | 2018–2025 |

Put the raw files in `data/raw/` (`paloalto.csv`, `dundee_*.csv`, `perth_*.csv`,
`trondheim_sessions.csv`), then run `python scripts/download_prices.py`.

## Pipeline

```bash
pip install -r requirements.txt
python src/data.py                     # harmonise sessions -> data/processed/*.parquet
python src/scenario.py                 # weekly episodes, chronological 60/20/20 splits, hub limits
python scripts/validate_acnportal.py   # cross-check our simulator against Caltech ACN-Sim
python scripts/run_baselines.py        # ASAP/EDF/LLF/PriceGreedy/MPC + clairvoyant LP optimum
python scripts/hpo.py --arch deepsets --trials 10 --worker 0   # (also mlp, attention; workers in parallel)
python scripts/run_experiments.py --workers 5 --steps 2000000
python scripts/analyze.py              # tables (results/tables) and figures (figures/)
```

## Design (what is in `src/`)

* `data.py` – one schema for four datasets: arrival, departure, energy, max rate.
* `env.py` – batched 15-min simulator. One episode = one week of real arrivals + 24 h tail.
  Hub limit per week = 0.8 × P95 of the uncontrolled load. Objective
  `J = Σ p_t E_t + λ U` (prices normalised by the week mean, λ = 3).
* `baselines.py` – ASAP, EDF, LLF, price-greedy, receding-horizon LP (MPC), offline LP optimum.
* `agent.py` – PPO with three backbones (flat MLP, DeepSets, self-attention). The action is the
  hourly hub set-point; a safety layer disaggregates it least-laxity-first and always serves
  the energy needed to stay feasible. Training uses an exactly equivalent centred-price reward
  and potential-based shaping on an EDF lower bound of undeliverable energy.

## Protocol

* Tuning only on Dundee validation weeks (Optuna TPE, median pruning, 1M steps per trial).
* Test weeks are always in the future of training weeks; test prices (2023–2025) are disjoint
  from training prices (2018–2019); a stress test uses 2022 crisis prices.
* Five seeds per configuration, Wilcoxon signed-rank tests over test weeks with Holm correction.

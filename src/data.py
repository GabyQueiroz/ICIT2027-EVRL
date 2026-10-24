"""Harmonise four public EV-charging session datasets into one schema.

Output schema (one parquet per site, data/processed/<site>.parquet):
    arrival   (naive local datetime)   plug-in time
    departure (naive local datetime)   plug-out time
    energy    (kWh)                    energy actually delivered (= user demand)
    pmax      (kW)                     max charging rate of the session
    station   (str)                    charge point identifier

Only AC (flexible) sessions are kept; DC rapid sessions have no temporal
flexibility and are out of scope.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

MIN_ENERGY = 0.5        # kWh
MIN_DWELL_H = 0.25      # one 15-min step
MAX_DWELL_H = 24.0      # sessions are truncated to 24 h of plug-in time
PMAX_CAP = 22.0         # kW, AC three-phase upper bound

# nominal per-site AC power (kW) used when the session does not reveal more
NOMINAL_KW = {"paloalto": 6.6, "dundee": 7.4, "perth": 7.4, "trondheim": 3.7}


def _finish(df: pd.DataFrame, site: str) -> pd.DataFrame:
    df = df.dropna(subset=["arrival", "departure", "energy"]).copy()
    dwell = (df.departure - df.arrival).dt.total_seconds() / 3600
    df = df[(dwell >= MIN_DWELL_H) & (df.energy >= MIN_ENERGY)].copy()
    dwell = (df.departure - df.arrival).dt.total_seconds() / 3600
    long = dwell > MAX_DWELL_H
    df.loc[long, "departure"] = df.loc[long, "arrival"] + pd.Timedelta(hours=MAX_DWELL_H)
    dwell = dwell.clip(upper=MAX_DWELL_H)
    nominal = df.pop("nominal") if "nominal" in df else NOMINAL_KW[site]
    # the observed average rate is a lower bound of the vehicle's max rate
    df["pmax"] = np.minimum(np.maximum(nominal, 1.1 * df.energy / dwell), PMAX_CAP)
    # energy must be deliverable within the dwell at pmax
    df["energy"] = np.minimum(df.energy, 0.98 * df.pmax * dwell)
    df["station"] = df.station.astype(str)
    df = df.sort_values("arrival").reset_index(drop=True)
    return df[["arrival", "departure", "energy", "pmax", "station"]]


def load_paloalto() -> pd.DataFrame:
    raw = pd.read_csv(RAW / "paloalto.csv", low_memory=False)
    arr = pd.to_datetime(raw["Start Date"], format="%m/%d/%Y %H:%M", errors="coerce")
    dur = pd.to_timedelta(raw["Total Duration (hh:mm:ss)"], errors="coerce")
    df = pd.DataFrame({
        "arrival": arr,
        "departure": arr + dur,
        "energy": pd.to_numeric(raw["Energy (kWh)"], errors="coerce"),
        "station": raw["Station Name"] + "#" + raw["Port Number"].astype(str),
        "nominal": np.where(raw["Port Type"] == "Level 1", 1.9, NOMINAL_KW["paloalto"]),
    })
    # the network was being built out until 2015 (1.2k sessions in 2011 vs 45k in
    # 2019); keep the mature period only
    df = df[df.arrival >= "2016-01-01"]
    return _finish(df, "paloalto")


def load_dundee() -> pd.DataFrame:
    raw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RAW / "dundee_*.csv")))])
    raw = raw.drop_duplicates(subset=["SDR ID"])
    raw = raw[raw["Connector Type"].str.strip().str.lower() == "ac"]
    df = pd.DataFrame({
        "arrival": pd.to_datetime(raw["Start"], format="%d/%m/%Y %H:%M", errors="coerce"),
        "departure": pd.to_datetime(raw["End"], format="%d/%m/%Y %H:%M", errors="coerce"),
        "energy": pd.to_numeric(raw["Consum(kWh)"], errors="coerce"),
        "station": raw["CP ID"],
    })
    return _finish(df, "dundee")


def load_perth() -> pd.DataFrame:
    raw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RAW / "perth_*.csv")))])
    raw = raw[raw["Model"].isin(["APT 22kW Dual Outlet", "APT 22kW Raption",
                                 "Siemens 7 kW Dual Outlet", "APT 7kW Dual Outlet"])]
    raw = raw.drop_duplicates(subset=["CP ID", "Connector", "Start Date", "Start Time"])

    def ts(d, t):
        return pd.to_datetime(raw[d].str.slice(0, 10) + " " + raw[t], format="%Y-%m-%d %H:%M",
                              errors="coerce")

    df = pd.DataFrame({
        "arrival": ts("Start Date", "Start Time"),
        "departure": ts("End Date", "End Time"),
        "energy": pd.to_numeric(raw["Total kWh"], errors="coerce"),
        "station": raw["CP ID"].astype(str) + "#" + raw["Connector"].astype(str),
    })
    return _finish(df, "perth")


def load_trondheim() -> pd.DataFrame:
    raw = pd.read_csv(RAW / "trondheim_sessions.csv", sep=";", decimal=",")
    df = pd.DataFrame({
        "arrival": pd.to_datetime(raw.Start_plugin, format="%d.%m.%Y %H:%M", errors="coerce"),
        "departure": pd.to_datetime(raw.End_plugout, format="%d.%m.%Y %H:%M", errors="coerce"),
        "energy": pd.to_numeric(raw.El_kWh, errors="coerce"),
        "station": raw.Garage_ID + "/" + raw.User_ID,
    })
    return _finish(df, "trondheim")


LOADERS = {"paloalto": load_paloalto, "dundee": load_dundee,
           "perth": load_perth, "trondheim": load_trondheim}
SITES = list(LOADERS)


def load(site: str) -> pd.DataFrame:
    path = PROC / f"{site}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    PROC.mkdir(parents=True, exist_ok=True)
    df = LOADERS[site]()
    df.to_parquet(path)
    return df


if __name__ == "__main__":
    rows = []
    for s in SITES:
        d = load(s)
        dwell = (d.departure - d.arrival).dt.total_seconds() / 3600
        flex = 1 - d.energy / (d.pmax * dwell)
        rows.append(dict(site=s, sessions=len(d), stations=d.station.nunique(),
                         start=d.arrival.min().date(), end=d.arrival.max().date(),
                         energy_mean=d.energy.mean(), dwell_mean=dwell.mean(),
                         flex_mean=flex.mean(), arr_hour_mean=d.arrival.dt.hour.mean()))
    print(pd.DataFrame(rows).to_string())

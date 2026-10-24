"""Download GB half-hourly Market Index Data (APX MID) from the Elexon Insights API.

Public, no API key. Output: data/raw/gb_mid_prices.csv (UTC start time, GBP/MWh).
"""
import io
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "gb_mid_prices.csv"
URL = ("https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index"
       "?from={a}T00:00Z&to={b}T00:00Z&dataProviders=APXMIDP&format=csv")


def fetch(a: date, b: date) -> pd.DataFrame:
    for attempt in range(5):
        try:
            with urlopen(URL.format(a=a.isoformat(), b=b.isoformat()), timeout=60) as r:
                return pd.read_csv(io.StringIO(r.read().decode()))
        except Exception as e:  # noqa: BLE001 - network retries
            print("retry", a, e, file=sys.stderr)
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"failed {a}")


def main(start="2018-01-01", end="2025-09-01", step_days=7):
    a = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    parts = []
    while a < stop:
        b = min(a + timedelta(days=step_days), stop)
        parts.append(fetch(a, b))
        a = b
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["StartTime"]).sort_values("StartTime")
    df[["StartTime", "Price", "Volume"]].to_csv(OUT, index=False)
    print(OUT, len(df), df.StartTime.min(), df.StartTime.max())


if __name__ == "__main__":
    main()

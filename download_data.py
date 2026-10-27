import time
import requests
import pandas as pd

URL = "https://api.binance.com/api/v3/klines"

def fetch(symbol="BTCUSDT", interval="1h", total=10000):
    rows = []
    end = None
    while len(rows) < total:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end:
            params["endTime"] = end
        r = requests.get(URL, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows = batch + rows
        end = batch[0][0] - 1
        time.sleep(0.2)

    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_base", "taker_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    return df.drop_duplicates("open_time").sort_values("open_time")

if __name__ == "__main__":
    df = fetch()
    df.to_csv("btcusdt_1h.csv", index=False)
    print(df.shape)
    print(df.head())
    print(df.tail())
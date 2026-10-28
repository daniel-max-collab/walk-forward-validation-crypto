import pandas as pd

HORIZON = 24


def load_data(path="btcusdt_1h.csv"):
    df = pd.read_csv(path, parse_dates=["open_time"])
    return df.sort_values("open_time").reset_index(drop=True)


def build_features(df):
    ret = df["close"].pct_change()
    return pd.DataFrame({
        "ret_1": ret,
        "ret_3": df["close"].pct_change(3),
        "ret_6": df["close"].pct_change(6),
        "ret_24": df["close"].pct_change(24),
        "vol_24": ret.rolling(24).std(),
        "vol_ratio": df["volume"] / df["volume"].rolling(24).mean(),
    })


def make_target(df, horizon=HORIZON):
    future_ret = df["close"].shift(-horizon) / df["close"] - 1
    y = (future_ret > 0).astype(int)
    return y, future_ret.notna()


def make_dataset(df, horizon=HORIZON):
    X = build_features(df)
    y, has_future = make_target(df, horizon)
    data = X.copy()
    data["y"] = y
    data = data[has_future].dropna()
    return data.drop(columns="y"), data["y"]
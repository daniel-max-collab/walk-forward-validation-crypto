import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from features import build_features, make_target, make_dataset


def make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "open_time": pd.date_range("2026-01-01", periods=n, freq="h"),
        "close": close,
        "volume": rng.uniform(50, 150, n),
    })


def test_features_ignore_the_future():
    df = make_df()
    before = build_features(df)
    changed = df.copy()
    changed.loc[150:, "close"] *= 10
    changed.loc[150:, "volume"] *= 10
    after = build_features(changed)
    pd.testing.assert_frame_equal(before.iloc[:150], after.iloc[:150])


def test_target_is_future_direction():
    df = make_df()
    y, _ = make_target(df, horizon=24)
    for i in [0, 50, 100]:
        assert y[i] == int(df["close"][i + 24] > df["close"][i])


def test_last_rows_have_no_target():
    df = make_df()
    _, has_future = make_target(df, horizon=24)
    assert not has_future.iloc[-24:].any()
    assert has_future.iloc[:-24].all()


def test_dataset_has_no_missing_values():
    X, y = make_dataset(make_df())
    assert not X.isna().any().any()
    assert len(X) == len(y)


def test_walk_forward_trains_on_past_only():
    X, _ = make_dataset(make_df(500))
    for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(X):
        assert train_idx.max() < test_idx.min()
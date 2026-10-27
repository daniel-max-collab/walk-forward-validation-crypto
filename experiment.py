import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score

HORIZON = 24  # predict direction 24 hours ahead

df = pd.read_csv("btcusdt_1h.csv", parse_dates=["open_time"])
df = df.sort_values("open_time").reset_index(drop=True)

ret = df["close"].pct_change()
X = pd.DataFrame({
    "ret_1": ret,
    "ret_3": df["close"].pct_change(3),
    "ret_6": df["close"].pct_change(6),
    "ret_24": df["close"].pct_change(24),
    "vol_24": ret.rolling(24).std(),
    "vol_ratio": df["volume"] / df["volume"].rolling(24).mean(),
})

future_ret = df["close"].shift(-HORIZON) / df["close"] - 1
y = (future_ret > 0).astype(int)

data = X.copy()
data["y"] = y
data = data[future_ret.notna()].dropna()
X, y = data.drop(columns="y"), data["y"]

print("rows used:", len(X))
print("baseline (always guess the majority):", round(max(y.mean(), 1 - y.mean()), 3))

model = RandomForestClassifier(
    n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1
)

random_cv = KFold(n_splits=5, shuffle=True, random_state=42)
walk_forward = TimeSeriesSplit(n_splits=5)

random_scores = cross_val_score(model, X, y, cv=random_cv, scoring="accuracy")
walk_scores = cross_val_score(model, X, y, cv=walk_forward, scoring="accuracy")

print("random CV accuracy:      ", random_scores.round(3), "mean", round(random_scores.mean(), 3))
print("walk-forward accuracy:   ", walk_scores.round(3), "mean", round(walk_scores.mean(), 3))
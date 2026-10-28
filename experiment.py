from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score

from features import load_data, make_dataset

X, y = make_dataset(load_data())

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
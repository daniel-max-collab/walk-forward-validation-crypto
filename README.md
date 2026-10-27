\# Why ordinary cross-validation breaks on time series



A small experiment on hourly BTCUSDT data (Binance public API, 10,000 rows,

2025-09-06 to 2026-10-27).



\## Question

Does a model that predicts the direction of price 24 hours ahead really work,

or does ordinary cross-validation just make it look like it does?



\## Method

\- Features use only past information: returns over 1/3/6/24 hours,

&#x20; 24-hour volatility, volume relative to its 24-hour average.

\- Target: 1 if the close is higher 24 hours later, else 0.

\- Model: RandomForestClassifier, scored with accuracy.

\- Compared shuffled 5-fold CV against walk-forward validation (TimeSeriesSplit).



\## Results (9,952 rows)

| Validation | Mean accuracy |

|---|---|

| Majority-class baseline | 0.505 |

| Random 5-fold CV | 0.579 |

| Walk-forward (5 splits) | 0.503 |



\## Takeaway

Random CV suggests a model with an edge. Walk-forward shows none: it scores

at the baseline. Shuffled splits leak future information, because the

24-hour targets of neighbouring hours overlap.



\## Limits

\- One asset, one horizon, one model, one period.

\- Overlapping targets can still leak slightly at walk-forward split

&#x20; boundaries. A purged/embargoed split would be stricter.

\- Accuracy ignores trading costs; this is not a trading strategy.



\## Run it

&#x20;   pip install -r requirements.txt

&#x20;   python download\_data.py

&#x20;   python experiment.py


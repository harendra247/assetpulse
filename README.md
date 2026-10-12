# AssetPulse: predictive-maintenance proof of concept

A small, explainable predictive-maintenance pipeline for one asset: an **AHU supply fan** (belt-driven
centrifugal fan, 7.5 kW induction motor, VFD). It simulates vibration, sound, temperature and electrical
signals, injects four progressive faults, and shows health, anomalies, the likely fault, days to maintenance
and a recommended action in a Streamlit dashboard.

> All data is **synthetic**. Results show the approach and that the pipeline works. They say nothing about
> accuracy on a real machine.

## Run it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py          # dashboard, http://localhost:8501
python scripts/validate.py    # five 60-day scenarios vs. the success criteria
python -m pytest -q           # 31 unit tests
```

## What is in it

| Module | Job |
| --- | --- |
| `assetpulse/config.py` | Every tunable number: scenarios, fault fingerprints, thresholds, actions |
| `assetpulse/simulator.py` | Healthy baseline + bearing wear, imbalance, belt slip, clogged filter (ground truth included) |
| `assetpulse/twin.py` | The asset: rated spec, components, operating state, per-component health |
| `assetpulse/analytics.py` | Baseline, z-scores, anomaly score, diagnosis, health index, days remaining, risk, advice |
| `assetpulse/pipeline.py` | Runs a scenario and answers "what is the state at reading N?" |
| `assetpulse/viz.py`, `app.py`, `.streamlit/config.toml` | Dark-theme panel dashboard: status tiles, health trend, fault donut, deviation bars, signal table, twin, signal charts |
| `scripts/validate.py`, `calibrate.py` | Success-criteria check; derives the failure-level numbers in config |

## How the analytics work

1. **Features**: 10 numbers per reading (vibration overall / 1x / high-frequency, sound level / high-frequency band,
   bearing and winding temperature rise over ambient, current, power, current ripple).
2. **Baseline**: for each feature, a smooth function of fan speed learned from the first 14 healthy days
   (cubic fit) plus a robust noise scale (MAD). Speed matters: power rises with speed cubed, so "normal" at 55% and 100% speed differ.
3. **Anomaly**: z-score per feature, smoothed over 6 h, combined as a root-mean-square. A flag needs the score above a
   threshold learned from the healthy period for 6 hours in a row, and stays on until it falls well below.
4. **Diagnosis**: cosine similarity between the observed deviation pattern and each fault's fingerprint
   (e.g. imbalance = vibration at 1x; clogged filter = power and current *down*). Below a minimum similarity the answer is "unclassified".
   Confidence and the supporting signals are shown.
5. **Health index** 0-100: how far along the diagnosed fault's fingerprint the signals have moved, relative to
   where they sit at failure. Healthy ≥ 70, Watch 40-70, Act < 40.
6. **Days to maintenance**: an accelerating power-law curve is fitted to the decline since the alarm and extrapolated to
   the Act line; the range comes from all fits nearly as good as the best.
7. **Risk and action**: risk from days remaining (>14 Low, 3-14 Medium, <3 High), then a (fault, risk) lookup.

## Validation results (seed 7; 20 seeds in brackets)

| Scenario | First alarm | Lead before failure | Diagnosis | Days-to-Act error at midpoint |
| --- | --- | --- | --- | --- |
| Healthy | none | n/a | n/a | 0 false alarms (20/20 seeds) |
| Bearing wear | day 38.8 | 17 d | correct (20/20) | 3% (20/20 within 30%) |
| Imbalance | day 39.7 | 18 d | correct (20/20) | 31% (11/20 within 30%) |
| Belt slip | day 28.3 | 24 d | correct (20/20) | 5% (20/20 within 30%) |
| Clogged filter | day 28.6 | 29 d | correct (20/20) | 9% (8/20 within 30%) |

Detection, lead time (needed ≥ 5 days) and diagnosis meet the criteria on every run. **The days-remaining
accuracy criterion is only partly met**: bearing and belt are accurate, imbalance is biased about 30% pessimistic
(predicts maintenance earlier than needed), and the clogged filter, whose signal is weak next to its noise, is
noisy (median error across all runs 9%, worst case about 190%). The quoted range contains the true value in
roughly 84% of runs.

Notes on how the PRD criteria were made testable:
- "True" days remaining is when the 24 h-smoothed health index actually crossed the Act line in that run,
  judged at the midpoint between the first alarm and that crossing.
- The failure-level numbers (`FAIL_Z`) and the power-law fit family were tuned on this simulator, so
  these results flatter the method. Real use would need failure data or alarm limits to set them.

## Limits

One fault at a time; faults and noise are simulated by the same author as the analytics; no real sensor
problems (dropouts, drift, miscalibration); a single asset; no database or alerts.

## Where it would go next

Replay real logged data through the same pipeline; learn fingerprints and failure levels from history instead
of hand-set values; handle several simultaneous faults; add sensor-health checks; compare against a learned
model (e.g. isolation forest or an RUL regressor) using the rule-based version as the explainable baseline.

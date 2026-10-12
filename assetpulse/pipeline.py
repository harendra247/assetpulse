"""Glue: simulate a scenario, run the analytics over it, and answer 'what is the state at time X?'."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import analytics as A
from . import config as C
from .simulator import simulate
from .twin import FanTwin, status_from_hi


@dataclass
class Result:
    scenario: C.Scenario
    df: pd.DataFrame
    feats: pd.DataFrame
    zs: pd.DataFrame            # smoothed z-scores
    score: pd.Series            # smoothed anomaly score
    thr: float
    alarm: np.ndarray           # latched anomaly flag
    episode_start: np.ndarray   # for each row: time (days) the current alarm episode began, NaN if none
    sim: pd.DataFrame
    hi: pd.Series
    baseline: A.Baseline
    n_train: int
    twin: FanTwin = field(default_factory=FanTwin)

    @property
    def t(self) -> np.ndarray:
        return self.df["t_days"].to_numpy()


def _latch(score: np.ndarray, thr: float):
    """Alarm turns on after SUSTAIN readings above thr; turns off after SUSTAIN readings below 0.6*thr."""
    n = len(score)
    alarm = np.zeros(n, bool)
    start = np.full(n, np.nan)
    on, above, below, ep_start = False, 0, 0, np.nan
    t_idx = np.arange(n)
    for i in range(n):
        above = above + 1 if score[i] > thr else 0
        below = below + 1 if score[i] < 0.6 * thr else 0
        if not on and above >= C.SUSTAIN:
            on, ep_start = True, i - C.SUSTAIN + 1
        elif on and below >= C.SUSTAIN:
            on = False
        alarm[i] = on
        start[i] = ep_start if on else np.nan
    return alarm, start


def run(scenario: str = "healthy", seed: int = C.DEFAULT_SEED, days: int = C.DAYS) -> Result:
    sc = C.SCENARIOS[scenario]
    df = simulate(sc, seed=seed, days=days)
    n_train = C.BASELINE_DAYS * C.SAMPLES_PER_DAY
    feats = A.make_features(df)
    base = A.fit_baseline(df, feats, n_train)
    z = A.z_scores(df, feats, base)
    zs = A.smooth(z)
    score = A.anomaly_score(zs)
    thr = A.learn_threshold(score, n_train)
    alarm, start_idx = _latch(score.to_numpy(), thr)
    t = df["t_days"].to_numpy()
    ep = np.where(np.isnan(start_idx), np.nan, t[np.nan_to_num(start_idx).astype(int)])
    sim = A.similarities(zs)
    hi = A.health_index(zs, sim)
    return Result(sc, df, feats, zs, score, thr, alarm, ep, sim, hi, base, n_train)


def assess(res: Result, idx: int | None = None) -> dict:
    """Everything the dashboard (or a test) needs to know about the asset at reading `idx`."""
    idx = len(res.df) - 1 if idx is None else int(idx)
    flagged = bool(res.alarm[idx])
    hi = float(res.hi.iloc[idx])
    diag = A.diagnose(res.sim.iloc[idx], res.zs.iloc[idx]) if flagged else None
    fault = diag["fault"] if diag else None
    ep = None if np.isnan(res.episode_start[idx]) else float(res.episode_start[idx])
    rul = A.remaining_useful_life(res.t, res.hi.to_numpy(), ep, idx) if flagged else \
        {"days": None, "low": None, "high": None, "model": None, "reason": "No active anomaly"}
    risk = A.risk_level(rul["days"], flagged, hi)
    return {
        "idx": idx, "t_days": float(res.t[idx]), "flagged": flagged, "hi": hi,
        "status": status_from_hi(hi, flagged), "diagnosis": diag, "rul": rul, "risk": risk,
        "action": A.recommend(fault, risk, flagged),
        "twin": res.twin.snapshot(res.df.iloc[idx], hi, fault, flagged),
    }

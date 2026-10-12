"""Detection, diagnosis, health index, trend and recommendation. Rule-based and explainable on purpose.

Pipeline, in plain words:
  1. features   : raw sensors -> the 10 numbers we track (temperatures become "rise over ambient")
  2. baseline   : learn what each feature normally is at each fan speed, from a healthy period
  3. z-scores   : how many "normal noise units" is each feature away from normal right now
  4. anomaly    : combine all features into one smoothed score; flag only if high for 6 h straight
  5. diagnosis  : which known fault's fingerprint does the pattern of deviations look like
  6. health     : turn the score into a 0-100 index relative to the diagnosed fault's failure level
  7. trend      : fit the index's recent decline and extrapolate to the 'Act' line -> days remaining
  8. advice     : (fault, risk) -> recommended action
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


# ------------------------------------------------------------------ 1. features
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    for k in ["vib_rms", "vib_1x", "vib_hf", "ac_db", "ac_hf", "current", "power", "current_ripple"]:
        f[k] = df[k]
    f["temp_bearing"] = df["temp_bearing_c"] - df["ambient"]
    f["temp_winding"] = df["temp_winding_c"] - df["ambient"]
    return f[C.FEATURES]


# ------------------------------------------------------------------ 2. baseline
@dataclass
class Baseline:
    coefs: dict          # feature -> polynomial coefficients in fan speed
    scale: dict          # feature -> robust noise scale (residual MAD * 1.4826, with a floor)

    def expected(self, speed: np.ndarray, feat: str) -> np.ndarray:
        return np.polyval(self.coefs[feat], speed)


def fit_baseline(df: pd.DataFrame, feats: pd.DataFrame, n_train: int) -> Baseline:
    """Learn normal behaviour from the first n_train rows (assumed healthy)."""
    s = df["speed"].to_numpy()[:n_train]
    coefs, scale = {}, {}
    for k in C.FEATURES:
        y = feats[k].to_numpy()[:n_train]
        p = np.polyfit(s, y, C.SPEED_POLY_DEG)
        res = y - np.polyval(p, s)
        mad = np.median(np.abs(res - np.median(res))) * 1.4826
        coefs[k] = p
        scale[k] = max(mad, C.SCALE_FLOOR_FRAC * abs(np.median(y)), 1e-6)
    return Baseline(coefs, scale)


# ------------------------------------------------------------------ 3. z-scores
def z_scores(df: pd.DataFrame, feats: pd.DataFrame, b: Baseline) -> pd.DataFrame:
    s = df["speed"].to_numpy()
    z = {k: np.clip((feats[k].to_numpy() - b.expected(s, k)) / b.scale[k], -C.Z_CLIP, C.Z_CLIP) for k in C.FEATURES}
    return pd.DataFrame(z, index=df.index)


# ------------------------------------------------------------------ 4. anomaly score and flag
def smooth(z: pd.DataFrame) -> pd.DataFrame:
    return z.ewm(span=C.EWMA_SPAN, adjust=False).mean()


def anomaly_score(zs: pd.DataFrame) -> pd.Series:
    """Root-mean-square of the smoothed z-scores: one number for 'how far from normal overall'."""
    return np.sqrt((zs ** 2).mean(axis=1))


def learn_threshold(score: pd.Series, n_train: int) -> float:
    warm = C.EWMA_SPAN                       # ignore the smoother's start-up
    return max(C.THRESHOLD_MARGIN * float(score.iloc[warm:n_train].max()), C.MIN_THRESHOLD)


def sustained_flag(score: pd.Series, thr: float) -> pd.Series:
    """True only when the score has been above thr for SUSTAIN consecutive readings."""
    return (score > thr).astype(int).rolling(C.SUSTAIN).min().fillna(0).astype(bool)


# ------------------------------------------------------------------ 5. diagnosis
def signature_matrix() -> tuple[list[str], np.ndarray]:
    faults = list(C.SIGNATURES)
    m = np.zeros((len(faults), len(C.FEATURES)))
    for i, f in enumerate(faults):
        for k, w in C.SIGNATURES[f].items():
            m[i, C.FEATURES.index(k)] = w
        m[i] /= np.linalg.norm(m[i])
    return faults, m


def similarities(zs: pd.DataFrame) -> pd.DataFrame:
    """Cosine similarity between the observed deviation pattern and each fault's fingerprint (-1..1)."""
    faults, sig = signature_matrix()
    v = zs[C.FEATURES].to_numpy()
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    sim = (v @ sig.T) / np.where(norm < 1e-9, 1, norm)
    return pd.DataFrame(sim, index=zs.index, columns=faults)


def diagnose(sim_row: pd.Series, z_row: pd.Series) -> dict:
    """Pick the best-matching fault, give a confidence and the signals that support it."""
    pos = sim_row.clip(lower=0) ** 2
    top = sim_row.idxmax()
    conf = float(pos[top] / pos.sum()) if pos.sum() > 0 else 0.0
    if sim_row[top] < C.MIN_SIMILARITY:
        return {"fault": None, "title": "Unclassified anomaly", "similarity": float(sim_row[top]),
                "confidence": conf, "ranking": sim_row.sort_values(ascending=False), "evidence": []}
    contrib = {k: float(z_row[k] * np.sign(w)) for k, w in C.SIGNATURES[top].items()}
    ev = [(k, float(z_row[k])) for k in sorted(contrib, key=lambda k: -abs(contrib[k] * abs(C.SIGNATURES[top][k])))]
    return {"fault": top, "title": C.FAULT_TITLES[top], "similarity": float(sim_row[top]), "confidence": conf,
            "ranking": sim_row.sort_values(ascending=False), "evidence": ev}


# ------------------------------------------------------------------ 6. health index
def health_index(zs: pd.DataFrame, sim: pd.DataFrame) -> pd.Series:
    """100 = normal, 0 = the best-matching fault has reached its failure level (config.FAIL_Z).

    Severity of fault f = how far along its fingerprint the smoothed z-scores have moved, as a fraction of
    where they sit at failure: (w . z) / (w . z_fail).
    """
    top = sim.idxmax(axis=1)
    sev = pd.DataFrame(index=zs.index)
    for f, fz in C.FAIL_Z.items():
        w = np.array([C.SIGNATURES[f].get(k, 0.0) for k in C.FEATURES])
        zf = np.array([fz.get(k, 0.0) for k in C.FEATURES])
        sev[f] = (zs[C.FEATURES].to_numpy() @ w) / float(w @ zf)
    cols = list(sev.columns)
    chosen = sev.to_numpy()[np.arange(len(top)), [cols.index(f) for f in top]]
    raw = pd.Series(100 * (1 - np.clip(chosen, 0, 1)), index=zs.index)
    # 24 h trailing mean removes the day/night swing (fault effects scale with fan speed) so the trend is readable
    return raw.rolling(C.SAMPLES_PER_DAY, min_periods=1).mean()


# ------------------------------------------------------------------ 7. trend -> days remaining
def remaining_useful_life(t_days: np.ndarray, hi: np.ndarray, flag_start: float | None, now_idx: int) -> dict:
    """Extrapolate the health decline to the 'Act' line with an accelerating (power-law) curve.

    Wear usually speeds up, so a straight line is too optimistic. We model the loss L = 100 - HI as
    c * (t - t0)^p, where t0 is when the degradation really began (somewhere before the alarm) and p > 1
    means 'speeding up'. We try a grid of (t0, p), keep the best fit, and report the spread of all fits that are
    nearly as good as the range. Returns days=None when there is no usable trend.
    """
    out = {"days": None, "low": None, "high": None, "model": "power-law", "reason": ""}
    t_now = float(t_days[now_idx])
    if flag_start is None or t_now - flag_start < C.RUL_MIN_DAYS:
        out["reason"] = "Not enough flagged history yet"
        return out
    start_idx = int(np.searchsorted(t_days, flag_start))
    idx = np.arange(now_idx, start_idx - 1, -6)[::-1]          # hourly points since the alarm began
    if len(idx) < 24:
        out["reason"] = "Not enough points"
        return out
    t, loss = t_days[idx], 100.0 - hi[idx]
    target = 100.0 - C.HI_ACT
    if loss[-1] >= target:
        return {**out, "days": 0.0, "low": 0.0, "high": 0.0, "model": "reached", "reason": "Act threshold reached"}

    fits = []
    for t0 in np.arange(flag_start - C.RUL_T0_BACK_DAYS, flag_start + 1e-9, 0.5):
        for p in np.arange(C.RUL_P_MIN, C.RUL_P_MAX + 1e-9, 0.25):
            u = np.clip(t - t0, 1e-6, None) ** p
            c = float(loss @ u) / float(u @ u)
            if c <= 0:
                continue
            sse = float(np.sum((loss - c * u) ** 2))
            fits.append((sse, t0 + (target / c) ** (1 / p) - t_now))
    if not fits:
        out["reason"] = "No clear downward trend"
        return out
    fits.sort()
    best = fits[0]
    near = [d for sse, d in fits if sse <= best[0] * C.RUL_RANGE_SSE_FACTOR + 1e-9]
    days = best[1]
    if days > C.RUL_MAX_DAYS:
        out["reason"] = "No clear downward trend"
        return out
    out.update(days=float(max(days, 0)), low=float(max(min(near), 0)),
               high=float(min(max(near), C.RUL_MAX_DAYS)), reason="")
    return out


# ------------------------------------------------------------------ 8. risk and advice
def risk_level(rul_days: float | None, flagged: bool, hi: float) -> str:
    if hi < C.HI_ACT:
        return "High"
    if not flagged:
        return "Low"
    if rul_days is None:
        return "Medium"
    if rul_days < C.RISK_HIGH_DAYS:
        return "High"
    if rul_days <= C.RISK_MEDIUM_DAYS:
        return "Medium"
    return "Low"


def recommend(fault: str | None, risk: str, flagged: bool) -> str:
    if not flagged:
        return C.HEALTHY_ACTION
    return C.ACTIONS[fault or "unknown"][risk]

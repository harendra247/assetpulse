"""Synthetic sensor data for the AHU fan: a healthy baseline plus progressive faults.

Every number here is invented to be plausible, not measured. The point is to have ground truth
(which fault, when it starts, when it fails) so the analytics can be tested against it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DAYS, DEFAULT_SEED, SAMPLE_MIN, SAMPLES_PER_DAY, SCENARIOS, Scenario
from .twin import Spec


def _ar1(rng, n, rho, sigma):
    """Slowly wandering noise (e.g. load changes, weather)."""
    x = np.zeros(n)
    e = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def _lag(x, tau_min):
    """First-order thermal lag: temperatures follow load with a delay."""
    alpha = 1 - np.exp(-SAMPLE_MIN / tau_min)
    return pd.Series(x).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def severity(t_days: np.ndarray, sc: Scenario) -> np.ndarray:
    """Ground-truth fault severity d(t) in [0,1]: 0 before the fault, 1 at functional failure, accelerating."""
    if sc.fault is None:
        return np.zeros_like(t_days)
    x = np.clip((t_days - sc.start_day) / (sc.failure_day - sc.start_day), 0, 1)
    return x ** sc.shape


def simulate(scenario: str | Scenario = "healthy", seed: int = DEFAULT_SEED, days: int = DAYS,
             spec: Spec | None = None) -> pd.DataFrame:
    sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    spec = spec or Spec()
    rng = np.random.default_rng(seed)
    n = days * SAMPLES_PER_DAY
    t = np.arange(n) * SAMPLE_MIN / 1440.0          # days
    hour = (t * 24) % 24

    # ---- operating condition: fan speed (fraction of rated) follows a day/night cycle
    ramp_up = 1 / (1 + np.exp(-(hour - 6.5) * 2.5))
    ramp_dn = 1 / (1 + np.exp((hour - 20.5) * 2.5))
    day_part = ramp_up * ramp_dn
    weekend = ((t.astype(int) % 7) >= 5) * 0.12
    s = 0.55 + (0.38 - weekend) * day_part + _ar1(rng, n, 0.98, 0.025)
    s = np.clip(s, 0.5, 1.0)
    ambient = 22 + 6 * np.sin(2 * np.pi * (hour - 9) / 24) + _ar1(rng, n, 0.9995, 1.8)

    # ---- healthy physics (simple, monotonic in speed)
    vib_1x = 0.6 + 1.2 * s ** 2
    vib_base = 0.3 + 0.4 * s
    vib_hf = 0.05 + 0.04 * s
    ac_db = 52 + 18 * s
    ac_hf = 35 + 6 * s
    power = spec.motor_kw * s ** 3 + 0.4
    current_ripple = np.full(n, 1.5)
    rise_w = _lag(15 + 35 * s ** 3, 30)
    rise_b = _lag(8 + 14 * s ** 2, 60)
    cur_scale = 1 / (1.732 * spec.voltage_v / 1000 * spec.pf)

    # ---- fault effects, scaled by ground-truth severity d(t)
    d = severity(t, sc)
    cur_mult = np.ones(n)
    if sc.fault == "bearing":
        vib_hf = vib_hf * (1 + 12 * d ** 1.5)
        ac_hf = ac_hf + 12 * d
        rise_b = rise_b + 28 * d ** 1.5
    elif sc.fault == "imbalance":
        vib_1x = vib_1x + 3.5 * d * s ** 2
    elif sc.fault == "belt":
        current_ripple = current_ripple * (1 + 10 * d)
        power = power * (1 - 0.10 * d)
        cur_mult = 1 - 0.04 * d
        squeal = rng.random(n) < 0.3                  # intermittent squeal bursts
        ac_hf = ac_hf + squeal * 8 * d
    elif sc.fault == "filter":
        power = power * (1 - 0.22 * d)
        cur_mult = 1 - 0.18 * d
        rise_w = rise_w - 1.5 * d

    # ---- measurement noise
    vib_1x_m = vib_1x * (1 + rng.normal(0, 0.04, n))
    vib_hf_m = vib_hf * (1 + rng.normal(0, 0.08, n))
    vib_rms_m = np.sqrt(vib_1x ** 2 + vib_base ** 2 + (4 * vib_hf) ** 2) * (1 + rng.normal(0, 0.03, n))
    power_m = power * (1 + rng.normal(0, 0.025, n))
    current_m = power * cur_scale * cur_mult * (1 + rng.normal(0, 0.02, n))
    amb_m = ambient + rng.normal(0, 0.2, n)

    df = pd.DataFrame({
        "t_days": t,
        "speed": s,
        "ambient": amb_m,
        "vib_rms": vib_rms_m,
        "vib_1x": vib_1x_m,
        "vib_hf": vib_hf_m,
        "ac_db": ac_db + rng.normal(0, 0.6, n),
        "ac_hf": ac_hf + rng.normal(0, 0.8, n),
        "temp_bearing_c": amb_m + rise_b + rng.normal(0, 0.4, n),
        "temp_winding_c": amb_m + rise_w + rng.normal(0, 0.4, n),
        "current": current_m,
        "power": power_m,
        "current_ripple": current_ripple * (1 + rng.normal(0, 0.06, n)),
        "true_severity": d,                           # ground truth: for validation only, never used by analytics
    })
    df.attrs["scenario"] = sc.name
    return df

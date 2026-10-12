import numpy as np
import pandas as pd
import pytest

from assetpulse import analytics as A
from assetpulse import config as C
from assetpulse import pipeline as P
from assetpulse.simulator import severity, simulate
from assetpulse.twin import FanTwin, Spec, status_from_hi


# ---------------------------------------------------------------- simulator
def test_same_seed_same_data():
    a, b = simulate("bearing", seed=3), simulate("bearing", seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_shape_and_columns():
    df = simulate("healthy")
    assert len(df) == C.DAYS * C.SAMPLES_PER_DAY
    assert {"vib_hf", "power", "current_ripple", "temp_bearing_c", "true_severity"} <= set(df.columns)


def test_severity_ramps_zero_to_one():
    sc = C.SCENARIOS["bearing"]
    t = np.array([0, sc.start_day, (sc.start_day + sc.failure_day) / 2, sc.failure_day, 60.0])
    d = severity(t, sc)
    assert d[0] == d[1] == 0 and d[3] == d[4] == 1 and 0 < d[2] < 0.5      # accelerating: below the straight line at the midpoint


@pytest.mark.parametrize("name,feature,sign", [("bearing", "vib_hf", 1), ("imbalance", "vib_1x", 1),
                                               ("belt", "current_ripple", 1), ("filter", "power", -1)])
def test_fault_moves_its_signature_feature(name, feature, sign):
    h, f = simulate("healthy"), simulate(name)
    late = slice(-144, None)
    assert sign * (f[feature].iloc[late].mean() - h[feature].iloc[late].mean()) > 0


def test_healthy_data_identical_before_fault_starts():
    h, f = simulate("healthy"), simulate("bearing")
    n = int(C.SCENARIOS["bearing"].start_day * C.SAMPLES_PER_DAY)
    assert np.allclose(h["vib_hf"][:n], f["vib_hf"][:n])


# ---------------------------------------------------------------- analytics
def test_signature_matrix_unit_rows():
    faults, m = A.signature_matrix()
    assert faults == list(C.SIGNATURES) and np.allclose(np.linalg.norm(m, axis=1), 1)


def test_baseline_z_is_about_zero_on_healthy_training_data():
    r = P.run("healthy")
    z = r.zs.iloc[C.EWMA_SPAN:r.n_train]
    assert abs(z.mean().mean()) < 0.1 and z.abs().max().max() < 4   # winding temp lags load, so it has the largest residual


def test_healthy_run_raises_no_alarm():
    assert not P.run("healthy").alarm.any()


def test_sustained_flag_ignores_short_spikes():
    s = pd.Series(np.r_[np.zeros(100), np.full(C.SUSTAIN - 1, 5.0), np.zeros(100), np.full(C.SUSTAIN + 5, 5.0)])
    f = A.sustained_flag(s, 1.0)
    assert not f.iloc[:200].any() and f.iloc[-1]


def test_diagnose_returns_unclassified_for_unmatched_pattern():
    z = pd.Series(0.0, index=C.FEATURES)
    z["ac_db"] = 5.0                                          # no fault has this fingerprint
    sim = pd.Series({"bearing": 0.1, "imbalance": 0.0, "belt": 0.1, "filter": 0.0})
    d = A.diagnose(sim, z)
    assert d["fault"] is None and d["title"] == "Unclassified anomaly"


@pytest.mark.parametrize("name", ["bearing", "imbalance", "belt", "filter"])
def test_each_fault_is_flagged_early_and_diagnosed(name):
    r = P.run(name)
    sc = r.scenario
    after = np.flatnonzero(r.alarm & (r.t >= sc.start_day))
    assert len(after) and sc.failure_day - r.t[after[0]] >= 5
    assert not r.alarm[r.t < sc.start_day].any()               # no false alarm before the fault
    a = P.assess(r, int(np.searchsorted(r.t, sc.failure_day)) - 1)
    assert a["diagnosis"]["fault"] == sc.fault and a["status"] in ("Watch", "Act")


def test_health_index_falls_over_time_for_a_fault():
    r = P.run("bearing")
    day = lambda d: r.hi.iloc[int(d * C.SAMPLES_PER_DAY)]
    assert day(25) > 95 > day(45) > day(54)


def test_rul_none_without_enough_history():
    r = P.run("bearing")
    out = A.remaining_useful_life(r.t, r.hi.to_numpy(), flag_start=r.t[-1] - 0.5, now_idx=len(r.t) - 1)
    assert out["days"] is None


def test_rul_range_brackets_the_point_estimate():
    r = P.run("bearing")
    first = int(np.flatnonzero(r.alarm & (r.t >= 30))[0])
    a = P.assess(r, first + 6 * C.SAMPLES_PER_DAY)
    rul = a["rul"]
    assert rul["days"] is not None and rul["low"] <= rul["days"] <= rul["high"]


@pytest.mark.parametrize("rul,flagged,hi,expected", [(20, True, 80, "Low"), (10, True, 60, "Medium"), (2, True, 50, "High"),
                                                     (None, True, 80, "Medium"), (None, False, 100, "Low"), (30, True, 30, "High")])
def test_risk_levels(rul, flagged, hi, expected):
    assert A.risk_level(rul, flagged, hi) == expected


def test_every_fault_and_risk_has_an_action():
    for f in list(C.SIGNATURES) + ["unknown"]:
        for risk in ("Low", "Medium", "High"):
            assert A.recommend(f, risk, True)
    assert A.recommend(None, "Low", False) == C.HEALTHY_ACTION


# ---------------------------------------------------------------- twin
def test_status_bands():
    assert status_from_hi(90) == "Healthy" and status_from_hi(60) == "Watch" and status_from_hi(30) == "Act"
    assert status_from_hi(90, flagged=True) == "Watch"


def test_rated_current_is_sensible():
    assert 12 < Spec().rated_current_a < 14


def test_twin_snapshot_marks_only_the_affected_component():
    r = P.run("belt")
    a = P.assess(r, len(r.df) - 1)
    rows = {c["Component"]: c for c in a["twin"]["components"]}
    assert rows["Drive belt"]["Health"] < 100 and rows["Air filter"]["Health"] == 100
    assert a["twin"]["operating"]["state"] == "Running"


def test_assess_on_healthy_gives_no_action_needed():
    a = P.assess(P.run("healthy"))
    assert a["status"] == "Healthy" and a["diagnosis"] is None and a["action"] == C.HEALTHY_ACTION

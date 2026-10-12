"""Every tunable number lives here: scenarios, features, fault signatures, thresholds, actions."""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------- data shape
SAMPLE_MIN = 10                       # one reading every 10 minutes
SAMPLES_PER_DAY = 24 * 60 // SAMPLE_MIN
DAYS = 60
BASELINE_DAYS = 14                    # healthy period used to learn "normal"
DEFAULT_SEED = 7

# ---------------------------------------------------------------- features (what the analytics sees)
FEATURES = ["vib_rms", "vib_1x", "vib_hf", "ac_db", "ac_hf",
            "temp_bearing", "temp_winding", "current", "power", "current_ripple"]
GROUPS = {                            # for plotting and for explaining which signal family moved
    "Vibration": ["vib_rms", "vib_1x", "vib_hf"],
    "Acoustic": ["ac_db", "ac_hf"],
    "Temperature": ["temp_bearing", "temp_winding"],
    "Electrical": ["current", "power", "current_ripple"],
}
LABELS = {
    "vib_rms": "Vibration overall (mm/s)", "vib_1x": "Vibration at 1x shaft speed (mm/s)",
    "vib_hf": "Vibration high-frequency (g)", "ac_db": "Sound level (dB)",
    "ac_hf": "Sound high-frequency band (dB)", "temp_bearing": "Bearing temp rise over ambient (C)",
    "temp_winding": "Winding temp rise over ambient (C)", "current": "Motor current (A)",
    "power": "Electrical power (kW)", "current_ripple": "Current ripple (%)",
}

SHORT = {                             # short names and units for tables
    "vib_rms": ("Vibration, overall", "mm/s"), "vib_1x": ("Vibration at 1x speed", "mm/s"),
    "vib_hf": ("Vibration, high-freq", "g"), "ac_db": ("Sound level", "dB"),
    "ac_hf": ("Sound, high-freq band", "dB"), "temp_bearing": ("Bearing temp rise", "C"),
    "temp_winding": ("Winding temp rise", "C"), "current": ("Motor current", "A"),
    "power": ("Electrical power", "kW"), "current_ripple": ("Current ripple", "%"),
}
Z_WATCH, Z_ALARM = 3.0, 10.0          # a signal beyond 3 noise units is flagged Watch, beyond 10 Alarm

# ---------------------------------------------------------------- fault scenarios (simulated ground truth)
@dataclass(frozen=True)
class Scenario:
    name: str
    fault: str | None        # key in SIGNATURES, None = healthy
    start_day: float         # fault begins
    failure_day: float       # severity reaches 1.0 (functional failure)
    shape: float             # >1 = degradation accelerates
    title: str


SCENARIOS = {
    "healthy": Scenario("healthy", None, 0, 0, 1, "Healthy (no fault)"),
    "bearing": Scenario("bearing", "bearing", 30, 56, 2.0, "Bearing wear"),
    "imbalance": Scenario("imbalance", "imbalance", 35, 58, 1.6, "Fan imbalance"),
    "belt": Scenario("belt", "belt", 25, 52, 1.6, "Belt slip / looseness"),
    "filter": Scenario("filter", "filter", 20, 58, 1.3, "Clogged filter"),
}

FAULT_TITLES = {s.fault: s.title for s in SCENARIOS.values() if s.fault}

# ---------------------------------------------------------------- how each fault moves the features (expected direction and weight)
# Used for diagnosis: the observed pattern of deviations is compared against these vectors.
SIGNATURES = {
    "bearing":   {"vib_hf": 1.0, "ac_hf": 0.8, "temp_bearing": 0.7, "vib_rms": 0.6},
    "imbalance": {"vib_1x": 1.0, "vib_rms": 0.7},
    "belt":      {"current_ripple": 1.0, "ac_hf": 0.6, "power": -0.2},
    "filter":    {"power": -1.0, "current": -0.9, "temp_winding": -0.3},
}

# ---------------------------------------------------------------- analytics settings
SPEED_POLY_DEG = 3                    # baseline = feature as a smooth function of fan speed
SCALE_FLOOR_FRAC = 0.01               # never let a noise scale go below 1% of the typical level
Z_CLIP = 1000.0                       # only guards against absurd sensor glitches
EWMA_SPAN = 36                        # 36 x 10 min = 6 h smoothing
SUSTAIN = 36                          # flag only if above threshold for 6 h straight
THRESHOLD_MARGIN = 1.6                # threshold = margin x highest healthy-period score
MIN_THRESHOLD = 0.35

HI_WATCH, HI_ACT = 70.0, 40.0         # health index bands: >=70 healthy, 40-70 watch, <40 act
RISK_HIGH_DAYS, RISK_MEDIUM_DAYS = 3.0, 14.0
MIN_SIMILARITY = 0.55                 # below this the pattern is "unclassified"
RUL_T0_BACK_DAYS = 12.0               # degradation may have begun this long before the alarm
RUL_RANGE_SSE_FACTOR = 3.0            # range = all fits whose error is within 3x the best fit's (about 84% coverage on synthetic runs)
RUL_P_MIN, RUL_P_MAX = 1.0, 3.0       # how strongly the decline accelerates
RUL_MIN_DAYS = 2.0                    # need this much flagged history before trusting a trend
RUL_MAX_DAYS = 90.0

# Smoothed z-score of each signature feature when the fault reaches functional failure (health index = 0).
# In a real plant this comes from alarm limits / standards / past failures. Here it is calibrated on the
# simulator (scripts/calibrate.py) and says nothing about real machines.
FAIL_Z = {
    "bearing": {"vib_hf": 140, "ac_hf": 14, "temp_bearing": 34, "vib_rms": 61},
    "imbalance": {"vib_1x": 42, "vib_rms": 46},
    "belt": {"current_ripple": 161, "ac_hf": 3, "power": -5},
    "filter": {"power": -12, "current": -24, "temp_winding": -2},
}

# ---------------------------------------------------------------- recommendations: (fault, risk) -> action
ACTIONS = {
    "bearing": {
        "Low": "Increase vibration/temperature monitoring; plan bearing inspection at next shutdown.",
        "Medium": "Schedule bearing replacement within 2 weeks; order spare bearing and book a technician.",
        "High": "Replace the fan bearing now; run at reduced speed until done to avoid seizure.",
    },
    "imbalance": {
        "Low": "Check impeller for dust build-up or loose weights at the next visit.",
        "Medium": "Schedule impeller cleaning and dynamic balancing within 2 weeks.",
        "High": "Balance or replace the impeller now; reduce speed to limit vibration and bearing load.",
    },
    "belt": {
        "Low": "Inspect belt tension and wear at the next routine visit.",
        "Medium": "Re-tension or replace the belt within 2 weeks; check pulley alignment.",
        "High": "Replace the belt and check pulleys now; airflow is likely already reduced.",
    },
    "filter": {
        "Low": "Plan a filter change at the next routine visit.",
        "Medium": "Replace the air filter within 2 weeks to restore airflow and efficiency.",
        "High": "Replace the air filter now; the fan is working against heavy restriction.",
    },
    "unknown": {
        "Low": "Keep watching; no clear pattern yet.",
        "Medium": "Send a technician to inspect: the signals are abnormal but do not match a known fault.",
        "High": "Inspect urgently: abnormal and getting worse, cause unclassified.",
    },
}
HEALTHY_ACTION = "No action needed. Continue routine monitoring."

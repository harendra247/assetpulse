"""AssetPulse dashboard. Run:  streamlit run app.py"""
import html

import numpy as np
import pandas as pd
import streamlit as st

from assetpulse import config as C
from assetpulse import pipeline as P
from assetpulse import viz

st.set_page_config(page_title="AssetPulse: AHU fan health", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1500px}
header[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none}
[data-testid="stVerticalBlockBorderWrapper"]{background:#131b2e;border:1px solid #243049;border-radius:10px;padding:0 !important;overflow:hidden}
[data-testid="stVerticalBlockBorderWrapper"] > div{gap:0.4rem}
.ph{background:linear-gradient(90deg,#27324d,#1a2339);padding:9px 16px;margin:0 0 6px 0;display:flex;justify-content:space-between;align-items:center;
    font-weight:600;font-size:0.95rem;color:#e6ecf7;letter-spacing:.2px}
.ph span{font-weight:400;font-size:.8rem;color:#93a1bd}
.pb{padding:0 16px}
.top{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
.top h1{font-size:1.5rem;margin:0;padding:0;font-weight:700;color:#e6ecf7}
.top .sub{color:#93a1bd;font-size:.9rem}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:6px 16px 4px}
.tile{background:#0f1729;border:1px solid #243049;border-radius:8px;padding:12px 14px}
.tile .t{font-size:.78rem;font-weight:700;color:#e6ecf7;text-align:center;margin-bottom:8px}
.tile .cols{display:grid;grid-template-columns:1fr 1px 1fr;gap:10px;align-items:start}
.tile .div{background:#2a3858;height:100%}
.tile .c{text-align:center}
.tile .l{font-size:.74rem;color:#93a1bd}
.tile .arrow{font-size:1.1rem;line-height:1.2}
.tile .v{font-size:1.9rem;font-weight:700;color:#e6ecf7;line-height:1.15}
.tile .v small{font-size:.8rem;font-weight:500;color:#93a1bd}
.tile .tg{font-size:.74rem;color:#93a1bd;margin-top:2px}
.facts{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;padding:4px 16px 6px}
.fact{border-left:3px solid #3987e5;padding:2px 0 2px 10px}
.fact .l{font-size:.74rem;color:#93a1bd}
.fact .v{font-size:1.15rem;font-weight:700;color:#e6ecf7}
.fact .s{font-size:.74rem;color:#93a1bd}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.78rem;font-weight:700;color:#0b1020}
.action{margin:8px 16px 14px;padding:12px 14px;border-radius:8px;background:#0f1729;border:1px solid #243049;border-left:4px solid var(--c);color:#e6ecf7;font-size:.95rem}
.action b{color:var(--c)}
table.sig{width:100%;border-collapse:collapse;font-size:.85rem}
table.sig th{color:#93a1bd;font-weight:600;text-align:right;padding:6px 8px;background:#1a2339;font-size:.76rem}
table.sig th:first-child,table.sig td:first-child{text-align:left}
table.sig td{padding:6px 8px;color:#e6ecf7;border-top:1px solid #1f2a45;text-align:right;white-space:nowrap}
table.sig tr:nth-child(even) td{background:#111a2e}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:baseline}
.legend{width:100%;font-size:.88rem;border-collapse:collapse}
.legend td{padding:5px 4px;color:#e6ecf7;border-top:1px solid #1f2a45}
.legend td:last-child{text-align:right;font-weight:700}
.bar{height:8px;border-radius:4px;background:#1c2640;overflow:hidden;min-width:90px}
.bar i{display:block;height:100%;border-radius:4px}
.muted{color:#93a1bd}
div[data-testid="stTabs"] button{color:#93a1bd}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load(scenario: str, seed: int):
    return P.run(scenario, seed=seed)


def panel_head(title: str, note: str = ""):
    st.markdown(f"<div class='ph'>{html.escape(title)}<span>{html.escape(note)}</span></div>", unsafe_allow_html=True)


STATUS_COLOR = {"Healthy": viz.GOOD, "Watch": viz.WARN, "Act": viz.CRIT}
RISK_COLOR = {"Low": viz.GOOD, "Medium": viz.WARN, "High": viz.CRIT}

# ------------------------------------------------------------------ top bar: what to look at, and when
titles = {k: v.title for k, v in C.SCENARIOS.items()}
st.markdown("<div class='top'><h1>AHU-01 supply fan</h1><span class='sub'>Predictive-maintenance monitor · simulated data</span></div>",
            unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns([1.3, 3.2, 1.2, 0.8], vertical_alignment="bottom")
qp = st.query_params                      # optional deep links, e.g. ?scenario=filter&day=40
_keys = list(titles)
_i = _keys.index(qp["scenario"]) if qp.get("scenario") in _keys else 1
scenario = k1.selectbox("Scenario", _keys, index=_i, format_func=titles.get)
with k4.popover("Seed"):
    seed = st.number_input("Random seed", 1, 9999, C.DEFAULT_SEED, help="Same seed = same data every time")
res = load(scenario, int(seed))
default_day = float(min(C.DAYS, (res.scenario.failure_day - 7) if res.scenario.fault else C.DAYS))
try:
    default_day = min(float(C.DAYS), max(15.0, float(qp.get("day", default_day))))
except ValueError:
    pass
day = k2.slider("Replay: show what the system knew at day", 15.0, float(C.DAYS), default_day, 0.25)
truth = k3.toggle("Show simulated truth", value=True, help="Dotted lines mark the real fault start and failure day from the simulator. The analytics never sees them.")
idx = min(int(np.searchsorted(res.t, day)), len(res.t) - 1)
a = P.assess(res, idx)
diag, rul = a["diagnosis"], a["rul"]
prev = max(0, idx - C.SAMPLES_PER_DAY)
hi_now, hi_prev = a["hi"], float(res.hi.iloc[prev])
sc_now, sc_prev = float(res.score.iloc[idx]), float(res.score.iloc[prev])


def arrow(delta, up_is_good):
    if abs(delta) < 0.02:
        return "<span class='arrow' style='color:#93a1bd'>▬</span>"
    good = (delta > 0) == up_is_good
    return f"<span class='arrow' style='color:{viz.GOOD if good else viz.CRIT}'>{'▲' if delta > 0 else '▼'}</span>"


def tile(title, prev_v, now_v, unit, target, up_is_good, pdelta):
    return (f"<div class='tile'><div class='t'>{title}</div><div class='cols'>"
            f"<div class='c'><div class='l'>24 h ago</div><div class='arrow'>&nbsp;</div><div class='v'>{prev_v}<small> {unit}</small></div></div>"
            f"<div class='div'></div>"
            f"<div class='c'><div class='l'>Now</div>{arrow(pdelta, up_is_good)}<div class='v'>{now_v}<small> {unit}</small></div></div>"
            f"</div><div class='tg' style='text-align:center'>{target}</div></div>")


# ------------------------------------------------------------------ row 1: status + health trend
c_left, c_right = st.columns([5, 6])
with c_left:
    with st.container(border=True):
        panel_head("Asset status", f"day {a['t_days']:.1f}")
        st.markdown(
            "<div class='tiles'>"
            + tile("Health index", f"{hi_prev:.0f}", f"{hi_now:.0f}", "/ 100", "Target: 70 or more", True, hi_now - hi_prev)
            + tile("Anomaly score", f"{sc_prev:.1f}", f"{sc_now:.1f}", "", f"Alarm above {res.thr:.1f}", False, sc_now - sc_prev)
            + "</div>", unsafe_allow_html=True)
        sc = STATUS_COLOR[a["status"]]
        rc = RISK_COLOR[a["risk"] if a["flagged"] else "Low"]
        fault_txt = html.escape(diag["title"]) if diag else "None detected"
        fault_sub = f"{diag['confidence'] * 100:.0f}% match" if diag else "all signals within normal"
        if rul["days"] is None:
            d_txt, d_sub = "n/a", html.escape(rul["reason"])
        else:
            d_txt, d_sub = f"{rul['days']:.0f} days", f"range {rul['low']:.0f} to {rul['high']:.0f}"
        st.markdown(
            "<div class='facts'>"
            f"<div class='fact' style='border-color:{sc}'><div class='l'>Status</div><div class='v'><span class='dot' style='background:{sc}'></span>{a['status']}</div><div class='s'>health bands 70 / 40</div></div>"
            f"<div class='fact'><div class='l'>Likely fault</div><div class='v'>{fault_txt}</div><div class='s'>{fault_sub}</div></div>"
            f"<div class='fact' style='border-color:{rc}'><div class='l'>Maintenance due in</div><div class='v'>{d_txt}</div><div class='s'>{d_sub}</div></div>"
            "</div>", unsafe_allow_html=True)
        risk_txt = a["risk"] if a["flagged"] else "Low"
        st.markdown(f"<div class='action' style='--c:{rc}'><b>Risk: {risk_txt}.</b> {html.escape(a['action'])}</div>",
                    unsafe_allow_html=True)
with c_right:
    with st.container(border=True):
        panel_head("Health trend", "Healthy ≥ 70 · Watch 40-70 · Act < 40")
        st.plotly_chart(viz.health_fig(res, idx, truth, rul if a["flagged"] else None), width="stretch",
                        config={"displayModeBar": False})
        st.markdown("<div class='pb muted' style='font-size:.78rem;padding-bottom:10px'>Red shading = anomaly flagged. "
                    "Orange = predicted time to reach the Act line, with its range.</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ row 2: diagnosis + deviations
d_left, d_right = st.columns([5, 6])
with d_left:
    with st.container(border=True):
        panel_head("Fault diagnosis", "match to known fault patterns")
        g1, g2 = st.columns([1, 1.2])
        with g1:
            st.plotly_chart(viz.diagnosis_fig(diag), width="stretch", config={"displayModeBar": False})
        with g2:
            shares = viz.fault_shares(diag)
            if shares is None:
                st.markdown("<div class='muted' style='padding:30px 10px'>Nothing is flagged, so there is nothing to diagnose. "
                            "The fault match appears here once an anomaly is flagged.</div>", unsafe_allow_html=True)
            else:
                rows = ""
                for f, v in shares.items():
                    top = f == diag["fault"]
                    col = viz.BLUE if top else "#3c4d75"
                    rows += (f"<tr><td><span class='dot' style='background:{col}'></span>{html.escape(C.FAULT_TITLES[f])}</td>"
                             f"<td style='width:90px'><div class='bar'><i style='width:{v * 100:.0f}%;background:{col}'></i></div></td>"
                             f"<td>{v * 100:.0f}%</td></tr>")
                st.markdown(f"<table class='legend' style='margin-top:18px'>{rows}</table>", unsafe_allow_html=True)
                if diag["evidence"]:
                    ev = ", ".join(f"{html.escape(C.SHORT[k][0].lower())} {z:+.0f}" for k, z in diag["evidence"][:3])
                    st.markdown(f"<div class='muted' style='font-size:.8rem;margin-top:8px'>Strongest evidence (noise units): {ev}</div>",
                                unsafe_allow_html=True)
with d_right:
    with st.container(border=True):
        panel_head("Signals furthest from normal", "top 6 of 10, now")
        st.plotly_chart(viz.deviation_fig(res, idx), width="stretch", config={"displayModeBar": False})

# ------------------------------------------------------------------ row 3: signal table + twin
t_left, t_right = st.columns([6, 5])
with t_left:
    with st.container(border=True):
        panel_head("Signal status", "last 6 h average vs normal at the current fan speed")
        body = ""
        for r in viz.deviation_rows(res, idx):
            arrow_s = "" if r["state"] == "Normal" else ("▲" if r["z"] > 0 else "▼")
            body += (f"<tr><td>{html.escape(r['name'])}</td><td class='muted'>{r['unit']}</td>"
                     f"<td>{r['expected']:.3g}</td><td>{r['actual']:.3g}</td><td>{r['z']:+.1f}</td>"
                     f"<td><span class='dot' style='background:{r['color']}'></span>{arrow_s} {r['state']}</td></tr>")
        st.markdown("<div class='pb'><table class='sig'><tr><th>Signal</th><th>Unit</th><th>Normal</th><th>Actual</th>"
                    f"<th>Noise units</th><th style='text-align:left'>State</th></tr>{body}</table></div>"
                    "<div class='pb muted' style='font-size:.78rem;padding:8px 16px 12px'>Normal and Actual are 6 h averages. "
                    "Watch = beyond 3 noise units, Alarm = beyond 10.</div>", unsafe_allow_html=True)
with t_right:
    with st.container(border=True):
        tw = a["twin"]
        o = tw["operating"]
        panel_head("Digital twin", tw["asset"])
        st.markdown(
            "<div class='facts' style='grid-template-columns:repeat(4,1fr)'>"
            f"<div class='fact'><div class='l'>State</div><div class='v'>{o['state']}</div></div>"
            f"<div class='fact'><div class='l'>Speed</div><div class='v'>{o['speed_pct']:.0f}%</div><div class='s'>{o['rpm']} rpm</div></div>"
            f"<div class='fact'><div class='l'>Load</div><div class='v'>{o['load_pct']:.0f}%</div><div class='s'>of {tw['spec'].motor_kw} kW</div></div>"
            f"<div class='fact'><div class='l'>Ambient</div><div class='v'>{o['ambient_c']:.0f} °C</div></div></div>",
            unsafe_allow_html=True)
        rows = ""
        for c in tw["components"]:
            col = STATUS_COLOR[c["Status"]]
            rows += (f"<tr><td><b>{html.escape(c['Component'])}</b><div class='muted' style='font-size:.74rem;white-space:normal'>{html.escape(c['Role'])}</div></td>"
                     f"<td style='width:110px'><div class='bar'><i style='width:{c['Health']}%;background:{col}'></i></div></td>"
                     f"<td>{c['Health']}</td><td style='text-align:left'><span class='dot' style='background:{col}'></span>{c['Status']}</td></tr>")
        st.markdown(f"<div class='pb'><table class='sig'><tr><th>Component</th><th>Health</th><th></th><th style='text-align:left'>Status</th></tr>{rows}</table></div>"
                    f"<div class='pb muted' style='font-size:.78rem;padding:8px 16px 12px'>Rated {tw['spec'].motor_kw} kW, {tw['spec'].poles}-pole motor, "
                    f"{tw['spec'].rated_current_a:.1f} A. Only the diagnosed fault's component loses health.</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ row 4: raw signals
with st.container(border=True):
    panel_head("Sensor signals vs normal", "grey band = normal range for the current fan speed (±3 noise units)")
    tabs = st.tabs(list(C.GROUPS))
    for tab, group in zip(tabs, C.GROUPS):
        with tab:
            st.plotly_chart(viz.signals_fig(res, idx, group, truth), width="stretch", config={"displayModeBar": False})

with st.expander("How this works, and what it is not"):
    st.markdown("""
1. **Simulate** 60 days of vibration, sound, temperature and electrical readings, with a progressive fault in the fault scenarios.
2. **Learn normal** for each signal as a function of fan speed from the first 14 healthy days.
3. **Score** how far every signal is from normal, combine into one smoothed anomaly score, and flag only if it stays high for 6 hours.
4. **Diagnose** by comparing the pattern of deviations to the fingerprint of each known fault.
5. **Track** a 0-100 health index and extrapolate its decline with an accelerating curve to estimate days to maintenance.

This is a proof of concept on synthetic data. It shows the approach and that the pipeline works, not accuracy on a real machine.
""")

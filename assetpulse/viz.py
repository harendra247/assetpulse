"""Plotly figures and small data helpers for the dashboard. Pure functions: result + 'as of' index in, figure out."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config as C
from .pipeline import Result

# ---- dark theme tokens (blue / orange / aqua are the validated categorical slots 1-3 on this surface)
PANEL = "#131b2e"
INK, MUTED, GRID = "#e6ecf7", "#93a1bd", "rgba(147,161,189,0.16)"
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
GOOD, WARN, CRIT = "#2fbf71", "#e0a100", "#e66767"          # reserved status colours, always shown with a text label
SLATE = ["#3c4d75", "#2f3e61", "#25324f"]                   # de-emphasised slices
STEP = 6                                                    # plot hourly points, not every 10 min


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=12), showlegend=False,
                      hoverlabel=dict(bgcolor="#0b1020", font_color=INK, bordercolor="#243049"))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def _alarm_spans(res: Result, idx: int):
    a, t = res.alarm[: idx + 1], res.t[: idx + 1]
    spans, start = [], None
    for i, on in enumerate(a):
        if on and start is None:
            start = t[i]
        if not on and start is not None:
            spans.append((start, t[i])); start = None
    if start is not None:
        spans.append((start, t[-1]))
    return spans


def _decorate(fig, res: Result, idx: int, truth: bool, row=None, col=None, label=False):
    kw = {} if row is None else {"row": row, "col": col}
    for x0, x1 in _alarm_spans(res, idx):
        fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(230,103,103,0.13)", line_width=0, **kw)
    if truth and res.scenario.fault:
        for x, text in ((res.scenario.start_day, "fault starts (truth)"), (res.scenario.failure_day, "failure (truth)")):
            if x <= res.t[idx]:
                fig.add_vline(x=x, line_dash="dot", line_color=MUTED, line_width=1, **kw)
                if label:
                    fig.add_annotation(x=x, y=1.0, yref="paper", text=text, showarrow=False, yanchor="bottom",
                                       font=dict(size=10, color=MUTED))


def health_fig(res: Result, idx: int, truth: bool, rul: dict | None = None) -> go.Figure:
    sl = slice(0, idx + 1, STEP)
    t, hi = res.t[sl], res.hi.to_numpy()[sl]
    fig = go.Figure()
    fig.add_hrect(y0=C.HI_WATCH, y1=100, fillcolor="rgba(47,191,113,0.07)", line_width=0)
    fig.add_hrect(y0=C.HI_ACT, y1=C.HI_WATCH, fillcolor="rgba(224,161,0,0.09)", line_width=0)
    fig.add_hrect(y0=0, y1=C.HI_ACT, fillcolor="rgba(230,103,103,0.09)", line_width=0)
    for y, txt in ((C.HI_WATCH, "Watch below 70"), (C.HI_ACT, "Act below 40")):
        fig.add_hline(y=y, line_dash="dot", line_color=MUTED, line_width=1,
                      annotation_text=txt, annotation_position="bottom right", annotation_font=dict(size=10, color=MUTED))
    fig.add_trace(go.Scatter(x=t, y=hi, mode="lines", line=dict(color=BLUE, width=2.2),
                             hovertemplate="day %{x:.1f}<br>health %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[t[-1]], y=[hi[-1]], mode="markers",
                             marker=dict(color=BLUE, size=10, line=dict(color=PANEL, width=2)), hoverinfo="skip"))
    xmax = C.DAYS
    if rul and rul.get("days") is not None and rul.get("model") == "power-law":
        now = res.t[idx]
        fig.add_vrect(x0=now + rul["low"], x1=now + rul["high"], fillcolor="rgba(217,89,38,0.16)", line_width=0)
        fig.add_vline(x=now + rul["days"], line_dash="dash", line_color=ORANGE, line_width=1.5)
        fig.add_annotation(x=now + rul["days"], y=96, text="predicted Act line", showarrow=False, xanchor="right",
                           xshift=-6, font=dict(size=11, color=ORANGE))
        xmax = max(xmax, now + rul["high"] + 2)
    _decorate(fig, res, idx, truth, label=True)
    fig.update_xaxes(title="Day", range=[0, xmax])
    fig.update_yaxes(title="Health index", range=[0, 104])
    return _base(fig, 330).update_layout(margin=dict(l=8, r=8, t=26, b=8))


def signals_fig(res: Result, idx: int, group: str, truth: bool) -> go.Figure:
    feats = C.GROUPS[group]
    fig = make_subplots(rows=len(feats), cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=[C.LABELS[f] for f in feats])
    starts = np.arange(0, idx + 1, STEP)
    sizes = np.diff(np.r_[starts, idx + 1])
    hourly = lambda a: np.add.reduceat(a[: idx + 1], starts) / sizes
    t = res.t[: idx + 1][::STEP]
    speed = res.df["speed"].to_numpy()
    for r, f in enumerate(feats, 1):
        exp, band = res.baseline.expected(speed, f), 3 * res.baseline.scale[f]
        fig.add_trace(go.Scatter(x=t, y=hourly(exp + band), mode="lines", line=dict(width=0), hoverinfo="skip"), row=r, col=1)
        fig.add_trace(go.Scatter(x=t, y=hourly(exp - band), mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(147,161,189,0.20)", hoverinfo="skip"), row=r, col=1)
        fig.add_trace(go.Scatter(x=t, y=hourly(res.feats[f].to_numpy()), mode="lines", line=dict(color=BLUE, width=1.4),
                                 hovertemplate="day %{x:.1f}<br>%{y:.3g}<extra></extra>"), row=r, col=1)
        _decorate(fig, res, idx, truth, row=r, col=1)
    fig.update_xaxes(title_text="Day", row=len(feats), col=1, range=[0, C.DAYS])
    for a in fig.layout.annotations:
        a.font = dict(size=12, color=INK)
        a.xanchor, a.x = "left", 0
    return _base(fig, 175 * len(feats) + 50)


def fault_shares(diag: dict | None):
    """Match to each known fault as a share that sums to 100% (squared positive similarity)."""
    if not diag:
        return None
    pos = diag["ranking"].clip(lower=0) ** 2
    return (pos / pos.sum()).sort_values(ascending=False) if pos.sum() > 0 else None


def diagnosis_fig(diag: dict | None) -> go.Figure:
    fig = go.Figure()
    shares = fault_shares(diag)
    if shares is None:
        fig.add_trace(go.Pie(values=[1], hole=0.72, marker=dict(colors=["#1c2640"], line=dict(color=PANEL, width=2)),
                             textinfo="none", hoverinfo="skip", sort=False))
        fig.add_annotation(text="No active<br>anomaly", showarrow=False, font=dict(size=14, color=MUTED))
    else:
        order = list(shares.index)
        colors = [BLUE if f == diag["fault"] else SLATE[min(i - 1, 2)] for i, f in enumerate(order)]
        fig.add_trace(go.Pie(labels=[C.FAULT_TITLES[f] for f in order], values=shares.values, hole=0.72, sort=False,
                             direction="clockwise", rotation=0, marker=dict(colors=colors, line=dict(color=PANEL, width=3)),
                             textinfo="none", hovertemplate="%{label}: %{percent}<extra></extra>"))
        top = diag["title"] if diag["fault"] else "Unclassified"
        fig.add_annotation(text=f"<b style='font-size:26px;color:{INK}'>{diag['confidence'] * 100:.0f}%</b><br>"
                                f"<span style='font-size:12px'>{top}</span>", showarrow=False, font=dict(color=MUTED))
    return _base(fig, 250)


def deviation_rows(res: Result, idx: int):
    """The 10 signals now: expected vs actual (last 6 h average), smoothed z-score and a state."""
    lo = max(0, idx - C.EWMA_SPAN + 1)
    speed = res.df["speed"].to_numpy()
    rows = []
    for f in C.FEATURES:
        z = float(res.zs[f].iloc[idx])
        name, unit = C.SHORT[f]
        az = abs(z)
        state, color = ("Normal", GOOD) if az < C.Z_WATCH else (("Watch", WARN) if az < C.Z_ALARM else ("Alarm", CRIT))
        rows.append({"key": f, "name": name, "unit": unit, "z": z, "state": state, "color": color,
                     "expected": float(res.baseline.expected(speed[lo:idx + 1], f).mean()),
                     "actual": float(res.feats[f].to_numpy()[lo:idx + 1].mean())})
    return rows


def deviation_fig(res: Result, idx: int, n: int = 6) -> go.Figure:
    rows = sorted(deviation_rows(res, idx), key=lambda r: -abs(r["z"]))[:n][::-1]
    vals = [max(abs(r["z"]), 0.3) for r in rows]
    fig = go.Figure(go.Bar(
        x=vals, y=[r["name"] for r in rows], orientation="h", base=0,
        marker=dict(color=[BLUE if abs(r["z"]) >= C.Z_WATCH else "#3c4d75" for r in rows], cornerradius=4),
        text=[f"{'▲' if r['z'] > 0 else '▼'} {abs(r['z']):.1f}" for r in rows], textposition="outside",
        textfont=dict(color=INK, size=12), cliponaxis=False,
        hovertemplate="%{y}<br>%{customdata:+.1f} noise units<extra></extra>", customdata=[r["z"] for r in rows]))
    fig.add_vline(x=C.Z_WATCH, line_color=WARN, line_width=1.5, line_dash="dash")
    fig.add_annotation(x=np.log10(C.Z_WATCH), y=1.0, yref="paper", text="normal limit: 3", showarrow=False,
                       yanchor="bottom", font=dict(size=11, color=WARN))
    top = max(vals)
    fig.update_xaxes(type="log", range=[np.log10(0.3), np.log10(max(top * 2.2, 10))], title="Distance from normal (noise units, log scale)",
                     tickvals=[1, 3, 10, 30, 100], ticktext=["1", "3", "10", "30", "100"])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=INK, size=12))
    return _base(fig, 290).update_layout(margin=dict(l=8, r=40, t=22, b=8), bargap=0.45)

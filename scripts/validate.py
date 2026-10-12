"""Runs the five 60-day scenarios and checks them against the PRD success criteria.

Usage:  python scripts/validate.py            (default seed)
        python scripts/validate.py --seeds 10 (repeat over seeds 1..10 and report the pass rate)
Everything here is synthetic data: a pass shows the pipeline works, not that it will work on a real machine.
"""
import argparse
import sys
import time

sys.path.insert(0, ".")
import numpy as np

from assetpulse import config as C
from assetpulse import pipeline as P


def evaluate(name: str, seed: int) -> dict:
    t0 = time.perf_counter()
    r = P.run(name, seed=seed)
    sc, t = r.scenario, r.t
    out = {"scenario": name, "seed": seed}
    post = np.arange(len(t)) >= r.n_train
    edges = np.flatnonzero(np.diff(np.r_[False, r.alarm]) & r.alarm)       # alarm rising edges
    if sc.fault is None:
        out["false_alarms"] = int(np.sum(t[edges] >= C.BASELINE_DAYS))
        out["pass"] = out["false_alarms"] <= 1
        out["checks"] = {"<=1 false alarm in 60 days": out["pass"]}
        out["secs"] = time.perf_counter() - t0
        return out
    # first alarm at/after the fault begins (an earlier edge would be a false alarm)
    before = int(np.sum(t[edges] < sc.start_day))
    after = edges[t[edges] >= sc.start_day]
    first = int(after[0]) if len(after) else None
    fail_idx = int(np.searchsorted(t, sc.failure_day)) - 1
    lead = (sc.failure_day - t[first]) if first is not None else None
    out.update(first_alarm_day=None if first is None else float(t[first]), lead_days=lead, false_alarms=before)

    # diagnosis: top-1 correct at the first alarm (and at the end of the alarm period)
    top_at_alarm = P.assess(r, first)["diagnosis"] if first is not None else None
    diag_ok = bool(top_at_alarm and top_at_alarm["fault"] == sc.fault)
    top_late = P.assess(r, fail_idx)["diagnosis"]
    diag_late_ok = bool(top_late and top_late["fault"] == sc.fault)

    # days-remaining accuracy, judged halfway between the first alarm and the point where the (24 h-smoothed)
    # health index actually crossed the Act line in this run. "True" days remaining = that crossing minus now.
    hi = r.hi.to_numpy()
    cross = np.flatnonzero(hi < C.HI_ACT)
    cross = cross[cross > (first or 0)]
    act_idx = int(cross[0]) if len(cross) else None
    pred = rul_range = true_days = rul_err = model = None
    if first is not None and act_idx is not None:
        hidx = (first + act_idx) // 2
        a = P.assess(r, hidx)
        true_days = t[act_idx] - t[hidx]
        pred, rul_range, model = a["rul"]["days"], (a["rul"]["low"], a["rul"]["high"]), a["rul"]["model"]
        if pred is not None:
            rul_err = abs(pred - true_days) / true_days
            rul_abs = abs(pred - true_days)
    in_range = bool(rul_range and rul_range[0] is not None and rul_range[0] <= true_days <= rul_range[1]) if true_days is not None else None
    out.update(in_range=in_range, true_days_to_act=true_days, pred_days_to_act=pred, rul_err=rul_err,
               rul_range=rul_range, rul_model=model)
    checks = {
        "flagged >=5 days before failure": lead is not None and lead >= 5,
        "no false alarm before fault": before == 0,
        "top-1 diagnosis correct at first alarm": diag_ok,
        "top-1 diagnosis correct near failure": diag_late_ok,
        "days remaining within 30% (or 1 day) at midpoint": rul_err is not None and (rul_err <= 0.30 or rul_abs <= 1.0),
    }
    out["checks"] = checks
    out["pass"] = all(checks.values())
    out["secs"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (1 = default seed only)")
    args = ap.parse_args()
    seeds = [C.DEFAULT_SEED] if args.seeds == 1 else list(range(1, args.seeds + 1))
    allres = [evaluate(n, s) for s in seeds for n in C.SCENARIOS]

    if args.seeds == 1:
        print(f"{'scenario':10s} {'first alarm':>11s} {'lead (d)':>8s} {'diagnosis':>9s} {'pred/true days-to-Act @mid':>28s} {'err':>14s} {'time':>6s}  result")
        for o in allres:
            c = o["checks"]
            if o["scenario"] == "healthy":
                print(f"{'healthy':10s} {'-':>11s} {'-':>8s} {'-':>9s} {'-':>28s} {'-':>5s} {o['secs']:5.1f}s  "
                      f"{'PASS' if o['pass'] else 'FAIL'}  ({o['false_alarms']} false alarms)")
                continue
            d = "ok" if c["top-1 diagnosis correct at first alarm"] and c["top-1 diagnosis correct near failure"] else "WRONG"
            fa = f"{o['first_alarm_day']:.1f}" if o["first_alarm_day"] is not None else "none"
            lead = f"{o['lead_days']:.1f}" if o["lead_days"] is not None else "-"
            pt = f"{o['pred_days_to_act']:.1f}" if o["pred_days_to_act"] is not None else "n/a"
            tt = f"{o['true_days_to_act']:.1f}" if o["true_days_to_act"] is not None else "n/a"
            er = f"{o['rul_err'] * 100:.0f}%" if o["rul_err"] is not None else "n/a"
            er += " in-range" if o.get("in_range") else " out-range"
            print(f"{o['scenario']:10s} {fa:>11s} {lead:>8s} {d:>9s} {pt + ' / ' + tt:>28s} {er:>14s} {o['secs']:5.1f}s  {'PASS' if o['pass'] else 'FAIL'}")
            for k, v in c.items():
                if not v:
                    print(f"           - failed: {k}")
        ok = all(o["pass"] for o in allres)
        print("\nOVERALL:", "ALL CRITERIA MET" if ok else "SOME CRITERIA MISSED", "(synthetic data; seed", seeds[0], ")")
        sys.exit(0 if ok else 1)

    # multi-seed robustness
    print(f"Robustness over seeds {seeds[0]}..{seeds[-1]}")
    crit = {}
    for o in allres:
        for k, v in o["checks"].items():
            crit.setdefault((o["scenario"], k), []).append(bool(v))
    for (sc, k), v in crit.items():
        print(f"  {sc:10s} {k:42s} {sum(v):2d}/{len(v)}")
    rng = [o["in_range"] for o in allres if o.get("in_range") is not None]
    if rng:
        print(f"  true days-to-Act fell inside the predicted range: {sum(rng)}/{len(rng)} (informational, not a pass/fail criterion)")
    errs = [o["rul_err"] for o in allres if o.get("rul_err") is not None]
    if errs:
        print(f"  days-remaining error at halfway: median {np.median(errs)*100:.0f}%, worst {max(errs)*100:.0f}%")


if __name__ == "__main__":
    main()

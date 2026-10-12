import sys; sys.path.insert(0, ".")
import numpy as np
from assetpulse import config as C, pipeline as P
for name in C.SCENARIOS:
    r = P.run(name)
    t = r.t; sc = r.scenario
    first = np.argmax(r.alarm) if r.alarm.any() else None
    healthy_max = r.score.iloc[r.n_train:].max()
    print(f"\n== {name}: thr={r.thr:.3f}  train max={r.score.iloc[36:r.n_train].max():.3f}  post max={healthy_max:.2f}")
    if first is not None:
        print(f"   first alarm day {t[first]:.1f}  (fault start {sc.start_day}, failure {sc.failure_day})")
    if sc.fault:
        i = int(np.searchsorted(t, sc.failure_day)) - 1
        print(f"   score at failure day: {r.score.iloc[i-6:i].mean():.2f}   top fault there: {r.sim.iloc[i].idxmax()}  sims={r.sim.iloc[i].round(2).to_dict()}")
        act = np.argmax(r.hi.to_numpy() < C.HI_ACT) if (r.hi < C.HI_ACT).any() else None
        print(f"   HI<40 at day {t[act]:.1f}" if act is not None else "   HI never <40")
    else:
        print(f"   alarm samples: {int(r.alarm.sum())}")

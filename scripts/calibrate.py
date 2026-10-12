"""Prints the smoothed z-score each fault's signature features reach at the failure point (config.FAIL_Z).

In a real plant these come from alarm limits, standards or past failures. Here they come from the
simulator, so they only show the maths works; they say nothing about real machines.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from assetpulse import config as C, pipeline as P

print("FAIL_Z = {")
for name, sc in C.SCENARIOS.items():
    if not sc.fault:
        continue
    acc = {k: [] for k in C.SIGNATURES[sc.fault]}
    for seed in range(1, 6):
        r = P.run(name, seed=seed)
        i = int(np.searchsorted(r.t, sc.failure_day)) - 1
        for k in acc:
            acc[k].append(r.zs[k].iloc[i - 144:i].mean())          # last day before failure
    body = ", ".join(f'"{k}": {np.mean(v):.0f}' for k, v in acc.items())
    print(f'    "{sc.fault}": {{{body}}},')
print("}")

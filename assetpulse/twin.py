"""A basic digital twin of an AHU supply fan: what it is made of, how it is running, how healthy each part is."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import HI_ACT, HI_WATCH


@dataclass(frozen=True)
class Spec:
    motor_kw: float = 7.5
    poles: int = 4
    rated_rpm: float = 1450.0
    voltage_v: float = 400.0
    pf: float = 0.85
    rated_airflow_m3h: float = 12000.0

    @property
    def rated_current_a(self) -> float:
        return self.motor_kw * 1000 / (1.732 * self.voltage_v * self.pf)


@dataclass(frozen=True)
class Component:
    key: str          # id
    name: str
    role: str
    fault: str | None  # the fault mode that degrades it (None = no modelled fault)


COMPONENTS = [
    Component("filter", "Air filter", "Cleans incoming air; restricts flow as it loads up", "filter"),
    Component("belt", "Drive belt", "Transmits motor torque to the fan shaft", "belt"),
    Component("bearing", "Fan bearings", "Support the shaft; wear raises vibration and heat", "bearing"),
    Component("impeller", "Fan impeller", "Moves the air; dust or damage unbalances it", "imbalance"),
    Component("motor", "Induction motor + VFD", "Drives the fan at the requested speed", None),
]


def status_from_hi(hi: float, flagged: bool = False) -> str:
    if hi < HI_ACT:
        return "Act"
    if hi < HI_WATCH or flagged:
        return "Watch"
    return "Healthy"


@dataclass
class FanTwin:
    """Holds the static description; `snapshot` combines it with the latest sensor row and analytics result."""
    spec: Spec = field(default_factory=Spec)
    asset_id: str = "AHU-01 supply fan"

    def operating_state(self, row) -> dict:
        speed = float(row["speed"])
        running = speed > 0.05
        return {
            "state": "Running" if running else "Stopped",
            "speed_pct": round(speed * 100, 1),
            "rpm": round(speed * self.spec.rated_rpm),
            "load_pct": round(float(row["power"]) / self.spec.motor_kw * 100, 1),
            "ambient_c": round(float(row["ambient"]), 1),
        }

    def component_table(self, hi: float, fault: str | None, flagged: bool) -> list[dict]:
        """The diagnosed fault's component carries the health index; others stay healthy.
        This is a deliberate simplification: one fault at a time."""
        rows = []
        for c in COMPONENTS:
            affected = flagged and fault is not None and c.fault == fault
            h = hi if affected else 100.0
            rows.append({"Component": c.name, "Role": c.role, "Health": round(h),
                         "Status": status_from_hi(h, flagged=affected)})
        return rows

    def snapshot(self, row, hi: float, fault: str | None, flagged: bool) -> dict:
        return {"asset": self.asset_id, "spec": self.spec, "operating": self.operating_state(row),
                "components": self.component_table(hi, fault, flagged)}

"""Core data types. Symptom names are the group names from cc_symptom_mapping.csv."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum


class Tier(IntEnum):
    SELF_CARE = 0
    SEE_DOCTOR_SOON = 1
    URGENT = 2        # needs care today
    EMERGENCY = 3     # call emergency services now


@dataclass
class Vitals:
    heart_rate: float | None = None
    systolic_bp: float | None = None
    resp_rate: float | None = None
    spo2: float | None = None
    temp_c: float | None = None


@dataclass
class Intake:
    age: float
    sex: str = "unknown"
    symptoms: set[str] = field(default_factory=set)    # symptom group names
    modifiers: set[str] = field(default_factory=set)   # e.g. sudden_onset, at_rest, stiff_neck
    severity: int | None = None                         # 0-10
    duration_hours: float | None = None
    pregnant: bool = False
    immunocompromised: bool = False
    on_anticoagulants: bool = False
    vitals: Vitals = field(default_factory=Vitals)
    free_text: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Intake":
        d = dict(d)
        vitals = Vitals(**(d.pop("vitals", None) or {}))
        d["symptoms"] = set(d.get("symptoms", []))
        d["modifiers"] = set(d.get("modifiers", []))
        return cls(vitals=vitals, **d)

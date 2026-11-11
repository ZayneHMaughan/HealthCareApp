"""Wrapper around the saved classifier bundle (see risk_model/ layout in the README)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .schema import Intake, Tier


class RiskEngine:
    def __init__(self, path: str | Path = "models/risk_model"):
        import joblib  # only load bundles you trained yourself: joblib can execute code on load
        p = Path(path)
        self.model = joblib.load(p / "model.joblib")
        self.schema = json.loads((p / "schema.json").read_text())
        self.th = json.loads((p / "thresholds.json").read_text())
        self.meta = json.loads((p / "metadata.json").read_text())

    def features(self, i: Intake) -> pd.DataFrame:
        groups = set(self.schema.get("symptom_groups", []))
        v = i.vitals
        known = {"age": i.age, "gender": i.sex,
                 "triage_vital_hr": v.heart_rate, "triage_vital_sbp": v.systolic_bp,
                 "triage_vital_rr": v.resp_rate, "triage_vital_o2": v.spo2, "triage_vital_temp": v.temp_c}
        row = {}
        for f in self.schema["features"]:
            if f in groups:
                row[f] = int(f in i.symptoms)
            else:
                row[f] = known.get(f, np.nan)
        return pd.DataFrame([row])[self.schema["features"]]

    def predict(self, i: Intake) -> tuple[float, Tier]:
        p = float(self.model.predict_proba(self.features(i))[0, 1])
        if p <= self.th["self_care_max"]:
            return p, Tier.SELF_CARE
        if p <= self.th["see_doctor_max"]:
            return p, Tier.SEE_DOCTOR_SOON
        return p, Tier.URGENT   # the model alone never returns EMERGENCY; only rules do

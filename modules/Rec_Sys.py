import json, joblib
import numpy as np
import pandas as pd
from pathlib import Path

TIERS = ["self_care", "see_doctor_soon", "urgent"]

class RiskEngine:
    def __init__(self, path="risk_model"):
        p = Path(path)
        self.model = joblib.load(p / "model.joblib")
        self.schema = json.loads((p / "schema.json").read_text())
        self.th = json.loads((p / "thresholds.json").read_text())
        self.meta = json.loads((p / "metadata.json").read_text())

    def predict(self, intake: dict) -> dict:
        row = {f: intake.get(f, np.nan) for f in self.schema["features"]}
        x = pd.DataFrame([row])[self.schema["features"]]   # enforce column order
        p = float(self.model.predict_proba(x)[0, 1])
        if p <= self.th["self_care_max"]:
            tier = "self_care"
        elif p <= self.th["see_doctor_max"]:
            tier = "see_doctor_soon"
        else:
            tier = "urgent"
        return {"risk_score": round(p, 2), "tier": tier,
                "model_version": self.meta.get("trained_on")}
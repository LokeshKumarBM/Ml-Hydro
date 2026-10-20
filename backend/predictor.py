"""Model loading, input checking and prediction, independent of FastAPI.

Keeping this separate from main.py means the logic can be unit-tested
without starting a web server.
"""

import json
import math
from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "ml" / "models" / "fatigue_ridge_model.joblib"
METADATA_PATH = PROJECT_DIR / "ml" / "models" / "fatigue_model_metadata.json"

WARNING_TEXT = (
    "Research estimate only. This result is not a material compatibility "
    "decision, equipment certification, or safety approval."
)


class InvalidConditionError(ValueError):
    """The request asks for a condition the model was never trained on."""


class FatiguePredictor:
    def __init__(self, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
        self.model = joblib.load(model_path)
        self.metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        self.conditions = {
            (c["material"], c["environment"]): c
            for c in self.metadata["tested_conditions"]
        }

    def options(self):
        """What the dashboard needs to build its dropdowns."""
        return {
            "model": self.metadata.get("model"),
            "training_records": self.metadata.get("training_records"),
            "validation": self.metadata.get("validation"),
            "tested_conditions": self.metadata["tested_conditions"],
        }

    def check(self, request):
        """Reject unknown categories; return notes for extrapolated numbers.

        The one-hot encoder silently ignores unseen categories, which would
        produce a confident-looking number that ignores the input. So unknown
        material/environment/orientation/method are hard errors here.
        """
        key = (request["material"], request["environment"])
        condition = self.conditions.get(key)
        if condition is None:
            valid = [f"{m} in {e}" for m, e in self.conditions]
            raise InvalidConditionError(
                f"No training data for material '{key[0]}' in environment "
                f"'{key[1]}'. Tested combinations: {', '.join(valid)}."
            )

        for field, allowed_key in (
            ("orientation", "orientations"),
            ("test_method", "test_methods"),
        ):
            if request[field] not in condition[allowed_key]:
                raise InvalidConditionError(
                    f"{field} '{request[field]}' was not tested for "
                    f"{key[0]} in {key[1]}. Allowed: {', '.join(condition[allowed_key])}."
                )

        notes = []
        if not math.isclose(request["pressure_mpa"], condition["pressure_mpa"], abs_tol=1e-6):
            notes.append(
                f"Pressure {request['pressure_mpa']:g} MPa differs from the tested "
                f"{condition['pressure_mpa']:g} MPa for this environment."
            )
        low, high = condition["delta_k_range"]
        dk = request["delta_k_mpa_sqrt_m"]
        if not low <= dk <= high:
            notes.append(
                f"Delta K {dk:g} is outside the tested range {low:g}-{high:g} "
                f"MPa*sqrt(m) for this condition; the estimate is an extrapolation."
            )
        if request["r_ratio"] not in condition["r_ratios"]:
            notes.append(
                f"R ratio {request['r_ratio']:g} was not tested for this condition "
                f"(tested: {', '.join(f'{v:g}' for v in condition['r_ratios'])})."
            )
        if request["frequency_hz"] not in condition["frequencies_hz"]:
            notes.append(
                f"Frequency {request['frequency_hz']:g} Hz was not tested for this "
                f"condition (tested: {', '.join(f'{v:g}' for v in condition['frequencies_hz'])})."
            )
        return notes

    def predict(self, request):
        notes = self.check(request)
        row = pd.DataFrame([request])
        log_rate = float(self.model.predict(row)[0])
        rate = 10 ** log_rate
        if not (math.isfinite(rate) and rate > 0):
            raise ArithmeticError("Model returned a non-finite prediction.")

        return {
            "material": request["material"],
            "environment": request["environment"],
            "predicted_da_dn_m_per_cycle": rate,
            "predicted_log10_da_dn": round(log_rate, 4),
            "unit": "m/cycle",
            "model": self.metadata.get("model", "Ridge regression"),
            "status": "extrapolated_estimate" if notes else "experimental_estimate",
            "notes": notes,
            "warning": WARNING_TEXT,
        }

"""Command-line example: one prediction from the saved model.

Run from the project root:
    python -m ml.predict_fatigue
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.predictor import FatiguePredictor  # noqa: E402


def main():
    predictor = FatiguePredictor()
    result = predictor.predict({
        "material": "7475",
        "environment": "H2: 103 / RT",
        "pressure_mpa": 103.0,
        "delta_k_mpa_sqrt_m": 8.0,
        "r_ratio": 0.3,
        "frequency_hz": 1.0,
        "orientation": "SL/ST",
        "test_method": "ASTM E647",
    })

    print("EXPERIMENTAL CRACK-GROWTH PREDICTION")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

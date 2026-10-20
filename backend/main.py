"""MLHydro API: fatigue crack-growth predictions and a simulated sensor feed.

Run from the project root:
    uvicorn backend.main:app --reload
Then open http://127.0.0.1:8000/ for the dashboard, or /docs for the API.
"""

import os
import random
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.predictor import (
    PROJECT_DIR,
    FatiguePredictor,
    InvalidConditionError,
)

DASHBOARD_FILE = PROJECT_DIR / "dashboard.html"

app = FastAPI(
    title="MLHydro API",
    description="Hydrogen sensor monitoring and experimental fatigue crack-growth prediction",
    version="1.2.0",
)

# Comma-separated list of allowed origins, e.g. your GitHub Pages URL.
# Defaults to "*" for local development.
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

try:
    predictor = FatiguePredictor()
except Exception as exc:  # keep the API up so /health can report the problem
    print(f"Warning: could not load ML model: {exc}")
    predictor = None


class FatiguePredictionRequest(BaseModel):
    material: str = Field(min_length=1, max_length=100, examples=["7475"])
    environment: str = Field(min_length=1, max_length=150, examples=["H2: 103 / RT"])
    pressure_mpa: float = Field(ge=0, le=2000, examples=[103])
    delta_k_mpa_sqrt_m: float = Field(gt=0, le=10000, examples=[8])
    r_ratio: float = Field(ge=-1, le=1, examples=[0.1])
    frequency_hz: float = Field(gt=0, le=100000, examples=[1])
    orientation: str = Field(min_length=1, max_length=100, examples=["SL/ST"])
    test_method: str = Field(min_length=1, max_length=100, examples=["ASTM E647"])


def require_model():
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Fatigue model is not loaded. Check the model file and server logs.",
        )
    return predictor


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(DASHBOARD_FILE)


@app.get("/health")
def health_check():
    return {"api_status": "healthy", "fatigue_model_loaded": predictor is not None}


@app.get("/sensor/simulated")
def simulated_sensor_reading():
    return {
        "source": "simulation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hydrogen_concentration_ppm": round(random.uniform(50, 200), 2),
        "temperature_c": round(random.uniform(25, 35), 2),
        "is_simulated": True,
    }


@app.get("/model/options")
def model_options():
    """Tested material/environment combinations, for building the input form."""
    return require_model().options()


@app.post("/predict/fatigue")
def predict_fatigue(request: FatiguePredictionRequest):
    model = require_model()
    try:
        return model.predict(request.model_dump())
    except InvalidConditionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

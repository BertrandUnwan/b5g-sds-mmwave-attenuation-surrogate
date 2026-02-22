"""FastAPI wrapper for the b5g-sds-atten-surrogate package.

Run (from project root):
    uvicorn api.main:app --reload

Env vars (optional):
    SDS_FREQ_GHZ=28|38
    SDS_ARTIFACTS_DIR=./models/mlp_streaming
    SDS_CHECKPOINT_TAG=2021-12
    SDS_DEVICE=cpu|mps|cuda
    SDS_MODEL_DIR=/absolute/or/relative/path   (optional override)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sds_atten_surrogate import Surrogate, SurrogateConfig
from sds_atten_surrogate.features import FEATURE_VARS


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    v = _env_str(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got: {v!r}") from e


def _validate_freq(freq_ghz: int) -> int:
    if int(freq_ghz) not in (28, 38):
        raise HTTPException(status_code=422, detail="freq_ghz must be 28 or 38")
    return int(freq_ghz)


def _build_config(freq_ghz: int) -> SurrogateConfig:
    freq_ghz = _validate_freq(freq_ghz)
    artifacts_dir = _env_str("SDS_ARTIFACTS_DIR", "models/mlp_streaming")
    checkpoint_tag = _env_str("SDS_CHECKPOINT_TAG", "2021-12")
    device = _env_str("SDS_DEVICE", None)
    model_dir = _env_str("SDS_MODEL_DIR", None)

    # Note: model_dir is optional override; keep it None unless user sets it.
    return SurrogateConfig(
        freq_ghz=freq_ghz,
        artifacts_dir=artifacts_dir,  # SurrogateConfig accepts str or Path (your code supports both)
        checkpoint_tag=checkpoint_tag,
        device=device,
        model_dir=model_dir,
    )


class PredictOneRequest(BaseModel):
    # A single row of features. Extra keys are allowed/ignored by the package.
    features: Dict[str, float] = Field(
        ..., description=f"Feature dict with required keys: {', '.join(FEATURE_VARS)}"
    )


class PredictManyRequest(BaseModel):
    rows: List[Dict[str, float]] = Field(
        ..., description="List of feature dict rows (each must contain all required features)."
    )


class PredictResponse(BaseModel):
    freq_ghz: int
    target: str
    predictions: List[float]


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[int]
    feature_count: int


app = FastAPI(
    title="B5G SDS Attenuation Surrogate API",
    version="0.1.0",
    description="Thin FastAPI wrapper around the b5g-sds-atten-surrogate Python package.",
)


# We keep one Surrogate per frequency in memory.
app.state.surrogates: Dict[int, Surrogate] = {}


@app.on_event("startup")
def _startup() -> None:
    """Load models once at process start.

    Default behavior: load BOTH 28 and 38 so the API can serve either.
    If you want only one, set SDS_FREQ_GHZ to 28 or 38.
    """

    freq_env = _env_str("SDS_FREQ_GHZ", None)
    freqs = []
    if freq_env in ("28", "38"):
        freqs = [int(freq_env)]
    else:
        freqs = [28, 38]

    loaded: Dict[int, Surrogate] = {}
    for f in freqs:
        cfg = _build_config(f)
        loaded[f] = Surrogate(cfg)

    app.state.surrogates = loaded


def _get_surrogate(freq_ghz: int) -> Surrogate:
    s = app.state.surrogates.get(int(freq_ghz))
    if s is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model for freq_ghz={freq_ghz} is not loaded. Loaded: {sorted(app.state.surrogates.keys())}",
        )
    return s


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=sorted(app.state.surrogates.keys()),
        feature_count=len(FEATURE_VARS),
    )


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    """Expose basic info for clients (features + target names)."""
    return {
        "features": list(FEATURE_VARS),
        "targets": {
            "28": "ATTEN_28_DB_PER_KM",
            "38": "ATTEN_38_DB_PER_KM",
        },
        "models_loaded": sorted(app.state.surrogates.keys()),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictOneRequest,
    freq_ghz: int = 38,
) -> PredictResponse:
    """Predict one row.

    Query param `freq_ghz` chooses the model (28 or 38).
    """
    freq_ghz = _validate_freq(freq_ghz)
    s = _get_surrogate(freq_ghz)

    try:
        y = s.predict_one(payload.features)
    except KeyError as e:
        # Missing required feature(s)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e

    return PredictResponse(
        freq_ghz=freq_ghz,
        target=f"ATTEN_{freq_ghz}_DB_PER_KM",
        predictions=[float(y)],
    )


@app.post("/predict_many", response_model=PredictResponse)
def predict_many(
    payload: PredictManyRequest,
    freq_ghz: int = 38,
) -> PredictResponse:
    """Predict many rows (list-of-dicts)."""
    freq_ghz = _validate_freq(freq_ghz)
    s = _get_surrogate(freq_ghz)

    try:
        y = s.predict_many(payload.rows)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e

    return PredictResponse(
        freq_ghz=freq_ghz,
        target=f"ATTEN_{freq_ghz}_DB_PER_KM",
        predictions=[float(v) for v in y],
    )
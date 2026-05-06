"""FastAPI inference service for EPA.

Run::

    uvicorn src.service.app:app --reload --port 8000

Then::

    curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d @example_payload.json

The service loads the trained pipeline lazily on the first request (or at
startup if ``EPA_MODEL_PATH`` is set), so a fresh container can come up
quickly. The exact same feature pipeline used during training drives
inference, ensuring train/serve parity.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field

from src.models.predict import predict_payload
from src.utils.io import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schema – kept liberal: emails are validated as strings (not
# EmailStr) because the EPA dataset routinely contains group aliases that
# look like emails but aren't strictly RFC-compliant.
# ---------------------------------------------------------------------------


class PersonRef(BaseModel):
    name: Optional[str] = ""
    email: str


class PredictRequest(BaseModel):
    sender: Union[PersonRef, str]
    to: List[Union[PersonRef, str]] = Field(default_factory=list)
    cc: List[Union[PersonRef, str]] = Field(default_factory=list)
    subject: str = ""
    body: str = ""
    task: str
    # Constrained to [0, 1] so an out-of-range threshold gets a 422
    # at the schema layer rather than producing nonsense scores.
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class Assignment(BaseModel):
    person: str
    name: str = ""
    email: str = ""
    role: str = ""
    score: float
    assigned: bool


class PredictResponse(BaseModel):
    task: str
    threshold: float
    assignments: List[Assignment]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _resolve_model_path() -> Path:
    env_path = os.environ.get("EPA_MODEL_PATH")
    if env_path:
        return Path(env_path)
    cfg = load_config(os.environ.get("EPA_CONFIG", "config.yaml"))
    return Path(cfg["paths"]["model_path"])


@lru_cache(maxsize=1)
def _load_bundle() -> Dict[str, Any]:
    path = _resolve_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Train one with `python -m src.models.train` "
            f"or set EPA_MODEL_PATH to point at an existing artifact."
        )
    logger.info("Loading EPA model bundle from %s", path)
    return joblib.load(path)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EPA – Email task-to-person assignment",
    version="0.1.0",
    description=(
        "Given an email plus an extracted task sentence, return a probability "
        "and an assignment decision for every candidate person (sender + To + Cc)."
    ),
)


@app.get("/health")
def health() -> Dict[str, Any]:
    """Lightweight liveness probe. Does not load the model."""
    model_path = _resolve_model_path()
    return {
        "status": "ok",
        "model_path": str(model_path),
        "model_present": model_path.exists(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest = Body(...),
    threshold: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Override the decision threshold (0..1).",
    ),
) -> PredictResponse:
    try:
        bundle = _load_bundle()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    payload = request.model_dump()
    chosen_threshold = (
        threshold
        if threshold is not None
        else (request.threshold if request.threshold is not None else None)
    )
    result = predict_payload(payload, bundle, threshold=chosen_threshold)
    return PredictResponse(**result)

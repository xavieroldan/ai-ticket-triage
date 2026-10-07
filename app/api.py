"""REST API exposing the ticket classifier."""

import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import __version__
from app.classifier import DEFAULT_MODEL, TicketClassifier, load_hf_pipeline
from app.config import load_config

MAX_BATCH = 50


class TicketIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000, examples=["I can't log in since the password reset"])


class BatchIn(BaseModel):
    tickets: list[TicketIn] = Field(min_length=1, max_length=MAX_BATCH)


class PredictionOut(BaseModel):
    category: str
    category_score: float
    urgency: str
    urgency_score: float
    needs_review: bool


@lru_cache
def get_classifier() -> TicketClassifier:
    """Single classifier instance per process; overridden in tests."""
    labels_path = os.environ.get("TRIAGE_LABELS")
    config = load_config(labels_path) if labels_path else load_config()
    model = os.environ.get("TRIAGE_MODEL", DEFAULT_MODEL)
    return TicketClassifier(config, lambda: load_hf_pipeline(model))


Classifier = Annotated[TicketClassifier, Depends(get_classifier)]

app = FastAPI(title="AI Ticket Triage", version=__version__)


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/labels")
def labels(classifier: Classifier):
    config = classifier.config
    return {
        "categories": config.categories,
        "urgency": config.urgency,
        "review_threshold": config.review_threshold,
    }


@app.post("/classify", response_model=PredictionOut)
def classify(ticket: TicketIn, classifier: Classifier):
    try:
        return classifier.classify(ticket.text).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/classify/batch", response_model=list[PredictionOut])
def classify_batch(batch: BatchIn, classifier: Classifier):
    try:
        return [classifier.classify(t.text).to_dict() for t in batch.tickets]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

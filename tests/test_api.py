import pytest
from fastapi.testclient import TestClient

from app.api import MAX_BATCH, app, get_classifier
from app.classifier import TicketClassifier
from app.config import load_config
from tests.fakes import FakePipeline


@pytest.fixture
def client():
    fake = FakePipeline({"Invoices": ["invoice"], "Blocks work": ["urgent"]})
    app.dependency_overrides[get_classifier] = lambda: TicketClassifier(load_config(), lambda: fake)
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_labels(client):
    body = client.get("/labels").json()
    assert "billing" in body["categories"]
    assert set(body["urgency"]) == {"high", "medium", "low"}


def test_classify(client):
    response = client.post("/classify", json={"text": "Urgent: invoice is wrong"})
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "billing"
    assert body["urgency"] == "high"
    assert set(body) == {"category", "category_score", "urgency", "urgency_score", "needs_review"}


def test_classify_batch(client):
    response = client.post("/classify/batch", json={"tickets": [{"text": "invoice"}, {"text": "urgent"}]})
    assert response.status_code == 200
    assert len(response.json()) == 2


@pytest.mark.parametrize(
    "payload",
    [{}, {"text": ""}, {"text": "x" * 5001}],
)
def test_classify_validation(client, payload):
    assert client.post("/classify", json=payload).status_code == 422


def test_whitespace_only_ticket_is_rejected(client):
    assert client.post("/classify", json={"text": "   "}).status_code == 422


def test_batch_limit(client):
    tickets = [{"text": "invoice"}] * (MAX_BATCH + 1)
    assert client.post("/classify/batch", json={"tickets": tickets}).status_code == 422

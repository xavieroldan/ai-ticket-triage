import pytest

from app.classifier import TicketClassifier
from app.config import load_config
from tests.fakes import FakePipeline

KEYWORDS = {
    "Invoices": ["invoice", "factura", "refund"],
    "Login": ["login", "password", "contraseña"],
    "slow": ["slow", "timeout", "lento"],
    "Blocks work": ["all users", "customers", "urgent", "todos"],
    "A question": ["question", "pregunta", "when you can"],
}


def make(top_score=0.9):
    fake = FakePipeline(KEYWORDS, top_score)
    return TicketClassifier(load_config(), lambda: fake), fake


def test_classifies_category_and_urgency():
    clf, _ = make()
    pred = clf.classify("The invoice for September is wrong and customers are calling, urgent")
    assert pred.category == "billing"
    assert pred.urgency == "high"
    assert pred.needs_review is False


def test_multilingual_keywords_reach_the_model():
    clf, fake = make()
    pred = clf.classify("No puedo entrar, la contraseña no funciona. Pregunta: ¿cuándo se arregla?")
    assert pred.category == "access"
    assert pred.urgency == "low"
    assert "contraseña" in fake.calls[0]["text"]


def test_low_confidence_flags_review():
    clf, _ = make(top_score=0.3)
    assert clf.classify("Something odd happened").needs_review is True


def test_uses_full_hypotheses_and_identity_template():
    clf, fake = make()
    clf.classify("invoice")
    first = fake.calls[0]
    assert first["template"] == "{}"
    assert all(label.startswith("This support ticket is about:") for label in first["labels"])


def test_model_is_loaded_once_and_lazily():
    loads = []
    fake = FakePipeline(KEYWORDS)

    def factory():
        loads.append(1)
        return fake

    clf = TicketClassifier(load_config(), factory)
    assert loads == []
    clf.classify("invoice")
    clf.classify("login")
    assert loads == [1]


def test_whitespace_is_normalised_and_text_truncated():
    clf, fake = make()
    clf.classify("  invoice \n\n  wrong  " + "x" * 5000)
    sent = fake.calls[0]["text"]
    assert sent.startswith("invoice wrong")
    assert len(sent) <= 2000


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_text_is_rejected(text):
    clf, _ = make()
    with pytest.raises(ValueError):
        clf.classify(text)

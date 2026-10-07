"""Zero-shot ticket classifier built on a Hugging Face NLI pipeline."""

from collections.abc import Callable
from dataclasses import asdict, dataclass

from app.config import LabelConfig

DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
MAX_CHARS = 2000

# A zero-shot pipeline: (text, candidate_labels=[...], hypothesis_template=..., multi_label=...) -> dict
ZeroShotPipeline = Callable[..., dict]


@dataclass(frozen=True)
class Prediction:
    category: str
    category_score: float
    urgency: str
    urgency_score: float
    needs_review: bool

    def to_dict(self) -> dict:
        return asdict(self)


def load_hf_pipeline(model_name: str = DEFAULT_MODEL) -> ZeroShotPipeline:
    """Create the real Hugging Face pipeline. Imported lazily so tests never need torch."""
    from transformers import pipeline

    return pipeline("zero-shot-classification", model=model_name)


class TicketClassifier:
    """Classify ticket text into a category and an urgency level."""

    def __init__(self, config: LabelConfig, pipeline_factory: Callable[[], ZeroShotPipeline]):
        self._config = config
        self._factory = pipeline_factory
        self._pipeline: ZeroShotPipeline | None = None

    @property
    def config(self) -> LabelConfig:
        return self._config

    def _get_pipeline(self) -> ZeroShotPipeline:
        # Load the model on first use: startup stays fast and health checks work immediately
        if self._pipeline is None:
            self._pipeline = self._factory()
        return self._pipeline

    def _best(self, text: str, hypotheses: dict) -> tuple:
        """Return (label_key, score) for the hypothesis the model finds most likely."""
        by_hypothesis = {h: key for key, h in hypotheses.items()}
        result = self._get_pipeline()(
            text,
            candidate_labels=list(by_hypothesis),
            hypothesis_template="{}",
            multi_label=False,
        )
        top_label, top_score = result["labels"][0], float(result["scores"][0])
        return by_hypothesis[top_label], round(top_score, 4)

    def classify(self, text: str) -> Prediction:
        clean = " ".join(str(text or "").split())[:MAX_CHARS]
        if not clean:
            raise ValueError("ticket text is empty")
        category, category_score = self._best(clean, self._config.category_hypotheses())
        urgency, urgency_score = self._best(clean, self._config.urgency_hypotheses())
        threshold = self._config.review_threshold
        return Prediction(
            category=category,
            category_score=category_score,
            urgency=urgency,
            urgency_score=urgency_score,
            needs_review=category_score < threshold or urgency_score < threshold,
        )

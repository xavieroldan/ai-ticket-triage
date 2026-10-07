"""Deterministic stand-in for the Hugging Face zero-shot pipeline."""


class FakePipeline:
    """Scores each candidate label by keyword overlap with the text."""

    def __init__(self, keywords: dict, top_score: float = 0.9):
        # keywords: {hypothesis_substring: [words that should select it]}
        self.keywords = keywords
        self.top_score = top_score
        self.calls = []

    def __call__(self, text, candidate_labels, hypothesis_template, multi_label):
        self.calls.append({"text": text, "labels": list(candidate_labels), "template": hypothesis_template})
        lowered = text.lower()

        def hits(label):
            return sum(
                word in lowered for key, words in self.keywords.items() if key in label for word in words
            )

        ranked = sorted(candidate_labels, key=hits, reverse=True)
        rest = (1 - self.top_score) / max(len(ranked) - 1, 1)
        return {"labels": ranked, "scores": [self.top_score] + [rest] * (len(ranked) - 1)}

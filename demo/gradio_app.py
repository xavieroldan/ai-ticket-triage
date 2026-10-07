"""Interactive demo for Hugging Face Spaces or local use: python -m demo.gradio_app"""

import os

import gradio as gr

from app.classifier import DEFAULT_MODEL, TicketClassifier, load_hf_pipeline
from app.config import load_config

classifier = TicketClassifier(
    load_config(), lambda: load_hf_pipeline(os.environ.get("TRIAGE_MODEL", DEFAULT_MODEL))
)

EXAMPLES = [
    ["The September invoice charges us twice and our customers are already complaining."],
    ["No puedo iniciar sesión desde que cambié la contraseña y tengo que entregar hoy."],
    ["L'aplicació va molt lenta des d'aquest matí, triga més d'un minut a carregar."],
    ["It would be great to export the report to Excel when you have time."],
]


def triage(text: str):
    pred = classifier.classify(text)
    review = "⚠️ Needs human review" if pred.needs_review else "✅ Confident"
    return (
        {pred.category: pred.category_score},
        {pred.urgency: pred.urgency_score},
        review,
    )


demo = gr.Interface(
    fn=triage,
    inputs=gr.Textbox(lines=4, label="Ticket text (English, Spanish, Catalan…)"),
    outputs=[gr.Label(label="Category"), gr.Label(label="Urgency"), gr.Textbox(label="Review")],
    examples=EXAMPLES,
    title="AI Ticket Triage",
    description="Zero-shot classification of support tickets with an open-source multilingual NLI model.",
)

if __name__ == "__main__":
    demo.launch()

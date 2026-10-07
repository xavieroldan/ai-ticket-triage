FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch keeps the image much smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-model.txt ./
RUN pip install --no-cache-dir -r requirements-model.txt

COPY app/ app/
COPY labels.yaml .

# Optionally bake the model into the image: docker build --build-arg PRELOAD_MODEL=1 .
ARG PRELOAD_MODEL=0
RUN if [ "$PRELOAD_MODEL" = "1" ]; then python -c "from app.classifier import load_hf_pipeline; load_hf_pipeline()"; fi

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

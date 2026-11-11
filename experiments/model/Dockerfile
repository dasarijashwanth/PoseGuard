# PoseGuard posture classifier API (Day 6, Render-ready Day 9).
#
# Build from the REPO ROOT (the build context), so the image gets the
# whole experiments/ package plus the trained model artifacts:
#
#     docker build -f experiments/model/Dockerfile -t poseguard-api .
#     docker run -p 8000:8000 poseguard-api
#
# Or with compose (from experiments/model/):
#
#     docker compose up --build
#
# The image already contains the trained models (COPY experiments/
# below includes artifacts/), so it serves immediately. To serve a
# freshly retrained model instead, mount it over the baked-in one:
#
#     docker run -p 8000:8000 \
#       -v ./experiments/model/artifacts:/app/experiments/model/artifacts:ro \
#       poseguard-api
#
# Day 9 changes (Render):
# - The image installs requirements-api.txt, the small runtime-only
#   set (FastAPI, scikit-learn, pandas, numpy, joblib). The old
#   requirements file also pulls torch and mlflow, and the default
#   torch install is the multi-gigabyte CUDA build, which makes a
#   Render build slow and the image huge. The API never imports
#   torch or mlflow, so they do not belong in its image.
# - The start command binds $PORT, which Render sets (10000 by
#   default, not 8000). Locally it falls back to 8000.
# - The MLflow service in docker-compose.yml still needs the full
#   requirements, so it passes REQUIREMENTS as a build arg.
FROM python:3.12-slim

WORKDIR /app

# Requirements first, so the layer is cached when only code changes.
# REQUIREMENTS defaults to the slim API set; pass the full eval
# requirements when building the MLflow image (see docker-compose.yml).
ARG REQUIREMENTS=experiments/model/requirements-api.txt
COPY ${REQUIREMENTS} ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY experiments/ ./experiments/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn experiments.model.api:app --host 0.0.0.0 --port ${PORT:-8000}"]






































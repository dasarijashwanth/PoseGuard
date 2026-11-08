# PoseGuard posture classifier API (Day 6).
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
FROM python:3.12-slim

WORKDIR /app

# Requirements first, so the layer is cached when only code changes.
COPY experiments/eval/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY experiments/ ./experiments/

EXPOSE 8000

CMD ["uvicorn", "experiments.model.api:app", "--host", "0.0.0.0", "--port", "8000"]

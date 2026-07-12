# ==============================================================================
# Dockerfile -- Heart Disease Risk Prediction API
# ==============================================================================
# Assignment Task 6 (Model Containerization). Builds a container that serves
# the trained scikit-learn pipeline via the FastAPI app in api/main.py.
#
# Design choices (and why):
#   1. Multi-stage-free but slim base image (python:3.14-slim) -- balances
#      build speed against image size; -slim avoids the ~1GB `python:3.14`
#      full image while still shipping apt/pip so wheels for scikit-learn/
#      xgboost/mlflow install without needing a full build toolchain (these
#      ship manylinux wheels on PyPI, so no gcc is even required here, but
#      slim keeps a small margin of safety for any package that DOES need
#      to compile). Pinned to 3.14 to match the Pipfile/requirements.txt
#      resolution -- newer numpy/scikit-learn/xgboost pins require 3.11+/3.12+.
#   2. Dependencies are installed in their OWN layer, copied BEFORE the
#      application code -- Docker layer caching means `pip install` only
#      re-runs when requirements.txt changes, not on every code edit. This
#      makes the local development inner-loop (edit code -> rebuild) fast,
#      which matters given the assignment's tight timeline.
#   3. A non-root user runs the final process -- standard container security
#      hardening; the API does not need root privileges to bind to a port
#      >1024 or read its own model file.
#   4. HEALTHCHECK hits the /health endpoint -- lets `docker ps` and
#      Kubernetes readiness/liveness probes detect a broken container
#      (e.g. model failed to load) without needing external tooling.
#   5. The trained model artifact (models/heart_disease_pipeline.joblib) is
#      COPIED INTO the image at build time. This is a deliberate choice for
#      this assignment: the FAQ says "the container should include ...
#      trained model", and baking it in means `docker run` works completely
#      offline with zero extra volume mounts -- simplest possible reviewer
#      experience. (In a larger production system you would instead pull
#      the model from an MLflow Model Registry / artifact store at startup,
#      to decouple model updates from image rebuilds -- noted as a
#      limitation/future-work item in the final report.)
# ==============================================================================

FROM python:3.14-slim

# Prevents Python from writing .pyc files and buffers stdout -- keeps
# container logs flowing to `docker logs` / `kubectl logs` immediately
# instead of being buffered, which matters for the Task 8 monitoring
# requirement (we want live request logs).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- Layer 1: dependencies only (maximizes Docker build-cache hits) --------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Layer 2: application code + trained model artifact ---------------------
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

# Create a non-root user and switch to it (security hardening).
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Kubernetes and `docker ps` both understand HEALTHCHECK; a failing check
# marks the container unhealthy so orchestrators stop routing traffic to it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

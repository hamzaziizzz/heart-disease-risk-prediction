"""
api/main.py
===========
FastAPI model-serving application for the Heart Disease classifier.
Covers Assignment Task 6 (Model Containerization -- the API this Dockerfile
wraps) and Task 8 (Monitoring & Logging -- request logging + Prometheus
metrics are wired up here).

Design overview
----------------
* On startup, loads the single serialized pipeline artifact produced by
  src/train.py (models/heart_disease_pipeline.joblib). This pipeline already
  contains BOTH the preprocessing ColumnTransformer and the fitted
  classifier, so the API code itself contains zero feature-engineering
  logic -- it cannot drift out of sync with what was used at training time.
* Exposes:
    - GET  /              -> basic liveness info
    - GET  /health         -> Kubernetes-style health/readiness probe
    - POST /predict         -> the core inference endpoint required by the
                                assignment (accepts JSON patient record,
                                returns prediction + confidence)
    - GET  /metrics         -> Prometheus scrape endpoint (via
                                prometheus-fastapi-instrumentator)
* Every request to /predict is logged (structured logging via the stdlib
  `logging` module) with the input feature summary, the resulting
  prediction, and the confidence score -- this is the "API request logging"
  required by Task 8 and is what a Grafana/Prometheus dashboard, or even
  `docker logs`/`kubectl logs`, would surface for monitoring.

Why FastAPI over Flask?
    The assignment FAQ marks FastAPI as "preferred". FastAPI additionally
    gives us: (1) automatic request validation via Pydantic (malformed JSON
    is rejected with a clear 422 error before it ever reaches the model),
    (2) automatic OpenAPI/Swagger docs at /docs for manual testing, and
    (3) native async support, useful if the model were ever swapped for a
    remote inference call.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from prometheus_fastapi_instrumentator import Instrumentator

# ---------------------------------------------------------------------------
# Logging configuration.
# Structured, timestamped logs to stdout -- Docker/Kubernetes capture stdout
# automatically, so no file-handler setup is required inside the container.
# This is intentional: containers should be stateless and log to stdout/stderr
# per the 12-factor app methodology, letting the orchestration layer
# (Docker logging driver / kubectl logs / a log aggregator) own persistence.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("heart_disease_api")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "heart_disease_pipeline.joblib"

# Module-level holder for the loaded model. Populated in the lifespan
# startup hook below rather than at import time, so that unit tests can
# monkeypatch/inject a fake model without needing the real artifact on disk.
_model = {"pipeline": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager: runs once at application startup and
    once at shutdown. We use it (rather than the deprecated `@app.on_event`
    decorator) to load the trained model pipeline into memory a single time,
    so that /predict requests don't pay disk I/O cost per-request.

    Why load-at-startup instead of load-per-request?
        Loading a joblib pipeline involves deserializing potentially large
        sklearn objects from disk; doing this on every request would add
        significant latency and is unnecessary since the model is immutable
        for the lifetime of the running container.
    """
    logger.info(f"Loading model pipeline from {MODEL_PATH} ...")
    if not MODEL_PATH.exists():
        # Fail loudly and early rather than crashing on the first request --
        # this matches the assignment's "Pipeline must fail on ... errors
        # and give clear logs" production-readiness requirement.
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run `python src/train.py` before starting the API."
        )
    _model["pipeline"] = joblib.load(MODEL_PATH)
    logger.info("Model pipeline loaded successfully.")
    yield
    logger.info("Shutting down API, clearing model from memory.")
    _model["pipeline"] = None


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description=(
        "Serves a scikit-learn classification pipeline trained on the UCI "
        "Heart Disease dataset. POST patient data to /predict to get a "
        "risk prediction and confidence score."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# --- Prometheus instrumentation (Task 8: Monitoring) ------------------------
# Automatically instruments every route with request-count, latency, and
# in-progress-request metrics, and exposes them at GET /metrics in the
# Prometheus text exposition format. This is what our docker-compose
# Prometheus service (see monitoring/prometheus.yml) scrapes.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    HTTP middleware that logs every incoming request and its latency.

    Why middleware (rather than logging only inside /predict)?
        A middleware guarantees ALL endpoints are logged uniformly
        (including 404s and validation failures), which is closer to what a
        real production monitoring setup needs -- you want to know about
        failed/malformed requests, not just successful predictions.

    Args:
        request (starlette.requests.Request): the incoming HTTP request.
        call_next (Callable): the next handler in the middleware chain.

    Returns:
        starlette.responses.Response: the response produced downstream.
    """
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} "
        f"({duration_ms:.2f} ms)"
    )
    return response


# ---------------------------------------------------------------------------
# Request / response schemas (Pydantic).
# ---------------------------------------------------------------------------
class PatientRecord(BaseModel):
    """
    Pydantic schema for a single patient's input features, mirroring the 13
    raw features consumed by src/features.ALL_FEATURES. Using Pydantic gives
    us automatic type validation and range documentation for free (visible
    in the interactive /docs Swagger UI).

    Field descriptions follow the UCI Heart Disease dataset documentation.
    """
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=260, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1=yes, 0=no)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment (1-3)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-3)")
    thal: int = Field(..., ge=3, le=7, description="3=normal, 6=fixed defect, 7=reversible defect")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
            }
        }
    )


class PredictionResponse(BaseModel):
    """
    Pydantic schema for the /predict response. Contains both the discrete
    prediction (required by the assignment: "return prediction") and the
    underlying probability (required: "return prediction and confidence"),
    plus a human-readable label to make manual testing/demo easier.
    """
    prediction: int = Field(..., description="0 = no heart disease, 1 = heart disease present")
    prediction_label: str = Field(..., description="Human-readable label for `prediction`")
    confidence: float = Field(..., description="Model's probability estimate for the predicted class")
    probability_disease: float = Field(..., description="Raw predicted probability of class 1 (disease)")


@app.get("/", tags=["meta"])
def root():
    """
    Basic liveness/info endpoint. Useful as a fast smoke-test after
    `docker run` -- if this returns 200, the container started successfully
    and FastAPI is serving.
    """
    return {
        "service": "heart-disease-risk-api",
        "status": "running",
        "docs_url": "/docs",
        "predict_endpoint": "/predict",
    }


@app.get("/health", tags=["meta"])
def health():
    """
    Kubernetes-style readiness/liveness probe endpoint. Returns 503 if the
    model failed to load (so a K8s readiness probe would correctly mark the
    pod as NOT ready and stop routing traffic to it), else 200.

    Returns:
        dict: {"status": "ok", "model_loaded": bool}
    """
    model_loaded = _model["pipeline"] is not None
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(patient: PatientRecord):
    """
    Core inference endpoint (Assignment Task 6 requirement).

    Accepts a JSON body matching `PatientRecord`, runs it through the loaded
    preprocessing + classification pipeline, and returns the predicted class
    plus a confidence score.

    Args:
        patient (PatientRecord): validated patient feature record (FastAPI
            has already rejected malformed/out-of-range input before this
            function body runs, thanks to Pydantic validation).

    Returns:
        PredictionResponse: prediction (0/1), human label, confidence, and
            raw disease probability.

    Raises:
        HTTPException(503): if the model has not finished loading.
        HTTPException(500): if inference fails unexpectedly (logged with
            full context for debugging -- satisfies the production
            requirement that the pipeline "must fail ... and give clear
            logs").
    """
    pipeline = _model["pipeline"]
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Convert the validated Pydantic model into a single-row DataFrame with
    # column order matching what the pipeline's ColumnTransformer expects.
    # Using a DataFrame (not a raw list/array) means the ColumnTransformer's
    # column-name-based selectors (see src/features.py) work identically to
    # how they worked during training.
    input_df = pd.DataFrame([patient.model_dump()])

    try:
        pred = int(pipeline.predict(input_df)[0])
        proba = pipeline.predict_proba(input_df)[0]
        probability_disease = float(proba[1])
        confidence = float(proba[pred])
    except Exception as exc:
        logger.error(f"Inference failed for input {patient.model_dump()}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    label = "Heart Disease Present" if pred == 1 else "No Heart Disease"

    logger.info(
        f"Prediction served: input_age={patient.age}, input_sex={patient.sex} "
        f"-> prediction={pred} ({label}), confidence={confidence:.4f}"
    )

    return PredictionResponse(
        prediction=pred,
        prediction_label=label,
        confidence=round(confidence, 4),
        probability_disease=round(probability_disease, 4),
    )

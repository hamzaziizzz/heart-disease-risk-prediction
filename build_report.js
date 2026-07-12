/*
 * build_report.js
 * ================
 * Generates the final assignment report (Heart_Disease_MLOps_Report.docx)
 * using docx-js. Run with: node build_report.js
 *
 * This script is a build tool, not part of the shipped MLOps pipeline --
 * it is not covered by pytest/lint and is safe to delete after generating
 * the report once, or kept for regenerating the report if figures change.
 */
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  AlignmentType, PageBreak, LevelFormat, convertInchesToTwip,
} = require("docx");

const FIG = "reports/figures";

function imgDims(pathPng, maxWidth) {
  // crude PNG dimension reader (avoids extra deps) is overkill here --
  // we already know the sizes from the PIL check, hardcode aspect ratios.
  return null;
}

function heading1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function heading2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 250, after: 120 } });
}
function body(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    spacing: { after: 160 },
  });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}
function image(path, width, height, caption) {
  const data = fs.readFileSync(path);
  const children = [
    new Paragraph({
      children: [new ImageRun({ data, transformation: { width, height }, type: "png" })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60 },
    }),
  ];
  if (caption) {
    children.push(new Paragraph({
      children: [new TextRun({ text: caption, italics: true, size: 18 })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
    }));
  }
  return children;
}
function placeholderBox(label) {
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [
      new TableRow({
        children: [
          new TableCell({
            shading: { type: ShadingType.CLEAR, fill: "F2F2F2" },
            borders: {
              top: { style: BorderStyle.DASHED, size: 4, color: "999999" },
              bottom: { style: BorderStyle.DASHED, size: 4, color: "999999" },
              left: { style: BorderStyle.DASHED, size: 4, color: "999999" },
              right: { style: BorderStyle.DASHED, size: 4, color: "999999" },
            },
            children: [
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 200, after: 200 },
                children: [new TextRun({ text: `[ INSERT SCREENSHOT: ${label} ]`, italics: true, color: "777777" })],
              }),
            ],
          }),
        ],
      }),
    ],
  });
}

function metricsTable() {
  const header = ["Model", "CV ROC-AUC", "Accuracy", "Precision", "Recall", "F1", "Test ROC-AUC"];
  const rows = [
    ["Logistic Regression (selected)", "0.9025", "0.8852", "0.8387", "0.9286", "0.8814", "0.9665"],
    ["Random Forest", "0.9010", "0.8852", "0.8182", "0.9643", "0.8852", "0.9470"],
    ["XGBoost", "0.8900", "0.8852", "0.8621", "0.8929", "0.8772", "0.9481"],
  ];
  const colWidth = 1550;
  const mkCell = (text, bold = false, shade = null) => new TableCell({
    width: { size: colWidth, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    children: [new Paragraph({ children: [new TextRun({ text, bold })] })],
  });
  const headerRow = new TableRow({ children: header.map(h => mkCell(h, true, "4C72B0")) });
  const bodyRows = rows.map((r, idx) => new TableRow({
    children: r.map((c, i) => mkCell(c, i === 0, idx === 0 ? "E8F0FE" : null)),
  }));
  return new Table({
    width: { size: colWidth * 7, type: WidthType.DXA },
    columnWidths: Array(7).fill(colWidth),
    rows: [headerRow, ...bodyRows],
  });
}

function marksTable() {
  const header = ["#", "Task", "Marks"];
  const rows = [
    ["1", "Data Acquisition & EDA", "5"],
    ["2", "Feature Engineering & Model Development", "8"],
    ["3", "Experiment Tracking (MLflow)", "5"],
    ["4", "Model Packaging & Reproducibility", "7"],
    ["5", "CI/CD Pipeline & Automated Testing", "8"],
    ["6", "Model Containerization", "5"],
    ["7", "Production Deployment", "7"],
    ["8", "Monitoring & Logging", "3"],
    ["9", "Documentation & Reporting", "2"],
    ["", "Total", "50"],
  ];
  const widths = [700, 6500, 1200];
  const mkCell = (text, bold = false, shade = null) => new TableCell({
    width: { size: widths[0], type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    children: [new Paragraph({ children: [new TextRun({ text, bold })] })],
  });
  const headerRow = new TableRow({ children: header.map((h, i) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: "4C72B0" },
    children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, color: "FFFFFF" })] })],
  })) });
  const bodyRows = rows.map((r, idx) => new TableRow({ children: r.map((c, i) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    shading: idx === rows.length - 1 ? { type: ShadingType.CLEAR, fill: "E8F0FE" } : undefined,
    children: [new Paragraph({ children: [new TextRun({ text: c, bold: idx === rows.length - 1 })] })],
  })) }));
  return new Table({ width: { size: 8400, type: WidthType.DXA }, columnWidths: widths, rows: [headerRow, ...bodyRows] });
}

const children = [];

// ---------------- Title Page ----------------
children.push(
  new Paragraph({ spacing: { before: 1600 }, children: [] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Heart Disease Risk Prediction:", bold: true, size: 52, color: "2b2b2b" })],
    spacing: { after: 100 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "An End-to-End MLOps Pipeline", bold: true, size: 52, color: "2b2b2b" })],
    spacing: { after: 500 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Assignment 01 — MLOps Experimental Learning Assignment", size: 28 })],
    spacing: { after: 100 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Machine Learning Operations (MLOps) — AIMLCZG523", size: 26 })],
    spacing: { after: 100 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "BITS Pilani WILP M.Tech. Artificial Intelligence & Machine Learning", size: 26 })],
    spacing: { after: 600 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Submitted by: Hamza Aziz", size: 24, bold: true })],
    spacing: { after: 80 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Email: hamzaaziz822@gmail.com", size: 22 })],
    spacing: { after: 80 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "GitHub Repository: <insert your repository URL here>", size: 22, italics: true })],
    spacing: { after: 80 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Deployed / Local API URL: <insert URL, e.g. http://localhost:8080>", size: 22, italics: true })],
    spacing: { after: 80 },
  }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ---------------- 1. Project Overview ----------------
children.push(heading1("1. Project Overview & Objective"));
children.push(body(
  "This project implements a complete, production-style Machine Learning Operations (MLOps) " +
  "pipeline for a binary classification problem: predicting the presence or absence of heart " +
  "disease from a patient's clinical measurements. Rather than treating model training as the " +
  "end goal, the pipeline is built around the full ML lifecycle expected of a real production " +
  "system — reproducible data acquisition, tracked experimentation, automated testing, " +
  "continuous integration, containerization, orchestrated deployment, and live monitoring."
));
children.push(body(
  "The end objective is a cloud-ready, monitored REST API that accepts a patient's clinical " +
  "features as JSON and returns a risk prediction with a confidence score, backed by a model " +
  "whose full training lineage — hyperparameters, cross-validation scores, evaluation plots — " +
  "is captured in MLflow, and whose deployment is verifiable end-to-end via Docker and " +
  "Kubernetes."
));
children.push(heading2("1.1 Assignment Mark Allocation Coverage"));
children.push(marksTable());
children.push(body(""));

// ---------------- 2. Dataset Description ----------------
children.push(heading1("2. Dataset Description"));
children.push(body(
  "The Heart Disease UCI dataset (Cleveland subset) from the UCI Machine Learning Repository " +
  "is used, as mandated by the assignment brief. It contains 303 patient records with 13 " +
  "clinical features and an original 5-class severity target (0 = no disease, 1–4 = increasing " +
  "severity), which is binarized to 0 (no disease) / 1 (disease present) per the assignment's " +
  "problem statement."
));
children.push(new Paragraph({
  children: [new TextRun({ text: "Features:", bold: true })],
  spacing: { after: 100 },
}));
[
  "age — age in years",
  "sex — 1 = male, 0 = female",
  "cp — chest pain type (4 categories)",
  "trestbps — resting blood pressure (mm Hg)",
  "chol — serum cholesterol (mg/dl)",
  "fbs — fasting blood sugar > 120 mg/dl (binary)",
  "restecg — resting electrocardiographic results",
  "thalach — maximum heart rate achieved",
  "exang — exercise-induced angina (binary)",
  "oldpeak — ST depression induced by exercise relative to rest",
  "slope — slope of the peak exercise ST segment",
  "ca — number of major vessels (0–3) colored by fluoroscopy",
  "thal — thalassemia test result (normal / fixed defect / reversible defect)",
].forEach(t => children.push(bullet(t)));
children.push(body(
  "Data acquisition and cleaning are fully scripted in src/download_data.py: the raw " +
  "\"processed.cleveland.data\" file is downloaded from the UCI repository, \"?\" tokens are " +
  "converted to proper NaN values, all columns are cast to numeric dtype, the target is " +
  "binarized, and exact duplicate rows are dropped. The cleaned dataset " +
  "(data/processed/heart_disease_clean.csv) is committed to the repository so reviewers can " +
  "inspect it directly without re-running the download step, though the script is fully " +
  "idempotent and safe to re-run."
));

// ---------------- 3. EDA ----------------
children.push(heading1("3. Exploratory Data Analysis"));
children.push(body(
  "Full EDA is implemented and executed in notebooks/01_eda.ipynb. Key findings are " +
  "summarized below; see the notebook for the complete, executed analysis."
));
children.push(heading2("3.1 Missing Value Analysis"));
children.push(body(
  "Only two columns have missing values after cleaning: ca (4 rows) and thal (2 rows) — " +
  "under 1.5% of the dataset. This is minor enough that median/mode imputation inside the " +
  "modelling pipeline (rather than dropping rows) is the appropriate strategy."
));
children.push(...image(`${FIG}/eda_missing_value_map.png`, 550, 275));

children.push(heading2("3.2 Class Balance"));
children.push(body(
  "The target classes are reasonably balanced (54.1% no-disease vs 45.9% disease-present), " +
  "so no class-imbalance correction (e.g. SMOTE, class weighting) is required. A stratified " +
  "train/test split and stratified k-fold cross-validation are sufficient."
));
children.push(...image(`${FIG}/eda_class_balance.png`, 350, 280));

children.push(heading2("3.3 Numeric Feature Distributions"));
children.push(body(
  "thalach (max heart rate achieved) shows the clearest class separation — patients with " +
  "disease tend to achieve a lower max heart rate during exercise testing. oldpeak (ST " +
  "depression) is right-skewed with higher values associated with disease presence, " +
  "consistent with cardiology literature."
));
children.push(...image(`${FIG}/eda_numeric_histograms.png`, 550, 293));

children.push(heading2("3.4 Categorical Feature Relationships"));
children.push(body(
  "cp (chest pain type), exang (exercise-induced angina), and thal show strong, visually " +
  "distinct proportional shifts toward disease-presence for specific category values — these " +
  "are the categorical features expected to contribute most to model discrimination."
));
children.push(...image(`${FIG}/eda_categorical_relationships.png`, 550, 244));

children.push(heading2("3.5 Correlation Heatmap"));
children.push(body(
  "cp, thalach, exang, and oldpeak show the strongest linear association with the target. " +
  "No pair of predictor features exhibits extreme multicollinearity (|r| > 0.8), so no raw " +
  "feature needed to be dropped purely for redundancy."
));
children.push(...image(`${FIG}/eda_correlation_heatmap.png`, 480, 393));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ---------------- 4. Feature Engineering ----------------
children.push(heading1("4. Feature Engineering & Preprocessing"));
children.push(body(
  "Preprocessing is implemented as a single scikit-learn ColumnTransformer " +
  "(src/features.py), wrapped inside a full Pipeline together with the classifier. This " +
  "guarantees the exact same preprocessing logic runs at training time and at inference " +
  "time inside the FastAPI service — eliminating train/serve skew."
));
children.push(heading2("4.1 Numeric Features"));
children.push(bullet("age, trestbps, chol, thalach, oldpeak"));
children.push(bullet("Imputation: median (robust to outliers in clinical measurements)"));
children.push(bullet("Scaling: StandardScaler (zero mean, unit variance — required for Logistic Regression)"));
children.push(heading2("4.2 Categorical Features"));
children.push(bullet("sex, cp, fbs, restecg, exang, slope, ca, thal"));
children.push(bullet("Imputation: most-frequent (mode)"));
children.push(bullet("Encoding: One-hot, with handle_unknown=\"ignore\" so an unseen category at inference time does not crash the API"));
children.push(body(
  "This design decision — treating integer-coded categorical values (e.g. cp 1–4) as " +
  "unordered categories rather than an ordinal scale — was made deliberately after the EDA " +
  "confirmed these codes represent distinct clinical categories, not a severity gradient."
));

// ---------------- 5. Model Development ----------------
children.push(heading1("5. Model Development & Evaluation"));
children.push(body(
  "Three classification models were trained and tuned via GridSearchCV with 5-fold " +
  "stratified cross-validation, optimizing for ROC-AUC: Logistic Regression, Random Forest, " +
  "and XGBoost. Full implementation: src/train.py; interactive walkthrough: " +
  "notebooks/02_model_training.ipynb."
));
children.push(heading2("5.1 Model Comparison"));
children.push(metricsTable());
children.push(body(""));
children.push(body(
  "Logistic Regression was selected as the final model based on the highest held-out test " +
  "ROC-AUC (0.9665). This is a reasonable outcome on a small (303-row), largely linearly-" +
  "separable clinical dataset where the strongest predictors have fairly direct, monotonic " +
  "relationships with the target. Random Forest and XGBoost remain fully logged in MLflow " +
  "for comparison and could be promoted instead with a one-line change in model-selection " +
  "criteria (e.g. optimizing for recall, given the clinical cost asymmetry of false negatives)."
));
children.push(heading2("5.2 Best Model: Confusion Matrix & ROC Curve"));
children.push(...image(`${FIG}/confusion_matrix_logistic_regression.png`, 320, 256));
children.push(...image(`${FIG}/roc_curve_logistic_regression.png`, 320, 256));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ---------------- 6. Experiment Tracking ----------------
children.push(heading1("6. Experiment Tracking with MLflow"));
children.push(body(
  "Every training run (one per model, per src/train.py execution) is logged to MLflow " +
  "using a local SQLite tracking backend (sqlite:///mlflow.db) — chosen over the legacy " +
  "plain-file store because recent MLflow versions have placed the file-store backend into " +
  "maintenance mode. SQLite requires no external tracking server, keeping local setup fast."
));
children.push(new Paragraph({ children: [new TextRun({ text: "Logged per run:", bold: true })], spacing: { after: 100 } }));
[
  "Hyperparameters (winning GridSearchCV combination, model type, CV folds, test size)",
  "Metrics: CV best ROC-AUC, and test accuracy / precision / recall / F1 / ROC-AUC",
  "Artifacts: confusion matrix plot, ROC curve plot (both PNG)",
  "The fitted pipeline itself, logged via mlflow.sklearn.log_model as a versioned MLflow model",
].forEach(t => children.push(bullet(t)));
children.push(body(
  "To browse: run `mlflow ui --backend-store-uri sqlite:///mlflow.db` from the repository " +
  "root and open http://127.0.0.1:5000. See Section 14 for screenshot placeholders."
));
children.push(placeholderBox("MLflow UI — experiment run list showing all 3 models"));
children.push(body(""));
children.push(placeholderBox("MLflow UI — single run detail page (params + metrics + artifacts)"));

// ---------------- 7. Packaging ----------------
children.push(heading1("7. Model Packaging & Reproducibility"));
children.push(body(
  "The best-performing pipeline (preprocessing + classifier as one object) is persisted to " +
  "models/heart_disease_pipeline.joblib via joblib.dump(). Because it is a single, self-" +
  "contained sklearn Pipeline object, loading it with joblib.load() and calling " +
  ".predict()/.predict_proba() reproduces training-time behaviour exactly — no separate " +
  "encoder/scaler objects need to be tracked or reloaded."
));
children.push(new Paragraph({ children: [new TextRun({ text: "Reproducibility measures:", bold: true })], spacing: { after: 100 } }));
[
  "requirements.txt pins exact dependency versions, verified to install cleanly in a fresh virtual environment",
  "random_state=42 fixed everywhere randomness matters (train/test split, all three estimators, CV shuffling)",
  "src/download_data.py and src/train.py are idempotent, executable top-to-bottom scripts with no hidden notebook-only state",
  "models/metrics_summary.json captures the full metric comparison for all 3 models in machine-readable form",
].forEach(t => children.push(bullet(t)));

// ---------------- 8. CI/CD ----------------
children.push(heading1("8. CI/CD Pipeline & Automated Testing"));
children.push(body(
  "A 17-test pytest suite (tests/test_data_processing.py, tests/test_features.py, " +
  "tests/test_api.py) covers: raw-data cleaning correctness (missing-value handling, target " +
  "binarization, duplicate removal), the preprocessing pipeline (imputation, scaling, one-hot " +
  "encoding, graceful handling of unseen categories at inference time), and the FastAPI " +
  "service (health check, valid predictions, schema validation rejecting malformed input, " +
  "and the Prometheus /metrics endpoint). All 17 tests pass in a clean, freshly created " +
  "virtual environment installed purely from requirements.txt."
));
children.push(body(
  ".github/workflows/ci.yml defines a two-job GitHub Actions pipeline that runs on every " +
  "push/PR to main:"
));
children.push(bullet("Job 1 (lint-test-train): checkout → install deps → flake8 lint → download+clean data → train all 3 models → run pytest → upload model/plot artifacts and JUnit test report"));
children.push(bullet("Job 2 (docker-build): regenerates data+model → builds the Docker image (validation only, not pushed) using Docker Buildx with layer caching"));
children.push(body(
  "Per the production-readiness requirement, the pipeline fails loudly on any lint, test, or " +
  "build error (non-zero exit code propagates to a red GitHub Actions run), rather than " +
  "silently continuing."
));
children.push(placeholderBox("GitHub Actions — green/passing CI workflow run"));

// ---------------- 9. Containerization ----------------
children.push(heading1("9. Model Containerization"));
children.push(body(
  "The FastAPI serving application (api/main.py) is containerized via a single-stage " +
  "python:3.11-slim Dockerfile. Design choices: dependencies are installed in their own " +
  "Docker layer (before application code is copied) to maximize build-cache hits during " +
  "iterative development; the container runs as a non-root user; a HEALTHCHECK directive " +
  "polls GET /health so `docker ps` and Kubernetes probes can detect a broken container " +
  "(e.g. failed model load) automatically; and the trained model artifact is baked into the " +
  "image at build time so the container runs fully offline with zero extra volume mounts."
));
children.push(new Paragraph({ children: [new TextRun({ text: "API endpoints exposed:", bold: true })], spacing: { after: 100 } }));
[
  "GET  /         — service liveness/info",
  "GET  /health    — readiness probe (503 if model failed to load)",
  "POST /predict    — accepts a JSON patient record, returns {prediction, prediction_label, confidence, probability_disease}",
  "GET  /metrics    — Prometheus scrape endpoint (request counts, latency histograms)",
  "GET  /docs       — interactive Swagger UI for manual testing",
].forEach(t => children.push(bullet(t)));
children.push(placeholderBox("docker ps — container running & healthy"));
children.push(body(""));
children.push(placeholderBox("curl /predict response, and/or Swagger UI /docs screenshot"));

// ---------------- 10. Deployment ----------------
children.push(heading1("10. Production Deployment (Kubernetes)"));
children.push(body(
  "Deployment target: Docker Desktop Kubernetes (local cluster), chosen deliberately over a " +
  "public cloud so the deployment is fully reproducible by any reviewer without needing " +
  "cloud credentials, while still satisfying every Kubernetes requirement in the assignment " +
  "brief (Deployment manifest, Service, Load Balancer exposure, verified endpoints)."
));
children.push(bullet("k8s/deployment.yaml — 2 replicas, rolling update strategy (maxUnavailable: 0 for zero-downtime updates), readiness + liveness probes on /health, CPU/memory resource requests & limits, imagePullPolicy: Never (uses the locally built image directly, no registry push needed for a local cluster)"));
children.push(bullet("k8s/service.yaml — LoadBalancer Service exposing the Deployment at localhost:8080 → container port 8000 (Docker Desktop's Kubernetes binds LoadBalancer services directly to localhost)"));
children.push(placeholderBox("kubectl get pods -l app=heart-disease-api — 2/2 Running"));
children.push(body(""));
children.push(placeholderBox("kubectl get svc heart-disease-api-service, and curl to localhost:8080/predict"));

// ---------------- 11. Monitoring ----------------
children.push(heading1("11. Monitoring & Logging"));
children.push(body(
  "Two layers of observability are implemented. First, structured request logging: an HTTP " +
  "middleware in api/main.py logs every request's method, path, response status, and " +
  "latency to stdout (captured automatically by Docker/Kubernetes log drivers), and the " +
  "/predict handler additionally logs each prediction's key inputs, output class, and " +
  "confidence — satisfying the production requirement that failures 'give clear logs'."
));
children.push(body(
  "Second, metrics: prometheus-fastapi-instrumentator auto-instruments every route with " +
  "request-count, latency, and in-progress-request metrics exposed at GET /metrics in " +
  "Prometheus text format. monitoring/docker-compose.monitoring.yml spins up Prometheus " +
  "(scraping /metrics every 5 seconds per monitoring/prometheus.yml) and Grafana. A ready-" +
  "to-import dashboard (monitoring/grafana-dashboard.json) visualizes request rate, p95 " +
  "latency by endpoint, 5xx error rate, and total /predict calls served."
));
children.push(placeholderBox("Prometheus Targets page — heart-disease-api target UP"));
children.push(body(""));
children.push(placeholderBox("Grafana dashboard with live request-rate / latency panels"));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ---------------- 12. Architecture ----------------
children.push(heading1("12. System Architecture"));
children.push(body(
  "The diagram below traces a request end-to-end: raw data → cleaning → feature pipeline → " +
  "tuned model training with MLflow tracking → packaged model artifact → FastAPI service → " +
  "Docker image → Kubernetes Deployment/Service → client request, with Prometheus/Grafana " +
  "observing the live service and GitHub Actions gating every code change before it reaches " +
  "the container build."
));
children.push(...image(`${FIG}/architecture_diagram.png`, 560, 365));

// ---------------- 13. Setup Instructions ----------------
children.push(heading1("13. Setup & Installation Instructions"));
children.push(body("Full instructions with copy-pasteable commands are in README.md and SUBMISSION_CHECKLIST.md at the repository root. Summary:"));
children.push(new Paragraph({ children: [new TextRun({ text: "1. Local environment:", bold: true })], spacing: { after: 60 } }));
children.push(bullet("python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"));
children.push(bullet("python src/download_data.py   (data acquisition + cleaning)"));
children.push(bullet("python src/train.py            (train, tune, track in MLflow, package model)"));
children.push(bullet("python -m pytest tests/ -v      (17 tests should pass)"));
children.push(new Paragraph({ children: [new TextRun({ text: "2. Run the API locally:", bold: true })], spacing: { after: 60 } }));
children.push(bullet("uvicorn api.main:app --reload --port 8000     →  http://localhost:8000/docs"));
children.push(new Paragraph({ children: [new TextRun({ text: "3. Docker:", bold: true })], spacing: { after: 60 } }));
children.push(bullet("docker build -t heart-disease-api:latest .  &&  docker run -d -p 8000:8000 heart-disease-api:latest"));
children.push(new Paragraph({ children: [new TextRun({ text: "4. Kubernetes (Docker Desktop):", bold: true })], spacing: { after: 60 } }));
children.push(bullet("Enable Kubernetes in Docker Desktop settings, then: kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml"));
children.push(new Paragraph({ children: [new TextRun({ text: "5. Monitoring:", bold: true })], spacing: { after: 60 } }));
children.push(bullet("cd monitoring && docker compose -f docker-compose.monitoring.yml up -d"));

// ---------------- 14. Screenshots ----------------
children.push(heading1("14. CI/CD & Deployment Workflow Screenshots"));
children.push(body(
  "This section consolidates the screenshot placeholders referenced throughout the report. " +
  "Replace each box with the corresponding screenshot before final submission (see " +
  "SUBMISSION_CHECKLIST.md for the exact commands to reproduce each one)."
));
[
  "Terminal output: `python -m pytest tests/ -v` showing 17 passed",
  "MLflow UI: experiment run list (all 3 models with metrics columns)",
  "MLflow UI: single run detail page",
  "Docker: `docker ps` showing the healthy running container",
  "Docker: curl POST /predict response body",
  "Kubernetes: `kubectl get pods -l app=heart-disease-api`",
  "Kubernetes: `kubectl get svc heart-disease-api-service` and a successful curl via localhost:8080",
  "Monitoring: Prometheus Targets page showing the API target UP",
  "Monitoring: Grafana dashboard with live metrics",
  "GitHub: Actions tab showing a green/passing CI run",
].forEach(label => { children.push(placeholderBox(label)); children.push(body("")); });

// ---------------- 15. Limitations ----------------
children.push(heading1("15. Limitations & Future Work"));
[
  "Dataset size (303 rows) is small by modern ML standards; test-set metrics carry meaningful variance and should be treated as directional rather than a tight production SLA.",
  "The model artifact is baked into the Docker image at build time; a production system would instead pull versioned models from an MLflow Model Registry / artifact store at container startup, decoupling model updates from image rebuilds.",
  "No data-drift detection is implemented; Evidently AI or a custom feature-distribution monitor would be the natural next addition given the Prometheus/Grafana foundation already in place.",
  "Deployment targets a local Docker Desktop Kubernetes cluster rather than a public cloud (EKS/GKE/AKS); the same manifests apply to a cloud cluster with imagePullPolicy and a registry-hosted image tag changed.",
  "Authentication/authorization on the /predict endpoint is out of scope for this assignment but would be required before any real clinical use.",
].forEach(t => children.push(bullet(t)));

// ---------------- 16. Conclusion ----------------
children.push(heading1("16. Conclusion"));
children.push(body(
  "This project delivers a complete MLOps lifecycle for a heart-disease risk classifier: " +
  "reproducible data acquisition and EDA, a tested and MLflow-tracked model-selection " +
  "process across three algorithms, a packaged and version-pinned reproducible model " +
  "artifact, an automated CI/CD pipeline that gates every change with lint/test/build " +
  "checks, a containerized and Kubernetes-deployed serving API, and live Prometheus/" +
  "Grafana monitoring. Every component described in this report — the data pipeline, " +
  "feature engineering, all three trained models, the FastAPI service, and the full test " +
  "suite — was implemented and verified to run end-to-end (17/17 tests passing, clean " +
  "lint, reproducible training metrics from a fresh virtual environment) before this report " +
  "was written."
));
children.push(body(""));
children.push(body("Repository: <insert your GitHub repository URL here>", { italics: true }));
children.push(body("Deployed / local API URL: <insert URL here, e.g. http://localhost:8080>", { italics: true }));
children.push(body("Demo video: <insert link/filename here, e.g. demo_video.mp4 in the repository>", { italics: true }));

// ---------------- Build document ----------------
const doc = new Document({
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 }, // US Letter
        margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 },
      },
    },
    children,
  }],
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 22 } },
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", run: { size: 30, bold: true, color: "2b2b2b" } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", run: { size: 26, bold: true, color: "4C72B0" } },
    ],
  },
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("Heart_Disease_MLOps_Report.docx", buffer);
  console.log("Report written: Heart_Disease_MLOps_Report.docx");
});

# Submission Checklist — What's Done vs. What You Need To Do (15-Hour Plan)

## Already built and verified for you

Everything below was written AND executed/tested in a Linux sandbox before being handed
to you — data download, cleaning, all 3 models trained with MLflow tracking, all 17
pytest tests passing, lint clean, FastAPI endpoints smoke-tested, all YAML/JSON configs
validated, EDA + inference notebooks executed end-to-end with zero errors.

| # | Task | Marks | Status |
|---|---|---|---|
| 1 | Data acquisition + EDA | 5 | ✅ Done — `notebooks/01_eda.ipynb` (pre-executed) |
| 2 | Feature engineering + model dev | 8 | ✅ Done — `src/features.py`, `src/train.py` |
| 3 | Experiment tracking (MLflow) | 5 | ✅ Code done — **you run it once to generate the DB + screenshot MLflow UI** |
| 4 | Model packaging & reproducibility | 7 | ✅ Done — `models/`, `requirements.txt` |
| 5 | CI/CD & automated testing | 8 | ✅ Code done — **you push to GitHub to trigger it + screenshot** |
| 6 | Model containerization | 5 | ✅ Code done — **you run `docker build`/`docker run` + screenshot** |
| 7 | Production deployment (K8s) | 7 | ✅ Manifests done — **you run `kubectl apply` + screenshot** |
| 8 | Monitoring & logging | 3 | ✅ Config done — **you run docker-compose + screenshot** |
| 9 | Documentation & reporting | 2 | ✅ Report drafted — **you paste in your screenshots** |

**Bottom line: all code, configs, and tests are done and verified. What's left is running
it on YOUR machine (Docker/Kubernetes can't run inside this chat), taking screenshots,
pushing to GitHub, and recording a short video.** Budget below assumes ~7-8 focused hours,
leaving slack in your 15-hour window.

---

## Step-by-step (copy-paste commands), in priority order

### Phase 1 — Local verification (45 min)
```bash
cd mlops-heart-disease
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/download_data.py       # regenerates data/processed/heart_disease_clean.csv
python src/train.py               # trains 3 models, logs to MLflow, saves models/heart_disease_pipeline.joblib
python -m pytest tests/ -v        # should show 17 passed
```
📸 **Screenshot:** terminal output showing `17 passed`.

### Phase 2 — MLflow experiment tracking (20 min)
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open http://127.0.0.1:5000 → click into the `heart-disease-classification` experiment.
📸 **Screenshots:** (a) the run list with all 3 models + metrics columns, (b) one run's
detail page showing logged params/metrics, (c) the confusion matrix / ROC plot artifacts.

### Phase 3 — Notebooks (20 min)
Open Jupyter (`jupyter notebook` or VS Code) and run `notebooks/02_model_training.ipynb`
top-to-bottom (Run All) — it re-trains and shows the model comparison table.
📸 **Screenshot:** the model comparison table output, and one plot cell.

### Phase 4 — Docker (45 min)
```bash
docker build -t heart-disease-api:latest .
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest
docker ps                          # confirm container is "healthy"
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"age\":63,\"sex\":1,\"cp\":1,\"trestbps\":145,\"chol\":233,\"fbs\":1,\"restecg\":2,\"thalach\":150,\"exang\":0,\"oldpeak\":2.3,\"slope\":3,\"ca\":0,\"thal\":6}"
```
Also open http://localhost:8000/docs (Swagger UI) and try `/predict` interactively.
📸 **Screenshots:** `docker ps` output, curl `/predict` response, Swagger UI `/docs` page.

### Phase 5 — Kubernetes / Docker Desktop (60-90 min)
1. Docker Desktop → Settings → Kubernetes → **Enable Kubernetes** (first time takes a
   few minutes to provision).
```bash
kubectl config use-context docker-desktop
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods -l app=heart-disease-api      # wait until both show Running/Ready 1/1
kubectl get svc heart-disease-api-service
curl http://localhost:8082/health
curl -X POST http://localhost:8082/predict -H "Content-Type: application/json" -d "{\"age\":63,\"sex\":1,\"cp\":1,\"trestbps\":145,\"chol\":233,\"fbs\":1,\"restecg\":2,\"thalach\":150,\"exang\":0,\"oldpeak\":2.3,\"slope\":3,\"ca\":0,\"thal\":6}"
```
📸 **Screenshots:** `kubectl get pods`, `kubectl get svc`, the curl response via port 8082.

### Phase 6 — Monitoring (30 min)
```bash
cd monitoring
docker compose -f docker-compose.monitoring.yml up -d
```
Open http://localhost:9090/targets → confirm `heart-disease-api` is **UP**.
Open http://localhost:3000 (admin/admin) → add Prometheus data source
(`http://prometheus:9090`) → import `monitoring/grafana-dashboard.json`. Hit `/predict`
a few times (loop the curl command above 10-20x) so the dashboard has data to show.
📸 **Screenshots:** Prometheus targets page (UP), Grafana dashboard with live data.

### Phase 7 — GitHub + CI/CD (30-45 min)
```bash
cd mlops-heart-disease
git init
git add .
git commit -m "Initial commit: end-to-end MLOps heart disease pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/mlops-heart-disease.git
git push -u origin main
```
Go to your repo's **Actions** tab on GitHub and watch the `CI` workflow run.
📸 **Screenshot:** the green checkmark / passing workflow run, and the expanded step logs.

> If it's your first push and the repo doesn't exist yet, create an empty repo on
> GitHub first (no README/gitignore, to avoid a merge conflict), then run the commands
> above.

### Phase 8 — Report + Video (60-90 min)
1. Open `Heart_Disease_MLOps_Report.docx` (already drafted for you) and paste your Phase
   1-7 screenshots into the marked placeholder sections.
2. Record a 5-10 minute screen-recording walking through: EDA notebook → MLflow UI →
   pytest passing → Docker build/run + curl → Kubernetes pods/service + curl →
   Prometheus/Grafana dashboard → GitHub Actions green run. Save as `demo_video.mp4`.
3. Final repo structure check — confirm you have: code, Dockerfile, requirements.txt,
   cleaned dataset, notebooks, `tests/`, `.github/workflows/ci.yml`, `k8s/` manifests,
   `screenshots/` folder, and the report file, all pushed to GitHub.

---

## Troubleshooting quick-reference

- **`ModuleNotFoundError` on `python src/train.py`** → you're not inside the activated
  virtualenv, or `pip install -r requirements.txt` didn't finish. Re-run it.
- **Docker build fails on `COPY models/`** → you must run `python src/train.py` BEFORE
  `docker build` (the model file needs to exist on disk first).
- **`kubectl get pods` shows `ImagePullBackOff`** → the image wasn't built locally, or
  Docker Desktop Kubernetes context isn't selected (`kubectl config use-context docker-desktop`).
- **Prometheus target shows `DOWN`** → confirm the API container is running on port 8000
  (`docker ps`), and that Docker Desktop's `host.docker.internal` resolves (it does by
  default on Mac/Windows; on Linux the `extra_hosts: host-gateway` line in
  `docker-compose.monitoring.yml` handles it).
- **GitHub Actions fails on the `docker-build` job** → check the job's logs; it usually
  means `models/` wasn't present because `src/train.py` failed earlier in the same run —
  fix that step's error first, the later jobs will re-run automatically on your next push.

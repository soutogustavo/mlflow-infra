# MLflow Tracking Server

A production-ready MLflow tracking server setup using Docker Compose, with PostgreSQL as the metadata backend and MinIO as artifact storage. Designed to run on an internal company server, with a clear migration path to cloud providers.

---

## Why This Matters

Without a centralized tracking server, a data science team quickly falls into a familiar set of problems: experiments live only on individual laptops, model results are shared over Slack or email, and nobody can confidently answer *"what was the best model we trained last quarter, and how was it built?"*

This setup solves that. Every experiment (i.e. parameters, metrics, artifacts, and the model itself) run by any team member is stored in one place, queryable by everyone, and reproducible.

**For the day-to-day of a DS/MLE team, this means:**

- **No more lost experiments.** Every run is logged automatically to a shared server. If a colleague trained a model six months ago, you can find it, inspect every parameter and metric, and load the model directly — without asking them.
- **Standardized workflow.** The whole team logs experiments the same way, making it easy to compare work across people and projects.
- **Model discoverability.** Instead of hunting through notebooks or asking teammates, you query the tracking server programmatically to find the best model for a given metric, filter by parameters, and load it in seconds.
- **Model Registry as a source of truth.** Validated models are promoted through `Staging → Production` stages. Production systems always load from the registry — no more hardcoded file paths or emailed `.pkl` files.
- **Auditability.** Every model in production has a traceable lineage: which dataset, which code, which hyperparameters, and who ran it.

**On the infrastructure side:**

This solution is intentionally simple today and painlessly scalable tomorrow. It runs on a single internal server with no cloud dependency. When the team grows or compliance requirements change, migrating to cloud is a matter of swapping the backing services — PostgreSQL → RDS, MinIO → S3 — while the MLflow server, all experiment code, and the entire team workflow remain exactly the same. No rewrites, no migrations of application logic, no retraining required.

The investment made now in a structured, centralized tracking setup compounds over time: every experiment the team runs from this point forward is an asset that can be searched, compared, and built upon.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   DS / MLE Team                     │
│         mlflow.log_metric(...) / UI Browser         │
└────────────────────┬────────────────────────────────┘
                     │ HTTP (port 5001)
                     ▼
┌─────────────────────────────────────────────────────┐
│              MLflow Tracking Server                 │
│                  (Docker container)                 │
└──────────┬──────────────────────┬───────────────────┘
           │                      │
           ▼                      ▼
┌──────────────────┐   ┌──────────────────────────────┐
│   PostgreSQL     │   │           MinIO              │
│  (metadata:      │   │  (artifacts: models,         │
│  runs, metrics,  │   │   plots, files...)           │
│  params, tags)   │   │                              │
└──────────────────┘   └──────────────────────────────┘
```

**Why this separation?**
- **PostgreSQL** handles metadata — fast queries, filtering, and comparisons between runs.
- **MinIO** handles artifact storage — cheap, efficient, and S3-compatible, making cloud migration trivial.

---

## Repository Structure

```
mlflow-infra/
├── mlflow/
│   ├── Dockerfile
│   └── requirements.txt
├── tests/
│   ├── test_infrastructure.py
├── environments/
│   ├── docker-compose.dev.yml
│   └── docker-compose.prod.yml
├── docker-compose.yml
├── .env.example
├── .env
├── .gitignore
├── .python-version
├── pyproject.toml
└── uv.lock
```

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (macOS/Windows) or Docker Engine (Linux)
- [UV](https://docs.astral.sh/uv/) for Python environment management

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone git@github.com:soutogustavo/mlflow-infra.git
cd mlflow-infra
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in the values:

```bash
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=<strong-password>
POSTGRES_DB=mlflow
MINIO_ROOT_USER=<minio-user>
MINIO_ROOT_PASSWORD=<strong-password>
MLFLOW_URL=http://localhost:5001
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
```

> **Security**: Never commit `.env`. If a secret is accidentally committed, rotate it immediately — git history cannot be fully erased.

### 3. Create the MinIO bucket

Before starting the services, you need to create the artifact bucket. Start MinIO first:

```bash
docker-compose -f docker-compose.yml -f environments/docker-compose.dev.yml up minio -d
```

Access the MinIO console at `http://localhost:9001`, log in with your credentials, and create a bucket named `mlflow-artifacts`.

### 4. Start all services

**macOS (development):**
```bash
docker-compose -f docker-compose.yml -f environments/docker-compose.dev.yml up -d
```

**Linux server (production):**
```bash
docker-compose -f docker-compose.yml -f environments/docker-compose.prod.yml up -d
```

### 5. Verify services are running

```bash
docker ps
```

All three containers should show `healthy` or `Up`:

```
mlflow-infra-mlflow-1    Up
mlflow-infra-postgres-1  Up (healthy)
mlflow-infra-minio-1     Up (healthy)
```

Access the MLflow UI at `http://localhost:5001`.

---

## Services

| Service    | Port  | Purpose                          |
|------------|-------|----------------------------------|
| MLflow UI  | 5001  | Experiment tracking interface    |
| MinIO API  | 9000  | S3-compatible artifact storage   |
| MinIO UI   | 9001  | MinIO web console                |
| PostgreSQL | 5432  | Metadata backend                 |

---

## Connecting to the Tracking Server

Each team member needs to set these environment variables (add to `~/.bashrc` or `~/.zshrc`):

```bash
export MLFLOW_TRACKING_URI=http://<SERVER_IP>:5001
export MLFLOW_S3_ENDPOINT_URL=http://<SERVER_IP>:9000
export AWS_ACCESS_KEY_ID=<minio-user>
export AWS_SECRET_ACCESS_KEY=<minio-password>
```

Then in any Python script or notebook — no extra configuration needed:

```python
import mlflow

with mlflow.start_run(experiment_name="my-project"):
    mlflow.log_param("model", "xgboost")
    mlflow.log_metric("auc", 0.87)
    mlflow.sklearn.log_model(model, "model")
```

---

## Team Conventions

To keep experiments organized and searchable, the team should follow these conventions:

- **Experiment names**: `<project>-<objective>` (e.g., `churn-binary-classification`)
- **Owner tag**: always set `mlflow.set_tag("owner", "your-name")`
- **Never use the `Default` experiment**: create one per project
- **Always log the model artifact**: not just metrics

---

## Querying Experiments Programmatically

```python
import mlflow

client = mlflow.MlflowClient()

# Find the best run in an experiment
experiment = client.get_experiment_by_name("churn-binary-classification")

best_run = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.auc > 0.85",
    order_by=["metrics.auc DESC"],
    max_results=1
)[0]

# Load the model directly from the server
model = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/model")
```

---

## Model Registry

Once a model is validated, register it for production use:

```python
# Register
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="churn-model"
)

# Promote to production
client.transition_model_version_stage(
    name="churn-model",
    version=1,
    stage="Production"
)

# Load in production: always pulls the current production model
model = mlflow.sklearn.load_model("models:/churn-model/Production")
```

---

## Running Tests

Tests verify that all services are healthy, connected, and working end-to-end.

```bash
# Set up the Python environment
uv venv
source .venv/bin/activate
uv add --dev pytest requests mlflow boto3 psycopg2-binary python-dotenv

# Run all tests
pytest tests/ -v

# Run only health checks
pytest tests/ -v -k "TestHealthChecks"

# Run only the end-to-end smoke test
pytest tests/ -v -k "TestEndToEnd"
```

Always run the tests after making any changes to the compose files or Dockerfile.

---

## Backup

With bind mounts on the production server, backup is straightforward:

```bash
# Backup PostgreSQL metadata
pg_dump -h localhost -U mlflow mlflow > backup_$(date +%Y%m%d).sql

# Backup MinIO artifacts
rsync -av /opt/mlflow/minio/ /your-backup-location/minio/
```

---

## Migrating to Cloud

This setup is designed to make cloud migration a low-effort task. The MLflow server itself requires zero changes, only the backing services are replaced.

### Steps

**1. Provision managed services:**
- AWS: RDS (PostgreSQL) + S3 bucket
- GCP: Cloud SQL + GCS bucket
- Azure: Azure Database for PostgreSQL + Azure Blob Storage

**2. Migrate existing data:**
```bash
# Import PostgreSQL data
psql -h <rds-endpoint> -U mlflow mlflow < backup.sql

# Sync artifacts to S3
aws s3 sync /opt/mlflow/minio s3://your-bucket
```

**3. Update `.env` with new endpoints:**
```bash
# Remove these (no longer needed)
# MLFLOW_S3_ENDPOINT_URL=...  ← remove for real S3/GCS

# Update backend store
POSTGRES_HOST=your-rds-endpoint.amazonaws.com
```

**4. Use `docker-compose.prod.yml` without the MinIO and PostgreSQL services** — they are now managed externally.

The MLflow container and all experiment code remain exactly the same.

---

## Troubleshooting

**MLflow container starts but uses SQLite instead of PostgreSQL**

This is a known issue with the `command: >` fold style in Docker Compose — it can corrupt argument parsing in MLflow 3.x. The solution is to use the list syntax with `|` in the compose file, which is already configured in this repo.

**`ERR_CONNECTION_RESET` on macOS**

Docker Desktop on macOS runs inside a VM and only mounts directories under `/Users` by default. Paths like `/opt/mlflow` will fail with permission errors. Use `~/mlflow-data/` as defined in `docker-compose.dev.yml`.

**`database does not exist` error on first start**

This happens when the PostgreSQL volume has stale state from a previous run where `POSTGRES_DB` was not set. Run `docker-compose down -v` to remove volumes and start fresh.

**Port 5000 already in use**

On macOS, AirPlay Receiver uses port 5000. This setup exposes MLflow on port 5001 externally to avoid this conflict.

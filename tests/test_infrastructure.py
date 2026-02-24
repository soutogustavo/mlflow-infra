"""Infrastructure tests"""

import os
from dotenv import load_dotenv
import requests
import psycopg2
import boto3
import mlflow

load_dotenv()

MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5001")
POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD")
}
MINIO_CONFIG = {
    "endpoint_url": os.getenv("MLFLOW_S3_ENDPOINT_URL"),
    "aws_access_key_id": os.getenv("MINIO_ROOT_USER"),
    "aws_secret_access_key": os.getenv("MINIO_ROOT_PASSWORD")
}


class TestHealthChecks:
    """Check if the infrastructure is up"""

    def test_mlflow_is_up(self):
        """MLflow UI should respond 200"""
        response = requests.get(MLFLOW_URL, timeout=5)
        assert response.status_code == 200

    def test_postgres_is_up(self):
        """Postgres should accept connections"""
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        assert conn.closed == 0
        conn.close()

    def test_minio_is_up(self):
        """MinIO should respond to health check"""
        response = requests.get(
            "http://localhost:9000/minio/health/live",
            timeout=5
        )
        assert response.status_code == 200


class TestIntegration:
    """Test integration between services"""

    def test_mlflow_connected_to_postgres(self):
        """MLflow should have created its tables in Postgres"""
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        expected_tables = ["experiments", "runs", "metrics", "params", "tags"]
        for table in expected_tables:
            assert table in tables, f"Table '{table}' not found in Postgres"

    def test_minio_bucket_exists(self):
        """Bucket mlflow-artifacts should exist in MinIO"""
        s3 = boto3.client("s3", **MINIO_CONFIG)
        buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
        assert "mlflow-artifacts" in buckets, "Bucket 'mlflow-artifacts' not found"

    def test_mlflow_api_returns_experiments(self):
        """MLflow API should return list of experiments"""
        response = requests.get(
            f"{MLFLOW_URL}/ajax-api/2.0/mlflow/experiments/search",
            params={"max_results": 10},
            timeout=5
        )
        assert response.status_code == 200
        data = response.json()
        assert "experiments" in data


class TestEndToEnd:
    """Test end-to-end functionality"""

    def test_create_experiment_and_log_run(self):
        """Should be able to create experiment, log run and retrieve metrics"""
        mlflow.set_tracking_uri(MLFLOW_URL)
        client = mlflow.MlflowClient()

        experiment_name = "test-infra-smoke"

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            if experiment.lifecycle_stage == "deleted":
                client.restore_experiment(experiment.experiment_id)
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(experiment_name)

        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.log_param("model", "test")
            mlflow.log_metric("accuracy", 0.99)
            run_id = run.info.run_id

        logged_run = client.get_run(run_id)
        assert logged_run.data.metrics["accuracy"] == 0.99
        assert logged_run.data.params["model"] == "test"

        client.delete_experiment(experiment_id)

    def test_artifact_storage_reachable(self):
        """MLflow should be able to resolve the artifact store"""
        mlflow.set_tracking_uri(MLFLOW_URL)
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name("Default")
        assert experiment is not None
        assert "s3://mlflow-artifacts" in experiment.artifact_location

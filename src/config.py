import os
from pathlib import Path

def is_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

ENV = "DATABRICKS" if is_databricks() else "LOCAL"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if ENV == "LOCAL":
    DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "housing.csv"
    MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    MLFLOW_ARTIFACT_URI = f"file:///{PROJECT_ROOT / 'mlflow_artifacts'}"
else:
    DATA_RAW_PATH = "dbfs:/FileStore/data/raw/housing.csv"
    MLFLOW_TRACKING_URI = None
    MLFLOW_ARTIFACT_URI = None

RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.8

SPARK_APP_NAME = "ml-pipeline-databricks-pyspark"

SPARK_SQL_SHUFFLE_PARTITIONS = 8

import os
from pathlib import Path
from databricks.sdk import WorkspaceClient

def is_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

ENV = "DATABRICKS" if is_databricks() else "LOCAL"

PROJECT_ROOT = Path.cwd()

# Nome da tabela no Databricks
DATABRICKS_TABLE = "workspace.default.housing"

if ENV == "LOCAL":    
    DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "housing.csv"
    MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    MLFLOW_ARTIFACT_URI = f"file:///{PROJECT_ROOT / 'mlflow_artifacts'}"
    EXPERIMENT_NAME = "california-housing-regression"
else:
    DATA_RAW_PATH = "workspace.default.housing_raw"
    MLFLOW_TRACKING_URI = None
    MLFLOW_ARTIFACT_URI = None

    w = WorkspaceClient()
    me = w.current_user.me()
    user_name = me.user_name
    EXPERIMENT_NAME = f"/Users/{user_name}/california-housing-regression"

RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.8

SPARK_APP_NAME = "ml-pipeline-databricks-pyspark"

SPARK_SQL_SHUFFLE_PARTITIONS = 8

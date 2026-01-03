import mlflow
import mlflow.spark
from pyspark.sql import DataFrame

from src.spark_session import get_spark_session
from src.ingest import ingest_data
from src.config import (
    DATA_RAW_PATH,
    MLFLOW_TRACKING_URI,
    ENV
)

MODEL_NAME = "california_housing_rf"
MODEL_STAGE = "Staging"

OUTPUT_PATH_LOCAL = "data/predictions"
OUTPUT_PATH_DATABRICKS = "dbfs:/FileStore/data/predictions"


def load_model():
    """
    Carrega o modelo a partir do Model Registry (Databricks)
    ou do último run local.
    """
    if ENV == "DATABRICKS":
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    else:
        # Local: pega o último modelo logado no experimento
        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name(
            "california-housing-regression"
        )
        
        if experiment is None:
            raise RuntimeError("Experimento não encontrado no MLflow")
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs:
            raise RuntimeError("Nenhuma run encontrada no MLflow local")

        model_uri = f"runs:/{runs[0].info.run_id}/model"

    return mlflow.spark.load_model(model_uri)


def run_batch_inference():
    spark = get_spark_session()

    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model = load_model()

    df = ingest_data(spark, str(DATA_RAW_PATH))

    preds = model.transform(df)

    output_path = (
        OUTPUT_PATH_DATABRICKS
        if ENV == "DATABRICKS"
        else OUTPUT_PATH_LOCAL
    )

    preds.select(
        "median_house_value",
        "prediction"
    ).write.mode("overwrite").parquet(output_path)

    print(f"Predições salvas em: {output_path}")

    spark.stop()


if __name__ == "__main__":
    run_batch_inference()

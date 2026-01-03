import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

from src.spark_session import get_spark_session
from src.ingest import ingest_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.config import (
    DATA_RAW_PATH,
    RANDOM_SEED,
    TRAIN_TEST_SPLIT,
    MLFLOW_TRACKING_URI,
    MLFLOW_ARTIFACT_URI,
    ENV
)

EXPERIMENT_NAME = "california-housing-regression"
MODEL_NAME = "california_housing_rf"


def main():
    spark = get_spark_session()

    client = MlflowClient()

    # =========================
    # MLflow configuration
    # =========================
    if ENV == "LOCAL":
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

        if experiment is None:
            client.create_experiment(
                name=EXPERIMENT_NAME,
                artifact_location=MLFLOW_ARTIFACT_URI
            )

        mlflow.set_experiment(EXPERIMENT_NAME)

    else:
        # Databricks: tracking e artifact store são gerenciados
        mlflow.set_experiment(EXPERIMENT_NAME)

    # =========================
    # Pipeline execution
    # =========================
    with mlflow.start_run():
        # Ingestão
        df = ingest_data(spark, str(DATA_RAW_PATH))

        # Split (antes do pipeline)
        train_df, test_df = df.randomSplit(
            [TRAIN_TEST_SPLIT, 1 - TRAIN_TEST_SPLIT],
            seed=RANDOM_SEED
        )

        # Treinamento (PipelineModel)
        model = train_model(train_df)

        # Avaliação
        rmse, r2 = evaluate_model(model, test_df)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # =========================
        # Registro do modelo
        # =========================
        if ENV == "DATABRICKS":
            mlflow.spark.log_model(
                spark_model=model,
                artifact_path="model",
                registered_model_name=MODEL_NAME
            )

            # Promoção automática para Staging
            versions = client.get_latest_versions(
                MODEL_NAME, stages=["None"]
            )

            if versions:
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=versions[0].version,
                    stage="Staging",
                    archive_existing_versions=True
                )

        else:
            # LOCAL: loga apenas o artefato (sem Registry)
            # mlflow.spark.log_model(
            #     spark_model=model,
            #     artifact_path="model"
            # )
            pass

        print(f"RMSE: {rmse:.2f}")
        print(f"R2: {r2:.4f}")

    spark.stop()


if __name__ == "__main__":
    main()

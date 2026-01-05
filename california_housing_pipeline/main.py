import os
import mlflow
import mlflow.spark
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from california_housing_pipeline.spark_session import get_spark_session
from california_housing_pipeline.ingest import ingest_data
from california_housing_pipeline.train import train_model
from california_housing_pipeline.evaluate import evaluate_model
from california_housing_pipeline.config import (
    RANDOM_SEED,
    TRAIN_TEST_SPLIT,
    MLFLOW_TRACKING_URI,
    MLFLOW_ARTIFACT_URI,
    EXPERIMENT_NAME,
    ENV
)

MODEL_NAME = "workspace.default.california_housing"


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
        os.environ["MLFLOW_DFS_TMP"] = "/Volumes/workspace/default/mlflow_tmp"

    # =========================
    # Pipeline execution
    # =========================
    with mlflow.start_run():
        # Ingestão
        df = ingest_data(spark)

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

            input_data_sample = train_df.limit(10)
            signature = infer_signature(input_data_sample, model.transform(input_data_sample))

            mlflow.spark.log_model(
                spark_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name=MODEL_NAME
            )

            client = MlflowClient()

            run_id = mlflow.active_run().info.run_id

            model_versions = client.search_model_versions(
                filter_string=f"name = '{MODEL_NAME}'"
            )

            model_version = None
            
            for mv in model_versions:
                if mv.run_id == run_id:
                    model_version = mv.version
                    break

            if model_version is None:
                raise RuntimeError("Could not find model version for current run")
            
            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias="champion",
                version=model_version
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

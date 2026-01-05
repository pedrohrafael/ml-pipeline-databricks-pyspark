from pyspark.sql import DataFrame
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from california_housing_pipeline.features import build_feature_pipeline
# from california_housing_pipeline.config import (
#     RANDOM_SEED
# )


def train_model(train_df: DataFrame):
    feature_pipeline = build_feature_pipeline()

    rf = LinearRegression(
        featuresCol="features",
        labelCol="median_house_value",
        regParam=0.1,
    )

    pipeline = Pipeline(stages=[feature_pipeline, rf])

    model = pipeline.fit(train_df)
    return model

from pyspark.sql import DataFrame
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from california_housing_pipeline.features import build_feature_pipeline

def train_model(train_df: DataFrame):
    feature_pipeline = build_feature_pipeline()

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="median_house_value",
        numTrees=100,
        maxDepth=10,
        seed=42
    )

    pipeline = Pipeline(stages=[feature_pipeline, rf])

    model = pipeline.fit(train_df)
    return model

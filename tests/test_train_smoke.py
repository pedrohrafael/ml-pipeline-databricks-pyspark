from pyspark.sql import SparkSession
from pyspark.sql.types import *
from california_housing_pipeline.batch.train import train_model
from california_housing_pipeline.features import build_feature_pipeline


def test_train_pipeline_smoke():
    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName("ci-train-smoke-test")
        .getOrCreate()
    )

    schema = StructType([
        StructField("longitude", DoubleType()),
        StructField("latitude", DoubleType()),
        StructField("housing_median_age", DoubleType()),
        StructField("total_rooms", DoubleType()),
        StructField("total_bedrooms", DoubleType()),
        StructField("population", DoubleType()),
        StructField("households", DoubleType()),
        StructField("median_income", DoubleType()),
        StructField("median_house_value", DoubleType()),
        StructField("ocean_proximity", StringType()),
    ])

    df = spark.createDataFrame([], schema)

    pipeline = build_feature_pipeline()

    # Smoke test: pipeline construído corretamente
    assert pipeline is not None
    assert len(pipeline.getStages()) > 0

    spark.stop()

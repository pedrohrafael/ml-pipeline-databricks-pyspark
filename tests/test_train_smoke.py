from pyspark.sql import SparkSession
from pyspark.sql.types import *
from california_housing_pipeline.train import train_model


def test_train_pipeline_smoke():
    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName("ci-train-smoke-test")
        .getOrCreate()
    )

    data = [
        (-122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252, 452600.0, "NEAR BAY"),
        (-122.22, 37.86, 21.0, 7099.0, 1106.0, 2401.0, 1138.0, 8.3014, 358500.0, "NEAR BAY"),
    ]

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

    df = spark.createDataFrame(data, schema)

    model = train_model(df)

    assert model is not None

    spark.stop()

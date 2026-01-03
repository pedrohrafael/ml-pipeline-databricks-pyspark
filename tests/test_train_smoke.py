from pyspark.sql import SparkSession
from pyspark.sql.types import *
from california_housing_pipeline.train import train_model


def test_train_pipeline_smoke():
    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName("ci-smoke-test")
        .getOrCreate()
    )

    data = [
        (-122.23, 37.88, 41, 880, 129, 322, 126, 8.3252, 452600, "NEAR BAY"),
        (-122.22, 37.86, 21, 7099, 1106, 2401, 1138, 8.3014, 358500, "NEAR BAY"),
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

from pyspark.sql import DataFrame
from pyspark.sql.types import (
    StructType, StructField, DoubleType, StringType
)

def get_schema():
    return StructType([
        StructField("longitude", DoubleType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("housing_median_age", DoubleType(), True),
        StructField("total_rooms", DoubleType(), True),
        StructField("total_bedrooms", DoubleType(), True),
        StructField("population", DoubleType(), True),
        StructField("households", DoubleType(), True),
        StructField("median_income", DoubleType(), True),
        StructField("median_house_value", DoubleType(), True),
        StructField("ocean_proximity", StringType(), True),
    ])

def ingest_data(spark, path: str) -> DataFrame:
    df = (
        spark.read
        .schema(get_schema())
        .option("header", True)
        .csv(path)
    )

    if df.count() == 0:
        raise ValueError("Dataset vazio após ingestão")

    return df

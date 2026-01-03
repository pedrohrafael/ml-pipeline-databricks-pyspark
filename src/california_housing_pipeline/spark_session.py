from pyspark.sql import SparkSession
from california_housing_pipeline.config import (
    ENV,
    SPARK_APP_NAME,
    SPARK_SQL_SHUFFLE_PARTITIONS
)

def get_spark_session() -> SparkSession:
    builder = SparkSession.builder.appName(SPARK_APP_NAME)

    if ENV == "LOCAL":
        builder = builder.master("local[*]")

    spark = (
        builder
        .config("spark.sql.shuffle.partitions", SPARK_SQL_SHUFFLE_PARTITIONS)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark

from pyspark.sql import SparkSession
from california_housing_pipeline.ingest import ingest_data
from california_housing_pipeline.config import ENV


def test_ingest_local(monkeypatch):
    monkeypatch.setenv("ENV", "LOCAL")

    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName("ci-ingest-test")
        .getOrCreate()
    )

    df = ingest_data(spark)

    assert df is not None
    assert df.count() > 0

    spark.stop()

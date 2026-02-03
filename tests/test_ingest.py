from pyspark.sql import SparkSession
from california_housing_pipeline.batch.ingest import ingest_data


def test_ingest_data_local(monkeypatch):
    monkeypatch.setenv("ENV", "LOCAL")

    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName("ci-ingest-test")
        .getOrCreate()
    )

    df = ingest_data(spark)

    assert df is not None
    assert df.columns is not None
    assert len(df.columns) > 0

    spark.stop()

from pyspark.sql import SparkSession
from src.ingest import get_schema


def test_schema_columns():
    schema = get_schema()
    expected_cols = {
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "median_house_value",
        "ocean_proximity",
    }

    assert set(schema.fieldNames()) == expected_cols

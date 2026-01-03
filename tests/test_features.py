from pyspark.sql import SparkSession
from california_housing_pipeline.features import build_feature_pipeline


def test_feature_pipeline_creation():
    pipeline = build_feature_pipeline()
    assert pipeline is not None
    assert len(pipeline.getStages()) > 0

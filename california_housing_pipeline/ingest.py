from pyspark.sql import SparkSession, DataFrame
from california_housing_pipeline.config import ENV, DATA_RAW_PATH, DATABRICKS_TABLE


def ingest_data(spark: SparkSession) -> DataFrame:
    """
    Ingestão de dados a partir de tabela Spark gerenciada.
    
    Exemplo de table_name:
        workspace.default.housing_raw
    """
    if ENV == "DATABRICKS":
        return spark.table(DATABRICKS_TABLE)
    else:
        return (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(str(DATA_RAW_PATH))
        )

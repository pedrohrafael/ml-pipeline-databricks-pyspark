from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    Imputer
)

CATEGORICAL_COLS = ["ocean_proximity"]

NUMERIC_COLS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",   # contém NULLs
    "population",
    "households",
    "median_income",
]

LABEL_COL = "median_house_value"


def build_feature_pipeline():
    # 1. Imputação de valores ausentes (mediana)
    imputer = Imputer(
        inputCols=NUMERIC_COLS,
        outputCols=[f"{c}_imputed" for c in NUMERIC_COLS],
        strategy="median"
    )

    imputed_cols = [f"{c}_imputed" for c in NUMERIC_COLS]

    # 2. Indexação categórica
    indexer = StringIndexer(
        inputCol="ocean_proximity",
        outputCol="ocean_proximity_idx",
        handleInvalid="keep"
    )

    # 3. One-hot encoding
    encoder = OneHotEncoder(
        inputCol="ocean_proximity_idx",
        outputCol="ocean_proximity_ohe"
    )

    # 4. VectorAssembler FINAL (sem NULLs)
    assembler = VectorAssembler(
        inputCols=imputed_cols + ["ocean_proximity_ohe"],
        outputCol="features"
    )

    pipeline = Pipeline(
        stages=[imputer, indexer, encoder, assembler]
    )

    return pipeline

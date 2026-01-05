from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler,
    Imputer
)

# Apenas colunas numéricas
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

    # 2. VectorAssembler final
    assembler = VectorAssembler(
        inputCols=imputed_cols,
        outputCol="features"
    )

    pipeline = Pipeline(
        stages=[imputer, assembler]
    )

    return pipeline

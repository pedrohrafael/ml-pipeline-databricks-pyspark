from pyspark.sql import DataFrame
from pyspark.ml.evaluation import RegressionEvaluator

def evaluate_model(model, test_df: DataFrame):
    predictions = model.transform(test_df)

    rmse = RegressionEvaluator(
        labelCol="median_house_value",
        predictionCol="prediction",
        metricName="rmse"
    ).evaluate(predictions)

    r2 = RegressionEvaluator(
        labelCol="median_house_value",
        predictionCol="prediction",
        metricName="r2"
    ).evaluate(predictions)

    return rmse, r2

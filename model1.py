# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 13:46:40 2019

@author: rmahajan14
"""

from loader1 import load_spark_df, load_pandas_df
import pyspark
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, MatrixFactorizationModel

dir_name = 'ml-latest-small'
ratings_spark_df = load_spark_df(dir_name, 'ratings', use_cache=True)

(training, test) = ratings_spark_df.randomSplit([0.8, 0.2])

als = ALS(
    maxIter=5,
    regParam=0.09,
    rank=25,
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True)
model = als.fit(training)

evaluator = RegressionEvaluator(
    metricName="rmse", labelCol="rating", predictionCol="prediction")

predictions = model.transform(test)
rmse = evaluator.evaluate(predictions)
print("RMSE=" + str(rmse))
predictions.show()

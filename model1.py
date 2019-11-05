# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 13:46:40 2019

@author: rmahajan14
"""

from loader1 import load_spark_df, load_pandas_df
import pandas as pd
import numpy as np

from time import time

import pyspark
from pyspark.sql.functions import split, explode
from pyspark.sql import Row
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from sklearn.neighbors import NearestNeighbors


def get_als_model(df,
                  rank,
                  split=[0.9, 0.1],
                  model='ALS',
                  evaluator='Regression'):
    train, test = df.randomSplit(split, seed=1)

    total_unique__movieids_train = train.select(
        'movieId').distinct().toPandas().values
    total_unique__movieids_test = test.select(
        'movieId').distinct().toPandas().values

    if model == 'ALS':
        model = ALS(maxIter=5,
                    regParam=0.09,
                    rank=rank,
                    userCol="userId",
                    itemCol="movieId",
                    ratingCol="rating",
                    coldStartStrategy="drop",
                    nonnegative=True)

    if evaluator == 'Regression':
        evaluator = RegressionEvaluator(metricName="rmse",
                                        labelCol="rating",
                                        predictionCol="prediction")
    start = time()
    model = model.fit(train)
    running_time = time() - start
    predictions = model.transform(test)
    rmse_test = evaluator.evaluate(model.transform(test))
    rmse_train = evaluator.evaluate(model.transform(train))

    pred_unique_movieids = calculate_coverage(model)
    subset_pred_train = [
        i for i in pred_unique_movieids if i in total_unique__movieids_train
    ]
    subset_pred_test = [
        i for i in pred_unique_movieids if i in total_unique__movieids_test
    ]
    coverage_train = len(subset_pred_train) / len(total_unique__movieids_train)
    coverage_test = len(subset_pred_test) / len(total_unique__movieids_test)

    return (predictions, model, rmse_train, rmse_test, coverage_train,
            coverage_test, running_time)


def get_nearest_neighbours_model():
    from scipy.sparse import csr_matrix
    #make an object for the NearestNeighbors Class.
    model_knn = NearestNeighbors(metric='cosine',
                                 algorithm='brute',
                                 n_neighbors=20,
                                 n_jobs=-1)
    # fit the dataset
    model_knn.fit(movie_user_mat_sparse)


def calculate_coverage(model):
    user_recos = model.recommendForAllUsers(numItems=10)
    #    breakpoint()

    df1 = user_recos.select(explode(user_recos.recommendations).alias('col1'))
    df2 = df1.select('col1.*')
    df3 = df2.select('movieId').distinct()
    df4 = df3.toPandas()
    movie_set = df4['movieId'].values
    return movie_set


def get_best_rank(df, ranks=[2**i for i in range(7)]):
    #based on rmse
    rmse_train_dict = dict()
    coverage_train_dict = dict()
    rmse_test_dict = dict()
    coverage_test_dict = dict()
    running_time_dict = dict()

    for rank in ranks:
        _, model, rmse_train, rmse_test, coverage_train, coverage_test, running_time = get_als_model(
            df, rank, model='ALS', evaluator='Regression')
        rmse_train_dict[rank] = rmse_train
        rmse_test_dict[rank] = rmse_test
        coverage_train_dict[rank] = coverage_train
        coverage_test_dict[rank] = coverage_test
        running_time_dict[rank] = running_time
    
    df = pd.DataFrame(data=np.asarray([list(rmse_train_dict.keys()),
                            list(rmse_train_dict.values()),
                            list(rmse_test_dict.values()),
                            list(coverage_train_dict.values()),
                            list(coverage_test_dict.values()),
                            list(running_time_dict.values())
                            ]).T,
                      columns=[
                          'Rank', 'RMSE_train', 'RMSE_test', 'Coverage_train',
                          'Coverage_test', 'Running_time'
                      ])
    
    return df


def get_rank_report(df):
    rank = 64
    predictions, model, rmse = get_als_model_rmse(df, rank)
    valuesAndPreds = predictions.rdd.map(lambda x: (x.rating, x.prediction))
    regressionmetrics = RegressionMetrics(valuesAndPreds)
    rankingmetrics = RankingMetrics(valuesAndPreds)
    print("MAE = {regressionmetrics.meanAbsoluteError}")


def cross_validation(df,
                     model='ALS',
                     evaluator='Regression',
                     param_grid=None,
                     k_folds=3):
    """
        Cross validation
    """
    train, test = df.randomSplit([0.9, 0.1], seed=1)

    if model == 'ALS':
        model = ALS(userCol="userId",
                    itemCol="movieId",
                    ratingCol="rating",
                    coldStartStrategy="drop",
                    nonnegative=True)

    if evaluator == 'Regression':
        evaluator = RegressionEvaluator(metricName="rmse",
                                        labelCol="rating",
                                        predictionCol="prediction")

    if not param_grid:
        param_grid = ParamGridBuilder() \
        .addGrid(model.maxIter, [3]) \
        .addGrid(model.regParam, [0.01,0.1]) \
        .addGrid(model.rank, [64, 128]) \
        .build()

    crossval = CrossValidator(estimator=model,
                              estimatorParamMaps=param_grid,
                              evaluator=evaluator,
                              numFolds=3)

    cvModel = crossval.fit(train)
    predictions = cvModel.transform(test)
    rmse = evaluator.evaluate(predictions)
    print(f'RMSE is {rmse}')
    print(cvModel.getEstimatorParamMaps()[0])


if __name__ == '__main__':
    dir_name = 'ml-latest-small'
    ratings_spark_df = load_spark_df(dir_name, 'ratings', use_cache=True)
    #rmse_dict = get_best_rank(ratings_spark_df)
    #    get_rank_report(ratings_spark_df)
    #    print("RMSE=" + str(rmse))
    #    predictions.show()
    cross_validation(ratings_spark_df)

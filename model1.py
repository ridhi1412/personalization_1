# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 13:46:40 2019

@author: rmahajan14
"""

from loader1 import load_spark_df, load_pandas_df
import pyspark
from pyspark.sql.functions import split, explode
from pyspark.sql import Row
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from sklearn.neighbors import NearestNeighbors


def get_als_model_rmse(df, rank):
    train, test = df.randomSplit([0.9, 0.1], seed=1)
    als = ALS(maxIter=5,
              regParam=0.09,
              rank=rank,
              userCol="userId",
              itemCol="movieId",
              ratingCol="rating",
              coldStartStrategy="drop",
              nonnegative=True)

    model = als.fit(train)
    evaluator = RegressionEvaluator(metricName="rmse",
                                    labelCol="rating",
                                    predictionCol="prediction")
    predictions = model.transform(test)
    rmse = evaluator.evaluate(predictions)
    return (predictions, model, rmse)


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
    #    recos_list = user_recos.select('recommendations').collect()
    #    recos_list = [el for el in recos_list]
    #    recos_list = [x for b in recos_list for x in b]
    #    recos_list = [item for sublist in recos_list for item in sublist]
    #    movie_list = [row['movieId'] for row in recos_list]
    #    movie_set = list(set(movise_list))
    return movie_set


def get_best_rank(df, ranks=[2**i for i in range(7)]):
    #based on rmse
    rmse_dict = {}
    coverage_dict = {}
    for rank in ranks:
        _, model, rmse = get_als_model_rmse(df, rank)
        print(f'RANK: {rank} RMSE : {rmse:.4f}')
        coverage = calculate_coverage(model)
        rmse_dict[rank] = rmse
        coverage_dict[rank] = coverage
    return rmse_dict, coverage_dict


def get_rank_report(df):
    rank = 64
    predictions, model, rmse = get_als_model_rmse(df, rank)
    valuesAndPreds = predictions.rdd.map(lambda x: (x.rating, x.prediction))
    regressionmetrics = RegressionMetrics(valuesAndPreds)
    rankingmetrics = RankingMetrics(valuesAndPreds)
    print("MAE = {regressionmetrics.meanAbsoluteError}")


def cross_validation(df, model='ALS', evaluator='Regression', param_grid=None, k_folds=3 ):
    """
        Cross validation
    """
    train, test = df.randomSplit([0.9, 0.1], seed=1)
    
    if model=='ALS':
        model = ALS(userCol="userId",
                  itemCol="movieId",
                  ratingCol="rating",
                  coldStartStrategy="drop",
                  nonnegative=True)

    if evaluator=='Regression':
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

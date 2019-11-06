# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 13:46:40 2019

@author: rmahajan14
"""
# conda install -c conda-forge scikit-surprise
# conda install -c conda-forge lightfm
# conda install tqdm
import math

from tqdm.auto import tqdm
from utils.data_loader import load_spark_df, load_pandas_df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from time import time

import pyspark
from pyspark.sql.functions import split, explode
from pyspark.sql import Row
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
try:
    from sample_df import sample_df_threshold_use_pandas
except:
    from utils.sample_df import sample_df_threshold_use_pandas


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


def calculate_coverage(model):
    """
        Returns all unique movies ids recommended atleast once to a user
    """
    user_recos = model.recommendForAllUsers(numItems=10)
    df1 = user_recos.select(explode(user_recos.recommendations).alias('col1'))
    df2 = df1.select('col1.*')
    df3 = df2.select('movieId').distinct()
    df4 = df3.toPandas()
    movie_set = df4['movieId'].values
    return movie_set


def get_best_rank(df, ranks=[2**i for i in range(7)]):
    """
        Returns a report of performance metrics for ALS model for diffrent ranks
    """
    rmse_train_dict = dict()
    coverage_train_dict = dict()
    rmse_test_dict = dict()
    coverage_test_dict = dict()
    running_time_dict = dict()

    for rank in tqdm(ranks):
        _, model, rmse_train, rmse_test, coverage_train, coverage_test, running_time = get_als_model(
            df, rank, model='ALS', evaluator='Regression')
        rmse_train_dict[rank] = rmse_train
        rmse_test_dict[rank] = rmse_test
        coverage_train_dict[rank] = coverage_train
        coverage_test_dict[rank] = coverage_test
        running_time_dict[rank] = running_time

    df = pd.DataFrame(data=np.asarray([
        list(rmse_train_dict.keys()),
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
    predictions = cvModel.bestModel.transform(test)
    rmse = evaluator.evaluate(predictions)
    print(f'RMSE of Best Model on Test Set: {rmse:.4f}')
    #print(cvModel.bestModel)
    #print('\n')

    label = list()
    value = list()

    temp = cvModel.getEstimatorParamMaps()[np.argmax(cvModel.avgMetrics)]
    list(temp.keys())[0].name
    list(temp.values())[0]

    k = list(temp.keys())
    v = list(temp.values())
    for i in range(len(temp)):
        label.append(k[i].name)
        value.append(v[i])
    best_hyper_parameter = pd.DataFrame(data=[label, value]).T
    best_hyper_parameter.columns = ['HyperParameter', 'Value']

    return (best_hyper_parameter, cvModel)


def get_best_sample_size(df, sample_size=[10**i for i in range(4, 6)]):
    """
        Returns a report for sample size
    """
    rmse_train_dict = dict()
    coverage_train_dict = dict()
    rmse_test_dict = dict()
    coverage_test_dict = dict()
    running_time_dict = dict()

    for size in tqdm(sample_size):

        temp_spark_df = sample_df_threshold_use_pandas(df,
                                                       n=size,
                                                       min_user_threshold=5,
                                                       min_item_threshold=5)

        predictions, model, rmse_train, rmse_test, coverage_train, coverage_test, running_time = get_als_model(
            temp_spark_df,
            rank=64,
            split=[0.9, 0.1],
            model='ALS',
            evaluator='Regression')

        rmse_train_dict[size] = rmse_train
        coverage_train_dict[size] = coverage_train
        rmse_test_dict[size] = rmse_test
        coverage_test_dict[size] = coverage_test
        running_time_dict[size] = running_time

    df = pd.DataFrame(data=np.asarray([
        list(rmse_train_dict.keys()),
        list(rmse_train_dict.values()),
        list(rmse_test_dict.values()),
        list(coverage_train_dict.values()),
        list(coverage_test_dict.values()),
        list(running_time_dict.values())
    ]).T,
                      columns=[
                          'Sample_size', 'RMSE_train', 'RMSE_test',
                          'Coverage_train', 'Coverage_test', 'Running_time'
                      ])

    return df


def plot_performance_als(report_df, report_type='rank'):
    if report_type == 'rank':
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))

        ax[0].plot(report_df['RMSE_train'])
        ax[0].plot(report_df['RMSE_test'])
        ax[0].legend(['Train', 'Test'])
        ax[0].title.set_text('Error vs Rank for ALS model')
        ax[0].set_ylabel('RMSE')
        ax[0].set_xlabel('Log_2(Rank)')

        ax[1].plot(report_df['Coverage_train'])
        ax[1].plot(report_df['Coverage_test'])
        ax[1].legend(['Train', 'Test'])
        ax[1].title.set_text('Coverage vs Rank for ALS model')
        ax[1].set_ylabel('Coverage')
        ax[1].set_xlabel('Log_2(Rank)')
        plt.show()

        plt.figure(figsize=(20, 5))
        plt.plot(report_df['Running_time'])
        plt.title('Running Time vs Rank for ALS model')
        plt.ylabel('Running Time (seconds)')
        plt.xlabel('Log_2(Rank)')
        plt.show()

        print('\n')

    elif report_type == 'sample':
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))
        low = int(math.log10(report_df['Sample_size'].iloc[0]))
        high = int(math.log10(report_df['Sample_size'].iloc[-1]))
        x_tick = [i for i in range(low, high)]
        ax[0].plot(report_df['RMSE_train'])
        ax[0].plot(report_df['RMSE_test'])
        ax[0].legend(['Train', 'Test'])
        ax[0].title.set_text('Error vs Sample Size for ALS model')
        ax[0].set_ylabel('RMSE')
        ax[0].set_xlabel('Log_10(Sample_size)+4')
        #ax[0].set_xticks(x_tick)

        ax[1].plot(report_df['Coverage_train'])
        ax[1].plot(report_df['Coverage_test'])
        ax[1].legend(['Train', 'Test'])
        ax[1].title.set_text('Coverage vs Sample Size for ALS model')
        ax[1].set_ylabel('Coverage')
        ax[1].set_xlabel('Log_10(Sample_size)+4')
        #ax[1].set_xticks(x_tick)
        plt.show()

        plt.figure(figsize=(20, 5))
        plt.plot(report_df['Running_time'])
        plt.title('Running Time vs Sample Size for ALS model')
        plt.ylabel('Running Time (seconds)')
        plt.xlabel('Log_10(Sample_size)+4')
        #plt.xticks(x_tick)
        plt.show()

        print('\n')


if __name__ == '__main__':
    dir_name = 'ml-latest-small'
    ratings_spark_df = load_spark_df(dir_name, 'ratings', use_cache=True)
    #rmse_dict = get_best_rank(ratings_spark_df)
    #    get_rank_report(ratings_spark_df)
    #    print("RMSE=" + str(rmse))
    #    predictions.show()
    cross_validation(ratings_spark_df)

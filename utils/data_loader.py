# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 12:40:57 2019

@author: rmahajan14
"""

import pandas as pd
import numpy as np
import os
import sys
from scipy import sparse

from pyspark import SparkContext
from pyspark.sql import SQLContext

#from utils.common import CACHE_DIR, DATA_DIR


def sample_data_frame(df,
                      ratio=0.2,
                      min_user_threshold=5,
                      min_item_threshold=5):
    """
        Samples and applies a threshold filter on the dataframe
    """
    print(f'Length before sampling: {df.count()}')
    sample_df = df.sample(False, ratio, 42)
    print(f'Length after sampling: {sample_df.count()}')
    sample_df = sample_df.filter(sample_df['userId'] >= min_user_threshold)
    sample_df = sample_df.filter(sample_df['movieId'] >= min_item_threshold)
    print(f'Length after thresholding: {sample_df.count()}')

    return sample_df


def load_pandas_df(dir_name,
                   file_name,
                   use_cache=True,
                   DATA_DIR=None,
                   CACHE_DIR=None):
    """
        Loads pandas data frame from memory or loads from cache file
    """
    #    breakpoint()
    cache_path = os.path.join(CACHE_DIR, f'{dir_name}_{file_name}.msgpack')
    if os.path.exists(cache_path) and use_cache:
        print(f'Loading from {cache_path}')
        df = pd.read_msgpack(cache_path)
    else:
        print('-------------------ss', DATA_DIR)
        csv_path = os.path.join(DATA_DIR, dir_name, file_name + '.csv')
        print('------------------- ', csv_path)
        df = pd.read_csv(csv_path)
        pd.to_msgpack(cache_path, df)
        print(f'Dumping to {cache_path}')
    return df


def load_spark_df(dir_name,
                  file_name,
                  use_cache=True,
                  DATA_DIR=None,
                  CACHE_DIR=None):
    """
        Converts a pandas data frame loaded from memory into a spark data frame
    """
    cache_path = os.path.join(CACHE_DIR,
                              f'spark_{dir_name}_{file_name}.msgpack')
    if os.path.exists(cache_path) and use_cache:
        print(f'Loading from {cache_path}')
        spark_df = pd.read_msgpack(cache_path)
    else:
        pandas_df = load_pandas_df(
            dir_name=dir_name,
            file_name=file_name,
            use_cache=True,
            DATA_DIR=DATA_DIR,
            CACHE_DIR=CACHE_DIR)
        #        sc = SparkContext.('local','example')  # if using locally
        sc = SparkContext.getOrCreate()  # else get multiple contexts error
        sql_sc = SQLContext(sc)
        spark_df = sql_sc.createDataFrame(pandas_df)
    return spark_df


def spark_to_sparse(spark_df, user_or_item='user'):
    """
        Makes a spark data frame sparse for models such as nearest neighbors
        and LightFM
    """
    df = spark_df.drop('timestamp')
    pd_df = df.toPandas()

    row = pd_df['userId'].values
    column = pd_df['movieId'].values
    values = pd_df['rating'].values

    num_rows = max(pd_df['userId'])
    num_columns = max(pd_df['movieId'])

    sparse_mat = np.empty([num_rows + 1, num_columns + 1])
    sparse_mat[row, column] = values
    if user_or_item == 'item':
        sparse_mat = sparse_mat.T
    elif user_or_item == 'user':
        pass
    else:
        sys.exit()

    sparse_mat = sparse.csr_matrix(sparse_mat)
    return sparse_mat


if __name__ == '__main__':
    dir_name = 'ml-latest-small'
    #    dir_name = 'ml-20m'
    movies_pandas_df = load_pandas_df(dir_name, 'movies', use_cache=True)
    ratings_pandas_df = load_pandas_df(dir_name, 'ratings', use_cache=True)

    movies_spark_df = load_spark_df(dir_name, 'movies', use_cache=True)
    ratings_spark_df = load_spark_df(dir_name, 'ratings', use_cache=True)

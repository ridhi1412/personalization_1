# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:20:08 2019

@author: rmahajan14
"""

#ratings_spark_df
from pyspark.mllib.linalg.distributed import CoordinateMatrix
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys
from scipy import sparse


def spark_to_sparse(spark_df, user_or_item='user'):
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


def get_nn(sparse_mat,
           num_neighbors=5,
           metric='euclidean',
           algorithm='auto',
           n_neighbors=5):

    model_knn = NearestNeighbors(metric='cosine',
                                 algorithm='brute',
                                 n_neighbors=5,
                                 n_jobs=-1)
    model_knn.fit(sparse_mat)
    distances, indices = model_knn.kneighbors(sparse_mat)
    return (distances, indices)


sparse_mat = spark_to_sparse(ratings_spark_df)
(distances, indices) = get_nn(sparse_mat)
#import scipy as sc
#check = sc.sparse.coo_matrix(aaa.values)
#ddd = check.tobsr()

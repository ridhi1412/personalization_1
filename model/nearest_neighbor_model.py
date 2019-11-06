#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 03:52:17 2019

@author: anirudh
"""

import pandas as pd
import numpy as np
import surprise
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms.knns import KNNWithZScore


def KNN(model, df):
    """
        Prints the result of Cross Validation method
    """
    ratings_pandas_df = df.drop('timestamp').toPandas()
    ratings_pandas_df.columns = ['userID', 'itemID', 'rating']

    reader = Reader(rating_scale=(0, 5.0))
    data = surprise.dataset.Dataset.load_from_df(df=ratings_pandas_df,
                                                 reader=reader)
    _ = cross_validate(model,
                       data,
                       measures=['RMSE', 'MAE'],
                       cv=5,
                       verbose=1,
                       n_jobs=-1)

    print('\n')

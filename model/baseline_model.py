import pandas as pd
import numpy as np
import surprise
from surprise import Reader
from surprise.model_selection import cross_validate


def baseline_bias_model(df):
    """
        Shows the performance of model based on just bias
    """
    ratings_pandas_df = df.drop('timestamp').toPandas()
    ratings_pandas_df.columns = ['userID', 'itemID', 'rating']

    reader = Reader(rating_scale=(-5.0, 5.0))
    data = surprise.dataset.Dataset.load_from_df(df=ratings_pandas_df,
                                                 reader=reader)
    _ = cross_validate(
        surprise.prediction_algorithms.baseline_only.BaselineOnly(),
        data,
        measures=['RMSE', 'MAE'],
        cv=5,
        verbose=1,
        n_jobs=-1)

    print('\n')

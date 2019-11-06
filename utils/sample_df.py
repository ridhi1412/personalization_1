# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:53:42 2019

@author: rmahajan14
"""
import os
import pandas as pd
try:
    from data_loader import load_spark_df, load_pandas_df, pandas_to_spark
except:
    from utils.data_loader import load_spark_df, load_pandas_df, pandas_to_spark

dir_name = 'ml-20m'
CACHE_DIR = r'P:\rmahajan14\columbia\fall 2019\Personalization\project_1\personalization_1\cache'
DATA_DIR = r'P:\rmahajan14\columbia\fall 2019\Personalization\project_1\personalization_1\data'


def random_sample(df, frac):
    df = df.sample(frac=frac, random_state=1)
    return df


def sample_df_threshold_use_pandas(df,
                                   n=100000,
                                   min_user_threshold=5,
                                   min_item_threshold=5,
                                   quite=True):
    """
        Samples and applies a threshold filter on the dataframe
    """
    #    print(f'Length before sampling: {df.count()}')
    #    sample_df = df.sample(False, ratio, 42)
    #    print(f'Length after sampling: {sample_df.count()}')
    if not quite:
        print(f'Length before sampling: {len(df)}')
        df = df.loc[df['userId'] >= min_user_threshold]
        df = df.loc[df['movieId'] >= min_item_threshold]
        print(f'Length after thresholding: {len(df)}')
        df = df.sample(n=n, random_state=1)
        print(f'Length after sampling: {len(df)}')
        spark_df = pandas_to_spark(df)
    else:
        df = df.loc[df['userId'] >= min_user_threshold]
        df = df.loc[df['movieId'] >= min_item_threshold]
        df = df.sample(n=n, random_state=1)

        spark_df = pandas_to_spark(df)

    return spark_df


def sample_df_threshold(df,
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


def sample_popular_df(df,
                      movie_count=10000,
                      user_count=50000,
                      final_sample_size=100000):
    movie_counts = df['movieId'].value_counts().reset_index()
    user_counts = df['userId'].value_counts().reset_index()

    popular_movies = movie_counts.iloc[:movie_count]  #top 10000 movies
    popular_users = user_counts.iloc[:user_count]  #top 100000 users

    df_popular_movies = df.loc[df['movieId'].isin(popular_movies['movieId'])]
    df_popular_users = df.loc[df['userId'].isin(popular_users['userId'])]

    #now merge these 2 to get intersection of most popular movies and users
    df_popular = pd.merge(df_popular_movies,
                          df_popular_users,
                          on=['userId', 'movieId', 'rating', 'timestamp'],
                          how='inner')
    df_sampled = df_popular.sample(n=final_sample_size)
    return df_sampled


if __name__ == '__main__':
    df = load_pandas_df(dir_name,
                        'ratings',
                        DATA_DIR=DATA_DIR,
                        CACHE_DIR=CACHE_DIR,
                        use_cache=True)

    #    df_sampled_big = sample_popular_df(
    #        df, movie_count=10000, user_count=50000, final_sample_size=100000)
    #
    #    df_sampled_medium = sample_popular_df(
    #        df, movie_count=5000, user_count=25000, final_sample_size=50000)
    #
    #    df_sampled_small = sample_popular_df(
    #        df, movie_count=5000, user_count=25000, final_sample_size=10000)

    ratings_spark_df = sample_df_threshold_use_pandas(df,
                                                      n=100000,
                                                      min_user_threshold=5,
                                                      min_item_threshold=5)

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:53:42 2019

@author: rmahajan14
"""
import os
import pandas as pd
from data_loader import load_spark_df, load_pandas_df

dir_name = 'ml-20m'
CACHE_DIR = r'P:\rmahajan14\columbia\fall 2019\Personalization\project_1\personalization_1\cache'
DATA_DIR = r'P:\rmahajan14\columbia\fall 2019\Personalization\project_1\personalization_1\data'

# Loading the Data Frames
df = load_pandas_df(
    dir_name,
    'ratings',
    DATA_DIR=DATA_DIR,
    CACHE_DIR=CACHE_DIR,
    use_cache=False)


def random_sample(large_df, frac):
  df = large_df.sample(frac=frac, random_state=1)
  return df

def choose_popular_movies():
  pass


#def counts_per_user():
movie_counts = df['movieId'].value_counts().reset_index()
user_counts = df['userId'].value_counts().reset_index()

popular_movies = movie_counts.iloc[:1000]
popular_users = user_counts.iloc[:10000]


df_popular_movies = df.loc[df['movieId'].isin(popular_movies['movieId'])]
df_popular_users = df.loc[df['userId'].isin(popular_users['userId'])]

















































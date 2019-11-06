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

def random_sample(df, frac):
  df = df.sample(frac=frac, random_state=1)
  return df

def choose_popular_movies(df, movie_count=10000, user_count=50000,
                          final_sample_size=100000):
  movie_counts = df['movieId'].value_counts().reset_index()
  user_counts = df['userId'].value_counts().reset_index()
  
  popular_movies = movie_counts.iloc[:movie_count] #top 10000 movies
  popular_users = user_counts.iloc[:user_count] #top 100000 users
  
  
  df_popular_movies = df.loc[df['movieId'].isin(popular_movies['movieId'])]
  df_popular_users = df.loc[df['userId'].isin(popular_users['userId'])]
  
  #now merge these 2 to get intersection of most popular movies and users
  df_popular = pd.merge(df_popular_movies, df_popular_users, on=[
      'userId', 'movieId', 'rating', 'timestamp'],
    how='inner')
  
  df_sampled = df_popular.sample(n=final_sample_size)
  return df_sampled

df = load_pandas_df(
    dir_name,
    'ratings',
    DATA_DIR=DATA_DIR,
    CACHE_DIR=CACHE_DIR,
    use_cache=True)

df_sampled = choose_popular_movies(df, movie_count=10000, user_count=50000,
                          final_sample_size=100000)


















































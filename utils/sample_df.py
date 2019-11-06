# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:53:42 2019

@author: rmahajan14
"""
import os
import pandas as pd



def load(use_cache=True):
    cache_path = os.path.join(CACHE_PATH, f'reader.msgpack')
    if use_cache and os.path.exists(cache_path):
        df = pd.read_msgpack(cache_path)
        print(f'Loading from {cache_path}')
    else:
        df_path = os.path.join(EXCEL_PATH, 'df.csv')
        df = pd.read_csv(df_path)
        pd.to_msgpack(cache_path, df)
        print(f'Dumping to {cache_path}')
    return df

# Setting Directory path
base_path = os.getcwd()
dir_name = 'ml-latest-small'
CACHE_DIR = base_path + '/cache/'
DATA_DIR =  base_path + '/data/'

# Loading the Data Frames
movies_spark_df = load_spark_df(dir_name=dir_name, 
                                file_name='movies', 
                                use_cache=True,
                                DATA_DIR=DATA_DIR,
                                CACHE_DIR=CACHE_DIR
                               )

ratings_spark_df = load_spark_df(dir_name=dir_name, 
                                 file_name='ratings', 
                                 use_cache=True,
                                 DATA_DIR=DATA_DIR,
                                 CACHE_DIR=CACHE_DIR)

if __name__ == '__main__':
    df = load()

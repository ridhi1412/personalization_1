# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 12:40:57 2019

@author: rmahajan14
"""

import pandas as pd
import os

from common import CACHE_DIR, DATA_DIR


def load_df(dir_name, file_name, use_cache=True):
    cache_path = os.path.join(CACHE_DIR, f'{dir_name}_{file_name}.msgpack')
    if os.path.exists(cache_path) and use_cache:
        print(f'Loading from {cache_path}')
        df = pd.read_msgpack(cache_path)
    else:
        csv_path = os.path.join(DATA_DIR, dir_name, file_name + '.csv')
        df = pd.read_csv(csv_path)
        
        pd.to_msgpack(cache_path, df)
        print(f'Dumping to {cache_path}')
    return df


if __name__ == '__main__':
    dir_name = 'ml-latest-small'
#    dir_name = 'ml-20m'
    movies_df = load_df(dir_name, 'movies', use_cache=True)
    ratings_df = load_df(dir_name, 'ratings', use_cache=True)

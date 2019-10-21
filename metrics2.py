# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:25:55 2019

@author: rmahajan14
"""


def calculate_coverage(model):
    user_recos = model.recommendForAllUsers()
    return user_recos
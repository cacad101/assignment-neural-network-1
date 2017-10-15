"""
Utils for preprocessing
"""

import numpy as np

def scale(data):
    """ scale data (from begin_project_1b.py) """
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    return (data - data_min)/(data_max - data_min)

def normalize(data):
    """ normalize data (from begin_project_1b.py) """
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    return (data - data_mean)/data_std
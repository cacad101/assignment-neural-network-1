"""
Preprocess Data:
- Load
- Divide into training and test set
- Normalize
"""

import numpy as np
from housing_preprocessing import utils

class Preprocessor:
    """ Preprocess class """

    def __init__(self):
        self.x_data = None
        self.y_data = None
        self.test_x = None
        self.test_y = None
        self.train_x = None
        self.train_y = None
        return


    def load_data(self, data_dir):
        """ Load data from data_dir """
        temp_data = np.loadtxt(data_dir, delimiter=',')
        self.x_data = temp_data[:,:8]
        self.y_data = Y_data = (np.asmatrix(temp_data[:,-1])).transpose()

    def divide_data(self, test_count, train_count):
        """ Divide data into training and test set """
        div_count = (test_count * self.x_data.shape[0]) // (test_count + train_count)
        self.test_x = self.x_data[:div_count]
        self.test_y = self.y_data[:div_count]
        self.train_x = self.x_data[div_count:]
        self.train_y = self.y_data[div_count:]

    def normalize_data(self):
        """ Scale and Normalize test and train data """
        self.test_x = utils.scale(self.test_x)
        self.train_x = utils.scale(self.train_x)

        self.test_x = utils.normalize(self.test_x)
        self.train_x = utils.normalize(self.train_x)
        
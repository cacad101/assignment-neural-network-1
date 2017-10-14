"""
Utils for Training
"""

import numpy as np
import theano

FLOAT_X = theano.config.floatX

def shuffle_data(samples, labels):
    """ Shuffle data (from begin_project_1b.py) """
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

def get_weigh(prev_layer, cur_layer):
    """ get weigh """
    return theano.shared(np.random.randn(prev_layer, cur_layer)*0.01, FLOAT_X)

def get_bias(cur_layer):
    """ get weigh """
    return theano.shared(np.random.randn(cur_layer)*0.01, FLOAT_X)

def get_fold_data(data, fold_num):
    """ split the data into different fold """
    train = []
    validation = []
    for i in range(len(data)):
        if i == fold_num:
            validation = data[i]
        else:
            if len(train) == 0:
                train = data[i]
            else:
                train = np.concatenate((train, data[i]))

    return train, validation
"""
Training Data:
- Approximation
- Mini-batch gradient descent
"""

import numpy as np
import theano
import theano.tensor as T

from housing_training import utils

class Approximation:
    """ Approximation Class """

    def __init__(self):
        self.train = None
        self.test = None

        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

        
    def create_model(self, list_of_neurons_on_hidden_layer, learning_rate):
        """ Create model from list of neurons """
        no_features = self.train_x.shape[1] 
        
        input_x = T.matrix('x') # data sample
        expected_y = T.matrix('d') # desired output
        no_samples = T.scalar('no_samples')

        # alpha = theano.shared(learning_rate, utils.FLOAT_X) 

        # # initialize weights and biases for hidden layer(s) and output layer
        # list_of_neurons_on_hidden_layer
        # weights = []
        # biases = []
        # prev_layer = no_features

        # for cur_layer in list_of_neurons_on_hidden_layer:
        #     weights.append(utils.get_weigh(prev_layer, cur_layer))
        #     biases.append(utils.get_bias(cur_layer))
        #     prev_layer = cur_layer

        # weights.append(utils.get_weigh(prev_layer,1))
        # biases.append(utils.get_bias(1))

        # #Define mathematical expression:
        # h_out = input_x
        # for w,b in zip(weights, biases):
        #     h_out = T.nnet.sigmoid(T.dot(h_out, w) + b)

        # cost = T.abs_(T.mean(T.sqr(expected_y - h_out)))
        # accuracy = T.mean(expected_y - h_out)

        # #define gradients
        # updates = []
        # for w, b in zip(weights, biases):
        #     dw, db = T.grad(cost, [w, b])
        #     updates.append([w, w - alpha * dw])
        #     updates.append([b, b - alpha * db])

        # self.train = theano.function(
        #     inputs = [input_x, expected_y],
        #     outputs = cost,
        #     updates = updates,
        #     allow_input_downcast=True
        #     )

        # self.test = theano.function(
        #     inputs = [input_x, expected_y],
        #     outputs = [h_out, cost, accuracy],
        #     allow_input_downcast=True
        #     )

        ############################################################################################################
        w_o = theano.shared(np.random.randn(list_of_neurons_on_hidden_layer[0])*.01, utils.FLOAT_X ) 
        b_o = theano.shared(np.random.randn()*.01, utils.FLOAT_X)
        w_h1 = theano.shared(np.random.randn(no_features, list_of_neurons_on_hidden_layer[0])*.01, utils.FLOAT_X )
        b_h1 = theano.shared(np.random.randn(list_of_neurons_on_hidden_layer[0])*0.01, utils.FLOAT_X)

        # learning rate
        alpha = theano.shared(learning_rate, utils.FLOAT_X) 


        #Define mathematical expression:
        h1_out = T.nnet.sigmoid(T.dot(input_x, w_h1) + b_h1)
        y = T.dot(h1_out, w_o) + b_o

        cost = T.abs_(T.mean(T.sqr(expected_y - y)))
        accuracy = T.mean(expected_y - y)

        #define gradients
        dw_o, db_o, dw_h, db_h = T.grad(cost, [w_o, b_o, w_h1, b_h1])
        ############################################################################################################


        self.train = theano.function(
            inputs = [input_x, expected_y],
            outputs = cost,
            updates = [[w_o, w_o - alpha*dw_o],
                   [b_o, b_o - alpha*db_o],
                   [w_h1, w_h1 - alpha*dw_h],
                   [b_h1, b_h1 - alpha*db_h]],
            allow_input_downcast=True
            )

        self.test = theano.function(
            inputs = [input_x, expected_y],
            outputs = [y, cost, accuracy],
            allow_input_downcast=True
            )

    def train_model(self, epochs, batch_size):
        """ train model based on self.train_x and self.train_y """
        train_cost = np.zeros(epochs)
        test_cost = np.zeros(epochs)
        test_accuracy = np.zeros(epochs)

        min_error = 1e+15

        for iter in range(epochs):
            # print("train: " + str(iter))
            self.train_x, self.train_y = utils.shuffle_data(self.train_x, self.train_y)
            train_cost[iter] = self.training_iter(batch_size)
            _, test_cost[iter], test_accuracy[iter] = self.test_model()

            if test_cost[iter] < min_error:
                min_error = test_cost[iter]
        
        return train_cost, test_cost, test_accuracy, min_error

    def training_iter(self, batch_size):
        cost = []

        for i in range(0, len(self.train_x), batch_size):
            end = i + batch_size
            if end > len(self.train_x):
                end = len(self.train_x)

            train_x_batch = self.train_x[i:end]
            train_y_batch = self.train_y[i:end]

            cost.append(self.train(train_x_batch, np.transpose(train_y_batch)))
            
        return np.mean(cost)

    def test_model(self):
        """ test model using independent test data """
        return self.test(self.test_x, np.transpose(self.test_y))

    def select_model(self, train_x, train_y, k_fold, epochs, batch_size, hidden_neurons, learning_rate):
        """ select model using K-Fold cross validation """
        data_fold_x = []
        data_fold_y = []

        div_count = train_x.shape[0] // k_fold

        for i in range(k_fold):
            data_fold_x.append(train_x[i*div_count:(i+1)*div_count])
            data_fold_y.append(train_y[i*div_count:(i+1)*div_count])

        min_err = 1e+15
        best_train_x = None
        best_train_y = None
        for i in range(k_fold):
            self.train_x, self.test_x = utils.get_fold_data(data_fold_x, i)
            self.train_y, self.test_y = utils.get_fold_data(data_fold_y, i)
            
            self.create_model(hidden_neurons, learning_rate)
            self.train_model(epochs, batch_size)
            _, new_err, _ = self.test_model()

            if new_err < min_err:
                min_err = new_err
                best_train_x = self.train_x
                best_train_y = self.train_y

        self.train_x = best_train_x
        self.train_y = best_train_y
        return min_err

    def set_x_train(self, train_x):
        """ self.train_x setter """
        self.train_x = train_x

    def set_y_train(self, train_y):
        """ self.train_y setter """
        self.train_y = train_y

    def set_x_test(self, test_x):
        """ self.test_x setter """
        self.test_x = test_x

    def set_y_test(self, test_y):
        """ self.test_y setter """
        self.test_y = test_y

        
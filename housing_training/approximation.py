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

        list_train_cost = []
        list_test_cost = []
        list_test_accuracy = []

        for i in range(k_fold):
            self.train_x, self.test_x = utils.get_fold_data(data_fold_x, i)
            self.train_y, self.test_y = utils.get_fold_data(data_fold_y, i)
            
            self.create_model(hidden_neurons, learning_rate)
            train_cost, test_cost, test_accuracy, new_err = self.train_model(epochs, batch_size)
            
            list_train_cost.append(train_cost)
            list_test_cost.append(test_cost)
            list_test_accuracy.append(test_accuracy)

            if new_err < min_err:
                min_err = new_err
                best_train_x = self.train_x
                best_train_y = self.train_y

        self.train_x = best_train_x
        self.train_y = best_train_y
        return list_train_cost, list_test_cost, list_test_accuracy, min_err
    
    def create_model(self, list_of_neurons_on_hidden_layer, learning_rate):
        """ Create model from list of neurons """
        no_features = self.train_x.shape[1] 
        
        input_x = T.matrix('x') # data sample
        expected_y = T.matrix('d') # desired output

        alpha = theano.shared(learning_rate, utils.FLOAT_X) 

        # initialize weights and biases for hidden layer(s) and output layer
        weights = []
        biases = []
        prev_layer = no_features

        for cur_layer in list_of_neurons_on_hidden_layer + [1]:
            weights.append(utils.get_weigh(prev_layer, cur_layer))
            biases.append(utils.get_bias(cur_layer))
            prev_layer = cur_layer

        # Define mathematical expression:
        h_out = input_x
        for w,b in zip(weights[:-1], biases[:-1]):
            h_out = T.nnet.sigmoid(T.dot(h_out, w) + b)

        h_out = T.dot(h_out, weights[-1]) + biases[-1]

        cost = T.abs_(T.mean(T.sqr(expected_y - h_out)))
        accuracy = T.mean(expected_y - h_out)

        #define gradients
        updates = []
        grad = T.grad(cost, weights + biases)
        grad_w = grad[:len(grad)//2]
        grad_b = grad[len(grad)//2:]
        for i in range(len(weights)):
            updates.append([weights[i], weights[i] - alpha * grad_w[i]])
            updates.append([biases[i], biases[i] - alpha * grad_b[i]])

        self.train = theano.function(
            inputs = [input_x, expected_y],
            outputs = cost,
            updates = updates,
            allow_input_downcast=True
            )

        self.test = theano.function(
            inputs = [input_x, expected_y],
            outputs = [h_out, cost, accuracy],
            allow_input_downcast=True
            )

    def train_model(self, epochs, batch_size):
        """ train model based on self.train_x and self.train_y """
        train_cost = np.zeros(epochs)
        test_cost = np.zeros(epochs)
        test_accuracy = np.zeros(epochs)

        min_error = 1e+15

        for iter in range(epochs):
            self.train_x, self.train_y = utils.shuffle_data(self.train_x, self.train_y)
            train_cost[iter] = self.training_iter(batch_size)
            _, test_cost[iter], test_accuracy[iter] = self.test_model()

            if iter%100 == 0:
                print("Iter: %s\n MSE: %s\n Test Accuracy: %s" %(iter, train_cost[iter], test_accuracy[iter]))
                print("----------------------------------------------------------------------")
            
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

            cost.append(self.train(train_x_batch, train_y_batch))
            
        return np.mean(cost)

    def test_model(self):
        """ test model using independent test data """
        return self.test(self.test_x, self.test_y)

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

        
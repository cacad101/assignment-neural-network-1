"""
Question 4:
Design a four-layer neural network and a five-layer neural network, with the first hidden
layer having number of neurons found in step (3) and other hidden layers having 20
neurons each. Use a learning rate = 10−4. Plot the test errors of the 4-layer network
and 5-layer network, and compare them with that of the 3-layer network.
"""

import numpy as np

from housing_preprocessing.preprocess import Preprocessor
from housing_training.approximation import Approximation
from housing_visualization.visualization import plot_graph

def main():
    print("Start Question 4")
    np.random.seed(10)

    k_fold = 0
    hidden_neuron_list = [[40], [40, 20], [40, 20, 20]]
    batch_size = 4048
    learning_rate = 1e-4
    epochs = 1000
    data_dir = "housing_data/cal_housing.data"

    # Preprocessing: Load data
    preprocessor = Preprocessor()
    preprocessor.load_data(data_dir)
    preprocessor.divide_data(3, 7)
    preprocessor.normalize_data()

    list_train_cost = []
    list_test_cost = []
    list_test_accuracy = []
    nn = Approximation()

    for hidden_neurons in hidden_neuron_list:
        if k_fold > 0:
            nn.select_model(
                train_x = preprocessor.train_x,
                train_y = preprocessor.train_y,
                k_fold = k_fold,
                epochs = epochs,
                batch_size = batch_size,
                hidden_neurons = hidden_neurons,
                learning_rate = learning_rate
            )
            # TODO:
            # Plot K-fold graphs

        else:
            nn.set_x_train(preprocessor.train_x)
            nn.set_y_train(preprocessor.train_y)
        
        nn.set_x_test(preprocessor.test_x)
        nn.set_y_test(preprocessor.test_y)
        nn.create_model(hidden_neurons, learning_rate)
        train_cost, test_cost, test_accuracy, min_err = nn.train_model(epochs=epochs, batch_size=batch_size)
        list_train_cost.append(train_cost)
        list_test_cost.append(test_cost)
        list_test_accuracy.append(test_accuracy)

    # Plot training error against number of epoch
    plot_graph(
        title='Train Errors for each Neuron',
        x_label="Epochs",
        y_label="MSE",
        x_val=range(epochs),
        y_vals=list_train_cost,
        data_labels=hidden_neuron_list,
    )

    # Plot test error of prediction against number of epoch
    plot_graph(
        title='Test Errors for each Neuron',
        x_label="Epochs",
        y_label="MSE",
        x_val=range(epochs),
        y_vals=list_test_cost,
        data_labels=hidden_neuron_list,
    )

    # Plot accuracy against number of epoch
    plot_graph(
        title="Test Accuracy",
        x_label="Epochs",
        y_label="Accuracy",
        x_val=range(epochs),
        y_vals=list_test_accuracy,
        data_labels=hidden_neuron_list,
    )

if __name__ == '__main__':
    main()
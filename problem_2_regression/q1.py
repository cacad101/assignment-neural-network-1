"""
Question 1:
Design a 3-layer feedforward neural network consisting of a hidden-layer of 30 neurons.
Use mini-batch gradient descent (with batch size of 32 and learning rate alpha = 10e−4) to
train the network. Use up to about 1000 epochs for this problem.
a) Plot the training error against number of epochs for the 3-layer network.
b) Plot the final test errors of prediction by the network.
"""

import numpy as np

from housing_preprocessing.preprocess import Preprocessor
from housing_training.approximation import Approximation
from housing_visualization.visualization import plot_graph

def main():
    print("Start Question 1")
    np.random.seed(10)

    k_fold = 5
    hidden_neurons = [30]
    batch_size = 32
    learning_rate = 1e-4
    epochs = 1000
    data_dir = "housing_data/cal_housing.data"

    # Preprocessing: Load data
    preprocessor = Preprocessor()
    preprocessor.load_data(data_dir)
    preprocessor.divide_data(3, 7)
    preprocessor.normalize_data()

    nn = Approximation()

    if k_fold > 0:
        list_train_cost, list_test_cost, list_test_accuracy, min_err = nn.select_model(
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
    train_cost, test_cost, accuracy, min_err = nn.train_model(epochs=epochs, batch_size=batch_size, verbose=True)

    # Plot training error against number of epoch
    # Plot test error of prediction against number of epoch
    plot_graph(
        title='Training and Test Errors at Alpha = %.3f'%learning_rate,
        x_label="Epochs",
        y_label="MSE",
        x_val=range(epochs),
        y_vals=[train_cost, test_cost],
        data_labels=["train", "test"],
    )

    # Plot accuracy against number of epoch
    plot_graph(
        title="Test Accuracy",
        x_label="Epochs",
        y_label="Accuracy",
        x_val=range(epochs),
        y_vals=[accuracy],
        data_labels=["Test Accuracy"],
    )

if __name__ == '__main__':
    main()
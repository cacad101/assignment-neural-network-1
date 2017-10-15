# CZ4042 Neural Network - Assignment 1

Part B: Approximation Problem

## Getting Started

### Prerequisite

- Python 2.7 or above (with pip)
- Jupyter

### Running

- Using python:
  - To run question 1: `python q1.py`
  - To run question 2: `python q2.py`
  - To run question 3: `python q3.py`
  - To run question 4: `python q4.py`
- Using jupyter:
  - execute `jupyter notebook` on the current directory.
  - execute code in [Approximation Problem](approximation_problem.ipynb).

## Content

### Question 1

Design a 3-layer feedforward neural network consisting of a hidden-layer of 30 neurons. Use mini-batch gradient descent (with batch size of 32 and learning rate alpha = 10e−4) to train the network. Use up to about 1000 epochs for this problem.

- Plot the training error against number of epochs for the 3-layer network.
- Plot the final test errors of prediction by the network.

### Question 2

Find the optimal learning rate for the 3-layer network designed. Set this as the learning rate in first hidden layer for the rest of the experiments.

- Plot the training errors and validation errors against number of epochs for the 3-layer network for different learning rates. Limit the search space to: {10e−3, 0.5 x 10e−3, 10e−4, 0.5 x 10e−4, 10e−5}
- Plot the test errors against number of epochs for the optimum learning rate.
- State the rationale behind selecting the optimal learning rate.

### Question 3

Find the optimal number of hidden neurons for the 3-layer network designed.

- Plot the training errors against number of epochs for the 3-layer network for different hidden-layer neurons. Limit search space to:{20,30,40,50,60}.
- Plot the test errors against number of epochs for the optimum number of hidden layer neurons.
- State the rationale behind selecting the optimal number of hidden neurons

### Question 4

Design a four-layer neural network and a five-layer neural network, with the first hidden layer having number of neurons found in step (3) and other hidden layers having 20 neurons each. Use a learning rate = 10−4. Plot the test errors of the 4-layer network and 5-layer network, and compare them with that of the 3-layer network.

### Question 5

Additionally, the project report should contain:

- An introduction to the problem of approximation of housing prices in the California Housing dataset and the use of multilayer feedforward networks for solving the prediction problem.
- The methods used in the experiments.
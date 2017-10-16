# CZ4042 Neural Network - Assignment 1
Part A: Classification Problem

## Getting Started

### Prerequisite

- Python 2.7 or above (with pip)
- Jupyter

### Running

- To run question 1: `python questions/q1.py`
- To run question 2: `python questions/q2.py`
- To run question 3: `python questions/q3.py`
- To run question 4: `python questions/q4.py`
- To run question 5: `python questions/q5.py`

## Content
s project aims at building neural network to classify Landsat satellite dataset:
https://sites.google.com/site/sukritsite/teaching
https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)
The dataset contains multispectral values of pixels in a 3x3 neighbourhoods in a satellite
images and class labels of the centre pixels in each neighbourhood. The aim is to predict class
labels in the test dataset after training the neural networks on the training data.
Training data: sat_train.txt, 4435 samples
Test data: sat_test.txt, 2000 samples
Read the data from the two files and use it as train and test set. Do not use the data in the
test dataset during training. It is reserved for the final performance measures. Think of it as
unseen data during all of your work.
Each data sample is a row of 37 values: 36 input attributes (4 spectral bands x 9 pixels in the
neighbourhood), and the class label. There are 6 class-labels: 1, 2, 3, 4, 5, 7.

### Question 1
Design a 3-layer feedforward neural network consisting of a hidden-layer of 10 neurons
having logistic activation function and an output softmax layer. Assume a learning rate
𝛼 = 0.01, decay parameter 𝛽 = 10−6, and batch size = 32. Use appropriate scaling of input features.

### Question 2
Find the optimal batch size for mini-batch gradient descent while training the neural
network by evaluating the performances for different batch sizes. Set this as the batch
size for the rest of the experiments.

- Plot the training errors and test accuracies against the number of epochs for the 3-layer network for different batch sizes. Limit search space to:{4,8,16,32,64}. 
- Plot the time taken to update parameters of the network against different batch sizes.
- State the rationale for selecting the optimal batch size. 

### Question 3
Find the optimal number of hidden neurons for the 3-layer network designed in part (2).
Set this number of neurons in the hidden layer for the rest of the experiments.

- Plot the training errors and test accuracies against the number of epochs for 3-layer network at hidden-layer neurons. Limit the search space to the set:{5,10,15,20,25}.
- Plot the time to update parameters of the network for different number of hiddenlayer neurons
- State the rationale for selecting the optimal number of hidden neurons

### Question 4
Find the optimal decay parameter for the 3-layer network designed in part (3).

- Plot the training errors against the number of epochs for the 3-layer network for different values of decay parameters in search space{0, 10−3, 10−6, 10−9, 10−12}.
- Plot the test accuracies against the different values of decay parameter.
- State the rationale for selecting the optimal decay parameter. 

### Question 5
After you are done with the 3-layer network, design a 4-layer network with two hiddenlayers, each consisting of 10 neurons with logistic activation functions, a batch size of 32 and decay parameter 10-6.
-  Plot the train and test accuracy of the 4-layer network.
-  Compare and comment on the performances on 3-layer and 4-layer networks.

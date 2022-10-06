# only_numpy_neural_network

This is a project where I've made a simple feedforward neural network (using only Numpy) which classifies MNIST dataset of handwritten digits.

Project is consisted of three .py files:

neural_functions.py - this is where all necessary functions are defined, such as activation functions (ReLu and Softmax), and one hot encoder for dataset labels

numpy_NN.py - this is where the actual network was defined, as a class with .fit and .predict methods

testing_numpy_NN.py - this is where the MNIST dataset was imported and the network was tested.

I decided to go with 2 hidden layers since that was a big step-up from one HL. Also, I made hidden layer size to be adjustable directly in .predict method, if anyone trying this out can tinker with it easier, but I made layers to have 100 neurons each in order to decrease the computational load on my old laptop.


Have fun

import numpy as np

def my_ReLu(x):
    return np.maximum(0,x)

def one_hot_encode(labels):
    output = np.zeros((labels.shape[0],10))
    for i in range(labels.shape[0]):
        output[i,np.int(labels[i])] = 1
    return output

def dReLu(x):
    return x > 0

def my_softmax(Z):
    normalized_Z = (Z.T - Z.max(axis = 1)).T
    numerator = np.exp(normalized_Z)
    denominator = np.exp(normalized_Z).sum(axis = 1)
    return (numerator.T/denominator).T


def accuracy(actual, predicted):
    arg_max_predicted = np.argmax(predicted,axis = 1)
    arg_max_actual = np.argmax(actual, axis = 1)
    return np.sum(arg_max_actual == arg_max_predicted)/arg_max_actual.size




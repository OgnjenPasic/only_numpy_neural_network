import numpy as np
#from matplotlib import pyplot as plt
import neural_functions as nf

# Importing MNIST dataset



""""""""" Training data """""""""""

#data available at:  http://yann.lecun.com/exdb/mnist/

f = open('MNISTdataset/Extracted/train-images-idx3-ubyte', 'rb')
f.read(16)
image_size = 28
num_images = 60000
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

X_train = data.reshape(num_images,784) # no preprocessing

# Import labels
labels = open('MNISTdataset/Extracted/train-labels-idx1-ubyte', 'rb')
labels.read(8)
buf2 = labels.read(num_images)
labels = np.frombuffer(buf2, dtype = np.uint8).astype(np.float32)

# One hot encoding is done for Y train
Y_train = nf.one_hot_encode(labels) 

m = Y_train.size


"""""""""  Testing data  """""""""


g = open('MNISTdataset/Extracted/t10k-images-idx3-ubyte', 'rb')
g.read(16)
num_images_test = 10000

buf3 = g.read(image_size * image_size * num_images_test)
data_test = np.frombuffer(buf3, dtype=np.uint8).astype(np.float32)
data_test = data_test.reshape(num_images_test, image_size, image_size, 1)


labels_test = open('MNISTdataset/Extracted/t10k-labels-idx1-ubyte', 'rb')
labels_test.read(8)

buf4 = labels_test.read(60000)
labels_test = np.frombuffer(buf4, dtype = np.uint8).astype(np.float32)

Y_test = nf.one_hot_encode(labels_test)

X_test = data_test.reshape(10000,784)

#%%

""""""""" Training the network """""""""

from NumpyNNClass import OnlyNumpyNeuralNetwork

my_net = OnlyNumpyNeuralNetwork()

my_net.fit(X_train, Y_train, learning_rate=0.01)


#%%


""""""""" Testing the accuracy on unseen data """""""""

my_net.predict(X_test, Y_test)








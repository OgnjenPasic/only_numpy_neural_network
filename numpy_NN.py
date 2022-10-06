import numpy as np
import neural_functions as nf


class OnlyNumpyNeuralNetwork:
    
    
    def __init__(self, HL1_size = 100, HL2_size = 100,input_size = 784,output_size = 10):
        
        self.W0 = np.random.rand(input_size,HL1_size)
        self.b0 = np.random.rand(HL1_size)
        self.W1 = np.random.rand(HL1_size,HL2_size)
        self.b1 = np.random.rand(HL2_size)
        self.W2 = np.random.rand(HL2_size,output_size)
        self.b2 = np.random.rand(output_size)

    
    def fit(self, X_train, Y_train, learning_rate = 0.1,n_iterations = 2000, print_results = True):
        
        """"""""" Forward propagation """""""""
        
        Z0 = X_train.dot(self.W0) + self.b0
        Y0 = nf.my_ReLu(Z0)
        Z1 = Y0.dot(self.W1) + self.b1
        Y1 = nf.my_ReLu(Z1)
        Z2 = Y1.dot(self.W2) + self.b2
        Y_predicted = nf.my_softmax(Z2) 

        m = X_train.shape[0]
        
        """"""""" Backpropagation """""""""
        
        for i in range(n_iterations):
            
            if (print_results and i % 2 == 0):
                print('Iteration', i, ', Accuracy: ',nf.accuracy(Y_train, Y_predicted))
                
                #x_axis.append(i)
                #y_axis.append(nf.accuracy(Y_train, Y_predicted))

            dZ2 = Y_train - Y_predicted
            dW2 = -(2/m)*Y1.T.dot(dZ2)
            db2 = -(2/m)*np.sum(dZ2,axis = 0)
            dY1 = -(2/m)*dZ2.dot(self.W2.T)
            dZ1 = -(2/m)*dY1*nf.dReLu(Z1)
            db1 = -(2/m)*np.sum(dZ1, axis = 0)
            dW1 = -(2/m)*Y0.T.dot(dZ1)
            dY0 = -(2/m)*dZ1.dot(self.W1.T)
            dZ0 = -(2/m)*dY0*nf.my_ReLu(Z0)
            db0 = -(2/m)*np.sum(dZ0, axis = 0)
            dW0 = -(2/m)*X_train.T.dot(dZ0)
            
            self.W0 -= learning_rate*dW0
            self.b0 -= learning_rate*db0
            self.W1 -= learning_rate*dW1
            self.b1 -= learning_rate*db1
            self.W2 -= learning_rate*dW2
            self.b2 -= learning_rate*db2
            
            Z0 = X_train.dot(self.W0) + self.b0
            Y0 = nf.my_ReLu(Z0)
            Z1 = Y0.dot(self.W1) + self.b1
            Y1 = nf.my_ReLu(Z1)
            Z2 = Y1.dot(self.W2) + self.b2
            Y_predicted = nf.my_softmax(Z2) 
         
    
    def predict(self, X_test, Y_test):
        
        Z0 = X_test.dot(self.W0) + self.b0
        Y0 = nf.my_ReLu(Z0)
        Z1 = Y0.dot(self.W1) + self.b1
        Y1 = nf.my_ReLu(Z1)
        Z2 = Y1.dot(self.W2) + self.b2
        Y_output = nf.my_softmax(Z2) 
        
        print('Test error:', 1 - np.sum(Y_output == Y_test)/Y_test.size)
                
    
    
    
    
    
    
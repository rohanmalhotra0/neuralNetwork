import numpy as np
from activation import softMax, relu, sigmoid, tanh
import forward as nn
import train as nn_train

class BackwardLayer:
    def __init__(self, forward_layer, dA):
        self.forward_layer = forward_layer
        self.dA = dA  # Gradient of the loss with respect to the activation output
        self.X = forward_layer.X
        self.weights = forward_layer.weights
        self.biases = forward_layer.biases
        self.activation = forward_layer.activation
        self.A = forward_layer.A  # Activation output from the forward pass
        
    def backward(self):
        m = self.X.shape[0]  # Number of examples
        
        # Compute dZ based on the activation function
        if self.activation == 'relu':
            dZ = self.dA * (self.A > 0)
        elif self.activation == 'sigmoid':
            dZ = self.dA * self.A * (1 - self.A)
        elif self.activation == 'tanh':
            dZ = self.dA * (1 - np.power(self.A, 2))
        elif self.activation == 'softmax':
            # For softmax, assuming dA is already the gradient w.r.t. Z
            dZ = self.dA
        else:
            raise ValueError("Unsupported activation function")
        
        # Compute gradients
        dW = np.dot(self.X.T, dZ) / m
        db = np.sum(dZ, axis=0) / m
        dX = np.dot(dZ, self.weights.T)
        
        return dX, dW, db
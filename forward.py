import np as np
from activation import softMax, relu, sigmoid, tanh

class ForwardLayer:
    
    def __init__(X, weights, biases, activation):
        self.X = X
        self.weights = weights
        self.biases = biases
        self.activation = activation
        
        
    def forward(self):
        z = np.dot(self.X, self.weights) + self.biases
        if self.activation == 'softmax':
            self.A = softMax(z)
        elif self.activation == 'relu':
            self.A = relu(z)
        elif self.activation == 'sigmoid':
            self.A = sigmoid(z)
        elif self.activation == 'tanh':
            self.A = tanh(z)
        return self.A

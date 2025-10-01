import numpy as np
from activation import softMax, relu, sigmoid, tanh
import forward as nn
import train as nn_train
import backPropagation as nn_bp

class gradientDescent:
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def update_parameters(self, layer, dW, db):
        layer.weights -= self.learning_rate * dW
        layer.biases -= self.learning_rate * db
        return layer
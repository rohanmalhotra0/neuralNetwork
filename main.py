import np as np
from activation import softMax, relu, sigmoid, tanh
import forward as nn
import train as nn_train
import backPropagation as nn_bp
import gradient as nn_gd

def main():
    
    for epoch in range(10):
        # Example data
        X = np.random.rand(10, 5)  # 10 samples, 5 features
        weights = np.random.rand(5, 3)  # 5 input features, 3 output neurons
        biases = np.random.rand(3)  # 3 output neurons
        
        # Create a forward layer
        layer = nn.ForwardLayer(X, weights, biases, activation='relu')
        layer = nn_bp.BackwardLayer(layer, dA=np.random.rand(10, 3))
        layer = nn_gd.gradientDescent(learning_rate=0.01)
        
        
        
        # Perform forward pass
        A = layer.forward()
        # Update parameters
        A = layer.gradientDescent.update_parameters(layer, dW=np.random.rand(5, 3), db=np.random.rand(3))
        # Perform backward pass
        A = layer.backward()
        #Repeat for a few epochs
        print(f"Epoch {epoch+1} completed.")
    
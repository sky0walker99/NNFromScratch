import numpy as np
from typing import Any
from src.layers.layer import Layer

class Dense(Layer):
    """
    A fully connected (dense) layer in a neural network.

    Initialization:
    1. When loading a trained model,saved weights and biases are used.
    2. For a new model, weights are randomly initialized in a small range for stable training. 

    Parameters:
    ---------
    n_inputs : int 
        Number of input featurs (or no of neurons in the previous layer).
    
    n_neurons : int 
        Number of neurons in this layer.

    seed : Any used to reproduce randomness in numpy
    
    Example:
    -------
    >>>>
    #Input data ( X ) with 3 samples and 4 features(inputs) per sample  - shape(3,4)
    X = np.random.randn(3,4)
    layer1 = Dense_layer(4,3)
    layer2 = Dense_layer(3,5)
    layer1.forward(X)
    layer2.forward(layer1.output)
    output = layer2.output
    print(output)
    """
    def __init__(self,n_inputs : int  , n_neurons : int, seed:Any=None):
        super().__init__()
        # initiating the parent class
        
        self.weights = 0.1*np.random.randn(n_inputs , n_neurons)
        self.biases = np.zeros((1,n_neurons))

        if seed is not None:
            np.random.seed(seed)
            
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
        return self.output

if __name__ == "__main__":
    X = np.random.randn(3,4)
    layer1 = Dense(4,3)
    layer2 = Dense(3,5)
    layer1.forward(X)

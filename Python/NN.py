import numpy as np

np.random.seed(0)

# input data ( X )with 3 samples
X = [[1,2.3,-2.6,3.4],
    [2.1,3.1,5.3,6.2],
    [3.2,1.4,-1.1,-2.1]] 


class Dense_layer:
    """
    A fully connected (dense) layer in a neural network.


    Initialization:

    1. When loading a trained model,saved weights and biases are used.
    2. For a new model, weights are randomly initialized in a small range for stable training. 

    """
    def __init__(self,n_inputs , n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs , n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases


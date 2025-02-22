import numpy as np
import math

e = math.e

class Activation:
    """
    # Activation

    Activation is an abstract class that is used to represent an activation function.

    Activation class implements all the required method as a template for an activation function.
    """
    def forward(self, x):
        raise NotImplementedError("Formward should be implemented by the child")
    
class ReLU(Activation):
    """
    ReLU (Rectified Linear Unit) is an activation function used in neural networks to introduce non-linearity into the model.
    
    """
    def forward(self,x):
        self.output = np.maximum(0,x)
        return self.output
    
class SoftMax(Activation):
    """
    1. Exponentiate the output values
    2. Then normalise them by dividing with the output of other neurons in layer.
    Softmax = Exponentiation + Normalisation
    Input -> Exponentiate -> Normalise -> Output
  
    """
    def forward(self,x):
        exp_values = np.exp(x - np.max(x,keepdims=True,axis=1) )
        normalised_score = exp_values / np.sum(exp_values,keepdims=True,axis=1)
        self.output = normalised_score
        return self.output 
    
class Sigmoid(Activation):
    """
    Sigmoid function is an activation function used in neural networks to introduce non-linearity into the model.
    Output in the range (0, 1)
    """

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
class Tanh(Activation):
    """
    Hyperbolic tangent function is an activation function used in neural networks to introduce non-linearity into the model.
    Output in the range (-1, 1)
    """
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    
class ELU(Activation):
    """
    Exponential Linear Unit (ELU) is an activation function used in neural networks to introduce non-linearity into the model.
    Output in the range (0, infinity)
    """
    def __init__(self, alpha:float):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        self.output = np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))
        return self.output
        
import numpy as np
import math

e = math.e

class ReLU:
    """
    ReLU (Rectified Linear Unit) is an activation function used in neural networks to introduce non-linearity into the model.
    
    """
    def forward(self,x):
        self.output = np.maximum(0,x)
        return self.output
    
class SoftMax:
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
        
import numpy as np
import math
from abc import ABC , abstractmethod


class Activation(ABC):
    """
    Activation is an abstract class that is used to represent an activation function.
    Activation class implements all the required method as a template for an activation function.
    
    """
    def __init__(self):
        self.output = None
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the activation function.
        
        Parameters:
        --------
        x : numpy.ndarray
            Input to the activation function
            
        Returns:
        --------
        numpy.ndarray
            Output after applying the activation function   
        """
        pass
    
    @abstractmethod
    def backward(self,grad_output):
        """
        Computes the gradient of the activation function.
        
        """
        pass

class ReLU(Activation):
    """
    ReLU (Rectified Linear Unit)
  
    
    ReLU(x)=max(0,x)
    
    if x is less than or equal to 0 then output is 0,
    else if x > 0  then output is x.
    
    Output in the range: [0, infinity)
    """
    
    def forward(self,x):
        self.input = x
        self.output = np.maximum(0,x)
        return self.output
    
class LeakyReLU(Activation):
    """
    Leaky ReLU activation function.
    
    For x > 0: LeakyReLU(x) = x
    For x ≤ 0: LeakyReLU(x) = α * x
    
    Where α is a small positive constant (typically 0.01).
    
    Output range: (-∞, ∞)
    """
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        if alpha <= 0:
            raise ValueError("Alpha must be positive")
        self.alpha = alpha
    
    def forward(self, x):
        self.input = x
        self.output = np.where(x >= 0, x,self.alpha * x)
        return self.output
        
class SoftMax(Activation):
    """
    Softmax activation function.
    
    1. Exponentiate the output values
    2. Then normalise them by dividing with the output of other neurons in layer.
    Softmax = Exponentiation + Normalisation
    Input -> Exponentiate -> Normalise -> Output
    
    Parameters
    ----------
    Input : Expects a 2D array (batch_size, num_classes) as input
    
    Used primarily in the output layer for multi-class classification.
    Converts logits to probabilities that sum to 1 across classes.


    """
    def forward(self,x):
        if x.ndim != 2:
            raise ValueError("Softmax expects a 2D array (batch_size, num_classes)")
        exp_values = np.exp(x - np.max(x,axis=1,keepdims=True) )
        normalised_score = exp_values / np.sum(exp_values,keepdims=True,axis=1)
        self.output = normalised_score
        return self.output 
    
class Sigmoid(Activation):
    """
    Sigmoid function transforms any input into a value between 0 and 1, creating an s shaped curve.
    
    Outputs in the range (0, 1)
    Centered at 0.5 when x = 0
    
    Often used for binary classification or in gates of LSTM.
    """
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
class Tanh(Activation):
    """
    The hyperbolic tangent (tanh) activation function serves as an alternative to sigmoid activation function.
    Outputs in the range (-1, 1).
    
    The function is centered at 0 (when x = 0, tanh(x) = 0)
    
    Often used in hidden layers of neural networks and in RNNs.
    """
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    
class ELU(Activation):
    """
    The ELU is introduced to improve upon the limitations of older functions like ReLU and sigmoid.
    
    For 𝑥>0 , ELU behaves like ReLU. 
    For negative inputs (x<0), it produces a smooth, exponential curve that approaches 𝛼 as x becomes very negative, rather than cutting off at zero like ReLU.
    
    For x > 0: ELU(x) = x
    For x ≤ 0: ELU(x) = α * (e^x - 1)
    
    Output in the range (-alpha, infinity)
    
    Parameters:
    -----------
    input    :  The input to the function (numpy.ndarray) 
    alpha(α) :  A positive hyperparameter (typically set to 1) that controls the value to which the function saturates for negative inputs.
    
    """
    def __init__(self, alpha: float = 1):
        super().__init__()
        if alpha <= 0:
            raise ValueError("Alpha must be positive")
        self.alpha = alpha
        
    def forward(self, x):
        self.input = x
        self.output = np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))
        return self.output
        
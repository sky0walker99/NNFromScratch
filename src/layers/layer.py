from abc import ABC, abstractmethod
import numpy as np

class Layer:
    """
    Abstract base class for neural network layers.
    This class defines the interface that all layer must follow.

    Attributes
    -------   
    inputs : np.ndarray
        The input datat to the layer.
    output : np.ndarray
        The output from the layer after the forward pass.
    grad_input  : np.ndarray
        The gradient of loss with respect to the inputs. (∂L/∂x)
        
    
    """
    def __init__(self):
        """
        Sets default values for inputs , output , gradients.
        """
        self.inputs = None
        self.output = None
        self.grad_input = None

    @abstractmethod
    def forward(self, x):
        """
        forward method is used to transform a given input of a layer to output.
        
        Parameters:
        ----------
        inputs : numpy.ndarray
            The input data to the layer
        Returns:
        -------
        numpy.ndarray
            The output of the layer after applying its transformation
        """
        pass
    
    
    @abstractmethod
    def backward(self, grad_output):
        """
        Perform the backward pass calculation.
        
        It calculates the gradient of the loss with respect to the layer's inputs
        based on the gradient with respect to the layer's outputs.
        
        Parameters:
        ----------
        grad_output : np.ndarray
            The gradient of the loss with respect to the layer's outputs.
            Recieved from the next layer in the network.

        Returns:
        -------
        np.ndarray
            The gradient of the loss with respect to the layer's inputs
            This will be used by the previous layer in backpropagation.
        """
        pass
class Layer:
    def __init__(self):
        pass

    def forward(self, x):
        """
        forward method is used to transform a given input of a layer to output.

        Parameters:
        x (numpy.ndarray): Input to the layer.
        """
        raise NotImplementedError("Subclasses must implement this method")
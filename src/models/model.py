from src.layers.layer import Layer

class Model:
    def __init__(self, layers:list[Layer] = []):
        self.layers:list[Layer] = layers

    def addLayer(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
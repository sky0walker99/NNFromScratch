from src.models import Model
from src.layers import Dense
import numpy as np
import unittest

class TestModel(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.model = Model()

    def test_add_layer(self):
        layers = [
            Dense(50, 10),
            Dense(10, 5),
            Dense(5, 1)
        ]

        for i, layer in enumerate(layers):
            self.model.addLayer(layer)
            self.assertEqual(self.model.layers[i], layer)
            self.assertEqual(len(self.model.layers), i+1)

    def test_output(self):
        layers = [
            Dense(50, 10),
            Dense(10, 5),
            Dense(5, 1)
        ]

        self.model = Model(layers)

        input_data = np.random.randn(30, 50)
        
        output = self.model.forward(input_data)

        self.assertAlmostEqual(output.shape, (30, 1))
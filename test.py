import unittest
from Python.NN import Dense_layer
import numpy as np

class DenseTest(unittest.TestCase):
    layer = Dense_layer(50, 10)
    # layer to be tested  

    def test_init(self):
        self.assertEqual(self.layer.weights.shape, (50, 10), f"The layer shape should be (50, 10) and not {self.layer.weights.shape}")

    def test_output(self):
        self.layer.forward(np.random.randn(30, 50))
        output = self.layer.output
        self.assertEqual(output.shape, (30, 10), f"The output shape should be (30, 10) and not {output.shape}")

    def test_error(self):
        self.assertRaises(ValueError, self.layer.forward, np.random.randn(30, 30))


if __name__ == "__main__":
    unittest.main()
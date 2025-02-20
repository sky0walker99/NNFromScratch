import unittest
from Python.NN import Dense_layer
import numpy as np

class DenseTest(unittest.TestCase):
    def setUp(self):
        self.layer = Dense_layer(50, 10)
        # layer to be tested  

    def test_init(self):
        self.assertEqual(self.layer.weights.shape, (50, 10), f"The weight shape should be (50, 10) and not {self.layer.weights.shape}")
        self.assertEqual(self.layer.biases.shape,(1,10),f"The bias shape should be (1,10) and not {self.layer.biases.shape}" )

    def test_forward_shape(self):
        self.layer.forward(np.random.randn(30, 50))
        output = self.layer.output
        self.assertEqual(output.shape, (30, 10), f"The output shape should be (30, 10) and not {output.shape}")

    def test_forward_computation(self):
        np.random.seed(3) 
        test_input  = np.random.randn(5, 50)
        self.layer.forward(test_input)
        expected_output  = np.dot(test_input ,self.layer.weights) + self.layer.biases
        self.layer.forward(test_input)
        np.testing.assert_array_equal(self.layer.output,expected_output, f"The output array should be {expected_output} and not {self.layer.output}")
        
    def test_error(self):    
        self.assertRaises(ValueError, self.layer.forward, np.random.randn(30, 30))


if __name__ == "__main__":
    unittest.main()

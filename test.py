import unittest
from Python.NN import Dense_layer
from Python.activation import ReLU, SoftMax, Sigmoid, ELU, Tanh
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


class TestActivationFunctions(unittest.TestCase):
    def test_relu(self):
        relu = ReLU()
        x = np.array([-2, -1, 0, 1, 2])
        expected = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_equal(relu.forward(x), expected)
    
    def test_softmax(self):
        softmax = SoftMax()
        x = np.array([[1, 2, 3], [1, 2, 3]])
        output = softmax.forward(x)
        self.assertTrue(np.allclose(np.sum(output, axis=1), np.ones(output.shape[0])))
    
    def test_sigmoid(self):
        sigmoid = Sigmoid()
        x = np.array([-1, 0, 1])
        expected = 1 / (1 + np.exp(-x))
        np.testing.assert_array_almost_equal(sigmoid.forward(x), expected)
    
    def test_tanh(self):
        tanh = Tanh()
        x = np.array([-1, 0, 1])
        expected = np.tanh(x)
        np.testing.assert_array_almost_equal(tanh.forward(x), expected)
    
    def test_elu(self):
        elu = ELU(alpha=1.0)
        x = np.array([-1, 0, 1])
        expected = np.where(x >= 0, x, 1.0 * (np.exp(x) - 1))
        np.testing.assert_array_almost_equal(elu.forward(x), expected)



if __name__ == "__main__":
    unittest.main()
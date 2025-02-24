import unittest
from src.activation import ReLU, SoftMax, Sigmoid, ELU, Tanh
import numpy as np

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
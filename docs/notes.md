# Neural Network Implementation Notes

Initialization:

1. When loading a trained model, the saved weights and biases are used.
2. For a new model, weights are randomly initialized in a small range for stable training. 

- randn uses a gaussian distribution bounded aroud zero but some values would be bigger so the weights are multiplied by 0.1.
- Scaling by 0.1 ensures smaller initial values, preventing large activations.

biases are initialized to 0 because they are learned during training. while in some scenarios setting them to zero would not be apt.

Activation Function
Why use one? why not just use weights and biases without activation function?
Without a non linear activation function the output will always be linear and we cant use any non-linear data and train the NN. The Activation function introdues non-linearity into the network.

The activation function is applied to the (inputs*weights + bias ).
- Z=XW+b
- A=Activation(Z)

1. RelU(if x is less than or equal to 0 then y is 0  , else if x > 0  y will be equal to x) Function - (Rectified Linear Unit)
-ReLU(x)=max(0,x)
-Less Vanishing gradient problems compared to sigmoid function.
by tweaking by the bias the activation func can be offsetted.

2. Softmax Activation Function
-
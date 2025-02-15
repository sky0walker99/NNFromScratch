# Neural Network Implementation Notes

Initialization:

1. When loading a trained model, the saved weights and biases are used.
2. For a new model, weights are randomly initialized in a small range for stable training. 

- randn uses a gaussian distribution bounded aroud zero but some values would be bigger so the weights are multiplied by 0.1.
- Scaling by 0.1 ensures smaller initial values, preventing large activations.

biases are initialized to 0 because they are learned during training.while in some scenarios setting them to zero would not be apt.

 
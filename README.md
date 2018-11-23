# Neural Network from Scratch 
Lightweight, fully operational, configurable neural network implemented with
only NumPy library use for vector operations. Documentation for
trainer, model, and data loader included within modules.
Performance evaluated on MNIST data set with 97% testing, 98% training accuracy 
obtained through preconfigured hyperparameters in main module. The 
Neuralnetwork class and trainer contain the following configurable
specifications and features: 
- Number of layers and units per layer 
- Back pass backpropagation, forward pass cross entropy loss and predictions
- Hidden layer activation functions (ReLU, sigmoid, tanh)
- Standard Momentum, adjustable gamma
- Minibatch SGD, adjustable batch size
- Training and validation loss plotting
- L2 Regularization, adjustable penalty 
- Early stopping with lowest validation loss model restored 
- ... and more! See module for documentation. 
This module can be imported and used for any single label, multiclass 
classification problem. It's documentation is easy to parse and its 
hyperparameters are rapidly configurable. Operations are optimized through
vectorization of minibatches and use of NumPy's matrix multiplication libraries.


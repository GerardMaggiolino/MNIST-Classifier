''' 
Contains Neuralnetwork class and methods to operate on the model.

Classes: 
  Neuralnetwork
    Trainable and modifiable neural network implemented in NumPy.

:q
:q
Functions: 
  load_data
    Reads data and returns np arrays of input and one-hot labels. 
  trainer
    Trains a model using custom config options and data.
  test
    Returns percentage of accurately classified images from test set.

Variables: 
  conf_default
    Dictionary of default values for Neuralnetwork model and training  

Configuration dictionary necessary for model initialization and 
training. Specifications of config beneath. 

  config['layer_specs']: [784, 50, 50, 10]
    List of integers, denoting input & output neurons per layer. 

  config['activation'] = 'sigmoid' 
    Hidden layer activations, as 'ReLU', 'tanh', or 'sigmoid'.

  config['batch_size'] = 128
    Number of training examples per batch of SGD.

  config['epochs'] = 100
    Number of full pass through of training set.

  config['early_stop'] = True  
    Enable early stopping, terminating training early when loss 
    increases on validation set. 

  config['early_stop_epoch'] = 5  
    Number of epochs for which loss increases during validation before
    training is terminated and the weights of the lowest loss epoch
    are restored. 

  config['L2_penalty'] = 0.0001
    L2 regularization term. 0 specifies no regularization applied.

  config['momentum'] = True  
    Enable standard momentum.

  config['momentum_gamma'] = 0.9
    Decay of momentum. 

  config['learning_rate'] = 0.0001
    Learning rate of gradient descent algorithm.

  config['plot'] = True
    Enable training and validation loss recording.

  config['plot_lists'] = [[], []]
    Stores lists to training and validation loss.
'''

import numpy as np
import pickle
import gzip

# Holds default specifications of trainer and model 
conf_default = { 
  'layer_specs': [784, 50, 50, 10],
  'activation': 'sigmoid',
  'batch_size': 128,
  'epochs': 100,
  'early_stop': True,
  'early_stop_epoch': 5,
  'L2_penalty': 0.0001,
  'momentum': True,
  'momentum_gamma': 0.9,
  'learning_rate': 0.0001,
  'plot': True, 
  'plot_lists': [[], []]
}

class Neuralnetwork():
  '''
  Trainable and modifiable neural network implemented in NumPy. 
  
  Softmax output with custom hidden layer activations and 
  specifications. Configuration defaults and options detailed below:
  
  config['layer_specs']: [784, 100, 100, 10]
    List of integers, denoting input & output neurons per layer. 

  config['activation'] = 'sigmoid' 
    Hidden layer activations, as 'ReLU', 'tanh', or 'sigmoid'.
  '''
  def __init__(self, config={}):
    '''
    Creates Layer / Activation class pairs according to config, with the last
    Layer class having no Activation class.
    '''
    # Parse the config, set unspecified variables to default values
    for k, v in conf_default.items(): 
      config[k] = _conf_check(config, k, v) 

    self.layers = []      # Holds activations and layers
    self.y = None         # Output vector of model 
    self.targets = None   # Targets of model 

    for i in range(len(config['layer_specs']) - 1):
      self.layers.append(_Layer(\
        config['layer_specs'][i], config['layer_specs'][i+1]))
      if i < len(config['layer_specs']) - 2:
        self.layers.append(_Activation(config['activation']))  
    
  def forward_pass(self, x, targets=None):
    '''
    Performs one forward pass on the network, returning loss and 
    classification for that example
    '''
    self.targets = targets

    # Take input of training example for first Layer
    raw_output = self.layers[0].forward_pass(x)

    # Pass over all Activation layer
    for i in range(1, len(self.layers) - 1, 2):
      activations = self.layers[i].forward_pass(raw_output)
      raw_output = self.layers[i + 1].forward_pass(activations)

    # Apply output function (softmax)
    self.y = _softmax(raw_output)
  
    loss = None if targets is None else self.loss_func(self.y, self.targets)

    return loss, self.y

  def loss_func(self, logits, targets):
    '''
    Returns cross entropy loss between targets and logits.
    '''
    return -np.sum(targets * np.log(logits + 1e-10))
    
  def backward_pass(self):
    '''
    Performs one backwards pass on the network, storing gradients
    '''
    # Process output layer
    deltas = self.targets - self.y 
    weighted_deltas = self.layers[len(self.layers) - 1].backward_pass(deltas)

    # Loop over all hidden layers in reversed order
    for i in range(len(self.layers) - 2, -1, -2):
      deltas = self.layers[i].backward_pass(weighted_deltas)
      weighted_deltas = self.layers[i - 1].backward_pass(deltas)


def load_data(fname):
  '''
  Reads data and converts images to np array with parallel one-hot 
  encoded labels. Set up for use with MNIST data set.

  Returns images, labels as 2d numpy arrays.
  '''
  with gzip.open(fname, 'rb') as f: 
    data = pickle.load(f)

  images = [np.split(n, [784])[0] for n in data]
  labels = [np.split(n, [784])[1] for n in data]

  # Transform labels to one-hot arrays
  for i in range(len(labels)): 
    one_hot = np.zeros(10)
    one_hot[int(labels[i])] = 1
    labels[i] = one_hot

  return np.array(images), np.array(labels)


def trainer(model, X_train, y_train, X_valid, y_valid, config={}):
  '''
  Trains the model using config options to decide on model parameters.

  Validation performed on passed validation set. Input expected as 
  1d np arrays in X_train with outer indices mapping to individual 
  examples. Outer indices of y_train must be parallel to X_train, 
  corresponding to one hot encoded labels for an input.

  Configuration options and default beneath: 

  config['batch_size'] = 128
    Number of training examples per batch of SGD.

  config['epochs'] = 100
    Number of full pass through of training set.

  config['early_stop'] = True  
    Enable early stopping, terminating training early when loss 
    increases on validation set. 

  config['early_stop_epoch'] = 5  
    Number of epochs for which loss increases during validation before
    training is terminated and the weights of the lowest loss epoch
    are restored. 

  config['L2_penalty'] = 0.0001
    L2 regularization term. 0 specifies no regularization applied.

  config['momentum'] = True  
    Enable standard momentum.

  config['momentum_gamma'] = 0.9
    Decay of momentum. 

  config['learning_rate'] = 0.0001
    Learning rate of gradient descent algorithm.

  config['plot'] = True 
    Enable training and validation loss recording.
      
  config['plot_lists'] = [[], []]
    Stores lists to training and validation loss.
  '''
  # Parse the config, set unspecified variables to default values
  for k, v in conf_default.items(): 
    config[k] = _conf_check(config, k, v) 

  # Set up early stop variables if specified in config
  if config['early_stop'] or config['plot']:
    # Holds validation loss at each epoch 
    cross_loss = [0] * config['epochs']
    # Lowest loss epoch
    best_epoch = 0
    if config['early_stop']:
      # Holds lowest loss weights
      best_w = [np.zeros_like(model.layers[i].w) \
        for i in range(0, len(model.layers), 2)]
      best_b = [np.zeros_like(model.layers[i].b) \
        for i in range(0, len(model.layers), 2)]
    if config['plot']:
      print("Epoch | Loss") 
      print("------------")


  #  Set up velocity if momentum specified
  if config['momentum']:
    # Zero arrays for each weight array per model's Layer
    velocity_w = [np.zeros_like(model.layers[i].w) \
      for i in range(0, len(model.layers), 2)]
    velocity_b = [np.zeros_like(model.layers[i].b) \
      for i in range(0, len(model.layers), 2)]

  # Configure batch size and number of batches 
  b_size = config['batch_size']
  num_batches = int(X_train.shape[0] / b_size) 
  num_val_batches = int(X_valid.shape[0] / b_size)

  # Cover training set for specified epochs
  for epoch in range(config['epochs']):

    # Add plotting training data
    if config['plot']:
      config['plot_lists'][0].append(0)

    # Shuffle all data between batches for each epoch
    state = np.random.get_state()
    np.random.shuffle(X_train)
    np.random.set_state(state)
    np.random.shuffle(y_train)

    # Train with mini-batches over the entire dataset
    for batch in range(num_batches):
      loss, _ = model.forward_pass( \
        X_train[(batch * b_size):((batch + 1) * b_size)], \
        y_train[(batch * b_size):((batch + 1) * b_size)])
      model.backward_pass()

      if config['plot']: 
        config['plot_lists'][0][epoch] += loss

      # Gradient descent for each Layer layer
      for i in range(0, len(model.layers), 2):
        grad_w = config['learning_rate'] * model.layers[i].d_w
        grad_b = config['learning_rate'] * model.layers[i].d_b
        # Add momentum to gradient, update velocity vector
        if config['momentum']:
          grad_w = velocity_w[i//2] = config['momentum_gamma'] * \
            velocity_w[i//2] + grad_w 
          grad_b = velocity_b[i//2] = config['momentum_gamma'] * \
            velocity_b[i//2] + grad_b 
        # Add regularization constant  
        grad_w -= config['L2_penalty'] * model.layers[i].w
        grad_b -= config['L2_penalty'] * model.layers[i].b
        # Apply gradient 
        model.layers[i].w += grad_w
        model.layers[i].b += grad_b

    if config['plot']: 
      config['plot_lists'][0][epoch] /= num_batches * b_size

    # Perform early stop and validation if flag on 
    if config['early_stop'] or config['plot']:
      # Perform validation, saving sum of loss for each example
      for batch in range(num_val_batches):
        # Save loss for a single batch, loop over batches
        loss, _ =  model.forward_pass(X_valid[(batch * b_size): \
          ((batch + 1) * b_size)], y_valid[(batch * b_size): \
          ((batch + 1) * b_size)])
        cross_loss[epoch] += loss 

      # Save and print if plotting 
      if config['plot']: 
        config['plot_lists'][1].append(cross_loss[epoch] \
          / (num_val_batches * b_size))
        print(f'{epoch + 1}:\t{np.round(config["plot_lists"][1][epoch], 4)}')

      # Current epoch has better weights, save these weights
      if config['early_stop']:
        if cross_loss[epoch] <= cross_loss[best_epoch]:
          best_epoch = epoch
          # Loop over layers, performing deep copy of weights
          for i in range(0, len(model.layers), 2):
            best_w[i//2][:] = model.layers[i].w
            best_b[i//2][:] = model.layers[i].b
        # Current epoch has worse weights, higher loss than best epoch
        elif best_epoch <= (epoch - config['early_stop_epoch']): 
          print("Early stop on epoch " + str(epoch + 1) + ", weights from "\
            + str(best_epoch + 1) + " saved in model.")
          # Change model weights to best weights
          for i in range(0, len(model.layers), 2):
            model.layers[i].w[:] = best_w[i//2]
            model.layers[i].b[:] = best_b[i//2]
          return


def test(model, X_test, y_test, config):
  '''
  Returns percentage of accurately classified images from test set.
  '''
  accuracy = 0
  b_size = config['batch_size']
  num_batches = int(X_test.shape[0]/b_size)

  # Loop over each batch 
  for batch in range(num_batches):
    _, prediction = model.forward_pass(X_test[(batch * b_size): \
      (batch + 1) * b_size])
    # Check for accuracy in retrieved batch
    for i in range(prediction.shape[0]):
      if np.argmax(prediction[i]) == np.argmax(y_test[i + batch * b_size]):
        accuracy += 1

  return accuracy / X_test.shape[0]


def _softmax(x):
  '''
  Performs softmax on input array, output array of distribution 
  '''
  # Softmax for each training example in batch 
  for i in range(x.shape[0]):
    x[i] = np.exp(x[i] - np.amax(x[i])) / np.sum(np.exp(x[i] - np.amax(x[i])))
  return x


class _Activation:
  '''
  Activation layer storing computed gradients for backprop.
  '''
  implemented_activations = ['sigmoid, ReLU, tanh']

  def __init__(self, activation_type):
    # Assign activation type to sigmoid if not correctly specified
    self.activation_type = activation_type if activation_type in \
      self.implemented_activations else 'sigmoid' 
    # Holds Layer output, prior to activation
    self.x = None 
  
  def forward_pass(self, a):
    '''
    Takes in output of this layer prior to activations, returns
    final activated output of this layer
    '''
    # Save output of Layer prior to activation 
    self.x = a

    if self.activation_type == "sigmoid":
      return self.sigmoid(a)
    elif self.activation_type == "tanh":
      return self.tanh(a)
    elif self.activation_type == "ReLU":
      return self.ReLU(a)
  
  def backward_pass(self, delta):
    '''
    Takes in weighted deltas of upper layer, returns deltas for current
    layer.
    '''
    if self.activation_type == "sigmoid":
      grad = self.grad_sigmoid()
    elif self.activation_type == "tanh":
      grad = self.grad_tanh()
    elif self.activation_type == "ReLU":
      grad = self.grad_ReLU()
    
    return grad * delta
      
  def sigmoid(self, x):
    '''
    Returns applied sigmoid activation function to all elements of x
    '''
    output = 1/(1 + np.exp(np.multiply(x, -1)))
    return output

  def tanh(self, x):
    '''
    Returns applied tanh activation function to all elements of x
    '''
    output = np.tanh(x)
    return output

  def ReLU(self, x):
    '''
    Returns applied tanh activation function to all elements of x
    '''
    output = np.maximum(x, 0)
    return output

  def grad_sigmoid(self):
    '''
    Returns gradient of sigmoid activation function for backprop
    '''
    return (1 / (1 + np.exp(np.multiply(self.x, -1)))) * \
      (1 - (1 / (1 + np.exp(np.multiply(self.x, -1)))))


  def grad_tanh(self):
    '''
    Returns gradient of tanh activation function for backprop
    '''
    return 1 - np.square(np.tanh(self.x))

  def grad_ReLU(self):
    '''
    Returns gradient of ReLu activation function for backprop
    '''
    return (self.x > 0).astype(int)


class _Layer():
  '''
  Linear layer for Neuralnetwork class.
  '''
  def __init__(self, in_units, out_units):
    # (in_units) number of rows, (out_units) number of columns / weights
    self.w = np.random.randn(in_units, out_units) 
    # one row,  (out_unit) number of columns
    self.b = np.zeros((1, out_units))  # Bias

    # Use Xavier initialization (tanh optimized) 
    self.w *= np.sqrt(1/(in_units))

    self.x = None     # Input to linear layer
    self.a = None     # Weighted output of linear layer

    self.d_w = None   # Gradient w.r.t w 
    self.d_b = None   # Gradient w.r.t b

    self.count = 0

  def forward_pass(self, x):
    '''
    Forward pass through a layer, without activation function
    '''
    self.x = x
    self.a = np.matmul(self.x, self.w) + self.b
    return self.a
  
  def backward_pass(self, delta):
    '''
    Backprop through a layer, takes in delta for the current layer. 
    Saves gradient of weights / bias, returns weighted sum of current
    delta for the previous layer's delta calculation.
    '''
    # Weighted sum of current deltas for previous layer. 
    self.d_x = np.matmul(self.w, delta.T).T

    # Gradient of weights, current delta * input from previous layer
    self.d_w = np.matmul(self.x.T, delta)

    # Gradient of bias as delta
    self.d_b = np.matmul(np.ones(delta.shape[0]).T, delta)

    return self.d_x

def _conf_check(config, key, default_val): 
  '''
  Checks if config contains appropriate type for default val.
  
  If of appropriate type, returns config val, otherwise default val.
  '''
  if type(config.get(key)) is not type(default_val):
    return default_val
  elif type(default_val) is list and default_val is not []: 
    for item in config[key]:
      if type(item) is not type(default_val[0]):
        return default_val
  return config[key]

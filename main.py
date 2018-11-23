''' 
Module contains main function which trains and runs an MNIST classifier.

Documentation for Neuralnetwork in its respective module. Main displays
training and validation loss over training epochs, along with testing
and training set accuracy after training. 
'''

import matplotlib.pyplot as plt
import neuralnet

def main(): 
  ''' 
  Loads, trains, tests, and displays performance of NN on MNIST.
  '''
  dirname = 'data'
  train_data_fname = f'{dirname}/MNIST_train.pkl.gz'
  valid_data_fname = f'{dirname}/MNIST_valid.pkl.gz'
  test_data_fname = f'{dirname}/MNIST_test.pkl.gz'

  # See neuralnet for documentation on config
  config = {
    'batch_size': 1000, 
    'learning_rate': 0.0003,
    'activation': 'ReLU'
  } 

  # Load data sets and untrained model
  model = neuralnet.Neuralnetwork(config)
  
  X_train, y_train = neuralnet.load_data(train_data_fname)
  X_valid, y_valid = neuralnet.load_data(valid_data_fname)
  X_test, y_test = neuralnet.load_data(test_data_fname)

  # Train the model on data sets with validation sets
  neuralnet.trainer(model, X_train, y_train, X_valid, y_valid, config)

  # Test on testing set and training set
  test_acc = neuralnet.test(model, X_test, y_test, config)
  train_acc = neuralnet.test(model, X_train, y_train, config)
  print(f'Training Acc: {train_acc}')
  print(f'Testing Acc: {test_acc}')

  # Plot the training and validation loss
  plt.figure(1)
  epochs = [i for i in range(1, len(config['plot_lists'][0]) + 1)]
  line2, = plt.plot(epochs, config['plot_lists'][0], 'b', label="Training")
  line1, = plt.plot(epochs, config['plot_lists'][1], 'g', label="Validation")
  plt.legend(handles=[line1, line2])
  plt.xlabel('Number of Epochs')
  plt.ylabel('Normalized Loss')
  plt.title(f'Cross Entropy Loss over Epochs')
  plt.show()

if __name__ == "__main__":
  main()


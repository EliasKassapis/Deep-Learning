"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch
import matplotlib.pyplot as plt



dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  predictions = predictions.detach().numpy()
  targets = targets.detach().numpy()
  accuracy = (predictions.argmax(axis=1) == targets.argmax(axis=1)).mean()

  # raise NotImplementedError
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  #load data
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

  #hyperparameters
  eta = FLAGS.learning_rate
  eps = 1e-6 # convergence criterion
  max_steps = FLAGS.max_steps
  b_size = FLAGS.batch_size


  #load test data
  x_test = cifar10["test"].images
  y_test = cifar10["test"].labels
  y_test = torch.tensor(y_test, requires_grad=False).type(dtype).to(device) ################################################################################


  n_inputs = np.size(x_test,0)
  n_classes = np.size(y_test,1)
  v_size = np.size(x_test,1) * np.size(x_test,2) * np.size(x_test,3)

  x_test = x_test.reshape((n_inputs, v_size))
  x_test = torch.tensor(x_test, requires_grad=False).type(dtype).to(device) #################################################################################

  #initialize the MLP model
  model = MLP(n_inputs = v_size, n_hidden = dnn_hidden_units, n_classes = n_classes)
  get_loss = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=eta)
  # optimizer = torch.optim.Adam(model.parameters(), lr=eta)


  train_loss = []
  test_loss = []
  train_acc = []
  test_acc = []

  for epoch in range(max_steps):

    #get batch
    x, y = cifar10['train'].next_batch(b_size) # NEED TO MAKE THEM TENSORS
    y = torch.tensor(y).type(dtype).to(device) ############ Check removing grad=!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #stretch input images into vectors
    x = x.reshape(b_size, v_size)
    x = torch.tensor(x).type(dtype).to(device)

    #forward pass
    pred = model.forward(x)

    #get loss
    current_loss = get_loss(pred,y.argmax(dim=1)) # check this again 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    optimizer.zero_grad()

    #get loss gradient
    current_loss.backward()

    optimizer.step()


    if (epoch % FLAGS.eval_freq) == 0:

        c_loss = current_loss.data.item()
        train_loss.append(c_loss)

        current_train_acc = accuracy(pred, y)
        train_acc.append(current_train_acc)

        test_pred = model.forward(x_test)
        current_test_loss = get_loss(test_pred, y_test.argmax(dim=1))
        c_test_loss = current_test_loss.data.item()
        test_loss.append(c_test_loss)
        current_test_acc = accuracy(test_pred, y_test)
        test_acc.append(current_test_acc)

        print('\nEpoch ',epoch, '\n------------\nTraining Loss = ', round(c_loss,4), ', Train Accuracy = ', current_train_acc, '\nTest Loss = ', round(c_test_loss,4), ', Test Accuracy = ', current_test_acc)

        if epoch > 0 and abs(train_loss[(int(epoch/FLAGS.eval_freq))] - train_loss[int(epoch/FLAGS.eval_freq)-1]) < eps:
                break


  plot_graphs(train_loss, 'Training Loss', 'orange',
                test_loss, 'Test Loss', 'blue',
                title='Stochastic gradient descent',
                ylabel='Loss',
                xlabel='Epochs')

  plot_graphs(train_acc, 'Training Accuracy', 'darkorange',
                test_acc, 'Test Accuracy', 'darkred',
                title='Stochastic gradient descent',
                ylabel='Accuracy',
                xlabel='Epochs')

  #save results:
  path = "./results/pytorch results/results"
  np.save(path, train_loss)
  np.save(path, train_acc)
  np.save(path, test_loss)
  np.save(path, test_acc)


  # raise NotImplementedError
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def plot_graphs(*args, title=None, ylabel=None, xlabel=None):
    y = args[0::3]
    legends = args[1::3]
    colors = args[2::3]

    if title != None:
        plt.title(title)

    if xlabel != None:
        plt.xlabel(xlabel)

    if ylabel != None:
        plt.ylabel(ylabel)

    for i, current_y in enumerate(y):
        plt.plot(current_y, label=legends[i], color=colors[i])

    plt.legend()
    plt.show()

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()

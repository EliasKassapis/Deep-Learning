"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
import matplotlib.pyplot as plt



dtype = torch.FloatTensor
# device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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


  accuracy = (predictions.argmax(dim=1) == targets.argmax(dim=1)).type(dtype).mean()
  accuracy = accuracy.detach().data.cpu().item()

  # raise NotImplementedError
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

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

  #test_data
  x_test = cifar10["test"].images
  y_test = cifar10["test"].labels

  n_channels = np.size(x_test,1)
  n_classes = np.size(y_test,1)
  n_batches = np.size(x_test,0)//b_size


  #initialize the ConvNet model
  model = ConvNet(n_channels, n_classes)
  get_loss = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=eta)


  model.to(device)

  train_loss = []
  test_loss = []
  train_acc = []
  test_acc = []

  for step in range(max_steps):
    #get batch
    x, y = cifar10['train'].next_batch(b_size)
    x = torch.tensor(x).type(dtype).to(device)
    y = torch.tensor(y).type(dtype).to(device)


    #forward pass
    pred = model.forward(x)

    #get training loss
    current_loss = get_loss(pred,y.argmax(dim=1))
    optimizer.zero_grad()

    #get training loss gradient
    current_loss.backward()

    #get training accuracy
    current_train_acc = accuracy(pred, y)

    optimizer.step()

    #free memory up
    pred.detach()
    x.detach()
    y.detach()

    #select evaluation step
    if (step % FLAGS.eval_freq) == 0:

        c_loss = current_loss.data.item()
        train_loss.append(c_loss)
        train_acc.append(current_train_acc)

        c_test_loss = 0
        current_test_acc = 0

        #loop through test set in batches
        for test_batch in range(n_batches):
          #load test data
          x_test, y_test = cifar10['test'].next_batch(b_size)
          x_test = torch.tensor(x_test, requires_grad=False).type(dtype).to(device)
          y_test = torch.tensor(y_test, requires_grad=False).type(dtype).to(device)

          #get test batch results
          test_pred = model.forward(x_test)
          current_test_loss = get_loss(test_pred, y_test.argmax(dim=1))

          c_test_loss += current_test_loss.data.item()
          current_test_acc += accuracy(test_pred, y_test)

          #free memory up
          test_pred.detach()
          x_test.detach()
          y_test.detach()

        #get full test set results
        c_test_loss = c_test_loss/n_batches
        current_test_acc = current_test_acc/n_batches
        test_loss.append(c_test_loss)
        test_acc.append(current_test_acc)

        print('\nStep ',step, '\n------------\nTraining Loss = ', round(c_loss,4), ', Train Accuracy = ', current_train_acc, '\nTest Loss = ', round(c_test_loss,4), ', Test Accuracy = ', round(current_test_acc,4))

        if step > 0 and abs(test_loss[(int(step/FLAGS.eval_freq))] - test_loss[int(step/FLAGS.eval_freq)-1]) < eps:
                break


  plot_graphs(train_loss, 'Training Loss', 'orange',
                test_loss, 'Test Loss', 'blue',
                title='Adams optimization',
                ylabel='Loss',
                xlabel='Steps')

  plot_graphs(train_acc, 'Training Accuracy', 'darkorange',
                test_acc, 'Test Accuracy', 'darkred',
                title='Adamns optimization',
                ylabel='Accuracy',
                xlabel='Steps')

  #save results:
  np.save('train_loss', train_loss)
  np.save('train_acc', train_acc)
  np.save('test_loss', test_loss)
  np.save('test_acc', test_acc)


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

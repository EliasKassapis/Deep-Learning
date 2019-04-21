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



dtype = torch.FloatTensor
# device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

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

  accuracy = (predictions.argmax(dim=1) == targets.argmax(dim=1)).type(dtype).mean()
  accuracy = accuracy.detach().data.cpu().item()

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
  y_test = torch.tensor(y_test, requires_grad=False).type(dtype).to(device)

  n_inputs = np.size(x_test,0)
  n_classes = np.size(y_test,1)
  v_size = np.size(x_test,1) * np.size(x_test,2) * np.size(x_test,3)
  n_batches = np.size(x_test,0)//b_size

  x_test = x_test.reshape((n_inputs, v_size))
  x_test = torch.tensor(x_test, requires_grad=False).type(dtype).to(device)


  # #load whole train data ############################################################
  # x_train = cifar10["train"].images
  # x_train = x_train.reshape((np.size(x_train,0), v_size))
  # x_train = torch.tensor(x_train, requires_grad=False).type(dtype).to(device)
  # y_train = cifar10["train"].labels
  # y_train = torch.tensor(y_train, requires_grad=False).type(dtype).to(device)


  #initialize the MLP model
  model = MLP(n_inputs = v_size, n_hidden = dnn_hidden_units, n_classes = n_classes, b_norm=FLAGS.b_norm)
  get_loss = torch.nn.CrossEntropyLoss()

  if FLAGS.optimizer == "adam":
      optimizer = torch.optim.Adam(model.parameters(), lr=eta)
  elif FLAGS.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)

  model.to(device)

  train_loss = []
  test_loss = []
  train_acc = []
  test_acc = []

  for step in range(max_steps):
    #get batch
    x, y = cifar10['train'].next_batch(b_size)
    y = torch.tensor(y).type(dtype).to(device)

    #stretch input images into vectors
    x = x.reshape(b_size, v_size)
    x = torch.tensor(x).type(dtype).to(device)

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
          y_test = torch.tensor(y_test, requires_grad=False).type(dtype).to(device)

          #stretch input images into vectors
          x_test = x_test.reshape(b_size, v_size)
          x_test = torch.tensor(x_test).type(dtype).to(device)

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

        if FLAGS.optimize == False:
            print('\nStep ',step, '\n------------\nTraining Loss = ', round(c_loss,4), ', Train Accuracy = ', current_train_acc, '\nTest Loss = ', round(c_test_loss,4), ', Test Accuracy = ', current_test_acc)

        if step > 0 and abs(test_loss[(int(step/FLAGS.eval_freq))] - test_loss[int(step/FLAGS.eval_freq)-1]) < eps:
                break

  # if FLAGS.optimize == False:
  #     plot_graphs(train_loss, 'Training Loss', 'orange',
  #                   test_loss, 'Test Loss', 'blue',
  #                   title='Stochastic gradient descent',
  #                   ylabel='Loss',
  #                   xlabel='Steps')
  #
  #     plot_graphs(train_acc, 'Training Accuracy', 'darkorange',
  #                   test_acc, 'Test Accuracy', 'darkred',
  #                   title='Stochastic gradient descent',
  #                   ylabel='Accuracy',
  #                   xlabel='Steps')
  #
  #     #save results:
  #     path = "./results/pytorch results/"
  #     np.save(path + 'train_loss', train_loss)
  #     np.save(path + 'train_acc', train_acc)
  #     np.save(path + 'test_loss', test_loss)
  #     np.save(path + 'test_acc', test_acc)

  return train_loss, test_loss, train_acc, test_acc

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

  FLAGS.optimizer = 'sgd'

  FLAGS.optimize = False

  FLAGS.b_norm = False

  main()



def get_setups():


    l_size_options = np.linspace(128,512,4).astype(int).astype(str)
    etas = np.linspace(1e-7, 1e-5, 3)
    opts = ["adam", "sgd"]
    batch_size = np.linspace(128, 384, 3).astype(int)
    n_hlayers = np.linspace(1,5,2).astype(int)
    b_norm = [True, False]

    setups = []

    #create all combos
    for i in range(3):
        for norm in b_norm:
            for n in n_hlayers:
                n_hidden = ''
                for layer in range(n):
                    l_size = np.random.choice(l_size_options)
                    if layer != (n-1):
                        n_hidden += l_size + ','
                    else:
                        n_hidden += l_size
                for b_size in batch_size:
                    for eta in etas:
                        for opt in opts:
                            setups.append([n_hidden, b_size, eta, opt,norm])

    return setups

setups = get_setups()

print('No of setups: ', len(setups))

def get_best_setup(setups):

    FLAGS.optimize = True
    setup_acc = []
    for i,setup in enumerate(setups):
        FLAGS.dnn_hidden_units = setup[0]
        FLAGS.batch_size = setup[1]
        FLAGS.learning_rate = setup[2]
        FLAGS.optimizer = setup[3]
        FLAGS.b_norm = setup[4]
        print('\nCurrent Setup hyperparameters (setup ',i,' of ',len(setups),'):\n----------------------\ndnn_hidden_units = ',FLAGS.dnn_hidden_units,
              '\nBatch size = ', FLAGS.batch_size,'\nTraining rate = ', FLAGS.learning_rate,
              '\nOptimizer = ', FLAGS.optimizer, '\nBatch Normalization =', FLAGS.b_norm)

        train_loss, test_loss, train_acc, test_acc = train()

        setup_acc.append(test_acc[-1])

        print('\nTraining Loss = ', round(train_loss[-1],4), ', Train Accuracy = ', round(train_acc[-1],4),
              '\nTest Loss = ', round(test_loss[-1],4), ', Test Accuracy = ', round(test_acc[-1],4))

        current_best = np.argmax(setup_acc)

        #Get current best setup parameters
        FLAGS.dnn_hidden_units = setups[current_best][0]
        FLAGS.batch_size = setups[current_best][1]
        FLAGS.learning_rate = setups[current_best][2]
        FLAGS.optimizer = setups[current_best][3]
        FLAGS.b_norm = setups[current_best][4]
        print('\nCurrent best = setup', current_best, ', Test accuracy = ', round(setup_acc[current_best],4))

    # best_setup_idx = np.argmax(setup_acc)
    #
    # #Get best setup parameters
    # FLAGS.dnn_hidden_units = setups[best_setup_idx][0]
    # FLAGS.batch_size = setups[best_setup_idx][1]
    # FLAGS.learning_rate = setups[best_setup_idx][2]
    # FLAGS.optimizer = setups[best_setup_idx][3]

    #Get results of best setup
    train_loss, test_loss, train_acc, test_acc = train()

    print('\nBest Setup hyperparameters:\n----------------------\ndnn_hidden_units = ', FLAGS.dnn_hidden_units,
          '\nBatch size = ', FLAGS.batch_size, '\nTraining rate = ', FLAGS.learning_rate,
          '\nOptimizer = ', FLAGS.optimizer, '\nBatch Normalization =', FLAGS.b_norm)

    #save results of best setup
    np.save('Best dnn_hidden_units', FLAGS.dnn_hidden_units)
    np.save('Best b_size', FLAGS.batch_size)
    np.save('Best eta', FLAGS.learning_rate)
    np.save('Best opt', FLAGS.optimizer)
    np.save('Best b_norm', FLAGS.b_norm)
    np.save('Best train_loss', train_loss)
    np.save('Best train_acc', train_acc)
    np.save('Best test_loss', test_loss)
    np.save('Best test_acc', test_acc)

n_hidden, b_size, eta, opt = get_best_setup(setups)

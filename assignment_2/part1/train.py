################################################################################
# MIT License
# 
# Copyright (c) 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################


def get_accuracy(predictions, targets, batch_size):
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

  acc = torch.mean((predictions.argmax(dim=1) == targets).to(torch.float))

  return acc



def train(config,n_run):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Train on T-1 first digits
    config.input_length = config.input_length - 1

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, device=device)
    elif config.model_type == 'LSTM':
        model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, device=device)


    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    model.to(device)

    train_loss = []
    train_acc = []
    t_loss = []
    t_acc = []

    #Convergence condition
    eps = 1e-6

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Clear stored gradient
        model.zero_grad()

        # Only for time measurement of step through network
        t1 = time.time()

        # Add more code here ...

        #Convert inputs and labels into tensors
        x = torch.tensor(batch_inputs, device=device)
        y = torch.tensor(batch_targets,device=device)


        #Forward pass
        pred = model.forward(x)
        loss = criterion(pred, y)
        t_loss.append(loss.item())
        optimizer.zero_grad()

        #Backward pass
        loss.backward()

        ############################################################################
        # QUESTION: what happens here and why?

        # ANSWER : the function torch.nn.utils.clip_grad_norm() is used to prevent
        # exploding gradients by ‘clipping’ the norm of the gradients, to restrain
        # the gradient values to a certain threshold. This essentially acts as a
        # limit to the size of the updates of the parameters of every layer, ensuring
        # that the parameter values don't change too much from their previous values.

        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        # Add more code here ...

        optimizer.step()
        accuracy = get_accuracy(pred,y, config.batch_size)
        t_acc.append(accuracy.item())

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))
            # print(f"x: {x[0,:]}, pred: {pred[0,:].argmax()}, y: {y[0]}") #######################################################################

        if step % 100 == 0:
            #Get loss and accuracy averages over 100 steps
            train_loss.append(np.mean(t_loss))
            train_acc.append(np.mean(t_acc))
            t_loss = []
            t_acc = []

            if step > 0 and abs(train_loss[-1] - train_loss[-2]) < eps:
                break


        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break


    print('Done training.')
    #
    #Save trained model and results
    if config.model_type == 'RNN':
        # #save model
        # torch.save(model, "./Results/RNN/" + str(config.input_length) + "_RNN_model")
        # #save train accuracy and loss
        # np.save("./Results/RNN/" + str(config.input_length) + "_RNN_accuracy", train_acc)
        # np.save("./Results/RNN/" + str(config.input_length) + "_RNN_loss", train_loss)

        #save model ####################################################################### For SURFsara
        torch.save(model, str(config.input_length) + "_RNN_model_" + str(n_run))
        #save train accuracy and loss
        np.save(str(config.input_length) + "_RNN_accuracy_" + str(n_run), train_acc)
        np.save(str(config.input_length) + "_RNN_loss_" + str(n_run), train_loss)

    elif config.model_type == 'LSTM':
        # #save model
        # torch.save(model, "./Results/LSTM/" + str(config.input_length) + "_LSTM_model")
        # #save train accuracy and loss
        # np.save("./Results/LSTM/" + str(config.input_length) + "_LSTM_accuracy", train_acc)
        # np.save("./Results/LSTM/" + str(config.input_length) + "_LSTM_loss", train_loss)

        #save model ####################################################################### For SURFsara
        torch.save(model,str(config.input_length) + "_LSTM_model_"  + str(n_run))
        #save train accuracy and loss
        np.save(str(config.input_length) + "_LSTM_accuracy_" + str(n_run), train_acc)
        np.save(str(config.input_length) + "_LSTM_loss_" + str(n_run), train_loss)



 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=5, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    # train(config)


#train models for different sequence lengths
for i in range(3):
    for model in ['RNN', 'LSTM']:
        print('Training', model)
        config.model_type = model
        for length in [5,10,15,20,25,30,35,40,45,50]:
            config.input_length = length
            train(config, i+1)


def test(config, seq_size, n_examples):

    # Initialize the dataset and data loader
    dataset = PalindromeDataset(seq_size)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    #Get one batch to test
    (batch_inputs, batch_targets)  = next(iter(data_loader))

    #Convert inputs and labels into tensors
    x = torch.tensor(batch_inputs, device=config.device)

    # Load the trained model
    model = torch.load('./Results/RNN/' + str(seq_size) + '_RNN_model', map_location='cpu')
    model.to(config.device)

    #get predictions for batch
    with torch.no_grad():
        pred = model.forward(x)

    print('\n----------------------\nSequence length: ',str(seq_size),'\n----------------------')

    for i in range(n_examples):
        print('\nTesting on palindrome',str(i+1),':\n---------------\n\nInput:',str(batch_inputs[i].tolist()),'\nPredicted last digit:',str(pred[i,:].argmax().item()),'\n')


# #Get qualitative results for models of different sizes
# for length in [5,10,15,20,25,30,35,40,45,50]:
#     test(config, length, 3)




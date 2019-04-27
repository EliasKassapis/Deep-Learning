# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import TextDataset
from model import TextGenerationModel

#Set default tensor type
torch.set_default_tensor_type('torch.FloatTensor')

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

  acc = torch.sum(predictions.argmax(dim=1) == targets).to(torch.float) /batch_size # check if torch.float is necessary

  return acc


def idx_2_onehot(input, vocab_size):

    #get dimensions of input
    length, batch_size = input.shape

    input = torch.unsqueeze(input, 2)

    one_hot = torch.zeros(length, batch_size, vocab_size).to(config.device) # change 188 to data.vocab_size
    one_hot.scatter_(2, input, 1)

    return one_hot


def get_next_char(char, model, hc, temperature, sampling="greedy"):  #input is (sentence length, batch_size, one_hot vec(char))

    #get model output
    pred, hc = model(char, hc) #pred = (sentence length, score of each char ,batch_size)

    #get char distributions
    p = F.softmax(pred.squeeze().to(torch.float)/temperature, dim=0)

    #sort characters according to probability mass
    p, idx = torch.sort(p)

    #sample one character
    if sampling == 'greedy':
        #get top character
        top_ch = idx[-1]
    elif sampling == 'egreedy':
        #get randomly one of top 3 characters
        top_ch = idx[-np.random.choice(range(3))]

    elif sampling == 'random':
        top_ch = torch.multinomial(pred.squeeze(),1).item()

    # print(top_ch)
    return top_ch.view(1,1), hc


def text_gen(model, s_length, vocab_size, temperature, sampling="greedy"):
    #get random first character in one-hot vector form
    char = torch.randint(0, vocab_size, (1,1))
    chars = [char.item()]
    char = idx_2_onehot(char, vocab_size)

    hc = None

    for i in range(s_length):
        next_char, hc = get_next_char(char, model, hc, temperature, sampling)
        chars.append(next_char.item())
        char = idx_2_onehot(next_char, vocab_size)

    return chars



def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, config.lstm_num_hidden, config.lstm_num_layers, device=device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss() ############################################################################################################
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    model.to(device)

    train_loss = []

    #Convergence criterion
    eps = 1e-6

    for epoch in range(100):

        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Clear stored gradient
            model.zero_grad()

            # Only for time measurement of step through network
            t1 = time.time()

            #######################################################
            # Add more code here ...

            #Convert list of tensors into one tensor for inputs and labels
            x = torch.stack(batch_inputs).to(device)
            y = torch.stack(batch_targets).to(device)


            #Convert input to one-hot vectors
            x = idx_2_onehot(x, dataset.vocab_size) #x = (sentence length, batch_size, one_hot vec(char))

            #Forward pass
            pred, _ = model.forward(x) #pred = (sentence length, score of each char ,batch_size)
            loss = criterion(pred, y)
            train_loss.append(loss.item())
            optimizer.zero_grad()


            #Backward pass
            loss.backward()
            optimizer.step()

            accuracy = get_accuracy(pred,y, config.batch_size)

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            # if step % config.print_every == 0:

                # print("[{}] Train Step {:04}/{:04}, Batch Size = {}, Examples/Sec = {:.2f}, "
                #       "Accuracy = {:.2f}, Loss = {:.3f}".format(
                #         datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                #         config.train_steps, config.batch_size, examples_per_second,
                #         accuracy, loss
                # ))



            if step % config.sample_every == 0:
                # Generate some sentences by sampling from the model
                #get text in int format
                text = text_gen(model, config.seq_length, dataset.vocab_size, 0.5, sampling='egreedy')
                #convert text to string
                text = dataset.convert_to_string(text)
                print('\nEpoch ',epoch+1,'/ 20, Training Step ',step,'/',int(config.train_steps),', Training Accuracy = ', accuracy.item(),'\n-----------------------------------------------\nGenerated text: ',text)

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

            if step % 5000 == 0:
                if step != 0:
                    #save model in each iteration just in case
                    torch.save(model, "step_" + str(step) +"_model")

        if step > 0 and abs(train_loss[step] - train_loss[step-1]) < eps:
            break



    print('Done training.')



 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default = './Ovid.txt', help="Path to a .txt file to train on") ############ May need to change to False
    parser.add_argument('--output_file', type=str, default = './gOvid.txt', help="Path to a .txt file to train on") ##################################################
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device used to train model')


    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)

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

# from dataset import PalindromeDataset ####

import torch
import torch.nn as nn

torch.set_default_tensor_type('torch.FloatTensor')
################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...


        #Save parameters
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device


        #initialize Weight matrices and biases
        self.Whx = nn.Parameter(torch.randn(num_hidden,input_dim))
        self.Whh = nn.Parameter(torch.randn(num_hidden,num_hidden))
        self.Wph = nn.Parameter(torch.randn(num_classes,num_hidden))
        self.bh = nn.Parameter(torch.randn(num_hidden))
        self.bh = nn.Parameter(torch.randn(num_hidden))
        self.bp = nn.Parameter(torch.randn(num_hidden))


    def forward(self, x):
        # Implementation here ...

        #initialize first hidden state
        self.h = torch.zeros(self.batch_size, self.num_hidden, device=self.device)

        #forward through all timesteps
        for t in range(self.seq_length):
            #get current input
            current_x = x[:,t].view(-1, self.input_dim)
            #get current h
            self.h = torch.tanh(self.Whx @ current_x.t() + self.Whh @ self.h + self.bh)

        #get output from final layer
        out = self.Wph @ self.h + self.bp

        return out.t()


# data = PalindromeDataset(3)
#
# palindrome = data.generate_palindrome()

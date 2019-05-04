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

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...

        #Save parameters
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        #Initialize input modulation gate params
        self.Wgx = nn.Parameter(torch.randn(num_hidden,input_dim))
        self.Wgh = nn.Parameter(torch.randn(num_hidden,num_hidden))
        self.bg = nn.Parameter(torch.randn(num_hidden))

        #Initialize input gate params
        self.Wix = nn.Parameter(torch.randn(num_hidden,input_dim))
        self.Wih = nn.Parameter(torch.randn(num_hidden,num_hidden))
        self.bi = nn.Parameter(torch.randn(num_hidden))

        #Initialize forget gate params
        self.Wfx = nn.Parameter(torch.randn(num_hidden,input_dim))
        self.Wfh = nn.Parameter(torch.randn(num_hidden,num_hidden))
        self.bf = nn.Parameter(torch.randn(num_hidden))

         #Initialize output gate params
        self.Wox = nn.Parameter(torch.randn(num_hidden,input_dim))
        self.Woh = nn.Parameter(torch.randn(num_hidden,num_hidden))
        self.bo = nn.Parameter(torch.randn(num_hidden))

        #Initialize linear output mapping params
        self.Wph = nn.Parameter(torch.randn(num_classes, num_hidden))
        self.bp = nn.Parameter(torch.randn(num_classes))


    def forward(self, x):
        # Implementation here ...

        #Initialize cell state and hidden state
        c = nn.Parameter(torch.zeros(self.batch_size, self.num_hidden, device=self.device))
        h = nn.Parameter(torch.zeros(self.batch_size, self.num_hidden, device=self.device))

        for t in range(self.seq_length):
            #get current input
            current_x = x[:,t].view(-1,self.input_dim)
            #input modulation gate
            g = torch.tanh(self.Wgx @ current_x.t() + self.Wgh @ h + self.bg)
            #input gate
            i = torch.sigmoid(self.Wix @ current_x.t() + self.Wih @ h + self.bi)
            #forget gate
            f = torch.sigmoid(self.Wfx @ current_x.t() + self.Wfh @ h + self.bf)
            #out gate
            o = torch.sigmoid(self.Wox @ current_x.t() + self.Woh @ h + self.bo)
            #get current c
            c = g * i + c * f
            #get current h
            h = torch.tanh(c) * o

        #get output from final layer
        out = (self.Wph @ h).t() + self.bp


        return out

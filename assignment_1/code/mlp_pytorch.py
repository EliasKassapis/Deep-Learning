"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes, b_norm = None, dropout = None): ####Added batch_norm
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################


    super(MLP,self).__init__()

    n_layers = n_hidden.copy()
    n_layers.insert(0,n_inputs)
    n_layers.append(n_classes)
    self.layers = []
    #
    for l in range(len(n_layers)-1):
      self.layers.append(nn.Linear(n_layers[l], n_layers[l+1]))
      if l != (len(n_hidden)):
        self.layers.append(nn.ReLU())
        if b_norm != None:
          self.layers.append(nn.BatchNorm1d(num_features=n_layers[l+1])) ##### batch normalization
        if dropout != None:
          self.layers.append(nn.Dropout(0.1))


    self.model = nn.Sequential(*self.layers)


    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    out = self.model(x)


    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

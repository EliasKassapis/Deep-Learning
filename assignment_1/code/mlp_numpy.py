"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
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
    self.n_hidden = n_hidden

    #initialize input layer
    a_0 = LinearModule(n_inputs,n_hidden[0])
    z_0 = ReLUModule()

    self.layers = [a_0]
    self.layer_out = [z_0]

    #initialize hidden_layers
    for l in range(len(n_hidden)-1):
      current_a = LinearModule(n_hidden[l],n_hidden[l + 1])
      current_z = ReLUModule()

      self.layers.append(current_a)
      self.layer_out.append(current_z)

    #initialize last layer
    a_N = LinearModule(n_hidden[-1],n_classes)
    z_N = SoftMaxModule()

    self.layers.append(a_N)
    self.layer_out.append(z_N)


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

    #forward pass
    for i in range(len(self.n_hidden)+1):
      x = self.layers[i].forward(x)
      x = self.layer_out[i].forward(x)

    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return x

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    ########################
    # PUT YOUR CODE HERE  #
    #######################

    for i in range(len(self.n_hidden), -1, -1):
        dout = self.layer_out[i].backward(dout)
        dout = self.layers[i].backward(dout)


    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return

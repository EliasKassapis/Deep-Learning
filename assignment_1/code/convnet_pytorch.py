"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    super(ConvNet,self).__init__()

    # self.n_channels = n_channels
    # self.n_classes = n_classes

    #Initializing layers
    self.conv1 = nn.Conv2d(n_channels,64, kernel_size=3, stride=1, padding=1)
    self.norm1 = nn.BatchNorm2d(64)
    self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.norm2 = nn.BatchNorm2d(128)
    self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.conv3_a = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.norm3_a = nn.BatchNorm2d(256)
    self.conv3_b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.norm3_b = nn.BatchNorm2d(256)
    self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.conv4_a = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
    self.norm4_a = nn.BatchNorm2d(512)
    self.conv4_b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.norm4_b = nn.BatchNorm2d(512)
    self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.conv5_a = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.norm5_a = nn.BatchNorm2d(512)
    self.conv5_b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.norm5_b = nn.BatchNorm2d(512)
    self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.avgpool = nn.AvgPool2d(kernel_size = 1, stride = 1, padding = 0)
    self.linear = nn.Linear(512,n_classes)


#    raise NotImplementedError
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

    x = F.relu(self.norm1(self.conv1(x)))
    x = self.maxpool1(x)

    x = F.relu(self.norm2(self.conv2(x)))
    x = self.maxpool2(x)

    x = F.relu(self.norm3_a(self.conv3_a(x)))
    x = F.relu(self.norm3_b(self.conv3_b(x)))
    x = self.maxpool3(x)

    x = F.relu(self.norm4_a(self.conv4_a(x)))
    x = F.relu(self.norm4_b(self.conv4_b(x)))
    x = self.maxpool4(x)

    x = F.relu(self.norm5_a(self.conv5_a(x)))
    x = F.relu(self.norm5_b(self.conv5_b(x)))
    x = self.maxpool5(x)

    x = self.avgpool(x)

    out = self.linear(x.view(x.shape[0], -1))


    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################


    self.in_features = in_features
    self.out_features = out_features

    mu = 0
    sigma = 0.0001

    p_w = np.random.normal(mu, sigma, (self.out_features, self.in_features))
    g_w = np.zeros((self.out_features, self.in_features))
    b = np.zeros((self.out_features,1))

    self.params = {'weight': p_w, 'bias': b}
    self.grads = {'weight': g_w, 'bias': b}

    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    out = ((self.params['weight'] @ x.T) + self.params['bias']).T

    self.x = x



    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################


    dx = dout @ self.params['weight']
    self.grads['weight'] = (dout.T.dot(self.x))
    # self.grads['bias'] = np.diag(dout.dot(np.identity(np.size(self.params['bias'])))).reshape(self.out_features,1)
    self.grads['bias'] = np.mean(dout, axis=0).reshape(self.out_features,1)

    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    out = np.maximum(x,0)

    self.x = x

    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    R = self.x.copy()
    R[R<0] = 0
    R[R>0] = 1


    dx = np.multiply(dout, R)




    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    b_dim = np.size(x,0)

    a = np.max(x,axis=1).reshape(b_dim,1)
    y = np.exp(x - a)
    out = y / np.sum(y, axis=1).reshape(b_dim,1)
    self.out = out

    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # print('x shape = ', np.shape(self.x))
    # print('dout shape = ', np.shape(dout))

    b_dim = np.size(self.out,0)
    x_dim = np.size(self.out,1)

    D = np.zeros([b_dim,x_dim,x_dim])

    index = np.arange(x_dim)

    D[:,index,index] = self.out

    d_x = D - np.multiply(self.out[:, :, None], self.out[:, None, :])
    dx = (dout[:, None, :] @ d_x).squeeze()


    # raise NotImplementedError##############################################
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    x = -np.log(x)
    D = x.dot(y.T)

    out = np.diag(D).mean()

    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    dx = np.divide(-y,x)/len(y)


    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

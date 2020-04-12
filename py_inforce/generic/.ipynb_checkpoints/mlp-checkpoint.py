import torch 
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
  """
  Multilayer perceptron template
  """
  def __init__(self, dims, activation = nn.ReLU, weight_init = None, LEARN_RATE = 0.01):
    """
    Generic template for multilayer perceptrons
    
    Args:
      dims: (1d iterable) eg list, contains sizes of each layer
      activation: (function) Activation function. Default is torch.nn.ReLU
      weight_init: (function: torch.nn.layer -> None) Initializes weights of layers. Default torch.nn.init.xavier_uniform_.
      LEARN_RATE: (float) learning rate. DEFAULT = 0.01. Passed on to optimizer
      
    Examples:
      cart_agent = MLP([4, 6, 6, 2])
      cart_agent = MLP([4, 6, 6, 2], nn.LeakyReLU)
      
      def weight_init(m):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01) 

      cart_agent = MLP([4, 6, 6, 2], nn.LeakyReLU, weight_init)
    """
    super(MLP, self).__init__()
    
    if weight_init == None:   
        def weight_init(m):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01) 
    
    layers = []

    layers.append(nn.Linear(dims[0], dims[1]))
    weight_init(layers[-1])
    
    for d in range(1, len(dims) - 1):
      layers.append(activation())
      layers.append(nn.Linear(dims[d], dims[d+1]))
      weight_init(layers[-1])
    
  
    self.model = nn.Sequential(*layers)

    self.lr = LEARN_RATE

  def forward(self, x):
    return self.model(x)
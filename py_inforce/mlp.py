
import torch 
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
  """
  Multilayer perceptron template
  """
  def __init__(self, dims, activation, weight_init, LEARN_RATE = 0.01):
    """
    Args:
      dims: (1d iterable) eg list, contains sizes of each layer
      activation: (function) Activation function
      weight_init: (function) Initializes weights
    """
    super(nn_policy_estimator, self).__init__()

    layers = []

    layers.append(nn.Linear(dims[0], dims[1]))
    weight_init(layers[-1].weight)

    for d in range(1, len(dims) - 1):
      layers.append(activation())
      layers.append(nn.Linear(dims[d], dims[d+1]))
      weight_init(layers[-1].weight)
      
  
    self.model = nn.Sequential(*layers)
    self.lr = LEARN_RATE

  def forward(self, x):
    return self.model(x)
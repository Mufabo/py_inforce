import torch
import torch.nn as nn
import numpy as np

def REINFORCE(agent, env, dist, optimizer, EPI_LEN , bf = lambda x: x, DISC_FACTOR = 0.95, MAX_EPISODES = 300, EARLY = lambda x: False):
  """
  A first implementation of the REINFORCE Algorithm.
  
  Works only on discrete action spaces so far

  Args:
    agent:   Acting agent/policy estimator eg a multilayer perceptron that will be trained 
    env:     Environment
    dist:    Parametric torch distribution
    optimizer: Torch optimizer
    EPI_LEN:   (uint) Max number of steps in an Episode.
    bf: (function: 1d torch.tensor -> 1d torch.tensor) basis function for scaling of returns. Default identity function
    DISC_FACTOR: (float) Discount factor aka gamma
    MAX_EPISODES: Number of episodes that should be sampled. Default = 300
    EARLY: (function: 1d-sequence -> Boolean) Function that returns true if training should be early stopped depending on achieved rewards
    

  Examples:
    from torch.distributions import Categorical
    import gym
    import torch.nn as nn
    from py_inforce.generic.mlp import MLP
    from py_inforce.policy_based.REINFORCE import REINFORCE
    import torch.optim as optim
    import torch
    import numpy as np

    env = gym.make('CartPole-v0')
    in_dim = env.observation_space.shape[0] # 4
    out_dim = env.action_space.n # 2
    cart_agent = MLP([in_dim, 128, 128, out_dim], nn.ReLU)
    optimizer = optim.Adam(cart_agent.parameters(), lr=cart_agent.lr)

    REINFORCE(cart_agent, env, Categorical, optimizer, 200, bf = lambda x: x - x.mean(), MAX_EPISODES=500, EARLY = lambda x: x == 200)
  """
  for episode in range(MAX_EPISODES):
    #####################
    # Sample a trajectory
    #####################
    done = False

    state = env.reset()

    rewards = torch.zeros(EPI_LEN)
    log_probs = torch.zeros(EPI_LEN)
    
    step = 0
    while not done:
      state = torch.tensor(state)
      pd = dist(logits=agent.forward(state))
      action = pd.sample()
      state, reward, done, _ = env.step(action.numpy())
      rewards[step] = reward
      log_probs[step] = pd.log_prob(value=action)
      step += 1


    ############################
    # Compute discounted Returns
    ############################

    returns = torch.zeros(step)
    ret = 0.0
    for t in reversed(range(step)):
      ret = rewards[t] + DISC_FACTOR * ret
      returns[t] = ret
    
    returns = bf(returns) 
    
    ###################    
    # Update policy net
    ###################
    
    log_probs = log_probs[:step]
    loss = torch.sum(- log_probs * returns)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if EARLY(sum(rewards)):
        break
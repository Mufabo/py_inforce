
import numpy as np
import torch.optim as optim

def REINFORCE(agent, env, dist, visualize = False, EPI_LEN = None, DISC_FACTOR = 0.95, MAX_EPISODES = 300):
  """
  A first implementation of the REINFORCE Algorithm.
  Works only on discrete action spaces so far

  Args:
  """
  for episode in range(MAX_EPISODES):
    #####################
    # Sample a trajectory
    #####################
    done = False

    state = env.reset()

    rewards = []
    log_probs = []

    while not done:
      state = torch.from_numpy(state.astype(np.float32))
      pd = dist(logits=agent.forward(state))
      action = pd.sample()
      state, reward, done, _ = env.step(action.numpy())
      rewards.append(reward)
      log_probs.append(pd.log_prob(value=action))

      if visualize:
        env.render()

    #################
    # Inner Loop
    #################

    returns = np.empty(len(rewards))
    ret = 0.0
    for t in reversed(range(len(rewards))):
      ret = rewards[t] + DISC_FACTOR * ret
      returns[t] = ret

    # Update policy net
    log_probs = torch.stack(log_probs)
    loss = - log_probs * torch.tensor(returns) # gradient terms
    loss = torch.sum(loss)

    optimizer = optim.Adam(agent.parameters(), lr=agent.lr)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




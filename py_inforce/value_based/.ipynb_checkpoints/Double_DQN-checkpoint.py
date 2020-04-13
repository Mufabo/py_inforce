import torch

def DQN(env, memory, q_net, t_net, optim, steps = 10000, eps = 1, disc_factor = 0.99, loss = torch.nn.MSELoss(), batch_sz = 128, tgt_update = 10, early = True,
        eps_decay = lambda eps, steps, step: eps - eps/steps,
        act = lambda s, eps, env, q_net: torch.tensor(env.action_space.sample()) if torch.rand(1) < eps else q_net(s).max(0)[1]):
    """
    Trains a neural network with Deep Q-Network algorithm
    
    Args:
      env         : openai gym environment
      memory      : Memory used to store samples, import from py_inforce.Generic.Memories
      q_net       : Neural Network to train, import from py_inforce.Generic.MLP
      t_net       : Target Net, copy of q_net
      optim       : Pytorch optimizer for q_net
      steps       : Integer, Max number of samples to collect.
                    Default = 10_000
      eps         : Float, probability for epsilon greedy policy
                    Default = 1
      disc_factor : Float, Discount factor aka gamma
                    Default = 0.99
      loss        : Pytorch compatible loss function
                    Default = torch.nn.MSELoss()
      batch_sz    : Int, number of samples for gradient descent
      tgt_updat   : Int, number of samples between update of t_net
      early       : Bool, indicates if conditions for early termination should be checked. 
                    At the moment the early termination is hardwired for the CartPole-v0 environment
                    Default = True
      eps_decay   : Function of eps, steps and the current step, computes decayed epsilon
                    Default = linear decay from 1 to 0 against steps
      act         : Function of env state s, eps and env, determines action
                    Default = Epsilon greedy
      
      Note:
       Based on:
      
       Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A.,Riedmiller, M., Fidjeland, A. K.,   
       Ostrovski, G. et al. (2015). Human-level control through deepreinforcement learning.Nature,518529–533.
    """   
    
    optimizer = optim(q_net.parameters(), lr = q_net.lr)
    ret = 0
    returns = []
    s = torch.tensor(env.reset(), dtype=torch.float32)  
    for step in range(steps):      
        a = act(s, eps, env, q_net)

        s_prime, r, done, _ = env.step(a.numpy())
        s_prime = torch.tensor(s_prime, dtype=torch.float32)
        eps = eps_decay(eps, steps, step)
        
        memory.push(s, a, r, s_prime, done)
        ret += r
        # Optimize
        if step >= batch_sz:
            s_, a_, r_, s_p, d_ = memory.sample(batch_sz)            
            y = r_ + disc_factor * t_net(s_p).max(1)[0] * (1 - d_)  
            predictions = q_net(s_).gather(1, a_.long()).flatten()          
            l = loss(y, predictions)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
        if step % tgt_update == 0:
            t_net.load_state_dict(q_net.state_dict())
        
        # Test for early break
        if early and done:
            ret = 0
            for _ in range(100):
                done = False
                state = torch.tensor(env.reset(), dtype=torch.float32)
                while not done:
                    s, r, done, _ = env.step(torch.argmax(q_net(s)).numpy())
                    s = torch.tensor(s, dtype=torch.float32)
                    ret += r
            if 195 <= ret/100:
                print('converged in %i steps' %step)
                break
                    
        s = torch.tensor(env.reset(), dtype=torch.float32) if done else s_prime

        
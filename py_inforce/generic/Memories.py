import torch

class ReplayMemory():
    '''
    ReplayMemory for DQN
    '''
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.s = torch.zeros([capacity, state_dim])
        self.a = torch.zeros([capacity, 1])
        self.r = torch.zeros([capacity])
        self.s_prime = torch.zeros([capacity, state_dim])
        self.done = torch.zeros([capacity])
        self.mem_ptr = 0

    def push(self, s, a, r, s_prime, done):
        self.s[self.mem_ptr] = s
        self.a[self.mem_ptr] = a
        self.r[self.mem_ptr] = r
        self.s_prime[self.mem_ptr] = s_prime
        self.done[self.mem_ptr] = done
        self.mem_ptr = (self.mem_ptr + 1) % self.capacity
        
    def sample(self, batch_size):
        idx = torch.randperm(self.s.shape[0])[:batch_size]
        return self.s[idx], self.a[idx], self.r[idx], self.s_prime[idx], self.done[idx]

    def __len__(self):
        return len(self.memory)
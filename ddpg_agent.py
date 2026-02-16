import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- The Neural Networks ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.out = nn.Linear(300, action_dim)
        
    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return torch.tanh(self.out(x)) # Outputs between -1 and 1

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fcs1 = nn.Linear(state_dim, 400)
        self.fca1 = nn.Linear(action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.q_value = nn.Linear(300, 1)

    def forward(self, state, action):
        s1 = F.relu(self.fcs1(state))
        a1 = self.fca1(action)
        x = F.relu(s1 + a1)
        x = F.relu(self.fc2(x))
        return self.q_value(x)

# --- The Memory (Replay Buffer) ---
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[i] for i in batch]
"""
This file contains the netowrk classes, Actor and Critic, along with replay buffer
"""

import random
from collections import deque

BATCH_SIZE = 128
 


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
# Actor Critic network
class Actor(nn.Module):
    """
    Actor class : Uses three layer, with the input as states vector and output the probility distribution of all actions
    fc1 : Input state -> 128 neurons or nodes
    fc2 : 128 -> 256 neurons
    fc3 : 256 -> Actions
    
    """
    def __init__(self,n_states,n_actions):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(n_states,128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,n_actions)
    
    def forward(self,state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = F.tanh(self.fc3(out))
        return out

class Critic(nn.Module):
    """
    Critic class : Used to compute value function for state action pair given. It consists of three layers.
    fc1: Input -> 128 nodes
    fc2: actions appended to output off fc1 -> 256 nodes
    fc3: 256 -> 1 node: this outputs just the value of the given action-value pair as input
    """
    def __init__(self,n_states,n_actions):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(n_states,128)
        self.fc2 = nn.Linear(128+n_actions,256)
        self.fc3 = nn.Linear(256,1)
    
    def forward(self,state,action):
        out = F.relu(self.fc1(state))
        out = torch.cat((out,action),1).cuda() #appending actions to output of fc1
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class ReplayBuffer:
    """
    Replay buffer stores the experience tuple of every step in sequence. While training, sample is randomly selected frm the stack so that
    the experience tuples remain uncorrelated
    """
    def __init__(self,buffer_size):
        self.size = buffer_size # defining the size of the buffer
        self.data = deque(maxlen=self.size)
    
    def length(self):
        return len(self.data)
    
    def sample(self, batch_size=BATCH_SIZE):
        """
        Function : Returns a tuple of list of states, actions rewards, next state and dones by random selection process. The size of the sampls is
        governed by batch_size parameter
        """
        if (self.length() < batch_size):
            warnings.warn('Get more samples!')
            return None
        else:
            batch = random.sample(self.data,batch_size)
            states = [s[0] for s in batch]
            actions = [s[1] for s in batch]
            rewards = [s[2] for s in batch]
            next_states = [s[3] for s in batch]
            dones = [s[4] for s in batch]
        
        return states, actions, rewards, next_states,dones
    
    def add(self,s):
        #Add experience s to the memory queue
        self.data.append(s)
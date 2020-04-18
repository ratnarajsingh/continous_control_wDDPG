"""
This file contains the definition of the DDPDG agent. The DDPG class defines basic construct of the algorigthm
"""


from model import Actor, Critic, ReplayBuffer
from copy import deepcopy
from noise import OUNoise
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

actor_lr = 1e-4
critic_lr = 1e-3
n_episodes = 2000
max_steps = 5000
tau = 0.01
gamma = 0.9
buffer_size = 10000000
batch_size = 512



def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class DDPG(object):
    """
    Class : defines the agent that operates on Deep deterministic policy gradient methods
    Contains 2 pair of networks, Actor and Critic.
    
    """
    def __init__(self,n_states,n_actions):
        """
        Inialize the class with input parametrs
        n_states  : size of the state vector
        n_actions : size of the action space
        """
        self.n_states = n_states
        self.n_actions = n_actions
        
        #hyper parameters
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        # setting up local and target actor classes and copying the weight of local netowrk to target
        self.actor = Actor(self.n_states,self.n_actions).cuda()
        self.actor_target = Actor(self.n_states,self.n_actions).cuda()
        self.actor_opt = Adam(self.actor.parameters(),lr = self.actor_lr)
        hard_update(self.actor_target,self.actor)
        
        self.critic = Critic(self.n_states,self.n_actions).cuda()
        self.critic_target = Critic(self.n_states,self.n_actions).cuda()
        self.critic_opt = Adam(self.critic.parameters(),lr = self.critic_lr)
        hard_update(self.critic_target,self.critic)
        
        #critic network optimizes itself by minimizing the MSE of TD error. Actor netowrk minizes the negative policy value
        self.critic_loss  = nn.MSELoss()
        
        self.buffer = ReplayBuffer(buffer_size)
        self.noise = OUNoise(self.n_actions,123)
        self.training=True
        
    def set_to_test(self):
        self.training = False
        
    def set_to_train(self):
        self.training = True
        
    def step(self, states, actions, rewards, next_states, dones):
        # Take a step in learning. Add the latest s,a,r,s,d tuple to the experience buffer
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.buffer.add([state, action, reward, next_state, done])
        if(self.buffer.length()>self.batch_size):
            # if sufficient samples are collected then learn
            self.learn_(self.batch_size)
        
        
    def act(self, state):
        # convert the states to cuda float tensor, set network to eval and then get an action after switching off grad. After taking action
        # set the network back to train
        state = torch.cuda.FloatTensor(state)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()
        self.actor.train()
        # Add noise to action distribution and return
        if(self.training):
            action += self.noise.noise()            
        return np.clip(action,-1,1)
    
    def reset(self):
        self.noise.reset()
        
    def learn_(self,batch_size):    
        """
        Function : train the network one step taking sample of 'batch_size'
        """
        
        # Load sample and convert to cuda float tensor
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size=self.batch_size)
        states = torch.cuda.FloatTensor(states)
        actions = torch.cuda.FloatTensor(actions)
        rewards = torch.cuda.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.cuda.FloatTensor(next_states)
        dones = torch.cuda.FloatTensor(dones).unsqueeze(1)
        
        #critic loss calculation, calculate current q value
        q_vals = self.critic(states,actions)
        # get next actions using next states and calcualte the target at next step
        next_actions = self.actor_target(next_states)
        q_target = self.critic_target(next_states,next_actions)
        
        # calculate expected rewards set the critic loss to TD error and actor loss to negative of total value of state
        q_prime = rewards + self.gamma*q_target*(1-dones)
        
        critic_loss = self.critic_loss(q_vals,q_prime)
        policy_loss  = -self.critic(states, self.actor(states)).mean()
        
        # update networks
        self.actor_opt.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()

        self.critic_opt.zero_grad()
        critic_loss.backward() 
        self.critic_opt.step()
        
        #soft target netowrk updates : changing target network parameters a little bit (tau times) towards the local network 
        soft_update(self.actor_target,self.actor,tau=self.tau)
        soft_update(self.critic_target,self.critic,tau=self.tau)
        
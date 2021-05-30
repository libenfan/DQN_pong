# -*- coding: utf-8 -*-
"""
Created on Mon May 17 09:29:20 2021

@author: 13271
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from collections import namedtuple

import gym, random, pickle, os.path, math, glob

from wrappers import *

from collections import namedtuple
from itertools import count
import numpy as np 
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
# from memory import ReplayMemory 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transion', ('state', 'action', 'next_state', 'reward'))

## 超参数
# epsilon = 0.9
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.02
EPS_DECAY = 1000000
TARGET_UPDATE = 1000
RENDER = False
lr = 1e-3
INITIAL_MEMORY = 10000
MEMORY_SIZE = 10 * INITIAL_MEMORY
n_episode = 2000

# 这里用colab运行时的路径
# MODEL_STORE_PATH = '/content/drive/My Drive/'+'DQN_pytorch_pong'
# modelname = 'DQN_Pong'
# madel_path = MODEL_STORE_PATH + '/' + 'model/' + 'DQN_Pong_episode60.pt'

# 本地运行时
MODEL_STORE_PATH = os.getcwd()
print(MODEL_STORE_PATH)
modelname = 'DQN_Pong'
madel_path = MODEL_STORE_PATH + '/' + 'model/' + 'DQN_Pong_episode900.pt'



class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)



class DQN_agent():
    def __init__(self,in_channels=1, action_space=[], learning_rate=1e-3, memory_size=100000, epsilon=0.99):
        
        self.in_channels = in_channels        
        self.action_space = action_space
        self.action_dim = self.action_space.n     
                
        self.memory_buffer = ReplayMemory(memory_size)
        self.stepdone = 0
        self.DQN = DQN(self.in_channels, self.action_dim).cuda()
        self.target_DQN = DQN(self.in_channels, self.action_dim).cuda()
        # 加载之前训练好的模型
        self.DQN.load_state_dict(torch.load(madel_path))
        
        self.target_DQN.load_state_dict(self.DQN.state_dict())
        self.optimizer = optim.RMSprop(self.DQN.parameters(),lr=learning_rate, eps=0.001, alpha=0.95)
        
        
        
    def select_action(self, state):
        
        self.stepdone += 1
        state = state.to(device)
        # epsilon = EPS_END + (EPS_START - EPS_END)* \
        #     math.exp(-1. * self.stepdone / EPS_DECAY) 
        epsilon = 0.99
        # print(epsilon)
        if random.random()<epsilon:
            action = torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)
        else:
            action = self.DQN(state).detach().max(1)[1].view(1,1)
            
        return action        
        
        
    def learn(self):
        
        if self.memory_buffer.__len__()<BATCH_SIZE:
            return
        
        transitions = self.memory_buffer.sample(BATCH_SIZE)
        
        batch = Transition(*zip(*transitions))
        # print(batch)
        actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action))) 
        rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward))) 
    
    
        
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.uint8).bool()
        
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).to('cuda')
        
        # print(type(batch.state))
        state_batch = torch.cat(batch.state).to('cuda')
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)
        
        state_action_values = self.DQN(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_DQN(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.DQN.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
    
            
class Trainer():
    def __init__(self, env, agent, n_episode):
        self.env = env
        self.n_episode = n_episode
        self.agent = agent
        # self.losslist = []
        self.rewardlist = []
        
        
    def get_state(self,obs):
        state = np.array(obs)
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state)
        return state.unsqueeze(0)    # 转化为四维的数据结构
        
    def train(self):

        for episode in range(900,self.n_episode):
            
            obs = self.env.reset()
            state = self.get_state(obs)
            episode_reward = 0.0
            
            # print('episode:',episode)
            for t in count():  
                # print(state.shape)                              
                action = self.agent.select_action(state)
                if RENDER:
                    self.env.render()
                
                
                obs,reward,done,info = self.env.step(action)
                episode_reward += reward
                
                if not done:
                    next_state = self.get_state(obs)
                else:
                    next_state = None
                # print(next_state.shape)
                reward = torch.tensor([reward], device=device)
                
                # 将四元组存到memory中
                '''
                state: batch_size channel h w    size: batch_size * 4
                action: size: batch_size * 1
                next_state: batch_size channel h w    size: batch_size * 4
                reward: size: batch_size * 1                
                '''
                self.agent.memory_buffer.push(state, action.to('cpu'), next_state, reward.to('cpu')) # 里面的数据都是Tensor
                state = next_state
                # 经验池满了之后开始学习
                if self.agent.stepdone > INITIAL_MEMORY:
                    self.agent.learn()
                    if self.agent.stepdone % TARGET_UPDATE == 0:
                        self.agent.target_DQN.load_state_dict(self.agent.DQN.state_dict())
                
                if done:
                    break
            # print(episode_reward)
            if episode % 20 == 0:
                torch.save(self.agent.DQN.state_dict(), MODEL_STORE_PATH + '/' + "model/{}_episode{}.pt".format(modelname, episode))
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(self.agent.stepdone, episode, t, episode_reward))
            
            self.rewardlist.append(episode_reward)
                
            
            self.env.close()
        return
        

    def plot_reward(self):
        
        plt.plot(self.rewardlist)
        plt.xlabel("episode")
        plt.ylabel("episode_reward")
        plt.title('train_reward')
        
        plt.show()
        

        
       
if __name__ == '__main__':
   
    # create environment
    env = gym.make("PongNoFrameskip-v4")
    env = make_env(env)
    action_space = env.action_space
    state_channel = env.observation_space.shape[2]  
    
    agent = DQN_agent(in_channels = state_channel, action_space= action_space)
    
    trainer = Trainer(env, agent, n_episode)
    trainer.train()
    trainer.plot_reward()
    
      
   





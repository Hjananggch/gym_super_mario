import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import os
from collections import deque

import torch
import torch.nn as nn
from torchvision import transforms    

import gym
import gym_super_mario_bros
from gym.spaces import Box
from gym.wrappers import FrameStack

from nes_py.wrappers import JoypadSpace


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
    
# skip 4 frames, take max over the last 2 frames
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8)   
    
    def observation(self, obs):
        transform = transforms.Grayscale()
        return transform(torch.tensor(np.transpose(obs, (2, 0, 1)).copy(), dtype=torch.float))

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env,shape):
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        
    def observation(self, obs):
        transform = transforms.Compose([
            transforms.Resize(self.shape, antialias=True),
            transforms.Normalize(0, 255)
        ])
        return transform(obs).squeeze(0)

class DDQNSolvere(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.online = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input,model):
        if model == 'online':
            return self.online(input)
        else:
            return self.target(input)

class DDQNAgent:
    def __init__(self,action_dim,save_dir):
        self.action_dim = action_dim
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.net = DDQNSolvere(self.action_dim).to(device)
        self.exploration_rate = 0.75 #控制随机探索的概率
        self.exploration_rate_decay = 0.999998 # 探索率衰减
        self.exploration_rate_min = 0.01 # 探索率最小值
        self.curr_step = 0 # 当前步数
        self.memory = deque(maxlen=10000) # 经验回放
        self.batch_size = 32 # 批量大小
        self.gamma = 0.78 # 折扣因子
        self.loss_fn = nn.SmoothL1Loss() # 损失函数
        self.sync_period = 10 # 同步频率
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=0.0001) # 优化器
        self.episode_reward = [] # 记录每局游戏的奖励
        self.moving_average_episode_reward = [] # 记录每局游戏的移动平均奖励
        self.current_episode_reward = 0.0 
    
    def log_episode(self):
        self.episode_reward.append(self.current_episode_reward)
        self.current_episode_reward = 0.0
    
    def log_period(self,episode,epsilon,step):
        self.moving_average_episode_reward.append(np.round(np.mean(self.episode_reward[-checkpoint_period:]),3))
        print('Episode:{},Step:{},Epsilon:{},Moving Average Reward:{}'.format(episode,step,epsilon,self.moving_average_episode_reward[-1]))
        return self.moving_average_episode_reward[-1]

    
    def remember(self,state,next_state,action,reward,done):
        self.memory.append((torch.tensor(state.__array__()),
        torch.tensor(next_state.__array__()),
        torch.tensor([action]),torch.tensor([reward]),torch.tensor([done])))

    def experience_replay(self,step_reward):
        self.current_episode_reward += step_reward # 记录当前局游戏的奖励
        if self.curr_step % self.sync_period == 0:
            self.net.target.load_state_dict(self.net.online.state_dict())

        if self.batch_size > len(self.memory):
            return

        state, next_state, action, reward, done = self.recall()

        q_values = self.net(state.to(device),model='online')[np.arange(0,self.batch_size),action.to(device).long()]

        with torch.no_grad():
            best_action = torch.argmax(self.net(next_state.to(device),model='online'),dim=1)

            next_q_values = self.net(next_state.to(device),model='target')[np.arange(0,self.batch_size),best_action]

            q_target = reward.to(device) + (1 - done.to(device).float()) * self.gamma * next_q_values

        loss = self.loss_fn(q_values.float(),q_target.float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    


    def recall(self):
        state, next_state,action, reward,done = map(torch.stack,zip(*random.sample(self.memory,self.batch_size)))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    
    def act(self,state):
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)
        else:
            action_values = self.net(torch.tensor(state.__array__()).to(device).unsqueeze(0),model='online')
            action = torch.argmax(action_values,dim=1).item()
        
        # 更新探索率
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min,self.exploration_rate)
        self.curr_step += 1
        return action
        
    def load(self,path):
        checkpoint = torch.load(path,map_location='cpu')
        self.net.load_state_dict(checkpoint['model'])
        self.exploration_rate = checkpoint['exploration_rate']
    
    def save_checkpoint(self):
        filename = os.path.join(self.save_dir,'checkpoint_{}.pth'.format(episode))
        torch.save(dict(model=self.net.state_dict(),exploration_rate=self.exploration_rate),filename)
        print('---checkpoint saved---')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, [['left'],['left',"A"],['right'], ['right', 'A']])
    obs = env.reset()
    print(obs.shape)

    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    env.seed(42)
    env.action_space.seed(42)
    torch.manual_seed(42)
    torch.random.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    obs = env.reset()


    checkpoint_period = 10
    save_dir = 'weights'
    agent = DDQNAgent(action_dim=env.action_space.n,save_dir=save_dir)
    agent.load('weights/checkpoint_662.pth')

    score = 0
    episode = 0
    while True:
        state = env.reset()
        step = 0
        while True:
            action = agent.act(state)
            
            env.render()

            next_state, reward, done, info = env.step(action)

            agent.remember(state, next_state, action, reward, done)
            agent.experience_replay(reward)

            state = next_state
            step += 1
            if done:
                agent.log_episode()
                episode += 1
                soc = agent.log_period(episode, agent.exploration_rate, agent.curr_step)
                if soc >score:
                    score = soc
                    agent.save_checkpoint()
                    print("best score:{}".format(score))

                if episode % checkpoint_period == 0:
                    
                    
                    agent.save_checkpoint()
                    torch.cuda.empty_cache()

                break

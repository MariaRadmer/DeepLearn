import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib as plt

from collections import deque  # for memory
from tqdm import tqdm          # for progress bar
import numpy as np

from copy import deepcopy
import random
from turtle import done, st


class Model(nn.Module):
    def __init__(self, observation_size, action_size):
        super(Model, self).__init__()
        self.dense1 = nn.Linear(observation_size, 100)

        self.dense2 = nn.Linear(100, 100)
        self.dense3 = nn.Linear(100, 100)
        # had to hardcode it to 4 becuse action_size is 2
        self.dense4 = nn.Linear(100, action_size)

        torch.nn.init.xavier_uniform_(self.dense1.weight)
        torch.nn.init.xavier_uniform_(self.dense2.weight)
        torch.nn.init.xavier_uniform_(self.dense3.weight)
        torch.nn.init.xavier_uniform_(self.dense4.weight)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = F.relu(x)
        x = self.dense3(x)
        x = F.relu(x)
        x = self.dense4(x)
        return x

    def predict(self, x):
        x = torch.tensor(x)
        x = self.forward(x)

        return torch.argmax(x)


class Agent:
    def __init__(self, observation_size, action_size):
        self.observation_size = observation_size
        self.action_size = action_size
        self.criterion = nn.MSELoss()
        self.model = Model(observation_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        # memory that stores N most new transitions
        self.memory = deque([], maxlen=2000)
        self.epsilon = 1.0
        self.decay_rate = 0.99
        # good place to store hyperparameters as attributes

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        rand = random.uniform(0, 1)
        if rand > self.epsilon:
            return int(self.model.predict(state))
        else:
            return int(env.action_space.sample())

    def replay(self, batch_size):
        # update model based on replay memory
        # you might want to make a self.train() helper method
        transition_set = random.sample(self.memory, batch_size)
        self.optimizer.zero_grad()

        for transition in transition_set:
            self.train(transition)

        self.epsilon *= self.decay_rate
        self.optimizer.step()

    def train(self, data_set):
        state = torch.tensor(data_set[0])
        action = torch.tensor(data_set[1])
        reward = torch.tensor(data_set[2])
        next_state = torch.tensor(data_set[3])
        done = torch.tensor(data_set[4])
        gamma = 0.9

        new_state = reward

        if not done:
            new_state = reward + gamma * \
                float(torch.max(self.model.forward(next_state)))

        prediction = self.model.forward(state)[action]
        loss = self.criterion(prediction, new_state)
        loss.backward()


filename_out = '5 - model_7_100x100x100.pth'


def train(envrioment, agent: Agent, episodes=1000, batch_size=64):  # train for many games
    for e in tqdm(range(episodes)):

        if e % 10 == 0:
            torch.save(agent.model.state_dict(), filename_out)
            print("save")

        state, _ = envrioment.reset()
        done = False
        total_r = 0
        while not done:
            # 1. make a move in game.
            action = agent.act(state)
            # 2. have the agent remember stuff.

            next_state, reward, done, _, _ = envrioment.step(action)
            total_r += reward
            agent.remember(state, action, reward, next_state, done)
            # 3. update state
            state = next_state
            # 4. if we have enough experiences in out memory, learn from a batch with replay.
            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)

    envrioment.close()


env = gym.make('CartPole-v1', render_mode='human')
agent = Agent(env.observation_space.shape[0], env.action_space.n)
train(env, agent)
env.close()
torch.save(agent.model.state_dict(), filename_out)

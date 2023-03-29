import math
import time

import flappy_bird_gym
import torch
import matplotlib.pyplot as plt
import matplotlib
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count

from FlappyBird.Network import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
env = flappy_bird_gym.make("FlappyBird-v0")

BATCH_SIZE = 128
GAMMA = 0.99
LR = 1e-4
TAU = 0.005
n_actions = env.action_space.n

state = env.reset()
n_observations = len(state)
policy_net = DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
policy_net.load_state_dict(torch.load("./data/model_parameter.pkl"))

def select_action(state, is_test):
    if is_test:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)

print("test")
obs = env.reset()
reward_sum = 0
while True:
    # Next action:
    # (feed the observation to your agent here)
    obs += 0.02
    action = select_action(torch.tensor(obs, dtype=torch.float32).unsqueeze(0), True)


    # Processing:
    obs, reward, done, info = env.step(action)
    print(obs, action)
    reward_sum += reward
    # Rendering the game:
    # (remove this two lines during training)
    env.render()
    time.sleep(1 / 144)  # FPS

    # Checking if the player is still alive
    if done:
        break
env.close()
print(reward_sum)

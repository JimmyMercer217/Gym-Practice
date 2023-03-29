import math
import time

import flappy_bird_gym
import torch
import matplotlib.pyplot as plt
import matplotlib
import random
from FlappyBird.Network import DQN, ReplayMemory, Transition
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
env = flappy_bird_gym.make("FlappyBird-v0")

BATCH_SIZE = 128
GAMMA = 0.99
LR = 1e-4
TAU = 0.005
n_actions = env.action_space.n

# 0.18.3版本下gym，reset没有info
state = env.reset()
n_observations = len(state)
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load("./data/model_parameter.pkl"))
# 使得policy_net和target_net的参数一致
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1e4

steps_done = 0


def select_action(state, is_test):
    global steps_done
    sample = random.random()
    # eps_threshold = 0.1
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    # print(eps_threshold)
    steps_done += 1
    if sample > eps_threshold or is_test:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def draw_pic(x, y, path):
    plt.plot(x, y)
    plt.xlabel("iteration")
    plt.ylabel("reward")
    plt.savefig(path)


if torch.cuda.is_available():
    # 调大迭代次数可提升效果
    num_episodes = 500
else:
    num_episodes = 50

reward_list = []

for i_episode in range(num_episodes):
    if (i_episode + 1) % 10 == 0:
        print(f'已经完成{i_episode + 1}代')
    # Initialize the environment and get it's state
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    reward_sum = 0
    for t in count():
        action = select_action(state, False)
        observation, reward, done, info = env.step(action.item())
        observation[1] += 0.02
        if (action == 0 and observation[1] < 0) or (action == 1 and observation[1] > 0):
            reward = -reward

        if done:
            reward -= 10
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            if next_state[0][0] - state[0][0] > 0:
                reward += 10
        reward_sum += reward
        reward = torch.tensor([reward], device=device)
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            reward_list.append(reward_sum)
            print(f'reward:{reward_sum}')
            break

print('Complete')
print(reward_list)
draw_pic(range(len(reward_list)), reward_list, "./iteration_reward_01.png")
torch.save(policy_net.state_dict(), "./data/model_parameter.pkl")

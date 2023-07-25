import warnings

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# avoid the warning message
gym.logger.set_level(40)


# PPO trick https://zhuanlan.zhihu.com/p/512327050?utm_id=0

class Memory:
    def __init__(self, mini_batch_size):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.mini_batch_size = mini_batch_size

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def sample(self):
        n_states = len(self.states)


# Trick 8—Orthogonal Initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()

        # actor
        # Trick 8—Orthogonal Initialization 输出层为0.01
        fc1 = nn.Linear(state_dim, hidden_dim)
        orthogonal_init(fc1)
        fc2 = nn.Linear(hidden_dim, hidden_dim)
        orthogonal_init(fc2)
        fc3 = nn.Linear(hidden_dim, action_dim)
        orthogonal_init(fc3, gain=0.01)
        # discrete
        self.action_layer = nn.Sequential(
            fc1,
            nn.Tanh(),
            fc2,
            nn.Tanh(),
            fc3,
            # 对action进行softmax
            nn.Softmax(dim=-1)
        )

        # critic
        fc1 = nn.Linear(state_dim, hidden_dim)
        orthogonal_init(fc1)
        fc2 = nn.Linear(hidden_dim, hidden_dim)
        orthogonal_init(fc2)
        fc3 = nn.Linear(hidden_dim, 1)
        orthogonal_init(fc3)
        # discrete
        self.value_layer = nn.Sequential(
            fc1,
            nn.Tanh(),
            fc2,
            nn.Tanh(),
            fc3
        )

    def forward(self):
        raise NotImplementedError

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return state, action, dist.log_prob(action)

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.value_layer(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, K_epochs, mini_batch_size, eps_clip, gae_lambda,
                 use_gae=False):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        # Trick 9—Adam Optimizer Epsilon Parameter
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.losses = []
        self.eps = np.finfo(np.float32).eps.item()
        self.gae_lambda = gae_lambda
        self.use_gae = use_gae
        # 用作缓冲区
        self.memory = Memory(mini_batch_size)

    def update_policy(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        if not self.use_gae:
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            # Normalizing the rewards:
            rewards = torch.tensor(rewards).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        else:
            rewards = self.memory.rewards
        # convert list to tensor
        old_states = torch.stack(self.memory.states).to(device).detach()
        old_actions = torch.stack(self.memory.actions).to(device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()
        old_is_terminal = self.memory.is_terminals
        # Optimize policy for K epochs:
        # todo try mini batch, replace two network to one and optimize gae calculation
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss:

            if not self.use_gae:
                # not gae , MC
                advantages = rewards - state_values.detach()
            else:
                # gae
                advantages = np.zeros(len(rewards), dtype=np.float32)
                for t in range(len(rewards) - 1):
                    discount = 1
                    a_t = 0
                    for k in range(t, len(rewards) - 1):
                        a_t += discount * (
                                rewards[k] + self.gamma * state_values.detach()[k + 1] * (1 - old_is_terminal[k]) -
                                state_values.detach()[k])
                        discount *= self.gamma * self.gae_lambda
                    advantages[t] = a_t
                advantages = torch.tensor(advantages).to(device)
                # Trick 1—Advantage Normalization
                advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # Trick 5—Policy Entropy
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values,
                                                                 advantages + state_values.detach()) - 0.01 * dist_entropy
            self.losses.append(loss.mean())
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()

            # Trick 7—Gradient clip
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def train_network(self, epsiodes=500):
        epsiode_rewards = []
        mean_rewards = []
        for epsiode in range(1, epsiodes + 1):
            state, info = env.reset()
            ep_reward = 0
            while True:
                # Running policy_old:
                state, action, log_prob = self.policy_old.select_action(state)
                self.memory.logprobs.append(log_prob)
                self.memory.states.append(state)
                self.memory.actions.append(action)
                # 下面与网络进行交互
                state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                # Saving reward and is_terminal:
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(1.0 if done else 0)
                ep_reward += reward
                if done:
                    self.update_policy()
                    self.memory.clear_memory()
                    break
            # logging
            epsiode_rewards.append(ep_reward)
            mean_rewards.append(torch.mean(torch.Tensor(epsiode_rewards[-30:])))
            print("第{}回合的奖励值是{:.2f},平均奖励是{:.2f}".format(epsiode, ep_reward, mean_rewards[-1]))
        return epsiode_rewards, mean_rewards


def draw_pic(x, y, path):
    plt.plot(x, y)
    plt.xlabel("iteration")
    plt.ylabel("reward")
    plt.savefig(path)


if __name__ == '__main__':
    ############## Hyperparameters ##############
    # "cuda:0" or "cpu"
    device = torch.device("cpu")
    # creating environment
    env = gym.make("CartPole-v1")
    # env = env.unwrapped
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    gae_lambda = 0.95
    hidden_dim = 64  # number of variables in hidden layer
    mini_batch_size = 32
    lr = 0.002
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    ppo = PPO(state_dim, action_dim, hidden_dim, lr, gamma, K_epochs, mini_batch_size, eps_clip, gae_lambda,
              use_gae=False)
    epsiode_rewards, mean_rewards = ppo.train_network(epsiodes=500)
    draw_pic(range(len(epsiode_rewards)), epsiode_rewards, "iteration-sumReward-no-gae")

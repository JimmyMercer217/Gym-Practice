# Gym-Practice

OpenAi-Gym环境熟悉与RL学习练习

| Environment |      Parameters      |   Algorithm |  Status |
|-------------|:-------------:|------------:|  ------------ |
| FrozenLake  |  is_slippery=False |  Q-Learning |  ✔ |
| FrozenLake  |    is_slippery=True   |  Q-Learning |  ✔ |
| Taxi        |  |  Q-Learning |  ✔ |
| FlappyBird  |  |         DQN |  ✔ |
| CartPole    |  |  Double-DQN |  ✔ |
| CartPole    |  |         DQN |  ✔ |
| LunarLander | default |  Double-DQN |  ✔ |
| LunarLander | default | D3QN | ✔ |

## 贝尔曼方程

$A$：动作集合
$S$：状态集合
$π(a|s) = P(a|s)$：表示在状态 $s$采取 $a$的概率，也称策略。
$P(s'|s)$：在状态 $s$下状态转移到 $s'$的概率。
$P(s'|a,s)$：在状态 $s$下采取动作 $a$后状态转移到 $s'$的概率。
$P(s',r|a,s)$：在状态 $s$下采取动作 $a$后状态转移到 $s'$并且获得奖励 $r$的概率。

定义累计回报为

$$
G_t=R_{t+1}+\gamma R_{t+2}+…=\sum_{k=0}^{∞} \gamma^k R_{t+k+1} \tag{1}
$$

表示在 $t$ 时刻智能体与环境的一次交互过程的累计奖励，其中 $\gamma$代表折扣因子。

有状态价值函数 $V_{π}(s)=\mathbb{E}[G_t|S_t=s]$ 表示智能体在状态 $s$的期望，将 $G_t$展开则有

$$
\begin{aligned}
V_{π}(s)&=\mathbb{E}_{π}[G_t|S_t=s] \\
& = \mathbb{E}_{π}[R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}…|S_t=s] \\
& = \mathbb{E}_{π}[R_{t+1}+\gamma (R_{t+2}+\gamma R_{t+3}…)|S_t=s] \\
& = \mathbb{E}_{π}[R_{t+1}+\gamma G_{t+1}|S_t=s]
\end{aligned} \tag{2}
$$

其中有

$$
\begin{aligned} 
&\mathbb{E}{_\pi }[{R_{t + 1}}|{S_t} = s] \\
& = \sum\limits_r {r \cdot p(r|s)} \\
& = \sum\limits_r {r\sum\limits_a {p(r,a|s)} } \\
& = \sum\limits_r {\sum\limits_a {r \cdot p(r,a|s)} } \\
& = \sum\limits_a {\sum\limits_r {r \cdot p(r,a|s)} } \\
& = \sum\limits_a {\sum\limits_r {r \cdot p(a|s)p(r|s,a)} } \\
& = \sum\limits_a {\sum\limits_r {r \cdot \pi (a|s)p(r|s,a)} } \\
& = \sum\limits_a {\pi (a|s)\sum\limits_r {r \cdot p(r|s,a)} } \\
& = \sum\limits_a {\pi (a|s)\sum\limits_r {r\sum\limits_{s'} {p(s',r|s,a)} } } \\
& = \sum\limits_a {\pi (a|s)\sum\limits_r {\sum\limits_{s'} {p(s',r|s,a)} } } r \end{aligned} \tag{3}
$$

以及

$$
\begin{aligned} 
&\mathbb{E}{_\pi }[{G_{t + 1}}|{S_t} = s] \\
= & \sum\limits_{{G_{t + 1}}} {{G_{t + 1}}p({G_{t + 1}}|s)} \\
= & \sum\limits_{{G_{t + 1}}} {{G_{t + 1}}\sum\limits_a {\sum\limits_{s'} {p({G_{t + 1}}|s',a)p(s',a|s)} } } \\
= & \sum\limits_{{G_{t + 1}}} {{G_{t + 1}}\sum\limits_a {\sum\limits_{s'} {p({G_{t + 1}}|s')p(s',a|s)} } } \\ 
= & \sum\limits_{{G_{t + 1}}} {{G_{t + 1}}\sum\limits_a {\sum\limits_{s'} {p({G_{t + 1}}|s')p(a|s)p(s'|s,a)} } } \\ 
= & \sum\limits_{{G_{t + 1}}} {{G_{t + 1}}\sum\limits_a {\sum\limits_{s'} {p({G_{t + 1}}|s')\pi (a|s)p(s'|s,a)} } } \\ 
= & \sum\limits_a {\sum\limits_{s'} {\pi (a|s)p(s'|s,a)\sum\limits_{{G_{t + 1}}} {{G_{t + 1}}p({G_{t + 1}}|s')} } } \\ 
= & \sum\limits_a {\sum\limits_{s'} {\pi (a|s)p(s'|s,a)} } {\mathbb{E}_\pi }[{G_{t + 1}}|{S_{t + 1}} = s']\\ 
= & \sum\limits_a {\sum\limits_{s'} {\pi (a|s)p(s'|s,a)} } {v_\pi }(s')\\ 
= & \sum\limits_a {\pi (a|s)\sum\limits_{s'} {p(s'|s,a)} } {v_\pi }(s')\\ 
= & \sum\limits_a {\pi (a|s)\sum\limits_r {\sum\limits_{s'} {p(s',r|s,a)} } } {v_\pi }(s') \end{aligned} \tag{4}
$$

其中第三个等式由于动作 $a$是状态 $s$转移到状态 $s'$的动作，而 $G_{t+1}$只会受到状态 $s'$以及其之后状态的动作的影响，故动作 $a$不影响 $G_{t+1}$，认为两者独立，而独立事件即有 $P(A|B) = P(A)$。

记

$$
\begin{aligned}
Q_{\pi}(s, a) &=\sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma v_{\pi}\left(s^{\prime}\right)\right]
\end{aligned} \tag{5}
$$

可得出

$$
\begin{aligned}
V_{\pi}(s) 
& = \mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s\right] \\
& = \mathbb{E}_{\pi}\left[R_{t+1}+\gamma G_{t+1} \mid S_{t}=s\right] \\
& = \sum_{a} \pi(a \mid s) \sum_{s^{\prime}} \sum_{r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma \mathbb{E}\left[G_{t+1} \mid S_{t+1}=s^{\prime}\right]\right]\\
& = \sum_{a} \pi(a \mid s) \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma v_{\pi}\left(s^{\prime}\right)\right] \\
& = \sum_{a} \pi(a \mid s)Q_{\pi}(s, a)
\end{aligned} \tag{6}
$$

将公式 $(6)$代入公式 $(5)$可得出

$$
\begin{aligned}
Q_{\pi}(s, a) =\sum_{s^{\prime},r} p\left(s^{\prime},r \mid s, a\right)\left[r+\gamma \sum_{a'} \pi(a' \mid s')Q_{\pi}(s', a'))\right]
\end{aligned} \tag{7}
$$

当在状态 $s$采取动作 $a$后转移的状态 $s'$以及获得的奖励 $r$固定时，公式 $(7)$退化为

$$
\begin{aligned}
Q_{\pi}(s, a) =r+\gamma \sum_{a'} \pi(a' \mid s')Q_{\pi}(s', a'))
\end{aligned} \tag{8}
$$

## Q-Learning

&emsp;&emsp;在强化学习中value-based方法要在当前状态 $s$下选择对应 $Q$值最大的action当作策略，因此对于每个状态下的动作 $a$对应的 $Q$值评估十分重要。其中，非常直觉的是，**Q-learning和DQN通过使得在确定状态 $s$下的动作 $a$的 $Q_{\pi}(s, a)$最大来评估动作 $a$好坏。** 以退化公式 $(7)$为例，则估计为

$$
\begin{aligned}
Q^{*}_{\pi}(s, a) =r+\gamma \max Q_{\pi}(s', a')
\end{aligned} \tag{9}
$$

其中通过引入学习率 $\alpha$来实现当前 $Q$值与最佳 $Q^*$的结合，从而得到如下公式：

$$
\begin{aligned}
Q_{\pi}(s, a) =  Q_{\pi}(s, a)+ \alpha[ r+\gamma \max Q_{\pi}(s', a')-Q_{\pi}(s, a)]
\end{aligned} \tag{10}
$$

&emsp; &emsp; 如上所述，网上大部分资料对于 $\alpha$的理解仅停留在对于当前 $Q$值与最佳 $Q^*$的结合。**但是事实上，当我们的环境中对于状态转移以及获取到的reward都唯一确定的时候， $\alpha != 1$只会引起收敛速度的减缓(指 $Q$表中的值的收敛，不影响策略的收敛，另外其实在完全确定情况下甚至不需要贪婪策略，因为对于状态下的动作都能唯一判断其好坏)，只有当环境中存在stochastic才需要引入 $\alpha$来使得 $Q$值以及策略能够收敛，且一般情况下 $\alpha$值较小。通用起见，可以设计 $\alpha$为一个较小值。**

&emsp; &emsp; 引用wiki上的一句话就是'In fully deterministic environments, a learning rate of $\alpha_t=1$  is optimal. When the problem is stochastic, the algorithm converges under some technical conditions on the learning rate that require it to decrease to zero.'

&emsp;&emsp;此外，可以通过frozenLake中 is_slippery=False，使用不同的学习率来验证fully deterministic environments下 $\alpha$的作用。在is_slippery=True条件下，分别假定已知状态转移概率来理解使用公式 $(7)$，以及未知状态转移概率验证 $\alpha$在stochastic environments的作用。

## DQN

## Double DQN

## Dueling DQN



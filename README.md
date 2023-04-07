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
| LunarLander |  | Dueling-DQN |   |
| LunarLander |  | D3QN |   |

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
& = \mathbb{E}_{π}[R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+2}…|S_t=s] \\
& = \mathbb{E}_{π}[R_{t+1}+\gamma (R_{t+2}+\gamma R_{t+2}…)|S_t=s] \\
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
Q_{\pi}(s, a) =\sum_{s^{\prime},r} p\left(s^{\prime} \mid s, a\right)\left[r+\gamma \sum_{a'} \pi(a' \mid s')Q_{\pi}(s', a'))\right]
\end{aligned} \tag{7}
$$

当在状态 $s$采取动作 $a$后转移的状态 $s'$以及获得的奖励 $r$固定时，公式 $(7)$退化为

$$
\begin{aligned}
Q_{\pi}(s, a) =r+\gamma \sum_{a'} \pi(a' \mid s')Q_{\pi}(s', a'))
\end{aligned} \tag{8}
$$

## Q-Learning&DQN

## Double DQN

## Dueling DQN

## D3QN With Prioritized Experience Replay Memory

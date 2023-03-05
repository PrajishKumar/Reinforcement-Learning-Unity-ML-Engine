# Report

### Learning Algorithm

Our agent uses [Multi-Agent Deep Deterministic Policy Gradient](https://arxiv.org/pdf/1706.02275v4.pdf)(MADDPG) as the
learning algorithm.
MADDPG is an off-policy, actor-critic based reinforcement learning approach that extends DDPG for training multiple
agents.

Each individual agent maintains an actor and a critic network.
Technically, there are 4 networks per individual agent, if you take in consideration the presence of both local and
target networks to avoid the moving-target problem in Q-learning.
The actors of all agents in an environment are only aware of their local observations and output the actions that they
think their agent should take.
The critics of each agent, however, have access to the observations of and the actions taken by all the agents.

The algorithm concurrently learns the actor (policy) and the critic (action-value function) of all agents.

Here's the psuedocode of the
algorithm [[source]](https://www.researchgate.net/publication/348367411_Multi-Agent_Reinforcement_Learning_using_the_Deep_Distributed_Distributional_Deterministic_Policy_Gradients_Algorithm)

<img src="https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/main/collaborative_tennis/media/maddpg_algo.png"  width="600" alt="MADDPG algorithm">

The actor network is parameterized as a neural network with 24 inputs, 2 outputs and 3 hidden layers with 512, 256 and
128 nodes.
ReLU is used as the activation function for the hidden layers.

The input to the actor network is a stack of observations from three consecutive frames.

The output corresponds to the actions the agent can take (left/right, up/down).
The output of the neural network is supposed to represent the actions in the range [-1, +1].
Therefore, the final layer is passed through a `tanh` activation function to clip the actions to the required range.


<img src="https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/main/collaborative_tennis/media/actor.png"  width="500" alt="Actor">

The critic network is parameterized as a neural network with 54 inputs, 1 output and 3 hidden layers with 512, 256 and
128 nodes.
ReLU is used as the activation function for the hidden layers.

The input to the critic network is a concatenated vector of the stacked observations and the stacked actions of the two
agents.
The stacked observations of two agents are of size $2 \times 24 = 48$ and for the stacked actions, $2 \times 2 = 4$.
That would explain the input size of $48 + 4 = 52$ of the critic.

The output is a scalar that represents the estimate of the action-value of the state-actions pair passed into the
network.

<img src="https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/main/collaborative_tennis/media/critic.png"  width="500" alt="Critic">

The hyperparameters used for training:

| **Hyperparameter**                                                        | **Value**               |
|---------------------------------------------------------------------------|-------------------------|
| replay buffer size                                                        | 100,000                 |
| minibatch size of sampling from replay buffer                             | 128                     |
| target network update frequency                                           | 1 updates every episode |
| discount factor                                                           | 0.99                    |
| learning rate of actor                                                    | 0.0001                  |
| learning rate of critic                                                   | 0.0001                  |
| weight decay of critic                                                    | 0                       |
| interpolation parameter for target network update                         | 0.001                   |
| maximum standard deviation of noise applied to the actions                | 0.5                     |
| minimum standard deviation of noise applied to the actions                | 0.01                    |
| decay rate standard deviation of noise applied to the actions per episode | 0.995                   |
| timesteps in an episode                                                   | 1000                    |

Policies for agents with continuous actions are known to perform well when they are trained with some noise on the
chosen actions.
Instead of Ornstein-Uhlenbeck noise, we opted for a simple Gaussian noise with a zero mean.
Over the course of training, we gradually decrease the applied noise.

### Result

Performance of the agent after training for 4300 episodes:

![](https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/main/collaborative_tennis/media/successful_test.gif)

#### Plot of score over time of training:

<img src="https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/main/collaborative_tennis/media/scores_plot.png" width="600" alt="Scores Plot"/>

Our agents received a score of more than 0.5 (averaged over 100 episodes) at episode number 4173.
However, we decided to run the training for longer in the hope that the agent might end up getting even more scores.
We stopped when the score was around 1.

```
Episode 100	Average Score: 0.003000
Episode 200	Average Score: 0.000000
Episode 300	Average Score: 0.000000
Episode 400	Average Score: 0.001000
Episode 500	Average Score: 0.001000
Episode 600	Average Score: 0.001000
Episode 700	Average Score: 0.001000
Episode 800	Average Score: 0.002900
Episode 900	Average Score: 0.003000
Episode 1000	Average Score: 0.000000
Episode 1100	Average Score: 0.013000
Episode 1200	Average Score: 0.005000
Episode 1300	Average Score: 0.037000
Episode 1400	Average Score: 0.046000
Episode 1500	Average Score: 0.028800
Episode 1600	Average Score: 0.031400
Episode 1700	Average Score: 0.018000
Episode 1800	Average Score: 0.066900
Episode 1900	Average Score: 0.096900
Episode 2000	Average Score: 0.058000
Episode 2100	Average Score: 0.067000
Episode 2200	Average Score: 0.064800
Episode 2300	Average Score: 0.050900
Episode 2400	Average Score: 0.067900
Episode 2500	Average Score: 0.069800
Episode 2600	Average Score: 0.074900
Episode 2700	Average Score: 0.062900
Episode 2800	Average Score: 0.071900
Episode 2900	Average Score: 0.071900
Episode 3000	Average Score: 0.051000
Episode 3100	Average Score: 0.049000
Episode 3200	Average Score: 0.054000
Episode 3300	Average Score: 0.048000
Episode 3400	Average Score: 0.057000
Episode 3500	Average Score: 0.106400
Episode 3600	Average Score: 0.142300
Episode 3700	Average Score: 0.164700
Episode 3800	Average Score: 0.213900
Episode 3900	Average Score: 0.197300
Episode 4000	Average Score: 0.208900
Episode 4100	Average Score: 0.302000
Episode 4173	Average Score: 0.505900
Our agent learnt to get a score of 0.5 in 4173 episodes!
Episode 4200	Average Score: 0.678800
Episode 4300	Average Score: 1.152500
```

### Ideas for Future Work

1. **Implementing a prioritized replay buffer.**

   Our agent takes way too long to learn how to play.
   Tweaking some hyperparameters could help, but that doesn't take away the issue that, in our environment, the rewards
   a
   little sparse.

   So, it becomes important that we favor experiences that produced high rewards when we sample experiences from the
   replay buffer.

2. **Providing a baseline in the policy improvement step.**

   In other actor-critic methods, such as A2C and A3C, we use a baseline in the policy update step.
   The baseline could be a simple average of rewards obtained, or could be an estimate of the advantage.
   The idea of a baseline is to selectively favor actions that lead to returns higher than the expected value from the
   baseline.

   This might be useful at the start since out agents seems to be not learning that well initially.
   A baseline will roughly split the experiences into two halves - ones from which we can learn to do better and the
   others on what not to do.
   This selective learning could improve the agents' performance even more.

3. **Applying batch normalization.**

   Applying batch normalization to neural networks is known to help the agent converge
   faster [source](https://arxiv.org/pdf/1502.03167.pdf).
   We could have tried that with our actor and critic networks.

4. **Using D4PG over DDPG.**

   Using D4PG seems promising for our
   application [source](https://www.researchgate.net/publication/348367411_Multi-Agent_Reinforcement_Learning_using_the_Deep_Distributed_Distributional_Deterministic_Policy_Gradients_Algorithm).
   D4PG is also an off-policy actor-critic learning algorithm but is known to provide better gradients and thus more
   stable learning signal.
   Moreover, the D4PG algorithm runs many actors in parallel with all feeding into a single replay buffer, executing a
   sophisticated distributed way to gather the experience, and as a result, a significant time saving can be achieved.


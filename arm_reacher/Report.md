# Report

### Learning Algorithm

Our agent uses [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971)(DDPG) as the learning algorithm.
DDPG is an off-policy, actor-critic based reinforcement learning approach that extends DQN for tasks with continuous
action spaces.

The algorithm concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

Here's the psuedocode of the algorithm [source](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

<img src="https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/9503738f9878386f399b3a000eb126d8f01021bd/arm_reacher/media/ddpg_algo.png"  width="600" alt="DDPG algorithm">

The actor network is parameterized as a neural network with 33 inputs, 4 outputs and 3 hidden layers with 512, 256 and
128 nodes.
ReLU is used as the activation function for the hidden layers.
The output of the neural network is supposed to represent the actions in the range [-1, +1].
Therefore, the final layer is passed through a `tanh` activation function to clip the actions to the required range.

<img src="https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/9503738f9878386f399b3a000eb126d8f01021bd/arm_reacher/media/actor.png"  width="500" alt="DQN algorithm">

The critic network is parameterized as a neural network with 37 inputs, 1 outputs and 3 hidden layers with 512, 256 and
128 nodes.
ReLU is used as the activation function for the hidden layers.

The input to the critic network is a concatenated vector of the 33 states and the 4 actions.
The output is a scalar that represents the estimate of the action-value of the state-action pair passed into the
network.

<img src="https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/9503738f9878386f399b3a000eb126d8f01021bd/arm_reacher/media/critic.png"  width="500" alt="DQN algorithm">

The hyperparameters used for training:

| **Hyperparameter**                                                        | **Value**                    |
|---------------------------------------------------------------------------|------------------------------|
| replay buffer size                                                        | 1,000,000                    |
| minibatch size of sampling from replay buffer                             | 128                          |
| target network update frequency                                           | 10 updates every 20 episodes |
| discount factor                                                           | 0.99                         |
| learning rate of actor                                                    | 0.0001                       |
| learning rate of critic                                                   | 0.0003                       |
| weight decay of critic                                                    | 0                            |
| interpolation parameter for target network update                         | 0.001                        |
| maximum standard deviation of noise applied to the actions                | 0.1                          |
| minimum standard deviation of noise applied to the actions                | 0.01                         |
| decay rate standard deviation of noise applied to the actions per episode | 0.99                         |

### Result

Performance of the agent after training for 2000 episodes:

![](https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/9503738f9878386f399b3a000eb126d8f01021bd/minion_navigation/media/banana_nav.gif)

Plot of score over time of training:

![](https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/9503738f9878386f399b3a000eb126d8f01021bd/minion_navigation/media/plot_result.png)

The agent first scored more than 13 (averaged over 100 episodes) at episode number 572.
However, we decided to run the training for longer in the hope that the agent might end up getting even more scores.

```
Episode 100	Average Score: 0.80
Episode 200	Average Score: 4.09
Episode 300	Average Score: 6.81
Episode 400	Average Score: 10.10
Episode 500	Average Score: 12.01
Episode 600	Average Score: 12.99
Episode 700	Average Score: 14.59
Episode 800	Average Score: 14.15
Episode 900	Average Score: 13.89
Episode 1000	Average Score: 15.45
Episode 1100	Average Score: 14.40
Episode 1200	Average Score: 15.11
Episode 1300	Average Score: 15.08
Episode 1400	Average Score: 14.27
Episode 1500	Average Score: 14.91
Episode 1600	Average Score: 14.98
Episode 1700	Average Score: 14.57
Episode 1800	Average Score: 15.09
Episode 1900	Average Score: 15.77
Episode 2000	Average Score: 16.34

Our minion learnt to get a score of 13.0 in 572 episodes.
```

### Ideas for Future Work

Upon testing the trained agent in the task, the agent seems to perform optimally.
The agent quickly realises not to waste time meandering around while collecting bananas.
The agent also adapts very quickly when new bananas are spawned nearby too.
Overall, it looks like the agent performs just as good, if not better, than how a human operator would have in this
task.

However, some improvements could be tried:

1. Implementing a double DQN is, honestly, a low-hanging fruit.
   One could attempt this minor change in TD target calculations and at least try to explore if it provides any
   significant benefit to the learning curve.

2. We see that the average score plateaus around 15 starting around 700 episodes.
   At this point, the agent is generally good.
   But to be exceptionally good, the agent, after some time has passed, might benefit from re-collecting those rare
   events in the past where it might have made a maneuver that resulting in collecting a lot of points in a short
   duration.  
   A prioritized experience replay would have weighed such experiences with a higher priority and we might have visited
   such tuples more often in our learning.  



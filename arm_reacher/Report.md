# Report

### Learning Algorithm

Our agent uses [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971)(DDPG) as the learning algorithm.
DDPG is an off-policy, actor-critic based reinforcement learning approach that extends DQN for tasks with continuous
action spaces.

The algorithm concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

Here's the psuedocode of the algorithm [source](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

<img src="https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/main/arm_reacher/media/ddpg_algo.png"  width="600" alt="DDPG algorithm">

The actor network is parameterized as a neural network with 33 inputs, 4 outputs and 3 hidden layers with 512, 256 and
128 nodes.
ReLU is used as the activation function for the hidden layers.
The output of the neural network is supposed to represent the actions in the range [-1, +1].
Therefore, the final layer is passed through a `tanh` activation function to clip the actions to the required range.

<img src="https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/main/arm_reacher/media/actor.png"  width="500" alt="DQN algorithm">

The critic network is parameterized as a neural network with 37 inputs, 1 outputs and 3 hidden layers with 512, 256 and
128 nodes.
ReLU is used as the activation function for the hidden layers.

The input to the critic network is a concatenated vector of the 33 states and the 4 actions.
The output is a scalar that represents the estimate of the action-value of the state-action pair passed into the
network.

<img src="https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/main/arm_reacher/media/critic.png"  width="500" alt="DQN algorithm">

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
| decay rate standard deviation of noise applied to the actions per episode | 0.995                        |

Policies for agents with continuous actions are known to perform well when they are trained with some noise on the chosen actions. 
Instead of Ornstein-Uhlenbeck noise, we opted for a simple Gaussian noise with a zero mean. 
However, over the course of training, we gradually decrease the applied noise. 

### Result

Performance of the agent after training for 2000 episodes:

![](https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/main/arm_reacher/media/successful_test.gif)

Plot of score over time of training:

![](https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/main/arm_reacher/media/scores_plot.png)

The agent first scored more than 30 (averaged over 100 episodes) at episode number 99.
However, we decided to run the training for longer in the hope that the agent might end up getting even more scores.

```
Episode 100	Average Score (over 100 consecutive episodes): 8.94
Episode 200	Average Score (over 100 consecutive episodes): 30.14
Episode 300	Average Score (over 100 consecutive episodes): 34.86

Our arm learnt to get a score of 30.0           in 99 episodes.
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



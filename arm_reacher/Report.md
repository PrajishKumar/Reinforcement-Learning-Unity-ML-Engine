# Report

### Learning Algorithm

Our agent uses [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971)(DDPG) as the learning algorithm.
DDPG is an off-policy, actor-critic based reinforcement learning approach that extends DQN for tasks with continuous
action spaces.

The algorithm concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

Here's the psuedocode of the algorithm [[source]](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

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
| timesteps in an episode                                                   | 1000                         |

Policies for agents with continuous actions are known to perform well when they are trained with some noise on the chosen actions. 
Instead of Ornstein-Uhlenbeck noise, we opted for a simple Gaussian noise with a zero mean. 
However, over the course of training, we gradually decrease the applied noise. 

### Result

Performance of the agent after training for 300 episodes:

![](https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/main/arm_reacher/media/successful_test.gif)

Plot of score over time of training:

<img src="https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/main/arm_reacher/media/scores_plot.png" width="500" alt="Scores Plot"/>

The agent first scored more than 30 (averaged over 100 episodes, and over 20 agents) at episode number 99.
As in, from episode 99 to 198, the averaged scores across all agents was consistently above 30.
However, we decided to run the training for longer in the hope that the agent might end up getting even more scores.

```
Episode 100	Average Score (over 100 consecutive episodes): 8.94
Episode 200	Average Score (over 100 consecutive episodes): 30.14
Episode 300	Average Score (over 100 consecutive episodes): 34.86

Our arm learnt to get a score of 30.0           in 99 episodes.
```

### Ideas for Future Work

Given that we run an episode for 1000 timesteps, the maximum score an agent can get is $1000 \times 0.1 = 100$. 
Our trained agent gets a score of just above 30 pretty soon, but the score does not really improve after. 
When one looks at the performance of the agent, it visually appears to track just fine.

However, some improvements could be tried:

1. **Implementing a prioritized replay buffer.**
   
   Our replay buffer has a capacity of 1 million experiences. 
   And for 20 agents, with an episode containing 1000 experiences, it is equivalent to storing 50 episodes worth of experiences. 
   That's a good enough number, however, we see that the agent's behaviour starts converging around 100 episodes. 
   That would mean that the experiences in the buffer, on average, produce the same behavior. 
   
   It's at this stage, one might want to treat experiences with higher expected returns favorably. 
   And we can do that by sampling those experiences with higher expected returns with higher probability.  

2. **Providing a baseline in the policy improvement step.**

   In other actor-critic methods, such as A2C and A3C, we use a baseline in the policy update step. 
   The baseline could be a simple average of rewards obtained, or could be an estimate of the advantage. 
   The idea of a baseline is to selectively favor actions that lead to returns higher than the expected value from the baseline. 
   
   This is very useful when the agent seems to have converged to suboptimal behaviors. 
   A baseline will roughly split the experiences into two halves - ones from which we can learn to do better and the others on what not to do. 
   This selective learning could improve the agent's performance even more. 


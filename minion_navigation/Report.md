# Report

### Learning Algorithm

Our agent uses [Deep Q Learning](https://www.nature.com/articles/nature14236) as the learning algorithm. 
Deep Q Learning is an off-policy, value-based reinforcement learning approach that describes the action-value function as a deep neural network. 

Here's an outline to the algorithm researches at DeepMind used in their paper reference above. 

![](https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/9503738f9878386f399b3a000eb126d8f01021bd/minion_navigation/media/dqn_algo.png width="500")

The Q network is parameterized as a neural network with 37 inputs, 4 outputs and 2 hidden layers of 128 nodes each. 
ReLU is used as the activation function.

![](https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/9503738f9878386f399b3a000eb126d8f01021bd/minion_navigation/media/q_network.png){width=50%}

The hyperparameters used for training:

| **Hyperparameter**                                | **Value** |
|---------------------------------------------------|-----------|
| replay buffer size                                | 100,000   |
| minibatch size of sampling from replay buffer     | 64        |
| target network update frequency                   | 4         |
| discount factor                                   | 0.99      |
| learning rate                                     | 0.0005    |
| interpolation parameter for target network update | 0.001     |
| initial exploration                               | 1.0       |
| final exploration                                 | 0.1       |
| exploration decay rate                            | 0.995     |


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
Overall, it looks like the agent performs just as good, if not better, than how a human operator would have in this task.

However, some improvements could be tried:

1. Implementing a double DQN is, honestly, a low-hanging fruit. 
One could attempt this minor change in TD target calculations and at least try to explore if it provides any significant benefit to the learning curve.

2. We see that the average score plateaus around 15 starting around 700 episodes.
At this point, the agent is generally good. 
But to be exceptionally good, the agent, after some time has passed, might benefit from re-collecting those rare events in the past where it might have made a maneuver that resulting in collecting a lot of points in a short duration.  
A prioritized experience replay would have weighed such experiences with a higher priority and we might have visited such tuples more often in our learning.  



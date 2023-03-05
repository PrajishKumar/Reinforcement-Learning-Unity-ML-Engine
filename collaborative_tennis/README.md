[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Collaborative Tennis

### Project Details

The environment contains two agents in control of a racket each.  
The task is to pass the ball to the other court (to the other agent), while not letting the ball touch the floor or go out of bounds.

We use the Unity ML-Agents environment. This is performed as part of a project for the course Deep Reinforcement Learning Nanodegree.
See example below.

<img src="https://github.com/PrajishKumar/Reinforcement-Learning-Unity-ML-Engine/blob/main/collaborative_tennis/media/successful_test.gif" width="300" alt=""/>

If an agent hits the ball over the net, it receives a reward of +0.1. 
If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. 
Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. 
Each agent receives its own, local observation. 
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic. 
In order to solve the environment, our agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).
That is, after each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. 
This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


### Getting Started

1. Set up a conda virtual environment.

```bash 
conda create --name collaborative_tennis python=3.6 
conda activate collaborative_tennis 
```

2. Clone the repository. 

```bash 
git clone git@github.com:PrajishKumar/Reinforcement-Learning-Unity-ML-Engine.git
```

3. Install all the requirements for this project.

```bash
cd collaborative_tennis/python
pip install . 
cd ../code
```

4. Create an IPython kernel for the environment. 

```bash
python -m ipykernel install --user --name collaborative_tennis --display-name "collaborative_tennis"
```

5. Open the notebook. 

```bash
jupyter notebook ArmReacher.ipynb
```

6. Before running code in a notebook, change the kernel to match the `collaborative_tennis` environment. 
In the menu bar, choose `Kernel` -> `Change kernel` -> `collaborative_tennis`.

### Instructions

Follow the instructions in `code/CollaborativeTennis.ipynb` to see the agent learn to track the moving target in play.
The instructions in the notebook should be sufficiently clear to guide you.
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Minion Navigation

### Project Details

The idea is to collect as many yellow bananas as possible from the environment while avoiding blue bananas.

We use the Unity ML-Agents environment. This is performed as part of a project for the course Deep Reinforcement Learning Nanodegree.
See example below. 

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. 
Thus, the goal of our agent (minion) is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic. 
The agent is judged with the average score over 100 consecutive episodes. 
To consider the task solved, we set a goal of average score >= 13 (over 100 consecutive episodes).

### Getting Started

1. Set up a conda virtual environment.

```bash 
conda create --name banana_nav python=3.6 
conda activate banana_nav 
```

2. Clone the repository. 

```bash 
git clone git@github.com:PrajishKumar/Reinforcement-Learning-Unity-ML-Engine.git
```

3. Install all the requirements for this project.

```bash
cd minion_navigation/python
pip install . 
cd ../code
```

4. Create an IPython kernel for the environment. 

```bash
python -m ipykernel install --user --name banana_nav --display-name "banana_nav"
```

5. Open the notebook. 

```bash
jupyter notebook MinionNavigation.ipynb
```

6. Before running code in a notebook, change the kernel to match the `banana_nav` environment. 
In the menu bar, choose `Kernel` -> `Change kernel` -> `banana_nav`.

### Instructions

Follow the instructions in `code/MinionNavigation.ipynb` to see the agent learn to collect bananas in play.
The instructions in the notebook should be sufficiently clear to guide you.